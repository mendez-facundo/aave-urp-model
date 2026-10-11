"""Minimal synchronous JSON-RPC client for the Ethereum archive endpoint.

The endpoint URL is read from ``ALCHEMY_RPC_URL`` (loaded from the project
``.env``). The URL embeds the API key, so it is never logged and is removed
from any error message before the message is raised.
"""

from __future__ import annotations

import itertools
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from requests import ConnectionError as RequestsConnectionError
from requests import RequestException, Session, Timeout

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

REQUEST_TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 6
RETRY_BASE_SECONDS = 1.0
RETRY_MAX_SECONDS = 30.0

# Throttling appears as these JSON-RPC error codes or messages on Alchemy.
RATE_LIMIT_ERROR_CODES = frozenset({429, -32005})
RATE_LIMIT_MESSAGE_MARKERS = (
    "rate limit",
    "too many requests",
    "compute units",
    "capacity",
)


class RpcError(RuntimeError):
    """Raised when an RPC call fails permanently or exhausts its retries."""


class _Transient(Exception):
    """Internal marker for a failure that should be retried."""


def _is_rate_limit_error(error: Any) -> bool:
    """Return True when a JSON-RPC error object signals throttling."""
    if not isinstance(error, dict):
        return False
    if error.get("code") in RATE_LIMIT_ERROR_CODES:
        return True
    message = str(error.get("message", "")).lower()
    return any(marker in message for marker in RATE_LIMIT_MESSAGE_MARKERS)


def _has_rate_limit_error(body: Any) -> bool:
    """Return True when a single or batch JSON-RPC response reports throttling."""
    if isinstance(body, dict):
        return _is_rate_limit_error(body.get("error"))
    if isinstance(body, list):
        return any(
            isinstance(item, dict) and _is_rate_limit_error(item.get("error"))
            for item in body
        )
    return False


def _block_param(number_or_tag: int | str) -> str:
    """Encode a block number as hex; pass tags such as ``latest`` through."""
    if isinstance(number_or_tag, int):
        if number_or_tag < 0:
            raise ValueError(f"Block number must be non-negative, got {number_or_tag}.")
        return hex(number_or_tag)
    return number_or_tag


def _with_int_fields(block: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a block object with ``number`` and ``timestamp`` as ints."""
    converted = dict(block)
    for field in ("number", "timestamp"):
        if isinstance(converted.get(field), str):
            converted[field] = int(converted[field], 16)
    return converted


class RpcClient:
    """Synchronous JSON-RPC client over HTTP with retries and batch support.

    Parameters
    ----------
    url:
        Endpoint URL. Defaults to the ``ALCHEMY_RPC_URL`` environment variable.

    Raises
    ------
    RpcError
        If no URL is configured.
    """

    def __init__(self, url: str | None = None) -> None:
        resolved = (url if url is not None else os.getenv("ALCHEMY_RPC_URL", "")).strip()
        if not resolved:
            raise RpcError("ALCHEMY_RPC_URL is missing. Set it in the project .env file.")
        self._url = resolved
        self._session = Session()
        self._ids = itertools.count(1)
        self._batch_supported = True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(endpoint=<redacted>)"

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def _redact(self, text: str) -> str:
        """Strip the endpoint URL and its path (which carries the API key) from text."""
        secrets = {self._url, urlparse(self._url).path}
        for secret in sorted((s for s in secrets if len(s) > 1), key=len, reverse=True):
            text = text.replace(secret, "<rpc-url>")
        return text

    def _format_error(self, error: Any) -> str:
        """Summarize a JSON-RPC error object for an exception message."""
        if isinstance(error, dict):
            message = self._redact(str(error.get("message", "")))
            return f"code {error.get('code')}: {message}"
        return self._redact(str(error))

    def _post(self, payload: Any, label: str) -> tuple[int, Any]:
        """Send one HTTP POST and decode the JSON body.

        Raises ``_Transient`` for HTTP 429, HTTP 5xx, timeouts, connection
        errors and non-JSON 200 responses. Other HTTP statuses are returned
        to the caller together with their decoded body (or None).
        """
        try:
            response = self._session.post(
                self._url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except (Timeout, RequestsConnectionError) as exc:
            # The exception text can echo the URL, so only the type name is kept.
            raise _Transient(type(exc).__name__) from None
        except RequestException as exc:
            raise RpcError(f"{label} failed: {type(exc).__name__}") from None

        status = response.status_code
        if status == 429 or status >= 500:
            raise _Transient(f"HTTP {status}")
        try:
            body = response.json()
        except ValueError:
            if status == 200:
                raise _Transient("non-JSON response") from None
            body = None
        return status, body

    def _request(self, payload: Any, label: str) -> tuple[int, Any]:
        """POST with exponential backoff on 429, 5xx, timeouts and rate-limit errors."""
        reason = "unknown"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                status, body = self._post(payload, label)
                if not _has_rate_limit_error(body):
                    return status, body
                reason = "JSON-RPC rate limit"
            except _Transient as transient:
                reason = str(transient)
            if attempt == MAX_ATTEMPTS:
                break
            wait_seconds = min(
                RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * 2 ** (attempt - 1)
            )
            logger.warning(
                "%s: %s (attempt %s/%s). Retrying in %.1fs.",
                label,
                reason,
                attempt,
                MAX_ATTEMPTS,
                wait_seconds,
            )
            time.sleep(wait_seconds)
        raise RpcError(f"{label} failed after {MAX_ATTEMPTS} attempts: {reason}")

    def _payload(self, method: str, params: list[Any]) -> dict[str, Any]:
        """Build one JSON-RPC request object with a fresh id."""
        return {
            "jsonrpc": "2.0",
            "id": next(self._ids),
            "method": method,
            "params": params,
        }

    def call(self, method: str, params: list[Any] | None = None) -> Any:
        """Run one JSON-RPC method and return its ``result``.

        Parameters
        ----------
        method:
            JSON-RPC method name, such as ``eth_blockNumber``.
        params:
            Positional parameters for the method.

        Returns
        -------
        Any
            The decoded ``result`` field.

        Raises
        ------
        RpcError
            On a permanent JSON-RPC error (for example a reverted ``eth_call``),
            an HTTP error other than 429/5xx, or after retries are exhausted.
        """
        label = f"RPC {method}"
        payload = self._payload(method, list(params or []))
        status, body = self._request(payload, label)
        if isinstance(body, dict) and "error" in body:
            raise RpcError(f"{label} failed: {self._format_error(body['error'])}")
        if status >= 400 or not isinstance(body, dict) or "result" not in body:
            raise RpcError(f"{label} failed: HTTP {status}, unexpected response.")
        return body["result"]

    def get_block(self, number_or_tag: int | str) -> dict[str, Any]:
        """Fetch a block with ``eth_getBlockByNumber`` (transactions not included).

        Parameters
        ----------
        number_or_tag:
            Block number, or a tag such as ``"latest"``.

        Returns
        -------
        dict[str, Any]
            Block object with ``number`` and ``timestamp`` converted to int.
            Other fields keep the node's hex-string encoding.
        """
        block = self.call(
            "eth_getBlockByNumber", [_block_param(number_or_tag), False]
        )
        if block is None:
            raise RpcError(f"Block {number_or_tag!r} was not found.")
        return _with_int_fields(block)

    def eth_call(self, to: str, data: str, block: int | str) -> str:
        """Run a read-only contract call at a given block.

        Parameters
        ----------
        to:
            Contract address (0x-prefixed).
        data:
            ABI-encoded calldata (0x-prefixed), for example from ``eth_abi``.
        block:
            Block number, or a tag such as ``"latest"``.

        Returns
        -------
        str
            Raw 0x-prefixed hex result. Decode it with ``eth_abi``.
        """
        result = self.call("eth_call", [{"to": to, "data": data}, _block_param(block)])
        if not isinstance(result, str):
            raise RpcError("eth_call returned a non-hex result.")
        return result

    def batch_call(self, requests: list[tuple[str, list[Any]]]) -> list[Any]:
        """Run several JSON-RPC calls and return their raw results in input order.

        Sends one JSON-RPC batch request. If the endpoint rejects batches, the
        calls are sent one by one instead, and later calls skip the batch attempt.

        Parameters
        ----------
        requests:
            ``(method, params)`` pairs.

        Returns
        -------
        list[Any]
            ``result`` for each call, in the order given. Results are not
            converted, so block objects keep hex fields.

        Raises
        ------
        RpcError
            If any call returns a JSON-RPC error or retries are exhausted.
        """
        if not requests:
            return []
        if self._batch_supported:
            payloads = [self._payload(method, list(params)) for method, params in requests]
            status, body = self._request(payloads, "RPC batch")
            if status < 400 and isinstance(body, list):
                return self._collect_batch(payloads, body)
            logger.info(
                "Endpoint did not accept JSON-RPC batches (HTTP %s); "
                "switching to sequential calls.",
                status,
            )
            self._batch_supported = False
        return [self.call(method, list(params)) for method, params in requests]

    def _collect_batch(self, payloads: list[dict[str, Any]], body: list[Any]) -> list[Any]:
        """Match batch responses to requests by id and return results in order."""
        by_id: dict[Any, dict[str, Any]] = {}
        for item in body:
            if not isinstance(item, dict) or "id" not in item:
                raise RpcError("RPC batch returned a malformed response.")
            by_id[item["id"]] = item

        results: list[Any] = []
        for payload in payloads:
            item = by_id.get(payload["id"])
            if item is None:
                raise RpcError(f"RPC batch response is missing {payload['method']}.")
            if "error" in item:
                raise RpcError(
                    f"RPC {payload['method']} failed: {self._format_error(item['error'])}"
                )
            results.append(item.get("result"))
        return results
