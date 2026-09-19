"""Blind snapshot extractor of Aave v3 Ethereum wallets at historical blocks."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "01_raw"
BASELINE_DIR = RAW_DATA_DIR / "baseline_mar2024"
STRESS_DIR = RAW_DATA_DIR / "stress_aug2024"

load_dotenv(PROJECT_ROOT / ".env")

GRAPH_API_KEY = os.getenv("GRAPH_API_KEY", "").strip()
AAVE_V3_ETHEREUM_SUBGRAPH_ID = "Cd2gEDVeqnjBn1hSeqFMitw8Q1iiyV9FYUZkLNRcL87g"
GRAPH_GATEWAY_BASE = "https://gateway-arbitrum.network.thegraph.com/api"

PAGE_SIZE = 1000
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2.0
PAGE_SLEEP_SECONDS = 0.35

BASELINE_BLOCK = 19_433_100
STRESS_BLOCK = 20_457_180

USERS_AT_BLOCK_QUERY = """
query FetchUsersAtBlock($first: Int!, $lastId: ID!, $blockNumber: Int!) {
  users(
    first: $first
    orderBy: id
    orderDirection: asc
    block: { number: $blockNumber }
    where: { id_gt: $lastId }
  ) {
    id
  }
}
"""


class GraphAPIError(RuntimeError):
    """Raised when The Graph gateway cannot be queried successfully."""


def _build_subgraph_url() -> str:
    """Build the decentralized Aave v3 Ethereum subgraph endpoint."""
    if not GRAPH_API_KEY:
        raise GraphAPIError(
            "GRAPH_API_KEY is missing. Set it in the project .env file."
        )
    return (
        f"{GRAPH_GATEWAY_BASE}/{GRAPH_API_KEY}/subgraphs/id/"
        f"{AAVE_V3_ETHEREUM_SUBGRAPH_ID}"
    )


def ensure_raw_scenario_directories() -> None:
    """Create raw data scenario directories if they do not already exist."""
    for directory in (BASELINE_DIR, STRESS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory exists: %s", directory)


def _post_graphql(
    query: str,
    variables: dict[str, Any],
    session: requests.Session,
) -> dict[str, Any]:
    """Execute a GraphQL request against The Graph with retries."""
    url = _build_subgraph_url()
    payload = {"query": query, "variables": variables}
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.post(
                url,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code in {429, 502, 503, 504}:
                wait_seconds = RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "The Graph returned HTTP %s (attempt %s/%s). Retrying in %.1fs.",
                    response.status_code,
                    attempt,
                    MAX_RETRIES,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise GraphAPIError(f"Unexpected GraphQL payload type: {type(body)}")
            if body.get("errors"):
                raise GraphAPIError(f"GraphQL errors: {body['errors']}")
            data = body.get("data")
            if not isinstance(data, dict):
                raise GraphAPIError(
                    f"GraphQL response is missing a data object: {body}"
                )
            return data
        except (requests.RequestException, ValueError, GraphAPIError) as exc:
            last_error = exc
            wait_seconds = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "The Graph request failed (attempt %s/%s): %s. Retrying in %.1fs.",
                attempt,
                MAX_RETRIES,
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise GraphAPIError(
        f"Failed to query The Graph after {MAX_RETRIES} attempts."
    ) from last_error


def fetch_users_at_block(
    block_number: int,
    session: requests.Session,
) -> list[dict[str, Any]]:
    """Fetch every Aave v3 ``users`` row indexed at a historical block.

    Pagination uses ``first: 1000`` and ``id_gt`` because The Graph caps
    each response at 1,000 entities and ``skip`` is not reliable past ~5,000.

    Parameters
    ----------
    block_number:
        Ethereum block height used as the GraphQL ``block: {number: ...}``
        time-travel argument.
    session:
        Shared ``requests.Session`` for connection reuse.

    Returns
    -------
    list[dict[str, Any]]
        Raw ``users`` objects containing at least an ``id`` field.
    """
    if block_number <= 0:
        raise ValueError(f"block_number must be a positive integer, got {block_number}.")

    records: list[dict[str, Any]] = []
    last_id = ""
    page_number = 0

    logger.info(
        "Starting users extraction at block %s (page size=%s).",
        block_number,
        PAGE_SIZE,
    )

    while True:
        page_number += 1
        variables = {
            "first": PAGE_SIZE,
            "lastId": last_id,
            "blockNumber": block_number,
        }
        data = _post_graphql(USERS_AT_BLOCK_QUERY, variables, session)
        page = data.get("users")
        if not isinstance(page, list):
            raise GraphAPIError(f"Expected a list for 'users', got {type(page)}.")

        records.extend(page)
        logger.info(
            "Fetched users page %s at block %s (%s rows, %s cumulative).",
            page_number,
            block_number,
            len(page),
            len(records),
        )

        if len(page) < PAGE_SIZE:
            break

        next_id = page[-1].get("id")
        if not next_id or next_id == last_id:
            logger.warning(
                "Pagination cursor did not advance at block %s; stopping at %s rows.",
                block_number,
                len(records),
            )
            break

        last_id = str(next_id)
        time.sleep(PAGE_SLEEP_SECONDS)

    logger.info(
        "Completed users extraction at block %s: %s raw rows.",
        block_number,
        len(records),
    )
    return records


def extract_unique_wallets(users: list[dict[str, Any]]) -> list[str]:
    """Return a sorted unique list of lowercase wallet addresses.

    Parameters
    ----------
    users:
        Raw GraphQL ``users`` payloads. Only the ``id`` field is consumed.

    Returns
    -------
    list[str]
        Deduplicated, lowercase Ethereum addresses, sorted for reproducibility.
    """
    wallets: set[str] = set()
    skipped = 0

    for user in users:
        wallet_id = user.get("id")
        if not isinstance(wallet_id, str) or not wallet_id.strip():
            skipped += 1
            continue
        wallets.add(wallet_id.strip().lower())

    if skipped:
        logger.warning("Skipped %s users with a missing or invalid id.", skipped)

    unique_wallets = sorted(wallets)
    logger.info("Standardized %s unique wallet addresses.", len(unique_wallets))
    return unique_wallets


def save_wallets_json(wallets: list[str], output_path: Path) -> None:
    """Write a JSON array of wallet address strings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(wallets, handle, ensure_ascii=False, indent=2)
    logger.info("Wrote %s wallets to %s", len(wallets), output_path)


def extract_active_wallets(block_number: int, output_path: Path) -> list[str]:
    """Extract a blind unique-wallet snapshot of Aave v3 at a given block.

    Parameters
    ----------
    block_number:
        Historical Ethereum block to time-travel the subgraph query.
    output_path:
        Destination JSON file for the unique wallet list.

    Returns
    -------
    list[str]
        Unique lowercase wallet addresses registered in the protocol
        up to ``block_number``.
    """
    try:
        with requests.Session() as session:
            users = fetch_users_at_block(block_number, session)
    except GraphAPIError:
        logger.exception(
            "Unable to extract Aave v3 users at block %s.",
            block_number,
        )
        raise

    wallets = extract_unique_wallets(users)
    save_wallets_json(wallets, output_path)
    return wallets


def run_extraction() -> None:
    """Download unique Aave v3 wallets for the baseline and stress blocks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    ensure_raw_scenario_directories()

    scenarios: list[tuple[str, int, Path]] = [
        ("baseline", BASELINE_BLOCK, BASELINE_DIR / "all_active_wallets.json"),
        ("stress", STRESS_BLOCK, STRESS_DIR / "all_active_wallets.json"),
    ]

    for name, block_number, output_path in scenarios:
        logger.info(
            "Starting %s wallet snapshot at block %s -> %s.",
            name,
            block_number,
            output_path,
        )
        try:
            wallets = extract_active_wallets(block_number, output_path)
            logger.info(
                "Finished %s snapshot: %s unique wallets at block %s.",
                name,
                len(wallets),
                block_number,
            )
        except (GraphAPIError, ValueError, requests.RequestException, OSError) as exc:
            logger.error(
                "Failed to extract %s wallets at block %s: %s",
                name,
                block_number,
                exc,
            )
            raise


if __name__ == "__main__":
    run_extraction()
