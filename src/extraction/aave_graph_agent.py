"""Aave v3 Ethereum daily snapshots from The Graph, queried with time-travel.

The panel for the Markov chain must look exactly as if it had been queried on
each day of 2024. For every UTC day ``D`` of a scenario window the extractor
resolves the block ``B`` (first block with ``timestamp >= D 00:00:00 UTC``, see
``src.utils.blocks.daily_blocks``) and asks the Aave v3 Ethereum subgraph for
its state *at that block* with ``block: {number: B}``. The queries, the schema
names and the balance formula were validated in the T1.1 spike
(``docs/spikes/T1.1_graph_spike.md``).

Windows (inclusive, 91 days each, ``[t - 60, t + 30]``):

* baseline: ``2024-01-15 .. 2024-04-14`` (t = 2024-03-15), folder ``baseline_mar2024``.
* stress: ``2024-06-06 .. 2024-09-04`` (t = 2024-08-05), folder ``stress_aug2024``.

Outputs, under ``<out-root>/<scenario_folder>/``:

* ``thegraph_daily/positions_<YYYY-MM-DD>.csv.gz``: one row per (user, reserve)
  of every borrower at block ``B`` (``borrowedReservesCount > 0``) with any
  non-zero balance. Columns: ``date, block, user_id, emode_category_id,
  reserve_symbol, underlying_asset, decimals, usage_as_collateral_enabled,
  scaled_a_token_balance, scaled_variable_debt, current_stable_debt,
  liquidity_index, variable_borrow_index`` (raw integers, written as digits) and
  ``a_token_balance, variable_debt, stable_debt`` (floats in token units).
  Read the raw-integer columns with ``dtype=str`` (the indices have 27 digits,
  which do not fit in int64). ``emode_category_id`` is ``0`` when the user has no
  eMode category (the subgraph returns null), as on chain.
* ``thegraph_daily/metadata.json``: subgraph id, window, and per day the block,
  borrower count, row count and extraction timestamp.
* ``protocol/reserve_params_daily.csv`` and ``protocol/emode_categories_daily.csv``:
  reserve and eMode parameters per day, in the schema read by
  ``src.utils.protocol_params`` (fractions; liquidation bonus as the fraction
  above par, raw values below 10000 map to 0.0; lowercase addresses). The reserve
  file has one extra column, ``emode_id`` (``Reserve.eMode.id``, ``0`` if none),
  which the loader ignores: read the CSV directly to get the reserve -> eMode link.
* ``protocol/daily_blocks.csv``: date to block cache shared with the other agents.
* ``thegraph_events/liquidation_calls.csv`` and ``repays.csv``: all events with
  ``timestamp`` in ``[start 00:00 UTC, end + 1 day 00:00 UTC)``, plus
  ``thegraph_events/metadata.json`` (union window and total row counts). Extending
  the window (``--events-only --start ... --end ...``) fetches only the requested
  days and merges them into the existing files; rows already there are kept as is.

Balance formula (never use the subgraph's ``currentATokenBalance`` /
``currentVariableDebt``: they are frozen at the user's last own event, a median
of 76 days before the block)::

    a_token_balance = rayMul(scaledATokenBalance, reserve.liquidityIndex)
    variable_debt   = rayMul(scaledVariableDebt,  reserve.variableBorrowIndex)

with both indices taken at the same block ``B`` and integer arithmetic. Stable
debt is zero for every row (stable borrowing is disabled) and is copied from
``currentStableDebt``; the extractor logs a warning if it is ever non-zero.

Event USD prices (``collateral_asset_price_usd``, ``borrow_asset_price_usd``,
``asset_price_usd``) are kept as the subgraph reports them but are **not
reliable**: they are 0 in about 20-27 % of liquidations. Use the raw amounts and
the Aave oracle series instead. ``Repay`` has no ``onBehalfOf`` field: ``user_id``
is the debtor and ``repayer_id`` the payer.

Robustness: every request retries with exponential backoff on transport errors,
HTTP 408/429/5xx and on HTTP-200 responses that carry a GraphQL ``errors`` array
(for example ``bad indexers``). A day is written only after all its pages were
fetched and validated: reserve parameters first, then the positions file, which
is the per-day checkpoint (an existing file means the day is skipped). Files are
written to a temporary name and renamed. Failed days are reported at the end and
can be retried by running the same command again.

Usage, from the repository root::

    python -m src.extraction.aave_graph_agent [--scenario baseline|stress|all]
        [--start YYYY-MM-DD --end YYYY-MM-DD] [--out-root PATH]
        [--events-only | --positions-only]

``GRAPH_API_KEY`` is read from ``.env``; neither the key nor the gateway URL is
ever logged.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    # Allow `python src/extraction/aave_graph_agent.py` from the repository root.
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.utils.blocks import daily_blocks  # noqa: E402
from src.utils.protocol_params import (  # noqa: E402
    load_emode_categories,
    load_reserve_params,
    raw_bonus_to_fraction,
    raw_bps_to_fraction,
)

logger = logging.getLogger(__name__)

load_dotenv(PROJECT_ROOT / ".env")

RAW_DATA_DIR = PROJECT_ROOT / "data" / "01_raw"

AAVE_V3_ETHEREUM_SUBGRAPH_ID = "Cd2gEDVeqnjBn1hSeqFMitw8Q1iiyV9FYUZkLNRcL87g"
GRAPH_GATEWAY_BASE = "https://gateway-arbitrum.network.thegraph.com/api"

DATE_FORMAT = "%Y-%m-%d"
PAGE_SIZE = 1000  # safe maximum: some indexers reject first > 1000
NESTED_RESERVES_LIMIT = 100  # observed maximum per borrower: 22
PAGE_SLEEP_SECONDS = 0.2
REQUEST_TIMEOUT_SECONDS = 120
MAX_ATTEMPTS = 8
RETRY_BASE_SECONDS = 2.0
RETRY_MAX_SECONDS = 60.0
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
# GraphQL errors that a retry cannot fix (bad query, not a bad indexer).
NON_RETRYABLE_ERROR_MARKERS = (
    "cannot query field",
    "has no field",
    "unknown argument",
    "unknown type",
    "validation error",
    "syntax error",
    "must be between 0 and",
)
MAX_CONSECUTIVE_DAY_FAILURES = 3

RAY = 10**27
HALF_RAY = RAY // 2

PROTOCOL_DIR = "protocol"
DAILY_DIR = "thegraph_daily"
EVENTS_DIR = "thegraph_events"
BLOCKS_FILE = "daily_blocks.csv"
RESERVE_PARAMS_FILE = "reserve_params_daily.csv"
EMODE_CATEGORIES_FILE = "emode_categories_daily.csv"
LIQUIDATIONS_FILE = "liquidation_calls.csv"
REPAYS_FILE = "repays.csv"
METADATA_FILE = "metadata.json"

POSITION_COLUMNS = [
    "date",
    "block",
    "user_id",
    "emode_category_id",
    "reserve_symbol",
    "underlying_asset",
    "decimals",
    "usage_as_collateral_enabled",
    "scaled_a_token_balance",
    "scaled_variable_debt",
    "current_stable_debt",
    "liquidity_index",
    "variable_borrow_index",
    "a_token_balance",
    "variable_debt",
    "stable_debt",
]
RESERVE_PARAM_COLUMNS = [
    "date",
    "block",
    "symbol",
    "underlying_asset",
    "decimals",
    "ltv",
    "liquidation_threshold",
    "liquidation_bonus",
    "usage_as_collateral_enabled",
    "emode_id",
]
RESERVE_PARAM_KEY = ["date", "symbol"]
EMODE_COLUMNS = [
    "date",
    "block",
    "emode_id",
    "label",
    "ltv",
    "liquidation_threshold",
    "liquidation_bonus",
]
EMODE_KEY = ["date", "emode_id"]
LIQUIDATION_COLUMNS = [
    "id",
    "tx_hash",
    "action",
    "timestamp",
    "date",
    "user_id",
    "liquidator",
    "collateral_reserve_id",
    "collateral_symbol",
    "collateral_underlying_asset",
    "collateral_decimals",
    "principal_reserve_id",
    "principal_symbol",
    "principal_underlying_asset",
    "principal_decimals",
    "collateral_amount",
    "principal_amount",
    "collateral_asset_price_usd",
    "borrow_asset_price_usd",
]
REPAY_COLUMNS = [
    "id",
    "tx_hash",
    "action",
    "timestamp",
    "date",
    "user_id",
    "repayer_id",
    "reserve_id",
    "reserve_symbol",
    "underlying_asset",
    "decimals",
    "amount",
    "asset_price_usd",
    "use_a_tokens",
]

# ---------------------------------------------------------------------------
# Queries (validated in the T1.1 spike, scripts/spikes/graph_spike.py).
# ---------------------------------------------------------------------------

USERS_SNAPSHOT_QUERY = """
query UsersSnapshot($first: Int!, $lastId: ID!, $block: Int!) {
  users(
    first: $first
    orderBy: id
    orderDirection: asc
    block: { number: $block }
    where: { borrowedReservesCount_gt: 0, id_gt: $lastId }
  ) {
    id
    borrowedReservesCount
    eModeCategoryId { id }
    reserves(first: 100) {
      usageAsCollateralEnabledOnUser
      scaledATokenBalance
      scaledVariableDebt
      principalStableDebt
      currentStableDebt
      reserve { id symbol underlyingAsset decimals liquidityIndex variableBorrowIndex }
    }
  }
}
"""

RESERVES_QUERY = """
query ReservesAtBlock($block: Int!) {
  reserves(first: 200, block: { number: $block }) {
    id
    symbol
    underlyingAsset
    decimals
    usageAsCollateralEnabled
    baseLTVasCollateral
    reserveLiquidationThreshold
    reserveLiquidationBonus
    eMode { id }
  }
  emodeCategories(first: 50, block: { number: $block }) {
    id
    label
    ltv
    liquidationThreshold
    liquidationBonus
  }
}
"""

LIQUIDATIONS_QUERY = """
query Liquidations($first: Int!, $lastId: ID!, $t0: Int!, $t1: Int!) {
  liquidationCalls(
    first: $first
    orderBy: id
    orderDirection: asc
    where: { timestamp_gte: $t0, timestamp_lt: $t1, id_gt: $lastId }
  ) {
    id
    txHash
    action
    timestamp
    liquidator
    user { id }
    collateralAmount
    principalAmount
    collateralAssetPriceUSD
    borrowAssetPriceUSD
    collateralReserve { id symbol underlyingAsset decimals }
    principalReserve { id symbol underlyingAsset decimals }
  }
}
"""

REPAYS_QUERY = """
query Repays($first: Int!, $lastId: ID!, $t0: Int!, $t1: Int!) {
  repays(
    first: $first
    orderBy: id
    orderDirection: asc
    where: { timestamp_gte: $t0, timestamp_lt: $t1, id_gt: $lastId }
  ) {
    id
    txHash
    action
    timestamp
    user { id }
    repayer { id }
    amount
    assetPriceUSD
    useATokens
    reserve { id symbol underlyingAsset decimals }
  }
}
"""


@dataclass(frozen=True)
class Scenario:
    """Folder and inclusive extraction window of one scenario."""

    name: str
    folder: str
    start: str
    end: str


SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario("baseline", "baseline_mar2024", "2024-01-15", "2024-04-14"),
    "stress": Scenario("stress", "stress_aug2024", "2024-06-06", "2024-09-04"),
}


class GraphAPIError(RuntimeError):
    """Raised when a request still fails after all retries."""


class GraphFatalError(GraphAPIError):
    """Raised for failures a retry cannot fix (bad key, bad query); aborts the run."""


# ---------------------------------------------------------------------------
# GraphQL client
# ---------------------------------------------------------------------------


class GraphClient:
    """GraphQL client for the Aave v3 Ethereum subgraph with retries.

    Counts every HTTP request (including retries) in ``request_count`` and every
    answered query in ``query_count``. The API key and the gateway URL are
    removed from every message that leaves the class.

    Parameters
    ----------
    api_key:
        Gateway API key. Defaults to the ``GRAPH_API_KEY`` environment variable.
    subgraph_id:
        Deployment id of the subgraph.

    Raises
    ------
    GraphFatalError
        If no API key is configured.
    """

    def __init__(
        self,
        api_key: str | None = None,
        subgraph_id: str = AAVE_V3_ETHEREUM_SUBGRAPH_ID,
    ) -> None:
        key = (api_key if api_key is not None else os.getenv("GRAPH_API_KEY", "")).strip()
        if not key:
            raise GraphFatalError("GRAPH_API_KEY is missing. Set it in the project .env file.")
        self._key = key
        self._url = f"{GRAPH_GATEWAY_BASE}/{key}/subgraphs/id/{subgraph_id}"
        self._session = requests.Session()
        self.request_count = 0
        self.query_count = 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(endpoint=<redacted>)"

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "GraphClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _scrub(self, text: str) -> str:
        """Remove the API key and the endpoint URL from ``text``."""
        return text.replace(self._url, "<url>").replace(self._key, "<key>")

    def query(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Run one GraphQL query and return its ``data`` object.

        Retries up to ``MAX_ATTEMPTS`` times with exponential backoff and jitter
        on transport errors, retryable HTTP statuses, non-JSON bodies and
        HTTP-200 responses carrying GraphQL ``errors`` (for example
        ``bad indexers``).

        Raises
        ------
        GraphFatalError
            For non-retryable HTTP statuses (bad key, quota) and for GraphQL
            errors that denote an invalid query.
        GraphAPIError
            If all attempts fail.
        """
        payload = {"query": query, "variables": variables}
        reason = "no attempt made"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.request_count += 1
            try:
                response = self._session.post(
                    self._url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
                )
            except requests.RequestException as exc:
                # Exception text can echo the URL, so only the type name is kept.
                reason = type(exc).__name__
            else:
                if response.status_code in RETRYABLE_STATUS:
                    reason = f"HTTP {response.status_code}"
                elif response.status_code != 200:
                    snippet = self._scrub(response.text[:200])
                    raise GraphFatalError(f"HTTP {response.status_code}: {snippet}")
                else:
                    try:
                        body = response.json()
                    except ValueError:
                        body = None
                        reason = "response is not JSON"
                    if isinstance(body, dict):
                        errors = body.get("errors")
                        if errors:
                            message = self._scrub(json.dumps(errors))[:400]
                            if any(m in message.lower() for m in NON_RETRYABLE_ERROR_MARKERS):
                                raise GraphFatalError(f"GraphQL errors: {message}")
                            reason = f"GraphQL errors: {message}"
                        elif isinstance(body.get("data"), dict):
                            self.query_count += 1
                            return body["data"]
                        else:
                            reason = "response has no data object"
                    elif body is not None:
                        reason = "unexpected JSON payload"

            if attempt == MAX_ATTEMPTS:
                break
            wait = min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * 2 ** (attempt - 1))
            wait += random.uniform(0.0, 1.0)
            logger.warning(
                "Graph request failed (attempt %s/%s): %s. Retrying in %.1fs.",
                attempt,
                MAX_ATTEMPTS,
                reason,
                wait,
            )
            time.sleep(wait)
        raise GraphAPIError(f"Graph request failed after {MAX_ATTEMPTS} attempts: {reason}")

    def paginate(
        self,
        query: str,
        entity: str,
        variables: dict[str, Any],
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield the pages of ``entity``, using ``first: 1000`` and ``id_gt`` cursors.

        ``query`` must declare ``$first: Int!`` and ``$lastId: ID!`` and order by
        ``id`` ascending. Iteration stops at the first page shorter than
        ``PAGE_SIZE``.
        """
        last_id = ""
        while True:
            data = self.query(query, {**variables, "first": PAGE_SIZE, "lastId": last_id})
            page = data.get(entity)
            if not isinstance(page, list):
                raise GraphAPIError(f"Expected a list for '{entity}', got {type(page).__name__}.")
            yield page
            if len(page) < PAGE_SIZE:
                return
            next_id = str(page[-1]["id"])
            if next_id <= last_id:
                raise GraphAPIError(f"Pagination cursor for '{entity}' did not advance.")
            last_id = next_id
            time.sleep(PAGE_SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def ray_mul(a: int, b: int) -> int:
    """Aave ``rayMul``: ``a * b / 1e27`` rounded half up, in integer arithmetic."""
    return (a * b + HALF_RAY) // RAY


def _parse_date(value: str) -> date:
    """Parse ``YYYY-MM-DD`` into a ``date``."""
    try:
        return datetime.strptime(value, DATE_FORMAT).date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a YYYY-MM-DD date, got {value!r}.") from exc


def _utc_midnight_ts(day: date) -> int:
    """Return the Unix timestamp of 00:00:00 UTC on ``day``."""
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def _date_range(start: str, end: str) -> list[date]:
    """Return every date in ``[start, end]``."""
    first, last = _parse_date(start), _parse_date(end)
    if last < first:
        raise ValueError(f"end {end} is before start {start}.")
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def _utc_date_of(timestamp: int) -> str:
    """Return the UTC calendar date of a Unix timestamp as ``YYYY-MM-DD``."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(DATE_FORMAT)


def _now_utc() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _bonus_fraction(raw: int, what: str) -> float:
    """Convert a raw liquidation bonus to the fraction above par; raw below par maps to 0.0."""
    if raw == 0:
        return 0.0
    if raw < 10000:
        logger.warning("%s: liquidation bonus %s is below par; stored as 0.0.", what, raw)
        return 0.0
    return raw_bonus_to_fraction(raw)


def _check_param_ranges(rows: list[dict[str, Any]], what: str) -> None:
    """Reject rows that ``protocol_params`` would refuse to load.

    Raises
    ------
    ValueError
        If ``0 <= ltv <= liquidation_threshold < 1`` or ``0 <= liquidation_bonus < 0.5``
        fails for any row.
    """
    problems = []
    for row in rows:
        label = row.get("symbol", row.get("emode_id"))
        ltv, threshold, bonus = row["ltv"], row["liquidation_threshold"], row["liquidation_bonus"]
        if not 0 <= ltv <= threshold < 1:
            problems.append(f"{label}: ltv={ltv}, liquidation_threshold={threshold}")
        if not 0 <= bonus < 0.5:
            problems.append(f"{label}: liquidation_bonus={bonus}")
    if problems:
        raise ValueError(f"{what}: parameters out of range: {problems[:5]}")


def flatten_users(
    users: list[dict[str, Any]],
    day: str,
    block: int,
) -> tuple[list[dict[str, Any]], int]:
    """Flatten borrowers into position rows, computing accrued balances.

    Parameters
    ----------
    users:
        ``users`` entities from ``USERS_SNAPSHOT_QUERY`` at ``block``.
    day:
        Snapshot date, ``YYYY-MM-DD``.
    block:
        Block the snapshot was taken at.

    Returns
    -------
    tuple
        ``(rows, stable_debt_rows)``: one dict per (user, reserve) with any
        non-zero balance (keys are ``POSITION_COLUMNS``), and the number of rows
        with a non-zero stable debt (expected 0).
    """
    rows: list[dict[str, Any]] = []
    stable_rows = 0
    for user in users:
        category = user.get("eModeCategoryId")
        emode_id = int(category["id"]) if category else 0
        user_reserves = user["reserves"]
        if len(user_reserves) >= NESTED_RESERVES_LIMIT:
            logger.warning("User %s may have more than %s reserves.", user["id"], NESTED_RESERVES_LIMIT)
        for entry in user_reserves:
            scaled_a = int(entry["scaledATokenBalance"])
            scaled_v = int(entry["scaledVariableDebt"])
            stable = int(entry["currentStableDebt"])
            principal_stable = int(entry["principalStableDebt"])
            if not (scaled_a or scaled_v or stable or principal_stable):
                continue
            if stable or principal_stable:
                stable_rows += 1
            reserve = entry["reserve"]
            decimals = int(reserve["decimals"])
            liquidity_index = int(reserve["liquidityIndex"])
            borrow_index = int(reserve["variableBorrowIndex"])
            unit = 10**decimals
            rows.append(
                {
                    "date": day,
                    "block": block,
                    "user_id": user["id"],
                    "emode_category_id": emode_id,
                    "reserve_symbol": reserve["symbol"],
                    "underlying_asset": reserve["underlyingAsset"].lower(),
                    "decimals": decimals,
                    "usage_as_collateral_enabled": bool(entry["usageAsCollateralEnabledOnUser"]),
                    "scaled_a_token_balance": str(scaled_a),
                    "scaled_variable_debt": str(scaled_v),
                    "current_stable_debt": str(stable),
                    "liquidity_index": str(liquidity_index),
                    "variable_borrow_index": str(borrow_index),
                    "a_token_balance": ray_mul(scaled_a, liquidity_index) / unit,
                    "variable_debt": ray_mul(scaled_v, borrow_index) / unit,
                    "stable_debt": stable / unit,
                }
            )
    return rows, stable_rows


def build_param_rows(
    data: dict[str, Any],
    day: str,
    block: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert a ``RESERVES_QUERY`` answer into reserve and eMode parameter rows.

    Returns
    -------
    tuple
        ``(reserve_rows, emode_rows)`` keyed by ``RESERVE_PARAM_COLUMNS`` and
        ``EMODE_COLUMNS``, with LTV and liquidation threshold as fractions and
        the liquidation bonus as the fraction above par.

    Raises
    ------
    ValueError
        If symbols repeat within a day or a row is outside the ranges accepted
        by ``src.utils.protocol_params``.
    """
    reserve_rows = []
    for item in sorted(data["reserves"], key=lambda r: r["symbol"]):
        emode = item.get("eMode")
        reserve_rows.append(
            {
                "date": day,
                "block": block,
                "symbol": item["symbol"],
                "underlying_asset": item["underlyingAsset"].lower(),
                "decimals": int(item["decimals"]),
                "ltv": raw_bps_to_fraction(int(item["baseLTVasCollateral"])),
                "liquidation_threshold": raw_bps_to_fraction(int(item["reserveLiquidationThreshold"])),
                "liquidation_bonus": _bonus_fraction(
                    int(item["reserveLiquidationBonus"]), f"reserve {item['symbol']} on {day}"
                ),
                "usage_as_collateral_enabled": bool(item["usageAsCollateralEnabled"]),
                "emode_id": int(emode["id"]) if emode else 0,
            }
        )
    symbols = [row["symbol"] for row in reserve_rows]
    if len(set(symbols)) != len(symbols):
        raise ValueError(f"Duplicate reserve symbols on {day} (block {block}).")
    if not reserve_rows:
        raise ValueError(f"No reserves returned for {day} (block {block}).")

    emode_rows = []
    for item in sorted(data["emodeCategories"], key=lambda c: int(c["id"])):
        emode_rows.append(
            {
                "date": day,
                "block": block,
                "emode_id": int(item["id"]),
                "label": item["label"] or f"category_{item['id']}",
                "ltv": raw_bps_to_fraction(int(item["ltv"])),
                "liquidation_threshold": raw_bps_to_fraction(int(item["liquidationThreshold"])),
                "liquidation_bonus": _bonus_fraction(
                    int(item["liquidationBonus"]), f"eMode {item['id']} on {day}"
                ),
            }
        )
    _check_param_ranges(reserve_rows, f"reserves on {day}")
    _check_param_ranges(emode_rows, f"eMode categories on {day}")
    return reserve_rows, emode_rows


def liquidation_row(item: dict[str, Any]) -> dict[str, Any]:
    """Flatten one ``liquidationCalls`` entity (raw amounts stay as digit strings)."""
    collateral, principal = item["collateralReserve"], item["principalReserve"]
    timestamp = int(item["timestamp"])
    return {
        "id": item["id"],
        "tx_hash": str(item["txHash"]).lower(),
        "action": item.get("action"),
        "timestamp": timestamp,
        "date": _utc_date_of(timestamp),
        "user_id": item["user"]["id"],
        "liquidator": str(item["liquidator"]).lower(),
        "collateral_reserve_id": collateral["id"],
        "collateral_symbol": collateral["symbol"],
        "collateral_underlying_asset": collateral["underlyingAsset"].lower(),
        "collateral_decimals": int(collateral["decimals"]),
        "principal_reserve_id": principal["id"],
        "principal_symbol": principal["symbol"],
        "principal_underlying_asset": principal["underlyingAsset"].lower(),
        "principal_decimals": int(principal["decimals"]),
        "collateral_amount": str(item["collateralAmount"]),
        "principal_amount": str(item["principalAmount"]),
        "collateral_asset_price_usd": item["collateralAssetPriceUSD"],
        "borrow_asset_price_usd": item["borrowAssetPriceUSD"],
    }


def repay_row(item: dict[str, Any]) -> dict[str, Any]:
    """Flatten one ``repays`` entity (``user`` is the debtor, ``repayer`` the payer)."""
    reserve = item["reserve"]
    timestamp = int(item["timestamp"])
    return {
        "id": item["id"],
        "tx_hash": str(item["txHash"]).lower(),
        "action": item.get("action"),
        "timestamp": timestamp,
        "date": _utc_date_of(timestamp),
        "user_id": item["user"]["id"],
        "repayer_id": item["repayer"]["id"],
        "reserve_id": reserve["id"],
        "reserve_symbol": reserve["symbol"],
        "underlying_asset": reserve["underlyingAsset"].lower(),
        "decimals": int(reserve["decimals"]),
        "amount": str(item["amount"]),
        "asset_price_usd": item["assetPriceUSD"],
        "use_a_tokens": bool(item["useATokens"]),
    }


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def _tmp_path(path: Path) -> Path:
    """Return the temporary sibling used for atomic writes."""
    return path.with_name(path.name + ".tmp")


def _write_frame(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    """Write ``frame`` to ``path`` atomically (temporary file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    frame.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Write ``payload`` as JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, or ``{}`` if the file is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _csv_text(value: Any) -> str:
    """Format one parameter cell as text (booleans lowercase, as the loader expects)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def upsert_param_rows(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
    key: list[str],
) -> None:
    """Merge ``rows`` into the parameter CSV at ``path``, de-duplicating on ``key``.

    Existing rows with the same key are replaced by the new ones. Cells are kept
    as text so values already on disk are never reformatted. The result is
    sorted by date (stable) and written atomically.
    """
    new = pd.DataFrame([{c: _csv_text(r[c]) for c in columns} for r in rows], columns=columns)
    if path.is_file():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        combined = pd.concat([old.reindex(columns=columns, fill_value=""), new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(subset=key, keep="last")
    combined = combined.sort_values("date", kind="stable").reset_index(drop=True)
    _write_frame(combined, path)


# ---------------------------------------------------------------------------
# Daily snapshots
# ---------------------------------------------------------------------------


def positions_path(scenario_dir: Path, day: str) -> Path:
    """Return the positions file of ``day`` (its existence is the day's checkpoint)."""
    return scenario_dir / DAILY_DIR / f"positions_{day}.csv.gz"


def extract_day(
    client: GraphClient,
    scenario_dir: Path,
    day: str,
    block: int,
) -> dict[str, Any]:
    """Fetch and write the snapshot of one day at one block.

    Everything is fetched and validated in memory first. Reserve and eMode
    parameters are written before the positions file, because the positions
    file is the checkpoint that marks the day as done.

    Returns
    -------
    dict
        Per-day metadata: ``block``, ``users`` (borrowers returned), ``rows``,
        ``stable_debt_rows``, ``graph_requests`` and ``extracted_at_utc``.
    """
    requests_before = client.request_count
    params = client.query(RESERVES_QUERY, {"block": block})
    reserve_rows, emode_rows = build_param_rows(params, day, block)

    n_users = 0
    rows: list[dict[str, Any]] = []
    stable_rows = 0
    for page in client.paginate(USERS_SNAPSHOT_QUERY, "users", {"block": block}):
        page_rows, page_stable_rows = flatten_users(page, day, block)
        n_users += len(page)
        rows.extend(page_rows)
        stable_rows += page_stable_rows
    if not n_users:
        raise GraphAPIError(f"No borrowers returned for {day} (block {block}).")
    if stable_rows:
        logger.warning(
            "%s: %s rows with non-zero stable debt (stored value may be stale).", day, stable_rows
        )

    protocol_dir = scenario_dir / PROTOCOL_DIR
    upsert_param_rows(
        protocol_dir / RESERVE_PARAMS_FILE, reserve_rows, RESERVE_PARAM_COLUMNS, RESERVE_PARAM_KEY
    )
    if emode_rows:
        upsert_param_rows(
            protocol_dir / EMODE_CATEGORIES_FILE, emode_rows, EMODE_COLUMNS, EMODE_KEY
        )
    _write_frame(
        pd.DataFrame(rows, columns=POSITION_COLUMNS),
        positions_path(scenario_dir, day),
        compression="gzip",
    )
    return {
        "block": block,
        "users": n_users,
        "rows": len(rows),
        "stable_debt_rows": stable_rows,
        "graph_requests": client.request_count - requests_before,
        "extracted_at_utc": _now_utc(),
    }


def _derive_day_metadata(path: Path) -> dict[str, Any]:
    """Rebuild a day's metadata from an existing positions file."""
    frame = pd.read_csv(path, usecols=["block", "user_id"], dtype={"user_id": str})
    return {
        "block": int(frame["block"].iloc[0]) if len(frame) else None,
        "users": int(frame["user_id"].nunique()),
        "rows": int(len(frame)),
        "derived_from_file": True,
        "extracted_at_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(
            timespec="seconds"
        ),
    }


def _write_daily_metadata(
    scenario: Scenario, scenario_dir: Path, days: dict[str, dict[str, Any]]
) -> None:
    """Write ``thegraph_daily/metadata.json`` for all days known so far."""
    ordered = {day: days[day] for day in sorted(days)}
    payload = {
        "subgraph_id": AAVE_V3_ETHEREUM_SUBGRAPH_ID,
        "scenario": scenario.folder,
        "first_date": next(iter(ordered), None),
        "last_date": next(reversed(ordered), None),
        "n_days": len(ordered),
        "balance_formula": "rayMul(scaled balance, reserve index at the snapshot block)",
        "extraction_timestamp_utc": _now_utc(),
        "days": ordered,
    }
    _write_json(payload, scenario_dir / DAILY_DIR / METADATA_FILE)


def extract_positions(
    client: GraphClient,
    scenario: Scenario,
    start: str,
    end: str,
    out_root: Path,
) -> list[str]:
    """Extract the daily snapshots of ``[start, end]`` for one scenario.

    Days whose positions file already exists are skipped. A day that fails after
    all retries is logged and skipped; the run aborts after
    ``MAX_CONSECUTIVE_DAY_FAILURES`` failures in a row or on a fatal error.

    Returns
    -------
    list[str]
        Dates that failed (empty when everything succeeded).
    """
    scenario_dir = out_root / scenario.folder
    days = [d.isoformat() for d in _date_range(start, end)]
    pending = [d for d in days if not positions_path(scenario_dir, d).is_file()]
    logger.info(
        "%s: %s days in [%s, %s], %s already done, %s to fetch.",
        scenario.folder,
        len(days),
        start,
        end,
        len(days) - len(pending),
        len(pending),
    )

    metadata_path = scenario_dir / DAILY_DIR / METADATA_FILE
    day_meta: dict[str, dict[str, Any]] = dict(_read_json(metadata_path).get("days", {}))
    derived = False
    for day in days:
        if day not in pending and day not in day_meta:
            day_meta[day] = _derive_day_metadata(positions_path(scenario_dir, day))
            derived = True

    failed: list[str] = []
    if pending:
        blocks = daily_blocks(
            pending[0], pending[-1], cache_path=scenario_dir / PROTOCOL_DIR / BLOCKS_FILE
        ).set_index("date")["block"]
        consecutive = 0
        for position, day in enumerate(pending, start=1):
            block = int(blocks[day])
            t0 = time.perf_counter()
            try:
                day_meta[day] = extract_day(client, scenario_dir, day, block)
            except GraphFatalError:
                _write_daily_metadata(scenario, scenario_dir, day_meta)
                raise
            except (GraphAPIError, ValueError) as exc:
                logger.error("%s: day failed (block %s): %s", day, block, exc)
                failed.append(day)
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_DAY_FAILURES:
                    _write_daily_metadata(scenario, scenario_dir, day_meta)
                    raise GraphAPIError(
                        f"{consecutive} consecutive day failures; aborting. Failed: {failed}"
                    ) from exc
                continue
            consecutive = 0
            info = day_meta[day]
            logger.info(
                "%s (%s/%s) block %s: %s borrowers, %s rows, %s requests, %.1fs.",
                day,
                position,
                len(pending),
                block,
                info["users"],
                info["rows"],
                info["graph_requests"],
                time.perf_counter() - t0,
            )
            _write_daily_metadata(scenario, scenario_dir, day_meta)

    if derived and not pending:
        _write_daily_metadata(scenario, scenario_dir, day_meta)
    return failed


def validate_protocol_files(scenario: Scenario, out_root: Path) -> None:
    """Load both parameter files with ``src.utils.protocol_params`` as a final check.

    Raises
    ------
    ValueError
        If either file fails the loader's validation.
    """
    reserves = load_reserve_params(scenario.folder, raw_dir=out_root)
    emodes = load_emode_categories(scenario.folder, raw_dir=out_root)
    logger.info(
        "%s: protocol files validated (%s reserve rows over %s dates, %s eMode rows).",
        scenario.folder,
        len(reserves),
        reserves["date"].nunique(),
        len(emodes),
    )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _fetch_events(
    client: GraphClient,
    query: str,
    entity: str,
    to_row: Any,
    t0: int,
    t1: int,
    columns: list[str],
) -> pd.DataFrame:
    """Fetch every ``entity`` event with ``t0 <= timestamp < t1`` as a sorted table."""
    rows: dict[str, dict[str, Any]] = {}
    for page in client.paginate(query, entity, {"t0": t0, "t1": t1}):
        for item in page:
            rows[item["id"]] = to_row(item)
    frame = pd.DataFrame(list(rows.values()), columns=columns)
    return frame.sort_values(["timestamp", "id"], kind="stable").reset_index(drop=True)


def _as_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with every cell as the text ``DataFrame.to_csv`` would write."""
    return pd.read_csv(io.StringIO(frame.to_csv(index=False)), dtype=str, keep_default_na=False)


def _read_event_csv(path: Path) -> pd.DataFrame:
    """Read an events CSV as text, so rewriting it reproduces the rows already on disk."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def merge_event_frames(
    existing: pd.DataFrame | None, new: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Merge a freshly fetched event table into the one already on disk.

    Rows are de-duplicated on ``id`` (the existing row wins) and sorted by
    ``(timestamp, id)``, the order of ``_fetch_events``. Every cell is returned as
    text (see ``_as_text``), so the rows of ``existing`` are written back unchanged;
    ``existing`` must have been read with ``dtype=str`` (see ``_read_event_csv``).

    Raises
    ------
    ValueError
        If ``existing`` does not have exactly ``columns``.
    """
    parts = [_as_text(new[columns])]
    if existing is not None:
        if list(existing.columns) != columns:
            raise ValueError(f"Existing events have columns {list(existing.columns)}, expected {columns}.")
        parts.insert(0, _as_text(existing))
    merged = pd.concat(parts, ignore_index=True).drop_duplicates("id", keep="first")
    merged = merged.assign(_timestamp=pd.to_numeric(merged["timestamp"]))
    merged = merged.sort_values(["_timestamp", "id"], kind="stable").drop(columns="_timestamp")
    return merged.reset_index(drop=True)


def plan_event_fetch(
    recorded: tuple[int, int] | None, t0: int, t1: int
) -> tuple[int, int, int, int] | None:
    """Decide what to fetch for the requested ``[t0, t1)`` given the recorded window.

    Returns ``None`` when the request lies inside ``recorded`` (nothing to do), else
    ``(fetch_t0, fetch_t1, union_t0, union_t1)``: the interval to fetch and the
    window the files cover afterwards. The fetch is the request itself, extended up
    to the recorded window when the two are disjoint with a gap, so that the union
    never has a hole.
    """
    if recorded is None:
        return t0, t1, t0, t1
    start, end = recorded
    if start <= t0 and t1 <= end:
        return None
    fetch_t0, fetch_t1 = t0, t1
    if t1 < start:
        fetch_t1 = start
    elif t0 > end:
        fetch_t0 = end
    return fetch_t0, fetch_t1, min(t0, start), max(t1, end)


def extract_events(
    client: GraphClient,
    scenario: Scenario,
    start: str,
    end: str,
    out_root: Path,
) -> dict[str, int]:
    """Extract liquidation calls and repays with timestamps in ``[start, end + 1 day)``.

    The window covered by the files is recorded in ``thegraph_events/metadata.json``.
    If the request lies inside it, nothing is fetched. Otherwise only the requested
    window is fetched and merged into the existing CSVs (see ``merge_event_frames``:
    existing rows are never changed), and the metadata records the union window and
    the total row counts. A rerun after an interruption between the CSV writes and
    the metadata write is safe: the same window is fetched again and de-duplicated.

    Returns
    -------
    dict
        Total row counts in the files: ``liquidation_calls`` and ``repays``.

    Raises
    ------
    ValueError
        If the CSVs exist but are incomplete or the metadata has no window; nothing
        is overwritten in that case.
    """
    events_dir = out_root / scenario.folder / EVENTS_DIR
    t0 = _utc_midnight_ts(_parse_date(start))
    t1 = _utc_midnight_ts(_parse_date(end) + timedelta(days=1))
    liquidations_path, repays_path = events_dir / LIQUIDATIONS_FILE, events_dir / REPAYS_FILE
    metadata_path = events_dir / METADATA_FILE

    previous = _read_json(metadata_path)
    on_disk = [path for path in (liquidations_path, repays_path) if path.is_file()]
    recorded: tuple[int, int] | None = None
    if on_disk:
        recorded_start, recorded_end = previous.get("start_timestamp"), previous.get("end_timestamp")
        if len(on_disk) < 2 or not isinstance(recorded_start, int) or not isinstance(recorded_end, int):
            raise ValueError(
                f"{events_dir}: events files are incomplete or {METADATA_FILE} has no window; "
                "refusing to overwrite them."
            )
        recorded = (recorded_start, recorded_end)

    plan = plan_event_fetch(recorded, t0, t1)
    if plan is None:
        logger.info("%s: events for [%s, %s] already extracted; skipping.", scenario.folder, start, end)
        return {k: int(previous.get(k, 0)) for k in ("liquidation_calls", "repays")}
    fetch_t0, fetch_t1, union_t0, union_t1 = plan
    if (fetch_t0, fetch_t1) != (t0, t1):
        logger.warning(
            "%s: [%s, %s] does not touch the recorded events window; fetching %s..%s to leave no gap.",
            scenario.folder,
            start,
            end,
            _utc_date_of(fetch_t0),
            _utc_date_of(fetch_t1 - 1),
        )

    liquidations = _fetch_events(
        client, LIQUIDATIONS_QUERY, "liquidationCalls", liquidation_row, fetch_t0, fetch_t1, LIQUIDATION_COLUMNS
    )
    repays = _fetch_events(client, REPAYS_QUERY, "repays", repay_row, fetch_t0, fetch_t1, REPAY_COLUMNS)
    fetched = {"liquidation_calls": len(liquidations), "repays": len(repays)}
    liquidations = merge_event_frames(
        _read_event_csv(liquidations_path) if recorded else None, liquidations, LIQUIDATION_COLUMNS
    )
    repays = merge_event_frames(_read_event_csv(repays_path) if recorded else None, repays, REPAY_COLUMNS)
    _write_frame(liquidations, liquidations_path)
    _write_frame(repays, repays_path)
    window_start, window_end = _utc_date_of(union_t0), _utc_date_of(union_t1 - 1)
    _write_json(
        {
            "subgraph_id": AAVE_V3_ETHEREUM_SUBGRAPH_ID,
            "scenario": scenario.folder,
            "window_start": window_start,
            "window_end_inclusive": window_end,
            "start_timestamp": union_t0,
            "end_timestamp": union_t1,
            "liquidation_calls": len(liquidations),
            "repays": len(repays),
            "extraction_timestamp_utc": _now_utc(),
            "note": "USD price fields come from the subgraph and are 0 in 20-27% of liquidations; do not use.",
        },
        metadata_path,
    )
    logger.info(
        "%s: fetched %s liquidation calls and %s repays in [%s, %s]; files now hold %s and %s in [%s, %s].",
        scenario.folder,
        fetched["liquidation_calls"],
        fetched["repays"],
        _utc_date_of(fetch_t0),
        _utc_date_of(fetch_t1 - 1),
        len(liquidations),
        len(repays),
        window_start,
        window_end,
    )
    return {"liquidation_calls": len(liquidations), "repays": len(repays)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract daily Aave v3 time-travel snapshots, reserve parameters and events."
    )
    parser.add_argument(
        "--scenario",
        choices=["baseline", "stress", "all"],
        default="all",
        help="Scenario window to extract (default: all).",
    )
    parser.add_argument("--start", help="Override window start, YYYY-MM-DD (needs --end).")
    parser.add_argument("--end", help="Override window end, YYYY-MM-DD (needs --start).")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=RAW_DATA_DIR,
        help="Root folder that holds the scenario folders (default: data/01_raw).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--events-only", action="store_true", help="Only extract the events.")
    mode.add_argument(
        "--positions-only",
        action="store_true",
        help="Only extract daily positions and reserve parameters.",
    )
    args = parser.parse_args(argv)
    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must be given together.")
    if args.start is not None:
        if args.scenario == "all":
            parser.error("--start/--end need a single --scenario (baseline or stress).")
        try:
            _date_range(args.start, args.end)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the extraction for the selected scenarios and return the exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    started = time.perf_counter()
    failures: dict[str, list[str]] = {}

    try:
        with GraphClient() as client:
            for name in names:
                scenario = SCENARIOS[name]
                start = args.start or scenario.start
                end = args.end or scenario.end
                if not args.events_only:
                    failed = extract_positions(client, scenario, start, end, args.out_root)
                    if failed:
                        failures[scenario.folder] = failed
                    try:
                        validate_protocol_files(scenario, args.out_root)
                    except (ValueError, FileNotFoundError) as exc:
                        logger.error("%s: protocol file validation failed: %s", scenario.folder, exc)
                        failures.setdefault(scenario.folder, []).append("protocol_files")
                if not args.positions_only:
                    extract_events(client, scenario, start, end, args.out_root)
            logger.info(
                "Done in %.1f min: %s Graph requests (%s answered queries).",
                (time.perf_counter() - started) / 60,
                client.request_count,
                client.query_count,
            )
    except GraphAPIError as exc:
        logger.error("Aborted: %s. Completed days are kept; rerun the same command to resume.", exc)
        return 1
    if failures:
        logger.error("Incomplete extraction, rerun to retry: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
