"""Aave v3 Ethereum The Graph extractor for the Markov actuarial pipeline."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
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

WINDOW_DAYS = 30
PAGE_SIZE = 1000
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
PAGE_SLEEP_SECONDS = 0.25

BASELINE_END = datetime(2024, 3, 15, tzinfo=timezone.utc)
STRESS_END = datetime(2024, 8, 5, tzinfo=timezone.utc)

USERS_QUERY = """
query FetchUsers(
  $first: Int!
  $lastId: ID!
  $startTimestamp: Int!
  $endTimestamp: Int!
) {
  users(
    first: $first
    orderBy: id
    orderDirection: asc
    where: { borrowedReservesCount_gt: 0, id_gt: $lastId }
  ) {
    id
    borrowedReservesCount
    eModeCategoryId {
      id
      ltv
      liquidationThreshold
      liquidationBonus
      label
    }
    reserves(first: 100) {
      id
      usageAsCollateralEnabledOnUser
      scaledATokenBalance
      currentATokenBalance
      scaledVariableDebt
      currentVariableDebt
      principalStableDebt
      currentStableDebt
      currentTotalDebt
      lastUpdateTimestamp
      reserve {
        id
        symbol
        name
        decimals
        underlyingAsset
        baseLTVasCollateral
        reserveLiquidationThreshold
        reserveLiquidationBonus
        liquidityIndex
        variableBorrowIndex
        lastUpdateTimestamp
        price {
          priceInEth
        }
      }
      aTokenBalanceHistory(
        first: 1000
        orderBy: timestamp
        orderDirection: asc
        where: { timestamp_gte: $startTimestamp, timestamp_lte: $endTimestamp }
      ) {
        id
        timestamp
        scaledATokenBalance
        currentATokenBalance
        index
      }
      vTokenBalanceHistory(
        first: 1000
        orderBy: timestamp
        orderDirection: asc
        where: { timestamp_gte: $startTimestamp, timestamp_lte: $endTimestamp }
      ) {
        id
        timestamp
        scaledVariableDebt
        currentVariableDebt
        index
      }
      sTokenBalanceHistory(
        first: 1000
        orderBy: timestamp
        orderDirection: asc
        where: { timestamp_gte: $startTimestamp, timestamp_lte: $endTimestamp }
      ) {
        id
        timestamp
        principalStableDebt
        currentStableDebt
        avgStableBorrowRate
      }
    }
  }
}
"""

LIQUIDATION_CALLS_QUERY = """
query FetchLiquidationCalls(
  $first: Int!
  $lastId: ID!
  $startTimestamp: Int!
  $endTimestamp: Int!
) {
  liquidationCalls(
    first: $first
    orderBy: id
    orderDirection: asc
    where: {
      timestamp_gte: $startTimestamp
      timestamp_lte: $endTimestamp
      id_gt: $lastId
    }
  ) {
    id
    txHash
    action
    timestamp
    liquidator
    collateralAmount
    principalAmount
    collateralAssetPriceUSD
    borrowAssetPriceUSD
    user {
      id
    }
    collateralReserve {
      id
      symbol
      decimals
      underlyingAsset
      reserveLiquidationThreshold
      reserveLiquidationBonus
    }
    principalReserve {
      id
      symbol
      decimals
      underlyingAsset
    }
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


def _window_bounds(end_timestamp: int) -> tuple[int, int]:
    """Return the inclusive [start, end] Unix timestamps for a 30-day window."""
    if end_timestamp <= 0:
        raise ValueError(f"end_timestamp must be a positive Unix time, got {end_timestamp}.")
    start_timestamp = end_timestamp - WINDOW_DAYS * 24 * 60 * 60
    return start_timestamp, end_timestamp


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
                raise GraphAPIError(f"GraphQL response is missing a data object: {body}")
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


def _paginate_entity(
    query: str,
    entity_key: str,
    start_timestamp: int,
    end_timestamp: int,
    session: requests.Session,
) -> list[dict[str, Any]]:
    """Fetch every page of a GraphQL entity using cursor pagination (id_gt).

    The Graph caps ``skip`` around 5,000 records, so this extractor pages with
    ``first: 1000`` and ``id_gt`` instead of incrementing ``skip``.
    """
    records: list[dict[str, Any]] = []
    last_id = ""
    page_number = 0

    while True:
        page_number += 1
        variables = {
            "first": PAGE_SIZE,
            "lastId": last_id,
            "startTimestamp": start_timestamp,
            "endTimestamp": end_timestamp,
        }
        data = _post_graphql(query, variables, session)
        page = data.get(entity_key)
        if not isinstance(page, list):
            raise GraphAPIError(
                f"Expected a list for '{entity_key}', got {type(page)}."
            )

        records.extend(page)
        logger.info(
            "Fetched %s page %s (%s rows, %s cumulative).",
            entity_key,
            page_number,
            len(page),
            len(records),
        )

        if len(page) < PAGE_SIZE:
            break

        next_id = page[-1].get("id")
        if not next_id or next_id == last_id:
            logger.warning(
                "Pagination cursor for %s did not advance; stopping at %s rows.",
                entity_key,
                len(records),
            )
            break
        last_id = str(next_id)
        time.sleep(PAGE_SLEEP_SECONDS)

    return records


def save_history_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write the consolidated extraction payload to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    logger.info("Wrote JSON payload to %s", output_path)


def fetch_30d_history(end_timestamp: int, output_path: Path | None = None) -> dict[str, Any]:
    """Fetch Aave v3 user positions and liquidations for a 30-day window.

    The Aave subgraph does not store Health Factor directly. This function
    extracts borrower ``users`` (collateral and debt balances plus 30-day
    a/v/s-token history) and ``liquidationCalls`` in
    ``[end_timestamp - 30 days, end_timestamp]`` so Health Factor can be
    computed downstream.

    Parameters
    ----------
    end_timestamp:
        Inclusive Unix timestamp (UTC seconds) that closes the 30-day window.
    output_path:
        Optional JSON destination. When provided, the consolidated payload is
        written immediately after the GraphQL extraction completes.

    Returns
    -------
    dict
        Consolidated payload with metadata, users, and liquidation calls.
    """
    start_timestamp, end_ts = _window_bounds(end_timestamp)
    start_iso = datetime.fromtimestamp(start_timestamp, tz=timezone.utc).isoformat()
    end_iso = datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()

    logger.info(
        "Extracting Aave v3 30-day history from %s to %s.",
        start_iso,
        end_iso,
    )

    try:
        with requests.Session() as session:
            users = _paginate_entity(
                USERS_QUERY,
                "users",
                start_timestamp,
                end_ts,
                session,
            )
            liquidation_calls = _paginate_entity(
                LIQUIDATION_CALLS_QUERY,
                "liquidationCalls",
                start_timestamp,
                end_ts,
                session,
            )
    except GraphAPIError:
        logger.exception(
            "Unable to extract Aave v3 history ending at timestamp %s.",
            end_timestamp,
        )
        raise

    payload: dict[str, Any] = {
        "protocol": "aave-v3",
        "network": "ethereum",
        "subgraph_id": AAVE_V3_ETHEREUM_SUBGRAPH_ID,
        "window_days": WINDOW_DAYS,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_ts,
        "start_datetime_utc": start_iso,
        "end_datetime_utc": end_iso,
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "user_count": len(users),
        "liquidation_call_count": len(liquidation_calls),
        "users": users,
        "liquidationCalls": liquidation_calls,
    }

    logger.info(
        "Extracted %s borrowers and %s liquidation calls.",
        len(users),
        len(liquidation_calls),
    )

    if output_path is not None:
        save_history_json(payload, output_path)

    return payload


def run_extraction() -> None:
    """Extract 30-day Aave v3 history for the baseline and stress scenarios."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    ensure_raw_scenario_directories()

    scenarios: list[tuple[str, datetime, Path]] = [
        ("baseline", BASELINE_END, BASELINE_DIR / "thegraph_hf_30d.json"),
        ("stress", STRESS_END, STRESS_DIR / "thegraph_hf_30d.json"),
    ]

    for name, end_dt, output_path in scenarios:
        end_timestamp = int(end_dt.timestamp())
        logger.info(
            "Starting %s extraction (end_timestamp=%s, %s).",
            name,
            end_timestamp,
            end_dt.isoformat(),
        )
        try:
            fetch_30d_history(end_timestamp, output_path=output_path)
        except (GraphAPIError, ValueError, requests.RequestException) as exc:
            logger.error("Failed to extract %s scenario: %s", name, exc)
            raise


if __name__ == "__main__":
    run_extraction()
