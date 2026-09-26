"""Binance historical daily klines extractor for the EVT actuarial pipeline."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
KLINE_INTERVAL = "1d"
KLINE_LIMIT = 1000
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "01_raw"
BASELINE_DIR = RAW_DATA_DIR / "baseline_mar2024"
STRESS_DIR = RAW_DATA_DIR / "stress_aug2024"

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]
OUTPUT_COLUMNS = ["timestamp", "date", "open", "high", "low", "close", "volume"]


class BinanceAPIError(RuntimeError):
    """Raised when the Binance public API cannot be queried successfully."""


def ensure_raw_scenario_directories() -> None:
    """Create raw data scenario directories if they do not already exist."""
    for directory in (BASELINE_DIR, STRESS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory exists: %s", directory)


def _parse_end_date(end_date: str) -> datetime:
    """Parse an ISO date string (YYYY-MM-DD) as UTC midnight."""
    try:
        parsed = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid end_date '{end_date}'. Expected ISO 8601 format YYYY-MM-DD."
        ) from exc
    return parsed.replace(tzinfo=timezone.utc)


def _request_klines(
    symbol: str,
    start_ms: int,
    end_ms: int,
    session: requests.Session,
) -> list[list[Any]]:
    """Fetch a single page of daily klines from the Binance public API."""
    params = {
        "symbol": symbol,
        "interval": KLINE_INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": KLINE_LIMIT,
    }

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                BINANCE_KLINES_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 429:
                wait_seconds = RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "Binance rate limit reached for %s (attempt %s/%s). Retrying in %.1fs.",
                    symbol,
                    attempt,
                    MAX_RETRIES,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise BinanceAPIError(
                    f"Unexpected Binance response payload for {symbol}: {payload}"
                )
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            wait_seconds = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "Binance request failed for %s (attempt %s/%s): %s. Retrying in %.1fs.",
                symbol,
                attempt,
                MAX_RETRIES,
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise BinanceAPIError(
        f"Failed to download klines for {symbol} after {MAX_RETRIES} attempts."
    ) from last_error


def _fetch_all_klines(
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[list[Any]]:
    """Download all daily klines between start_ms and end_ms, paginating if needed."""
    klines: list[list[Any]] = []
    cursor_ms = start_ms

    with requests.Session() as session:
        while cursor_ms <= end_ms:
            page = _request_klines(symbol, cursor_ms, end_ms, session)
            if not page:
                break

            klines.extend(page)
            last_open_time = int(page[-1][0])
            next_cursor = last_open_time + 1
            if next_cursor <= cursor_ms:
                break
            cursor_ms = next_cursor

            if len(page) < KLINE_LIMIT:
                break

    return klines


def _klines_to_dataframe(
    klines: list[list[Any]],
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Convert raw Binance klines into the pipeline output schema."""
    if not klines:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    frame = pd.DataFrame(klines, columns=KLINE_COLUMNS)
    frame["timestamp"] = pd.to_numeric(frame["open_time"], errors="raise").astype("int64")
    frame["date"] = (
        pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        .dt.strftime("%Y-%m-%d")
    )

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    start_iso = start_date.strftime("%Y-%m-%d")
    end_iso = end_date.strftime("%Y-%m-%d")
    frame = frame.loc[(frame["date"] >= start_iso) & (frame["date"] <= end_iso)]
    frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    return frame[OUTPUT_COLUMNS].reset_index(drop=True)


def download_historical_prices(
    asset: str,
    end_date: str,
    days_back: int = 730,
) -> pd.DataFrame:
    """Download daily OHLCV candles from Binance for an asset.

    Parameters
    ----------
    asset:
        Binance trading pair symbol, e.g. ``ETHUSDT``.
    end_date:
        Inclusive last candle date in ISO 8601 format ``YYYY-MM-DD``.
    days_back:
        Number of calendar days before ``end_date`` to include. The resulting
        window is ``[end_date - days_back, end_date]``.

    Returns
    -------
    pandas.DataFrame
        Columns: timestamp (UTC ms), date (YYYY-MM-DD), open, high, low,
        close, volume.
    """
    if days_back <= 0:
        raise ValueError(f"days_back must be a positive integer, got {days_back}.")
    if not asset or not asset.strip():
        raise ValueError("asset must be a non-empty Binance symbol.")

    symbol = asset.strip().upper()
    end_dt = _parse_end_date(end_date)
    start_dt = end_dt - timedelta(days=days_back)
    start_ms = int(start_dt.timestamp() * 1000)
    # Inclusive end of the last UTC day so the end_date daily candle is returned.
    end_ms = int((end_dt + timedelta(days=1) - timedelta(milliseconds=1)).timestamp() * 1000)

    logger.info(
        "Downloading %s daily klines from %s to %s.",
        symbol,
        start_dt.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
    )

    try:
        klines = _fetch_all_klines(symbol, start_ms, end_ms)
    except BinanceAPIError:
        logger.exception("Unable to download historical prices for %s.", symbol)
        raise

    prices = _klines_to_dataframe(klines, start_dt, end_dt)
    if prices.empty:
        logger.warning("No klines returned for %s in the requested window.", symbol)
    else:
        logger.info("Downloaded %s daily candles for %s.", len(prices), symbol)

    return prices


def save_prices_csv(prices: pd.DataFrame, output_path: Path) -> None:
    """Export a prices DataFrame to CSV, creating parent directories if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index=False)
    logger.info("Wrote %s rows to %s", len(prices), output_path)


def run_extraction() -> None:
    """Download historical daily prices for the baseline and stress scenarios."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    ensure_raw_scenario_directories()

    scenarios: list[tuple[str, Path]] = [
        ("2024-03-15", BASELINE_DIR),
        ("2024-08-05", STRESS_DIR),
    ]
    assets: list[tuple[str, str]] = [
        ("ETHUSDT", "binance_prices_eth.csv"),
        ("UNIUSDT", "binance_prices_uni.csv"),
        ("BTCUSDT", "binance_prices_btc.csv"),
        ("LINKUSDT", "binance_prices_link.csv"),
    ]

    for end_date, output_dir in scenarios:
        for asset, filename in assets:
            try:
                prices = download_historical_prices(asset=asset, end_date=end_date)
                save_prices_csv(prices, output_dir / filename)
            except (BinanceAPIError, ValueError, requests.RequestException) as exc:
                logger.error(
                    "Failed to extract %s for end_date=%s: %s",
                    asset,
                    end_date,
                    exc,
                )
                raise


if __name__ == "__main__":
    run_extraction()
