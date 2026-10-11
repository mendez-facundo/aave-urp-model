"""Build the long-format Binance daily price file for each scenario from existing CSVs.

Reads the per-ticker ``binance_prices_*.csv`` files already on disk (no API calls),
writes ``prices/binance_daily.csv`` in each scenario directory, and verifies the
result against the source files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.binance_agent import (  # noqa: E402
    ASSETS,
    LONG_OUTPUT_COLUMNS,
    LONG_PRICES_RELATIVE_PATH,
    SCENARIOS,
    combine_prices_long,
    save_prices_csv,
)

VALUE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def verify_against_sources(
    long_frame: pd.DataFrame,
    prices_by_ticker: dict[str, pd.DataFrame],
) -> None:
    """Assert the long file on disk matches the source CSV data for every ticker."""
    assert list(long_frame.columns) == LONG_OUTPUT_COLUMNS, list(long_frame.columns)
    assert not long_frame.duplicated(["ticker", "date"]).any(), "duplicate (ticker, date)"

    keys = list(zip(long_frame["ticker"], long_frame["date"]))
    assert keys == sorted(keys), "long file is not sorted by ticker then date"
    assert set(long_frame["ticker"]) == set(prices_by_ticker), "ticker set mismatch"

    for ticker, source in prices_by_ticker.items():
        source = source.sort_values("date").reset_index(drop=True)
        rows = long_frame.loc[long_frame["ticker"] == ticker].reset_index(drop=True)

        assert len(rows) == len(source), f"{ticker}: {len(rows)} rows vs {len(source)} source"
        assert (rows["date"] == source["date"]).all(), f"{ticker}: date mismatch"
        pd.testing.assert_series_equal(
            rows["close"], source["close"], check_exact=True
        )
        pd.testing.assert_frame_equal(
            rows[VALUE_COLUMNS], source[VALUE_COLUMNS], check_exact=True
        )


def migrate_scenario(scenario_dir: Path) -> None:
    """Build and verify the long price file for one scenario directory."""
    prices_by_ticker = {
        ticker: pd.read_csv(scenario_dir / filename) for ticker, filename in ASSETS
    }

    long_path = scenario_dir / LONG_PRICES_RELATIVE_PATH
    save_prices_csv(combine_prices_long(prices_by_ticker), long_path)

    long_frame = pd.read_csv(long_path)
    verify_against_sources(long_frame, prices_by_ticker)

    print(f"{scenario_dir.name}: {long_path}")
    for ticker, group in long_frame.groupby("ticker", sort=True):
        print(
            f"  {ticker:<9} rows={len(group):>4}  "
            f"{group['date'].min()} .. {group['date'].max()}  close=identical"
        )
    print(f"  total rows={len(long_frame)}  verification=OK")


def main() -> None:
    for _end_date, scenario_dir in SCENARIOS:
        migrate_scenario(scenario_dir)


if __name__ == "__main__":
    main()
