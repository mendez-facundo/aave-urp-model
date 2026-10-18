"""Shared helpers for ``check_raw_data.py`` and ``make_dev_sample.py``.

Holds the scenario definitions and the **approximate** position valuation used
by both scripts: USD value at the Aave oracle price of the day, a Health Factor
from the daily reserve and eMode parameters, and the A/B profile flags.

The valuation is a diagnostic, not the thesis methodology. It ignores
isolation mode, supply/borrow caps, custom eMode oracles and dust rules, and it
treats the balances in the CSV (floats in token units) as exact.

HF formula, per user and day::

    HF = sum_i(collateral_usd_i * LT_i) / debt_usd

where a reserve is collateral when the user enabled it as collateral and the
reserve itself is collateral-capable (reserve parameter flag), and ``LT_i`` is
the eMode category LT when the user's eMode id is > 0 and the reserve belongs to
that category (``emode_id`` column of the reserve parameters), else the reserve
LT.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "01_raw"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

DATE_FORMAT = "%Y-%m-%d"
WINDOW_BEFORE_DAYS = 60  # window = [t - 60, t + 30], 91 dates
WINDOW_AFTER_DAYS = 30
CALIBRATION_DAYS = 30  # Markov calibration window = [t - 30, t]; the diagnostics use its first day

DAILY_DIR = "thegraph_daily"
EVENTS_DIR = "thegraph_events"
PROTOCOL_DIR = "protocol"
PRICES_DIR = "prices"
LIQUIDATIONS_FILE = "liquidation_calls.csv"
REPAYS_FILE = "repays.csv"
METADATA_FILE = "metadata.json"
ORACLE_FILE = "aave_oracle_daily.csv"
BINANCE_FILE = "binance_daily.csv"
BLOCKS_FILE = "daily_blocks.csv"


@dataclass(frozen=True)
class Scenario:
    """One scenario: valuation date ``t`` and its window ``[t - 60, t + 30]``."""

    key: str
    folder: str
    t: str
    start: str
    end: str
    t_block: int  # first block at or after t 00:00 UTC (CLAUDE.md)


SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario("baseline", "baseline_mar2024", "2024-03-15", "2024-01-15", "2024-04-14", 19_436_645),
    "stress": Scenario("stress", "stress_aug2024", "2024-08-05", "2024-06-06", "2024-09-04", 20_458_638),
}

# Reserve symbols per group. Only symbols listed on the market on a given day
# matter; the others are simply absent from the data.
ETH_GROUP: frozenset[str] = frozenset({"WETH", "wstETH", "rETH", "cbETH", "weETH", "osETH", "ETHx"})
UNI_LINK_GROUP: frozenset[str] = frozenset({"UNI", "LINK"})
STABLE_GROUP: frozenset[str] = frozenset(
    {"USDC", "USDT", "DAI", "FRAX", "LUSD", "GHO", "PYUSD", "crvUSD", "USDe"}
)
PROFILE_SHARE = 0.85

# Health Factor states used by the diagnostic: <=1, (1,1.1), [1.1,1.25), [1.25,1.5), [1.5,2), >=2.
HF_STATE_LABELS: list[str] = ["<=1", "(1,1.1)", "[1.1,1.25)", "[1.25,1.5)", "[1.5,2)", ">=2"]

POSITION_USECOLS: list[str] = [
    "date",
    "block",
    "user_id",
    "emode_category_id",
    "reserve_symbol",
    "underlying_asset",
    "usage_as_collateral_enabled",
    "a_token_balance",
    "variable_debt",
    "stable_debt",
]
POSITION_DTYPES: dict[str, object] = {
    "date": str,
    "block": "int64",
    "user_id": str,
    "emode_category_id": "int64",
    "reserve_symbol": str,
    "underlying_asset": str,
    "usage_as_collateral_enabled": bool,
    "a_token_balance": "float64",
    "variable_debt": "float64",
    "stable_debt": "float64",
}


def window_dates(scenario: Scenario) -> list[str]:
    """Return the 91 consecutive ``YYYY-MM-DD`` dates of ``[t - 60, t + 30]``.

    Raises:
        ValueError: If ``scenario.start`` / ``scenario.end`` are not ``t - 60`` / ``t + 30``.
    """
    dates = [shift_day(scenario.t, offset) for offset in range(-WINDOW_BEFORE_DAYS, WINDOW_AFTER_DAYS + 1)]
    if (dates[0], dates[-1]) != (scenario.start, scenario.end):
        raise ValueError(f"{scenario.key}: window {scenario.start}..{scenario.end} is not [t - 60, t + 30] for t={scenario.t}.")
    return dates


def utc_midnight_ts(day: str) -> int:
    """Unix timestamp of ``day`` 00:00:00 UTC."""
    parsed = datetime.strptime(day, DATE_FORMAT).replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def shift_day(day: str, offset: int) -> str:
    """Return ``day`` shifted by ``offset`` calendar days."""
    parsed = datetime.strptime(day, DATE_FORMAT).date()
    return (parsed + timedelta(days=offset)).strftime(DATE_FORMAT)


def positions_file(scenario_dir: Path, day: str) -> Path:
    """Path of the daily positions file under a scenario folder."""
    return Path(scenario_dir) / DAILY_DIR / f"positions_{day}.csv.gz"


def read_positions(path: Path) -> pd.DataFrame:
    """Read the columns needed for valuation from one daily positions file."""
    return pd.read_csv(path, usecols=POSITION_USECOLS, dtype=POSITION_DTYPES, keep_default_na=False)


def hf_state(hf: np.ndarray) -> np.ndarray:
    """Map Health Factors to state indices ``0..5`` (see ``HF_STATE_LABELS``)."""
    hf = np.asarray(hf, dtype="float64")
    conditions = [hf <= 1.0, hf < 1.1, hf < 1.25, hf < 1.5, hf < 2.0]
    return np.select(conditions, [0, 1, 2, 3, 4], default=5)


def value_positions(
    positions: pd.DataFrame,
    prices: pd.Series,
    reserves: pd.DataFrame,
    emodes: pd.DataFrame,
) -> pd.DataFrame:
    """Value one day of positions per user (approximate, see the module docstring).

    Args:
        positions: One day from ``read_positions``.
        prices: Oracle USD price indexed by ``underlying_asset`` for the same day.
        reserves: ``load_reserve_params`` output for that day (needs ``emode_id``).
        emodes: ``load_emode_categories`` output for that day.

    Returns:
        DataFrame indexed by ``user_id`` with ``coll_usd``, ``coll_lt_usd``,
        ``eth_usd``, ``unil_usd``, ``debt_usd``, ``debt_nonstable_usd``, ``hf``
        (``inf`` when there is no debt), ``profile_a`` and ``profile_b``.
        Rows whose asset has no price or no parameters add nothing (the caller
        checks coverage separately).
    """
    res = reserves.set_index("underlying_asset")
    asset = positions["underlying_asset"]
    emode_user = positions["emode_category_id"]

    price = asset.map(prices)
    reserve_lt = asset.map(res["liquidation_threshold"])
    reserve_emode = asset.map(res["emode_id"])
    reserve_flag = asset.map(res["usage_as_collateral_enabled"]).fillna(False).astype(bool)
    category_lt = emode_user.map(emodes.set_index("emode_id")["liquidation_threshold"])

    use_category = (emode_user > 0) & (reserve_emode == emode_user) & category_lt.notna()
    lt = pd.Series(np.where(use_category, category_lt, reserve_lt), index=positions.index)

    is_collateral = positions["usage_as_collateral_enabled"] & reserve_flag
    symbol = positions["reserve_symbol"]

    collateral = (positions["a_token_balance"] * price).where(is_collateral, 0.0)
    debt = (positions["variable_debt"] + positions["stable_debt"]) * price

    frame = pd.DataFrame(
        {
            "user_id": positions["user_id"],
            "coll_usd": collateral,
            "coll_lt_usd": collateral * lt,
            "eth_usd": collateral.where(symbol.isin(ETH_GROUP), 0.0),
            "unil_usd": collateral.where(symbol.isin(UNI_LINK_GROUP), 0.0),
            "debt_usd": debt,
            "debt_nonstable_usd": debt.where(~symbol.isin(STABLE_GROUP), 0.0),
        }
    )
    users = frame.groupby("user_id", sort=False).sum()

    has_debt = users["debt_usd"] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        users["hf"] = np.where(has_debt, users["coll_lt_usd"] / users["debt_usd"], np.inf)
    stable_only = has_debt & (users["debt_nonstable_usd"] <= 0)
    has_collateral = users["coll_usd"] > 0
    eth_share = users["eth_usd"] / users["coll_usd"].where(has_collateral)
    unil_share = users["unil_usd"] / users["coll_usd"].where(has_collateral)
    users["profile_a"] = stable_only & has_collateral & (eth_share >= PROFILE_SHARE)
    users["profile_b"] = stable_only & has_collateral & (unil_share >= PROFILE_SHARE)
    return users


def day_prices(oracle: pd.DataFrame, day: str) -> pd.Series:
    """Oracle USD price by ``underlying_asset`` for one day."""
    rows = oracle.loc[oracle["date"] == day]
    return rows.set_index("underlying_asset")["price_usd"]

