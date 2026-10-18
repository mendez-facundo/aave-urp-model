# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Markov Calibration of Health Factor States on Aave v3 Daily Snapshots
#
# This notebook reads the **daily time-travel snapshots** of Aave v3 (The Graph, one positions file per UTC day), values every position in USD with the **Aave oracle** price of the day, applies the actuarial purity filters, and reconstructs the empirical **Health Factor (HF)** panel used to calibrate a discrete-time Markov chain over the states S1-S7.
#
# Both valuation windows are processed in parallel:
#
# - **Baseline scenario:** t = 15 March 2024 (`data/01_raw/baseline_mar2024/`)
# - **Stress scenario:** t = 5 August 2024 (`data/01_raw/stress_aug2024/`)
#
# Each scenario folder holds 61 daily snapshots for $[t-30, t+30]$. **Calibration uses $[t-30, t]$ only**: the transition counts $\mathbf{N}$ and the matrices $\mathbf{P}$ come from the 30 daily transitions inside that window. The days after $t$ are reserved for the out-of-time validation: they never enter $\mathbf{N}$ or $\mathbf{P}$, but the export of Section 18 records what that backtest needs (`oot_outcome`, `oot_event_date`).
#
# Positions are valued as `collateral_usd = a_token_balance * oracle price` and `debt_usd = (variable_debt + stable_debt) * oracle price`. Liquidation thresholds (LT), liquidation bonuses (LB) and eMode categories are read **day by day** from `src.utils.protocol_params`; the fixed constants of `src.utils.config` are no longer used. Binance prices are not used in this notebook.
#
# **State vocabulary.** `S1` is a *repayment observed on-chain*; `S7` is a *liquidation event* (technical default): Health Factor $\le 1$ in the daily snapshot, or a `liquidationCall` on the wallet. `S1` and `S7` are **absorbing by definition**: the trajectory ends at the first hit. **Censoring** is a different thing and only applies to wallets whose outcome is not observed (`censored_window_end`, `censored_filter_exit`).
#
# **Development.** The raw folder is configurable through `RAW_DATA_DIR` (default `../data/01_raw`; set the environment variable `MARKOV_RAW_DATA_DIR=../data/sample` to develop on the small sample). A run on the sample writes its exports under `data/sample/02_interim/` so that it can never be mistaken for a full run.

# %% [markdown]
# ## 1. Imports

# %%
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy.special import gammaincc, kolmogorov

sys.path.append(os.path.abspath(".."))
from src.utils.protocol_params import (
    get_param,
    load_emode_categories,
    load_reserve_params,
)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (10, 5),
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.frameon": True,
        "axes.spines.top": False,
    }
)

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 80)


def chi2_sf(statistic: float, dof: int) -> float:
    """Chi-square survival function (``scipy.special`` only: ``scipy.stats`` is blocked on some Windows setups)."""
    return float(gammaincc(dof / 2.0, statistic / 2.0))


def ks_two_sample(first: pd.Series, second: pd.Series) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and its asymptotic p-value."""
    a, b = np.sort(first.to_numpy(dtype=float)), np.sort(second.to_numpy(dtype=float))
    grid = np.concatenate([a, b])
    gap = np.abs(np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b))
    statistic = float(gap.max())
    effective_n = np.sqrt(len(a) * len(b) / (len(a) + len(b)))
    return statistic, float(kolmogorov(effective_n * statistic))


# %% [markdown]
# ## 2. Configuration and data loading
#
# `RAW_DATA_DIR` points to the raw folder; each scenario sub-folder is read through the layout written by the extraction agents:
#
# - `thegraph_daily/positions_<date>.csv.gz`: accrued balances per (user, reserve) at the first block at or after 00:00 UTC (`a_token_balance`, `variable_debt`, `stable_debt` in token units).
# - `thegraph_events/{liquidation_calls,repays}.csv`: the on-chain events of the window.
# - `protocol/{reserve_params_daily,emode_categories_daily,daily_blocks}.csv`: LT, LB, LTV, eMode and snapshot blocks.
# - `prices/aave_oracle_daily.csv`: USD price used by the protocol, one row per (date, reserve).
#
# Wallets and days are also indexed by integers (`wid`, `day` with `t-30 = 0` and `t = 30`) and combined in a single key, `key = wid * DAY_STRIDE + day`. This turns "the same wallet on the next day" into `key + 1` and keeps the transition logic of Section 14 vectorised.

# %%
RAW_DATA_DIR = Path(os.environ.get("MARKOV_RAW_DATA_DIR", "../data/01_raw"))
SAMPLE_DATA_DIR = Path("../data/sample")
IS_SAMPLE_RUN = RAW_DATA_DIR.resolve() == SAMPLE_DATA_DIR.resolve()
INTERIM_DATA_DIR = SAMPLE_DATA_DIR / "02_interim" if IS_SAMPLE_RUN else Path("../data/02_interim")

SCENARIO_KEYS = [
    ("baseline", "baseline_mar2024"),
    ("stress", "stress_aug2024"),
]
SCENARIO_LABEL = dict(SCENARIO_KEYS)
VALUATION_DATE = {
    "baseline": pd.Timestamp("2024-03-15"),
    "stress": pd.Timestamp("2024-08-05"),
}

CALIBRATION_DAYS = 30
OOT_DAYS = 30
T_IDX = CALIBRATION_DAYS  # day index of the valuation date t
LAST_IDX = CALIBRATION_DAYS + OOT_DAYS  # day index of t + 30
N_DAYS = LAST_IDX + 1
DAY_STRIDE = 64  # key = wid * DAY_STRIDE + day; DAY_STRIDE > N_DAYS so keys never cross wallets

DAY1_DATE = {key: date - pd.Timedelta(CALIBRATION_DAYS, unit="D") for key, date in VALUATION_DATE.items()}
OOT_END_DATE = {key: date + pd.Timedelta(OOT_DAYS, unit="D") for key, date in VALUATION_DATE.items()}

POSITION_DTYPES = {
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


def load_positions(scenario_dir: Path) -> pd.DataFrame:
    """Read and stack the daily positions files of a scenario (one row per user, reserve and day)."""
    files = sorted((scenario_dir / "thegraph_daily").glob("positions_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No positions_*.csv.gz files under {scenario_dir.resolve()}")
    frames = [
        pd.read_csv(path, usecols=list(POSITION_DTYPES), dtype=POSITION_DTYPES, keep_default_na=False)
        for path in files
    ]
    positions = pd.concat(frames, ignore_index=True)
    positions["date"] = pd.to_datetime(positions["date"])
    return positions


print(f"RAW_DATA_DIR = {RAW_DATA_DIR.resolve()}  (sample run: {IS_SAMPLE_RUN})")
print(f"INTERIM_DATA_DIR = {INTERIM_DATA_DIR.resolve()}")

positions_by_scenario: dict[str, pd.DataFrame] = {}
for scenario_key, scenario_label in SCENARIO_KEYS:
    positions_by_scenario[scenario_key] = load_positions(RAW_DATA_DIR / scenario_label)
    positions = positions_by_scenario[scenario_key]
    print(
        f"[{scenario_key}] {scenario_label}: {len(positions):,} position rows, "
        f"{positions['date'].nunique()} days, {positions['user_id'].nunique():,} distinct borrowers"
    )

# %% [markdown]
# ## 3. Inspection of the new files
#
# A compact look at what the extraction wrote: file groups and sizes, columns, metadata and one day of positions. The last check recomputes the token balances from the scaled balances and the Aave indices (`balance = scaled * index / 1e27 / 10**decimals`) on day $t$, to confirm that the float columns used below are the accrued balances and not the stale `current*` fields of the subgraph (the files store them rounded to 6 decimals).

# %%
FILE_GROUPS = {
    "thegraph_daily": "positions_*.csv.gz",
    "thegraph_events": "*.csv",
    "protocol": "*.csv",
    "prices": "*.csv",
}

file_records = []
columns_by_file: dict[str, list[str]] = {}
for scenario_key, scenario_label in SCENARIO_KEYS:
    scenario_dir = RAW_DATA_DIR / scenario_label
    for sub_folder, pattern in FILE_GROUPS.items():
        paths = sorted((scenario_dir / sub_folder).glob(pattern))
        groups = {f"{sub_folder}/positions_<date>.csv.gz": paths} if sub_folder == "thegraph_daily" else {
            f"{sub_folder}/{path.name}": [path] for path in paths
        }
        for name, group in groups.items():
            file_records.append(
                {
                    "scenario": scenario_label,
                    "file": name,
                    "n_files": len(group),
                    "size_mb": round(sum(path.stat().st_size for path in group) / 1e6, 2),
                }
            )
            if scenario_key == "baseline" and group:
                columns_by_file[name] = list(pd.read_csv(group[0], nrows=0).columns)

display(pd.DataFrame(file_records).set_index(["scenario", "file"]))

print("Columns (baseline files):")
for name, columns in columns_by_file.items():
    print(f"  {name}: {columns}")

import json

for scenario_key, scenario_label in SCENARIO_KEYS:
    meta_path = RAW_DATA_DIR / scenario_label / "thegraph_daily" / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        shown = {k: metadata[k] for k in ("sample", "scenario", "first_date", "last_date", "n_days") if k in metadata}
        print(f"\n[{scenario_key}] thegraph_daily/metadata.json: {shown}")

example_day = VALUATION_DATE["baseline"]
example = positions_by_scenario["baseline"].loc[positions_by_scenario["baseline"]["date"] == example_day]
print(f"\nBaseline positions on {example_day.date()} (first rows):")
display(example.head())

# Balance recomputation on day t, both scenarios.
for scenario_key, scenario_label in SCENARIO_KEYS:
    path = RAW_DATA_DIR / scenario_label / "thegraph_daily" / f"positions_{VALUATION_DATE[scenario_key].date()}.csv.gz"
    raw_day = pd.read_csv(path, dtype=str, keep_default_na=False)
    scale = 10.0 ** raw_day["decimals"].astype(float)
    liquidity_index = raw_day["liquidity_index"].astype(float)
    borrow_index = raw_day["variable_borrow_index"].astype(float)
    a_check = raw_day["scaled_a_token_balance"].astype(float) * liquidity_index / 1e27 / scale
    v_check = raw_day["scaled_variable_debt"].astype(float) * borrow_index / 1e27 / scale
    a_file = raw_day["a_token_balance"].astype(float)
    v_file = raw_day["variable_debt"].astype(float)
    # The file stores balances rounded to 6 decimals, so the tolerance is half a unit of that rounding
    # plus a relative term for the float arithmetic of the recomputation.
    err_a = ((a_check - a_file).abs() - 1e-9 * a_file.abs()).clip(lower=0).max()
    err_v = ((v_check - v_file).abs() - 1e-9 * v_file.abs()).clip(lower=0).max()
    print(f"[{scenario_key}] recomputed balances: max excess error beyond the 6-decimal rounding: aToken={err_a:.2e}, variable debt={err_v:.2e} tokens")
    assert err_a < 5.1e-7 and err_v < 5.1e-7, "File balances differ from scaled balance x index."

# %% [markdown]
# ## 4. Daily positions panel
#
# The extractor already writes the long format (one row per user, reserve and day), so no nested history has to be unnested: the flattening of the old notebook disappears. This section only validates the panel:
#
# - the 61 UTC days $[t-30, t+30]$ are present and consecutive,
# - every day uses the snapshot block of `protocol/daily_blocks.csv`,
# - there is no duplicated (day, user, reserve) row.
#
# A wallet is **observed** on a day when it appears in that day's snapshot. The universe is not fixed: wallets enter when they borrow and leave when they repay or are liquidated.

# %%
daily_blocks_by_scenario: dict[str, pd.DataFrame] = {}
for scenario_key, scenario_label in SCENARIO_KEYS:
    positions = positions_by_scenario[scenario_key]
    expected_days = pd.date_range(DAY1_DATE[scenario_key], OOT_END_DATE[scenario_key], freq="D")
    observed_days = pd.DatetimeIndex(sorted(positions["date"].unique()))
    assert observed_days.equals(expected_days), f"{scenario_label}: positions days are not the 61 consecutive window days."

    blocks = pd.read_csv(RAW_DATA_DIR / scenario_label / "protocol" / "daily_blocks.csv", parse_dates=["date"])
    daily_blocks_by_scenario[scenario_key] = blocks
    block_by_day = positions.groupby("date")["block"].agg(["min", "max"])
    assert (block_by_day["min"] == block_by_day["max"]).all(), f"{scenario_label}: more than one block on a day."
    assert block_by_day["min"].equals(blocks.set_index("date")["block"].reindex(block_by_day.index)), (
        f"{scenario_label}: positions blocks differ from daily_blocks.csv."
    )
    assert not positions.duplicated(["date", "user_id", "reserve_symbol"]).any(), f"{scenario_label}: duplicated rows."
    print(
        f"[{scenario_key}] OK: {len(observed_days)} days {observed_days[0].date()} -> {observed_days[-1].date()}, "
        f"t block = {int(blocks.set_index('date').loc[VALUATION_DATE[scenario_key], 'block']):,}"
    )

# %% [markdown]
# ## 5. Liquidation and repayment events
#
# `liquidationCalls` define one of the two triggers of `S7`; `repays` define `S1`. Both are keyed by the **borrower** (`user_id`) and the UTC date of the event. A snapshot is taken at the first block at or after 00:00 UTC of its date, so an event dated $d$ happens **after** snapshot $d$ and **before** snapshot $d+1$. Section 13 uses this timing to attach each event to a transition.
#
# The cell reports event counts and how many of them belong to a user present in the snapshot of the event day or the day before. The USD price fields of the events are unreliable (0 in 19-27% of liquidations) and are never used.

# %%
def load_events(scenario_dir: Path, scenario_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the liquidationCall and repay events of a scenario."""
    events_dir = scenario_dir / "thegraph_events"
    read_options = {"dtype": str, "keep_default_na": False}
    liquidations = pd.read_csv(
        events_dir / "liquidation_calls.csv",
        usecols=["timestamp", "date", "user_id", "collateral_symbol", "principal_symbol"],
        **read_options,
    )
    repays = pd.read_csv(
        events_dir / "repays.csv",
        usecols=["timestamp", "date", "user_id", "reserve_symbol", "use_a_tokens"],
        **read_options,
    )
    for frame in (liquidations, repays):
        frame["date"] = pd.to_datetime(frame["date"])
        frame.insert(0, "scenario", scenario_key)
    return liquidations, repays


liquidation_parts, repay_parts = [], []
for scenario_key, scenario_label in SCENARIO_KEYS:
    liquidations_s, repays_s = load_events(RAW_DATA_DIR / scenario_label, scenario_key)
    liquidation_parts.append(liquidations_s)
    repay_parts.append(repays_s)
liquidation_calls = pd.concat(liquidation_parts, ignore_index=True)
repays = pd.concat(repay_parts, ignore_index=True)


def share_with_snapshot(events: pd.DataFrame, positions: pd.DataFrame) -> float:
    """Share of events whose user appears in the snapshot of the event day or the day before."""
    presence = positions[["user_id", "date"]].drop_duplicates().assign(present=True)
    same_day = events[["user_id", "date"]].merge(presence, on=["user_id", "date"], how="left")["present"].notna()
    previous = events[["user_id", "date"]].assign(date=events["date"] - pd.Timedelta(1, unit="D"))
    day_before = previous.merge(presence, on=["user_id", "date"], how="left")["present"].notna()
    return float((same_day.to_numpy() | day_before.to_numpy()).mean()) if len(events) else float("nan")


event_records = []
for scenario_key, scenario_label in SCENARIO_KEYS:
    liq_s = liquidation_calls.loc[liquidation_calls["scenario"] == scenario_key]
    rep_s = repays.loc[repays["scenario"] == scenario_key]
    positions = positions_by_scenario[scenario_key]
    event_records.append(
        {
            "scenario": scenario_label,
            "liquidation_calls": len(liq_s),
            "liquidated_users": liq_s["user_id"].nunique(),
            "liq_first_date": liq_s["date"].min().date(),
            "liq_last_date": liq_s["date"].max().date(),
            "liq_user_in_snapshot": share_with_snapshot(liq_s, positions),
            "repays": len(rep_s),
            "repaying_users": rep_s["user_id"].nunique(),
            "repay_user_in_snapshot": share_with_snapshot(rep_s, positions),
        }
    )
display(pd.DataFrame(event_records).set_index("scenario").T)

# %% [markdown]
# ## 6. Protocol parameters
#
# LT, LB, LTV and the reserve eMode category come from `load_reserve_params(scenario, date)`, and the eMode category parameters from `load_emode_categories(scenario, date)`: the single source of protocol parameters. They change over time (for example UNI and LINK), which is why they are merged **by day** in Section 9. The table shows the LT of the reserves that matter for the profiles on the first day, on $t$ and on the last day of the window.

# %%
reserve_params: dict[str, pd.DataFrame] = {}
emode_categories: dict[str, pd.DataFrame] = {}
PARAM_SYMBOLS = ["WETH", "wstETH", "rETH", "cbETH", "WBTC", "UNI", "LINK", "USDC", "USDT", "DAI"]

for scenario_key, scenario_label in SCENARIO_KEYS:
    reserves = load_reserve_params(scenario_label, raw_dir=RAW_DATA_DIR)
    emodes = load_emode_categories(scenario_label, raw_dir=RAW_DATA_DIR)
    reserves["date"] = pd.to_datetime(reserves["date"])
    emodes["date"] = pd.to_datetime(emodes["date"])
    window = pd.date_range(DAY1_DATE[scenario_key], OOT_END_DATE[scenario_key], freq="D")
    assert set(window) <= set(reserves["date"]), f"{scenario_label}: reserve parameters miss window days."
    assert set(window) <= set(emodes["date"]), f"{scenario_label}: eMode parameters miss window days."
    reserve_params[scenario_key] = reserves
    emode_categories[scenario_key] = emodes

    rows = {}
    for label, date in (("day1", DAY1_DATE[scenario_key]), ("t", VALUATION_DATE[scenario_key]), ("t+30", OOT_END_DATE[scenario_key])):
        params_day = load_reserve_params(scenario_label, date=str(date.date()), raw_dir=RAW_DATA_DIR)
        rows[f"LT {label}"] = {
            symbol: get_param(params_day, symbol, "liquidation_threshold")
            for symbol in PARAM_SYMBOLS
            if symbol in set(params_day["symbol"])
        }
        if label == "t":
            rows["LB t"] = {
                symbol: get_param(params_day, symbol, "liquidation_bonus")
                for symbol in PARAM_SYMBOLS
                if symbol in set(params_day["symbol"])
            }
    print(f"\n[{scenario_key}] reserve parameters: {reserves['symbol'].nunique()} reserves, {reserves['date'].nunique()} days")
    display(pd.DataFrame(rows))
    emodes_t = emodes.loc[emodes["date"] == VALUATION_DATE[scenario_key]]
    print("eMode categories on t:")
    display(emodes_t)

# %% [markdown]
# ## 7. Daily universe
#
# Number of borrowers per day (all snapshot users, before any profile filter) and the number of reserves with debt per borrower on $t$. The universe changes from day to day, which is what the old extract could not show: the previous panel was built from a single set of users observed in 2026.

# %%
universe_by_day: dict[str, pd.Series] = {}
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (scenario_key, scenario_label) in zip(axes, SCENARIO_KEYS):
    positions = positions_by_scenario[scenario_key]
    universe = positions.groupby("date")["user_id"].nunique().rename("borrowers")
    universe_by_day[scenario_key] = universe
    assert universe.nunique() > 1, f"{scenario_label}: the universe does not change between days."
    print(
        f"[{scenario_key}] borrowers per day: min={universe.min():,} median={int(universe.median()):,} max={universe.max():,}"
    )
    display(universe.iloc[[0, 1, 2, T_IDX - 1, T_IDX, T_IDX + 1, LAST_IDX]].to_frame().T)

    debt_reserves = (
        positions.loc[(positions["date"] == VALUATION_DATE[scenario_key]) & (positions["variable_debt"] > 0)]
        .groupby("user_id")
        .size()
    )
    debt_reserves.value_counts().sort_index().plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title(f"{scenario_label}: reserves with debt per borrower on t")
    ax.set_xlabel("Reserves with variable debt")
    ax.set_ylabel("Number of borrowers")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Aave oracle prices
#
# `prices/aave_oracle_daily.csv` holds the price that the protocol itself uses to liquidate (`AaveOracle.getAssetsPrices` at the daily block), for every reserve and day. It values the ETH liquid staking tokens at their own price (wstETH trades above ETH) and the stablecoins at their oracle price instead of a fixed $1.00. The cell checks that the key (date, asset) is unique, that the prices are positive, and that every asset held in the positions has a price on the day it is held.

# %%
oracle_by_scenario: dict[str, pd.DataFrame] = {}
for scenario_key, scenario_label in SCENARIO_KEYS:
    oracle = pd.read_csv(
        RAW_DATA_DIR / scenario_label / "prices" / "aave_oracle_daily.csv",
        usecols=["date", "symbol", "underlying_asset", "price_usd"],
        dtype={"date": str, "symbol": str, "underlying_asset": str},
    )
    oracle["date"] = pd.to_datetime(oracle["date"])
    assert not oracle.duplicated(["date", "underlying_asset"]).any(), f"{scenario_label}: duplicated oracle rows."
    assert (oracle["price_usd"] > 0).all(), f"{scenario_label}: non-positive oracle prices."
    oracle_by_scenario[scenario_key] = oracle

    held = positions_by_scenario[scenario_key][["date", "underlying_asset"]].drop_duplicates()
    merged = held.merge(oracle[["date", "underlying_asset", "price_usd"]], on=["date", "underlying_asset"], how="left")
    n_missing = int(merged["price_usd"].isna().sum())
    print(f"[{scenario_key}] oracle rows: {len(oracle):,}, assets held without a price: {n_missing}")
    assert n_missing == 0, f"{scenario_label}: some held assets have no oracle price."

    stable_like = oracle.loc[oracle["symbol"].isin(["USDC", "USDT", "DAI"]), "price_usd"]
    wsteth = oracle.loc[oracle["symbol"] == "wstETH"].set_index("date")["price_usd"]
    weth = oracle.loc[oracle["symbol"] == "WETH"].set_index("date")["price_usd"]
    ratio = (wsteth / weth).dropna()
    print(
        f"  stablecoin oracle range: [{stable_like.min():.4f}, {stable_like.max():.4f}]; "
        f"wstETH/WETH range: [{ratio.min():.4f}, {ratio.max():.4f}]"
    )

# %% [markdown]
# ## 9. Daily USD valuation
#
# Each position row of a snapshot is valued with the oracle price **of its own day**:
#
# - `collateral_usd = a_token_balance * price`, counted only when the user enabled the reserve as collateral (`usage_as_collateral_enabled`) **and** the reserve is collateral-capable in the protocol parameters of that day.
# - `debt_usd = (variable_debt + stable_debt) * price`.
# - `lt` / `lb`: the reserve LT / LB of the day, or the **eMode category** LT / LB when the user's eMode id is $> 0$ and the reserve belongs to that category.
#
# Sub-microdollar residuals (`DUST_USD`) are treated as zero so that leftover wei does not contaminate the purity tests; this is a numerical floor and is unrelated to the repayment threshold `DUST_DEBT_USD` of Section 13. A position with a balance but no oracle price would be unvalued; the cell asserts that there is none, and that each valued price equals the price of `aave_oracle_daily.csv` for the same day.

# %%
ETH_CORE = {"WETH", "wstETH", "rETH", "cbETH", "weETH", "osETH", "ETHx"}  # WETH and ETH liquid staking tokens
WBTC_SET = {"WBTC"}
PROFILE_B_CORE = {"UNI", "LINK"}
STABLE_SYMBOLS = {"USDC", "USDT", "DAI", "FRAX", "LUSD", "GHO", "PYUSD", "crvUSD", "USDe"}  # USD stablecoins (debt rule)
STABLE_WRAPPERS = {"sDAI", "sUSDe"}  # yield-bearing wrappers of USD stablecoins: stable collateral, valued at the oracle price
STABLE_COLLATERAL = STABLE_SYMBOLS | STABLE_WRAPPERS
SYMBOL_GROUP = {
    **{symbol: "eth_core" for symbol in ETH_CORE},
    "WBTC": "wbtc",
    "UNI": "uni",
    "LINK": "link",
    **{symbol: "stable" for symbol in STABLE_COLLATERAL},
}
GROUPS = ["eth_core", "wbtc", "uni", "link", "stable", "other"]
DUST_USD = 1e-4
AUDIT_SAMPLE_ROWS = 5000
AUDIT_SEED = 20240315


def value_scenario_positions(scenario_key: str) -> pd.DataFrame:
    """Value one scenario at row level and aggregate it to one row per (user, day)."""
    positions = positions_by_scenario[scenario_key]
    oracle = oracle_by_scenario[scenario_key]
    reserves = reserve_params[scenario_key]
    emodes = emode_categories[scenario_key]

    reserve_columns = reserves[
        ["date", "underlying_asset", "liquidation_threshold", "liquidation_bonus", "usage_as_collateral_enabled", "emode_id"]
    ].rename(
        columns={
            "liquidation_threshold": "reserve_lt",
            "liquidation_bonus": "reserve_lb",
            "usage_as_collateral_enabled": "reserve_collateral_enabled",
            "emode_id": "reserve_emode_id",
        }
    )
    emode_columns = emodes[["date", "emode_id", "liquidation_threshold", "liquidation_bonus"]].rename(
        columns={"emode_id": "emode_category_id", "liquidation_threshold": "emode_lt", "liquidation_bonus": "emode_lb"}
    )

    work = positions.merge(
        oracle[["date", "underlying_asset", "price_usd"]], on=["date", "underlying_asset"], how="left", validate="many_to_one"
    )
    work = work.merge(reserve_columns, on=["date", "underlying_asset"], how="left", validate="many_to_one")
    work = work.merge(emode_columns, on=["date", "emode_category_id"], how="left", validate="many_to_one")
    assert len(work) == len(positions), "A merge changed the number of position rows."

    has_balance = (work["a_token_balance"] > 0) | (work["variable_debt"] > 0) | (work["stable_debt"] > 0)
    n_unpriced = int((work["price_usd"].isna() & has_balance).sum())
    n_no_params = int(work["reserve_lt"].isna().sum())
    assert n_no_params == 0, f"{scenario_key}: {n_no_params} rows without reserve parameters."

    audit = work.sample(n=min(AUDIT_SAMPLE_ROWS, len(work)), random_state=AUDIT_SEED)
    oracle_lookup = oracle.set_index(["date", "underlying_asset"])["price_usd"]
    expected_price = oracle_lookup.reindex(pd.MultiIndex.from_frame(audit[["date", "underlying_asset"]])).to_numpy()
    price_gap = float(np.nanmax(np.abs(audit["price_usd"].to_numpy() - expected_price)))

    use_emode = (
        (work["emode_category_id"] > 0)
        & (work["reserve_emode_id"] == work["emode_category_id"])
        & work["emode_lt"].notna()
    )
    work["lt"] = np.where(use_emode, work["emode_lt"], work["reserve_lt"])
    work["lb"] = np.where(use_emode, work["emode_lb"], work["reserve_lb"])

    is_collateral = work["usage_as_collateral_enabled"] & work["reserve_collateral_enabled"].fillna(False).astype(bool)
    price = work["price_usd"].fillna(0.0)
    work["collateral_usd"] = (work["a_token_balance"] * price).where(is_collateral, 0.0)
    work["debt_usd"] = (work["variable_debt"] + work["stable_debt"]) * price
    work.loc[work["collateral_usd"] < DUST_USD, "collateral_usd"] = 0.0
    work.loc[work["debt_usd"] < DUST_USD, "debt_usd"] = 0.0

    work["unpriced"] = work["price_usd"].isna() & has_balance
    work["symbol_group"] = work["reserve_symbol"].map(SYMBOL_GROUP).fillna("other")
    for group in GROUPS:
        work[f"{group}_usd"] = work["collateral_usd"].where(work["symbol_group"] == group, 0.0)
    work["coll_lt_usd"] = work["collateral_usd"] * work["lt"]
    work["coll_lb_usd"] = work["collateral_usd"] * work["lb"]
    work["stable_debt_usd"] = work["debt_usd"].where(work["reserve_symbol"].isin(STABLE_SYMBOLS), 0.0)

    users = work.groupby(["user_id", "date"], sort=False).agg(
        emode_category_id=("emode_category_id", "first"),
        total_collateral_usd=("collateral_usd", "sum"),
        total_debt_usd=("debt_usd", "sum"),
        stable_debt_usd=("stable_debt_usd", "sum"),
        thr_weighted_collateral_usd=("coll_lt_usd", "sum"),
        lb_weighted_collateral_usd=("coll_lb_usd", "sum"),
        eth_core_usd=("eth_core_usd", "sum"),
        wbtc_usd=("wbtc_usd", "sum"),
        uni_usd=("uni_usd", "sum"),
        link_usd=("link_usd", "sum"),
        stable_usd=("stable_usd", "sum"),
        other_usd=("other_usd", "sum"),
        has_unpriced=("unpriced", "any"),
    ).reset_index()
    users.insert(0, "scenario", scenario_key)

    n_emode_rows = int(use_emode.sum())
    print(f"[{scenario_key}] valued rows: {len(work):,} -> user-days: {len(users):,}")
    print(
        f"  unpriced rows with balance: {n_unpriced}; eMode LT/LB applied on {n_emode_rows:,} rows; "
        f"max |price - oracle| on {len(audit):,} sampled rows: {price_gap:.1e}"
    )
    assert n_unpriced == 0, f"{scenario_key}: unpriced positions."
    assert price_gap == 0.0, f"{scenario_key}: valued price differs from the oracle price."
    return users


user_day_parts = [value_scenario_positions(scenario_key) for scenario_key, _ in SCENARIO_KEYS]
user_day = pd.concat(user_day_parts, ignore_index=True)
del user_day_parts
positions_by_scenario.clear()  # the row-level positions are not needed after the (user, day) aggregation

# %% [markdown]
# ## 10. Aggregation and actuarial dominance filters
#
# Positions are summed to the `(user_id, date)` level, keeping the collateral USD by asset group: ETH core (`eth_core_usd`: WETH and the ETH liquid staking tokens present in the data), `wbtc_usd`, `uni_usd`, `link_usd`, `stable_usd` and `other_usd` (any other reserve).
#
# **Debt rule:** 100% of `total_debt_usd` must be in USD stablecoins (`STABLE_SYMBOLS`: USDC, USDT, DAI, FRAX, LUSD, GHO, PYUSD, crvUSD, USDe).
#
# **Profile A (ETH dominance):** at least 85% of `total_collateral_usd` in `eth_core`. The residual (at most 15%) may only be stablecoins or WBTC.
#
# **Profile B (UNI+LINK high-risk dominance):** at least 85% of `total_collateral_usd` in `UNI` + `LINK`. The residual (at most 15%) may only be stablecoins, WETH, ETH liquid staking tokens or WBTC.
#
# Only reserves enabled as collateral count. For the remainder rule and for `stable_usd`, **stable collateral** includes the yield-bearing wrappers of USD stablecoins present in the data (`STABLE_WRAPPERS`: sDAI, sUSDe), valued at their own oracle price. A wrapper is collateral only: it is not in the debt list, so a wallet that borrowed it would fail the debt rule (Aave does not list them as borrowable). The minimum debt of the panel is applied in Section 11.

# %%
DOMINANCE_THRESHOLD = 0.85


def classify_dominance(users: pd.DataFrame) -> pd.DataFrame:
    """Add shares, profile flags, Health Factor and weighted LT / LB to the (user, day) table."""
    out = users.copy()
    total = out["total_collateral_usd"]
    has_collateral = total > 0
    collateral = total.where(has_collateral)

    out["eth_share"] = (out["eth_core_usd"] / collateral).fillna(0.0)
    out["high_risk_share"] = ((out["uni_usd"] + out["link_usd"]) / collateral).fillna(0.0)
    a_disallowed = (out["other_usd"] + out["uni_usd"] + out["link_usd"]) > 0
    b_disallowed = out["other_usd"] > 0

    debt_ok = (out["total_debt_usd"] > 0) & (out["stable_debt_usd"] >= out["total_debt_usd"] * (1.0 - 1e-8))
    base = debt_ok & has_collateral & ~out["has_unpriced"]
    out["is_profile_a"] = base & ~a_disallowed & (out["eth_share"] >= DOMINANCE_THRESHOLD)
    out["is_profile_b"] = base & ~b_disallowed & (out["high_risk_share"] >= DOMINANCE_THRESHOLD)

    debt = out["total_debt_usd"]
    out["health_factor"] = np.where(debt > 0, out["thr_weighted_collateral_usd"] / debt.where(debt > 0), np.inf)
    out["lt_w"] = (out["thr_weighted_collateral_usd"] / collateral).astype("float64")
    out["lb_w"] = (out["lb_weighted_collateral_usd"] / collateral).astype("float64")
    return out


user_day = classify_dominance(user_day)

# Integer indices shared by every table below (see Section 2).
wallet_labels = user_day["scenario"] + "|" + user_day["user_id"]
user_day["wid"], WALLET_UNIQUES = pd.factorize(wallet_labels)
WALLET_LOOKUP = pd.Series(np.arange(len(WALLET_UNIQUES), dtype="int64"), index=WALLET_UNIQUES)
user_day["day"] = (user_day["date"] - user_day["scenario"].map(DAY1_DATE)).dt.days.astype("int64")
user_day["key"] = user_day["wid"].astype("int64") * DAY_STRIDE + user_day["day"]
user_day = user_day.sort_values("key", kind="mergesort").reset_index(drop=True)
assert user_day["key"].is_unique, "Duplicated (wallet, day) keys."

# Lookup arrays over ALL snapshot borrowers (not only profile wallets), used by Sections 14 and 18.
UD_INDEX = pd.Index(user_day["key"].to_numpy())
UD_DEBT = user_day["total_debt_usd"].to_numpy()
UD_HF = user_day["health_factor"].to_numpy()

for scenario_key, scenario_label in SCENARIO_KEYS:
    block = user_day.loc[user_day["scenario"] == scenario_key]
    print(f"[{scenario_key}] user-days: {len(block):,}  wallets: {block['wid'].nunique():,}")
    print(f"  Profile A user-days: {int(block['is_profile_a'].sum()):,}   Profile B user-days: {int(block['is_profile_b'].sum()):,}")
    print(
        f"  wallets with a Profile A day: {block.loc[block['is_profile_a'], 'wid'].nunique():,}   "
        f"with a Profile B day: {block.loc[block['is_profile_b'], 'wid'].nunique():,}"
    )

# %% [markdown]
# ## 11. Empirical Health Factor
#
# For qualifying user-days in both scenarios:
#
# $$
# HF_t = \frac{\sum_i (\text{Collateral USD}_{i,t} \times LT_{i,t})}{\text{Total Debt USD}_t}
# $$
#
# $LT_{i,t}$ is the liquidation threshold of reserve $i$ **on day $t$**, taken from `load_reserve_params` (or the eMode category LT when the user is in eMode and the reserve belongs to the category). The same weights give the collateral-weighted liquidation threshold `lt_w` and bonus `lb_w` that the LGD module needs.
#
# **Minimum debt.** A user-day enters the profile panel only if `total_debt_usd >= MIN_DEBT_USD` (USD 100). Smaller positions are dust that liquidators ignore, and the previous diagnostic showed that the wallets stuck at $HF \le 1$ were almost all dust. A USD 1,000 minimum is reported as a sensitivity in Section 17.5.
#
# If a user-day matches both profiles (not possible with an 85% threshold), it is labelled `A+B`. Wallet counts below use `.value_counts()` on the distinct profile labels of the surviving wallets.

# %%
MIN_DEBT_USD = 100.0
PANEL_COLUMNS = [
    "scenario", "user_id", "wid", "date", "day", "key", "profile", "is_profile_a", "is_profile_b",
    "total_collateral_usd", "total_debt_usd", "thr_weighted_collateral_usd", "health_factor", "lt_w", "lb_w",
    "eth_core_usd", "wbtc_usd", "uni_usd", "link_usd", "stable_usd", "other_usd",
]  # fmt: skip


def assign_profile_label(is_profile_a: np.ndarray, is_profile_b: np.ndarray) -> np.ndarray:
    """Profile label of each user-day: A, B or A+B."""
    return np.select([is_profile_a & is_profile_b, is_profile_a, is_profile_b], ["A+B", "A", "B"], default="")


def qualifying_panel(users: pd.DataFrame, min_debt_usd: float) -> pd.DataFrame:
    """User-days that satisfy Profile A and/or B and carry at least ``min_debt_usd`` of debt."""
    mask = (users["is_profile_a"] | users["is_profile_b"]) & (users["total_debt_usd"] >= min_debt_usd)
    panel = users.loc[mask].copy()
    panel["profile"] = assign_profile_label(panel["is_profile_a"].to_numpy(), panel["is_profile_b"].to_numpy())
    return panel[PANEL_COLUMNS].reset_index(drop=True)


hf_panel_df = qualifying_panel(user_day, MIN_DEBT_USD)

assert np.isfinite(hf_panel_df["health_factor"]).all(), "HF is NaN or infinite for a qualifying user-day."
assert (hf_panel_df["total_collateral_usd"] > 0).all(), "Ghost wallet: debt without collateral."
assert (hf_panel_df["total_debt_usd"] >= MIN_DEBT_USD).all()
assert hf_panel_df["lt_w"].between(0, 1).all() and hf_panel_df["lb_w"].ge(0).all()

for scenario_key, scenario_label in SCENARIO_KEYS:
    block = hf_panel_df.loc[hf_panel_df["scenario"] == scenario_key]
    wallet_profiles = block.groupby("wid")["profile"].agg(lambda labels: ",".join(sorted(set(labels))))
    print(f"=== {scenario_key} ===")
    print(f"Qualifying user-days (debt >= USD {MIN_DEBT_USD:,.0f}): {len(block):,}  wallets: {block['wid'].nunique():,}")
    print("Wallets by profile label over the whole window (value_counts):")
    print(wallet_profiles.value_counts().to_string())
    per_day = block.groupby("day")["wid"].nunique()
    print(
        f"Profile wallets per day: min={per_day.min():,} median={int(per_day.median()):,} max={per_day.max():,} "
        f"(day1={per_day.get(0, 0):,}, t={per_day.get(T_IDX, 0):,}, t+30={per_day.get(LAST_IDX, 0):,})"
    )
    assert per_day.nunique() > 1, f"{scenario_key}: the profile universe does not change between days."
    print()

print("Combined HF panel (head):")
display(hf_panel_df.head())

# %% [markdown]
# ## 12. Purity filter
#
# This section starts from `hf_panel_df` and builds the pure wallet panel:
#
# 1. **Purity filter.** A wallet's profile is the set of daily labels it shows **inside the calibration window** $[t-30, t]$ (days after $t$ are never used to classify a wallet). Hybrid wallets, Profile A on some days and B on others, are dropped; only strictly **A** or **B** wallets remain.
# 2. **Filter membership.** A pure wallet is *in the filter* on a day when that day's label equals its profile. After $t$ the same rule applies, which lets the out-of-time follow-up use the same definition of "leaving the profile filter".
#
# The result, `pure_panel`, is the observation table of the chain: one row per pure wallet and day in the filter.

# %%
def drop_hybrid_wallets(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Keep wallets whose calibration-window labels are all A or all B; return the panel and wallet labels."""
    calibration = panel.loc[panel["day"] <= T_IDX]
    flags = calibration.groupby("wid").agg(has_a=("is_profile_a", "max"), has_b=("is_profile_b", "max"))
    wallet_profile = pd.Series(
        np.select([flags["has_a"] & ~flags["has_b"], flags["has_b"] & ~flags["has_a"]], ["A", "B"], default="A,B"),
        index=flags.index,
        name="wallet_profile",
    )
    pure = panel.merge(wallet_profile, left_on="wid", right_index=True, how="inner")
    pure = pure.loc[pure["profile"].eq(pure["wallet_profile"])].drop(columns="wallet_profile")
    return pure.reset_index(drop=True), wallet_profile


pure_panel, wallet_profile = drop_hybrid_wallets(hf_panel_df)
wallet_scenario = pd.Series(WALLET_UNIQUES.str.split("|").str[0], index=np.arange(len(WALLET_UNIQUES)))

for scenario_key, scenario_label in SCENARIO_KEYS:
    labels = wallet_profile.loc[wallet_profile.index.isin(wallet_scenario.index[wallet_scenario == scenario_key])]
    print(f"[{scenario_key}] wallet labels inside [t-30, t] (value_counts):")
    print(labels.value_counts().to_string())
    block = pure_panel.loc[pure_panel["scenario"] == scenario_key]
    print(
        f"  pure wallets: {block.loc[block['day'] <= T_IDX, 'wid'].nunique():,}   "
        f"pure rows (calibration): {int((block['day'] <= T_IDX).sum()):,}   rows after t: {int((block['day'] > T_IDX).sum()):,}"
    )
print()
print(
    "Pure wallets by profile:",
    pure_panel.loc[pure_panel["day"] <= T_IDX].drop_duplicates("wid").groupby(["scenario", "profile"]).size().to_dict(),
)

# %% [markdown]
# ## 13. Current-state mapping
#
# Every observation (pure wallet, day in the filter) receives a **state**:
#
# | State | Definition |
# |---|---|
# | S2 | $HF \ge 2.0$ |
# | S3 | $1.5 \le HF < 2.0$ |
# | S4 | $1.25 \le HF < 1.5$ |
# | S5 | $1.10 \le HF < 1.25$ |
# | S6 | $1.0 < HF < 1.10$ |
# | **S7** | **liquidation event (technical default)**: $HF \le 1$ in the day's snapshot (`S7_hf`) **or** a `liquidationCall` on the wallet in the snapshot interval ending that day (`S7_liquidation_call`) |
# | **S1** | **observed repayment**: it is never the state of an observation, only the *next* state of a wallet that repaid (Section 14) |
#
# `pd.cut` assigns bins with a closed right edge. The single boundary $HF == 1.10$ is remapped from `S6` to `S5` so that the target band is $1.10 \le HF < 1.25$ and the critical band is $1.0 < HF < 1.10$.
#
# **Timing of the liquidation trigger.** A snapshot is taken at 00:00 UTC. A real liquidation happens minutes after $HF$ crosses 1 and restores the $HF$ before the next snapshot, so the daily $HF$ alone would miss the event. An event dated $d$ happens between snapshot $d$ and snapshot $d+1$: with `LIQUIDATION_DAY_OFFSET = 0` it is attached to the transition $d \to d+1$, and the wallet is `S7` on day $d+1$ (the first snapshot that follows the event, where its $HF$ is already restored). This keeps the calibration causal: a liquidation that happens after the snapshot of $t$ is not used to calibrate the chain, it belongs to the out-of-time window. Setting `LIQUIDATION_DAY_OFFSET = 1` instead attaches the liquidations dated $d+1$ to the transition $d \to d+1$, so that the state of day $d+1$ is overwritten by the liquidations of day $d+1$ itself (the liquidations of day $t$, which follow the snapshot of $t$, then enter the calibration). **Consequence of the default:** the liquidations of the valuation day itself (in particular the crash day, 2024-08-05, in the stress scenario) fall in the out-of-time period, not in the stress calibration. Repayments are always read the first way (`REPAY_DAY_OFFSET = 0`), because `S1` is defined by the debt that the **next** snapshot shows after the repay.
#
# **Repayment threshold.** `DUST_DEBT_USD = 1.0`: a repay event followed by a total debt of at most USD 1 (or by the wallet disappearing from the next snapshot) is `S1`.

# %%
DUST_DEBT_USD = 1.0  # S1: total debt after a repay event is <= this amount (USD)
# An event dated d belongs to the transition (d - offset) -> (d - offset + 1); see the text above.
LIQUIDATION_DAY_OFFSET = 0  # 0: events dated d -> transition d -> d+1 (causal); 1: same-day variant
REPAY_DAY_OFFSET = 0  # keep at 0: S1 is read from the next snapshot
LIQUIDATION_HF_THRESHOLD = 1.0

HF_STATE_BINS = [-np.inf, 1.0, 1.10, 1.25, 1.50, 2.0, np.inf]
HF_STATE_LABELS = ["S7", "S6", "S5", "S4", "S3", "S2"]
STATE_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
TRANSIENT_STATES = ["S2", "S3", "S4", "S5", "S6"]
ABSORBING_STATES = ["S1", "S7"]
EXIT_REASONS = ["S1_repay", "S7_hf", "S7_liquidation_call", "censored_filter_exit", "censored_window_end"]


def map_health_factor_to_state(health_factor: pd.Series) -> pd.Series:
    """Map Health Factor to the HF-based states S2-S7 (the liquidationCall trigger is added below)."""
    state = pd.cut(
        health_factor,
        bins=HF_STATE_BINS,
        labels=HF_STATE_LABELS,
        right=True,
        include_lowest=True,
    ).astype("object")
    # pd.cut(right=True) puts HF == 1.10 in S6; the target band starts at 1.10 inclusive.
    state = state.mask(np.isclose(health_factor, 1.10), "S5")
    return pd.Series(state, index=health_factor.index, name="state_hf")


def attach_wallet_codes(events: pd.DataFrame) -> pd.DataFrame:
    """Add the wallet code ``wid`` (NaN for users that never appear in a snapshot) and the day index."""
    out = events.copy()
    out["wid"] = WALLET_LOOKUP.reindex(out["scenario"] + "|" + out["user_id"]).to_numpy()
    out["day"] = (out["date"] - out["scenario"].map(DAY1_DATE)).dt.days.astype("int64")
    return out


def event_origin_keys(events: pd.DataFrame, day_offset: int) -> np.ndarray:
    """Keys (wid, origin day) of the transition ``origin -> origin + 1`` in which each event happened."""
    origin = events["day"] - day_offset
    valid = events["wid"].notna() & origin.between(0, LAST_IDX - 1)
    keys = events.loc[valid, "wid"].astype("int64") * DAY_STRIDE + origin[valid].astype("int64")
    return np.unique(keys.to_numpy())


def event_day_keys(events: pd.DataFrame) -> np.ndarray:
    """Keys (wid, event day) of the events, sorted, used to find the first liquidationCall after a day."""
    valid = events["wid"].notna() & events["day"].between(0, LAST_IDX)
    keys = events.loc[valid, "wid"].astype("int64") * DAY_STRIDE + events.loc[valid, "day"].astype("int64")
    return np.unique(keys.to_numpy())


liquidation_calls = attach_wallet_codes(liquidation_calls)
repays = attach_wallet_codes(repays)
LIQ_ORIGIN_KEYS = event_origin_keys(liquidation_calls, LIQUIDATION_DAY_OFFSET)
REPAY_ORIGIN_KEYS = event_origin_keys(repays, REPAY_DAY_OFFSET)
LIQ_EVENT_KEYS = event_day_keys(liquidation_calls)
print(
    f"Events mapped to a snapshot wallet: liquidationCalls {int(liquidation_calls['wid'].notna().sum()):,}/{len(liquidation_calls):,}, "
    f"repays {int(repays['wid'].notna().sum()):,}/{len(repays):,}"
)


def assign_day_states(panel: pd.DataFrame) -> pd.DataFrame:
    """Observation table with the day-level state: S7 by HF <= 1 or by a liquidationCall, else the HF bin."""
    obs = panel.sort_values("key", kind="mergesort").reset_index(drop=True)
    if "state_hf" not in obs.columns:
        obs["state_hf"] = map_health_factor_to_state(obs["health_factor"])
    hf_at_or_below_1 = obs["health_factor"].le(LIQUIDATION_HF_THRESHOLD).to_numpy()
    assert ((obs["state_hf"] == "S7").to_numpy() == hf_at_or_below_1).all(), "S7 bin and HF <= 1 disagree."
    key = obs["key"].to_numpy()
    liq_landing = np.isin(key - 1, LIQ_ORIGIN_KEYS)  # a liquidationCall in the interval that ends on this day
    obs["s7_trigger"] = np.where(hf_at_or_below_1, "S7_hf", np.where(liq_landing, "S7_liquidation_call", ""))
    obs["state"] = np.where(obs["s7_trigger"] != "", "S7", obs["state_hf"])
    return obs


day_states = assign_day_states(pure_panel)
calibration_states = day_states.loc[day_states["day"] <= T_IDX]

print("\nCurrent-state value counts by scenario and profile (calibration window, wallet-days):")
for (scenario_key, profile), block in calibration_states.groupby(["scenario", "profile"]):
    print(f"\n{scenario_key} | Profile {profile}")
    counts = block["state"].value_counts().reindex(STATE_ORDER[1:]).fillna(0).astype(int)
    triggers = block["s7_trigger"].value_counts()
    print(counts.to_string())
    print(
        f"  S7 by trigger: S7_hf={int(triggers.get('S7_hf', 0))}, "
        f"S7_liquidation_call={int(triggers.get('S7_liquidation_call', 0))}"
    )

# %% [markdown]
# ## 14. Absorbing states, censoring and transitions without gaps
#
# The old notebook inferred `S1` from a wallet disappearing and forward-filled the first absorbing state to the end of the window. Both are replaced by observed events.
#
# For each wallet-day observation $d$ in a **transient** state (S2-S6), the interval to the next snapshot, $d \to d+1$, ends in exactly one outcome (checked in this order):
#
# 1. **`S7_hf` / `S7_liquidation_call`**: a `liquidationCall` happened in the interval, or the wallet is observed on $d+1$ with $HF \le 1$. The label is `S7_hf` when the $d+1$ snapshot shows $HF \le 1$, `S7_liquidation_call` when the event is the only evidence. The wallet need not be in the filter on $d+1$: a liquidation changes its collateral mix and debt.
# 2. **`S1_repay`**: a `repay` event happened in the interval and the wallet's total debt on the $d+1$ snapshot is at most `DUST_DEBT_USD` (or the wallet is absent from that snapshot). Liquidation wins a tie with repayment.
# 3. **A transient state** $S_j$: the wallet is observed on $d+1$, in the filter and not absorbed.
# 4. **No transition**: the wallet is not observed on $d+1$ and nothing explains it (a gap, or a filter exit).
#
# **Absorbing, not censored.** `S1` and `S7` are absorbing by definition. The entry transition $S_i(d) \to S1/S7(d+1)$ is counted once and the trajectory **ends** at the first hit: nothing is forward-filled. A trajectory that is observed for the first time already in `S7` ($HF \le 1$) ends there without an entry transition. Section 16 sets the `S1` and `S7` rows of $\mathbf{P}$ to the identity by definition, and their rows of $\mathbf{N}$ stay at 0.
#
# **Spells.** A repayment closes a *spell*: if the wallet borrows again inside the window, its next observation opens a **new spell**, a new trajectory that counts as a late entrant from its re-entry day (`spell_id`, internal key `sid = wid * 16 + spell`). `exit_reason` is therefore reported per spell, and the number of spells is the number of wallets plus the re-entries after `S1`. After `S7` the wallet is **not followed again** inside the window (project rule): its later observations are ignored for $\mathbf{N}$ and the cell reports how many there are. The bootstrap of Section 17.2 still resamples wallets, so that all the spells of a wallet stay together.
#
# **Transitions only between consecutive observed days ($\Delta = 1$ day).** A transition that crosses a gap is not counted and no state is carried over, but the wallet resumes contributing from its next pair of consecutive days.
#
# **Censoring** applies only to wallets whose outcome is not observed, and cuts the trajectory at the last observation (no `S1`):
#
# - `censored_filter_exit`: the wallet leaves the profile filter (debt under the minimum, a different collateral mix, a different label) with debt $> 0$ and **never returns** inside the window.
# - `censored_window_end`: the wallet is still transient on $t$.
#
# `exit_reason` takes five values: `S1_repay`, `S7_hf`, `S7_liquidation_call`, `censored_filter_exit`, `censored_window_end`. Section 15 compares the Health Factor at censoring with the other wallets to look for informative censoring.
#
# The same function runs over the 61 days. Calibration only keeps landings on day $\le t$; the out-of-time outcome of Section 18 uses the landings in $(t, t+30]$.

# %%
ABSORBING_PRIORITY = {"S7_hf": 0, "S7_liquidation_call": 1, "S1_repay": 2}
SPELL_STRIDE = 16  # at most 16 spells per wallet (repay, borrow again, ...); sid = wid * SPELL_STRIDE + spell


def lookup_snapshot(values: np.ndarray, query_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Value of the all-borrowers snapshot table at each (wid, day) key, and whether the key exists."""
    position = UD_INDEX.get_indexer(query_keys)
    found = position >= 0
    out = np.full(len(query_keys), np.nan)
    out[found] = values[position[found]]
    return out, found


def build_trajectories(obs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Apply the Section 14 rules: outcome of every interval ``d -> d+1`` and the absorbing events.

    Returns the observation table with the interval outcome of each row and the table of absorbing
    events ``(wid, landing day, type)``: the day on which a wallet enters S1 or S7.
    """
    obs = obs.sort_values("key", kind="mergesort").reset_index(drop=True)
    key = obs["key"].to_numpy()
    n_obs = len(obs)
    state = obs["state"].to_numpy(dtype=object)
    health_factor = obs["health_factor"].to_numpy()

    # The same wallet observed on the next day: key + 1 (keys never cross wallets).
    next_pos = np.minimum(np.searchsorted(key, key + 1), n_obs - 1)
    has_next = key[next_pos] == key + 1
    next_obs_state = np.where(has_next, state[next_pos], "")
    next_hf_at_or_below_1 = has_next & (health_factor[next_pos] <= LIQUIDATION_HF_THRESHOLD)

    liq_in_interval = np.isin(key, LIQ_ORIGIN_KEYS)
    repay_in_interval = np.isin(key, REPAY_ORIGIN_KEYS)
    debt_next, in_snapshot_next = lookup_snapshot(UD_DEBT, key + 1)

    origin_transient = state != "S7"
    to_s7 = origin_transient & (liq_in_interval | next_hf_at_or_below_1)
    to_s1 = (
        origin_transient
        & ~to_s7
        & repay_in_interval
        & (~in_snapshot_next | (np.nan_to_num(debt_next, nan=0.0) <= DUST_DEBT_USD))
    )
    to_transient = origin_transient & ~to_s7 & ~to_s1 & has_next
    assert not (to_transient & (next_obs_state == "S7")).any(), "A transient transition lands on S7."

    outcome = np.select(
        [to_s7 & next_hf_at_or_below_1, to_s7, to_s1, to_transient],
        ["S7_hf", "S7_liquidation_call", "S1_repay", "transient"],
        default="none",
    )
    outcome = np.where(origin_transient, outcome, "")  # rows already in S7 are absorbed: no outgoing interval
    next_state = np.select(
        [np.isin(outcome, ["S7_hf", "S7_liquidation_call"]), outcome == "S1_repay", outcome == "transient"],
        ["S7", "S1", next_obs_state],
        default="",
    )

    obs = obs.assign(
        has_next_obs=has_next,
        liq_in_interval=liq_in_interval,
        repay_in_interval=repay_in_interval,
        outcome=outcome,
        next_state=next_state,
    )

    # Absorbing events: (a) observations already in S7, (b) intervals that end in S7 or S1.
    entered_s7_obs = obs.loc[~origin_transient, ["wid", "day", "s7_trigger"]].rename(
        columns={"day": "landing", "s7_trigger": "type"}
    )
    ended_interval = obs.loc[np.isin(outcome, list(ABSORBING_PRIORITY)), ["wid", "day", "outcome"]].rename(
        columns={"day": "landing", "outcome": "type"}
    )
    ended_interval["landing"] += 1
    entered_s7_obs["origin_day"] = entered_s7_obs["landing"]
    ended_interval["origin_day"] = ended_interval["landing"] - 1
    events = pd.concat([entered_s7_obs, ended_interval], ignore_index=True)
    events["priority"] = events["type"].map(ABSORBING_PRIORITY)
    obs, events = assign_spells(obs, events)
    return {"obs": obs, "events": events}


def assign_spells(obs: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each wallet into spells: a repayment (S1) closes a spell and the next observation opens a new one.

    When the first absorbing event of a spell is an S1, the observations and events of the wallet that
    originate after its landing day move to the next spell. After an S7 nothing moves (the wallet is not
    followed again). Observations and events get ``spell`` and ``sid = wid * SPELL_STRIDE + spell``.
    """
    obs, events = obs.assign(spell=0), events.assign(spell=0)
    for _ in range(SPELL_STRIDE):
        first = events.sort_values(["wid", "spell", "landing", "priority"], kind="mergesort").drop_duplicates(["wid", "spell"])
        cut = first.loc[first["type"] == "S1_repay", ["wid", "spell", "landing"]].rename(columns={"landing": "cut_day"})
        obs_cut = obs[["wid", "spell"]].merge(cut, on=["wid", "spell"], how="left")["cut_day"].to_numpy()
        events_cut = events[["wid", "spell"]].merge(cut, on=["wid", "spell"], how="left")["cut_day"].to_numpy()
        move_obs = obs["day"].to_numpy() > obs_cut  # False where there is no cut (NaN)
        move_events = events["origin_day"].to_numpy() > events_cut
        if not (move_obs.any() or move_events.any()):
            break
        obs["spell"] += move_obs.astype("int64")
        events["spell"] += move_events.astype("int64")
    else:
        raise ValueError(f"A wallet has more than {SPELL_STRIDE} spells.")
    for frame in (obs, events):
        frame["sid"] = frame["wid"].astype("int64") * SPELL_STRIDE + frame["spell"]
    return obs, events


def first_absorption(events: pd.DataFrame, first_day: int, last_day: int) -> pd.DataFrame:
    """First absorbing event of each spell with landing day in ``[first_day, last_day]``, indexed by ``sid``."""
    window = events.loc[events["landing"].between(first_day, last_day)]
    first = window.sort_values(["sid", "landing", "priority"], kind="mergesort").drop_duplicates("sid")
    return first.set_index("sid")[["landing", "type"]]


def summarise_wallets(trajectories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per spell observed in the calibration window: outcome, exit_reason and last observation."""
    obs = trajectories["obs"]
    calibration = obs.loc[obs["day"] <= T_IDX]
    first_hit = first_absorption(trajectories["events"], 0, T_IDX)

    grouped = calibration.groupby("sid", sort=True)
    summary = grouped.agg(
        wid=("wid", "first"),
        spell=("spell", "first"),
        scenario=("scenario", "first"),
        user_id=("user_id", "first"),
        profile=("profile", "first"),
        first_day=("day", "min"),
        last_day=("day", "max"),
        n_obs=("day", "size"),
    )
    summary = summary.join(first_hit)
    absorbed = summary["type"].notna()

    # Last observation that belongs to the trajectory (strictly before the absorbing landing day).
    landing = summary["landing"].reindex(calibration["sid"]).to_numpy()
    in_trajectory = calibration.loc[np.isnan(landing) | (calibration["day"].to_numpy() < landing)]
    last_row = in_trajectory.groupby("sid").tail(1).set_index("sid")
    summary["last_traj_day"] = last_row["day"]
    summary["last_traj_hf"] = last_row["health_factor"]
    summary["last_traj_state"] = last_row["state"]
    summary["last_traj_debt_usd"] = last_row["total_debt_usd"]

    summary["exit_reason"] = np.where(
        absorbed,
        summary["type"],
        np.where(summary["last_day"] == T_IDX, "censored_window_end", "censored_filter_exit"),
    )
    summary["final_state"] = np.select(
        [summary["type"] == "S1_repay", summary["type"].isin(["S7_hf", "S7_liquidation_call"])],
        ["S1", "S7"],
        default=calibration.groupby("sid").tail(1).set_index("sid")["state"].reindex(summary.index),
    )
    summary["event_day"] = np.where(absorbed, summary["landing"], summary["last_day"]).astype("int64")
    summary["is_late_entrant"] = summary["first_day"] > 0
    return summary


def counted_transitions(trajectories: dict[str, pd.DataFrame], summary: pd.DataFrame) -> pd.DataFrame:
    """Calibration transitions: from transient observations, before the first absorption of their spell."""
    obs = trajectories["obs"]
    landing = summary["landing"].reindex(obs["sid"]).to_numpy()
    before_absorption = np.isnan(landing) | (obs["day"].to_numpy() < landing)
    counted = (
        obs["outcome"].isin(["S7_hf", "S7_liquidation_call", "S1_repay", "transient"])
        & (obs["day"] + 1 <= T_IDX)
        & before_absorption
    )
    transitions = obs.loc[counted, ["scenario", "user_id", "wid", "sid", "profile", "date", "day", "state", "next_state"]].copy()
    transitions["next_date"] = transitions["date"] + pd.Timedelta(1, unit="D")
    transitions["transition_pair"] = transitions["state"] + " -> " + transitions["next_state"]
    return transitions.reset_index(drop=True)


def run_chain(panel: pd.DataFrame) -> dict:
    """Section 13 and 14 end to end for any observation panel (used by the sensitivities of Section 17)."""
    trajectories = build_trajectories(assign_day_states(panel))
    summary = summarise_wallets(trajectories)
    return {"traj": trajectories, "summary": summary, "transitions": counted_transitions(trajectories, summary)}


chain = {"traj": build_trajectories(day_states)}
chain["summary"] = summarise_wallets(chain["traj"])
chain["transitions"] = counted_transitions(chain["traj"], chain["summary"])

wallet_summary = chain["summary"]
markov_transitions_df = chain["transitions"]

# Observations that belong to a trajectory: calibration days up to (and including) the absorbing landing day.
obs_all = chain["traj"]["obs"]
hit_landing = wallet_summary["landing"].reindex(obs_all["sid"]).to_numpy()
hit_type = wallet_summary["type"].reindex(obs_all["sid"]).to_numpy(dtype=object)
in_calibration = obs_all["day"].to_numpy() <= T_IDX
after_hit = in_calibration & (obs_all["day"].to_numpy() > hit_landing)
assert not (after_hit & (hit_type == "S1_repay")).any(), "Observations left in a spell after its S1."
markov_states_df = obs_all.loc[in_calibration & ~after_hit].copy()
n_post_s7_days = int(after_hit.sum())  # S7 closes the wallet: later observations are ignored
n_post_s7_wallets = int(obs_all.loc[after_hit, "wid"].nunique())
n_new_spells = int((wallet_summary["spell"] > 0).sum())

events_cal = chain["traj"]["events"]
print("Absorbing events by type, calibration window (all events, before keeping the first per spell):")
print(events_cal.loc[events_cal["landing"] <= T_IDX, "type"].value_counts().to_string())
print()
print(
    f"Pure wallets observed in [t-30, t]: {wallet_summary['wid'].nunique():,}; spells (trajectories): {len(wallet_summary):,} "
    f"({n_new_spells:,} opened by a wallet that repaid and borrowed again)"
)
print(f"Valid daily transitions (N): {len(markov_transitions_df):,}")
print(f"Wallet-days after an S7, ignored (the wallet is not followed again): {n_post_s7_days:,} ({n_post_s7_wallets:,} wallets re-enter the filter after S7)")
print(
    "Spells whose first observation is already S7 (no entry transition): "
    f"{int(((wallet_summary['first_day'] == wallet_summary['landing']) & wallet_summary['type'].notna()).sum()):,}"
)
print()
for pair in ("S2 -> S1", "S5 -> S6", "S5 -> S7", "S6 -> S7"):
    n_pair = int((markov_transitions_df["transition_pair"] == pair).sum())
    print(f"  {pair}: {n_pair:,}")

zombie = markov_transitions_df["state"].isin(ABSORBING_STATES)
print(f"Transitions out of absorbing states (should be 0): {int(zombie.sum()):,}")

# %% [markdown]
# ## 15. Sample-size verification
#
# Cross-sectional `state` counts (wallet-days that belong to a trajectory), the `exit_reason` of every spell, and the empirical $t \to t+1$ pair counts ($\mathbf{N}$) are reported **separately** for each market regime (`baseline_mar2024`, `stress_aug2024`) and collateral profile (A, B), before assembling the transition matrices. The cell also asserts the rules of Section 14 against the raw events:
#
# - `exit_reason` counts (one per spell) add up to the number of spells, and the spells belong to the pure wallets;
# - no transition spans more than one day, and no transition leaves an absorbing state or follows the first absorption;
# - no `censored_filter_exit` wallet ends in `S1`;
# - `S1` only after an observed repayment and `S7` only with $HF \le 1$ or a `liquidationCall` (no contaminated exits).
#
# **Informative-censoring check.** If wallets left the filter *because* they were close to liquidation, ignoring them would bias $\mathbf{P}$. The table compares the Health Factor of the last observation of `censored_filter_exit` wallets with the other exit reasons (for absorbed wallets: the last observation before the event) and tests it against `censored_window_end` with a Kolmogorov-Smirnov test. `exit_detail` shows what happened to a filter-exit wallet on the day after its last observation.

# %%
SCENARIO_PRINT_ORDER = ["baseline", "stress"]
PROFILE_PRINT_ORDER = ["A", "B"]
PROFILE_KEYS = ["A", "B"]
PROFILE_TITLES = {
    "A": "Profile A (ETH dominance)",
    "B": "Profile B (UNI+LINK dominance)",
}

print("=== Current-state distribution by scenario and profile (wallet-days in a trajectory) ===")
for (scenario, profile), block in markov_states_df.groupby(["scenario", "profile"], sort=False):
    n_wallets = block["user_id"].nunique()
    print(f"\n{SCENARIO_LABEL.get(scenario, scenario)} | Profile {profile}  (n_rows={len(block):,}, n_wallets={n_wallets:,})")
    print(block["state"].value_counts().reindex(STATE_ORDER[1:]).fillna(0).astype(int).to_string())

# --- exit_reason counts ---------------------------------------------------------------------------
segment_index = pd.MultiIndex.from_product([SCENARIO_PRINT_ORDER, PROFILE_PRINT_ORDER], names=["scenario", "profile"])
exit_table = (
    wallet_summary.groupby(["scenario", "profile", "exit_reason"])
    .size()
    .unstack("exit_reason")
    .reindex(index=segment_index, columns=EXIT_REASONS)
    .fillna(0)
    .astype(int)
)
exit_table["total_spells"] = exit_table.sum(axis=1)
exit_table.insert(0, "wallets", wallet_summary.groupby(["scenario", "profile"])["wid"].nunique().reindex(segment_index, fill_value=0))
print("\n=== exit_reason counts (one row per spell observed in [t-30, t]; a spell is a wallet's trajectory) ===")
display(exit_table)

n_pure_wallets = pure_panel.loc[pure_panel["day"] <= T_IDX].groupby(["scenario", "profile"])["wid"].nunique()
n_pure_spells = chain["traj"]["obs"].loc[lambda frame: frame["day"] <= T_IDX].groupby(["scenario", "profile"])["sid"].nunique()
assert (exit_table["wallets"].to_numpy() == n_pure_wallets.reindex(segment_index, fill_value=0).to_numpy()).all(), (
    "The spells do not cover the pure wallets."
)
assert (exit_table["total_spells"].to_numpy() == n_pure_spells.reindex(segment_index, fill_value=0).to_numpy()).all(), (
    "exit_reason counts do not add up to the number of spells."
)

# --- audits against the raw events ----------------------------------------------------------------
assert ((markov_transitions_df["next_date"] - markov_transitions_df["date"]).dt.days == 1).all(), "Transition with delta > 1 day."
assert not markov_transitions_df["state"].isin(ABSORBING_STATES).any(), "Transition out of an absorbing state."
assert markov_transitions_df["state"].isin(TRANSIENT_STATES).all()
first_landing = wallet_summary["landing"].reindex(markov_transitions_df["sid"]).to_numpy()
assert (np.isnan(first_landing) | (markov_transitions_df["day"].to_numpy() < first_landing)).all(), "Transition after the first absorption."

# Independent check of consecutive days: every transient -> transient pair is two rows one calendar day apart.
transient_pairs = markov_transitions_df.loc[~markov_transitions_df["next_state"].isin(ABSORBING_STATES)]
check = transient_pairs.merge(
    day_states[["scenario", "user_id", "date", "state"]].rename(columns={"date": "next_date", "state": "state_on_next_date"}),
    on=["scenario", "user_id", "next_date"],
    how="left",
)
assert (check["state_on_next_date"] == check["next_state"]).all(), "A transient transition does not match the next-day observation."
assert not wallet_summary.loc[wallet_summary["exit_reason"] == "censored_filter_exit", "final_state"].eq("S1").any()
assert wallet_summary.loc[wallet_summary["exit_reason"].str.startswith("censored"), "final_state"].isin(TRANSIENT_STATES).all()

absorbed_wallets = wallet_summary.loc[wallet_summary["type"].notna()].copy()
absorbed_wallets["event_date"] = absorbed_wallets["scenario"].map(DAY1_DATE) + pd.to_timedelta(absorbed_wallets["event_day"], unit="D")
origin_date = absorbed_wallets["event_date"] - pd.Timedelta(1, unit="D")


def has_event(frame: pd.DataFrame, wallets: pd.DataFrame, origin_dates: pd.Series, day_offset: int) -> pd.Series:
    """Whether each wallet has a raw event dated ``origin_date + day_offset`` (checked on the raw event table)."""
    keys = set(zip(frame["scenario"], frame["user_id"], frame["date"]))
    wanted = zip(wallets["scenario"], wallets["user_id"], origin_dates + pd.Timedelta(day_offset, unit="D"))
    return pd.Series([item in keys for item in wanted], index=wallets.index)


landing_keys = absorbed_wallets["wid"].to_numpy().astype("int64") * DAY_STRIDE + absorbed_wallets["landing"].to_numpy().astype("int64")
snapshot_debt, in_snapshot = lookup_snapshot(UD_DEBT, landing_keys)
snapshot_hf, _ = lookup_snapshot(UD_HF, landing_keys)
s1_rows = absorbed_wallets["type"] == "S1_repay"
s7_hf_rows = absorbed_wallets["type"] == "S7_hf"
s7_liq_rows = absorbed_wallets["type"] == "S7_liquidation_call"

s1_ok = has_event(repays, absorbed_wallets, origin_date, REPAY_DAY_OFFSET) & (
    ~in_snapshot | (np.nan_to_num(snapshot_debt, nan=0.0) <= DUST_DEBT_USD)
)
assert s1_ok[s1_rows].all(), "S1 without an observed repayment that leaves the debt at dust."
assert (snapshot_hf[s7_hf_rows.to_numpy()] <= LIQUIDATION_HF_THRESHOLD).all(), "S7_hf with HF > 1."
assert has_event(liquidation_calls, absorbed_wallets, origin_date, LIQUIDATION_DAY_OFFSET)[s7_liq_rows].all(), (
    "S7_liquidation_call without a liquidationCall."
)
print(
    f"\nAudits passed: {int(s1_rows.sum()):,} S1 spells with an observed repay, {int(s7_hf_rows.sum()):,} S7_hf with HF <= 1, "
    f"{int(s7_liq_rows.sum()):,} S7_liquidation_call with a liquidationCall; 0 contaminated exits."
)

# --- informative censoring ------------------------------------------------------------------------
exit_rows = wallet_summary.loc[wallet_summary["exit_reason"] == "censored_filter_exit"]
next_key = exit_rows["wid"].to_numpy().astype("int64") * DAY_STRIDE + exit_rows["last_day"].to_numpy().astype("int64") + 1
debt_after, present_after = lookup_snapshot(UD_DEBT, next_key)
exit_detail = pd.Series(
    np.select(
        [~present_after, debt_after <= DUST_DEBT_USD, debt_after < MIN_DEBT_USD],
        ["vanished_no_event", "dust_debt_no_repay_event", "below_min_debt"],
        default="profile_or_label_change",
    ),
    index=exit_rows.index,
    name="exit_detail",
)
print("\n=== What happens to a censored_filter_exit wallet on the day after its last observation ===")
display(exit_detail.groupby([exit_rows["scenario"], exit_rows["profile"], exit_detail]).size().unstack(fill_value=0))

hf_quantiles = (
    wallet_summary.groupby(["scenario", "profile", "exit_reason"])["last_traj_hf"]
    .describe(percentiles=[0.1, 0.5, 0.9])[["count", "10%", "50%", "90%"]]
    .round(3)
)
print("\n=== Health Factor of the last observation in the trajectory, by exit_reason ===")
display(hf_quantiles)

ks_records = []
for scenario_key, _ in SCENARIO_KEYS:
    for profile in PROFILE_KEYS:
        block = wallet_summary.loc[(wallet_summary["scenario"] == scenario_key) & (wallet_summary["profile"] == profile)]
        exits = block.loc[block["exit_reason"] == "censored_filter_exit", "last_traj_hf"].dropna()
        stays = block.loc[block["exit_reason"] == "censored_window_end", "last_traj_hf"].dropna()
        record = {"scenario": scenario_key, "profile": profile, "n_filter_exit": len(exits), "n_window_end": len(stays)}
        if len(exits) >= 5 and len(stays) >= 5:
            ks_statistic, ks_p_value = ks_two_sample(exits, stays)
            record.update(
                median_filter_exit=exits.median(), median_window_end=stays.median(), ks_statistic=ks_statistic, ks_p_value=ks_p_value
            )
        ks_records.append(record)
print("\nKolmogorov-Smirnov: HF at filter exit versus HF of wallets still transient on t")
display(pd.DataFrame(ks_records).set_index(["scenario", "profile"]).round(4))

# --- transition pair counts -----------------------------------------------------------------------
transition_counts = (
    markov_transitions_df.groupby(["scenario", "profile", "transition_pair"], sort=False).size().rename("n").reset_index()
)

print("\n=== Daily transition pairs by scenario and profile (empirical N) ===")
for scenario in SCENARIO_PRINT_ORDER:
    scenario_label = SCENARIO_LABEL[scenario]
    print(f"\n-------- {scenario_label} --------")
    for profile in PROFILE_PRINT_ORDER:
        block = transition_counts.loc[
            (transition_counts["scenario"] == scenario) & (transition_counts["profile"] == profile)
        ].sort_values("n", ascending=False)
        n_total = int(block["n"].sum())
        print(f"\n{scenario_label} | Profile {profile}  (N={n_total:,})")
        if block.empty:
            print("  (no transitions)")
            continue
        pair_counts = block.set_index("transition_pair")["n"]
        print(pair_counts.to_string())
        for critical_pair in ("S5 -> S6", "S6 -> S7", "S5 -> S7", "S2 -> S1"):
            print(f"  focus {critical_pair}: {int(pair_counts.get(critical_pair, 0)):,}")

print("\n=== Transition-pair counts grouped by scenario, profile, and pair ===")
transition_counts_display = transition_counts.sort_values(["scenario", "profile", "n"], ascending=[True, True, False]).reset_index(drop=True)
display(transition_counts_display)
print("\nTotal empirical transitions N by scenario and profile:")
print(
    markov_transitions_df.groupby(["scenario", "profile"])
    .size()
    .rename("N")
    .reindex(pd.MultiIndex.from_product([SCENARIO_PRINT_ORDER, PROFILE_PRINT_ORDER], names=["scenario", "profile"]))
    .fillna(0)
    .astype(int)
    .to_string()
)

# %% [markdown]
# ## 16. Empirical transition matrices
#
# Four daily count matrices $\mathbf{N}$ and probability matrices $\mathbf{P}$ are assembled from `transition_pair` in the calibration transitions: one per market regime (`baseline_mar2024`, `stress_aug2024`) and collateral profile (A, B).
#
# Rows and columns follow `STATE_ORDER`. Each transient row (S2-S6) is normalized by its sum. **The rows of `S1` and `S7` are set to the identity by definition** ($P_{S1,S1} = P_{S7,S7} = 1$): they are absorbing states, so no transition out of them is ever observed and their rows of $\mathbf{N}$ are 0 (the old notebook obtained them from forward-filled frozen paths). A transient row with no observed transition cannot be estimated; it takes the identity so that $\mathbf{P}$ stays row-stochastic, and it is flagged in the printout (its `n` is 0, `row_empty` is `True` in the export, and its $\mathbf{P}^{30}$ row is left empty there).

# %%
STATE_INDEX = {state: idx for idx, state in enumerate(STATE_ORDER)}
N_STATES = len(STATE_ORDER)
ABSORBING_IDX = [STATE_INDEX[state] for state in ABSORBING_STATES]
S7_STATE_IDX = STATE_INDEX["S7"]


def row_normalize(counts: np.ndarray, empty_value: float) -> np.ndarray:
    """Row-normalize stacked count matrices (..., K, K); empty rows take empty_value."""
    row_totals = counts.sum(axis=-1, keepdims=True)
    safe_totals = np.where(row_totals > 0, row_totals, 1.0)
    return np.where(row_totals > 0, counts / safe_totals, empty_value)


def set_absorbing_rows(matrix: np.ndarray) -> np.ndarray:
    """Set the S1 and S7 rows of stacked matrices (..., K, K) to the identity (absorbing by definition)."""
    out = matrix.copy()
    for idx in ABSORBING_IDX:
        out[..., idx, :] = 0.0
        out[..., idx, idx] = 1.0
    return out


def fill_empty_rows(matrix: np.ndarray) -> np.ndarray:
    """Replace the all-NaN rows (transient states never observed as origin) by identity rows."""
    empty = np.isnan(matrix).all(axis=-1, keepdims=True)
    return np.where(empty, np.eye(N_STATES), matrix)


def build_empirical_transition_matrices(subset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return empirical count matrix N and row-stochastic matrix P.

    N is the frequency of transition_pair values (S1 and S7 rows are 0). In P the S1 and S7 rows are
    identity by definition, and transient rows without observed transitions are identity as well.
    """
    counts = pd.DataFrame(0.0, index=STATE_ORDER, columns=STATE_ORDER)
    if not subset.empty:
        counts = (
            pd.crosstab(subset["state"], subset["next_state"])
            .reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
            .astype(float)
        )
    assert counts.loc[ABSORBING_STATES].to_numpy().sum() == 0, "Counts found on an absorbing row."
    daily = fill_empty_rows(set_absorbing_rows(row_normalize(counts.to_numpy(dtype=float), np.nan)))
    return counts, pd.DataFrame(daily, index=STATE_ORDER, columns=STATE_ORDER)


def unobserved_origin_states(counts_n: pd.DataFrame) -> list[str]:
    """Transient states without a single observed transition."""
    return [state for state in TRANSIENT_STATES if counts_n.loc[state].sum() == 0]


empirical_N: dict[tuple[str, str], pd.DataFrame] = {}
empirical_P: dict[tuple[str, str], pd.DataFrame] = {}
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

plot_idx = 0
for scenario_key, scenario_label in SCENARIO_KEYS:
    for profile in PROFILE_KEYS:
        subset = markov_transitions_df.loc[
            (markov_transitions_df["scenario"] == scenario_key) & (markov_transitions_df["profile"] == profile)
        ].copy()
        counts_n, transition_p = build_empirical_transition_matrices(subset)
        empirical_N[(scenario_label, profile)] = counts_n
        empirical_P[(scenario_label, profile)] = transition_p
        assert np.allclose(transition_p.sum(axis=1), 1.0), "A row of P does not sum to 1."

        ax = axes[plot_idx]
        sns.heatmap(
            transition_p,
            ax=ax,
            annot=True,
            fmt=".4f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.4,
            cbar_kws={"label": "Transition probability"},
        )
        ax.set_title(f"Empirical daily P  |  {scenario_label}  |  {PROFILE_TITLES[profile]}")
        ax.set_xlabel("Next state")
        ax.set_ylabel("Current state")

        print(
            f"{scenario_label} | {PROFILE_TITLES[profile]}: N={len(subset):,}  "
            f"N(S2..S6 -> S1)={int(counts_n['S1'].sum()):,}  N(S2..S6 -> S7)={int(counts_n['S7'].sum()):,}  "
            f"P(S1->S1)={transition_p.loc['S1', 'S1']:.1f}  P(S7->S7)={transition_p.loc['S7', 'S7']:.1f}"
        )
        print(f"  row sums: {transition_p.sum(axis=1).round(6).tolist()}")
        missing_states = unobserved_origin_states(counts_n)
        if missing_states:
            print(f"  WARNING: no observed transitions from {missing_states}; their rows are identity (not estimated).")
        plot_idx += 1

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 17. Statistical robustness validation (actuarial audit)
#
# The empirical daily matrices in Section 16 are point estimates. This section adds five audit checks on the calibration transitions `markov_transitions_df`, always split by market regime (`baseline_mar2024`, `stress_aug2024`) and collateral profile (A, B):
#
# 1. **Absolute frequency matrices $\mathbf{N}$**: the sample depth behind every probability.
# 2. **Wallet-level bootstrap**: 95% confidence intervals for the one-step probability of a liquidation event $P_{i,S7}$ and for the 30-day PD implied by $\mathbf{P}^{30}$.
# 3. **Time homogeneity**: $\mathbf{P}$ estimated on days 1-15 versus days 16-30, with a likelihood-ratio test per origin state.
# 4. **Threshold sensitivity**: $\mathbf{P}$ re-estimated after shifting the intermediate Health Factor cut points by $\pm 0.05$.
# 5. **Minimum-debt sensitivity**: transition counts and 30-day PD by origin state with a USD 1,000 minimum debt instead of USD 100.
#
# The cells reuse `STATE_ORDER`, `SCENARIO_KEYS`, `PROFILE_KEYS`, `PROFILE_TITLES`, and `build_empirical_transition_matrices` from Section 16, `run_chain` from Section 14, and `HF_STATE_BINS` / `HF_STATE_LABELS` from Section 13.

# %% [markdown]
# ### 17.1 Absolute frequency matrices N
#
# Rows are the state at day $t$ and columns the state at $t+1$, both ordered `S1` to `S7`. Row totals give the number of observed transitions behind each row of $\mathbf{P}$; the `S1` and `S7` rows are 0 by construction. Origin states with fewer than `MIN_ROW_DEPTH` transitions are flagged as thin; their probabilities should be read together with the bootstrap intervals in 17.2.

# %%
MIN_ROW_DEPTH = 30


def subset_transitions(transitions: pd.DataFrame, scenario_key: str, profile: str) -> pd.DataFrame:
    """Return the transitions of one market regime and collateral profile."""
    mask = transitions["scenario"].eq(scenario_key) & transitions["profile"].eq(profile)
    return transitions.loc[mask]


absolute_N: dict[tuple[str, str], pd.DataFrame] = {}

for scenario_key, scenario_label in SCENARIO_KEYS:
    for profile in PROFILE_KEYS:
        subset = subset_transitions(markov_transitions_df, scenario_key, profile)
        counts_n, _ = build_empirical_transition_matrices(subset)
        counts_n = counts_n.astype(int)
        counts_n.index.name = "current_state"
        counts_n.columns.name = "next_state"
        absolute_N[(scenario_label, profile)] = counts_n

        frequency_table = counts_n.copy()
        frequency_table["row_total"] = frequency_table.sum(axis=1)
        frequency_table.loc["column_total"] = frequency_table.sum(axis=0)

        row_depth = counts_n.loc[TRANSIENT_STATES].sum(axis=1)
        thin_rows = row_depth.loc[row_depth < MIN_ROW_DEPTH]
        print(
            f"\n{scenario_label} | {PROFILE_TITLES[profile]}  "
            f"(N={int(row_depth.sum()):,}, wallets with a transition={subset['user_id'].nunique():,})"
        )
        display(frequency_table)
        if thin_rows.empty:
            print(f"  All transient origin states have at least {MIN_ROW_DEPTH} transitions.")
        else:
            thin_text = ", ".join(f"{state}={n}" for state, n in thin_rows.items())
            print(f"  Thin origin states (< {MIN_ROW_DEPTH} transitions): {thin_text}")

# %% [markdown]
# ### 17.2 Wallet-level bootstrap confidence intervals
#
# Daily transitions of the same wallet are serially dependent, so resampling individual transitions would understate the variance. The bootstrap resamples **wallets** (the cluster) with replacement and keeps each wallet's full path.
#
# Implementation: each wallet's transitions are compressed into a 7x7 count vector $C_w$. A replicate draws multinomial weights $w \sim \text{Multinomial}(n, 1/n)$ over the $n$ wallets, which is equivalent to sampling `user_id` with replacement, and its count matrix is $\sum_w w_w C_w$. All $B = 500$ replicates are obtained with a single matrix product. In every replicate the `S1` and `S7` rows are the identity by definition.
#
# For each origin state the table reports the point estimate and the 2.5% / 97.5% percentiles of the one-step probability $P_{i,S7}$ of a liquidation event, plus the same interval for the 30-day PD $[\mathbf{P}^{30}]_{i,S7}$. Replicates in which an origin state is never visited are excluded from that row's daily percentiles (`n_valid_draws`); for the 30-day PD such a row follows the Section 16 convention (an identity row).

# %%
N_BOOTSTRAP = 500
BOOTSTRAP_SEED = 20240315
CI_PERCENTILES = (2.5, 97.5)
HORIZON_DAYS = 30


def wallet_count_matrix(subset: pd.DataFrame) -> np.ndarray:
    """Return per-wallet transition counts with shape (n_wallets, n_states**2)."""
    wallet_codes, _ = pd.factorize(subset["user_id"])
    origin = subset["state"].map(STATE_INDEX)
    target = subset["next_state"].map(STATE_INDEX)
    if origin.isna().any() or target.isna().any():
        raise ValueError("Transitions contain states outside STATE_ORDER.")

    n_cells = N_STATES * N_STATES
    n_wallets = int(wallet_codes.max()) + 1
    flat_index = (
        wallet_codes.astype(np.int64) * n_cells
        + origin.to_numpy(dtype=np.int64) * N_STATES
        + target.to_numpy(dtype=np.int64)
    )
    counts = np.bincount(flat_index, minlength=n_wallets * n_cells)
    return counts.reshape(n_wallets, n_cells).astype(np.float64)


def wallet_bootstrap_s7_ci(
    subset: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """Wallet-level (cluster) bootstrap of the transition probabilities into S7.

    user_ids are resampled with replacement through multinomial weights, so
    every replicate keeps complete wallet paths and their serial dependence.
    Returns point estimates and percentile intervals for the one-step
    probability P[i, S7] and the horizon PD [P^horizon_days][i, S7]. An empty
    subset returns a table of NaN.
    """
    if subset.empty:
        empty_columns = [
            "row_depth_N", "n_valid_draws", "pd_daily_point", "pd_daily_ci_low", "pd_daily_ci_high",
            "pd_daily_boot_se", f"pd_{horizon_days}d_point", f"pd_{horizon_days}d_ci_low", f"pd_{horizon_days}d_ci_high",
        ]  # fmt: skip
        return pd.DataFrame(np.nan, index=pd.Index(STATE_ORDER, name="current_state"), columns=empty_columns)

    wallet_counts = wallet_count_matrix(subset)
    n_wallets = wallet_counts.shape[0]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_wallets, np.full(n_wallets, 1.0 / n_wallets), size=n_bootstrap).astype(np.float64)
    boot_counts = (weights @ wallet_counts).reshape(n_bootstrap, N_STATES, N_STATES)
    point_counts = wallet_counts.sum(axis=0).reshape(N_STATES, N_STATES)

    point_daily = set_absorbing_rows(row_normalize(point_counts, empty_value=np.nan))
    boot_daily = set_absorbing_rows(row_normalize(boot_counts, empty_value=np.nan))
    point_horizon = np.linalg.matrix_power(fill_empty_rows(point_daily), horizon_days)
    boot_horizon = np.linalg.matrix_power(fill_empty_rows(boot_daily), horizon_days)

    boot_pd_daily = boot_daily[:, :, S7_STATE_IDX]
    boot_pd_horizon = boot_horizon[:, :, S7_STATE_IDX]
    low_q, high_q = CI_PERCENTILES
    with warnings.catch_warnings():
        # Origin states never visited in any replicate produce all-NaN slices.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        daily_low, daily_high = np.nanpercentile(boot_pd_daily, [low_q, high_q], axis=0)
        daily_se = np.nanstd(boot_pd_daily, axis=0, ddof=1)
    horizon_low, horizon_high = np.percentile(boot_pd_horizon, [low_q, high_q], axis=0)

    return pd.DataFrame(
        {
            "row_depth_N": point_counts.sum(axis=1).astype(int),
            "n_valid_draws": np.isfinite(boot_pd_daily).sum(axis=0),
            "pd_daily_point": point_daily[:, S7_STATE_IDX],
            "pd_daily_ci_low": daily_low,
            "pd_daily_ci_high": daily_high,
            "pd_daily_boot_se": daily_se,
            f"pd_{horizon_days}d_point": point_horizon[:, S7_STATE_IDX],
            f"pd_{horizon_days}d_ci_low": horizon_low,
            f"pd_{horizon_days}d_ci_high": horizon_high,
        },
        index=pd.Index(STATE_ORDER, name="current_state"),
    )


bootstrap_s7_ci: dict[tuple[str, str], pd.DataFrame] = {}

for scenario_key, scenario_label in SCENARIO_KEYS:
    for profile in PROFILE_KEYS:
        subset = subset_transitions(markov_transitions_df, scenario_key, profile)
        ci_table = wallet_bootstrap_s7_ci(subset)
        bootstrap_s7_ci[(scenario_label, profile)] = ci_table

        if not subset.empty:
            reference_pd = empirical_P[(scenario_label, profile)]["S7"].to_numpy()
            point_pd = ci_table["pd_daily_point"].to_numpy()
            observed_rows = ~np.isnan(point_pd)
            if not np.allclose(point_pd[observed_rows], reference_pd[observed_rows]):
                raise AssertionError(f"Bootstrap point estimates do not match Section 16 for {scenario_label} | {profile}.")

        print(
            f"\n{scenario_label} | {PROFILE_TITLES[profile]}  "
            f"(wallets with a transition={subset['user_id'].nunique():,}, B={N_BOOTSTRAP}, "
            f"CI={CI_PERCENTILES[0]}%-{CI_PERCENTILES[1]}%)"
        )
        display(ci_table.round(6))

# %% [markdown]
# ### 17.3 Time-homogeneity check
#
# The $\mathbf{P}^{30}$ projection assumes the daily matrix is constant within the 30-day horizon. Transitions are split by the date of day $t$ (the origin of the transition): days 1-15 of the calibration window versus days 16-30. $\mathbf{P}$ is estimated on each half and shown side by side, together with the difference (days 16-30 minus days 1-15).
#
# For each transient origin state $i$ (`S2` to `S6`) the Anderson-Goodman likelihood-ratio statistic compares the two halves:
#
# $$G^2_i = 2 \sum_{h=1}^{2} \sum_{j} n_{hij} \ln \frac{n_{hij}}{\hat{E}_{hij}}, \qquad \hat{E}_{hij} = \frac{n_{hi\cdot}\, n_{\cdot ij}}{n_{\cdot i \cdot}}$$
#
# with $K_i - 1$ degrees of freedom, where $K_i$ is the number of next states observed from $i$. A pooled statistic sums $G^2_i$ and the degrees of freedom across the testable transient rows.
#
# `S1` and `S7` are excluded from the test: they are absorbing by definition and their rows are the identity in both halves. A transient row observed in only one half is not testable and its row difference is left empty. Within-wallet dependence makes the chi-square reference approximate (p-values tend to be too small on large samples), so the test is read together with the size of the differences (`max_abs_row_diff`).

# %%
HALF_WINDOW_DAYS = 15
HOMOGENEITY_ALPHA = 0.05
FIRST_HALF, SECOND_HALF = "days_01_15", "days_16_30"
scenario_window_start = dict(DAY1_DATE)


def label_window_half(transitions: pd.DataFrame) -> pd.Series:
    """Label each transition by the half of the calibration window that contains day t."""
    split_date = transitions["scenario"].map(scenario_window_start) + pd.to_timedelta(HALF_WINDOW_DAYS, unit="D")
    is_first_half = transitions["date"].lt(split_date)
    return pd.Series(
        np.where(is_first_half, FIRST_HALF, SECOND_HALF),
        index=transitions.index,
        name="window_half",
    )


def homogeneity_lr_test(
    counts_first: pd.DataFrame,
    counts_second: pd.DataFrame,
    p_first: pd.DataFrame,
    p_second: pd.DataFrame,
    origin_states: list[str] = TRANSIENT_STATES,
) -> pd.DataFrame:
    """Anderson-Goodman likelihood-ratio test of equal transition rows across two periods."""
    records = []
    for origin in origin_states:
        table = np.vstack(
            [
                counts_first.loc[origin].to_numpy(dtype=float),
                counts_second.loc[origin].to_numpy(dtype=float),
            ]
        )
        table = table[:, table.sum(axis=0) > 0]
        n_first, n_second = table.sum(axis=1) if table.shape[1] else (0.0, 0.0)
        dof = table.shape[1] - 1
        if dof < 1 or n_first == 0 or n_second == 0:
            g_stat, p_value, dof = np.nan, np.nan, 0
        else:
            expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
            ratio = np.where(table > 0, table / expected, 1.0)
            g_stat = float(2.0 * np.sum(table * np.log(ratio)))
            p_value = chi2_sf(g_stat, dof)
        both_halves = n_first > 0 and n_second > 0
        records.append(
            {
                "current_state": origin,
                f"n_{FIRST_HALF}": int(n_first),
                f"n_{SECOND_HALF}": int(n_second),
                f"pd_daily_{FIRST_HALF}": p_first.loc[origin, "S7"] if n_first > 0 else np.nan,
                f"pd_daily_{SECOND_HALF}": p_second.loc[origin, "S7"] if n_second > 0 else np.nan,
                "max_abs_row_diff": float((p_second.loc[origin] - p_first.loc[origin]).abs().max()) if both_halves else np.nan,
                "g_statistic": g_stat,
                "dof": dof,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(records).set_index("current_state")

    tested = result["dof"] > 0
    pooled_g = float(result.loc[tested, "g_statistic"].sum())
    pooled_dof = int(result.loc[tested, "dof"].sum())
    pooled_row = pd.DataFrame(
        {
            f"n_{FIRST_HALF}": [int(result[f"n_{FIRST_HALF}"].sum())],
            f"n_{SECOND_HALF}": [int(result[f"n_{SECOND_HALF}"].sum())],
            f"pd_daily_{FIRST_HALF}": [np.nan],
            f"pd_daily_{SECOND_HALF}": [np.nan],
            "max_abs_row_diff": [float(result["max_abs_row_diff"].max())],
            "g_statistic": [pooled_g if pooled_dof > 0 else np.nan],
            "dof": [pooled_dof],
            "p_value": [chi2_sf(pooled_g, pooled_dof) if pooled_dof > 0 else np.nan],
        },
        index=pd.Index(["pooled_transient"], name="current_state"),
    )
    result = pd.concat([result, pooled_row])
    result["homogeneous_at_5pct"] = np.where(
        result["p_value"].isna(),
        "not testable",
        np.where(result["p_value"] >= HOMOGENEITY_ALPHA, "yes", "no"),
    )
    return result


transitions_by_half = markov_transitions_df.assign(window_half=label_window_half(markov_transitions_df))
homogeneity_P: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
homogeneity_tests: dict[tuple[str, str], pd.DataFrame] = {}

for scenario_key, scenario_label in SCENARIO_KEYS:
    for profile in PROFILE_KEYS:
        subset = subset_transitions(transitions_by_half, scenario_key, profile)
        n_first, p_first = build_empirical_transition_matrices(subset.loc[subset["window_half"].eq(FIRST_HALF)])
        n_second, p_second = build_empirical_transition_matrices(subset.loc[subset["window_half"].eq(SECOND_HALF)])
        p_diff = p_second - p_first
        homogeneity_P[(scenario_label, profile)] = {
            FIRST_HALF: p_first,
            SECOND_HALF: p_second,
            "difference": p_diff,
        }
        test_table = homogeneity_lr_test(n_first, n_second, p_first, p_second)
        homogeneity_tests[(scenario_label, profile)] = test_table

        diff_limit = max(float(p_diff.abs().to_numpy().max()), 0.01)
        panels = [
            (p_first, f"Days 1-15  (N={int(n_first.to_numpy().sum()):,})", "Blues", 0.0, 1.0),
            (p_second, f"Days 16-30  (N={int(n_second.to_numpy().sum()):,})", "Blues", 0.0, 1.0),
            (p_diff, "Difference (days 16-30 minus days 1-15)", "RdBu_r", -diff_limit, diff_limit),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        for ax, (matrix, subtitle, cmap, vmin, vmax) in zip(axes, panels):
            sns.heatmap(matrix, ax=ax, annot=True, fmt=".4f", cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0.4)
            ax.set_title(subtitle)
            ax.set_xlabel("Next state")
            ax.set_ylabel("Current state")
        fig.suptitle(
            f"Time-homogeneity check  |  {scenario_label}  |  {PROFILE_TITLES[profile]}",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()

        print(f"{scenario_label} | {PROFILE_TITLES[profile]}: likelihood-ratio test by origin state")
        display(test_table.round(6))

# %% [markdown]
# ### 17.4 Threshold sensitivity analysis
#
# `compute_transition_matrix(panel, bins, labels)` rebuilds the chain from the observed pure-wallet Health Factor panel (Section 12) for any set of cut points:
#
# 1. Map HF to states with left-closed bins `[a, b)` and force `S7` for `HF <= 1.0`. The liquidation threshold is a protocol rule, not a modelling choice, so it is never shifted.
# 2. Re-apply the rules of Section 14 end to end through `run_chain`: day-level `S7` (HF or `liquidationCall`), observed `S1`, first absorption, censoring and no transitions across gaps.
# 3. Return $\mathbf{P}$ for each regime and profile.
#
# The cut points that bound the intermediate states `S4`, `S5`, `S6` (1.10, 1.25, 1.50) are shifted by $-0.05$ and $+0.05$; the liquidation boundary 1.0 and the `S3`/`S2` boundary 2.0 stay fixed. Because a shifted label covers a different HF band, the summary also reports the 30-day PD of the **performing portfolio**: the distribution of each wallet's first observed state over `S2` to `S6`, multiplied by $[\mathbf{P}^{30}]_{\cdot,S7}$. Figures that move little across the three runs indicate that the PD is not an artefact of the chosen boundaries.
#
# A sanity check first confirms that the base cut points reproduce the Section 16 matrices.

# %%
SENSITIVITY_SHIFT = 0.05
INTERMEDIATE_BOUNDARIES = (1.10, 1.25, 1.50)


def map_hf_to_states(health_factor: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    """Discretize HF with left-closed bins [a, b); HF <= 1.0 is always S7."""
    edges = np.asarray(bins, dtype=float)
    if len(labels) != len(edges) - 1:
        raise ValueError(f"Expected {len(edges) - 1} labels for {len(edges)} bin edges, got {len(labels)}.")
    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"Bin edges must be strictly increasing: {list(bins)}")
    unknown_labels = set(labels) - set(STATE_ORDER)
    if unknown_labels:
        raise ValueError(f"Labels outside STATE_ORDER: {sorted(unknown_labels)}")

    states = pd.cut(health_factor, bins=edges, labels=labels, right=False).astype("object")
    states = states.mask(health_factor.le(LIQUIDATION_HF_THRESHOLD), "S7")
    n_unmapped = int(states.isna().sum())
    if n_unmapped:
        raise ValueError(f"{n_unmapped:,} Health Factor values fall outside the bin edges.")
    return states.rename("state_hf")


def compute_transition_matrix(
    panel: pd.DataFrame, bins: list[float], labels: list[str]
) -> dict[tuple[str, str], pd.DataFrame]:
    """Recompute the empirical P for custom Health Factor boundaries (Section 14 rules, calibration window).

    Returns one row-stochastic matrix per (scenario_label, profile).
    """
    panel = panel.assign(state_hf=map_hf_to_states(panel["health_factor"], bins, labels))
    transitions = run_chain(panel)["transitions"]

    matrices: dict[tuple[str, str], pd.DataFrame] = {}
    for scenario_key, scenario_label in SCENARIO_KEYS:
        for profile in PROFILE_KEYS:
            _, transition_p = build_empirical_transition_matrices(subset_transitions(transitions, scenario_key, profile))
            matrices[(scenario_label, profile)] = transition_p
    return matrices


def shift_intermediate_boundaries(
    bins: list[float], shift: float, boundaries: tuple[float, ...] = INTERMEDIATE_BOUNDARIES
) -> list[float]:
    """Shift only the selected cut points; every other edge is left unchanged."""
    return [edge + shift if any(np.isclose(edge, target) for target in boundaries) else edge for edge in bins]


def horizon_pd(transition_p: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> pd.Series:
    """Cumulative probability of reaching S7 within horizon_days, by origin state."""
    power = np.linalg.matrix_power(transition_p.to_numpy(), horizon_days)
    return pd.Series(power[:, S7_STATE_IDX], index=transition_p.index)


def performing_portfolio_pd(
    panel: pd.DataFrame,
    bins: list[float],
    labels: list[str],
    transition_p: pd.DataFrame,
    scenario_key: str,
    profile: str,
    horizon_days: int = HORIZON_DAYS,
) -> float:
    """30-day PD of wallets whose first observed state is transient (S2 to S6)."""
    segment = panel.loc[(panel["scenario"] == scenario_key) & (panel["profile"] == profile)]
    first_obs = segment.sort_values("day", kind="mergesort").drop_duplicates("wid", keep="first")
    if first_obs.empty:
        return np.nan
    initial_states = map_hf_to_states(first_obs["health_factor"], bins, labels)
    weights = initial_states.value_counts().reindex(TRANSIENT_STATES, fill_value=0).astype(float)
    if weights.sum() == 0:
        return np.nan
    weights = weights / weights.sum()
    return float((weights * horizon_pd(transition_p, horizon_days).loc[TRANSIENT_STATES]).sum())


pure_hf_panel = pure_panel.loc[pure_panel["day"] <= T_IDX].reset_index(drop=True)

sensitivity_bins = {
    f"-{SENSITIVITY_SHIFT:.2f}": shift_intermediate_boundaries(HF_STATE_BINS, -SENSITIVITY_SHIFT),
    "base": list(HF_STATE_BINS),
    f"+{SENSITIVITY_SHIFT:.2f}": shift_intermediate_boundaries(HF_STATE_BINS, SENSITIVITY_SHIFT),
}
print("Health Factor bin edges by run (labels: " + ", ".join(HF_STATE_LABELS) + "):")
for run_name, edges in sensitivity_bins.items():
    print(f"  {run_name:>6}: {[round(edge, 4) for edge in edges]}")

sensitivity_P = {
    run_name: compute_transition_matrix(pure_hf_panel, edges, HF_STATE_LABELS)
    for run_name, edges in sensitivity_bins.items()
}

print("\nSanity check: base bins versus Section 16 matrices")
for key, matrix in sensitivity_P["base"].items():
    max_gap = float((matrix - empirical_P[key]).abs().to_numpy().max())
    print(f"  {key[0]} | Profile {key[1]}: max |P_recomputed - P_section16| = {max_gap:.2e}")
    assert max_gap < 1e-12, "Base-bin chain does not reproduce Section 16."

portfolio_records = []
for scenario_key, scenario_label in SCENARIO_KEYS:
    for profile in PROFILE_KEYS:
        key = (scenario_label, profile)
        daily_pd = pd.DataFrame({run: sensitivity_P[run][key]["S7"] for run in sensitivity_bins})
        cumulative_pd = pd.DataFrame({run: horizon_pd(sensitivity_P[run][key]) for run in sensitivity_bins})
        sensitivity_table = pd.concat({"pd_daily": daily_pd, f"pd_{HORIZON_DAYS}d": cumulative_pd}, axis=1)
        sensitivity_table[("max_abs_delta_vs_base", "pd_daily")] = daily_pd.sub(daily_pd["base"], axis=0).abs().max(axis=1)
        sensitivity_table[("max_abs_delta_vs_base", f"pd_{HORIZON_DAYS}d")] = cumulative_pd.sub(cumulative_pd["base"], axis=0).abs().max(axis=1)
        sensitivity_table.index.name = "current_state"

        print(f"\n{scenario_label} | {PROFILE_TITLES[profile]}: S7 column under shifted boundaries")
        display(sensitivity_table.round(6))

        record = {"scenario": scenario_label, "profile": profile}
        for run_name, edges in sensitivity_bins.items():
            record[run_name] = performing_portfolio_pd(
                pure_hf_panel, edges, HF_STATE_LABELS, sensitivity_P[run_name][key], scenario_key, profile
            )
        portfolio_records.append(record)

portfolio_sensitivity = pd.DataFrame(portfolio_records).set_index(["scenario", "profile"])
run_columns = list(sensitivity_bins)
portfolio_sensitivity["max_relative_change_vs_base"] = (
    portfolio_sensitivity[run_columns]
    .sub(portfolio_sensitivity["base"], axis=0)
    .abs()
    .max(axis=1)
    .div(portfolio_sensitivity["base"].replace(0, np.nan))
)
print(f"\nPerforming-portfolio {HORIZON_DAYS}-day PD by boundary run")
display(portfolio_sensitivity.round(6))

# %% [markdown]
# ### 17.5 Minimum-debt sensitivity
#
# The base panel keeps user-days with at least USD 100 of debt (`MIN_DEBT_USD`). This check re-runs the **same functions** (`qualifying_panel`, `drop_hybrid_wallets`, `run_chain`) with a USD 1,000 minimum, so purity, `S1` detection and censoring are recomputed under the stricter filter, and compares the transition counts $N$ and the 30-day PD by origin state. It is a compact comparison, not a second full pipeline. A higher minimum removes the small positions that liquidators ignore; if the PD of the larger positions differs markedly from the base, the choice of minimum debt matters for the thesis.

# %%
MIN_DEBT_SENSITIVITY_USD = 1000.0

panel_sensitivity, _ = drop_hybrid_wallets(qualifying_panel(user_day, MIN_DEBT_SENSITIVITY_USD))
panel_sensitivity = panel_sensitivity.loc[panel_sensitivity["day"] <= T_IDX].reset_index(drop=True)
panel_sensitivity["state_hf"] = map_health_factor_to_state(panel_sensitivity["health_factor"])
chain_sensitivity = run_chain(panel_sensitivity)

min_debt_records = []
for scenario_key, scenario_label in SCENARIO_KEYS:
    for profile in PROFILE_KEYS:
        n_base, p_base = empirical_N[(scenario_label, profile)], empirical_P[(scenario_label, profile)]
        n_alt, p_alt = build_empirical_transition_matrices(
            subset_transitions(chain_sensitivity["transitions"], scenario_key, profile)
        )
        pd_base, pd_alt = horizon_pd(p_base), horizon_pd(p_alt)
        for state in TRANSIENT_STATES:
            min_debt_records.append(
                {
                    "scenario": scenario_label,
                    "profile": profile,
                    "current_state": state,
                    f"N_min{MIN_DEBT_USD:.0f}": int(n_base.loc[state].sum()),
                    f"N_min{MIN_DEBT_SENSITIVITY_USD:.0f}": int(n_alt.loc[state].sum()),
                    f"pd30_min{MIN_DEBT_USD:.0f}": pd_base[state] if n_base.loc[state].sum() > 0 else np.nan,
                    f"pd30_min{MIN_DEBT_SENSITIVITY_USD:.0f}": pd_alt[state] if n_alt.loc[state].sum() > 0 else np.nan,
                }
            )
min_debt_sensitivity = pd.DataFrame(min_debt_records).set_index(["scenario", "profile", "current_state"])
print(f"Transition counts N and 30-day PD by origin state: minimum debt USD {MIN_DEBT_USD:,.0f} versus USD {MIN_DEBT_SENSITIVITY_USD:,.0f}")
display(min_debt_sensitivity.round(6))
print("Total N by segment:")
display(min_debt_sensitivity.groupby(["scenario", "profile"])[[f"N_min{MIN_DEBT_USD:.0f}", f"N_min{MIN_DEBT_SENSITIVITY_USD:.0f}"]].sum())

# %% [markdown]
# ## 18. Export of segment parameters and wallet cohorts (not for production)
#
# One export step writes two files to `data/02_interim/` (to `data/sample/02_interim/` in a sample run). The previous `notebook_export_*` files are left untouched.
#
# **1. `markov_segment_params_not_for_prod.csv`.** One row per `(scenario, profile, current_state)`: the transition counts `n_S1..n_S7` (Section 17.1), the full row of $\mathbf{P}^{30}$ (`p30_S1..p30_S7`, from the Section 16 matrices), the 30-day PD `pd30_point` with its wallet-bootstrap interval `pd30_ci_low` / `pd30_ci_high` (Section 17.2), `min_debt_usd`, `row_depth` (the number of transitions behind the row) and `row_empty`. Rows of `S1` and `S7` are identity by definition. A transient state without observed transitions has no estimate: `row_empty` is `True` and its `p30_*` and `pd30_*` are left empty (the cell prints a warning that lists them).
#
# **2. `markov_wallet_cohorts_not_for_prod.csv`.** One row per wallet and cohort (`cohort`):
#
# - `day1`: the wallets observed on $t-30$ (no late entrants), with the state, Health Factor, EAD (`total_debt_usd`) and collateral of that day, and the calibration outcome at $t$: `final_state`, `event_date`, `exit_reason`. This is the cohort of the in-sample backtest.
# - `valuation`: the wallets observed on $t$, with their state **as observed that day** (no forward fill). `is_late_entrant` marks wallets whose spell did not start on $t-30$ (they entered later, or re-entered after a repayment). For the out-of-time backtest each row also carries `oot_outcome` (`S7`, `S1`, `transient` or `censored`) and `oot_event_date`, from the days $(t, t+30]$ with the rules of Section 14: the first absorption wins; a wallet still in the filter on $t+30$ is `transient`; one that left the filter and never came back is `censored`; a wallet already in `S7` on $t$ keeps `S7` with `oot_event_date = t`.
#
# Collateral is split by asset group (`eth_core_usd` includes the ETH liquid staking tokens, `wbtc_usd`, `uni_usd`, `link_usd`, `stable_usd`) and summarised by the collateral-weighted `lt_w` and `lb_w`. For every `S7` entry linked to a row, `s7_trigger` gives its trigger (`hf` or `liquidation_call`), `hf_at_s7_entry` the Health Factor of the entry day, and `days_s7_to_liquidation` the days from the first snapshot with $HF \le 1$ to the first `liquidationCall`. If the entry was triggered by a `liquidationCall` and the snapshot $HF$ of that day is above 1 (the liquidation already restored it), `hf_at_s7_entry` takes the $HF$ of the previous day's snapshot and `hf_at_s7_entry_prev_day` is `True`. A `liquidationCall` entry has no snapshot with $HF \le 1$ (crossing and liquidation fall in the same daily interval), so its `days_s7_to_liquidation` is 0; for an `hf` entry it is `NaN` when no liquidation follows within the window. **Use of the trigger downstream:** the previous-day $HF$ of a `liquidation_call` entry is above 1 by construction, so the LGD module should treat those entries with $HF_{entry} = 1$ (the Health Factor at the moment of the liquidation) and keep the observed $HF$ for `hf` entries.
#
# Before writing, the cell checks the tables; after writing, both files are read back to confirm an exact round trip, and $\mathbf{N}$, $\mathbf{P}^{30}$, the intervals and the `exit_reason` counts are recomputed from the files and compared with the values in memory.

# %%
SEGMENT_EXPORT_PATH = INTERIM_DATA_DIR / "markov_segment_params_not_for_prod.csv"
COHORT_EXPORT_PATH = INTERIM_DATA_DIR / "markov_wallet_cohorts_not_for_prod.csv"
STOCHASTIC_TOLERANCE = 1e-9
ROUND_TRIP_TOLERANCE = 1e-12
DATE_FORMAT = "%Y-%m-%d"

SEGMENT_COLUMNS = (
    ["scenario", "profile", "current_state"]
    + [f"n_{state}" for state in STATE_ORDER]
    + [f"p30_{state}" for state in STATE_ORDER]
    + ["pd30_point", "pd30_ci_low", "pd30_ci_high", "min_debt_usd", "row_depth", "row_empty"]
)
COHORT_COLUMNS = [
    "scenario", "profile", "user_id", "cohort", "date", "state", "health_factor", "total_debt_usd",
    "total_collateral_usd", "eth_core_usd", "wbtc_usd", "uni_usd", "link_usd", "stable_usd", "lt_w", "lb_w",
    "is_late_entrant", "final_state", "event_date", "exit_reason", "s7_trigger", "hf_at_s7_entry",
    "hf_at_s7_entry_prev_day", "days_s7_to_liquidation", "oot_outcome", "oot_event_date",
]  # fmt: skip


def compute_horizon_matrix(transition_p: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> pd.DataFrame:
    """Return the full horizon_days-step transition matrix P^horizon_days."""
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be a positive integer, got {horizon_days}.")
    if list(transition_p.index) != STATE_ORDER or list(transition_p.columns) != STATE_ORDER:
        raise ValueError("Daily transition matrix rows and columns must follow STATE_ORDER.")
    daily = transition_p.to_numpy(dtype=float)
    if not np.isfinite(daily).all():
        raise ValueError("Daily transition matrix contains non-finite values.")
    return pd.DataFrame(
        np.linalg.matrix_power(daily, horizon_days),
        index=pd.Index(STATE_ORDER, name="current_state"),
        columns=STATE_ORDER,
    )


def validate_horizon_matrix(
    horizon_p: pd.DataFrame, daily_p: pd.DataFrame, segment: str, horizon_days: int = HORIZON_DAYS, tolerance: float = STOCHASTIC_TOLERANCE
) -> None:
    """Raise if P^horizon_days is not a valid stochastic matrix consistent with Section 17.4."""
    values = horizon_p.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{segment}: P^{horizon_days} contains non-finite values.")
    if (values < -tolerance).any():
        raise ValueError(f"{segment}: P^{horizon_days} contains negative probabilities.")
    if not np.allclose(values.sum(axis=1), 1.0, atol=tolerance):
        raise ValueError(f"{segment}: P^{horizon_days} rows do not sum to 1.")
    reference_pd = horizon_pd(daily_p, horizon_days).to_numpy(dtype=float)
    if not np.allclose(values[:, S7_STATE_IDX], reference_pd, atol=tolerance):
        raise AssertionError(f"{segment}: S7 column does not match horizon_pd from Section 17.4.")


def build_segment_params_table() -> pd.DataFrame:
    """Stack N, P^30 and the bootstrap interval of every regime, profile and origin state."""
    rows = []
    for scenario_key, scenario_label in SCENARIO_KEYS:
        for profile in PROFILE_KEYS:
            counts_n = absolute_N[(scenario_label, profile)]
            daily_p = empirical_P[(scenario_label, profile)]
            horizon_p = compute_horizon_matrix(daily_p)
            validate_horizon_matrix(horizon_p, daily_p, f"{scenario_label} | {profile}")
            ci_table = bootstrap_s7_ci[(scenario_label, profile)]
            for state in STATE_ORDER:
                row_depth = int(counts_n.loc[state].sum())
                row_empty = state in TRANSIENT_STATES and row_depth == 0  # identity row in P, not estimated
                estimated = not row_empty
                row = {"scenario": scenario_label, "profile": profile, "current_state": state}
                row.update({f"n_{target}": int(counts_n.loc[state, target]) for target in STATE_ORDER})
                row.update({f"p30_{target}": horizon_p.loc[state, target] if estimated else np.nan for target in STATE_ORDER})
                row["pd30_point"] = horizon_p.loc[state, "S7"] if estimated else np.nan
                row["pd30_ci_low"] = ci_table.loc[state, f"pd_{HORIZON_DAYS}d_ci_low"] if estimated else np.nan
                row["pd30_ci_high"] = ci_table.loc[state, f"pd_{HORIZON_DAYS}d_ci_high"] if estimated else np.nan
                row["min_debt_usd"] = MIN_DEBT_USD
                row["row_depth"] = row_depth
                row["row_empty"] = row_empty
                rows.append(row)
    return pd.DataFrame(rows)[SEGMENT_COLUMNS]


def s7_entry_details(wid: np.ndarray, landing: np.ndarray, trigger: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Health Factor at S7 entry, its previous-day flag, and days to the first liquidationCall.

    ``landing`` is the day on which the wallet is in S7. If the entry comes from a liquidationCall and the
    snapshot HF of that day is above 1 (or the wallet is absent from it), the HF of the previous day's
    snapshot is used and flagged. An S7_hf entry counts the days from its snapshot to the first
    liquidationCall on or after it (NaN if none); an S7_liquidation_call entry is 0 days by construction.
    """
    entry_key = wid * DAY_STRIDE + landing
    hf_landing, _ = lookup_snapshot(UD_HF, entry_key)
    hf_previous, _ = lookup_snapshot(UD_HF, entry_key - 1)
    use_landing = (trigger == "S7_hf") | (np.isfinite(hf_landing) & (hf_landing <= LIQUIDATION_HF_THRESHOLD))
    hf_entry = np.where(use_landing, hf_landing, hf_previous)

    days_to_liquidation = np.full(len(wid), np.nan)
    if len(LIQ_EVENT_KEYS):
        position = np.searchsorted(LIQ_EVENT_KEYS, entry_key, side="left")
        candidate = LIQ_EVENT_KEYS[np.minimum(position, len(LIQ_EVENT_KEYS) - 1)]
        found = (position < len(LIQ_EVENT_KEYS)) & (candidate // DAY_STRIDE == wid)
        days_to_liquidation = np.where(found, candidate % DAY_STRIDE - landing, np.nan)
    days_to_liquidation = np.where(trigger == "S7_liquidation_call", 0.0, days_to_liquidation)
    return hf_entry, ~use_landing, days_to_liquidation


def oot_outcomes(trajectories: dict[str, pd.DataFrame], valuation: pd.DataFrame) -> pd.DataFrame:
    """Out-of-time outcome of the valuation wallets over the days (t, t+30]; same rules as the calibration."""
    obs = trajectories["obs"]
    first_hit = first_absorption(trajectories["events"], T_IDX + 1, LAST_IDX).reindex(valuation["sid"])
    seen_at_end = set(obs.loc[obs["day"] == LAST_IDX, "sid"])
    after_t = obs.loc[obs["day"] > T_IDX].groupby("sid")["day"].max()
    last_day_after_t = after_t.reindex(valuation["sid"]).fillna(T_IDX)
    already_s7 = (valuation["state"] == "S7").to_numpy()
    hit = first_hit["type"].notna().to_numpy()
    at_end = valuation["sid"].isin(seen_at_end).to_numpy()
    hit_type = first_hit["type"].to_numpy(dtype=object)

    outcome = np.select(
        [already_s7, hit & np.isin(hit_type, ["S7_hf", "S7_liquidation_call"]), hit, at_end],
        ["S7", "S7", "S1", "transient"],
        default="censored",
    )
    event_day = np.select(
        [already_s7, hit, at_end],
        [np.full(len(valuation), T_IDX), first_hit["landing"].fillna(T_IDX).to_numpy(), np.full(len(valuation), LAST_IDX)],
        default=last_day_after_t.to_numpy(),
    ).astype("int64")
    trigger = np.where(already_s7, valuation["s7_trigger"].to_numpy(dtype=object), np.where(hit, hit_type, ""))
    trigger = np.where(outcome == "S7", trigger, "")
    return pd.DataFrame(
        {"oot_outcome": outcome, "oot_day": event_day, "oot_trigger": trigger}, index=valuation.index
    )


def to_date_string(scenario: pd.Series, day: pd.Series) -> pd.Series:
    """Calendar date (YYYY-MM-DD) of a day index; NaN stays empty."""
    dates = scenario.map(DAY1_DATE) + pd.to_timedelta(day, unit="D")
    return dates.dt.strftime(DATE_FORMAT)


def build_wallet_cohorts_table(chain_result: dict) -> pd.DataFrame:
    """One row per wallet and cohort (day1, valuation) with state, exposure, collateral and outcomes."""
    obs = chain_result["traj"]["obs"]
    summary = chain_result["summary"]
    base_columns = [
        "scenario", "profile", "user_id", "wid", "sid", "day", "state", "s7_trigger", "health_factor", "total_debt_usd",
        "total_collateral_usd", "eth_core_usd", "wbtc_usd", "uni_usd", "link_usd", "stable_usd", "lt_w", "lb_w",
    ]  # fmt: skip
    late = summary["is_late_entrant"]

    day1 = obs.loc[obs["day"] == 0, base_columns].copy()
    day1["cohort"] = "day1"
    day1 = day1.join(summary[["final_state", "event_day", "exit_reason", "landing", "type"]], on="sid")
    day1["is_late_entrant"] = False
    day1["s7_landing"] = np.where(day1["final_state"] == "S7", day1["landing"], np.nan)
    day1["s7_entry_trigger"] = np.where(day1["final_state"] == "S7", day1["type"], "")

    valuation = obs.loc[obs["day"] == T_IDX, base_columns].copy()
    valuation["cohort"] = "valuation"
    valuation["is_late_entrant"] = late.reindex(valuation["sid"]).to_numpy()
    prior = summary.reindex(valuation["sid"])
    n_prior_s7 = int((prior["type"].isin(["S7_hf", "S7_liquidation_call"]) & (prior["landing"] < T_IDX)).sum())
    print(
        f"Valuation wallets observed on t that had already entered S7 earlier in the window (re-observed, state as observed): {n_prior_s7:,}; "
        f"valuation rows that belong to a spell opened after a repayment: {int((prior['spell'] > 0).sum()):,}"
    )
    oot = oot_outcomes(chain_result["traj"], valuation)
    valuation = valuation.join(oot)
    valuation["s7_landing"] = np.where(valuation["oot_outcome"] == "S7", valuation["oot_day"], np.nan)
    valuation["s7_entry_trigger"] = valuation["oot_trigger"]

    cohorts = pd.concat([day1, valuation], ignore_index=True)
    has_s7 = cohorts["s7_landing"].notna().to_numpy()
    cohorts["hf_at_s7_entry"] = np.nan
    cohorts["hf_at_s7_entry_prev_day"] = False
    cohorts["days_s7_to_liquidation"] = np.nan
    if has_s7.any():
        wid = cohorts.loc[has_s7, "wid"].to_numpy().astype("int64")
        landing = cohorts.loc[has_s7, "s7_landing"].to_numpy().astype("int64")
        trigger = cohorts.loc[has_s7, "s7_entry_trigger"].to_numpy(dtype=object)
        hf_entry, previous_day, days = s7_entry_details(wid, landing, trigger)
        cohorts.loc[has_s7, "hf_at_s7_entry"] = hf_entry
        cohorts.loc[has_s7, "hf_at_s7_entry_prev_day"] = previous_day
        cohorts.loc[has_s7, "days_s7_to_liquidation"] = days
    # Exported trigger: "hf" (HF <= 1 in the snapshot) or "liquidation_call" (only the event shows the S7); a
    # liquidation_call entry carries the previous-day HF in hf_at_s7_entry, so LGD can use HF_entry = 1 for it.
    cohorts["s7_trigger"] = np.where(has_s7, cohorts["s7_entry_trigger"], "")
    cohorts["s7_trigger"] = cohorts["s7_trigger"].replace({"S7_hf": "hf", "S7_liquidation_call": "liquidation_call", "": np.nan})

    cohorts["date"] = to_date_string(cohorts["scenario"], cohorts["day"])
    cohorts["event_date"] = to_date_string(cohorts["scenario"], cohorts["event_day"])
    cohorts["oot_event_date"] = to_date_string(cohorts["scenario"], cohorts["oot_day"])
    cohorts["scenario"] = cohorts["scenario"].map(SCENARIO_LABEL)
    cohorts.loc[cohorts["cohort"] == "valuation", ["final_state", "exit_reason"]] = np.nan
    cohorts.loc[cohorts["cohort"] == "day1", ["oot_outcome"]] = np.nan
    return cohorts[COHORT_COLUMNS].sort_values(["scenario", "profile", "cohort", "user_id"], kind="mergesort").reset_index(drop=True)


def validate_cohorts_table(cohorts: pd.DataFrame) -> None:
    """Consistency checks of the cohort table before writing."""
    assert not cohorts.duplicated(["scenario", "user_id", "cohort"]).any(), "A wallet appears twice in a cohort."
    assert cohorts["state"].isin(TRANSIENT_STATES + ["S7"]).all(), "A cohort row has a state outside S2-S7."
    assert (cohorts.loc[cohorts["cohort"] == "day1", "is_late_entrant"] == False).all()  # noqa: E712
    groups = cohorts[["eth_core_usd", "wbtc_usd", "uni_usd", "link_usd", "stable_usd"]].sum(axis=1)
    assert np.allclose(groups, cohorts["total_collateral_usd"], rtol=1e-9, atol=1e-6), "Collateral groups do not add up."
    assert np.isfinite(cohorts[["health_factor", "total_debt_usd", "lt_w", "lb_w"]]).all().all()
    day1 = cohorts.loc[cohorts["cohort"] == "day1"]
    assert day1["exit_reason"].isin(EXIT_REASONS).all() and day1["final_state"].isin(STATE_ORDER).all()
    valuation = cohorts.loc[cohorts["cohort"] == "valuation"]
    assert valuation["oot_outcome"].isin(["S7", "S1", "transient", "censored"]).all()
    s7_rows = cohorts["s7_trigger"].notna()
    assert cohorts.loc[s7_rows, "hf_at_s7_entry"].notna().sum() > 0 or not s7_rows.any()
    assert cohorts.loc[~s7_rows, "hf_at_s7_entry"].isna().all()
    assert cohorts.loc[s7_rows & ~cohorts["hf_at_s7_entry_prev_day"].astype(bool), "hf_at_s7_entry"].le(LIQUIDATION_HF_THRESHOLD).all(), (
        "An S7 entry without the previous-day flag has HF > 1."
    )


def write_csv_exact(table: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Write a table to CSV and return it read back with exact float parsing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table.to_csv(output_path, index=False)
        return pd.read_csv(output_path, float_precision="round_trip")
    except OSError as exc:
        raise OSError(f"Failed to write or reload {output_path.resolve()}: {exc}") from exc


segment_params_df = build_segment_params_table()
empty_rows = segment_params_df.loc[segment_params_df["row_empty"], ["scenario", "profile", "current_state"]]
thin_rows = segment_params_df.loc[
    segment_params_df["current_state"].isin(TRANSIENT_STATES) & ~segment_params_df["row_empty"] & (segment_params_df["row_depth"] < MIN_ROW_DEPTH),
    ["scenario", "profile", "current_state", "row_depth"],
]
if len(empty_rows):
    print("WARNING: transient rows without observed transitions (identity in P, p30_* / pd30_* left empty, row_empty = True):")
    print(empty_rows.to_string(index=False))
if len(thin_rows):
    print(f"Note: transient rows with fewer than {MIN_ROW_DEPTH} transitions (read them with the bootstrap interval):")
    print(thin_rows.to_string(index=False))
cohorts_df = build_wallet_cohorts_table(chain)
validate_cohorts_table(cohorts_df)

segment_reloaded = write_csv_exact(segment_params_df, SEGMENT_EXPORT_PATH)
cohorts_reloaded = write_csv_exact(cohorts_df, COHORT_EXPORT_PATH)

# --- round trip: files versus tables ------------------------------------------------------------------
pd.testing.assert_frame_equal(segment_reloaded, segment_params_df, check_dtype=False, rtol=0, atol=0)
pd.testing.assert_frame_equal(cohorts_reloaded, cohorts_df, check_dtype=False, rtol=0, atol=0)

# --- round trip: N, P^30, interval and exit_reason recomputed from the files ------------------------
max_gap = 0.0
for scenario_key, scenario_label in SCENARIO_KEYS:
    for profile in PROFILE_KEYS:
        block = segment_reloaded.loc[
            (segment_reloaded["scenario"] == scenario_label) & (segment_reloaded["profile"] == profile)
        ].set_index("current_state").loc[STATE_ORDER]
        counts_file = block[[f"n_{state}" for state in STATE_ORDER]].to_numpy(dtype=float)
        assert np.array_equal(counts_file, absolute_N[(scenario_label, profile)].to_numpy(dtype=float)), "N differs after reload."

        p_file = fill_empty_rows(set_absorbing_rows(row_normalize(counts_file, np.nan)))
        assert np.allclose(p_file, empirical_P[(scenario_label, profile)].to_numpy(), atol=ROUND_TRIP_TOLERANCE, rtol=0)
        p30_file = np.linalg.matrix_power(p_file, HORIZON_DAYS)
        p30_exported = block[[f"p30_{state}" for state in STATE_ORDER]].to_numpy(dtype=float)
        observed = ~np.isnan(p30_exported).any(axis=1)
        assert observed[[STATE_INDEX[s] for s in ABSORBING_STATES]].all()
        gap = float(np.abs(p30_exported[observed] - p30_file[observed]).max())
        assert np.allclose(p30_exported[observed].sum(axis=1), 1.0, atol=ROUND_TRIP_TOLERANCE), "A P^30 row does not sum to 1."
        ci = bootstrap_s7_ci[(scenario_label, profile)]
        for column, source in (("pd30_ci_low", f"pd_{HORIZON_DAYS}d_ci_low"), ("pd30_ci_high", f"pd_{HORIZON_DAYS}d_ci_high")):
            gap = max(gap, float(np.abs(block[column].to_numpy()[observed] - ci[source].to_numpy()[observed]).max()))
        gap = max(gap, float(np.abs(block["pd30_point"].to_numpy()[observed] - p30_file[observed, S7_STATE_IDX]).max()))
        max_gap = max(max_gap, gap)
        assert gap <= ROUND_TRIP_TOLERANCE, f"{scenario_label} | {profile}: round-trip gap {gap:.2e}."

exit_memory = (
    wallet_summary.loc[wallet_summary["first_day"] == 0]
    .assign(scenario=lambda frame: frame["scenario"].map(SCENARIO_LABEL))
    .groupby(["scenario", "profile", "exit_reason"])
    .size()
)
exit_file = cohorts_reloaded.loc[cohorts_reloaded["cohort"] == "day1"].groupby(["scenario", "profile", "exit_reason"]).size()
pd.testing.assert_series_equal(exit_memory.sort_index(), exit_file.sort_index(), check_names=False)

print("Exports written (manual notebook export, NOT for production):")
print(f"  segment params: {SEGMENT_EXPORT_PATH.resolve()}  ({len(segment_params_df)} rows)")
print(f"  wallet cohorts: {COHORT_EXPORT_PATH.resolve()}  ({len(cohorts_df):,} rows)")
print(f"  round trip OK: max |gap| in P^30 and intervals = {max_gap:.2e}; exit_reason counts identical")
print("\nSegment parameters (first rows):")
display(segment_params_df.head(14).round(6))
print("Wallet cohorts: rows per scenario, profile and cohort")
display(cohorts_df.groupby(["scenario", "profile", "cohort"]).size().unstack("cohort"))
print("Out-of-time outcome of the valuation cohort")
display(
    cohorts_df.loc[cohorts_df["cohort"] == "valuation"]
    .groupby(["scenario", "profile", "oot_outcome"])
    .size()
    .unstack("oot_outcome", fill_value=0)
)
print("S7 entries by trigger: Health Factor at entry, previous-day flag and days to liquidation")
display(
    cohorts_df.loc[cohorts_df["s7_trigger"].notna()]
    .groupby(["cohort", "s7_trigger"])
    .agg(
        n=("user_id", "size"),
        hf_entry_median=("hf_at_s7_entry", "median"),
        prev_day_flag=("hf_at_s7_entry_prev_day", "sum"),
        days_to_liq_median=("days_s7_to_liquidation", "median"),
        n_without_liquidation=("days_s7_to_liquidation", lambda s: int(s.isna().sum())),
    )
)
