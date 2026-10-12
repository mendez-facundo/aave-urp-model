# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Exploratory Data Analysis: Aave v3 The Graph Extracts for Markov Calibration
#
# This notebook inspects the **raw** Aave v3 payloads from The Graph, flattens nested position histories, values them in USD, applies actuarial purity filters, and reconstructs an empirical **Health Factor (HF)** panel for Markov calibration.
#
# Both valuation windows are processed in parallel:
#
# - **Baseline scenario:** 30-day window ending 15 March 2024 (`data/01_raw/baseline_mar2024/`)
# - **Stress scenario:** 30-day window ending 5 August 2024 (`data/01_raw/stress_aug2024/`)
#
# Each Graph payload contains **`users`** (borrower positions, ~49k records) and **`liquidationCalls`**. Nested reserve histories (`aTokenBalanceHistory`, `vTokenBalanceHistory`) are unnested, expanded to a daily calendar, and crossed with Binance close prices. Protocol constants (`STABLECOINS`, `LST_ASSETS`, `PRICE_PROXIES`, `LIQUIDATION_THRESHOLDS`) are imported from `src.utils.config`.

# %% [markdown]
# ## 1. Imports

# %%
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(".."))
from src.utils.config import (
    LST_ASSETS,
    LIQUIDATION_THRESHOLDS,
    PRICE_PROXIES,
    STABLECOINS,
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

print("STABLECOINS:", STABLECOINS)
print("LST_ASSETS:", LST_ASSETS)
print("PRICE_PROXIES:", PRICE_PROXIES)
print("LIQUIDATION_THRESHOLDS keys:", list(LIQUIDATION_THRESHOLDS.keys()))

# %% [markdown]
# ## 2. Data loading
#
# Both Graph JSON extracts are resolved relative to this notebook (`notebooks/`). Folders are iterated as `['baseline_mar2024', 'stress_aug2024']`. Each file is large (~250 MB); both payloads are kept in memory under `payloads`.

# %%
RAW_DATA_DIR = Path("../data/01_raw")
SCENARIO_DIRS = ["baseline_mar2024", "stress_aug2024"]
SCENARIO_KEY = {
    "baseline_mar2024": "baseline",
    "stress_aug2024": "stress",
}

payloads: dict[str, dict] = {}

for folder_name in SCENARIO_DIRS:
    scenario_key = SCENARIO_KEY[folder_name]
    json_path = RAW_DATA_DIR / folder_name / "thegraph_hf_30d.json"
    print(f"[{scenario_key}] {json_path.resolve()}")
    print(f"  exists={json_path.exists()} size_mb={json_path.stat().st_size / 1e6:.1f}")

    with json_path.open("r", encoding="utf-8") as handle:
        payloads[scenario_key] = json.load(handle)

    payload = payloads[scenario_key]
    print(
        f"  users={len(payload['users']):,} "
        f"liquidationCalls={len(payload['liquidationCalls']):,} "
        f"window={payload.get('start_datetime_utc')} → {payload.get('end_datetime_utc')}"
    )

# Alias retained so the structure-inspection cells below still run on baseline.
baseline_payload = payloads["baseline"]
print(f"\nLoaded scenarios: {list(payloads.keys())}")

# %% [markdown]
# ## 3. JSON structure inspection
#
# The extractor writes metadata fields alongside the two GraphQL collections. The collections required for Markov calibration are **`users`** and **`liquidationCalls`**. Lengths should match the extraction logs (~49k borrowers).

# %%
print("Top-level keys:")
for key in baseline_payload.keys():
    print(f"  - {key}")

print()
for collection_name in ("users", "liquidationCalls"):
    collection = baseline_payload[collection_name]
    print(
        f"{collection_name}: type={type(collection).__name__}, "
        f"n_records={len(collection):,}"
    )

print()
print("Extractor metadata (if present):")
for meta_key in (
    "protocol",
    "network",
    "window_days",
    "start_datetime_utc",
    "end_datetime_utc",
    "user_count",
    "liquidation_call_count",
):
    if meta_key in baseline_payload:
        print(f"  {meta_key}: {baseline_payload[meta_key]}")

# %% [markdown]
# ## 4. Conversion to DataFrames
#
# ### 4.1 Users
#
# Each row is one borrower. Nested objects (`eModeCategoryId`, `reserves`) are stored as Python objects in object-dtype columns.

# %%
users_df = pd.DataFrame(baseline_payload["users"])

print(f"users_df shape: {users_df.shape}")
display(users_df.head())
users_df.info()

# %% [markdown]
# ### 4.2 Liquidation calls
#
# Each row is one `liquidationCall` event inside the 30-day baseline window.

# %%
liquidation_calls_df = pd.DataFrame(baseline_payload["liquidationCalls"])

print(f"liquidation_calls_df shape: {liquidation_calls_df.shape}")
display(liquidation_calls_df.head())
liquidation_calls_df.info()


# %% [markdown]
# ## 5. Nested completeness and live-debt example
#
# Position history for Health Factor reconstruction lives **inside** each user: `reserves` is a list of per-asset snapshots, and each reserve may contain `aTokenBalanceHistory`, `vTokenBalanceHistory`, and `sTokenBalanceHistory`.
#
# This cell:
#
# 1. Identifies nested (list/dict) columns in `users_df`.
# 2. Summarizes how often those nested histories are non-empty.
# 3. Prints one borrower with `borrowedReservesCount > 0` as a structural example.

# %%
def classify_nested_column(series: pd.Series) -> str:
    """Return a short label for the first non-null nested Python type."""
    sample = series.dropna().iloc[0] if series.notna().any() else None
    if isinstance(sample, list):
        return f"list (example length={len(sample)})"
    if isinstance(sample, dict):
        return f"dict (keys={list(sample.keys())})"
    return type(sample).__name__


print("Nested / object columns in users_df:")
for column in users_df.columns:
    if users_df[column].dtype == "object":
        print(f"  - {column}: {classify_nested_column(users_df[column])}")

n_users = len(users_df)
n_with_reserves = users_df["reserves"].map(lambda value: isinstance(value, list) and len(value) > 0).sum()
n_live_debt = (users_df["borrowedReservesCount"] > 0).sum()

history_counts = {"aTokenBalanceHistory": 0, "vTokenBalanceHistory": 0, "sTokenBalanceHistory": 0}
n_reserves_total = 0
for reserves in users_df["reserves"]:
    if not isinstance(reserves, list):
        continue
    n_reserves_total += len(reserves)
    for reserve in reserves:
        if not isinstance(reserve, dict):
            continue
        for history_key in history_counts:
            history = reserve.get(history_key, [])
            if isinstance(history, list) and len(history) > 0:
                history_counts[history_key] += 1

print()
print(f"Users: {n_users:,}")
print(f"Users with borrowedReservesCount > 0: {n_live_debt:,}")
print(f"Users with a non-empty reserves list: {n_with_reserves:,}")
print(f"Total nested reserve records: {n_reserves_total:,}")
print("Reserves with non-empty 30-day histories:")
for history_key, count in history_counts.items():
    print(f"  - {history_key}: {count:,}")

live_debt_mask = users_df["borrowedReservesCount"] > 0
if not live_debt_mask.any():
    raise ValueError("No user with borrowedReservesCount > 0 was found in the baseline extract.")

example_user = users_df.loc[live_debt_mask].iloc[0]
example_reserves = example_user["reserves"] if isinstance(example_user["reserves"], list) else []

print()
print("Example borrower with live debt (borrowedReservesCount > 0):")
print(f"  id: {example_user['id']}")
print(f"  borrowedReservesCount: {example_user['borrowedReservesCount']}")
print(f"  eModeCategoryId: {example_user.get('eModeCategoryId')}")
print(f"  n_reserves: {len(example_reserves)}")

if example_reserves:
    example_reserve = example_reserves[0]
    reserve_meta = example_reserve.get("reserve", {})
    print()
    print("First nested reserve (structural sample):")
    print(f"  symbol: {reserve_meta.get('symbol')}")
    print(f"  currentATokenBalance: {example_reserve.get('currentATokenBalance')}")
    print(f"  currentVariableDebt: {example_reserve.get('currentVariableDebt')}")
    print(f"  currentTotalDebt: {example_reserve.get('currentTotalDebt')}")
    print(f"  len(aTokenBalanceHistory): {len(example_reserve.get('aTokenBalanceHistory') or [])}")
    print(f"  len(vTokenBalanceHistory): {len(example_reserve.get('vTokenBalanceHistory') or [])}")
    print(f"  len(sTokenBalanceHistory): {len(example_reserve.get('sTokenBalanceHistory') or [])}")

fig, ax = plt.subplots()
users_df["borrowedReservesCount"].value_counts().sort_index().plot(
    kind="bar", ax=ax, color="#4C72B0"
)
ax.set_title("Baseline: distribution of borrowedReservesCount")
ax.set_xlabel("borrowedReservesCount")
ax.set_ylabel("Number of users")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 6. Flatten nested histories into a longitudinal panel
#
# Each borrower stores collateral and variable-debt paths as **separate nested lists** (`aTokenBalanceHistory`, `vTokenBalanceHistory`) under `reserves`. Markov calibration needs a **long panel**: one row per `(user_id, reserve_symbol, timestamp)`.
#
# **Constant-state rule.** The Graph only emits a history point when the scaled balance changes. If `currentATokenBalance` or `currentVariableDebt` is strictly greater than zero but the matching history list is empty, the position was live and **unchanged** over the 30-day window. Those balances are inferred at the window start and window end timestamps so the panel spans the full valuation interval.
#
# Idle reserves (zero current balances and empty histories) are dropped. Collateral and debt series are aligned on the union of timestamps, with last-observation-carried-forward within each user-reserve.
#
# Both scenarios are flattened into `flat_panels = {'baseline': ..., 'stress': ...}`.

# %%
def _as_int(value) -> int:
    """Coerce GraphQL numeric strings to int; missing values become 0."""
    if value in (None, ""):
        return 0
    return int(value)


def _history_by_timestamp(history, balance_key: str) -> dict[int, int]:
    """Map Unix timestamps to raw token balances (last write wins)."""
    points: dict[int, int] = {}
    if not history:
        return points
    for item in history:
        timestamp = _as_int(item.get("timestamp"))
        points[timestamp] = _as_int(item.get(balance_key))
    return points


def flatten_user_reserve_histories(
    users: list,
    window_start: int,
    window_end: int,
) -> list[tuple]:
    """Unnest Aave v3 user reserves into a longitudinal list of tuples.

    For each user-reserve the function:
    1. Collects aToken (collateral) and vToken (variable debt) history points.
    2. If a live current balance has an empty history, fills that constant
       balance at ``window_start`` and ``window_end``.
    3. Walks the union of timestamps and forward-fills the other side.

    Returns
    -------
    list[tuple]
        ``(user_id, timestamp, reserve_symbol, aToken_balance, vToken_balance)``
        with balances scaled by the reserve decimals.
    """
    rows: list[tuple] = []
    append = rows.append

    for user in users:
        user_id = user.get("id")
        for reserve in user.get("reserves") or ():
            meta = reserve.get("reserve") or {}
            symbol = meta.get("symbol") or "UNKNOWN"
            decimals = _as_int(meta.get("decimals")) or 18
            scale = 10 ** decimals

            a_map = _history_by_timestamp(
                reserve.get("aTokenBalanceHistory"),
                "currentATokenBalance",
            )
            v_map = _history_by_timestamp(
                reserve.get("vTokenBalanceHistory"),
                "currentVariableDebt",
            )
            a_now = _as_int(reserve.get("currentATokenBalance"))
            v_now = _as_int(reserve.get("currentVariableDebt"))

            # Constant collateral: live balance, no change events in the window.
            if not a_map and a_now > 0:
                a_map[window_start] = a_now
                a_map[window_end] = a_now
            # Constant variable debt: live balance, no change events in the window.
            if not v_map and v_now > 0:
                v_map[window_start] = v_now
                v_map[window_end] = v_now

            if not a_map and not v_map:
                continue

            last_a = 0
            last_v = 0
            for timestamp in sorted(a_map.keys() | v_map.keys()):
                if timestamp in a_map:
                    last_a = a_map[timestamp]
                if timestamp in v_map:
                    last_v = v_map[timestamp]
                append(
                    (
                        user_id,
                        timestamp,
                        symbol,
                        last_a / scale,
                        last_v / scale,
                    )
                )

    return rows


# %%
def payload_to_flat_panel(payload: dict) -> pd.DataFrame:
    """Flatten one Graph payload into a user-reserve timestamp panel."""
    window_start = int(payload["start_timestamp"])
    window_end = int(payload["end_timestamp"])
    rows = flatten_user_reserve_histories(
        payload["users"],
        window_start=window_start,
        window_end=window_end,
    )
    panel = pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "timestamp",
            "reserve_symbol",
            "aToken_balance",
            "vToken_balance",
        ],
    )
    panel["date"] = pd.to_datetime(panel["timestamp"], unit="s", utc=True)
    return panel[
        [
            "user_id",
            "timestamp",
            "date",
            "reserve_symbol",
            "aToken_balance",
            "vToken_balance",
        ]
    ]


flat_panels: dict[str, pd.DataFrame] = {}

for scenario_key, payload in payloads.items():
    panel = payload_to_flat_panel(payload)
    window_start = int(payload["start_timestamp"])
    window_end = int(payload["end_timestamp"])
    n_constant_boundary = int(
        ((panel["timestamp"] == window_start) | (panel["timestamp"] == window_end)).sum()
    )
    print(f"[{scenario_key}] window=[{window_start}, {window_end}]")
    print(f"  flattened rows: {len(panel):,}")
    print(f"  unique users: {panel['user_id'].nunique():,}")
    print(f"  unique reserves: {panel['reserve_symbol'].nunique():,}")
    print(f"  rows at window start or end (includes inferred constants): {n_constant_boundary:,}")
    flat_panels[scenario_key] = panel

flat_panel_base = flat_panels["baseline"]
flat_panel_stress = flat_panels["stress"]
flat_panel_df = flat_panel_base  # alias for the baseline verification cell below

# %% [markdown]
# ## 7. Flattened panel verification
#
# Sort each scenario by `user_id` and `timestamp`, then inspect shape, dtypes, and the first ten rows.

# %%
for scenario_key, panel in flat_panels.items():
    flat_panels[scenario_key] = (
        panel.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    )
    print(f"=== {scenario_key} ===")
    print(f"shape: {flat_panels[scenario_key].shape}")
    print()
    flat_panels[scenario_key].info()
    print()
    display(flat_panels[scenario_key].head(10))
    print()

flat_panel_base = flat_panels["baseline"]
flat_panel_stress = flat_panels["stress"]
flat_panel_df = flat_panel_base

# %% [markdown]
# ## 8. Market data ingestion
#
# For each scenario folder, load the three Binance daily extracts (`binance_prices_eth.csv`, `binance_prices_uni.csv`, `binance_prices_btc.csv`) and keep the **close** as the USD proxy.
#
# Asset-to-ticker assignment uses `PRICE_PROXIES` from `src.utils.config` (ETH-correlated LSTs inherit the ETHUSDT close; WBTC inherits BTCUSDT; UNI uses UNIUSDT). Every symbol in `STABLECOINS` is priced at a strict **$1.00**.

# %%
BINANCE_CSV_BY_TICKER = {
    "ETHUSDT": "binance_prices_eth.csv",
    "UNIUSDT": "binance_prices_uni.csv",
    "BTCUSDT": "binance_prices_btc.csv",
}

STABLECOIN_SET = set(STABLECOINS)
ETH_CORE = set(LST_ASSETS) | {"WETH", "ETH"}
PROFILE_A_REMAINDER = STABLECOIN_SET | {"WBTC"}
PROFILE_B_CORE = {"UNI", "LINK"}
PROFILE_B_REMAINDER = STABLECOIN_SET | {"WETH", "ETH", "WBTC"} | set(LST_ASSETS)
DOMINANCE_THRESHOLD = 0.85
DUST_USD = 1e-4


def load_scenario_close_prices(folder_name: str) -> pd.DataFrame:
    """Load Binance daily closes for one scenario, keyed by (date, ticker)."""
    frames = []
    scenario_dir = RAW_DATA_DIR / folder_name
    for ticker, filename in BINANCE_CSV_BY_TICKER.items():
        csv_path = scenario_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing Binance extract: {csv_path.resolve()}")
        frame = pd.read_csv(csv_path, parse_dates=["date"])
        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
        frame["ticker"] = ticker
        frames.append(frame[["date", "ticker", "close"]])
    prices = pd.concat(frames, ignore_index=True)
    prices["close"] = pd.to_numeric(prices["close"], errors="raise")
    return prices.drop_duplicates(subset=["date", "ticker"], keep="last")


price_maps: dict[str, pd.DataFrame] = {}
for folder_name in SCENARIO_DIRS:
    scenario_key = SCENARIO_KEY[folder_name]
    prices = load_scenario_close_prices(folder_name)
    price_maps[scenario_key] = prices
    print(f"[{scenario_key}] close-price rows: {len(prices):,}")
    print(prices.groupby("ticker")["date"].agg(["min", "max", "count"]))
    print()


# %% [markdown]
# ## 9. Daily USD valuation
#
# Sparse Graph snapshots are expanded onto the scenario calendar so that **constant token balances still revalue daily** with Binance closes. Then:
#
# - `collateral_usd = aToken_balance * price`
# - `debt_usd = vToken_balance * price`
#
# Stablecoins receive `price = 1.0`. All other symbols take `PRICE_PROXIES[symbol]` and the matching scenario close. Positions with a positive token balance but no proxy are left unpriced and later fail the purity screens.

# %%
def expand_to_daily_panel(
    panel: pd.DataFrame,
    window_start: int,
    window_end: int,
) -> pd.DataFrame:
    """Forward-fill user-reserve balances onto every UTC calendar day in the window."""
    start = pd.to_datetime(window_start, unit="s", utc=True).normalize()
    end = pd.to_datetime(window_end, unit="s", utc=True).normalize()
    calendar = pd.DataFrame({"date": pd.date_range(start, end, freq="D", tz="UTC")})
    pairs = panel[["user_id", "reserve_symbol"]].drop_duplicates()
    grid = pairs.merge(calendar, how="cross")

    sparse = (
        panel.assign(date=lambda df: df["date"].dt.normalize())
        .sort_values(["user_id", "reserve_symbol", "date", "timestamp"])
        .groupby(["user_id", "reserve_symbol", "date"], as_index=False)
        .agg(
            aToken_balance=("aToken_balance", "last"),
            vToken_balance=("vToken_balance", "last"),
        )
    )
    daily = grid.merge(sparse, on=["user_id", "reserve_symbol", "date"], how="left")
    daily = daily.sort_values(["user_id", "reserve_symbol", "date"], kind="mergesort")
    daily[["aToken_balance", "vToken_balance"]] = daily.groupby(
        ["user_id", "reserve_symbol"], sort=False
    )[["aToken_balance", "vToken_balance"]].ffill()
    daily = daily.dropna(subset=["aToken_balance", "vToken_balance"], how="all")
    live = (daily["aToken_balance"] > 0) | (daily["vToken_balance"] > 0)
    return daily.loc[live].copy()


def value_panel_usd(daily_panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach daily USD prices and compute collateral_usd / debt_usd."""
    valued = daily_panel.copy()
    valued["ticker"] = valued["reserve_symbol"].map(PRICE_PROXIES)
    valued["is_stablecoin"] = valued["reserve_symbol"].isin(STABLECOIN_SET)
    valued = valued.merge(prices, on=["date", "ticker"], how="left")
    valued["price"] = valued["close"]
    valued.loc[valued["is_stablecoin"], "price"] = 1.0
    valued["collateral_usd"] = valued["aToken_balance"] * valued["price"]
    valued["debt_usd"] = valued["vToken_balance"] * valued["price"]
    return valued


usd_panels: dict[str, pd.DataFrame] = {}

for folder_name in SCENARIO_DIRS:
    scenario_key = SCENARIO_KEY[folder_name]
    payload = payloads[scenario_key]
    daily = expand_to_daily_panel(
        flat_panels[scenario_key],
        window_start=int(payload["start_timestamp"]),
        window_end=int(payload["end_timestamp"]),
    )
    valued = value_panel_usd(daily, price_maps[scenario_key])
    usd_panels[scenario_key] = valued
    n_unpriced = int(
        (
            (~valued["is_stablecoin"])
            & valued["price"].isna()
            & ((valued["aToken_balance"] > 0) | (valued["vToken_balance"] > 0))
        ).sum()
    )
    print(f"[{scenario_key}] daily USD rows: {len(valued):,}")
    print(f"  unique users: {valued['user_id'].nunique():,}")
    print(f"  unpriced live rows (no PRICE_PROXIES match): {n_unpriced:,}")
    display(valued.head())
    print()

# %% [markdown]
# ## 10. Aggregation and actuarial dominance filters
#
# Positions are summed to the `(user_id, date)` level. Sub-microdollar residuals (`DUST_USD`) are treated as zero so leftover wei does not contaminate purity tests.
#
# **Debt rule:** 100% of `total_debt_usd` must be in `STABLECOINS`.
#
# **Profile A (ETH dominance):** at least 85% of `total_collateral_usd` in `WETH` or `LST_ASSETS`. The residual (at most 15%) may only be `STABLECOINS` or `WBTC`.
#
# **Profile B (UNI+LINK high-risk dominance):** at least 85% of `total_collateral_usd` in the high-risk basket `UNI` + `LINK`, i.e. `(collateral_UNI_usd + collateral_LINK_usd) / total_collateral_usd >= 0.85`. The residual (at most 15%) may only be lower-volatility assets: `STABLECOINS`, `WETH`, `LST_ASSETS`, or `WBTC`.
#
# A wallet is retained if it satisfies Profile A and/or Profile B on **at least one** calendar day. The Health Factor panel below keeps only those qualifying user-days.

# %%
# Profile B core: joint high-risk basket. Remainder is unchanged (lower-volatility assets only).
PROFILE_B_CORE = {"UNI", "LINK"}
FOLDER_BY_SCENARIO = {key: folder for folder, key in SCENARIO_KEY.items()}


def attach_link_usd_prices(usd_panel: pd.DataFrame, folder_name: str) -> pd.DataFrame:
    """Fill LINK USD using binance_prices_link.csv so LINK can enter the Profile B numerator."""
    csv_path = RAW_DATA_DIR / folder_name / "binance_prices_link.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing Binance extract: {csv_path.resolve()}")
    link_prices = pd.read_csv(csv_path, parse_dates=["date"])
    link_prices["date"] = pd.to_datetime(link_prices["date"], utc=True).dt.normalize()
    link_prices = link_prices[["date", "close"]].rename(columns={"close": "link_close"})
    panel = usd_panel.merge(link_prices, on="date", how="left")
    link_mask = panel["reserve_symbol"].eq("LINK")
    panel.loc[link_mask, "ticker"] = "LINKUSDT"
    panel.loc[link_mask, "price"] = panel.loc[link_mask, "link_close"]
    panel.loc[link_mask, "collateral_usd"] = (
        panel.loc[link_mask, "aToken_balance"] * panel.loc[link_mask, "price"]
    )
    panel.loc[link_mask, "debt_usd"] = (
        panel.loc[link_mask, "vToken_balance"] * panel.loc[link_mask, "price"]
    )
    return panel.drop(columns=["link_close"])


for scenario_key in list(usd_panels):
    usd_panels[scenario_key] = attach_link_usd_prices(
        usd_panels[scenario_key], FOLDER_BY_SCENARIO[scenario_key]
    )


def apply_dust_floor(usd_panel: pd.DataFrame) -> pd.DataFrame:
    """Zero out sub-microdollar USD residuals from integer wei dust."""
    cleaned = usd_panel.copy()
    cleaned["collateral_usd"] = cleaned["collateral_usd"].fillna(0.0)
    cleaned["debt_usd"] = cleaned["debt_usd"].fillna(0.0)
    cleaned.loc[cleaned["collateral_usd"] < DUST_USD, "collateral_usd"] = 0.0
    cleaned.loc[cleaned["debt_usd"] < DUST_USD, "debt_usd"] = 0.0
    return cleaned


def classify_dominance(usd_panel: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (user_id, date) with Profile A / B flags and USD totals."""
    work = apply_dust_floor(usd_panel)
    work["has_unpriced_collateral"] = (
        (~work["is_stablecoin"])
        & work["price"].isna()
        & (work["aToken_balance"] > 0)
    )

    totals = work.groupby(["user_id", "date"], as_index=False).agg(
        total_collateral_usd=("collateral_usd", "sum"),
        total_debt_usd=("debt_usd", "sum"),
        has_unpriced_collateral=("has_unpriced_collateral", "any"),
    )

    eth_core_usd = (
        work.loc[work["reserve_symbol"].isin(ETH_CORE)]
        .groupby(["user_id", "date"])["collateral_usd"]
        .sum()
        .rename("eth_core_collateral_usd")
    )
    high_risk_usd = (
        work.loc[work["reserve_symbol"].isin(PROFILE_B_CORE)]
        .groupby(["user_id", "date"])["collateral_usd"]
        .sum()
        .rename("high_risk_collateral_usd")
    )
    stable_debt_usd = (
        work.loc[work["reserve_symbol"].isin(STABLECOIN_SET)]
        .groupby(["user_id", "date"])["debt_usd"]
        .sum()
        .rename("stable_debt_usd")
    )

    a_disallowed = work.loc[
        (work["collateral_usd"] > 0)
        & ~work["reserve_symbol"].isin(ETH_CORE | PROFILE_A_REMAINDER)
    ][["user_id", "date"]].drop_duplicates()
    a_disallowed["profile_a_disallowed"] = True

    b_disallowed = work.loc[
        (work["collateral_usd"] > 0)
        & ~work["reserve_symbol"].isin(PROFILE_B_CORE | PROFILE_B_REMAINDER)
    ][["user_id", "date"]].drop_duplicates()
    b_disallowed["profile_b_disallowed"] = True

    classified = (
        totals.merge(eth_core_usd, on=["user_id", "date"], how="left")
        .merge(high_risk_usd, on=["user_id", "date"], how="left")
        .merge(stable_debt_usd, on=["user_id", "date"], how="left")
        .merge(a_disallowed, on=["user_id", "date"], how="left")
        .merge(b_disallowed, on=["user_id", "date"], how="left")
    )
    classified["eth_core_collateral_usd"] = classified["eth_core_collateral_usd"].fillna(0.0)
    classified["high_risk_collateral_usd"] = classified["high_risk_collateral_usd"].fillna(0.0)
    classified["stable_debt_usd"] = classified["stable_debt_usd"].fillna(0.0)
    classified["eth_share"] = (
        classified["eth_core_collateral_usd"] / classified["total_collateral_usd"].replace(0, pd.NA)
    ).fillna(0.0)
    classified["high_risk_share"] = (
        classified["high_risk_collateral_usd"] / classified["total_collateral_usd"].replace(0, pd.NA)
    ).fillna(0.0)

    classified["profile_a_disallowed"] = classified["profile_a_disallowed"].fillna(False).astype(bool)
    classified["profile_b_disallowed"] = classified["profile_b_disallowed"].fillna(False).astype(bool)

    debt_ok = (classified["total_debt_usd"] > 0) & (
        classified["stable_debt_usd"] >= classified["total_debt_usd"] * (1.0 - 1e-8)
    )
    collat_ok = classified["total_collateral_usd"] > 0
    priced_ok = ~classified["has_unpriced_collateral"]

    classified["is_profile_a"] = (
        debt_ok
        & collat_ok
        & priced_ok
        & ~classified["profile_a_disallowed"]
        & (classified["eth_share"] >= DOMINANCE_THRESHOLD)
    )
    classified["is_profile_b"] = (
        debt_ok
        & collat_ok
        & priced_ok
        & ~classified["profile_b_disallowed"]
        & (classified["high_risk_share"] >= DOMINANCE_THRESHOLD)
    )
    return classified


dominance_panels: dict[str, pd.DataFrame] = {}
eligible_wallets: dict[str, set] = {}

for scenario_key, usd_panel in usd_panels.items():
    classified = classify_dominance(usd_panel)
    dominance_panels[scenario_key] = classified
    surviving = classified.loc[
        classified["is_profile_a"] | classified["is_profile_b"], "user_id"
    ].unique()
    eligible_wallets[scenario_key] = set(surviving)
    print(f"[{scenario_key}] user-days: {len(classified):,}")
    print(f"  Profile A user-days: {int(classified['is_profile_a'].sum()):,}")
    print(f"  Profile B user-days: {int(classified['is_profile_b'].sum()):,}")
    print(f"  wallets passing on at least one day: {len(surviving):,}")
    print()

profile_b_wallet_labels = []
for scenario_key, classified in dominance_panels.items():
    n_profile_b = classified.loc[classified["is_profile_b"], "user_id"].nunique()
    profile_b_wallet_labels.extend([scenario_key] * n_profile_b)
    print(f"[{scenario_key}] Profile B wallets (UNI+LINK dominance): {n_profile_b:,}")

print()
print("Profile B surviving wallets under joint UNI+LINK dominance (value_counts):")
print(pd.Series(profile_b_wallet_labels, name="scenario").value_counts())


# %% [markdown]
# ## 11. Empirical Health Factor
#
# For qualifying user-days in both scenarios:
#
# $$
# HF_t = \frac{\sum_i (\text{Collateral USD}_{i,t} \times LT_i)}{\text{Total Debt USD}_t}
# $$
#
# `LT_i` is taken **only** from `LIQUIDATION_THRESHOLDS`. Symbols missing from that map — including any stablecoin not listed explicitly — use `LIQUIDATION_THRESHOLDS['DEFAULT']`.
#
# If a user-day matches both profiles (should be rare), it is labelled `A+B`. Wallet counts below use `.value_counts()` on the distinct profile labels of surviving wallets.

# %%
def liquidation_threshold(symbol: str) -> float:
    """Look up LT from config; stablecoins and unknown assets use DEFAULT."""
    if symbol in LIQUIDATION_THRESHOLDS:
        return LIQUIDATION_THRESHOLDS[symbol]
    if symbol in STABLECOIN_SET:
        return LIQUIDATION_THRESHOLDS["DEFAULT"]
    return LIQUIDATION_THRESHOLDS["DEFAULT"]


def assign_profile_label(is_profile_a: bool, is_profile_b: bool) -> str:
    if is_profile_a and is_profile_b:
        return "A+B"
    if is_profile_a:
        return "A"
    if is_profile_b:
        return "B"
    return ""


def compute_health_factor_panel(
    usd_panel: pd.DataFrame,
    classified: pd.DataFrame,
) -> pd.DataFrame:
    """HF for user-days that satisfy Profile A and/or Profile B."""
    qualifying = classified.loc[
        classified["is_profile_a"] | classified["is_profile_b"],
        [
            "user_id",
            "date",
            "total_collateral_usd",
            "total_debt_usd",
            "is_profile_a",
            "is_profile_b",
        ],
    ].copy()
    qualifying["profile"] = [
        assign_profile_label(is_a, is_b)
        for is_a, is_b in zip(qualifying["is_profile_a"], qualifying["is_profile_b"])
    ]

    asset_level = apply_dust_floor(usd_panel)
    asset_level["lt"] = asset_level["reserve_symbol"].map(liquidation_threshold)
    asset_level["threshold_weighted_collateral_usd"] = (
        asset_level["collateral_usd"] * asset_level["lt"]
    )
    weighted = (
        asset_level.groupby(["user_id", "date"], as_index=False)[
            "threshold_weighted_collateral_usd"
        ].sum()
    )

    hf = qualifying.merge(weighted, on=["user_id", "date"], how="left")
    hf["health_factor"] = hf["threshold_weighted_collateral_usd"] / hf["total_debt_usd"]
    hf = hf.sort_values(["profile", "user_id", "date"], kind="mergesort").reset_index(drop=True)
    return hf[
        [
            "user_id",
            "date",
            "profile",
            "total_collateral_usd",
            "total_debt_usd",
            "threshold_weighted_collateral_usd",
            "health_factor",
        ]
    ]


hf_panels: dict[str, pd.DataFrame] = {}
wallet_profile_counts: dict[str, pd.Series] = {}

for scenario_key in payloads:
    hf_df = compute_health_factor_panel(usd_panels[scenario_key], dominance_panels[scenario_key])
    hf_df.insert(0, "scenario", scenario_key)
    hf_panels[scenario_key] = hf_df

    wallet_profiles = (
        hf_df.groupby("user_id")["profile"]
        .agg(lambda labels: ",".join(sorted(set(labels))))
        .rename("profile")
    )
    wallet_profile_counts[scenario_key] = wallet_profiles.value_counts()

    print(f"=== {scenario_key} ===")
    print("Pure wallets by profile (value_counts):")
    print(wallet_profile_counts[scenario_key])
    print()
    print(f"HF panel shape: {hf_df.shape}")
    display(hf_df.head())
    print()

hf_panel_df = pd.concat(hf_panels.values(), ignore_index=True)
print("Combined HF panel:")
print(hf_panel_df.shape)
display(hf_panel_df.head())

# %% [markdown]
# ## 12. Markov state discretization and daily transitions
#
# This section starts from `hf_panel_df` and builds the empirical transition sample:
#
# 1. **Purity filter** — drop hybrid wallets whose profile label is `A,B` (or `A+B`). Keep strictly **A** or **B**.
# 2. **Current state** — map `health_factor` to `{S2,...,S7}`. Voluntary prepayment `S1` is assigned only as a *next* state.
# 3. **Next state** — within `(scenario, user_id)`, shift the current state one observation forward. Wallets that exit before the scenario window end are mapped to `S1` (HF > 1) or absorbed in `S7` (HF <= 1). The last day of each 30-day window has no valid t to t+1 transition and is excluded from the counts.

# %%
wallet_profile = (
    hf_panel_df.groupby(["scenario", "user_id"], sort=False)["profile"]
    .agg(lambda labels: ",".join(sorted(set(labels))))
    .rename("wallet_profile")
    .reset_index()
)

hybrid_mask = wallet_profile["wallet_profile"].isin(["A,B", "A+B", "B,A"])
n_hybrid_wallets = int(hybrid_mask.sum())
print(f"Hybrid wallets dropped (A,B / A+B): {n_hybrid_wallets:,}")
print(wallet_profile["wallet_profile"].value_counts())
print()

pure_wallets = wallet_profile.loc[
    wallet_profile["wallet_profile"].isin(["A", "B"]),
    ["scenario", "user_id", "wallet_profile"],
]

markov_states_df = hf_panel_df.merge(pure_wallets, on=["scenario", "user_id"], how="inner")
markov_states_df = markov_states_df.loc[markov_states_df["profile"].isin(["A", "B"])].copy()
markov_states_df["profile"] = markov_states_df["wallet_profile"]
markov_states_df = markov_states_df.drop(columns=["wallet_profile"])

print(f"Pure HF rows retained: {len(markov_states_df):,}")
print(f"Pure wallets retained: {markov_states_df.groupby(['scenario', 'user_id']).ngroups:,}")
print(markov_states_df.drop_duplicates(["scenario", "user_id"])["profile"].value_counts())

# %% [markdown]
# ## 13. Current-state mapping
#
# `pd.cut` assigns bins with a closed right edge. The single boundary `HF == 1.10` is remapped from `S6` to `S5` so that the target band is `1.10 <= HF < 1.25` and the critical band is `1.0 < HF < 1.10`.

# %%
import numpy as np

HF_STATE_BINS = [-np.inf, 1.0, 1.10, 1.25, 1.50, 2.0, np.inf]
HF_STATE_LABELS = ["S7", "S6", "S5", "S4", "S3", "S2"]
STATE_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]


def map_health_factor_to_state(health_factor: pd.Series) -> pd.Series:
    """Map Health Factor to current Markov states S2-S7 (S1 is transition-only)."""
    state = pd.cut(
        health_factor,
        bins=HF_STATE_BINS,
        labels=HF_STATE_LABELS,
        right=True,
        include_lowest=True,
    ).astype("object")
    # pd.cut(right=True) puts HF == 1.10 in S6; the target band starts at 1.10 inclusive.
    state = state.mask(np.isclose(health_factor, 1.10), "S5")
    return pd.Series(state, index=health_factor.index, name="state")


markov_states_df["state"] = map_health_factor_to_state(markov_states_df["health_factor"])

print("Current-state value counts by profile:")
for profile, block in markov_states_df.groupby("profile"):
    print(f"\nProfile {profile}")
    print(block["state"].value_counts().reindex(STATE_ORDER[1:]).fillna(0).astype(int))

# %% [markdown]
# ## 14. Longitudinal next state and actuarial right-censoring
#
# Zero collateral and debt rows are dropped upstream, so `total_debt_usd == 0` is not observed. Prepayment (`S1`) is therefore the **dropout** rule: if a wallet's last observed date is strictly before the scenario window end and its last Health Factor is above 1.0, it enters `S1` on `last_date + 1 day`.
#
# Default (`S7`) is the first observed day with `health_factor <= 1.0`.
#
# Each wallet is expanded to a continuous daily index from its first observation through `window_end_date`. The first absorbing hit (`S7` wins a same-day tie) is forward-filled to the window end, which masks restorative liquidations and creates the empirical `S1 -> S1` and `S7 -> S7` transitions. `next_state` is the next day within `(scenario, user_id)`. The window-end day has no `t+1` and is excluded.

# %%
scenario_window_end = {
    key: pd.to_datetime(payload["end_datetime_utc"], utc=True).normalize()
    for key, payload in payloads.items()
}

observed = (
    markov_states_df.sort_values(["scenario", "user_id", "date"], kind="mergesort")
    .reset_index(drop=True)
    .copy()
)
observed["date"] = observed["date"].dt.normalize()
observed["window_end_date"] = observed["scenario"].map(scenario_window_end)

wallet_span = (
    observed.groupby(["scenario", "user_id"], sort=False)
    .agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        window_end_date=("window_end_date", "first"),
        profile=("profile", "first"),
    )
    .reset_index()
)
last_obs = (
    observed.groupby(["scenario", "user_id"], sort=False)
    .tail(1)[["scenario", "user_id", "health_factor"]]
    .rename(columns={"health_factor": "last_health_factor"})
)
wallet_span = wallet_span.merge(last_obs, on=["scenario", "user_id"], how="left")

# S7: first observed day with HF <= 1.0.
s7_hits = (
    observed.loc[observed["health_factor"] <= 1.0, ["scenario", "user_id", "date"]]
    .groupby(["scenario", "user_id"], sort=False, as_index=False)["date"]
    .min()
    .rename(columns={"date": "event_date"})
)
s7_hits["absorbing_state"] = "S7"

# S1: dropout before the window end while the last observed HF is still above 1.
# Entry is the calendar day after the last observation (zero balances are absent from the panel).
dropout_mask = (wallet_span["last_date"] < wallet_span["window_end_date"]) & (
    wallet_span["last_health_factor"] > 1.0
)
s1_hits = wallet_span.loc[dropout_mask, ["scenario", "user_id", "last_date"]].copy()
s1_hits["event_date"] = s1_hits["last_date"] + pd.Timedelta(days=1)
s1_hits["absorbing_state"] = "S1"
s1_hits = s1_hits[["scenario", "user_id", "event_date", "absorbing_state"]]

events = pd.concat([s7_hits, s1_hits], ignore_index=True)
events["tie_rank"] = events["absorbing_state"].map({"S7": 0, "S1": 1})
events = events.sort_values(
    ["scenario", "user_id", "event_date", "tie_rank"], kind="mergesort"
)
first_event = events.groupby(["scenario", "user_id"], sort=False, as_index=False).head(1)
first_event = first_event[["scenario", "user_id", "event_date", "absorbing_state"]].copy()

n_s1_dropouts = int(len(s1_hits))
n_s7_hits = int(len(s7_hits))
n_s1_effective = int((first_event["absorbing_state"] == "S1").sum())
n_s7_effective = int((first_event["absorbing_state"] == "S7").sum())

wallet_span["n_days"] = (
    wallet_span["window_end_date"] - wallet_span["first_date"]
).dt.days + 1
wallet_span = wallet_span.loc[wallet_span["n_days"] > 0].copy()

expanded = wallet_span.loc[wallet_span.index.repeat(wallet_span["n_days"])].copy()
expanded["day_offset"] = expanded.groupby(["scenario", "user_id"], sort=False).cumcount()
expanded["date"] = expanded["first_date"] + pd.to_timedelta(expanded["day_offset"], unit="D")

observed_cols = ["scenario", "user_id", "date", "state", "health_factor"]
if "total_debt_usd" in observed.columns:
    observed_cols.append("total_debt_usd")
observed_daily = observed[observed_cols].drop_duplicates(
    ["scenario", "user_id", "date"], keep="last"
)
expanded = expanded.merge(observed_daily, on=["scenario", "user_id", "date"], how="left")
expanded = expanded.merge(first_event, on=["scenario", "user_id"], how="left")
expanded = expanded.sort_values(
    ["scenario", "user_id", "date"], kind="mergesort"
).reset_index(drop=True)

group_keys = ["scenario", "user_id"]
expanded["state_observed"] = expanded.groupby(group_keys, sort=False)["state"].ffill()

hit_mask = expanded["event_date"].notna() & expanded["date"].eq(expanded["event_date"])
expanded["absorbing_hit"] = expanded["absorbing_state"].where(hit_mask)
expanded["absorbing_frozen"] = expanded.groupby(group_keys, sort=False)["absorbing_hit"].ffill()
expanded["final_state"] = expanded["absorbing_frozen"].fillna(expanded["state_observed"])

expanded["is_window_end"] = expanded["date"].eq(expanded["window_end_date"])
expanded["next_state"] = expanded.groupby(group_keys, sort=False)["final_state"].shift(-1)

n_frozen = int(
    expanded["absorbing_frozen"].notna().sum() - hit_mask.sum()
)

markov_states_df = expanded[
    [
        "scenario",
        "user_id",
        "profile",
        "date",
        "health_factor",
        "state_observed",
        "final_state",
        "window_end_date",
        "is_window_end",
        "next_state",
    ]
].copy()
markov_states_df["state"] = markov_states_df["final_state"]

markov_transitions_df = markov_states_df.loc[
    ~markov_states_df["is_window_end"]
    & markov_states_df["state"].notna()
    & markov_states_df["next_state"].notna()
].copy()
markov_transitions_df["transition_pair"] = (
    markov_transitions_df["state"].astype(str)
    + " -> "
    + markov_transitions_df["next_state"].astype(str)
)

print(f"S1 dropout candidates (last date before window end, HF > 1): {n_s1_dropouts:,}")
print(f"S1 first absorbing events after tie-break: {n_s1_effective:,}")
print(f"S7 hits (first day with HF <= 1.0): {n_s7_hits:,}")
print(f"S7 first absorbing events after tie-break: {n_s7_effective:,}")
print(f"Subsequent rows frozen through window end: {n_frozen:,}")
print(f"Window-end rows excluded (no t+1): {int(markov_states_df['is_window_end'].sum()):,}")
print(f"Valid daily transitions (N): {len(markov_transitions_df):,}")
print()
for pair in ("S1 -> S1", "S7 -> S7", "S2 -> S1", "S5 -> S7", "S6 -> S7", "S7 -> S5"):
    n_pair = int((markov_transitions_df["transition_pair"] == pair).sum())
    print(f"  {pair}: {n_pair:,}")

zombie_pairs = [
    "S7 -> S2",
    "S7 -> S3",
    "S7 -> S4",
    "S7 -> S5",
    "S7 -> S6",
    "S1 -> S2",
    "S1 -> S3",
    "S1 -> S4",
    "S1 -> S5",
    "S1 -> S6",
    "S1 -> S7",
    "S7 -> S1",
]
zombie_n = int(markov_transitions_df["transition_pair"].isin(zombie_pairs).sum())
print(f"Contaminated exits from absorbing states (should be 0): {zombie_n:,}")

# %% [markdown]
# ## 15. Sample-size verification
#
# Cross-sectional `state` counts and empirical `t -> t+1` pair counts (`N`) are reported **separately** for each market regime (`baseline_mar2024`, `stress_aug2024`) and collateral profile (A, B), before assembling the transition matrices.

# %%
SCENARIO_LABEL = {
    "baseline": "baseline_mar2024",
    "stress": "stress_aug2024",
}
SCENARIO_PRINT_ORDER = ["baseline", "stress"]
PROFILE_PRINT_ORDER = ["A", "B"]

print("=== Current-state distribution by scenario and profile (value_counts) ===")
for (scenario, profile), block in markov_states_df.groupby(
    ["scenario", "profile"], sort=False
):
    n_wallets = block["user_id"].nunique()
    print(
        f"\n{SCENARIO_LABEL.get(scenario, scenario)} | Profile {profile}  "
        f"(n_rows={len(block):,}, n_wallets={n_wallets:,})"
    )
    print(block["state"].value_counts().reindex(STATE_ORDER[1:]).fillna(0).astype(int))

transition_counts = (
    markov_transitions_df.groupby(["scenario", "profile", "transition_pair"], sort=False)
    .size()
    .rename("n")
    .reset_index()
)

print("\n=== Daily transition pairs by scenario and profile (empirical N) ===")
for scenario in SCENARIO_PRINT_ORDER:
    scenario_label = SCENARIO_LABEL[scenario]
    print(f"\n-------- {scenario_label} --------")
    for profile in PROFILE_PRINT_ORDER:
        block = transition_counts.loc[
            (transition_counts["scenario"] == scenario)
            & (transition_counts["profile"] == profile)
        ].sort_values("n", ascending=False)
        n_total = int(block["n"].sum())
        print(f"\n{scenario_label} | Profile {profile}  (N={n_total:,})")
        if block.empty:
            print("  (no transitions)")
            continue
        pair_counts = block.set_index("transition_pair")["n"]
        print(pair_counts)
        for critical_pair in ("S5 -> S6", "S6 -> S7", "S5 -> S7", "S7 -> S7"):
            n_pair = int(pair_counts.get(critical_pair, 0))
            print(f"  focus {critical_pair}: {n_pair:,}")

print("\n=== Transition-pair counts grouped by scenario, profile, and pair ===")
transition_counts_display = transition_counts.sort_values(
    ["scenario", "profile", "n"], ascending=[True, True, False]
).reset_index(drop=True)
display(transition_counts_display)
print(
    "\nTotal empirical transitions N by scenario and profile:"
)
print(
    markov_transitions_df.groupby(["scenario", "profile"])
    .size()
    .rename("N")
    .reindex(
        pd.MultiIndex.from_product(
            [SCENARIO_PRINT_ORDER, PROFILE_PRINT_ORDER],
            names=["scenario", "profile"],
        )
    )
    .fillna(0)
    .astype(int)
)

# %% [markdown]
# ## 16. Empirical transition matrices
#
# Four daily count matrices $\mathbf{N}$ and probability matrices $\mathbf{P}$ are assembled from `transition_pair` in the right-censored panel: one per market regime (`baseline_mar2024`, `stress_aug2024`) and collateral profile (A, B).
#
# Rows and columns follow `STATE_ORDER`. There is **no** identity injection. `S1 -> S1` and `S7 -> S7` are empirical frequencies produced by the forward-filled absorbing paths in Section 14. Each row is normalized by its sum; a row with zero mass stays at zero.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

STATE_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
SCENARIO_KEYS = [
    ("baseline", "baseline_mar2024"),
    ("stress", "stress_aug2024"),
]
PROFILE_KEYS = ["A", "B"]
PROFILE_TITLES = {
    "A": "Profile A (ETH dominance)",
    "B": "Profile B (UNI+LINK dominance)",
}


def build_empirical_transition_matrices(
    subset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return empirical count matrix N and row-stochastic matrix P.

    N is the frequency of transition_pair values. No absorbing row is
    overwritten. Rows with zero mass stay at 0 after normalization.
    """
    zeros = pd.DataFrame(0.0, index=STATE_ORDER, columns=STATE_ORDER)
    if subset.empty:
        return zeros.copy(), zeros.copy()

    counts = pd.crosstab(subset["state"], subset["next_state"])
    counts = counts.reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0).astype(float)
    row_totals = counts.sum(axis=1)
    transition_p = counts.div(row_totals.replace(0, pd.NA), axis=0).fillna(0.0)
    return counts, transition_p


empirical_N: dict[tuple[str, str], pd.DataFrame] = {}
empirical_P: dict[tuple[str, str], pd.DataFrame] = {}
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

plot_idx = 0
for scenario_key, scenario_label in SCENARIO_KEYS:
    for profile in PROFILE_KEYS:
        subset = markov_transitions_df.loc[
            (markov_transitions_df["scenario"] == scenario_key)
            & (markov_transitions_df["profile"] == profile)
        ].copy()
        counts_n, transition_p = build_empirical_transition_matrices(subset)
        empirical_N[(scenario_label, profile)] = counts_n
        empirical_P[(scenario_label, profile)] = transition_p

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
        ax.set_title(
            f"Empirical daily P  |  {scenario_label}  |  {PROFILE_TITLES[profile]}"
        )
        ax.set_xlabel("Next state")
        ax.set_ylabel("Current state")

        n_s1_stay = int(counts_n.loc["S1", "S1"])
        n_s7_stay = int(counts_n.loc["S7", "S7"])
        p_s1_stay = float(transition_p.loc["S1", "S1"])
        p_s7_stay = float(transition_p.loc["S7", "S7"])
        print(
            f"{scenario_label} | {PROFILE_TITLES[profile]}: "
            f"N={len(subset):,}  "
            f"N(S1->S1)={n_s1_stay:,} P(S1->S1)={p_s1_stay:.4f}  "
            f"N(S7->S7)={n_s7_stay:,} P(S7->S7)={p_s7_stay:.4f}"
        )
        print(f"  row sums: {transition_p.sum(axis=1).round(6).tolist()}")
        plot_idx += 1

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 17. Statistical robustness validation (actuarial audit)
#
# The empirical daily matrices in Section 16 are point estimates. This section adds four audit checks on the right-censored panel `markov_transitions_df`, always split by market regime (`baseline_mar2024`, `stress_aug2024`) and collateral profile (A, B):
#
# 1. **Absolute frequency matrices $\mathbf{N}$**: the sample depth behind every probability.
# 2. **Wallet-level bootstrap**: 95% confidence intervals for the one-step default probability $P_{i,S7}$ and for the 30-day PD implied by $\mathbf{P}^{30}$.
# 3. **Time homogeneity**: $\mathbf{P}$ estimated on days 1-15 versus days 16-30, with a likelihood-ratio test per origin state.
# 4. **Threshold sensitivity**: $\mathbf{P}$ re-estimated after shifting the intermediate Health Factor cut points by $\pm 0.05$.
#
# The cells reuse `STATE_ORDER`, `SCENARIO_KEYS`, `PROFILE_KEYS`, `PROFILE_TITLES`, and `build_empirical_transition_matrices` from Section 16, `scenario_window_end` from Section 14, and `HF_STATE_BINS` / `HF_STATE_LABELS` from Section 13.

# %% [markdown]
# ### 17.1 Absolute frequency matrices N
#
# Rows are the state at day $t$ and columns the state at $t+1$, both ordered `S1` to `S7`. Row totals give the number of observed transitions behind each row of $\mathbf{P}$. Origin states with fewer than `MIN_ROW_DEPTH` transitions are flagged as thin; their probabilities should be read together with the bootstrap intervals in 17.2.

# %%
MIN_ROW_DEPTH = 30


def subset_transitions(
    transitions: pd.DataFrame, scenario_key: str, profile: str
) -> pd.DataFrame:
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

        row_depth = counts_n.sum(axis=1)
        thin_rows = row_depth.loc[row_depth < MIN_ROW_DEPTH]
        print(
            f"\n{scenario_label} | {PROFILE_TITLES[profile]}  "
            f"(N={int(row_depth.sum()):,}, wallets={subset['user_id'].nunique():,})"
        )
        display(frequency_table)
        if thin_rows.empty:
            print(f"  All origin states have at least {MIN_ROW_DEPTH} transitions.")
        else:
            thin_text = ", ".join(f"{state}={n}" for state, n in thin_rows.items())
            print(f"  Thin origin states (< {MIN_ROW_DEPTH} transitions): {thin_text}")

# %% [markdown]
# ### 17.2 Wallet-level bootstrap confidence intervals
#
# Daily transitions of the same wallet are serially dependent, so resampling individual transitions would understate the variance. The bootstrap resamples **wallets** (the cluster) with replacement and keeps each wallet's full path.
#
# Implementation: each wallet's transitions are compressed into a 7x7 count vector $C_w$. A replicate draws multinomial weights $w \sim \text{Multinomial}(n, 1/n)$ over the $n$ wallets, which is equivalent to sampling `user_id` with replacement, and its count matrix is $\sum_w w_w C_w$. All $B = 500$ replicates are obtained with a single matrix product.
#
# For each origin state the table reports the point estimate and the 2.5% / 97.5% percentiles of the one-step default probability $P_{i,S7}$, plus the same interval for the 30-day PD $[\mathbf{P}^{30}]_{i,S7}$. Replicates in which an origin state is never visited are excluded from that row's daily percentiles (`n_valid_draws`); for the 30-day PD such a row follows the Section 16 convention (a zero row).

# %%
import warnings

N_BOOTSTRAP = 500
BOOTSTRAP_SEED = 20240315
CI_PERCENTILES = (2.5, 97.5)
HORIZON_DAYS = 30
N_STATES = len(STATE_ORDER)
STATE_INDEX = {state: idx for idx, state in enumerate(STATE_ORDER)}
DEFAULT_STATE_IDX = STATE_INDEX["S7"]


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


def row_normalize(counts: np.ndarray, empty_value: float) -> np.ndarray:
    """Row-normalize stacked count matrices (..., K, K); empty rows take empty_value."""
    row_totals = counts.sum(axis=-1, keepdims=True)
    safe_totals = np.where(row_totals > 0, row_totals, 1.0)
    return np.where(row_totals > 0, counts / safe_totals, empty_value)


def wallet_bootstrap_default_ci(
    subset: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """Wallet-level (cluster) bootstrap of the transition probabilities into S7.

    user_ids are resampled with replacement through multinomial weights, so
    every replicate keeps complete wallet paths and their serial dependence.
    Returns point estimates and percentile intervals for the one-step
    probability P[i, S7] and the horizon PD [P^horizon_days][i, S7].
    """
    if subset.empty:
        raise ValueError("Cannot bootstrap an empty transition subset.")

    wallet_counts = wallet_count_matrix(subset)
    n_wallets = wallet_counts.shape[0]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        n_wallets, np.full(n_wallets, 1.0 / n_wallets), size=n_bootstrap
    ).astype(np.float64)
    boot_counts = (weights @ wallet_counts).reshape(n_bootstrap, N_STATES, N_STATES)
    point_counts = wallet_counts.sum(axis=0).reshape(N_STATES, N_STATES)

    point_daily = row_normalize(point_counts, empty_value=np.nan)
    boot_daily = row_normalize(boot_counts, empty_value=np.nan)
    point_horizon = np.linalg.matrix_power(np.nan_to_num(point_daily, nan=0.0), horizon_days)
    boot_horizon = np.linalg.matrix_power(np.nan_to_num(boot_daily, nan=0.0), horizon_days)

    boot_pd_daily = boot_daily[:, :, DEFAULT_STATE_IDX]
    boot_pd_horizon = boot_horizon[:, :, DEFAULT_STATE_IDX]
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
            "pd_daily_point": point_daily[:, DEFAULT_STATE_IDX],
            "pd_daily_ci_low": daily_low,
            "pd_daily_ci_high": daily_high,
            "pd_daily_boot_se": daily_se,
            f"pd_{horizon_days}d_point": point_horizon[:, DEFAULT_STATE_IDX],
            f"pd_{horizon_days}d_ci_low": horizon_low,
            f"pd_{horizon_days}d_ci_high": horizon_high,
        },
        index=pd.Index(STATE_ORDER, name="current_state"),
    )


bootstrap_default_ci: dict[tuple[str, str], pd.DataFrame] = {}

for scenario_key, scenario_label in SCENARIO_KEYS:
    for profile in PROFILE_KEYS:
        subset = subset_transitions(markov_transitions_df, scenario_key, profile)
        ci_table = wallet_bootstrap_default_ci(subset)
        bootstrap_default_ci[(scenario_label, profile)] = ci_table

        reference_pd = empirical_P[(scenario_label, profile)]["S7"].to_numpy()
        if not np.allclose(np.nan_to_num(ci_table["pd_daily_point"].to_numpy()), reference_pd):
            raise AssertionError(
                f"Bootstrap point estimates do not match Section 16 for {scenario_label} | {profile}."
            )

        print(
            f"\n{scenario_label} | {PROFILE_TITLES[profile]}  "
            f"(wallets={subset['user_id'].nunique():,}, B={N_BOOTSTRAP}, "
            f"CI={CI_PERCENTILES[0]}%-{CI_PERCENTILES[1]}%)"
        )
        display(ci_table.round(6))

# %% [markdown]
# ### 17.3 Time-homogeneity check
#
# The $\mathbf{P}^{30}$ projection assumes the daily matrix is constant within the 30-day horizon. Transitions are split by the date of day $t$: days 1-15 of the scenario window versus days 16-30. $\mathbf{P}$ is estimated on each half and shown side by side, together with the difference (days 16-30 minus days 1-15).
#
# For each transient origin state $i$ (`S2` to `S6`) the Anderson-Goodman likelihood-ratio statistic compares the two halves:
#
# $$G^2_i = 2 \sum_{h=1}^{2} \sum_{j} n_{hij} \ln \frac{n_{hij}}{\hat{E}_{hij}}, \qquad \hat{E}_{hij} = \frac{n_{hi\cdot}\, n_{\cdot ij}}{n_{\cdot i \cdot}}$$
#
# with $K_i - 1$ degrees of freedom, where $K_i$ is the number of next states observed from $i$. A pooled statistic sums $G^2_i$ and the degrees of freedom across the testable transient rows.
#
# `S1` and `S7` are excluded from the test: they are absorbing by construction, and their counts grow in the second half because frozen wallets accumulate. Within-wallet dependence makes the chi-square reference approximate (p-values tend to be too small on large samples), so the test is read together with the size of the differences (`max_abs_row_diff`).

# %%
from scipy.stats import chi2

HALF_WINDOW_DAYS = 15
TRANSIENT_STATES = ["S2", "S3", "S4", "S5", "S6"]
HOMOGENEITY_ALPHA = 0.05
FIRST_HALF, SECOND_HALF = "days_01_15", "days_16_30"

scenario_window_start = {
    key: pd.to_datetime(payload["start_datetime_utc"], utc=True).normalize()
    for key, payload in payloads.items()
}


def label_window_half(transitions: pd.DataFrame) -> pd.Series:
    """Label each transition by the half of the scenario window that contains day t."""
    split_date = transitions["scenario"].map(scenario_window_start) + pd.to_timedelta(
        HALF_WINDOW_DAYS, unit="D"
    )
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
        n_first, n_second = table.sum(axis=1)
        dof = table.shape[1] - 1
        if dof < 1 or n_first == 0 or n_second == 0:
            g_stat, p_value, dof = np.nan, np.nan, 0
        else:
            expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
            ratio = np.where(table > 0, table / expected, 1.0)
            g_stat = float(2.0 * np.sum(table * np.log(ratio)))
            p_value = float(chi2.sf(g_stat, dof))
        records.append(
            {
                "current_state": origin,
                f"n_{FIRST_HALF}": int(n_first),
                f"n_{SECOND_HALF}": int(n_second),
                f"pd_daily_{FIRST_HALF}": p_first.loc[origin, "S7"],
                f"pd_daily_{SECOND_HALF}": p_second.loc[origin, "S7"],
                "max_abs_row_diff": float((p_second.loc[origin] - p_first.loc[origin]).abs().max()),
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
            "p_value": [float(chi2.sf(pooled_g, pooled_dof)) if pooled_dof > 0 else np.nan],
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


transitions_by_half = markov_transitions_df.assign(
    window_half=label_window_half(markov_transitions_df)
)
homogeneity_P: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
homogeneity_tests: dict[tuple[str, str], pd.DataFrame] = {}

for scenario_key, scenario_label in SCENARIO_KEYS:
    for profile in PROFILE_KEYS:
        subset = subset_transitions(transitions_by_half, scenario_key, profile)
        n_first, p_first = build_empirical_transition_matrices(
            subset.loc[subset["window_half"].eq(FIRST_HALF)]
        )
        n_second, p_second = build_empirical_transition_matrices(
            subset.loc[subset["window_half"].eq(SECOND_HALF)]
        )
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
            sns.heatmap(
                matrix,
                ax=ax,
                annot=True,
                fmt=".4f",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.4,
            )
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
# `compute_transition_matrix(hf_panel, bins, labels)` rebuilds the chain from the observed pure-wallet Health Factor panel (Section 12, before censoring) for any set of cut points:
#
# 1. Map HF to states with left-closed bins `[a, b)` and force `S7` for `HF <= 1.0`. The liquidation threshold is a protocol rule, not a modelling choice, so it is never shifted.
# 2. Re-apply the right-censoring rules of Section 14: S1 dropout, first S7 hit, forward-filled absorption to the window end.
# 3. Return $\mathbf{P}$ for each regime and profile.
#
# The cut points that bound the intermediate states `S4`, `S5`, `S6` (1.10, 1.25, 1.50) are shifted by $-0.05$ and $+0.05$; the default boundary 1.0 and the `S3`/`S2` boundary 2.0 stay fixed. Because a shifted label covers a different HF band, the summary also reports the 30-day PD of the **performing portfolio**: the distribution of each wallet's first observed state over `S2` to `S6`, multiplied by $[\mathbf{P}^{30}]_{\cdot,S7}$. Figures that move little across the three runs indicate that the PD is not an artefact of the chosen boundaries.
#
# A sanity check first confirms that the base cut points reproduce the Section 16 matrices.

# %%
DEFAULT_HF_THRESHOLD = 1.0
SENSITIVITY_SHIFT = 0.05
INTERMEDIATE_BOUNDARIES = (1.10, 1.25, 1.50)
WALLET_KEYS = ["scenario", "user_id"]


def map_hf_to_states(
    health_factor: pd.Series, bins: list[float], labels: list[str]
) -> pd.Series:
    """Discretize HF with left-closed bins [a, b); HF <= 1.0 is always S7."""
    edges = np.asarray(bins, dtype=float)
    if len(labels) != len(edges) - 1:
        raise ValueError(
            f"Expected {len(edges) - 1} labels for {len(edges)} bin edges, got {len(labels)}."
        )
    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"Bin edges must be strictly increasing: {list(bins)}")
    unknown_labels = set(labels) - set(STATE_ORDER)
    if unknown_labels:
        raise ValueError(f"Labels outside STATE_ORDER: {sorted(unknown_labels)}")

    states = pd.cut(health_factor, bins=edges, labels=labels, right=False).astype("object")
    states = states.mask(health_factor.le(DEFAULT_HF_THRESHOLD), "S7")
    n_unmapped = int(states.isna().sum())
    if n_unmapped:
        raise ValueError(f"{n_unmapped:,} Health Factor values fall outside the bin edges.")
    return states.rename("state")


def apply_right_censoring(
    observed_panel: pd.DataFrame, window_end_by_scenario: dict[str, pd.Timestamp]
) -> pd.DataFrame:
    """Expand wallets to daily paths and freeze the first absorbing state (Section 14 rules)."""
    observed = (
        observed_panel[WALLET_KEYS + ["profile", "date", "health_factor", "state"]]
        .assign(date=lambda df: df["date"].dt.normalize())
        .sort_values(WALLET_KEYS + ["date"], kind="mergesort")
        .drop_duplicates(WALLET_KEYS + ["date"], keep="last")
    )
    observed["window_end_date"] = observed["scenario"].map(window_end_by_scenario)

    span = (
        observed.groupby(WALLET_KEYS, sort=False)
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            window_end_date=("window_end_date", "first"),
            profile=("profile", "first"),
            last_health_factor=("health_factor", "last"),
        )
        .reset_index()
    )

    s7_events = (
        observed.loc[observed["health_factor"].le(DEFAULT_HF_THRESHOLD), WALLET_KEYS + ["date"]]
        .groupby(WALLET_KEYS, sort=False, as_index=False)["date"]
        .min()
        .rename(columns={"date": "event_date"})
        .assign(absorbing_state="S7", tie_rank=0)
    )
    is_dropout = span["last_date"].lt(span["window_end_date"]) & span["last_health_factor"].gt(
        DEFAULT_HF_THRESHOLD
    )
    s1_events = span.loc[is_dropout, WALLET_KEYS].assign(
        event_date=span.loc[is_dropout, "last_date"] + pd.to_timedelta(1, unit="D"),
        absorbing_state="S1",
        tie_rank=1,
    )
    first_events = (
        pd.concat([s7_events, s1_events], ignore_index=True)
        .sort_values(WALLET_KEYS + ["event_date", "tie_rank"], kind="mergesort")
        .drop_duplicates(WALLET_KEYS, keep="first")[WALLET_KEYS + ["event_date", "absorbing_state"]]
    )

    span["n_days"] = (span["window_end_date"] - span["first_date"]).dt.days + 1
    span = span.loc[span["n_days"] > 0]
    grid = span.loc[
        span.index.repeat(span["n_days"]), WALLET_KEYS + ["profile", "first_date", "window_end_date"]
    ].reset_index(drop=True)
    grid["date"] = grid["first_date"] + pd.to_timedelta(
        grid.groupby(WALLET_KEYS, sort=False).cumcount(), unit="D"
    )
    grid = (
        grid.merge(observed[WALLET_KEYS + ["date", "state"]], on=WALLET_KEYS + ["date"], how="left")
        .merge(first_events, on=WALLET_KEYS, how="left")
        .sort_values(WALLET_KEYS + ["date"], kind="mergesort")
        .reset_index(drop=True)
    )

    wallet_groups = [grid["scenario"], grid["user_id"]]
    state_observed = grid.groupby(WALLET_KEYS, sort=False)["state"].ffill()
    absorbing_hit = grid["absorbing_state"].where(grid["date"].eq(grid["event_date"]))
    absorbing_frozen = absorbing_hit.groupby(wallet_groups, sort=False).ffill()
    grid["state"] = absorbing_frozen.fillna(state_observed)
    grid["next_state"] = grid.groupby(WALLET_KEYS, sort=False)["state"].shift(-1)

    is_valid = (
        grid["date"].lt(grid["window_end_date"])
        & grid["state"].notna()
        & grid["next_state"].notna()
    )
    return grid.loc[is_valid, WALLET_KEYS + ["profile", "date", "state", "next_state"]].reset_index(
        drop=True
    )


def compute_transition_matrix(
    hf_panel: pd.DataFrame,
    bins: list[float],
    labels: list[str],
    window_end_by_scenario: dict[str, pd.Timestamp] | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Recompute the right-censored empirical P for custom Health Factor boundaries.

    Returns one row-stochastic matrix per (scenario_label, profile).
    """
    window_end_by_scenario = window_end_by_scenario or scenario_window_end
    panel = hf_panel.assign(state=map_hf_to_states(hf_panel["health_factor"], bins, labels))
    transitions = apply_right_censoring(panel, window_end_by_scenario)

    matrices: dict[tuple[str, str], pd.DataFrame] = {}
    for scenario_key, scenario_label in SCENARIO_KEYS:
        for profile in PROFILE_KEYS:
            _, transition_p = build_empirical_transition_matrices(
                subset_transitions(transitions, scenario_key, profile)
            )
            matrices[(scenario_label, profile)] = transition_p
    return matrices


def shift_intermediate_boundaries(
    bins: list[float], shift: float, boundaries: tuple[float, ...] = INTERMEDIATE_BOUNDARIES
) -> list[float]:
    """Shift only the selected cut points; every other edge is left unchanged."""
    return [
        edge + shift if any(np.isclose(edge, target) for target in boundaries) else edge
        for edge in bins
    ]


def horizon_pd(transition_p: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> pd.Series:
    """Cumulative probability of reaching S7 within horizon_days, by origin state."""
    power = np.linalg.matrix_power(transition_p.to_numpy(), horizon_days)
    return pd.Series(power[:, DEFAULT_STATE_IDX], index=transition_p.index)


def performing_portfolio_pd(
    hf_panel: pd.DataFrame,
    bins: list[float],
    labels: list[str],
    transition_p: pd.DataFrame,
    scenario_key: str,
    profile: str,
    horizon_days: int = HORIZON_DAYS,
) -> float:
    """30-day PD of wallets whose first observed state is transient (S2 to S6)."""
    segment = subset_transitions(hf_panel, scenario_key, profile)
    first_obs = segment.sort_values("date", kind="mergesort").drop_duplicates("user_id", keep="first")
    initial_states = map_hf_to_states(first_obs["health_factor"], bins, labels)
    weights = initial_states.value_counts().reindex(TRANSIENT_STATES, fill_value=0).astype(float)
    if weights.sum() == 0:
        return np.nan
    weights = weights / weights.sum()
    return float((weights * horizon_pd(transition_p, horizon_days).loc[TRANSIENT_STATES]).sum())


pure_hf_panel = (
    hf_panel_df.merge(pure_wallets, on=WALLET_KEYS, how="inner")
    .assign(profile=lambda df: df["wallet_profile"])
    .drop(columns=["wallet_profile"])
)

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

portfolio_records = []
for scenario_key, scenario_label in SCENARIO_KEYS:
    for profile in PROFILE_KEYS:
        key = (scenario_label, profile)
        daily_pd = pd.DataFrame({run: sensitivity_P[run][key]["S7"] for run in sensitivity_bins})
        cumulative_pd = pd.DataFrame({run: horizon_pd(sensitivity_P[run][key]) for run in sensitivity_bins})
        sensitivity_table = pd.concat(
            {"pd_daily": daily_pd, f"pd_{HORIZON_DAYS}d": cumulative_pd}, axis=1
        )
        sensitivity_table[("max_abs_delta_vs_base", "pd_daily")] = (
            daily_pd.sub(daily_pd["base"], axis=0).abs().max(axis=1)
        )
        sensitivity_table[("max_abs_delta_vs_base", f"pd_{HORIZON_DAYS}d")] = (
            cumulative_pd.sub(cumulative_pd["base"], axis=0).abs().max(axis=1)
        )
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
# ## 18. Manual export of $\mathbf{P}^{30}$ (not for production)
#
# The full 30-step matrix $\mathbf{P}^{30}$ is computed from the Section 16 daily matrices `empirical_P` for every market regime (`baseline_mar2024`, `stress_aug2024`) and collateral profile (A, B), and written to `data/02_interim/` in long format: one row per `(scenario, profile, current_state)` and one column per next state `S1` to `S7`.
#
# Before writing, each matrix is checked for finite, non-negative entries and row mass (rows sum to 1 whenever the daily matrix is row-stochastic), and its `S7` column is matched against `horizon_pd` from Section 17.4. The file is read back after writing to confirm a lossless round trip. The file name marks it as a manual notebook export so it cannot be mistaken for an output of the production pipeline.

# %%
INTERIM_DATA_DIR = Path("../data/02_interim")
P30_EXPORT_PATH = INTERIM_DATA_DIR / "notebook_export_markov_p30_not_for_prod.csv"
STOCHASTIC_TOLERANCE = 1e-9
EXPORT_KEY_COLUMNS = ["scenario", "profile", "horizon_days", "current_state"]


def compute_horizon_matrix(
    transition_p: pd.DataFrame, horizon_days: int = HORIZON_DAYS
) -> pd.DataFrame:
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
    horizon_p: pd.DataFrame,
    daily_p: pd.DataFrame,
    segment: str,
    horizon_days: int = HORIZON_DAYS,
    tolerance: float = STOCHASTIC_TOLERANCE,
) -> None:
    """Raise if P^horizon_days is not a valid (sub-)stochastic matrix consistent with Section 17.4."""
    values = horizon_p.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{segment}: P^{horizon_days} contains non-finite values.")
    if (values < -tolerance).any():
        raise ValueError(f"{segment}: P^{horizon_days} contains negative probabilities.")

    row_mass = values.sum(axis=1)
    if (row_mass > 1.0 + tolerance).any():
        raise ValueError(f"{segment}: P^{horizon_days} has rows with mass above 1.")
    daily_mass = daily_p.to_numpy(dtype=float).sum(axis=1)
    if np.allclose(daily_mass, 1.0, atol=tolerance) and not np.allclose(row_mass, 1.0, atol=tolerance):
        raise ValueError(
            f"{segment}: P^{horizon_days} rows do not sum to 1 although the daily P is row-stochastic."
        )

    reference_pd = horizon_pd(daily_p, horizon_days).to_numpy(dtype=float)
    if not np.allclose(values[:, DEFAULT_STATE_IDX], reference_pd, atol=tolerance):
        raise AssertionError(f"{segment}: S7 column does not match horizon_pd from Section 17.4.")


def build_horizon_export_table(
    daily_matrices: dict[tuple[str, str], pd.DataFrame],
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """Stack the validated P^horizon_days of every regime and profile into one long table."""
    expected_keys = [
        (scenario_label, profile)
        for _, scenario_label in SCENARIO_KEYS
        for profile in PROFILE_KEYS
    ]
    missing_keys = [key for key in expected_keys if key not in daily_matrices]
    if missing_keys:
        raise KeyError(f"Daily transition matrices missing for segments: {missing_keys}")

    blocks = []
    for scenario_label, profile in expected_keys:
        daily_p = daily_matrices[(scenario_label, profile)]
        horizon_p = compute_horizon_matrix(daily_p, horizon_days)
        validate_horizon_matrix(horizon_p, daily_p, f"{scenario_label} | {profile}", horizon_days)

        block = horizon_p.reset_index()
        block.insert(0, "horizon_days", horizon_days)
        block.insert(0, "profile", profile)
        block.insert(0, "scenario", scenario_label)
        blocks.append(block)

    return pd.concat(blocks, ignore_index=True)[EXPORT_KEY_COLUMNS + STATE_ORDER]


def export_horizon_table(table: pd.DataFrame, output_path: Path) -> None:
    """Write the table to CSV and verify the file reloads without loss."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table.to_csv(output_path, index=False)
        reloaded = pd.read_csv(output_path, float_precision="round_trip")
    except OSError as exc:
        raise OSError(f"Failed to write or reload {output_path.resolve()}: {exc}") from exc

    if list(reloaded.columns) != list(table.columns) or len(reloaded) != len(table):
        raise IOError(f"Reloaded file shape or columns differ from the exported table: {output_path}")
    keys_match = (
        reloaded[EXPORT_KEY_COLUMNS].astype(str).to_numpy()
        == table[EXPORT_KEY_COLUMNS].astype(str).to_numpy()
    ).all()
    values_match = np.allclose(
        reloaded[STATE_ORDER].to_numpy(dtype=float),
        table[STATE_ORDER].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-15,
    )
    if not (keys_match and values_match):
        raise IOError(f"Round-trip check failed for {output_path}")


p30_export_df = build_horizon_export_table(empirical_P, HORIZON_DAYS)
export_horizon_table(p30_export_df, P30_EXPORT_PATH)

print(f"P^{HORIZON_DAYS} exported (manual notebook export, NOT for production):")
print(f"  path: {P30_EXPORT_PATH.resolve()}")
print(f"  rows: {len(p30_export_df)} ({p30_export_df.groupby(['scenario', 'profile']).ngroups} segments x {len(STATE_ORDER)} states)")
display(p30_export_df.round(6))

# %% [markdown]
# ## 19. Export empirical absolute transition counts $\mathbf{N}$ and initial wallet states (not for production)
#
# Two intermediate files are written to `data/02_interim/` for the downstream ECL model validation. Their key columns identify every row, so the positional index is not written.
#
# 1. **Absolute transition counts** (`notebook_export_markov_counts_N_not_for_prod.csv`): the integer matrices `absolute_N` from Section 17.1, with the same layout as the $\mathbf{P}^{30}$ export in Section 18: one row per `(scenario, profile, current_state)` and one integer column per next state `S1` to `S7`. They preserve the absolute empirical Markov frequencies behind each daily $\mathbf{P}$. Row totals are wallet-day transition counts (one per wallet and observed day, Section 14), not distinct wallets.
# 2. **Initial wallet states** (`notebook_export_markov_initial_states_not_for_prod.csv`): one row per `(scenario, profile, user_id)` with the wallet's first day in `markov_states_df` (`first_date`, UTC calendar date) and its state on that day (`initial_state`). Counting wallets by `initial_state` gives the **Initial Wallets** per state as distinct wallets. Wallets that enter after the window start keep their own `first_date`.
#
# Before writing, each count matrix is checked for the `STATE_ORDER` layout and non-negative integer counts, and its total is matched against the transitions of the same segment in `markov_transitions_df`. Every initial state must belong to `STATE_ORDER` and every wallet to a known regime and profile. Both files are read back after writing to confirm an exact round trip. The cell reuses `INTERIM_DATA_DIR` from Section 18 and `subset_transitions` from Section 17.1.

# %%
# Intermediate file: preserves the absolute empirical Markov frequencies N (the sample depth
# behind each daily P) for the downstream ECL model validation.
COUNTS_EXPORT_PATH = INTERIM_DATA_DIR / "notebook_export_markov_counts_N_not_for_prod.csv"
COUNTS_KEY_COLUMNS = ["scenario", "profile", "current_state"]
# Intermediate file: the state of every wallet on its first day in markov_states_df, consumed
# downstream by the ECL model validation to compute the Initial Wallets per state.
INITIAL_STATES_EXPORT_PATH = INTERIM_DATA_DIR / "notebook_export_markov_initial_states_not_for_prod.csv"
INITIAL_STATES_COLUMNS = ["scenario", "profile", "user_id", "first_date", "initial_state"]


def validate_count_matrix(counts_n: pd.DataFrame, expected_total: int, segment: str) -> None:
    """Raise if N is not a STATE_ORDER matrix of non-negative integer counts summing to expected_total."""
    if list(counts_n.index) != STATE_ORDER or list(counts_n.columns) != STATE_ORDER:
        raise ValueError(f"{segment}: count matrix rows and columns must follow STATE_ORDER.")
    if not all(pd.api.types.is_integer_dtype(dtype) for dtype in counts_n.dtypes):
        raise TypeError(f"{segment}: count matrix must hold integer counts.")

    values = counts_n.to_numpy()
    if (values < 0).any():
        raise ValueError(f"{segment}: count matrix contains negative counts.")
    total = int(values.sum())
    if total != expected_total:
        raise AssertionError(
            f"{segment}: count matrix total {total:,} differs from the "
            f"{expected_total:,} transitions in markov_transitions_df."
        )


def build_counts_export_table(
    count_matrices: dict[tuple[str, str], pd.DataFrame],
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten the validated N of every regime and profile into one table.

    One row per (scenario, profile, current_state) and one integer column per next
    state. Counts are kept absolute (not row-normalized): row sums are the wallet-day
    transitions observed in each origin state, not distinct wallets.
    """
    missing_keys = [
        (scenario_label, profile)
        for _, scenario_label in SCENARIO_KEYS
        for profile in PROFILE_KEYS
        if (scenario_label, profile) not in count_matrices
    ]
    if missing_keys:
        raise KeyError(f"Absolute count matrices missing for segments: {missing_keys}")

    blocks = []
    for scenario_key, scenario_label in SCENARIO_KEYS:
        for profile in PROFILE_KEYS:
            counts_n = count_matrices[(scenario_label, profile)]
            n_transitions = len(subset_transitions(transitions, scenario_key, profile))
            validate_count_matrix(counts_n, n_transitions, f"{scenario_label} | {profile}")

            block = counts_n.rename_axis(index="current_state", columns=None).reset_index()
            block.insert(0, "profile", profile)
            block.insert(0, "scenario", scenario_label)
            blocks.append(block)

    return pd.concat(blocks, ignore_index=True)[COUNTS_KEY_COLUMNS + STATE_ORDER]


def build_initial_states_table(states: pd.DataFrame) -> pd.DataFrame:
    """Return one row per wallet with its Markov state on its first day in the daily panel.

    first_date is the UTC calendar date of the wallet's first observation (Section 14)
    and initial_state its state on that day. Counting rows by initial_state gives the
    distinct wallets per state, unlike the wallet-day row sums of N.
    """
    wallet_keys = ["scenario", "user_id"]
    first_days = states.sort_values(wallet_keys + ["date"]).drop_duplicates(wallet_keys, keep="first")

    scenario_labels = dict(SCENARIO_KEYS)
    known_segment = first_days["scenario"].isin(list(scenario_labels)) & first_days["profile"].isin(PROFILE_KEYS)
    if not known_segment.all():
        raise ValueError(
            f"{int((~known_segment).sum()):,} wallets belong to a regime or profile "
            "outside SCENARIO_KEYS / PROFILE_KEYS."
        )
    valid_state = first_days["state"].isin(STATE_ORDER)
    if not valid_state.all():
        raise ValueError(f"{int((~valid_state).sum()):,} wallets have a first-day state outside STATE_ORDER.")

    table = pd.DataFrame(
        {
            "scenario": first_days["scenario"].map(scenario_labels),
            "profile": first_days["profile"],
            "user_id": first_days["user_id"],
            "first_date": first_days["date"].dt.strftime("%Y-%m-%d"),
            "initial_state": first_days["state"],
        }
    )
    return table.sort_values(["scenario", "profile", "user_id"], ignore_index=True)[INITIAL_STATES_COLUMNS]


def export_flat_table(table: pd.DataFrame, output_path: Path) -> None:
    """Write a flat table to CSV without its positional index and verify an exact round trip."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table.to_csv(output_path, index=False)
        reloaded = pd.read_csv(output_path)
    except OSError as exc:
        raise OSError(f"Failed to write or reload {output_path.resolve()}: {exc}") from exc

    try:
        pd.testing.assert_frame_equal(reloaded, table, check_dtype=False)
    except AssertionError as exc:
        raise IOError(f"Round-trip check failed for {output_path}: {exc}") from exc


counts_export_df = build_counts_export_table(absolute_N, markov_transitions_df)
export_flat_table(counts_export_df, COUNTS_EXPORT_PATH)

print("Absolute transition counts N exported (manual notebook export, NOT for production):")
print(f"  path: {COUNTS_EXPORT_PATH.resolve()}")
print(f"  rows: {len(counts_export_df)} ({counts_export_df.groupby(['scenario', 'profile']).ngroups} segments x {len(STATE_ORDER)} states)")
display(counts_export_df)

initial_states_df = build_initial_states_table(markov_states_df)
export_flat_table(initial_states_df, INITIAL_STATES_EXPORT_PATH)

initial_wallets_by_state = (
    initial_states_df.groupby(["scenario", "profile", "initial_state"])
    .size()
    .unstack(["scenario", "profile"], fill_value=0)
    .reindex(STATE_ORDER, fill_value=0)
)
print("\nInitial wallet states exported (manual notebook export, NOT for production):")
print(f"  path: {INITIAL_STATES_EXPORT_PATH.resolve()}")
print(f"  rows: {len(initial_states_df):,} wallets ({initial_states_df.groupby(['scenario', 'profile']).ngroups} segments)")
print("  Initial Wallets per state (distinct wallets by first-day state):")
display(initial_wallets_by_state)
