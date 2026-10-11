"""Validate the re-extracted raw data of both scenarios and run an early diagnostic.

Checks (per scenario, window ``[t - 30, t + 30]`` = 61 days):

1. Dates and blocks: positions, oracle prices, daily blocks, reserve parameters
   and eMode categories cover the same 61 consecutive dates and share the same
   block on each date; ``t`` maps to the block documented in ``CLAUDE.md``.
2. Coverage: every ``(date, underlying_asset)`` in positions has an oracle
   price and reserve parameters on that day.
3. Universe: users per day and entries/exits per day (a constant universe is a
   failure: it was the defect of the old panel).
4. Events: inside the window (timestamp, date and block from the event id) and
   the share of liquidated / repaying users present in the positions of the
   event day or of the day before.
5. Parameters: ``0 <= ltv <= LT < 1`` and ``0 <= LB < 0.5`` (validated by
   ``src.utils.protocol_params``); LT/LB of ten reference assets at ``t``.
6. Binance daily prices cover the 730 days that end at ``t`` (EVT input).

Section "diagnostic" (APPROXIMATE, not the thesis methodology): on days
``t - 30`` and ``t``, positions are valued at the Aave oracle price and a Health
Factor is computed per user (see ``raw_data_common``). Profile A (>= 85% of the
collateral in ETH and ETH LSTs) and profile B (>= 85% in UNI + LINK), both with
stablecoin-only debt, are counted and their HF is binned by state.

The report is printed and written to ``<raw-dir>/<folder>/raw_data_check.json``.
Exit code: 1 if any hard check fails, else 0.

Usage, from the repository root::

    venv/Scripts/python.exe scripts/check_raw_data.py [--scenario baseline|stress|all]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from raw_data_common import (  # noqa: E402
    BINANCE_FILE,
    BLOCKS_FILE,
    DAILY_DIR,
    EVENTS_DIR,
    HF_STATE_LABELS,
    LIQUIDATIONS_FILE,
    METADATA_FILE,
    ORACLE_FILE,
    PRICES_DIR,
    PROTOCOL_DIR,
    RAW_DIR,
    REPAYS_FILE,
    SCENARIOS,
    WINDOW_HALF_DAYS,
    Scenario,
    day_prices,
    hf_state,
    positions_file,
    read_positions,
    shift_day,
    utc_midnight_ts,
    value_positions,
    window_dates,
)

from src.utils.protocol_params import load_emode_categories, load_reserve_params  # noqa: E402

logger = logging.getLogger(__name__)

REPORT_FILE = "raw_data_check.json"
REFERENCE_SYMBOLS = ["WETH", "wstETH", "rETH", "cbETH", "WBTC", "UNI", "LINK", "USDC", "USDT", "DAI"]
BINANCE_TICKERS = ["BTCUSDT", "ETHUSDT", "UNIUSDT", "LINKUSDT"]
EVT_DAYS = 730
BLOCK_TIMESTAMP_MAX_LAG_SECONDS = 120
COVERAGE_WARN_THRESHOLD = 0.90
DUST_REFERENCE_USD = 100.0  # only to tell dust from material positions among HF <= 1 wallets
MAX_EXAMPLES = 5
OLD_PANEL_REFERENCE = "OLD invalid panel, baseline day 1: ~8,000 A wallets, ~590 B wallets, 203 wallets with HF <= 1"
DIAGNOSTIC_NOTE = (
    "APPROXIMATE diagnostic, not the thesis methodology: oracle-priced balances, HF from reserve/eMode LT, "
    "no isolation mode, caps or dust rules."
)
_FILE_DATE = re.compile(r"positions_(\d{4}-\d{2}-\d{2})\.csv\.gz")


class Checks:
    """Ordered list of named checks; ``hard`` failures drive the exit code."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: str, hard: bool = True) -> bool:
        status = "PASS" if ok else ("FAIL" if hard else "WARN")
        self.items.append({"name": name, "status": status, "hard": hard, "detail": detail})
        return ok

    def info(self, name: str, detail: str) -> None:
        self.items.append({"name": name, "status": "INFO", "hard": False, "detail": detail})

    @property
    def hard_failures(self) -> int:
        return sum(1 for item in self.items if item["status"] == "FAIL")


def _examples(values: list[Any]) -> str:
    shown = ", ".join(str(value) for value in values[:MAX_EXAMPLES])
    return shown + (f", ... (+{len(values) - MAX_EXAMPLES})" if len(values) > MAX_EXAMPLES else "")


def _stats(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype="float64")
    return {"min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max())}


def _load(checks: Checks, name: str, loader: Callable[[], Any]) -> Any:
    """Run ``loader``; record a hard failure and return ``None`` if it raises."""
    try:
        return loader()
    except (OSError, ValueError, KeyError) as exc:
        message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        checks.add(name, False, f"cannot load: {message}")
        return None


def _dates_match(found: list[str], expected: list[str]) -> str:
    """Return '' if ``found`` equals ``expected``, else a short description."""
    if found == expected:
        return ""
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    return f"{len(found)} dates; missing {_examples(missing)}; unexpected {_examples(extra)}"


def _blocks_by_date(frame: pd.DataFrame) -> dict[str, set[int]]:
    return {day: set(group) for day, group in frame.groupby("date")["block"]}


def _block_alignment(frame: pd.DataFrame, block_of: dict[str, int]) -> list[str]:
    """Dates where the frame's blocks differ from the daily-blocks block."""
    bad = []
    for day, found in _blocks_by_date(frame).items():
        if found != {block_of.get(day)}:
            bad.append(f"{day}:{sorted(found)}")
    return bad


def check_protocol_and_prices(
    sc: Scenario, folder: Path, expected: list[str], checks: Checks
) -> dict[str, Any] | None:
    """Checks 1 (blocks/dates) and 5 (parameters) for the non-position files."""
    blocks = _load(
        checks,
        "daily_blocks_load",
        lambda: pd.read_csv(folder / PROTOCOL_DIR / BLOCKS_FILE, dtype={"date": str, "timestamp": "int64", "block": "int64"}),
    )
    oracle = _load(
        checks,
        "oracle_load",
        lambda: pd.read_csv(
            folder / PRICES_DIR / ORACLE_FILE,
            usecols=["date", "block", "symbol", "underlying_asset", "price_usd"],
            dtype={"date": str, "block": "int64", "symbol": str, "underlying_asset": str, "price_usd": "float64"},
        ),
    )
    reserves = _load(checks, "reserve_params_valid", lambda: load_reserve_params(sc.folder, raw_dir=folder.parent))
    emodes = _load(checks, "emode_params_valid", lambda: load_emode_categories(sc.folder, raw_dir=folder.parent))
    binance = _load(
        checks,
        "binance_load",
        lambda: pd.read_csv(folder / PRICES_DIR / BINANCE_FILE, usecols=["date", "ticker", "close"], dtype={"date": str, "ticker": str}),
    )
    if blocks is None or oracle is None or reserves is None or emodes is None:
        return None

    # --- daily blocks
    msg = _dates_match(list(blocks["date"]), expected)
    checks.add("dates_daily_blocks", not msg, msg or f"{len(blocks)} consecutive dates {expected[0]}..{expected[-1]}")
    block_of = dict(zip(blocks["date"], blocks["block"]))
    increasing = bool((blocks["block"].diff().dropna() > 0).all())
    checks.add("daily_blocks_increasing", increasing, "blocks strictly increasing" if increasing else "blocks not increasing")
    lag = blocks["timestamp"].to_numpy() - np.array([utc_midnight_ts(day) for day in blocks["date"]])
    lag_ok = bool(((lag >= 0) & (lag < BLOCK_TIMESTAMP_MAX_LAG_SECONDS)).all())
    checks.add(
        "daily_blocks_midnight",
        lag_ok,
        f"block timestamp lag after 00:00 UTC: min {lag.min()}s, max {lag.max()}s",
        hard=False,
    )
    t_block = block_of.get(sc.t)
    checks.add("t_block", t_block == sc.t_block, f"t={sc.t} -> block {t_block} (documented {sc.t_block})")

    # --- oracle
    msg = _dates_match(sorted(oracle["date"].unique()), expected)
    checks.add("dates_oracle", not msg, msg or "61 consecutive dates")
    bad = _block_alignment(oracle, block_of)
    checks.add("blocks_oracle", not bad, "aligned with daily_blocks" if not bad else f"misaligned on {_examples(bad)}")
    duplicated = int(oracle.duplicated(["date", "underlying_asset"]).sum())
    checks.add("oracle_unique", duplicated == 0, f"{duplicated} duplicated (date, asset) rows")
    prices_ok = bool(np.isfinite(oracle["price_usd"]).all() and (oracle["price_usd"] > 0).all())
    checks.add("oracle_prices_positive", prices_ok, "all price_usd finite and > 0" if prices_ok else "non-positive or non-finite price_usd")
    per_day = oracle.groupby("date").size()
    checks.info("oracle_reserves_per_day", f"{per_day.min()}..{per_day.max()} reserves per day")

    # --- reserve params and eMode categories
    msg = _dates_match(sorted(reserves["date"].unique()), expected)
    checks.add("dates_params", not msg, msg or "61 consecutive dates")
    bad = _block_alignment(reserves, block_of)
    checks.add("blocks_params", not bad, "aligned with daily_blocks" if not bad else f"misaligned on {_examples(bad)}")
    msg = _dates_match(sorted(emodes["date"].unique()), expected)
    checks.add("dates_emode", not msg, msg or "61 consecutive dates")
    bad = _block_alignment(emodes, block_of)
    checks.add("blocks_emode", not bad, "aligned with daily_blocks" if not bad else f"misaligned on {_examples(bad)}")
    oracle_sets = oracle.groupby("date")["underlying_asset"].agg(frozenset)
    params_sets = reserves.groupby("date")["underlying_asset"].agg(frozenset)
    diff_days = [day for day in expected if oracle_sets.get(day) != params_sets.get(day)]
    checks.add(
        "oracle_params_same_reserves",
        not diff_days,
        "same reserve set on every day" if not diff_days else f"reserve sets differ on {_examples(diff_days)}",
    )
    emode_days = emodes.groupby("date")["emode_id"].agg(frozenset)
    orphan = [
        day
        for day, ids in reserves.loc[reserves["emode_id"] > 0].groupby("date")["emode_id"].agg(frozenset).items()
        if not ids <= emode_days.get(day, frozenset())
    ]
    checks.add("reserve_emode_in_categories", not orphan, "every reserve emode_id has a category row" if not orphan else f"orphans on {_examples(orphan)}")
    checks.add(
        "params_plausible",
        True,
        f"validated by protocol_params: 0<=ltv<=LT<1, 0<=LB<0.5 ({len(reserves)} reserve rows, {len(emodes)} eMode rows)",
    )

    # --- Binance (EVT input)
    if binance is not None:
        first_needed = shift_day(sc.t, -(EVT_DAYS - 1))
        need = set(pd.date_range(first_needed, sc.t).strftime("%Y-%m-%d"))
        gaps = {ticker: len(need - set(binance.loc[binance["ticker"] == ticker, "date"])) for ticker in BINANCE_TICKERS}
        close_ok = bool((binance["close"] > 0).all())
        checks.add(
            "binance_covers_evt_window",
            all(v == 0 for v in gaps.values()) and close_ok,
            f"{EVT_DAYS} days {first_needed}..{sc.t} for {BINANCE_TICKERS}; missing days {gaps}",
        )

    return {"blocks": block_of, "oracle": oracle, "reserves": reserves, "emodes": emodes}


def check_positions(
    sc: Scenario, folder: Path, expected: list[str], loaded: dict[str, Any], checks: Checks
) -> dict[str, Any]:
    """Checks 1-3 on the daily positions, plus the diagnostic valuation on two days."""
    block_of: dict[str, int] = loaded["blocks"]
    oracle: pd.DataFrame = loaded["oracle"]
    reserves: pd.DataFrame = loaded["reserves"]
    emodes: pd.DataFrame = loaded["emodes"]

    daily_dir = folder / DAILY_DIR
    found_days = sorted(m.group(1) for p in daily_dir.glob("positions_*.csv.gz") if (m := _FILE_DATE.fullmatch(p.name)))
    msg = _dates_match(found_days, expected)
    checks.add("dates_positions", not msg, msg or "61 consecutive daily files")

    oracle_assets = {day: set(group) for day, group in oracle.groupby("date")["underlying_asset"]}
    params_assets = {day: set(group) for day, group in reserves.groupby("date")["underlying_asset"]}
    diag_days = {shift_day(sc.t, -WINDOW_HALF_DAYS): "t-30", sc.t: "t"}

    users_by_day: dict[str, set[str]] = {}
    rows_by_day: dict[str, int] = {}
    block_problems: list[str] = []
    file_problems: list[str] = []
    no_price: list[str] = []
    no_params: list[str] = []
    meta_problems: list[str] = []
    diag_frames: dict[str, pd.DataFrame] = {}

    try:
        metadata = json.loads((daily_dir / METADATA_FILE).read_text(encoding="utf-8")).get("days", {})
    except (OSError, ValueError):
        metadata = {}

    for day in found_days:
        if day not in block_of:
            continue
        try:
            pos = read_positions(positions_file(folder, day))
        except (OSError, ValueError) as exc:
            file_problems.append(f"{day}: unreadable ({str(exc).splitlines()[0]})")
            continue

        users_by_day[day] = set(pos["user_id"])
        rows_by_day[day] = len(pos)
        if set(pos["date"]) != {day}:
            file_problems.append(f"{day}: date column {sorted(set(pos['date']))[:3]}")
        if set(pos["block"]) != {block_of[day]}:
            block_problems.append(f"{day}:{sorted(set(pos['block']))[:3]}")
        duplicated = int(pos.duplicated(["user_id", "underlying_asset"]).sum())
        numbers = pos[["a_token_balance", "variable_debt", "stable_debt"]].to_numpy()
        if duplicated:
            file_problems.append(f"{day}: {duplicated} duplicated (user, asset) rows")
        if not np.isfinite(numbers).all() or (numbers < 0).any():
            file_problems.append(f"{day}: negative or non-finite balances")

        assets = set(pos["underlying_asset"])
        absent = sorted(assets - oracle_assets.get(day, set()))
        if absent:
            no_price.append(f"{day}:{absent[:2]}")
        absent = sorted(assets - params_assets.get(day, set()))
        if absent:
            no_params.append(f"{day}:{absent[:2]}")

        meta = metadata.get(day)
        if meta is not None and (meta.get("block") != block_of[day] or meta.get("rows") != len(pos) or meta.get("users") != len(users_by_day[day])):
            meta_problems.append(f"{day}: metadata block/rows/users {meta.get('block')}/{meta.get('rows')}/{meta.get('users')} vs {block_of[day]}/{len(pos)}/{len(users_by_day[day])}")

        if day in diag_days:
            diag_frames[day] = value_positions(
                pos,
                day_prices(oracle, day),
                reserves.loc[reserves["date"] == day],
                emodes.loc[emodes["date"] == day],
            )

    checks.add("blocks_positions", not block_problems, "aligned with daily_blocks" if not block_problems else f"misaligned on {_examples(block_problems)}")
    checks.add("positions_files_sane", not file_problems, "dates, uniqueness and balances OK" if not file_problems else _examples(file_problems))
    checks.add("positions_assets_have_price", not no_price, "every (date, asset) has an oracle price" if not no_price else f"missing on {_examples(no_price)}")
    checks.add("positions_assets_have_params", not no_params, "every (date, asset) has reserve params" if not no_params else f"missing on {_examples(no_params)}")
    checks.add("positions_metadata_consistent", not meta_problems, "metadata.json matches the files" if not meta_problems else _examples(meta_problems), hard=False)

    # --- universe
    days = [day for day in expected if day in users_by_day]
    counts = [len(users_by_day[day]) for day in days]
    entries = [len(users_by_day[b] - users_by_day[a]) for a, b in zip(days, days[1:])]
    exits = [len(users_by_day[a] - users_by_day[b]) for a, b in zip(days, days[1:])]
    universe: dict[str, Any] = {}
    if counts:
        universe = {
            "users_per_day": _stats(counts),
            "entries_per_day": _stats(entries) if entries else None,
            "exits_per_day": _stats(exits) if exits else None,
            "rows_per_day": _stats(list(rows_by_day.values())),
            "distinct_users_in_window": len(set().union(*users_by_day.values())),
        }
        varies = len(set(counts)) > 1 and sum(entries) > 0 and sum(exits) > 0
        checks.add(
            "universe_varies",
            varies,
            f"users/day min/median/max {universe['users_per_day']['min']:.0f}/{universe['users_per_day']['median']:.0f}/"
            f"{universe['users_per_day']['max']:.0f}; entries/day {_fmt_stats(universe['entries_per_day'])}; "
            f"exits/day {_fmt_stats(universe['exits_per_day'])}; {universe['distinct_users_in_window']} distinct users",
        )
    return {"users_by_day": users_by_day, "universe": universe, "diag_frames": diag_frames, "diag_days": diag_days}


def _fmt_stats(stats: dict[str, float] | None) -> str:
    if not stats:
        return "n/a"
    return f"{stats['min']:.0f}/{stats['median']:.0f}/{stats['max']:.0f}"


def check_events(
    sc: Scenario, folder: Path, expected: list[str], blocks: dict[str, int], users_by_day: dict[str, set[str]], checks: Checks
) -> dict[str, Any]:
    """Check 4: events inside the window and presence of their users in the positions."""
    start_ts = utc_midnight_ts(sc.start)
    end_ts = utc_midnight_ts(sc.end) + 86_400
    ordered_blocks = np.array([blocks[day] for day in expected if day in blocks])
    summary: dict[str, Any] = {}

    for label, filename in (("liquidation_calls", LIQUIDATIONS_FILE), ("repays", REPAYS_FILE)):
        events = _load(
            checks,
            f"events_{label}_load",
            lambda f=filename: pd.read_csv(folder / EVENTS_DIR / f, usecols=["id", "timestamp", "date", "user_id"], dtype={"id": str, "date": str, "user_id": str, "timestamp": "int64"}),
        )
        if events is None:
            continue

        inside = (events["timestamp"] >= start_ts) & (events["timestamp"] < end_ts)
        derived = pd.to_datetime(events["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
        date_ok = derived == events["date"]
        in_window_days = events["date"].isin(expected)
        block = pd.to_numeric(events["id"].str.split(":").str[0], errors="coerce")
        day_index = np.searchsorted(ordered_blocks, block.fillna(-1).to_numpy(), side="right") - 1
        expected_index = events["date"].map({day: i for i, day in enumerate(expected)}).fillna(-2).to_numpy()
        block_ok = block.notna().to_numpy() & (day_index == expected_index)
        duplicated = int(events["id"].duplicated().sum())

        bad = int((~inside).sum() + (~date_ok).sum() + (~in_window_days).sum())
        checks.add(
            f"events_{label}_in_window",
            bad == 0 and bool(block_ok.all()) and duplicated == 0,
            f"{len(events)} events; {int((~inside).sum())} outside [start, end+1d); {int((~date_ok).sum())} date!=timestamp day; "
            f"{int((~block_ok).sum())} block (from id) outside its day; {duplicated} duplicate ids",
        )

        same = previous = missing = 0
        for day, user in zip(events["date"], events["user_id"]):
            if user in users_by_day.get(day, ()):
                same += 1
            elif day != sc.start and user in users_by_day.get(shift_day(day, -1), ()):
                previous += 1
            else:
                missing += 1
        total = max(len(events), 1)
        covered = (same + previous) / total
        summary[label] = {
            "events": int(len(events)),
            "distinct_users": int(events["user_id"].nunique()),
            "pct_in_positions_same_or_previous_day": 100 * covered,
            "pct_same_day": 100 * same / total,
            "pct_previous_day_only": 100 * previous / total,
            "pct_not_found": 100 * missing / total,
            "events_on_first_window_day": int((events["date"] == sc.start).sum()),
        }
        checks.add(
            f"events_{label}_users_in_positions",
            covered >= COVERAGE_WARN_THRESHOLD,
            f"{100 * covered:.1f}% of {len(events)} events have the user in positions on the event day ({100 * same / total:.1f}%) "
            f"or the day before (+{100 * previous / total:.1f}%); {100 * missing / total:.1f}% not found",
            hard=False,
        )
    return summary


def reference_params(reserves: pd.DataFrame, emodes: pd.DataFrame, day: str) -> dict[str, Any]:
    """LT / LB of the reference assets (and the eMode categories) at ``day``."""
    rows = reserves.loc[reserves["date"] == day].set_index("symbol")
    table: dict[str, Any] = {}
    for symbol in REFERENCE_SYMBOLS:
        if symbol in rows.index:
            row = rows.loc[symbol]
            table[symbol] = {
                "lt": float(row["liquidation_threshold"]),
                "lb": float(row["liquidation_bonus"]),
                "ltv": float(row["ltv"]),
                "emode_id": int(row["emode_id"]),
            }
        else:
            table[symbol] = None
    for _, row in emodes.loc[emodes["date"] == day].iterrows():
        table[f"eMode {int(row['emode_id'])} ({row['label']})"] = {
            "lt": float(row["liquidation_threshold"]),
            "lb": float(row["liquidation_bonus"]),
            "ltv": float(row["ltv"]),
            "emode_id": int(row["emode_id"]),
        }
    return table


def diagnostic(diag_frames: dict[str, pd.DataFrame], diag_days: dict[str, str]) -> dict[str, Any]:
    """Approximate profile counts and HF state distributions for the diagnostic days."""
    days: dict[str, Any] = {}
    for day, role in diag_days.items():
        frame = diag_frames.get(day)
        if frame is None:
            continue
        borrowers = frame.loc[frame["debt_usd"] > 0]
        entry: dict[str, Any] = {
            "role": role,
            "users": int(len(frame)),
            "borrowers_with_debt": int(len(borrowers)),
            "hf_le_1_all_borrowers": int((borrowers["hf"] <= 1).sum()),
            "hf_le_1_all_borrowers_debt_ge_100usd": int(((borrowers["hf"] <= 1) & (borrowers["debt_usd"] >= DUST_REFERENCE_USD)).sum()),
            "profile_overlap_a_and_b": int((frame["profile_a"] & frame["profile_b"]).sum()),
        }
        for name, flag in (("A", "profile_a"), ("B", "profile_b")):
            sub = frame.loc[frame[flag]]
            states = np.bincount(hf_state(sub["hf"].to_numpy()), minlength=len(HF_STATE_LABELS))
            entry[name] = {
                "wallets": int(len(sub)),
                "hf_le_1": int(states[0]),
                "hf_le_1_debt_ge_100usd": int(((sub["hf"] <= 1) & (sub["debt_usd"] >= DUST_REFERENCE_USD)).sum()),
                "hf_le_1_median_debt_usd": float(sub.loc[sub["hf"] <= 1, "debt_usd"].median()) if states[0] else None,
                "states": {label: int(count) for label, count in zip(HF_STATE_LABELS, states)},
                "debt_usd_total": float(sub["debt_usd"].sum()),
                "median_hf": float(sub["hf"].median()) if len(sub) else None,
            }
        days[day] = entry
    return {"approximate": True, "note": DIAGNOSTIC_NOTE, "reference": OLD_PANEL_REFERENCE, "days": days}


def check_scenario(sc: Scenario, raw_dir: Path) -> dict[str, Any]:
    """Run every check for one scenario and return the report dictionary."""
    folder = raw_dir / sc.folder
    expected = window_dates(sc)
    checks = Checks()
    report: dict[str, Any] = {
        "scenario": sc.folder,
        "t": sc.t,
        "window": [sc.start, sc.end],
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    logger.info("Checking %s", sc.folder)

    loaded = check_protocol_and_prices(sc, folder, expected, checks)
    if loaded is not None:
        positions = check_positions(sc, folder, expected, loaded, checks)
        report["universe"] = positions["universe"]
        report["events"] = check_events(sc, folder, expected, loaded["blocks"], positions["users_by_day"], checks)
        report["params_at_t"] = reference_params(loaded["reserves"], loaded["emodes"], sc.t)
        report["diagnostic"] = diagnostic(positions["diag_frames"], positions["diag_days"])
    report["hard_failures"] = checks.hard_failures
    report["status"] = "PASS" if checks.hard_failures == 0 else "FAIL"
    report["checks"] = checks.items
    return report


def _clean(value: Any) -> Any:
    """Make a report JSON-serialisable (numpy scalars, NaN/inf)."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def print_report(reports: dict[str, dict[str, Any]]) -> None:
    """Print the compact console report."""
    for sc_key, report in reports.items():
        print(f"\n=== {report['scenario']}  t={report['t']}  window {report['window'][0]}..{report['window'][1]}  -> {report['status']} ===")
        for item in report["checks"]:
            print(f"[{item['status']:4}] {item['name']}: {item['detail']}")
        for label, summary in report.get("events", {}).items():
            print(
                f"  events {label}: {summary['events']} events, {summary['distinct_users']} users; "
                f"{summary['pct_in_positions_same_or_previous_day']:.1f}% in positions same/previous day"
            )

    symbols = REFERENCE_SYMBOLS + sorted(
        {name for r in reports.values() for name in r.get("params_at_t", {}) if name.startswith("eMode")}
    )
    if any("params_at_t" in r for r in reports.values()):
        print("\n--- Liquidation threshold / bonus at t ---")
        header = f"{'asset':<22}" + "".join(f"{r['scenario'] + ' ' + r['t']:>34}" for r in reports.values())
        print(header)
        for symbol in symbols:
            cells = []
            for report in reports.values():
                row = report.get("params_at_t", {}).get(symbol)
                cells.append("n/a" if not row else f"LT {row['lt']:.3f}  LB {row['lb']:.3f}  LTV {row['ltv']:.3f}")
            print(f"{symbol:<22}" + "".join(f"{cell:>34}" for cell in cells))

    print(f"\n--- Diagnostic ({DIAGNOSTIC_NOTE}) ---")
    print(f"Reference: {OLD_PANEL_REFERENCE}")
    for report in reports.values():
        for day, entry in report.get("diagnostic", {}).get("days", {}).items():
            print(
                f"\n{report['scenario']} {day} ({entry['role']}): {entry['users']} users, {entry['borrowers_with_debt']} with debt, "
                f"HF<=1 among all borrowers: {entry['hf_le_1_all_borrowers']}; A: {entry['A']['wallets']}, B: {entry['B']['wallets']}"
            )
            print(f"  {'state':<12}{'A':>8}{'B':>8}")
            for label in HF_STATE_LABELS:
                print(f"  {label:<12}{entry['A']['states'][label]:>8}{entry['B']['states'][label]:>8}")
            print(
                f"  HF<=1 on {day}: A {entry['A']['hf_le_1']}, B {entry['B']['hf_le_1']} "
                f"(with debt >= $100: A {entry['A']['hf_le_1_debt_ge_100usd']}, B {entry['B']['hf_le_1_debt_ge_100usd']}, "
                f"all borrowers {entry['hf_le_1_all_borrowers_debt_ge_100usd']})"
            )


def main(argv: list[str] | None = None) -> int:
    """Run the checks for the selected scenarios; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", choices=["baseline", "stress", "all"], default="all")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="root of the raw data (default: data/01_raw)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    selected = list(SCENARIOS.values()) if args.scenario == "all" else [SCENARIOS[args.scenario]]
    reports: dict[str, dict[str, Any]] = {}
    for sc in selected:
        report = check_scenario(sc, args.raw_dir)
        reports[sc.key] = report
        out = args.raw_dir / sc.folder / REPORT_FILE
        out.write_text(json.dumps(_clean(report), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote %s", out)

    print_report(reports)
    failed = sum(report["hard_failures"] for report in reports.values())
    print(f"\nOverall: {'PASS' if failed == 0 else 'FAIL'} ({failed} hard failure(s))")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
