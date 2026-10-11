"""Build the development sample ``data/sample/`` from the raw extracts.

The sample mirrors the raw layout of each scenario (``baseline_mar2024`` and
``stress_aug2024``) but keeps ALL 61 days and only ~300 users::

    data/sample/<folder>/thegraph_daily/positions_<date>.csv.gz (+ metadata.json)
    data/sample/<folder>/thegraph_events/{liquidation_calls,repays}.csv (+ metadata.json)
    data/sample/<folder>/protocol/*   (copied unchanged)
    data/sample/<folder>/prices/*     (copied unchanged)
    data/sample/<folder>/sample_users.csv   (user_id, stratum, approximate profile flags)
    data/sample/README.md

Users are drawn in three strata, in this order (a user is counted once):

1. ``liquidated``: users with >= 1 ``liquidationCall`` in the window that appear
   in the positions on some day, capped at 100. When there are more, users that
   are approximate A/B at some day come first (random inside each group).
2. ``repay_to_zero``: ~80 users with a ``repay`` event on day D whose debt is
   >= $100 at the D snapshot and < $1 (or absent) at the D+1 snapshot, and that
   were not liquidated. Same A/B-first rule.
3. ``profile_ab``: the rest, random among approximate A/B users (profile A, with
   about one third of this stratum drawn from profile B when available, since B
   is much smaller). If the A/B pool runs out, ``borrower`` fills with random
   borrowers.

"Approximate A/B" is the diagnostic of ``raw_data_common.value_positions`` (oracle
prices, HF from reserve / eMode LT): at least once in the window, >= 85% of the
collateral in ETH and ETH LSTs (A) or UNI + LINK (B), and stablecoin-only debt.

Positions and events are filtered by byte-exact line copy (same text, same CRLF
line endings), so the files have exactly the format of the raw extracts. The
selection is seeded and reproducible (``--seed``, default 20240315).

Usage, from the repository root::

    venv/Scripts/python.exe scripts/make_dev_sample.py [--scenario baseline|stress|all]
        [--seed N] [--n-users 300] [--out-root data/sample]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from raw_data_common import (  # noqa: E402
    DAILY_DIR,
    EVENTS_DIR,
    LIQUIDATIONS_FILE,
    METADATA_FILE,
    PRICES_DIR,
    PROTOCOL_DIR,
    RAW_DIR,
    REPAYS_FILE,
    SAMPLE_DIR,
    SCENARIOS,
    Scenario,
    day_prices,
    positions_file,
    read_positions,
    shift_day,
    value_positions,
    window_dates,
)

from src.utils.protocol_params import load_emode_categories, load_reserve_params  # noqa: E402

import check_raw_data  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_SEED = 20240315
DEFAULT_N_USERS = 300
LIQUIDATED_CAP = 100
REPAY_ZERO_TARGET = 80
MIN_DEBT_BEFORE_REPAY_USD = 100.0
MAX_DEBT_AFTER_REPAY_USD = 1.0
PROFILE_B_SHARE_OF_STRATUM = 1 / 3
USERS_FILE = "sample_users.csv"
README_FILE = "README.md"
GITKEEP_FILE = ".gitkeep"
EVENT_USER_COLUMN = "user_id"


def scan_positions(sc: Scenario, folder: Path) -> dict[str, Any]:
    """Pass 1: per-day valuation to find approximate A/B users and daily debts."""
    oracle = pd.read_csv(folder / PRICES_DIR / "aave_oracle_daily.csv", usecols=["date", "underlying_asset", "price_usd"], dtype={"date": str})
    reserves = load_reserve_params(sc.folder, raw_dir=folder.parent)
    emodes = load_emode_categories(sc.folder, raw_dir=folder.parent)

    ever_a: set[str] = set()
    ever_b: set[str] = set()
    borrowers: set[str] = set()
    debt_by_day: dict[str, pd.Series] = {}
    for day in window_dates(sc):
        positions = read_positions(positions_file(folder, day))
        valued = value_positions(
            positions,
            day_prices(oracle, day),
            reserves.loc[reserves["date"] == day],
            emodes.loc[emodes["date"] == day],
        )
        borrowers.update(valued.index)
        ever_a.update(valued.index[valued["profile_a"]])
        ever_b.update(valued.index[valued["profile_b"]])
        debt_by_day[day] = valued["debt_usd"]
    return {"ever_a": ever_a, "ever_b": ever_b, "borrowers": borrowers, "debt_by_day": debt_by_day}


def repay_to_zero_users(sc: Scenario, repays: pd.DataFrame, debt_by_day: dict[str, pd.Series]) -> set[str]:
    """Users with a repay on day D that takes their debt from >= $100 to < $1 by D+1."""
    found: set[str] = set()
    for day, user in zip(repays["date"], repays["user_id"]):
        if day >= sc.end:
            continue  # no D+1 snapshot to see the outcome
        before = float(debt_by_day[day].get(user, 0.0))
        after = float(debt_by_day[shift_day(day, 1)].get(user, 0.0))
        if before >= MIN_DEBT_BEFORE_REPAY_USD and after < MAX_DEBT_AFTER_REPAY_USD:
            found.add(user)
    return found


def _pick(rng: np.random.Generator, candidates: list[str], count: int, prefer: set[str] | None = None) -> list[str]:
    """Pick up to ``count`` users at random; users in ``prefer`` come first."""
    if count <= 0 or not candidates:
        return []
    ordered = sorted(candidates)
    if prefer:
        first = [user for user in ordered if user in prefer]
        rest = [user for user in ordered if user not in prefer]
        picked = [first[i] for i in rng.permutation(len(first))] + [rest[i] for i in rng.permutation(len(rest))]
    else:
        picked = [ordered[i] for i in rng.permutation(len(ordered))]
    return picked[:count]


def select_users(
    sc: Scenario, scan: dict[str, Any], liquidations: pd.DataFrame, repays: pd.DataFrame, n_users: int, seed: int
) -> tuple[dict[str, str], dict[str, int]]:
    """Choose the sample users. Returns ``{user_id: stratum}`` and pool sizes."""
    rng = np.random.default_rng(seed)
    ever_a, ever_b, borrowers = scan["ever_a"], scan["ever_b"], scan["borrowers"]
    ever_ab = ever_a | ever_b
    chosen: dict[str, str] = {}

    liquidated_all = set(liquidations["user_id"])
    liquidated_pool = sorted(liquidated_all & borrowers)
    for user in _pick(rng, liquidated_pool, LIQUIDATED_CAP, prefer=ever_ab):
        chosen[user] = "liquidated"

    repay_pool = sorted(repay_to_zero_users(sc, repays, scan["debt_by_day"]) - liquidated_all - set(chosen))
    for user in _pick(rng, repay_pool, REPAY_ZERO_TARGET, prefer=ever_ab):
        chosen[user] = "repay_to_zero"

    remaining = n_users - len(chosen)
    pool_b = sorted(ever_b - set(chosen))
    pool_a = sorted((ever_a - ever_b) - set(chosen))
    n_b = min(len(pool_b), int(round(remaining * PROFILE_B_SHARE_OF_STRATUM)))
    for user in _pick(rng, pool_b, n_b):
        chosen[user] = "profile_ab"
    for user in _pick(rng, pool_a, remaining - n_b):
        chosen[user] = "profile_ab"
    leftover_b = sorted(ever_b - set(chosen))  # B users left over if the A pool was too small
    for user in _pick(rng, leftover_b, n_users - len(chosen)):
        chosen[user] = "profile_ab"
    others = sorted(borrowers - set(chosen))
    for user in _pick(rng, others, n_users - len(chosen)):
        chosen[user] = "borrower"

    pools = {
        "liquidated_pool": len(liquidated_pool),
        "repay_to_zero_pool": len(repay_pool),
        "profile_a_pool": len(ever_a - ever_b),
        "profile_b_pool": len(ever_b),
        "borrowers": len(borrowers),
    }
    return chosen, pools


def filter_gzip_positions(source: Path, target: Path, users: set[str]) -> tuple[int, int]:
    """Copy the header and the lines of ``source`` whose user is in ``users``.

    Returns the number of kept rows and distinct kept users. The line text and
    line endings are preserved byte for byte; the gzip header is deterministic.
    """
    kept_rows = 0
    kept_users: set[str] = set()
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rt", encoding="utf-8", newline="") as src, open(target, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
            header = src.readline()
            columns = header.rstrip("\r\n").split(",")
            user_index = columns.index("user_id")
            out.write(header.encode("utf-8"))
            for line in src:
                user = line.split(",", user_index + 1)[user_index]
                if user in users:
                    out.write(line.encode("utf-8"))
                    kept_rows += 1
                    kept_users.add(user)
    return kept_rows, len(kept_users)


def filter_events(source: Path, target: Path, users: set[str]) -> int:
    """Copy the header and the event lines whose ``user_id`` is in ``users`` (byte-exact)."""
    kept = 0
    with open(source, "r", encoding="utf-8", newline="") as src, open(target, "w", encoding="utf-8", newline="") as out:
        header = src.readline()
        user_index = next(csv.reader([header])).index(EVENT_USER_COLUMN)
        out.write(header)
        for line in src:
            if next(csv.reader([line]))[user_index] in users:
                out.write(line)
                kept += 1
    return kept


def write_sample(sc: Scenario, folder: Path, out_folder: Path, chosen: dict[str, str]) -> dict[str, Any]:
    """Pass 2: write the filtered positions and events and copy protocol / prices."""
    users = set(chosen)
    out_folder.mkdir(parents=True, exist_ok=True)
    for sub in (PROTOCOL_DIR, PRICES_DIR):
        shutil.copytree(folder / sub, out_folder / sub, dirs_exist_ok=True)

    source_meta = json.loads((folder / DAILY_DIR / METADATA_FILE).read_text(encoding="utf-8"))
    days_meta: dict[str, Any] = {}
    total_rows = 0
    per_day_users: list[int] = []
    for day in window_dates(sc):
        rows, n_users = filter_gzip_positions(
            positions_file(folder, day), positions_file(out_folder, day), users
        )
        original = source_meta.get("days", {}).get(day, {})
        days_meta[day] = {
            "block": original.get("block"),
            "users": n_users,
            "rows": rows,
            "source_users": original.get("users"),
            "source_rows": original.get("rows"),
        }
        total_rows += rows
        per_day_users.append(n_users)
    daily_meta = {
        "sample": True,
        "note": "Dev sample: positions filtered to the sampled users. See data/sample/README.md.",
        "subgraph_id": source_meta.get("subgraph_id"),
        "scenario": sc.folder,
        "first_date": sc.start,
        "last_date": sc.end,
        "n_days": len(days_meta),
        "days": days_meta,
    }
    (out_folder / DAILY_DIR / METADATA_FILE).write_text(json.dumps(daily_meta, indent=2), encoding="utf-8")

    events_out = out_folder / EVENTS_DIR
    events_out.mkdir(parents=True, exist_ok=True)
    liq_kept = filter_events(folder / EVENTS_DIR / LIQUIDATIONS_FILE, events_out / LIQUIDATIONS_FILE, users)
    rep_kept = filter_events(folder / EVENTS_DIR / REPAYS_FILE, events_out / REPAYS_FILE, users)
    events_meta = json.loads((folder / EVENTS_DIR / METADATA_FILE).read_text(encoding="utf-8"))
    events_meta.update(
        {
            "sample": True,
            "source_liquidation_calls": events_meta.get("liquidation_calls"),
            "source_repays": events_meta.get("repays"),
            "liquidation_calls": liq_kept,
            "repays": rep_kept,
        }
    )
    (events_out / METADATA_FILE).write_text(json.dumps(events_meta, indent=2), encoding="utf-8")

    return {
        "position_rows": total_rows,
        "users_per_day_min": min(per_day_users),
        "users_per_day_max": max(per_day_users),
        "liquidation_events": liq_kept,
        "repay_events": rep_kept,
    }


def write_users_file(out_folder: Path, chosen: dict[str, str], scan: dict[str, Any]) -> None:
    """Write ``sample_users.csv`` with the stratum and approximate profile flags."""
    rows = [
        {
            "user_id": user,
            "stratum": stratum,
            "ever_profile_a": user in scan["ever_a"],
            "ever_profile_b": user in scan["ever_b"],
        }
        for user, stratum in sorted(chosen.items())
    ]
    pd.DataFrame(rows).to_csv(out_folder / USERS_FILE, index=False, lineterminator="\n")


def build_scenario(sc: Scenario, raw_dir: Path, out_root: Path, n_users: int, seed: int) -> dict[str, Any]:
    """Build the sample of one scenario and return its summary."""
    folder = raw_dir / sc.folder
    out_folder = out_root / sc.folder
    logger.info("Scanning %s", sc.folder)
    scan = scan_positions(sc, folder)

    cols = {"dtype": str, "keep_default_na": False}
    liquidations = pd.read_csv(folder / EVENTS_DIR / LIQUIDATIONS_FILE, usecols=["date", "user_id"], **cols)
    repays = pd.read_csv(folder / EVENTS_DIR / REPAYS_FILE, usecols=["date", "user_id"], **cols)
    chosen, pools = select_users(sc, scan, liquidations, repays, n_users, seed)

    logger.info("Writing %s (%d users)", out_folder, len(chosen))
    summary = write_sample(sc, folder, out_folder, chosen)
    write_users_file(out_folder, chosen, scan)

    strata = pd.Series(chosen).value_counts().to_dict()
    sample_a = len(set(chosen) & scan["ever_a"])
    sample_b = len(set(chosen) & scan["ever_b"])
    summary.update(
        {
            "scenario": sc.folder,
            "users": len(chosen),
            "strata": {name: int(strata.get(name, 0)) for name in ("liquidated", "repay_to_zero", "profile_ab", "borrower")},
            "pools": pools,
            "sample_ever_a": sample_a,
            "sample_ever_b": sample_b,
        }
    )

    report = check_raw_data.check_scenario(sc, out_root)
    summary["check_status"] = report["status"]
    summary["check_not_pass"] = [f"{c['status']} {c['name']}: {c['detail']}" for c in report["checks"] if c["status"] not in ("PASS", "INFO")]
    return summary


def write_readme(out_root: Path, summaries: list[dict[str, Any]], seed: int, n_users: int) -> None:
    """Write ``README.md`` describing how the sample was built and its counts."""
    lines = [
        "# Development sample",
        "",
        "Small, reproducible subset of `data/01_raw/` for developing and testing the pipeline. "
        "All 61 days of each scenario window are kept; only the users are subsampled. "
        "Git-ignored (only `.gitkeep` is tracked). Full runs use the raw data.",
        "",
        "## How it was built",
        "",
        "Script: `scripts/make_dev_sample.py` (shared helpers in `scripts/raw_data_common.py`). "
        f"Seed: `{seed}` (`numpy.random.default_rng`); target: {n_users} users per scenario. "
        "Re-running with the same raw data and seed gives the same sample.",
        "",
        "Strata, in this order (a user is counted once; candidates are sorted before drawing):",
        "",
        f"1. `liquidated`: users with >= 1 liquidation in the window that appear in the positions, capped at {LIQUIDATED_CAP}. "
        "If there are more, users that are approximate A/B at some day are drawn first.",
        f"2. `repay_to_zero`: ~{REPAY_ZERO_TARGET} users, not liquidated, with a repay on day D, debt >= ${MIN_DEBT_BEFORE_REPAY_USD:.0f} "
        f"at the D snapshot and < ${MAX_DEBT_AFTER_REPAY_USD:.0f} (or absent) at the D+1 snapshot. Same A/B-first rule.",
        "3. `profile_ab`: the rest, random among approximate A/B users (about one third from profile B when available, "
        "because B is much smaller than A).",
        "4. `borrower`: random borrowers, only if the A/B pool runs out.",
        "",
        "Approximate A/B (diagnostic, not the thesis methodology): at least once in the window, >= 85% of the collateral "
        "(oracle USD, reserves enabled as collateral) in WETH/wstETH/rETH/cbETH/weETH/osETH/ETHx (A) or UNI+LINK (B), "
        "and stablecoin-only debt. HF uses the reserve LT, or the eMode category LT for eMode users and category assets.",
        "",
        "Positions and events are filtered by user with a byte-exact line copy (same columns, text and CRLF line endings "
        "as the raw files). `protocol/` and `prices/` are copied unchanged. "
        "`thegraph_*/metadata.json` are the raw ones with the sample counts (`source_*` keep the originals). "
        f"`<folder>/{USERS_FILE}` lists the users with their stratum and approximate profile flags.",
        "",
        "## Counts",
        "",
        "| scenario | users | liquidated | repay_to_zero | profile_ab | borrower | position rows | liquidation events | repay events | users/day (min-max) | ever A | ever B |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        st = s["strata"]
        lines.append(
            f"| {s['scenario']} | {s['users']} | {st['liquidated']} | {st['repay_to_zero']} | {st['profile_ab']} | {st['borrower']} "
            f"| {s['position_rows']} | {s['liquidation_events']} | {s['repay_events']} "
            f"| {s['users_per_day_min']}-{s['users_per_day_max']} | {s['sample_ever_a']} | {s['sample_ever_b']} |"
        )
    lines += [
        "",
        "Candidate pools in the full raw data:",
        "",
        "| scenario | liquidated users in positions | repay-to-zero users | ever A (not B) | ever B | all borrowers |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        p = s["pools"]
        lines.append(
            f"| {s['scenario']} | {p['liquidated_pool']} | {p['repay_to_zero_pool']} | {p['profile_a_pool']} | {p['profile_b_pool']} | {p['borrowers']} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- The sample is stratified, not representative: liquidated and repaid wallets are over-represented. "
        "Do not estimate frequencies (PD, state shares) from it.",
        "- Entries and exits of the sampled users are real; the daily universe of the sample is not the daily universe of the market.",
        "- `scripts/check_raw_data.py --raw-dir data/sample` runs the structural checks on the sample.",
        "",
    ]
    (out_root / README_FILE).write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Build the sample for the selected scenarios."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", choices=["baseline", "stress", "all"], default="all")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-users", type=int, default=DEFAULT_N_USERS)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-root", type=Path, default=SAMPLE_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    selected = list(SCENARIOS.values()) if args.scenario == "all" else [SCENARIOS[args.scenario]]
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / GITKEEP_FILE).touch()

    summaries = [build_scenario(sc, args.raw_dir, args.out_root, args.n_users, args.seed) for sc in selected]
    if args.scenario == "all":
        write_readme(args.out_root, summaries, args.seed, args.n_users)

    for s in summaries:
        print(
            f"{s['scenario']}: {s['users']} users {s['strata']}, {s['position_rows']} position rows, "
            f"{s['liquidation_events']} liquidations, {s['repay_events']} repays, "
            f"ever A/B {s['sample_ever_a']}/{s['sample_ever_b']}, structural check {s['check_status']}"
        )
        for item in s["check_not_pass"]:
            print(f"  note: {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
