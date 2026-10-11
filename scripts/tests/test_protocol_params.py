"""Plain-assert checks for src.utils.protocol_params (run directly, no pytest needed).

All CSVs are written to temporary directories, never under data/.
Run from the repository root with: venv/Scripts/python.exe scripts/tests/test_protocol_params.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.utils.protocol_params import (  # noqa: E402
    get_param,
    load_emode_categories,
    load_reserve_params,
    raw_bonus_to_fraction,
    raw_bps_to_fraction,
    resolve_scenario_folder,
)

RESERVE_HEADER = (
    "date,block,symbol,underlying_asset,decimals,ltv,"
    "liquidation_threshold,liquidation_bonus,usage_as_collateral_enabled"
)
EMODE_HEADER = "date,block,emode_id,label,ltv,liquidation_threshold,liquidation_bonus"

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"

# Two dates x three symbols. Columns after the header: date, block, symbol,
# underlying_asset, decimals, ltv, liquidation_threshold, liquidation_bonus, usage.
GOOD_RESERVE_ROWS: List[str] = [
    f"2024-03-15,19300000,WETH,{WETH},18,0.8,0.825,0.05,true",
    f"2024-03-15,19300000,USDC,{USDC},6,0.77,0.8,0.045,true",
    f"2024-03-15,19300000,WBTC,{WBTC},8,0.7,0.75,0.065,false",
    f"2024-03-18,19320000,WETH,{WETH},18,0.805,0.83,0.05,true",
    f"2024-03-18,19320000,USDC,{USDC},6,0.78,0.8,0.045,true",
    f"2024-03-18,19320000,WBTC,{WBTC},8,0.7,0.75,0.065,false",
]

GOOD_EMODE_ROWS: List[str] = [
    "2024-03-15,19300000,0,None,0,0,0",
    "2024-03-15,19300000,1,ETH correlated,0.93,0.95,0.01",
    "2024-03-18,19320000,0,None,0,0,0",
    "2024-03-18,19320000,1,ETH correlated,0.93,0.95,0.01",
]


def write_csv(raw_dir: Path, folder: str, filename: str, header: str, rows: List[str]) -> Path:
    """Write a protocol CSV under raw_dir/<folder>/protocol/ and return its path."""
    target_dir = raw_dir / folder / "protocol"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def write_reserves(raw_dir: Path, rows: List[str], folder: str = "baseline_mar2024") -> None:
    """Write the reserve parameter CSV for a scenario folder."""
    write_csv(raw_dir, folder, "reserve_params_daily.csv", RESERVE_HEADER, rows)


def expect_raises(
    exc_type: Type[BaseException],
    fn: Callable[..., object],
    *args: object,
    contains: Optional[str] = None,
    **kwargs: object,
) -> None:
    """Assert that fn raises exc_type, optionally with a message containing a substring."""
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        if contains is not None:
            assert contains in str(exc), f"expected {contains!r} in message, got: {exc}"
        return
    raise AssertionError(f"{exc_type.__name__} was not raised")


def test_resolve_scenario_folder() -> None:
    assert resolve_scenario_folder("baseline") == "baseline_mar2024"
    assert resolve_scenario_folder("stress") == "stress_aug2024"
    assert resolve_scenario_folder("baseline_mar2024") == "baseline_mar2024"
    assert resolve_scenario_folder("stress_aug2024") == "stress_aug2024"
    expect_raises(ValueError, resolve_scenario_folder, "crash2020", contains="Unknown scenario")
    expect_raises(ValueError, resolve_scenario_folder, "", contains="Unknown scenario")


def test_load_all_rows_and_dtypes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        write_reserves(raw_dir, GOOD_RESERVE_ROWS)
        df = load_reserve_params("baseline", raw_dir=raw_dir)
        assert len(df) == 6, len(df)
        assert df["block"].dtype.kind == "i"
        assert df["decimals"].dtype.kind == "i"
        assert df["ltv"].dtype.kind == "f"
        assert df["usage_as_collateral_enabled"].dtype == bool
        assert set(df["date"]) == {"2024-03-15", "2024-03-18"}


def test_as_of_lookup() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        write_reserves(raw_dir, GOOD_RESERVE_ROWS)

        exact = load_reserve_params("baseline_mar2024", date="2024-03-18", raw_dir=raw_dir)
        assert len(exact) == 3, len(exact)
        assert set(exact["date"]) == {"2024-03-18"}
        assert get_param(exact, "WETH", "liquidation_threshold") == 0.83

        # 2024-03-16 is missing: the latest earlier date, 2024-03-15, must be used.
        missing = load_reserve_params("baseline", date="2024-03-16", raw_dir=raw_dir)
        assert len(missing) == 3, len(missing)
        assert set(missing["date"]) == {"2024-03-15"}
        assert get_param(missing, "WETH", "liquidation_threshold") == 0.825

        # A date after the last observation resolves to the last observation.
        after = load_reserve_params("baseline", date="2024-03-30", raw_dir=raw_dir)
        assert set(after["date"]) == {"2024-03-18"}

        # Nothing on or before the requested date.
        expect_raises(
            ValueError,
            load_reserve_params,
            "baseline",
            date="2024-03-14",
            raw_dir=raw_dir,
            contains="no rows on or before 2024-03-14",
        )
        # Malformed date argument.
        expect_raises(ValueError, load_reserve_params, "baseline", date="15/03/2024", raw_dir=raw_dir)


def test_scenario_alias_reads_matching_folder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        write_reserves(raw_dir, GOOD_RESERVE_ROWS, folder="baseline_mar2024")
        stress_rows = [row.replace("0.83", "0.9") for row in GOOD_RESERVE_ROWS]
        write_reserves(raw_dir, stress_rows, folder="stress_aug2024")

        by_alias = load_reserve_params("stress", date="2024-03-18", raw_dir=raw_dir)
        by_folder = load_reserve_params("stress_aug2024", date="2024-03-18", raw_dir=raw_dir)
        pd.testing.assert_frame_equal(by_alias, by_folder)
        assert get_param(by_alias, "WETH", "liquidation_threshold") == 0.9

        baseline = load_reserve_params("baseline", date="2024-03-18", raw_dir=raw_dir)
        assert get_param(baseline, "WETH", "liquidation_threshold") == 0.83

        expect_raises(ValueError, load_reserve_params, "unknown", raw_dir=raw_dir)


def test_validation_errors() -> None:
    cases: List[Tuple[str, List[str], str]] = []

    def with_row(index: int, row: str) -> List[str]:
        rows = list(GOOD_RESERVE_ROWS)
        rows[index] = row
        return rows

    cases.append((
        "ltv above liquidation threshold",
        with_row(0, f"2024-03-15,19300000,WETH,{WETH},18,0.9,0.825,0.05,true"),
        "liquidation_threshold",
    ))
    cases.append((
        "liquidation threshold at 1",
        with_row(0, f"2024-03-15,19300000,WETH,{WETH},18,0.8,1.0,0.05,true"),
        "liquidation_threshold",
    ))
    cases.append((
        "bonus at 0.5",
        with_row(0, f"2024-03-15,19300000,WETH,{WETH},18,0.8,0.825,0.5,true"),
        "liquidation_bonus",
    ))
    cases.append((
        "negative bonus (raw 9500 not converted)",
        with_row(0, f"2024-03-15,19300000,WETH,{WETH},18,0.8,0.825,-0.05,true"),
        "liquidation_bonus",
    ))
    cases.append((
        "uppercase address",
        with_row(0, f"2024-03-15,19300000,WETH,{WETH.upper().replace('0X', '0x')},18,0.8,0.825,0.05,true"),
        "underlying_asset",
    ))
    cases.append((
        "non-boolean usage flag",
        with_row(0, f"2024-03-15,19300000,WETH,{WETH},18,0.8,0.825,0.05,yes"),
        "usage_as_collateral_enabled",
    ))
    cases.append((
        "bad date format",
        with_row(0, f"15/03/2024,19300000,WETH,{WETH},18,0.8,0.825,0.05,true"),
        "date",
    ))
    cases.append((
        "non-integer block",
        with_row(0, f"2024-03-15,19300000.5,WETH,{WETH},18,0.8,0.825,0.05,true"),
        "block",
    ))
    cases.append((
        "duplicate date and symbol",
        GOOD_RESERVE_ROWS + [f"2024-03-15,19300000,WETH,{WETH},18,0.8,0.825,0.05,true"],
        "duplicate",
    ))

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        for label, rows, must_contain in cases:
            write_reserves(raw_dir, rows)
            try:
                load_reserve_params("baseline", raw_dir=raw_dir)
            except ValueError as exc:
                assert must_contain in str(exc), f"{label}: {must_contain!r} not in {exc}"
            else:
                raise AssertionError(f"{label}: no ValueError raised")

        # Missing required column.
        bad_header = RESERVE_HEADER.replace(",liquidation_bonus", "")
        write_csv(raw_dir, "baseline_mar2024", "reserve_params_daily.csv", bad_header, GOOD_RESERVE_ROWS)
        expect_raises(ValueError, load_reserve_params, "baseline", raw_dir=raw_dir, contains="missing required columns")

    with tempfile.TemporaryDirectory() as tmp:
        # Missing file.
        expect_raises(FileNotFoundError, load_reserve_params, "baseline", raw_dir=Path(tmp))


def test_emode_as_of_lookup() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        write_csv(raw_dir, "baseline_mar2024", "emode_categories_daily.csv", EMODE_HEADER, GOOD_EMODE_ROWS)
        df = load_emode_categories("baseline", date="2024-03-16", raw_dir=raw_dir)
        assert set(df["date"]) == {"2024-03-15"}
        assert len(df) == 2
        assert df.loc[df["emode_id"] == 1, "liquidation_threshold"].iloc[0] == 0.95
        expect_raises(ValueError, load_emode_categories, "baseline", date="2024-03-01", raw_dir=raw_dir)


def test_get_param() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        write_reserves(raw_dir, GOOD_RESERVE_ROWS)
        single_date = load_reserve_params("baseline", date="2024-03-15", raw_dir=raw_dir)
        assert get_param(single_date, "USDC", "ltv") == 0.77
        assert get_param(single_date, "WBTC", "liquidation_bonus") == 0.065
        assert isinstance(get_param(single_date, "WETH", "ltv"), float)

        expect_raises(KeyError, get_param, single_date, "DAI", "ltv", contains="DAI")
        expect_raises(KeyError, get_param, single_date, "WETH", "no_such_field", contains="no_such_field")

        # Without a date the table spans two days, so the lookup is ambiguous.
        all_dates = load_reserve_params("baseline", raw_dir=raw_dir)
        expect_raises(ValueError, get_param, all_dates, "WETH", "ltv")


def test_raw_conversion_helpers() -> None:
    assert abs(raw_bonus_to_fraction(10500) - 0.05) < 1e-12
    assert abs(raw_bonus_to_fraction(10000)) < 1e-12
    assert abs(raw_bps_to_fraction(8050) - 0.805) < 1e-12
    assert abs(raw_bps_to_fraction(8250) - 0.825) < 1e-12


def main() -> int:
    tests: Dict[str, Callable[[], None]] = {
        name: fn for name, fn in globals().items() if name.startswith("test_") and callable(fn)
    }
    failures = 0
    for name, fn in tests.items():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report every failing check
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
