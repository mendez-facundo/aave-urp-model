"""Validated loaders for daily Aave v3 reserve and eMode parameters.

The extractor writes two CSV files per scenario under
``data/01_raw/<scenario_folder>/protocol/``:

* ``reserve_params_daily.csv``: one row per (date, symbol) reserve.
* ``emode_categories_daily.csv``: one row per (date, emode_id) eMode category.

Conventions shared with the extractor:

* ``ltv`` and ``liquidation_threshold`` are fractions (raw basis points / 10000).
* ``liquidation_bonus`` is the fraction above par (raw value / 10000 - 1).
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "01_raw"

PROTOCOL_SUBDIR = "protocol"
RESERVE_PARAMS_FILE = "reserve_params_daily.csv"
EMODE_CATEGORIES_FILE = "emode_categories_daily.csv"

# Accepted scenario names (canonical folder names and short aliases) -> folder.
SCENARIO_FOLDERS: Dict[str, str] = {
    "baseline_mar2024": "baseline_mar2024",
    "stress_aug2024": "stress_aug2024",
    "baseline": "baseline_mar2024",
    "stress": "stress_aug2024",
}

# Column -> kind. Kinds drive parsing and validation of each CSV cell.
_RESERVE_SCHEMA: Dict[str, str] = {
    "date": "date",
    "block": "uint",
    "symbol": "str",
    "underlying_asset": "address",
    "decimals": "uint",
    "ltv": "fraction",
    "liquidation_threshold": "fraction",
    "liquidation_bonus": "bonus",
    "usage_as_collateral_enabled": "bool",
}
_RESERVE_KEY: List[str] = ["date", "symbol"]

_EMODE_SCHEMA: Dict[str, str] = {
    "date": "date",
    "block": "uint",
    "emode_id": "uint",
    "label": "str",
    "ltv": "fraction",
    "liquidation_threshold": "fraction",
    "liquidation_bonus": "bonus",
}
_EMODE_KEY: List[str] = ["date", "emode_id"]

_DTYPES: Dict[str, str] = {
    "uint": "int64",
    "fraction": "float64",
    "bonus": "float64",
    "bool": "bool",
}

_DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_ADDRESS_PATTERN = re.compile(r"0x[0-9a-f]{40}")
_UINT_PATTERN = re.compile(r"[0-9]+")
_MAX_REPORTED_PROBLEMS = 20


def resolve_scenario_folder(scenario: str) -> str:
    """Map a scenario name or alias to its raw data folder name.

    Args:
        scenario: Canonical folder name (``baseline_mar2024``, ``stress_aug2024``)
            or alias (``baseline``, ``stress``).

    Returns:
        The folder name under ``data/01_raw``.

    Raises:
        ValueError: If the scenario is not recognised.
    """
    folder = SCENARIO_FOLDERS.get(scenario.strip()) if isinstance(scenario, str) else None
    if folder is None:
        raise ValueError(
            f"Unknown scenario {scenario!r}; expected one of {sorted(SCENARIO_FOLDERS)}"
        )
    return folder


def load_reserve_params(
    scenario: str,
    date: Optional[str] = None,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Load and validate the daily Aave v3 reserve parameters for a scenario.

    Args:
        scenario: Scenario name or alias accepted by ``resolve_scenario_folder``.
        date: Optional ``YYYY-MM-DD`` date. If given, only rows for that date are
            returned. If that exact date is missing, the latest earlier date is
            used (as-of semantics). If ``None``, all dates are returned.
        raw_dir: Root of the raw data directory.

    Returns:
        DataFrame with the reserve parameter columns. ``block`` and ``decimals``
        are int64, the fraction columns are float64, and
        ``usage_as_collateral_enabled`` is bool.

    Raises:
        ValueError: If the scenario is unknown, the file is invalid, or no rows
            exist on or before ``date``.
        FileNotFoundError: If the parameter file does not exist.
    """
    path = _protocol_file(scenario, RESERVE_PARAMS_FILE, raw_dir)
    df = _read_protocol_csv(path, _RESERVE_SCHEMA, _RESERVE_KEY, "reserve parameters")
    return _select_as_of(df, date, path)


def load_emode_categories(
    scenario: str,
    date: Optional[str] = None,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Load and validate the daily Aave v3 eMode category parameters for a scenario.

    Args:
        scenario: Scenario name or alias accepted by ``resolve_scenario_folder``.
        date: Optional ``YYYY-MM-DD`` date, with the same as-of semantics as
            ``load_reserve_params``.
        raw_dir: Root of the raw data directory.

    Returns:
        DataFrame with the eMode category columns, with the same dtypes as
        ``load_reserve_params`` where applicable.

    Raises:
        ValueError: If the scenario is unknown, the file is invalid, or no rows
            exist on or before ``date``.
        FileNotFoundError: If the parameter file does not exist.
    """
    path = _protocol_file(scenario, EMODE_CATEGORIES_FILE, raw_dir)
    df = _read_protocol_csv(path, _EMODE_SCHEMA, _EMODE_KEY, "eMode categories")
    return _select_as_of(df, date, path)


def get_param(params: pd.DataFrame, symbol: str, field: str) -> float:
    """Return one numeric parameter for a reserve from a loaded parameter table.

    Args:
        params: Output of ``load_reserve_params``, restricted to a single date.
        symbol: Reserve symbol, matched exactly (e.g. ``"WETH"``, ``"wstETH"``).
        field: Column name, e.g. ``"liquidation_threshold"``.

    Returns:
        The parameter value as a float.

    Raises:
        KeyError: If the field or the symbol is not present.
        ValueError: If the table holds more than one row for the symbol, which
            happens when it spans several dates.
    """
    if field not in params.columns:
        raise KeyError(f"Unknown parameter field {field!r}; available: {list(params.columns)}")
    if "symbol" not in params.columns:
        raise KeyError("Parameter table has no 'symbol' column")

    rows = params.loc[params["symbol"] == symbol]
    if rows.empty:
        available = sorted(params["symbol"].unique())
        raise KeyError(f"No reserve parameters for symbol {symbol!r}; available: {available}")
    if len(rows) > 1:
        raise ValueError(
            f"{len(rows)} rows found for symbol {symbol!r}; load a single date first "
            "(pass date= to load_reserve_params)"
        )
    return float(rows[field].iloc[0])


def raw_bonus_to_fraction(raw: int) -> float:
    """Convert a raw Aave liquidation bonus (basis points of par) to a fraction above par.

    Example: ``10500`` -> ``0.05``. Raw values below 10000 (for example ``0``, which
    Aave uses for categories with no bonus) map to negative numbers, so the caller
    must handle them before writing the CSV.
    """
    return (raw - 10000) / 10000


def raw_bps_to_fraction(raw: int) -> float:
    """Convert a raw Aave basis-point value (LTV, liquidation threshold) to a fraction.

    Example: ``8050`` -> ``0.805``.
    """
    return raw / 10000


def _protocol_file(scenario: str, filename: str, raw_dir: Path) -> Path:
    """Build the path of a protocol CSV for a scenario."""
    return Path(raw_dir) / resolve_scenario_folder(scenario) / PROTOCOL_SUBDIR / filename


def _validate_date_text(text: str) -> str:
    """Return ``text`` if it is a real calendar date in ``YYYY-MM-DD`` form.

    Raises:
        ValueError: If the text is not a valid date in that format.
    """
    if not _DATE_PATTERN.fullmatch(text):
        raise ValueError("expected a date in YYYY-MM-DD form")
    datetime.strptime(text, "%Y-%m-%d")  # rejects impossible dates such as 2024-02-30
    return text


def _parse_cell(text: str, kind: str):
    """Convert one stripped CSV cell according to its kind.

    Raises:
        ValueError: If the text does not match the expected kind.
    """
    if kind == "str":
        if not text:
            raise ValueError("expected a non-empty string")
        return text
    if kind == "date":
        return _validate_date_text(text)
    if kind == "uint":
        if not _UINT_PATTERN.fullmatch(text):
            raise ValueError("expected a non-negative integer")
        return int(text)
    if kind == "address":
        if not _ADDRESS_PATTERN.fullmatch(text):
            raise ValueError("expected a lowercase 0x-prefixed 40-hex-digit address")
        return text
    if kind == "bool":
        lowered = text.lower()
        if lowered not in ("true", "false"):
            raise ValueError("expected true or false")
        return lowered == "true"
    # "fraction" and "bonus" are both finite floats; their range is checked per row.
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("expected a finite number")
    return value


def _check_fraction_row(line: int, record: Dict[str, object]) -> List[str]:
    """Check the cross-field range constraints for one parsed row."""
    problems: List[str] = []
    if {"ltv", "liquidation_threshold"} <= record.keys():
        ltv = record["ltv"]
        threshold = record["liquidation_threshold"]
        if not (0 <= ltv <= threshold < 1):
            problems.append(
                f"line {line}: require 0 <= ltv <= liquidation_threshold < 1, "
                f"got ltv={ltv}, liquidation_threshold={threshold}"
            )
    if "liquidation_bonus" in record:
        bonus = record["liquidation_bonus"]
        if not (0 <= bonus < 0.5):
            problems.append(f"line {line}: require 0 <= liquidation_bonus < 0.5, got {bonus}")
    return problems


def _read_protocol_csv(
    path: Path,
    schema: Dict[str, str],
    key: List[str],
    label: str,
) -> pd.DataFrame:
    """Read, validate and type-convert one protocol parameter CSV.

    Every invalid cell or row is collected and reported together, and duplicate
    keys are rejected.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If columns are missing, the file is empty, any row is invalid,
            or the key is duplicated.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")

    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [str(column).strip() for column in raw.columns]

    missing = [column for column in schema if column not in raw.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    if raw.empty:
        raise ValueError(f"{path}: no data rows")

    problems: List[str] = []
    records: List[Dict[str, object]] = []
    for position, row in enumerate(raw.to_dict("records")):
        line = position + 2  # line 1 is the header
        record: Dict[str, object] = {}
        for column, kind in schema.items():
            text = str(row[column]).strip()
            try:
                record[column] = _parse_cell(text, kind)
            except ValueError as exc:
                problems.append(f"line {line}: {column}={text!r}: {exc}")
        if len(record) == len(schema):
            problems.extend(_check_fraction_row(line, record))
        records.append(record)

    if problems:
        shown = problems[:_MAX_REPORTED_PROBLEMS]
        message = f"{path}: {len(problems)} invalid row(s)\n  " + "\n  ".join(shown)
        if len(problems) > len(shown):
            message += f"\n  ... and {len(problems) - len(shown)} more"
        raise ValueError(message)

    df = pd.DataFrame(records, columns=list(schema))
    duplicated = df.duplicated(subset=key, keep=False)
    if duplicated.any():
        keys = sorted({tuple(item) for item in df.loc[duplicated, key].itertuples(index=False)})
        raise ValueError(f"{path}: duplicate {key} rows, e.g. {keys[:5]}")

    df = df.astype({col: _DTYPES[kind] for col, kind in schema.items() if kind in _DTYPES})
    return df.sort_values(key, kind="stable").reset_index(drop=True)


def _select_as_of(df: pd.DataFrame, date: Optional[str], path: Path) -> pd.DataFrame:
    """Return the rows for ``date``, or for the latest earlier date if it is missing."""
    if date is None:
        return df

    if not isinstance(date, str):
        raise ValueError(f"date must be a YYYY-MM-DD string, got {date!r}")
    try:
        target = _validate_date_text(date.strip())
    except ValueError as exc:
        raise ValueError(f"invalid date {date!r}: {exc}") from None

    earlier = df.loc[df["date"] <= target]
    if earlier.empty:
        raise ValueError(
            f"{path}: no rows on or before {target} (earliest date is {df['date'].min()})"
        )

    effective = earlier["date"].max()
    if effective != target:
        logger.info("No protocol parameters for %s in %s; using as-of date %s", target, path.name, effective)
    return earlier.loc[earlier["date"] == effective].reset_index(drop=True)
