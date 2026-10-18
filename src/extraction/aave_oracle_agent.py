"""Aave v3 Ethereum oracle extractor: daily USD price of every reserve.

For each UTC day in a scenario window, the agent reads the price that the Aave
oracle reports for every reserve of the Aave v3 Ethereum market at that day's
block (the first block with ``timestamp >= D 00:00:00 UTC``, see
``src.utils.blocks.daily_blocks``).

At each block the addresses are resolved on chain, because they can change:

1. ``PoolAddressesProvider.getPool()`` and ``getPriceOracle()``.
2. ``Pool.getReservesList()`` and ``AaveOracle.BASE_CURRENCY_UNIT()``.
3. ``AaveOracle.getAssetsPrices(address[])``, one call per day.

Token ``symbol()`` and ``decimals()`` are read once per asset and cached.
Calls go through ``src.utils.rpc.RpcClient`` as JSON-RPC batches and are
encoded and decoded with ``eth_abi``. The endpoint comes from
``ALCHEMY_RPC_URL`` in ``.env`` and is never logged.

Outputs, under ``<out-root>/<scenario_folder>/``:

* ``prices/aave_oracle_daily.csv``: long format, one row per (date, reserve)
  with ``date, block, symbol, underlying_asset, price_usd, price_raw,
  base_currency_unit``.
* ``prices/aave_oracle_metadata.json``: provider, pool and oracle addresses
  seen, reserves per day, extraction timestamp.
* ``protocol/daily_blocks.csv``: date to block cache shared with other agents.

The run is idempotent and resumable: dates already present in the output CSV
are not fetched again.

Default windows are ``[t - 60, t + 30]`` (91 days): baseline ``2024-01-15 .. 2024-04-14``
(t = 2024-03-15) and stress ``2024-06-06 .. 2024-09-04`` (t = 2024-08-05). Extending a
window only fetches the dates that are missing; existing rows are kept.

Usage, from the repository root::

    python src/extraction/aave_oracle_agent.py [--scenario baseline|stress|all]
        [--start YYYY-MM-DD --end YYYY-MM-DD] [--out-root PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    # Allow `python src/extraction/aave_oracle_agent.py` from the repository root.
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from eth_abi import decode, encode  # noqa: E402
from eth_abi.exceptions import DecodingError  # noqa: E402
from eth_utils import function_signature_to_4byte_selector  # noqa: E402

from src.utils.blocks import daily_blocks  # noqa: E402
from src.utils.rpc import RpcClient, RpcError  # noqa: E402

logger = logging.getLogger(__name__)

RAW_DATA_DIR = PROJECT_ROOT / "data" / "01_raw"

# Aave v3 Ethereum (core market) PoolAddressesProvider. Pool and oracle are
# resolved from it at every block.
POOL_ADDRESSES_PROVIDER = "0x2f39d218133AFaB8F2B819B1066c7E434Ad94E9e"
EXPECTED_POOL = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
EXPECTED_BASE_CURRENCY_UNIT = 10**8  # USD with 8 decimals

DATE_FORMAT = "%Y-%m-%d"
CHUNK_DAYS = 10  # days per batch round trip; the CSV is saved after each chunk

OUTPUT_COLUMNS = [
    "date",
    "block",
    "symbol",
    "underlying_asset",
    "price_usd",
    "price_raw",
    "base_currency_unit",
]
STRING_COLUMNS = {
    "date": str,
    "symbol": str,
    "underlying_asset": str,
    "price_raw": str,
}
PRICES_RELATIVE_DIR = Path("prices")
OUTPUT_FILENAME = "aave_oracle_daily.csv"
METADATA_FILENAME = "aave_oracle_metadata.json"
BLOCKS_RELATIVE_PATH = Path("protocol") / "daily_blocks.csv"


@dataclass(frozen=True)
class Scenario:
    """Extraction window of one scenario."""

    folder: str
    start: str
    end: str
    valuation_date: str


SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario("baseline_mar2024", "2024-01-15", "2024-04-14", "2024-03-15"),
    "stress": Scenario("stress_aug2024", "2024-06-06", "2024-09-04", "2024-08-05"),
}


class OracleExtractionError(RuntimeError):
    """Raised when an on-chain answer cannot be decoded or is inconsistent."""


def _calldata(signature: str, abi_types: list[str] | None = None, args: list[Any] | None = None) -> str:
    """Return ABI-encoded calldata for ``signature`` as a 0x-prefixed hex string."""
    selector = function_signature_to_4byte_selector(signature)
    body = encode(abi_types, args) if abi_types else b""
    return "0x" + (selector + body).hex()


SEL_GET_POOL = _calldata("getPool()")
SEL_GET_PRICE_ORACLE = _calldata("getPriceOracle()")
SEL_GET_RESERVES_LIST = _calldata("getReservesList()")
SEL_BASE_CURRENCY_UNIT = _calldata("BASE_CURRENCY_UNIT()")
SEL_SYMBOL = _calldata("symbol()")
SEL_DECIMALS = _calldata("decimals()")


def _eth_call_request(to: str, data: str, block: int) -> tuple[str, list[Any]]:
    """Build one ``eth_call`` entry for ``RpcClient.batch_call``."""
    return "eth_call", [{"to": to, "data": data}, hex(block)]


def _raw_bytes(raw: Any, what: str) -> bytes:
    """Convert a 0x-prefixed hex result into bytes, rejecting empty answers."""
    if not isinstance(raw, str) or not raw.startswith("0x"):
        raise OracleExtractionError(f"{what} returned a non-hex result.")
    data = bytes.fromhex(raw[2:])
    if not data:
        raise OracleExtractionError(
            f"{what} returned empty data (no contract at that block, or the call reverted)."
        )
    return data


def _decode_single(abi_type: str, raw: Any, what: str) -> Any:
    """Decode a single return value of type ``abi_type``."""
    try:
        return decode([abi_type], _raw_bytes(raw, what))[0]
    except DecodingError as exc:
        raise OracleExtractionError(f"{what}: cannot decode {abi_type}.") from exc


def _decode_address(raw: Any, what: str) -> str:
    """Decode an ``address`` return value to a lowercase 0x string."""
    return str(_decode_single("address", raw, what)).lower()


def _decode_address_list(raw: Any, what: str) -> list[str]:
    """Decode an ``address[]`` return value to lowercase 0x strings."""
    return [str(item).lower() for item in _decode_single("address[]", raw, what)]


def _decode_text(raw: Any, what: str) -> str:
    """Decode ``symbol()``, which is a ``string`` for most tokens and ``bytes32`` for some (MKR)."""
    data = _raw_bytes(raw, what)
    text: str | None = None
    if len(data) != 32:
        try:
            text = decode(["string"], data)[0]
        except (DecodingError, ValueError):
            text = None
    if text is None:
        text = data[:32].rstrip(b"\x00").decode("utf-8", errors="ignore")
    return text.replace("\x00", "").strip()


def _fetch_asset_info(
    client: RpcClient, unknown: dict[str, int]
) -> dict[str, dict[str, Any]]:
    """Read ``symbol()`` and ``decimals()`` for assets, each at the block where it was first seen.

    A batch is tried first. If a token reverts on either call, the batch fails
    as a whole, so the calls are repeated one by one and the first failing
    token is reported.
    """
    addresses = list(unknown)
    requests: list[tuple[str, list[Any]]] = []
    for address in addresses:
        block = unknown[address]
        requests.append(_eth_call_request(address, SEL_SYMBOL, block))
        requests.append(_eth_call_request(address, SEL_DECIMALS, block))

    try:
        results = client.batch_call(requests)
    except RpcError:
        logger.info("Batch of token metadata calls failed; retrying one by one.")
        results = []
        for method, params in requests:
            try:
                results.append(client.call(method, params))
            except RpcError:
                results.append(None)

    info: dict[str, dict[str, Any]] = {}
    for index, address in enumerate(addresses):
        symbol = _decode_text(results[2 * index], f"symbol() of {address}")
        decimals = int(_decode_single("uint8", results[2 * index + 1], f"decimals() of {address}"))
        if not symbol:
            raise OracleExtractionError(f"symbol() of {address} is empty.")
        info[address] = {"symbol": symbol, "decimals": decimals}
        logger.info("New asset %s: %s (%s decimals).", address, symbol, decimals)
    return info


def _fetch_chunk(
    client: RpcClient,
    days: list[tuple[str, int]],
    assets: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Fetch oracle prices for a list of ``(date, block)`` pairs.

    Parameters
    ----------
    client:
        Connected RPC client.
    days:
        ``(YYYY-MM-DD, block)`` pairs to read.
    assets:
        Cache ``address -> {"symbol", "decimals"}``. New assets are added in place.

    Returns
    -------
    tuple
        Output rows (one per date and reserve) and per-date context
        ``{date: {"block", "pool", "oracle"}}``.
    """
    # Step 1: provider -> pool and oracle at every block.
    step1: list[tuple[str, list[Any]]] = []
    for _, block in days:
        step1.append(_eth_call_request(POOL_ADDRESSES_PROVIDER, SEL_GET_POOL, block))
        step1.append(_eth_call_request(POOL_ADDRESSES_PROVIDER, SEL_GET_PRICE_ORACLE, block))
    out1 = client.batch_call(step1)
    pools = [_decode_address(out1[2 * i], "getPool()") for i in range(len(days))]
    oracles = [_decode_address(out1[2 * i + 1], "getPriceOracle()") for i in range(len(days))]

    # Step 2: reserves list from the pool, quote unit from the oracle.
    step2: list[tuple[str, list[Any]]] = []
    for (_, block), pool, oracle in zip(days, pools, oracles):
        step2.append(_eth_call_request(pool, SEL_GET_RESERVES_LIST, block))
        step2.append(_eth_call_request(oracle, SEL_BASE_CURRENCY_UNIT, block))
    out2 = client.batch_call(step2)
    reserves = [_decode_address_list(out2[2 * i], "getReservesList()") for i in range(len(days))]
    units = [int(_decode_single("uint256", out2[2 * i + 1], "BASE_CURRENCY_UNIT()")) for i in range(len(days))]

    # Step 3: symbol and decimals of assets not seen before.
    unknown: dict[str, int] = {}
    for (_, block), day_reserves in zip(days, reserves):
        for asset in day_reserves:
            if asset not in assets and asset not in unknown:
                unknown[asset] = block
    if unknown:
        assets.update(_fetch_asset_info(client, unknown))

    # Step 4: one getAssetsPrices call per day.
    step4 = [
        _eth_call_request(
            oracle,
            _calldata("getAssetsPrices(address[])", ["address[]"], [day_reserves]),
            block,
        )
        for (_, block), oracle, day_reserves in zip(days, oracles, reserves)
    ]
    out4 = client.batch_call(step4)

    rows: list[dict[str, Any]] = []
    context: dict[str, dict[str, Any]] = {}
    for index, (day, block) in enumerate(days):
        prices = [int(p) for p in _decode_single("uint256[]", out4[index], "getAssetsPrices()")]
        if len(prices) != len(reserves[index]):
            raise OracleExtractionError(
                f"{day}: got {len(prices)} prices for {len(reserves[index])} reserves."
            )
        if pools[index] != EXPECTED_POOL.lower():
            logger.warning("%s: Pool is %s, expected %s.", day, pools[index], EXPECTED_POOL.lower())
        if units[index] != EXPECTED_BASE_CURRENCY_UNIT:
            logger.warning(
                "%s: BASE_CURRENCY_UNIT is %s, expected %s.",
                day,
                units[index],
                EXPECTED_BASE_CURRENCY_UNIT,
            )
        symbols = [assets[asset]["symbol"] for asset in reserves[index]]
        if len(set(symbols)) != len(symbols):
            logger.warning("%s: duplicate symbols among reserves; use underlying_asset as key.", day)

        for asset, symbol, raw_price in zip(reserves[index], symbols, prices):
            if raw_price <= 0:
                logger.warning("%s: non-positive oracle price for %s.", day, symbol)
            rows.append(
                {
                    "date": day,
                    "block": block,
                    "symbol": symbol,
                    "underlying_asset": asset,
                    "price_usd": raw_price / units[index],
                    "price_raw": str(raw_price),
                    "base_currency_unit": units[index],
                }
            )
        context[day] = {"block": block, "pool": pools[index], "oracle": oracles[index]}
        logger.info(
            "%s block %s: %s reserves, pool %s, oracle %s.",
            day,
            block,
            len(reserves[index]),
            pools[index],
            oracles[index],
        )
    return rows, context


def _read_existing(output_path: Path) -> pd.DataFrame:
    """Load the previous output CSV, or return an empty frame with the right columns."""
    if not output_path.is_file():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    try:
        frame = pd.read_csv(output_path, dtype=STRING_COLUMNS)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    missing = set(OUTPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{output_path} is missing columns: {sorted(missing)}.")
    return frame[OUTPUT_COLUMNS]


def _sorted_output(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by date then symbol and reset the index."""
    return (
        frame.sort_values(["date", "symbol"], kind="stable")
        .reset_index(drop=True)[OUTPUT_COLUMNS]
    )


def _atomic_write_text(path: Path, writer: Any) -> None:
    """Write through a temporary file and rename it, so an interrupted run never truncates ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    writer(tmp_path)
    os.replace(tmp_path, path)


def _write_output(frame: pd.DataFrame, output_path: Path) -> None:
    """Write the long-format CSV atomically."""
    _atomic_write_text(output_path, lambda tmp: frame.to_csv(tmp, index=False))
    logger.info("Wrote %s rows to %s.", len(frame), output_path)


def _write_metadata(
    frame: pd.DataFrame,
    context: dict[str, dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    metadata_path: Path,
) -> None:
    """Write provider, pool and oracle addresses seen, reserves per day and run timestamp.

    ``daily`` (date -> block, pool, oracle) is merged with the previous file so
    that a resumed run keeps the addresses of days fetched earlier.
    """
    daily: dict[str, dict[str, Any]] = {}
    if metadata_path.is_file():
        try:
            daily = dict(json.loads(metadata_path.read_text(encoding="utf-8")).get("daily", {}))
        except (ValueError, OSError):
            logger.warning("Could not read previous metadata; rebuilding it.")
    daily.update(context)
    known_dates = set(frame["date"])
    daily = {day: daily[day] for day in sorted(daily) if day in known_dates}

    used_assets = set(frame["underlying_asset"])
    metadata = {
        "provider": POOL_ADDRESSES_PROVIDER.lower(),
        "pool_addresses": sorted({entry["pool"] for entry in daily.values()}),
        "oracle_addresses": sorted({entry["oracle"] for entry in daily.values()}),
        "base_currency_units": sorted(int(unit) for unit in frame["base_currency_unit"].unique()),
        "n_days": int(frame["date"].nunique()),
        "first_date": str(frame["date"].min()),
        "last_date": str(frame["date"].max()),
        "reserves_per_day": {
            str(day): int(count) for day, count in frame.groupby("date").size().items()
        },
        "assets": {a: assets[a] for a in sorted(used_assets) if a in assets},
        "daily": daily,
        "extracted_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _atomic_write_text(
        metadata_path,
        lambda tmp: tmp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8"),
    )
    logger.info("Wrote metadata to %s.", metadata_path)


def _window_dates(start: str, end: str) -> list[str]:
    """Return every ``YYYY-MM-DD`` date in ``[start, end]``."""
    first = datetime.strptime(start, DATE_FORMAT).date()
    last = datetime.strptime(end, DATE_FORMAT).date()
    if last < first:
        raise ValueError(f"end {end} is before start {start}.")
    return [(first + timedelta(days=n)).isoformat() for n in range((last - first).days + 1)]


def extract_scenario(
    folder: str,
    start: str,
    end: str,
    out_root: Path = RAW_DATA_DIR,
    assets: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Extract (or resume) the daily oracle prices of one scenario window.

    Parameters
    ----------
    folder:
        Scenario folder name, such as ``baseline_mar2024``.
    start, end:
        Inclusive window, ``YYYY-MM-DD``.
    out_root:
        Root directory that contains the scenario folders.
    assets:
        Optional cache ``address -> {"symbol", "decimals"}`` shared between calls.

    Returns
    -------
    pandas.DataFrame
        The complete output table of the scenario after the run.
    """
    assets = assets if assets is not None else {}
    scenario_dir = out_root / folder
    output_path = scenario_dir / PRICES_RELATIVE_DIR / OUTPUT_FILENAME
    metadata_path = scenario_dir / PRICES_RELATIVE_DIR / METADATA_FILENAME

    wanted = _window_dates(start, end)
    existing = _read_existing(output_path)
    done = set(existing["date"])
    missing = [day for day in wanted if day not in done]
    logger.info(
        "%s: %s dates requested, %s already extracted, %s to fetch.",
        folder,
        len(wanted),
        len(wanted) - len(missing),
        len(missing),
    )
    if not missing:
        return _sorted_output(existing)

    blocks = daily_blocks(start, end, cache_path=scenario_dir / BLOCKS_RELATIVE_PATH)
    block_by_date = dict(zip(blocks["date"], blocks["block"].astype(int)))
    pending = [(day, int(block_by_date[day])) for day in missing]

    client = RpcClient()
    frame = existing
    try:
        for offset in range(0, len(pending), CHUNK_DAYS):
            chunk = pending[offset : offset + CHUNK_DAYS]
            rows, context = _fetch_chunk(client, chunk, assets)
            new_rows = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
            combined = new_rows if frame.empty else pd.concat([frame, new_rows], ignore_index=True)
            frame = _sorted_output(combined)
            _write_output(frame, output_path)
            _write_metadata(frame, context, assets, metadata_path)
    finally:
        client.close()
    return frame


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract daily Aave v3 oracle USD prices for every reserve."
    )
    parser.add_argument(
        "--scenario",
        choices=["baseline", "stress", "all"],
        default="all",
        help="Scenario window to extract (default: all).",
    )
    parser.add_argument("--start", help="Override window start, YYYY-MM-DD (needs --end).")
    parser.add_argument("--end", help="Override window end, YYYY-MM-DD (needs --start).")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=RAW_DATA_DIR,
        help="Root folder that holds the scenario folders (default: data/01_raw).",
    )
    args = parser.parse_args(argv)
    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must be given together.")
    if args.start is not None and args.scenario == "all":
        parser.error("--start/--end need a single --scenario (baseline or stress).")
    return args


def main(argv: list[str] | None = None) -> None:
    """Run the extraction for the selected scenarios."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    assets: dict[str, dict[str, Any]] = {}
    for name in names:
        scenario = SCENARIOS[name]
        start = args.start or scenario.start
        end = args.end or scenario.end
        frame = extract_scenario(scenario.folder, start, end, args.out_root, assets)
        logger.info(
            "%s done: %s rows, %s dates, %s reserves on the last date.",
            scenario.folder,
            len(frame),
            frame["date"].nunique(),
            int((frame["date"] == frame["date"].max()).sum()),
        )


if __name__ == "__main__":
    main()
