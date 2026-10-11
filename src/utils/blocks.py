"""Map UTC calendar dates to Ethereum mainnet block numbers.

The block for a date ``D`` is the first block whose timestamp is greater than
or equal to ``D 00:00:00 UTC``. A 12-second slot model provides a starting
point, which is then refined with ``eth_getBlockByNumber`` until the exact
boundary block is found.

Run the demo from the repository root with ``python -m src.utils.blocks``.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.utils.rpc import RpcClient

logger = logging.getLogger(__name__)

SLOT_SECONDS = 12
DATE_FORMAT = "%Y-%m-%d"
CSV_COLUMNS = ["date", "timestamp", "block"]

# After this many estimate-driven probes, switch to plain bisection so the
# worst case stays logarithmic in the search range.
MAX_ESTIMATE_PROBES = 6

DEMO_DATES: tuple[str, ...] = ("2024-03-15", "2024-08-05")


class BlockLookupError(RuntimeError):
    """Raised when a block lookup cannot be answered (for example, a future timestamp)."""


def estimate_block(target_ts: int, ref_block: int, ref_ts: int) -> int:
    """Estimate the block at ``target_ts`` assuming exact 12-second slots.

    Parameters
    ----------
    target_ts:
        Target Unix timestamp in seconds (UTC).
    ref_block:
        Known block number used as the anchor.
    ref_ts:
        Timestamp of ``ref_block`` in seconds.

    Returns
    -------
    int
        ``ref_block`` plus the number of slots needed to reach ``target_ts``,
        rounded up so the estimate is the first slot at or after the target.
        Clamped to 0 for targets before the anchor's chain start. Missed slots
        make the estimate drift, so treat it as a starting point only.
    """
    delta = target_ts - ref_ts
    # Integer ceiling division: -((-x) // n) == ceil(x / n).
    slots = -((-delta) // SLOT_SECONDS)
    return max(0, ref_block + slots)


def _search_block(client: RpcClient, target_ts: int, hint: int | None) -> tuple[int, int]:
    """Locate the first block whose timestamp is at or after ``target_ts``.

    The search keeps a bracket ``(low, high]`` with ``timestamp(low) < target_ts
    <= timestamp(high)``. Each probe proposes the next guess from the 12-second
    model anchored at the latest probe. After ``MAX_ESTIMATE_PROBES`` probes it
    bisects the bracket instead.

    Returns
    -------
    tuple[int, int]
        ``(block_number, block_timestamp)`` of the boundary block.
    """
    latest = client.get_block("latest")
    latest_number = int(latest["number"])
    latest_ts = int(latest["timestamp"])
    if target_ts > latest_ts:
        raise BlockLookupError(
            f"Timestamp {target_ts} is later than the chain head (block {latest_number})."
        )

    known_ts: dict[int, int] = {latest_number: latest_ts}

    def block_ts(number: int) -> int:
        """Return a block timestamp, fetching it at most once per lookup."""
        if number not in known_ts:
            known_ts[number] = int(client.get_block(number)["timestamp"])
        return known_ts[number]

    low, high = -1, latest_number  # -1 stands for "before genesis"
    start = (
        hint
        if hint is not None
        else estimate_block(target_ts, latest_number, latest_ts)
    )
    guess = min(max(start, 0), latest_number - 1)

    probes = 0
    while high - low > 1:
        probes += 1
        guess_ts = block_ts(guess)
        if guess_ts >= target_ts:
            high = guess
        else:
            low = guess
        if high - low <= 1:
            break

        if probes <= MAX_ESTIMATE_PROBES:
            candidate = estimate_block(target_ts, guess, guess_ts)
        else:
            candidate = (low + high) // 2
        # Keep the next probe strictly inside the bracket so it always shrinks.
        guess = min(max(candidate, low + 1), high - 1)

    return high, block_ts(high)


def find_block_by_timestamp(
    client: RpcClient,
    target_ts: int,
    hint: int | None = None,
) -> int:
    """Return the first block whose timestamp is at or after ``target_ts``.

    Parameters
    ----------
    client:
        Connected ``RpcClient`` (see ``src.utils.rpc``).
    target_ts:
        Target Unix timestamp in seconds (UTC).
    hint:
        Optional block number expected to be close to the answer, such as the
        block found for the previous day. It is used as the first probe instead
        of the 12-second estimate from the chain head.

    Returns
    -------
    int
        Exact block number. A typical lookup uses about 3 to 6 RPC calls.

    Raises
    ------
    BlockLookupError
        If ``target_ts`` lies after the chain head.
    RpcError
        If the RPC call fails permanently or exhausts its retries.
    """
    block, _ = _search_block(client, target_ts, hint)
    return block


def _parse_date(value: str) -> date:
    """Parse a ``YYYY-MM-DD`` string into a ``date``."""
    try:
        return datetime.strptime(value, DATE_FORMAT).date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a YYYY-MM-DD date, got {value!r}.") from exc


def _utc_midnight_ts(day: date) -> int:
    """Return the Unix timestamp of 00:00:00 UTC on ``day``."""
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def _read_cache(cache_path: Path | None) -> dict[date, tuple[int, int]]:
    """Load ``date -> (block, timestamp)`` rows from ``cache_path`` if it exists."""
    if cache_path is None or not cache_path.is_file():
        return {}
    try:
        frame = pd.read_csv(cache_path, dtype={"date": str})
    except pd.errors.EmptyDataError:
        return {}

    missing_columns = set(CSV_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Cache file {cache_path} is missing columns: {sorted(missing_columns)}."
        )

    rows: dict[date, tuple[int, int]] = {}
    for raw_date, raw_ts, raw_block in zip(
        frame["date"], frame["timestamp"], frame["block"]
    ):
        rows[_parse_date(str(raw_date))] = (int(raw_block), int(raw_ts))
    logger.info("Loaded %s cached rows from %s.", len(rows), cache_path)
    return rows


def _write_cache(known: dict[date, tuple[int, int]], cache_path: Path) -> None:
    """Write all known rows to ``cache_path`` as CSV, sorted by date."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {"date": day.isoformat(), "timestamp": ts, "block": block}
            for day, (block, ts) in sorted(known.items())
        ],
        columns=CSV_COLUMNS,
    )
    frame.to_csv(cache_path, index=False)
    logger.info("Wrote %s rows to %s.", len(frame), cache_path)


def _hint_from_nearest(
    known: dict[date, tuple[int, int]],
    day: date,
    target_ts: int,
) -> int | None:
    """Estimate the block for ``target_ts`` from the closest already-known date."""
    if not known:
        return None
    nearest = min(known, key=lambda d: abs((d - day).days))
    block, ts = known[nearest]
    return estimate_block(target_ts, block, ts)


def daily_blocks(
    start_date: str,
    end_date: str,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Return the first block of every UTC day in ``[start_date, end_date]``.

    Parameters
    ----------
    start_date:
        First date, ``YYYY-MM-DD``. Included.
    end_date:
        Last date, ``YYYY-MM-DD``. Included.
    cache_path:
        Optional CSV with columns ``date, timestamp, block``. Dates already in
        the file are reused and only missing dates are queried. The merged table
        (sorted by date) is written back to this path. The RPC client is created
        only when at least one date needs a query, and it reads
        ``ALCHEMY_RPC_URL``.

    Returns
    -------
    pandas.DataFrame
        One row per requested date with columns ``date`` (``YYYY-MM-DD``),
        ``timestamp`` (int, actual block timestamp in seconds) and ``block``
        (int).
    """
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}.")

    wanted = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]

    known = _read_cache(cache_path)
    missing = [day for day in wanted if day not in known]

    if missing:
        client = RpcClient()
        logger.info("Querying %s missing date(s) via RPC.", len(missing))
        try:
            for day in missing:
                target_ts = _utc_midnight_ts(day)
                hint = _hint_from_nearest(known, day, target_ts)
                known[day] = _search_block(client, target_ts, hint)
                logger.info("Resolved %s to block %s.", day.isoformat(), known[day][0])
        finally:
            client.close()
    else:
        logger.info("All %s dates found in cache; no RPC calls made.", len(wanted))

    if cache_path is not None:
        _write_cache(known, cache_path)

    rows = [
        {"date": day.isoformat(), "timestamp": known[day][1], "block": known[day][0]}
        for day in wanted
    ]
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def main() -> None:
    """Print the first block of each demo date."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    frames = [daily_blocks(day, day) for day in DEMO_DATES]
    print(pd.concat(frames, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
