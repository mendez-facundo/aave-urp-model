"""Offline plain-assert checks for the events merge of src.extraction.aave_graph_agent.

Covers ``merge_event_frames``, ``plan_event_fetch`` and ``extract_events`` with a fake
client (no network). Everything is written under a temporary directory, never data/.
Run from the repository root with: venv/Scripts/python.exe tests/test_events_merge.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

import src.extraction.aave_graph_agent as agent  # noqa: E402

SCENARIO = agent.Scenario("baseline", "baseline_mar2024", "2024-01-15", "2024-04-14")


def expect_raises(exc_type: type, fn: Callable[[], object], contains: str = "") -> BaseException:
    """Assert that ``fn`` raises ``exc_type`` with ``contains`` in its message."""
    try:
        fn()
    except exc_type as exc:
        assert contains in str(exc), f"{contains!r} not in {str(exc)!r}"
        return exc
    raise AssertionError(f"{exc_type.__name__} not raised")


def day_ts(day: str, hour: int = 0) -> int:
    """Unix timestamp of ``day`` at ``hour``:00 UTC."""
    return agent._utc_midnight_ts(agent._parse_date(day)) + hour * 3600


def event_id(number: int) -> str:
    """Event id in the subgraph format ``<block>:<log>:<tx>:...`` (made-up, unique per number)."""
    return f"{19_000_000 + number}:{number % 7}:0x{number:064x}"


def liquidation_item(number: int, timestamp: int, amount: str = "1000") -> dict[str, Any]:
    """One ``liquidationCalls`` entity in the shape the subgraph returns."""
    reserve = {"id": f"r{number}", "symbol": "WETH", "underlyingAsset": "0xABC", "decimals": 18}
    return {
        "id": event_id(number),
        "txHash": f"0x{number:064x}",
        "action": "LiquidationCall",
        "timestamp": timestamp,
        "user": {"id": f"0xuser{number}"},
        "liquidator": "0xLIQ",
        "collateralReserve": reserve,
        "principalReserve": {**reserve, "symbol": "USDC", "decimals": 6},
        "collateralAmount": amount,
        "principalAmount": "5577051147",
        "collateralAssetPriceUSD": "1.00006323",
        "borrowAssetPriceUSD": "0",
    }


def repay_item(number: int, timestamp: int) -> dict[str, Any]:
    """One ``repays`` entity in the shape the subgraph returns."""
    return {
        "id": event_id(number),
        "txHash": f"0x{number:064x}",
        "action": "Repay",
        "timestamp": timestamp,
        "user": {"id": f"0xuser{number}"},
        "repayer": {"id": "0xpayer"},
        "reserve": {"id": f"r{number}", "symbol": "RPL", "underlyingAsset": "0xDEF", "decimals": 18},
        "amount": "5707109406801454068106",
        "assetPriceUSD": "29.43657165",
        "useATokens": bool(number % 2),
    }


class FakeClient:
    """Stands in for ``GraphClient``: serves stored entities filtered like the real query."""

    def __init__(self, liquidations: list[dict[str, Any]], repays: list[dict[str, Any]]) -> None:
        self.store = {"liquidationCalls": liquidations, "repays": repays}
        self.calls: list[tuple[str, int, int]] = []

    def paginate(self, query: str, entity: str, variables: dict[str, Any]) -> Iterator[list[dict[str, Any]]]:
        t0, t1 = variables["t0"], variables["t1"]
        self.calls.append((entity, t0, t1))
        yield [item for item in self.store[entity] if t0 <= int(item["timestamp"]) < t1]


def make_client() -> FakeClient:
    """Events on 2024-02-10 .. 2024-02-16 (ids 1..7 liquidations, 101..107 repays), two per day on day 12."""
    days = ["2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16"]
    liquidations = [liquidation_item(i + 1, day_ts(day, 5)) for i, day in enumerate(days)]
    repays = [repay_item(101 + i, day_ts(day, 7)) for i, day in enumerate(days)]
    repays.append(repay_item(150, day_ts("2024-02-12", 20)))
    return FakeClient(liquidations, repays)


def read_text(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def test_merge_keeps_existing_and_sorts() -> None:
    """Duplicates on ``id`` keep the existing row; the result is sorted by (timestamp, id)."""
    cols = agent.LIQUIDATION_COLUMNS
    old = pd.DataFrame(
        [agent.liquidation_row(liquidation_item(2, day_ts("2024-02-11"), "111")),
         agent.liquidation_row(liquidation_item(3, day_ts("2024-02-12"), "222"))],
        columns=cols,
    )
    old_text = agent._as_text(old)
    new =pd.DataFrame(
        [agent.liquidation_row(liquidation_item(3, day_ts("2024-02-12"), "CHANGED")),
         agent.liquidation_row(liquidation_item(1, day_ts("2024-02-10"), "333")),
         agent.liquidation_row(liquidation_item(9, day_ts("2024-02-12"), "444"))],
        columns=cols,
    )
    merged = agent.merge_event_frames(old_text, new, cols)
    assert list(merged.columns) == cols
    assert list(merged["id"]) == [event_id(1), event_id(2), event_id(3), event_id(9)]
    assert list(merged["collateral_amount"]) == ["333", "111", "222", "444"]  # existing row 3 wins
    assert merged.iloc[0]["timestamp"] == str(day_ts("2024-02-10"))  # cells are text
    # existing rows are reproduced cell for cell
    assert merged.set_index("id").loc[old_text["id"], cols[1:]].equals(old_text.set_index("id")[cols[1:]])
    # idempotent: merging the same new rows again changes nothing
    again = agent.merge_event_frames(merged, new, cols)
    assert again.equals(merged)


def test_merge_sorts_timestamps_numerically() -> None:
    """The sort key is the integer timestamp, not its text."""
    cols = agent.REPAY_COLUMNS
    old = pd.DataFrame([agent.repay_row(repay_item(1, 10_000))], columns=cols)
    new = pd.DataFrame([agent.repay_row(repay_item(2, 9_999))], columns=cols)
    merged = agent.merge_event_frames(agent._as_text(old), new, cols)
    assert list(merged["timestamp"]) == ["9999", "10000"]


def test_merge_edge_cases() -> None:
    """No existing table, empty new table and a column mismatch."""
    cols = agent.REPAY_COLUMNS
    new = pd.DataFrame([agent.repay_row(repay_item(5, 50)), agent.repay_row(repay_item(4, 40))], columns=cols)
    fresh = agent.merge_event_frames(None, new, cols)
    assert list(fresh["id"]) == [event_id(4), event_id(5)] and fresh["use_a_tokens"].tolist() == ["False", "True"]
    empty = pd.DataFrame(columns=cols)
    assert agent.merge_event_frames(fresh, empty, cols).equals(fresh)
    assert agent.merge_event_frames(None, empty, cols).empty
    expect_raises(ValueError, lambda: agent.merge_event_frames(fresh.drop(columns="amount"), new, cols), "columns")


def test_merge_roundtrip_is_byte_exact() -> None:
    """Reading a CSV written by the agent and merging nothing into it reproduces the file."""
    cols = agent.LIQUIDATION_COLUMNS
    rows = [agent.liquidation_row(liquidation_item(n, day_ts("2024-02-10", n))) for n in (3, 1, 2)]
    frame = pd.DataFrame(rows, columns=cols).sort_values(["timestamp", "id"]).reset_index(drop=True)
    with tempfile.TemporaryDirectory() as tmp:
        source, target = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
        agent._write_frame(frame, source)
        merged = agent.merge_event_frames(agent._read_event_csv(source), pd.DataFrame(columns=cols), cols)
        agent._write_frame(merged, target)
        assert target.read_bytes() == source.read_bytes()


def test_plan_event_fetch() -> None:
    """Skip when inside the recorded window; otherwise fetch the request without leaving a gap."""
    assert agent.plan_event_fetch(None, 100, 200) == (100, 200, 100, 200)
    assert agent.plan_event_fetch((100, 200), 100, 200) is None
    assert agent.plan_event_fetch((100, 200), 120, 180) is None
    assert agent.plan_event_fetch((100, 200), 50, 100) == (50, 100, 50, 200)  # adjacent before
    assert agent.plan_event_fetch((100, 200), 200, 260) == (200, 260, 100, 260)  # adjacent after
    assert agent.plan_event_fetch((100, 200), 50, 150) == (50, 150, 50, 200)  # overlapping before
    assert agent.plan_event_fetch((100, 200), 150, 260) == (150, 260, 100, 260)  # overlapping after
    assert agent.plan_event_fetch((100, 200), 10, 60) == (10, 100, 10, 200)  # gap before is bridged
    assert agent.plan_event_fetch((100, 200), 260, 300) == (200, 300, 100, 300)  # gap after is bridged
    assert agent.plan_event_fetch((100, 200), 50, 300) == (50, 300, 50, 300)  # superset


def test_extract_events_merges_windows() -> None:
    """Extending the window fetches only the new days, keeps the old rows and records the union."""
    client = make_client()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        events_dir = root / SCENARIO.folder / agent.EVENTS_DIR
        liq_path, rep_path = events_dir / agent.LIQUIDATIONS_FILE, events_dir / agent.REPAYS_FILE
        meta_path = events_dir / agent.METADATA_FILE

        # 1. fresh extraction of 2024-02-13 .. 2024-02-14
        counts = agent.extract_events(client, SCENARIO, "2024-02-13", "2024-02-14", root)
        assert counts == {"liquidation_calls": 2, "repays": 2}
        assert client.calls == [("liquidationCalls", day_ts("2024-02-13"), day_ts("2024-02-15")),
                                ("repays", day_ts("2024-02-13"), day_ts("2024-02-15"))]
        original_liq, original_rep = liq_path.read_bytes(), rep_path.read_bytes()

        # 2. request inside the recorded window: nothing is fetched or rewritten
        client.calls.clear()
        assert agent.extract_events(client, SCENARIO, "2024-02-14", "2024-02-14", root) == counts
        assert agent.extract_events(client, SCENARIO, "2024-02-13", "2024-02-14", root) == counts
        assert client.calls == [] and liq_path.read_bytes() == original_liq

        # 3. the subgraph now returns a different amount for an event that is already stored
        client.store["liquidationCalls"][3] = liquidation_item(4, day_ts("2024-02-13", 5), "CHANGED")
        client.store["liquidationCalls"][4] = liquidation_item(5, day_ts("2024-02-14", 5), "CHANGED")

        # 4. extend backwards, overlapping by one day: only the request is fetched, old rows win
        counts = agent.extract_events(client, SCENARIO, "2024-02-11", "2024-02-13", root)
        assert client.calls == [("liquidationCalls", day_ts("2024-02-11"), day_ts("2024-02-14")),
                                ("repays", day_ts("2024-02-11"), day_ts("2024-02-14"))]
        liquidations, repays = read_text(liq_path), read_text(rep_path)
        assert list(liquidations["date"]) == ["2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14"]
        assert list(liquidations["collateral_amount"]) == ["1000"] * 4  # event 4 (overlap) keeps its stored amount
        assert list(repays["id"]) == [event_id(102), event_id(103), event_id(150), event_id(104), event_id(105)]
        assert counts == {"liquidation_calls": 4, "repays": 5}
        for old_line in original_liq.decode().splitlines(keepends=True)[1:]:
            assert old_line in liq_path.read_bytes().decode()  # existing rows are untouched, byte for byte
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert (meta["window_start"], meta["window_end_inclusive"]) == ("2024-02-11", "2024-02-14")
        assert (meta["start_timestamp"], meta["end_timestamp"]) == (day_ts("2024-02-11"), day_ts("2024-02-15"))
        assert (meta["liquidation_calls"], meta["repays"]) == (4, 5)

        # 5. a disjoint request is extended up to the recorded window so the union has no hole
        client.calls.clear()
        counts = agent.extract_events(client, SCENARIO, "2024-02-07", "2024-02-09", root)
        assert client.calls[0] == ("liquidationCalls", day_ts("2024-02-07"), day_ts("2024-02-11"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert (meta["window_start"], meta["window_end_inclusive"]) == ("2024-02-07", "2024-02-14")
        assert counts == {"liquidation_calls": 5, "repays": 6} and (meta["liquidation_calls"], meta["repays"]) == (5, 6)

        # 6. extend forwards past the end
        counts = agent.extract_events(client, SCENARIO, "2024-02-15", "2024-02-16", root)
        assert counts == {"liquidation_calls": 7, "repays": 8}
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert (meta["window_start"], meta["window_end_inclusive"]) == ("2024-02-07", "2024-02-16")
        liquidations = read_text(liq_path)
        assert not liquidations["id"].duplicated().any()
        assert list(liquidations["timestamp"].astype("int64")) == sorted(liquidations["timestamp"].astype("int64"))


def test_extract_events_refuses_to_overwrite_incomplete_state() -> None:
    """One CSV missing, or no window in the metadata: raise and leave the files alone."""
    client = make_client()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        events_dir = root / SCENARIO.folder / agent.EVENTS_DIR
        agent.extract_events(client, SCENARIO, "2024-02-13", "2024-02-14", root)
        client.calls.clear()

        rep_path = events_dir / agent.REPAYS_FILE
        saved = rep_path.read_bytes()
        rep_path.unlink()
        expect_raises(ValueError, lambda: agent.extract_events(client, SCENARIO, "2024-02-10", "2024-02-12", root), "incomplete")
        rep_path.write_bytes(saved)

        (events_dir / agent.METADATA_FILE).unlink()
        liq_before = (events_dir / agent.LIQUIDATIONS_FILE).read_bytes()
        expect_raises(ValueError, lambda: agent.extract_events(client, SCENARIO, "2024-02-10", "2024-02-12", root), "no window")
        assert client.calls == []
        assert (events_dir / agent.LIQUIDATIONS_FILE).read_bytes() == liq_before and rep_path.read_bytes() == saved


def test_default_windows() -> None:
    """Default scenario windows are [t - 60, t + 30]."""
    assert (agent.SCENARIOS["baseline"].start, agent.SCENARIOS["baseline"].end) == ("2024-01-15", "2024-04-14")
    assert (agent.SCENARIOS["stress"].start, agent.SCENARIOS["stress"].end) == ("2024-06-06", "2024-09-04")
    for scenario, t in ((agent.SCENARIOS["baseline"], "2024-03-15"), (agent.SCENARIOS["stress"], "2024-08-05")):
        t_date = agent._parse_date(t)
        assert (t_date - agent._parse_date(scenario.start)).days == 60
        assert (agent._parse_date(scenario.end) - t_date).days == 30


def main() -> None:
    """Run every ``test_*`` function in this module."""
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} checks passed")


if __name__ == "__main__":
    main()
