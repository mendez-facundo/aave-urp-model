"""Offline plain-assert checks for src.extraction.aave_graph_agent (no network).

The HTTP session is replaced by a fake and ``time.sleep`` is disabled, so the
checks run in well under a second and never touch data/.
Run from the repository root with: venv/Scripts/python.exe tests/test_aave_graph_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests  # noqa: E402

import src.extraction.aave_graph_agent as agent  # noqa: E402

agent.time.sleep = lambda seconds: None

KEY = "SECRETKEY"


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status: int, body: Any = None, text: str = "") -> None:
        self.status_code, self._body, self.text = status, body, text

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("not JSON")
        return self._body


def client_with(responses: list[Any]) -> agent.GraphClient:
    """Return a client whose session replays ``responses`` (exceptions are raised)."""
    client = agent.GraphClient(api_key=KEY)
    queue: Iterator[Any] = iter(responses)

    def post(url: str, json: Any = None, timeout: Any = None) -> FakeResponse:
        item = next(queue)
        if isinstance(item, Exception):
            raise item
        return item

    client._session.post = post  # type: ignore[method-assign]
    return client


def expect_raises(exc_type: type, fn: Callable[[], object], contains: str = "") -> BaseException:
    """Assert that ``fn`` raises ``exc_type`` with ``contains`` in its message."""
    try:
        fn()
    except exc_type as exc:
        assert contains in str(exc), f"{contains!r} not in {str(exc)!r}"
        return exc
    raise AssertionError(f"{exc_type.__name__} not raised")


def test_ray_math() -> None:
    """rayMul rounds half up in integer arithmetic."""
    assert agent.ray_mul(10**27, 10**27) == 10**27
    assert agent.ray_mul(1, agent.RAY // 2) == 1
    assert agent.ray_mul(1, agent.RAY // 2 - 1) == 0
    assert agent.ray_mul(3 * 10**18, 105 * 10**25) == 315 * 10**16  # 3e18 * 1.05


def test_flatten_users() -> None:
    """Accrued balances use the block's indices; all-zero rows are dropped."""
    reserve = {
        "id": "r1",
        "symbol": "WETH",
        "underlyingAsset": "0xC02AAA39B223FE8D0A0E5C4F27EAD9083C756CC2",
        "decimals": 18,
        "liquidityIndex": str(105 * 10**25),  # 1.05
        "variableBorrowIndex": str(11 * 10**26),  # 1.10
    }
    zero = {"scaledATokenBalance": "0", "scaledVariableDebt": "0",
            "principalStableDebt": "0", "currentStableDebt": "0"}
    users = [
        {
            "id": "0xabc",
            "eModeCategoryId": None,
            "reserves": [
                {**zero, "scaledATokenBalance": str(2 * 10**18), "usageAsCollateralEnabledOnUser": True,
                 "reserve": reserve},
                {**zero, "scaledVariableDebt": str(10**18), "usageAsCollateralEnabledOnUser": False,
                 "reserve": reserve},
                {**zero, "usageAsCollateralEnabledOnUser": False, "reserve": reserve},
            ],
        },
        {"id": "0xdef", "eModeCategoryId": {"id": "1"}, "reserves": []},
    ]
    rows, stable_rows = agent.flatten_users(users, "2024-03-14", 123)
    assert stable_rows == 0 and len(rows) == 2
    collateral, debt = rows
    assert list(collateral) == agent.POSITION_COLUMNS
    assert collateral["a_token_balance"] == 2.1 and collateral["variable_debt"] == 0.0
    assert debt["variable_debt"] == 1.1 and debt["a_token_balance"] == 0.0
    assert collateral["underlying_asset"] == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    assert collateral["emode_category_id"] == 0 and collateral["scaled_a_token_balance"] == str(2 * 10**18)
    assert collateral["usage_as_collateral_enabled"] is True and debt["usage_as_collateral_enabled"] is False


def test_param_rows() -> None:
    """Parameters are fractions, the bonus is above par, raw 0 maps to 0.0."""
    data = {
        "reserves": [
            {"symbol": "WETH", "underlyingAsset": "0xAB", "decimals": 18, "usageAsCollateralEnabled": True,
             "baseLTVasCollateral": 8050, "reserveLiquidationThreshold": 8300,
             "reserveLiquidationBonus": 10500, "eMode": {"id": "1"}},
            {"symbol": "GHO", "underlyingAsset": "0xCD", "decimals": 18, "usageAsCollateralEnabled": False,
             "baseLTVasCollateral": 0, "reserveLiquidationThreshold": 0,
             "reserveLiquidationBonus": 0, "eMode": None},
        ],
        "emodeCategories": [{"id": "1", "label": "ETH correlated", "ltv": 9300,
                             "liquidationThreshold": 9500, "liquidationBonus": 10100}],
    }
    reserves, emodes = agent.build_param_rows(data, "2024-03-14", 123)
    gho, weth = reserves  # sorted by symbol
    assert (weth["ltv"], weth["liquidation_threshold"], weth["liquidation_bonus"]) == (0.805, 0.83, 0.05)
    assert weth["emode_id"] == 1 and gho["emode_id"] == 0 and weth["underlying_asset"] == "0xab"
    assert (gho["ltv"], gho["liquidation_threshold"], gho["liquidation_bonus"]) == (0.0, 0.0, 0.0)
    assert emodes[0]["liquidation_bonus"] == 0.01 and emodes[0]["emode_id"] == 1
    bad = {"reserves": [{**data["reserves"][0], "baseLTVasCollateral": 9000}], "emodeCategories": []}
    expect_raises(ValueError, lambda: agent.build_param_rows(bad, "2024-03-14", 123), "out of range")


def test_retries() -> None:
    """Transport errors, 5xx, bad JSON and HTTP-200 GraphQL errors are retried."""
    bad_indexers = FakeResponse(200, {"errors": [{"message": f"bad indexers: {{0x1: Unavailable}} {KEY}"}]})
    ok = FakeResponse(200, {"data": {"users": []}})
    client = client_with([
        bad_indexers, FakeResponse(502), requests.Timeout(f"https://x/{KEY}"),
        FakeResponse(200, None), FakeResponse(200, {"data": None}), ok,
    ])
    assert client.query("q", {}) == {"users": []}
    assert client.request_count == 6 and client.query_count == 1

    exhausted = client_with([bad_indexers] * agent.MAX_ATTEMPTS)
    error = expect_raises(agent.GraphAPIError, lambda: exhausted.query("q", {}), "bad indexers")
    assert not isinstance(error, agent.GraphFatalError) and KEY not in str(error)
    assert exhausted.request_count == agent.MAX_ATTEMPTS


def test_fatal_errors() -> None:
    """Auth failures and invalid queries are not retried, and never leak the key."""
    unauthorized = client_with([FakeResponse(401, text=f"denied {KEY}")])
    error = expect_raises(agent.GraphFatalError, lambda: unauthorized.query("q", {}), "HTTP 401")
    assert KEY not in str(error) and unauthorized.request_count == 1
    invalid = client_with([FakeResponse(200, {"errors": [{"message": "Type `User` has no field `x`"}]})])
    expect_raises(agent.GraphFatalError, lambda: invalid.query("q", {}))
    assert invalid.request_count == 1
    assert KEY not in repr(invalid)
    expect_raises(agent.GraphFatalError, lambda: agent.GraphClient(api_key=" "), "GRAPH_API_KEY")


def test_pagination() -> None:
    """Pages are chained with id_gt and stop at the first short page."""
    client = agent.GraphClient(api_key=KEY)
    seen: list[str] = []
    pages = {"": ["a", "b"], "b": ["c", "d"], "d": ["e"]}

    def fake_query(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        seen.append(variables["lastId"])
        return {"things": [{"id": i} for i in pages[variables["lastId"]]]}

    original = agent.PAGE_SIZE
    agent.PAGE_SIZE = 2
    try:
        client.query = fake_query  # type: ignore[method-assign]
        assert [[x["id"] for x in p] for p in client.paginate("q", "things", {})] == [["a", "b"], ["c", "d"], ["e"]]
        assert seen == ["", "b", "d"]
        client.query = lambda q, v: {"things": [{"id": "a"}, {"id": "b"}]}  # type: ignore[method-assign]
        expect_raises(agent.GraphAPIError, lambda: list(client.paginate("q", "things", {})), "did not advance")
    finally:
        agent.PAGE_SIZE = original


def main() -> None:
    """Run every ``test_*`` function in this module."""
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} checks passed")


if __name__ == "__main__":
    main()
