"""T1.1 spike: verify The Graph time-travel on the Aave v3 Ethereum subgraph.

Reproducible, step-based script. Each step issues a handful of GraphQL calls
(a full ``all`` run is ~65 Graph calls plus 3 RPC calls) and prints a summary. Raw results
are written as JSON to ``--out`` (default: ``data/02_interim/spikes``, which is
git-ignored) so the findings document can be cross-checked.

Usage (from the repo root)::

    venv/Scripts/python.exe scripts/spikes/graph_spike.py all
    venv/Scripts/python.exe scripts/spikes/graph_spike.py schema
    venv/Scripts/python.exe scripts/spikes/graph_spike.py timetravel
    venv/Scripts/python.exe scripts/spikes/graph_spike.py count
    venv/Scripts/python.exe scripts/spikes/graph_spike.py balances
    venv/Scripts/python.exe scripts/spikes/graph_spike.py reserves
    venv/Scripts/python.exe scripts/spikes/graph_spike.py events
    venv/Scripts/python.exe scripts/spikes/graph_spike.py lean
    venv/Scripts/python.exe scripts/spikes/graph_spike.py windows
    venv/Scripts/python.exe scripts/spikes/graph_spike.py blocks

The API key is read from ``.env`` (``GRAPH_API_KEY``). Neither the key nor the
gateway URL is ever printed; exception text is scrubbed before logging.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GRAPH_API_KEY = os.getenv("GRAPH_API_KEY", "").strip()
SUBGRAPH_ID = "Cd2gEDVeqnjBn1hSeqFMitw8Q1iiyV9FYUZkLNRcL87g"
GATEWAY_BASE = "https://gateway-arbitrum.network.thegraph.com/api"

B1 = 19_433_100  # baseline valuation block, 2024-03-15
B0 = B1 - 30 * 7_200  # ~30 days earlier
B_STRESS = 20_457_180  # stress valuation block, 2024-08-05

# One-day event window: 2024-03-14 00:00 UTC to 2024-03-15 00:00 UTC.
T_START = int(datetime(2024, 3, 14, tzinfo=timezone.utc).timestamp())
T_END = int(datetime(2024, 3, 15, tzinfo=timezone.utc).timestamp())

REQUEST_TIMEOUT_SECONDS = 90
MAX_RETRIES = 3
PAGE_SLEEP_SECONDS = 0.2


class SpikeError(RuntimeError):
    """Raised when a GraphQL call fails after retries (message is scrubbed)."""


class Client:
    """Thin GraphQL client that counts calls and never leaks the key."""

    def __init__(self) -> None:
        if not GRAPH_API_KEY:
            raise SpikeError("GRAPH_API_KEY is missing in .env")
        self._url = f"{GATEWAY_BASE}/{GRAPH_API_KEY}/subgraphs/id/{SUBGRAPH_ID}"
        self._session = requests.Session()
        self.calls = 0
        self.latencies: list[float] = []

    def _scrub(self, text: str) -> str:
        return text.replace(GRAPH_API_KEY, "<KEY>").replace(self._url, "<URL>")

    def raw(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """POST and return the full JSON body (data and/or errors), with retries
        on transport failures and 429/5xx only. GraphQL errors are returned."""
        payload = {"query": query, "variables": variables or {}}
        last: str = ""
        for attempt in range(1, MAX_RETRIES + 1):
            self.calls += 1
            t0 = time.perf_counter()
            try:
                resp = self._session.post(
                    self._url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
                )
            except requests.RequestException as exc:
                last = self._scrub(f"{type(exc).__name__}: {exc}")
                time.sleep(2.0 * attempt)
                continue
            self.latencies.append(time.perf_counter() - t0)
            if resp.status_code in {429, 502, 503, 504}:
                last = f"HTTP {resp.status_code}"
                time.sleep(2.0 * attempt)
                continue
            try:
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                raise SpikeError(self._scrub(f"{type(exc).__name__}: {exc}")) from None
        raise SpikeError(f"failed after {MAX_RETRIES} attempts: {last}")

    def query(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Like ``raw`` but raises on GraphQL errors and returns ``data``."""
        body = self.raw(query, variables)
        if body.get("errors"):
            raise SpikeError(f"GraphQL errors: {json.dumps(body['errors'])[:600]}")
        return body["data"]


# ---------------------------------------------------------------------------
# Validated queries (these are the strings the extractor should reuse).
# ---------------------------------------------------------------------------

Q_USERS_SNAPSHOT_PAGE = """
query UsersSnapshotPage($first: Int!, $lastId: ID!, $block: Int!) {
  users(
    first: $first
    orderBy: id
    orderDirection: asc
    block: { number: $block }
    where: { borrowedReservesCount_gt: 0, id_gt: $lastId }
  ) {
    id
    borrowedReservesCount
    eModeCategoryId { id }
    reserves(first: 100) {
      id
      usageAsCollateralEnabledOnUser
      scaledATokenBalance
      currentATokenBalance
      scaledVariableDebt
      currentVariableDebt
      principalStableDebt
      currentStableDebt
      lastUpdateTimestamp
      reserve {
        id
        symbol
        underlyingAsset
        decimals
        liquidityIndex
        variableBorrowIndex
      }
    }
  }
}
"""

Q_USERS_SNAPSHOT_LEAN = """
query UsersSnapshotLean($first: Int!, $lastId: ID!, $block: Int!) {
  users(
    first: $first
    orderBy: id
    orderDirection: asc
    block: { number: $block }
    where: { borrowedReservesCount_gt: 0, id_gt: $lastId }
  ) {
    id
    borrowedReservesCount
    eModeCategoryId { id }
    reserves(first: 100) {
      usageAsCollateralEnabledOnUser
      scaledATokenBalance
      scaledVariableDebt
      principalStableDebt
      reserve { id liquidityIndex variableBorrowIndex }
    }
  }
}
"""

Q_RESERVES_AT_BLOCK = """
query ReservesAtBlock($block: Int!) {
  reserves(first: 200, block: { number: $block }) {
    id
    symbol
    underlyingAsset
    decimals
    usageAsCollateralEnabled
    borrowingEnabled
    isActive
    isFrozen
    isPaused
    isDropped
    baseLTVasCollateral
    reserveLiquidationThreshold
    reserveLiquidationBonus
    reserveFactor
    liquidityIndex
    variableBorrowIndex
    eMode { id }
    lastUpdateTimestamp
  }
  emodeCategories(first: 50, block: { number: $block }) {
    id
    label
    ltv
    liquidationThreshold
    liquidationBonus
    oracle
    assets { id }
  }
}
"""

Q_LIQUIDATIONS = """
query Liquidations($first: Int!, $lastId: ID!, $t0: Int!, $t1: Int!) {
  liquidationCalls(
    first: $first
    orderBy: id
    orderDirection: asc
    where: { timestamp_gte: $t0, timestamp_lt: $t1, id_gt: $lastId }
  ) {
    id
    txHash
    action
    timestamp
    liquidator
    user { id }
    collateralAmount
    principalAmount
    collateralAssetPriceUSD
    borrowAssetPriceUSD
    collateralReserve { id symbol underlyingAsset decimals }
    principalReserve { id symbol underlyingAsset decimals }
  }
}
"""

Q_REPAYS = """
query Repays($first: Int!, $lastId: ID!, $t0: Int!, $t1: Int!) {
  repays(
    first: $first
    orderBy: id
    orderDirection: asc
    where: { timestamp_gte: $t0, timestamp_lt: $t1, id_gt: $lastId }
  ) {
    id
    txHash
    action
    timestamp
    user { id }
    repayer { id }
    amount
    assetPriceUSD
    useATokens
    reserve { id symbol underlyingAsset decimals }
  }
}
"""

TYPE_FIELDS_FRAGMENT = """
fragment TypeRef on __Type {
  kind name ofType { kind name ofType { kind name ofType { kind name } } }
}
"""


def type_ref_to_str(t: dict[str, Any] | None) -> str:
    """Render an introspection type reference like ``[UserReserve!]!``."""
    if t is None:
        return "?"
    kind, name, inner = t.get("kind"), t.get("name"), t.get("ofType")
    if kind == "NON_NULL":
        return f"{type_ref_to_str(inner)}!"
    if kind == "LIST":
        return f"[{type_ref_to_str(inner)}]"
    return str(name)


def introspect_types(client: Client, names: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Return {type_name: [(field, type_str), ...]} using ``__type``."""
    aliases = "\n".join(
        f'  t{i}: __type(name: "{n}") {{ name fields {{ name type {{ ...TypeRef }} }} }}'
        for i, n in enumerate(names)
    )
    data = client.query(f"query {{\n{aliases}\n}}\n{TYPE_FIELDS_FRAGMENT}")
    out: dict[str, list[tuple[str, str]]] = {}
    for i, n in enumerate(names):
        node = data.get(f"t{i}")
        if not node:
            out[n] = []
            continue
        out[n] = [(f["name"], type_ref_to_str(f["type"])) for f in node["fields"]]
    return out


def write_json(out_dir: Path, name: str, obj: Any) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")


def rpc_block_timestamp(block: int) -> int | None:
    """Block timestamp via the Alchemy RPC in .env (``_meta`` returns null).

    Returns None when ALCHEMY_RPC_URL is unset or the call fails; the URL is
    never printed.
    """
    url = os.getenv("ALCHEMY_RPC_URL", "").strip()
    if not url:
        return None
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getBlockByNumber", "params": [hex(block), False]}
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return int(resp.json()["result"]["timestamp"], 16)
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return None


RAY = 10**27
SECONDS_PER_YEAR = 31_536_000


def ray_mul(a: int, b: int) -> int:
    """Aave rayMul: round half up."""
    return (a * b + RAY // 2) // RAY


def linear_interest(rate: int, dt: int) -> int:
    """Aave MathUtils.calculateLinearInterest (used for the liquidity index)."""
    return RAY + (rate * dt) // SECONDS_PER_YEAR


def compounded_interest(rate: int, dt: int) -> int:
    """Aave MathUtils.calculateCompoundedInterest (3-term Taylor, used for debt)."""
    if dt == 0:
        return RAY
    exp_m1 = dt - 1
    exp_m2 = dt - 2 if dt > 2 else 0
    base2 = ray_mul(rate, rate) // (SECONDS_PER_YEAR * SECONDS_PER_YEAR)
    base3 = ray_mul(base2, rate) // SECONDS_PER_YEAR
    second = dt * exp_m1 * base2 // 2
    third = dt * exp_m1 * exp_m2 * base3 // 6
    return RAY + (rate * dt) // SECONDS_PER_YEAR + second + third


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_schema(c: Client, out: Path) -> None:
    print("== STEP 1: schema introspection ==")
    data = c.query("{ __schema { queryType { fields { name } } types { name kind } } }")
    roots = sorted(f["name"] for f in data["__schema"]["queryType"]["fields"])
    print("root collections:", [r for r in roots if r.endswith("s")][:80])
    type_names = sorted(t["name"] for t in data["__schema"]["types"] if t["kind"] == "OBJECT")
    print("object types:", type_names)
    wanted = [
        "User", "UserReserve", "Reserve", "EModeCategory", "LiquidationCall", "Repay",
        "ATokenBalanceHistoryItem", "VTokenBalanceHistoryItem", "STokenBalanceHistoryItem",
        "ReserveParamsHistoryItem", "Borrow", "Supply", "PriceOracleAsset",
        "EModeCategoryConfig", "UserEModeSet", "Pool",
    ]
    fields = introspect_types(c, [w for w in wanted if w in type_names])
    for name, fl in fields.items():
        print(f"\n[{name}]")
        for fname, ftype in fl:
            print(f"  {fname}: {ftype}")
    write_json(out, "schema.json", {"roots": roots, "objects": type_names, "fields": fields})


def _users_page(c: Client, block: int, last_id: str, first: int) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    data = c.query(
        Q_USERS_SNAPSHOT_PAGE, {"first": first, "lastId": last_id, "block": block}
    )
    return data["users"], time.perf_counter() - t0


def step_timetravel(c: Client, out: Path) -> None:
    print("== STEP 2: time-travel on users ==")
    # 2a. indexer meta + page-size limits (ids only, then the full query)
    meta = c.query("{ _meta { block { number timestamp } hasIndexingErrors } }")["_meta"]
    print(f"  _meta: head block={meta['block']['number']} hasIndexingErrors={meta['hasIndexingErrors']}")
    ids_q = (
        "query($f:Int!,$b:Int!){ users(first:$f, orderBy:id, block:{number:$b}, "
        "where:{borrowedReservesCount_gt:0}) { id } }"
    )
    for first in (1000, 1001, 2000, 5000):
        t0 = time.perf_counter()
        body = c.raw(ids_q, {"f": first, "b": B1})
        dt = time.perf_counter() - t0
        if body.get("errors"):
            msg = json.dumps(body["errors"])
            reason = "first-limit" if "must be between 0 and 1000" in msg else "other"
            print(f"  users(first={first}) ids only: ERROR ({reason}) {dt:.2f}s")
        else:
            print(f"  users(first={first}) ids only: OK rows={len(body['data']['users'])} {dt:.2f}s")
    for first in (1000, 2000):
        t0 = time.perf_counter()
        body = c.raw(Q_USERS_SNAPSHOT_PAGE, {"first": first, "lastId": "", "block": B1})
        dt = time.perf_counter() - t0
        n = len(body["data"]["users"]) if not body.get("errors") else "ERROR"
        kb = len(json.dumps(body)) // 1024
        print(f"  full snapshot query first={first}: rows={n} {dt:.2f}s payload~{kb} KB")
    # 2b. nested reserves(first: N) limits
    for nested in (100, 1000, 1001):
        q = Q_USERS_SNAPSHOT_PAGE.replace("reserves(first: 100)", f"reserves(first: {nested})")
        body = c.raw(q, {"first": 5, "lastId": "", "block": B1})
        status = "ERROR " + json.dumps(body["errors"])[:200] if body.get("errors") else "OK"
        print(f"  nested reserves(first: {nested}): {status}")
    # 2c. same first page at B0 and B1
    p0, l0 = _users_page(c, B0, "", 1000)
    p1, l1 = _users_page(c, B1, "", 1000)
    print(f"  B0 page1: {len(p0)} rows in {l0:.2f}s ; B1 page1: {len(p1)} rows in {l1:.2f}s")
    p1_again, _ = _users_page(c, B1, "", 1000)
    print(f"  deterministic (B1 page1 fetched twice identical): {p1 == p1_again}")
    ids0 = {u["id"] for u in p0}
    ids1 = {u["id"] for u in p1}
    print(f"  page1 overlap: {len(ids0 & ids1)}; only B0: {len(ids0 - ids1)}; only B1: {len(ids1 - ids0)}")
    by0 = {u["id"]: u for u in p0}
    changed = same = 0
    examples: list[dict] = []
    for u in p1:
        if u["id"] not in by0:
            continue
        a = {r["id"]: (r["scaledATokenBalance"], r["scaledVariableDebt"]) for r in by0[u["id"]]["reserves"]}
        b = {r["id"]: (r["scaledATokenBalance"], r["scaledVariableDebt"]) for r in u["reserves"]}
        if a != b:
            changed += 1
            if len(examples) < 2:
                examples.append({"user": u["id"], "B0": a, "B1": b})
        else:
            same += 1
    print(f"  common users with different scaled balances B0 vs B1: {changed} / {changed + same}")
    for ex in examples:
        print("   example:", json.dumps(ex)[:500])
    max_res = max((len(u["reserves"]) for u in p1), default=0)
    max_bor = max((int(u["borrowedReservesCount"]) for u in p1), default=0)
    print(f"  max len(reserves) in page: {max_res}; max borrowedReservesCount: {max_bor}")
    write_json(out, "timetravel_sample.json", {"B0": p0[:3], "B1": p1[:3], "examples": examples})


def step_count(c: Client, out: Path, max_pages: int = 80) -> None:
    print("== STEP 3: count users with borrowedReservesCount_gt 0 (baseline B1 and stress block) ==")
    summary: dict[str, Any] = {}
    for label, block in (("B1", B1), ("B_STRESS", B_STRESS)):
        last_id, total, pages = "", 0, 0
        lat: list[float] = []
        exhausted = False
        n_reserve_rows = 0
        max_nested = 0
        while pages < max_pages:
            page, dt = _users_page(c, block, last_id, 1000)
            pages += 1
            lat.append(dt)
            total += len(page)
            n_reserve_rows += sum(len(u["reserves"]) for u in page)
            max_nested = max([max_nested, *(len(u["reserves"]) for u in page)])
            print(f"  [{label}] page {pages}: {len(page)} rows ({dt:.2f}s) cumulative={total}")
            if len(page) < 1000:
                exhausted = True
                break
            last_id = page[-1]["id"]
            time.sleep(PAGE_SLEEP_SECONDS)
        print(
            f"  [{label}] total borrowers={total} exhausted={exhausted} pages={pages} "
            f"userReserve rows={n_reserve_rows} max nested={max_nested} "
            f"lat mean={sum(lat)/len(lat):.2f}s p50={pct(lat,.5):.2f}s p90={pct(lat,.9):.2f}s"
        )
        summary[label] = {"block": block, "total": total, "pages": pages, "exhausted": exhausted,
                          "user_reserve_rows": n_reserve_rows, "max_nested_reserves": max_nested,
                          "latencies": lat}
    write_json(out, "count.json", summary)


Q_BALANCE_CHECK = """
query BalanceCheck($first: Int!, $block: Int!) {
  users(first: $first, orderBy: id, block: { number: $block },
        where: { borrowedReservesCount_gt: 0 }) {
    id
    borrowedReservesCount
    reserves(first: 100) {
      id
      scaledATokenBalance
      currentATokenBalance
      scaledVariableDebt
      currentVariableDebt
      principalStableDebt
      currentStableDebt
      lastUpdateTimestamp
      reserve {
        symbol decimals liquidityIndex variableBorrowIndex
        liquidityRate variableBorrowRate lastUpdateTimestamp
      }
    }
  }
}
"""


def _summ(label: str, rel: list[float]) -> None:
    if not rel:
        print(f"  {label}: n=0")
        return
    nz = sum(1 for x in rel if x > 1e-9)
    print(
        f"  {label}: n={len(rel)} nonzero(>1e-9)={nz} median={pct(rel,.5):.2e} "
        f"p90={pct(rel,.9):.2e} p99={pct(rel,.99):.2e} max={max(rel):.2e}"
    )


def step_balances(c: Client, out: Path) -> None:
    print("== STEP 4: stored vs accrued balances at block B1 ==")
    ts = rpc_block_timestamp(B1)
    print(f"  B1 timestamp (RPC): {ts}")
    data = c.query(Q_BALANCE_CHECK, {"first": 1000, "block": B1})
    users = data["users"]
    # consistency of the time-travelled borrowedReservesCount with nested debts
    mism = 0
    for u in users:
        n_debt = sum(1 for r in u["reserves"]
                     if int(r["scaledVariableDebt"]) > 0 or int(r["principalStableDebt"]) > 0)
        if n_debt != int(u["borrowedReservesCount"]):
            mism += 1
    print(f"  users with borrowedReservesCount != #reserves with debt>0: {mism}/{len(users)}")
    ur_with_debt = sum(1 for u in users for r in u["reserves"] if int(r["scaledVariableDebt"]) > 0)
    stable = sum(1 for u in users for r in u["reserves"] if int(r["principalStableDebt"]) > 0)
    print(f"  user-reserves with variable debt: {ur_with_debt}; with stable debt: {stable}")
    rel = {k: [] for k in ("a_stored_vs_idx", "a_stored_vs_accr", "a_idx_vs_accr",
                           "d_stored_vs_idx", "d_stored_vs_accr", "d_idx_vs_accr")}
    ages_ur: list[float] = []
    ages_res: list[float] = []
    rows: list[dict[str, Any]] = []
    for u in users:
        for r in u["reserves"]:
            res = r["reserve"]
            li, vi = int(res["liquidityIndex"]), int(res["variableBorrowIndex"])
            lr, vr = int(res["liquidityRate"]), int(res["variableBorrowRate"])
            sa, sd = int(r["scaledATokenBalance"]), int(r["scaledVariableDebt"])
            ca, cd = int(r["currentATokenBalance"]), int(r["currentVariableDebt"])
            if ts is not None:
                dt_res = max(ts - int(res["lastUpdateTimestamp"]), 0)
                ages_res.append(dt_res / 3600)
                ages_ur.append(max(ts - int(r["lastUpdateTimestamp"]), 0) / 3600)
                accr_a = ray_mul(sa, ray_mul(li, linear_interest(lr, dt_res)))
                accr_d = ray_mul(sd, ray_mul(vi, compounded_interest(vr, dt_res)))
            else:
                accr_a, accr_d = ray_mul(sa, li), ray_mul(sd, vi)
            idx_a, idx_d = ray_mul(sa, li), ray_mul(sd, vi)
            if ca > 0:
                rel["a_stored_vs_idx"].append(abs(idx_a - ca) / ca)
                rel["a_stored_vs_accr"].append(abs(accr_a - ca) / ca)
                rel["a_idx_vs_accr"].append(abs(accr_a - idx_a) / accr_a if accr_a else 0)
            if cd > 0:
                rel["d_stored_vs_idx"].append(abs(idx_d - cd) / cd)
                rel["d_stored_vs_accr"].append(abs(accr_d - cd) / cd)
                rel["d_idx_vs_accr"].append(abs(accr_d - idx_d) / accr_d if accr_d else 0)
            if (ca > 0 or cd > 0) and len(rows) < 400:
                rows.append({"symbol": res["symbol"], "dec": int(res["decimals"]),
                             "sa": sa, "ca": ca, "idx_a": idx_a, "accr_a": accr_a,
                             "sd": sd, "cd": cd, "idx_d": idx_d, "accr_d": accr_d,
                             "ur_age_h": (ts - int(r["lastUpdateTimestamp"])) / 3600 if ts else None,
                             "res_age_h": (ts - int(res["lastUpdateTimestamp"])) / 3600 if ts else None})
    for k, v in rel.items():
        _summ(k, v)
    if ages_ur:
        print(f"  age of userReserve.lastUpdateTimestamp vs B1 (h): median={pct(ages_ur,.5):.1f} p90={pct(ages_ur,.9):.1f} max={max(ages_ur):.1f}")
        print(f"  age of reserve.lastUpdateTimestamp vs B1 (h): median={pct(ages_res,.5):.3f} p90={pct(ages_res,.9):.3f} max={max(ages_res):.1f}")
    # show examples with an old userReserve update (where staleness would show)
    old = sorted((x for x in rows if x["ur_age_h"] is not None and x["ur_age_h"] > 24),
                 key=lambda x: -x["ur_age_h"])[:6]
    print("  examples with old userReserve update (units of token):")
    for x in old:
        d = 10 ** x["dec"]
        print(
            f"   {x['symbol']:>7} age_h={x['ur_age_h']:.0f} aTok stored={x['ca']/d:.6g} "
            f"scaled*idx={x['idx_a']/d:.6g} accrued={x['accr_a']/d:.6g} | "
            f"vDebt stored={x['cd']/d:.6g} scaled*idx={x['idx_d']/d:.6g} accrued={x['accr_d']/d:.6g}"
        )
    write_json(out, "balances.json", {"ts": ts, "examples": old, "mismatch_borrowedCount": mism})

    # Where does the stored value come from? Check the newest aToken history
    # item (at B1) of the user-reserve with the oldest update.
    cand = [(int(r["lastUpdateTimestamp"]), r) for u in users for r in u["reserves"]
            if int(r["currentATokenBalance"]) > 0]
    if cand:
        _, oldest = min(cand, key=lambda t: t[0])
        q = ("query($id:ID!,$b:Int!){ userReserve(id:$id, block:{number:$b}) { lastUpdateTimestamp "
             "aTokenBalanceHistory(first:1, orderBy:timestamp, orderDirection:desc) "
             "{ timestamp index scaledATokenBalance currentATokenBalance } } }")
        h = c.query(q, {"id": oldest["id"], "b": B1})["userReserve"]["aTokenBalanceHistory"]
        if h:
            it = h[0]
            same = ray_mul(int(it["scaledATokenBalance"]), int(it["index"])) == int(it["currentATokenBalance"])
            print(
                f"  history item ts={it['timestamp']} == userReserve.lastUpdateTimestamp "
                f"({oldest['lastUpdateTimestamp']}): {int(it['timestamp']) == int(oldest['lastUpdateTimestamp'])}; "
                f"history current == rayMul(scaled, index_at_that_time): {same}; "
                f"history current == userReserve.currentATokenBalance: {it['currentATokenBalance'] == oldest['currentATokenBalance']}"
            )


def step_blocks(c: Client, out: Path) -> None:
    print("== STEP 0: real timestamps of the reference blocks (RPC, not The Graph) ==")
    res: dict[str, Any] = {}
    for label, block in (("B0", B0), ("B1", B1), ("B_STRESS", B_STRESS)):
        ts = rpc_block_timestamp(block)
        iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None
        res[label] = {"block": block, "timestamp": ts, "utc": iso}
        print(f"  {label}: block {block} -> {ts} ({iso})")
    for label, day in (("2024-03-15 00:00", datetime(2024, 3, 15, tzinfo=timezone.utc)),
                       ("2024-08-05 00:00", datetime(2024, 8, 5, tzinfo=timezone.utc))):
        ref = res["B1"] if label.startswith("2024-03") else res["B_STRESS"]
        if ref["timestamp"]:
            est = ref["block"] + round((int(day.timestamp()) - ref["timestamp"]) / 12)
            print(f"  est. block at {label} UTC (12 s/block from the reference block): ~{est}")
    write_json(out, "blocks.json", res)


def step_reserves(c: Client, out: Path) -> None:
    print("== STEP 5: reserve params + eMode at B1 and stress block ==")
    keep = {"WETH", "wstETH", "WBTC", "UNI", "LINK", "USDC"}
    snap: dict[str, Any] = {}
    for label, block in (("baseline_B1", B1), ("stress", B_STRESS)):
        body = c.raw(Q_RESERVES_AT_BLOCK, {"block": block})
        if body.get("errors"):
            print(f"  {label}: ERRORS {json.dumps(body['errors'])[:400]}")
            continue
        data = body["data"]
        snap[label] = data
        print(f"  [{label}] block={block} reserves={len(data['reserves'])} emodeCategories={len(data['emodeCategories'])}")
        for r in data["reserves"]:
            if r["symbol"] in keep:
                print(
                    f"    {r['symbol']:>7} dec={r['decimals']} LTV={r['baseLTVasCollateral']} "
                    f"LT={r['reserveLiquidationThreshold']} LB={r['reserveLiquidationBonus']} "
                    f"collat={r['usageAsCollateralEnabled']} borrow={r['borrowingEnabled']} "
                    f"eMode={r['eMode']}"
                )
        for e in data["emodeCategories"]:
            print(f"    eMode {e['id']}: {e['label']} ltv={e['ltv']} lt={e['liquidationThreshold']} lb={e['liquidationBonus']}")
    write_json(out, "reserves_emode.json", snap)

    # Cross-check: do the *nested* reserve params inside a time-travelled users
    # page match reserves(block)? And how is User.eModeCategoryId distributed?
    nested_q = Q_USERS_SNAPSHOT_PAGE.replace(
        "variableBorrowIndex\n      }",
        "variableBorrowIndex\n        reserveLiquidationThreshold\n"
        "        reserveLiquidationBonus\n        baseLTVasCollateral\n      }",
    )
    assert nested_q != Q_USERS_SNAPSHOT_PAGE
    page = c.query(nested_q, {"first": 1000, "lastId": "", "block": B_STRESS})["users"]
    ref = {r["id"]: r for r in snap["stress"]["reserves"]}
    checked = bad = 0
    for u in page:
        for ur in u["reserves"]:
            rr = ur["reserve"]
            r0 = ref.get(rr["id"])
            if r0 is None:
                bad += 1
                continue
            checked += 1
            if (rr["reserveLiquidationThreshold"], rr["reserveLiquidationBonus"], rr["baseLTVasCollateral"]) != (
                r0["reserveLiquidationThreshold"], r0["reserveLiquidationBonus"], r0["baseLTVasCollateral"]
            ):
                bad += 1
    emode_counts: dict[str, int] = {}
    for u in page:
        k = (u["eModeCategoryId"] or {}).get("id", "none")
        emode_counts[k] = emode_counts.get(k, 0) + 1
    print(f"  nested reserve params vs reserves(block=stress): checked={checked} mismatches={bad}")
    print(f"  User.eModeCategoryId distribution in first 1000 borrowers at stress block: {emode_counts}")


def _paginate_events(c: Client, query: str, key: str, page_size: int = 1000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    last_id = ""
    while True:
        data = c.query(query, {"first": page_size, "lastId": last_id, "t0": T_START, "t1": T_END})
        page = data[key]
        rows.extend(page)
        if len(page) < page_size:
            return rows
        last_id = page[-1]["id"]
        time.sleep(PAGE_SLEEP_SECONDS)


def step_events(c: Client, out: Path) -> None:
    print("== STEP 6: events in 2024-03-14 00:00 .. 2024-03-15 00:00 UTC ==")
    liq = _paginate_events(c, Q_LIQUIDATIONS, "liquidationCalls")
    rep = _paginate_events(c, Q_REPAYS, "repays")
    print(f"  liquidationCalls: {len(liq)} ; repays: {len(rep)}")
    # Validate multi-page id_gt pagination by forcing small pages (first=100).
    rep_small = _paginate_events(c, Q_REPAYS, "repays", page_size=100)
    ids_small = [r["id"] for r in rep_small]
    print(
        f"  repays re-fetched with first=100 ({-(-len(rep_small) // 100)} pages): "
        f"same ids={ids_small == [r['id'] for r in rep]} unique={len(set(ids_small)) == len(ids_small)} "
        f"sorted={ids_small == sorted(ids_small)}"
    )
    for name, rows in (("liquidationCalls", liq), ("repays", rep)):
        if rows:
            ts = [int(r["timestamp"]) for r in rows]
            print(f"  {name}: ts range ok={T_START <= min(ts) and max(ts) < T_END}; sample={json.dumps(rows[0])[:700]}")
    if rep:
        print("  repays useATokens counts:", {k: sum(1 for r in rep if str(r["useATokens"]) == k) for k in ("True", "False", "None")})
    write_json(out, "events.json", {"liquidationCalls": liq[:5], "repays": rep[:5], "n_liq": len(liq), "n_rep": len(rep)})


def step_lean(c: Client, out: Path) -> None:
    print("== STEP 3b: lean snapshot query (recommended) over all borrowers at B1 ==")
    last_id, pages, total = "", 0, 0
    lat: list[float] = []
    kb: list[float] = []
    rows_total = zero_rows = stable_rows = 0
    first_page: list[dict[str, Any]] = []
    while True:
        t0 = time.perf_counter()
        body = c.query(Q_USERS_SNAPSHOT_LEAN, {"first": 1000, "lastId": last_id, "block": B1})
        lat.append(time.perf_counter() - t0)
        page = body["users"]
        kb.append(len(json.dumps(page)) / 1024)
        if pages == 0:
            first_page = page
        pages += 1
        total += len(page)
        for u in page:
            for r in u["reserves"]:
                rows_total += 1
                if int(r["scaledATokenBalance"]) == 0 and int(r["scaledVariableDebt"]) == 0                         and int(r["principalStableDebt"]) == 0:
                    zero_rows += 1
                if int(r["principalStableDebt"]) > 0:
                    stable_rows += 1
        if len(page) < 1000:
            break
        last_id = page[-1]["id"]
        time.sleep(PAGE_SLEEP_SECONDS)
    print(
        f"  pages={pages} users={total} userReserve rows={rows_total} "
        f"(all-zero rows={zero_rows}, stable-debt rows={stable_rows})"
    )
    print(
        f"  latency mean={sum(lat)/len(lat):.2f}s p90={pct(lat,.9):.2f}s ; payload per page "
        f"mean={sum(kb)/len(kb):.0f} KB (full query was ~2256 KB/page)"
    )
    full_first, _ = _users_page(c, B1, "", 1000)
    same = all(
        a["id"] == b["id"]
        and sorted((r["reserve"]["id"], r["scaledATokenBalance"], r["scaledVariableDebt"]) for r in a["reserves"])
        == sorted((r["reserve"]["id"], r["scaledATokenBalance"], r["scaledVariableDebt"]) for r in b["reserves"])
        for a, b in zip(first_page, full_first)
    ) and len(first_page) == len(full_first)
    print(f"  lean page 1 == full page 1 (ids + scaled balances): {same}")
    write_json(out, "lean.json", {"pages": pages, "users": total, "rows": rows_total, "zero_rows": zero_rows,
                                  "stable_rows": stable_rows, "latencies": lat, "kb": kb})


def step_windows(c: Client, out: Path) -> None:
    print("== STEP 6b: time-travel at the window edges (+-30 d) and event totals per 61-day window ==")
    ids_q = (
        "query($b:Int!){ users(first:1, orderBy:id, block:{number:$b}, "
        "where:{borrowedReservesCount_gt:0}) { id } }"
    )
    for label, centre in (("baseline", B1), ("stress", B_STRESS)):
        for off in (-30, +30):
            block = centre + off * 7_200
            body = c.raw(ids_q, {"b": block})
            ok = not body.get("errors")
            print(f"  {label} {off:+d}d block {block}: {'OK' if ok else 'ERROR ' + json.dumps(body['errors'])[:200]}")
    liq_q = Q_LIQUIDATIONS
    for label, centre in (("baseline", datetime(2024, 3, 15, tzinfo=timezone.utc)),
                          ("stress", datetime(2024, 8, 5, tzinfo=timezone.utc))):
        t0 = int(centre.timestamp()) - 30 * 86400
        t1 = int(centre.timestamp()) + 31 * 86400
        data = c.query(liq_q, {"first": 1000, "lastId": "", "t0": t0, "t1": t1})["liquidationCalls"]
        zero_coll = sum(1 for r in data if float(r["collateralAssetPriceUSD"]) == 0)
        zero_borr = sum(1 for r in data if float(r["borrowAssetPriceUSD"]) == 0)
        syms: dict[str, int] = {}
        for r in data:
            k = r["collateralReserve"]["symbol"]
            syms[k] = syms.get(k, 0) + 1
        print(
            f"  {label} window: liquidationCalls={len(data)} (page full: {len(data) == 1000}) "
            f"collateralAssetPriceUSD==0: {zero_coll}; borrowAssetPriceUSD==0: {zero_borr}; "
            f"by collateral: {dict(sorted(syms.items(), key=lambda kv: -kv[1])[:6])}"
        )


STEPS = {
    "blocks": step_blocks,
    "schema": step_schema,
    "timetravel": step_timetravel,
    "count": step_count,
    "lean": step_lean,
    "balances": step_balances,
    "reserves": step_reserves,
    "events": step_events,
    "windows": step_windows,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=[*STEPS, "all"])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "02_interim" / "spikes")
    args = parser.parse_args()
    client = Client()
    selected = list(STEPS) if args.step == "all" else [args.step]
    try:
        for name in selected:
            STEPS[name](client, args.out)
    except SpikeError as exc:
        print(f"SPIKE ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        mean = sum(client.latencies) / len(client.latencies) if client.latencies else float("nan")
        print(f"\n[api calls this run: {client.calls}; mean latency {mean:.2f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
