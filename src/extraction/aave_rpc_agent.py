"""Blind on-chain snapshot of Aave v3 collateral and debt at historical blocks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from aiohttp import ClientError, ClientResponseError
from dotenv import load_dotenv
from web3 import AsyncHTTPProvider, AsyncWeb3
from web3.contract import AsyncContract
from web3.exceptions import ContractLogicError, Web3Exception

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "01_raw"
BASELINE_DIR = RAW_DATA_DIR / "baseline_mar2024"
STRESS_DIR = RAW_DATA_DIR / "stress_aug2024"

load_dotenv(PROJECT_ROOT / ".env")

BASELINE_BLOCK = 19_433_100
STRESS_BLOCK = 20_457_180

AAVE_V3_DATA_PROVIDER = "0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3"

MAX_CONCURRENT_RPC_CALLS = 15
MAX_RETRIES = 8
RETRY_BASE_SECONDS = 1.0
RETRY_MAX_SECONDS = 60.0
REQUEST_TIMEOUT_SECONDS = 60
WALLET_CHUNK_SIZE = 25
PROGRESS_EVERY_CALLS = 250
INCLUDE_ZERO_POSITIONS = False

_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")

# Official Ethereum Mainnet underlying asset addresses listed on Aave v3.
TARGET_ASSETS: dict[str, str] = {
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    "FRAX": "0x853d955aCEf822Db058eb8505911ED77F175b99e",
    "wstETH": "0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0",
    "rETH": "0xae78736Cd615f374B308BD3F52Ea38009B31B0c7",
    "cbETH": "0xBe9895146f7AF43049ca1c1AE358B0541Ea49704",
}

AAVE_DATA_PROVIDER_ABI: list[dict[str, Any]] = [
    {
        "inputs": [
            {"internalType": "address", "name": "asset", "type": "address"},
            {"internalType": "address", "name": "user", "type": "address"},
        ],
        "name": "getUserReserveData",
        "outputs": [
            {"internalType": "uint256", "name": "currentATokenBalance", "type": "uint256"},
            {"internalType": "uint256", "name": "currentStableDebt", "type": "uint256"},
            {"internalType": "uint256", "name": "currentVariableDebt", "type": "uint256"},
            {"internalType": "uint256", "name": "principalStableDebt", "type": "uint256"},
            {"internalType": "uint256", "name": "scaledVariableDebt", "type": "uint256"},
            {"internalType": "uint256", "name": "liquidityRate", "type": "uint256"},
            {"internalType": "uint256", "name": "stableBorrowRate", "type": "uint256"},
            {"internalType": "uint256", "name": "stableRateLastUpdated", "type": "uint256"},
            {"internalType": "bool", "name": "usageAsCollateralEnabled", "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    }
]


class AlchemyRPCError(RuntimeError):
    """Raised when the Alchemy archive RPC cannot be queried successfully."""


class ReservePosition(TypedDict):
    """Single wallet-asset exposure observed at a historical block."""

    wallet: str
    asset_symbol: str
    asset_address: str
    current_a_token_balance: str
    current_stable_debt: str
    current_variable_debt: str
    total_debt: str
    usage_as_collateral_enabled: bool


class RateLimitGate:
    """Pause concurrent workers after HTTP 429 / CU-capacity errors."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._resume_at = 0.0

    async def wait_if_cooling_down(self) -> None:
        """Block until the shared cooldown window has elapsed."""
        while True:
            async with self._lock:
                remaining = self._resume_at - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(remaining)

    async def trigger_cooldown(self, seconds: float) -> None:
        """Extend the shared cooldown so workers do not stampede the RPC."""
        async with self._lock:
            self._resume_at = max(self._resume_at, time.monotonic() + seconds)


def _configure_event_loop() -> None:
    """Use the selector loop on Windows so aiohttp can run reliably."""
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _require_rpc_url() -> str:
    """Return ``ALCHEMY_RPC_URL`` or raise a configuration error."""
    rpc_url = os.getenv("ALCHEMY_RPC_URL", "").strip()
    if not rpc_url:
        raise AlchemyRPCError(
            "ALCHEMY_RPC_URL is missing. Set it in the project .env file."
        )
    return rpc_url


def _checksum(address: str) -> str:
    """Normalize an Ethereum address to EIP-55 checksum format."""
    return AsyncWeb3.to_checksum_address(address)


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True when the RPC failure is transient and should be retried."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    if isinstance(exc, ClientResponseError) and exc.status in {429, 500, 502, 503, 504}:
        return True
    if isinstance(exc, ClientError):
        return True

    message = str(exc).lower()
    retryable_tokens = (
        "429",
        "too many requests",
        "rate limit",
        "over rate limit",
        "compute units",
        "capacity",
        "timeout",
        "temporar",
        "connection reset",
        "server disconnected",
        "503",
        "502",
        "504",
        "http 429",
    )
    return any(token in message for token in retryable_tokens)


def _is_rate_limited(exc: BaseException) -> bool:
    """Return True when Alchemy is throttling the caller."""
    if isinstance(exc, ClientResponseError) and exc.status == 429:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "429",
            "too many requests",
            "rate limit",
            "over rate limit",
            "compute units",
        )
    )


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, capped to ``RETRY_MAX_SECONDS``."""
    base = min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
    jitter = random.uniform(0.0, 0.25 * base)
    return base + jitter


def _parse_user_reserve_data(result: Any) -> tuple[int, int, int, bool]:
    """Unpack ``getUserReserveData`` into collateral, debts and collateral flag."""
    try:
        a_token = int(result[0])
        stable_debt = int(result[1])
        variable_debt = int(result[2])
        usage_enabled = bool(result[8])
    except (TypeError, ValueError, IndexError) as exc:
        raise AlchemyRPCError(
            f"Unexpected getUserReserveData payload: {result!r}"
        ) from exc
    return a_token, stable_debt, variable_debt, usage_enabled


def load_wallets(wallets_path: Path) -> list[str]:
    """Load unique lowercase wallet addresses from ``all_active_wallets.json``.

    Parameters
    ----------
    wallets_path:
        JSON file produced by ``extract_active_wallets.py``. Expected payload
        is a list of address strings.

    Returns
    -------
    list[str]
        Deduplicated, lowercase Ethereum addresses, preserving first-seen order.
    """
    if not wallets_path.is_file():
        raise FileNotFoundError(
            f"Wallet universe file not found: {wallets_path}. "
            "Run extract_active_wallets.py before the RPC snapshot."
        )

    with wallets_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(
            f"Expected a JSON array of wallets in {wallets_path}, got {type(payload)}."
        )

    wallets: list[str] = []
    seen: set[str] = set()
    skipped = 0

    for item in payload:
        raw = item.get("id") if isinstance(item, dict) else item
        if not isinstance(raw, str) or not _ADDRESS_RE.match(raw.strip()):
            skipped += 1
            continue
        wallet = raw.strip().lower()
        if wallet in seen:
            continue
        seen.add(wallet)
        wallets.append(wallet)

    if skipped:
        logger.warning(
            "Skipped %s invalid wallet entries in %s.",
            skipped,
            wallets_path,
        )
    if not wallets:
        raise ValueError(f"No valid wallet addresses found in {wallets_path}.")

    logger.info("Loaded %s unique wallets from %s.", len(wallets), wallets_path)
    return wallets


def save_snapshot_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write the RPC snapshot payload as UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    logger.info(
        "Wrote %s positions for %s wallets to %s.",
        payload.get("position_count", 0),
        payload.get("wallet_count", 0),
        output_path,
    )


async def _disconnect_provider(w3: AsyncWeb3) -> None:
    """Close the HTTP session if the provider exposes a disconnect hook."""
    disconnect = getattr(w3.provider, "disconnect", None)
    if not callable(disconnect):
        return
    result = disconnect()
    if asyncio.iscoroutine(result) or asyncio.isfuture(result):
        await result


async def _build_web3() -> AsyncWeb3:
    """Create an ``AsyncWeb3`` client bound to the Alchemy archive endpoint."""
    rpc_url = _require_rpc_url()
    w3 = AsyncWeb3(
        AsyncHTTPProvider(
            rpc_url,
            request_kwargs={"timeout": REQUEST_TIMEOUT_SECONDS},
        )
    )
    if not await w3.is_connected():
        raise AlchemyRPCError("Unable to connect to ALCHEMY_RPC_URL.")
    logger.info("Connected to Alchemy archive RPC (endpoint redacted).")
    return w3


async def _call_user_reserve_data(
    contract: AsyncContract,
    asset: str,
    wallet: str,
    block_number: int,
    semaphore: asyncio.Semaphore,
    rate_gate: RateLimitGate,
) -> tuple[int, int, int, bool]:
    """Call ``getUserReserveData`` at ``block_number`` with retries.

    Concurrency is capped by ``semaphore``. HTTP 429 and other transient
    Alchemy errors trigger exponential backoff and a shared cooldown gate.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        await rate_gate.wait_if_cooling_down()
        try:
            async with semaphore:
                result = await contract.functions.getUserReserveData(
                    asset,
                    wallet,
                ).call(block_identifier=block_number)
            return _parse_user_reserve_data(result)
        except ContractLogicError as exc:
            logger.warning(
                "getUserReserveData reverted for %s/%s at block %s: %s. "
                "Treating the position as zero exposure.",
                wallet,
                asset,
                block_number,
                exc,
            )
            return 0, 0, 0, False
        except (Web3Exception, ClientError, asyncio.TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if not _is_retryable_error(exc) or attempt == MAX_RETRIES:
                break

            wait_seconds = _backoff_seconds(attempt)
            if _is_rate_limited(exc):
                await rate_gate.trigger_cooldown(wait_seconds)
                logger.warning(
                    "Alchemy rate limit on %s/%s (attempt %s/%s). Cooling down %.1fs.",
                    wallet,
                    asset,
                    attempt,
                    MAX_RETRIES,
                    wait_seconds,
                )
            else:
                logger.warning(
                    "RPC call failed for %s/%s (attempt %s/%s): %s. Retrying in %.1fs.",
                    wallet,
                    asset,
                    attempt,
                    MAX_RETRIES,
                    exc,
                    wait_seconds,
                )
            await asyncio.sleep(wait_seconds)

    raise AlchemyRPCError(
        f"Failed getUserReserveData(asset={asset}, user={wallet}) "
        f"at block {block_number} after {MAX_RETRIES} attempts."
    ) from last_error


async def _fetch_single_position(
    contract: AsyncContract,
    wallet: str,
    asset_symbol: str,
    asset_address: str,
    block_number: int,
    semaphore: asyncio.Semaphore,
    rate_gate: RateLimitGate,
    progress: dict[str, int],
    progress_lock: asyncio.Lock,
    total_calls: int,
) -> ReservePosition | None:
    """Fetch one wallet-asset pair and optionally drop empty balances."""
    checksum_wallet = _checksum(wallet)
    a_token, stable_debt, variable_debt, usage_enabled = await _call_user_reserve_data(
        contract=contract,
        asset=asset_address,
        wallet=checksum_wallet,
        block_number=block_number,
        semaphore=semaphore,
        rate_gate=rate_gate,
    )
    total_debt = stable_debt + variable_debt

    async with progress_lock:
        progress["completed"] += 1
        completed = progress["completed"]
        if total_debt == 0 and a_token == 0:
            progress["zero_positions"] += 1
        if completed % PROGRESS_EVERY_CALLS == 0 or completed == total_calls:
            logger.info(
                "RPC progress: %s/%s calls (%.1f%%) at block %s.",
                completed,
                total_calls,
                100.0 * completed / total_calls,
                block_number,
            )

    if not INCLUDE_ZERO_POSITIONS and a_token == 0 and total_debt == 0:
        return None

    return ReservePosition(
        wallet=wallet,
        asset_symbol=asset_symbol,
        asset_address=asset_address.lower(),
        current_a_token_balance=str(a_token),
        current_stable_debt=str(stable_debt),
        current_variable_debt=str(variable_debt),
        total_debt=str(total_debt),
        usage_as_collateral_enabled=usage_enabled,
    )


async def fetch_snapshot_at_block(
    block_number: int,
    wallets_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Extract a cross-sectional Aave v3 snapshot at a historical block.

    The function reads the full wallet universe from ``wallets_path`` and
    queries ``AaveProtocolDataProvider.getUserReserveData`` for every
    target asset, pinning state with ``block_identifier``.

    Parameters
    ----------
    block_number:
        Ethereum block height used as the archive-state identifier.
    wallets_path:
        JSON array of wallet addresses (``all_active_wallets.json``).
    output_path:
        Optional destination for ``alchemy_snapshot.json``.

    Returns
    -------
    dict[str, Any]
        Snapshot payload with metadata and per-wallet reserve positions.
        ``current_a_token_balance`` is collateral; ``total_debt`` is
        ``current_stable_debt + current_variable_debt``. Amounts are
        native-token wei strings to preserve uint256 precision.
    """
    if block_number <= 0:
        raise ValueError(f"block_number must be a positive integer, got {block_number}.")

    wallets = load_wallets(wallets_path)
    checksum_assets = {
        symbol: _checksum(address) for symbol, address in TARGET_ASSETS.items()
    }
    total_calls = len(wallets) * len(checksum_assets)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_RPC_CALLS)
    rate_gate = RateLimitGate()
    progress = {"completed": 0, "zero_positions": 0}
    progress_lock = asyncio.Lock()
    positions: list[ReservePosition] = []

    w3 = await _build_web3()
    try:
        latest = await w3.eth.block_number
        if block_number > latest:
            raise AlchemyRPCError(
                f"Requested block {block_number} is ahead of the node head {latest}."
            )
        logger.info(
            "Fetching Aave v3 snapshot at block %s for %s wallets x %s assets "
            "(%s eth_call requests, max concurrency=%s).",
            block_number,
            len(wallets),
            len(checksum_assets),
            total_calls,
            MAX_CONCURRENT_RPC_CALLS,
        )

        contract = w3.eth.contract(
            address=_checksum(AAVE_V3_DATA_PROVIDER),
            abi=AAVE_DATA_PROVIDER_ABI,
        )

        for chunk_start in range(0, len(wallets), WALLET_CHUNK_SIZE):
            wallet_chunk = wallets[chunk_start : chunk_start + WALLET_CHUNK_SIZE]
            tasks = [
                _fetch_single_position(
                    contract=contract,
                    wallet=wallet,
                    asset_symbol=symbol,
                    asset_address=asset_address,
                    block_number=block_number,
                    semaphore=semaphore,
                    rate_gate=rate_gate,
                    progress=progress,
                    progress_lock=progress_lock,
                    total_calls=total_calls,
                )
                for wallet in wallet_chunk
                for symbol, asset_address in checksum_assets.items()
            ]
            chunk_results = await asyncio.gather(*tasks)
            positions.extend(item for item in chunk_results if item is not None)
    finally:
        await _disconnect_provider(w3)

    payload: dict[str, Any] = {
        "protocol": "aave-v3",
        "network": "ethereum",
        "data_provider": AAVE_V3_DATA_PROVIDER.lower(),
        "block_number": block_number,
        "assets": {symbol: address.lower() for symbol, address in TARGET_ASSETS.items()},
        "wallet_count": len(wallets),
        "rpc_call_count": total_calls,
        "zero_position_count": progress["zero_positions"],
        "position_count": len(positions),
        "include_zero_positions": INCLUDE_ZERO_POSITIONS,
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "positions": positions,
    }

    logger.info(
        "Completed snapshot at block %s: %s non-empty positions "
        "(%s zero positions omitted=%s).",
        block_number,
        len(positions),
        progress["zero_positions"],
        not INCLUDE_ZERO_POSITIONS,
    )

    if output_path is not None:
        save_snapshot_json(payload, output_path)

    return payload


def ensure_raw_scenario_directories() -> None:
    """Create raw data scenario directories if they do not already exist."""
    for directory in (BASELINE_DIR, STRESS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory exists: %s", directory)


async def run_extraction() -> None:
    """Run the baseline and stress RPC snapshots sequentially."""
    ensure_raw_scenario_directories()

    scenarios: list[tuple[str, int, Path, Path]] = [
        (
            "baseline",
            BASELINE_BLOCK,
            BASELINE_DIR / "all_active_wallets.json",
            BASELINE_DIR / "alchemy_snapshot.json",
        ),
        (
            "stress",
            STRESS_BLOCK,
            STRESS_DIR / "all_active_wallets.json",
            STRESS_DIR / "alchemy_snapshot.json",
        ),
    ]

    for name, block_number, wallets_path, output_path in scenarios:
        logger.info(
            "Starting %s RPC snapshot at block %s (%s -> %s).",
            name,
            block_number,
            wallets_path,
            output_path,
        )
        try:
            await fetch_snapshot_at_block(
                block_number=block_number,
                wallets_path=wallets_path,
                output_path=output_path,
            )
            logger.info("Finished %s RPC snapshot at block %s.", name, block_number)
        except (AlchemyRPCError, FileNotFoundError, ValueError, OSError) as exc:
            logger.error(
                "Failed %s RPC snapshot at block %s: %s",
                name,
                block_number,
                exc,
            )
            raise


def main() -> None:
    """CLI entry point for the two-scenario archive snapshot."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _configure_event_loop()
    asyncio.run(run_extraction())


if __name__ == "__main__":
    main()
