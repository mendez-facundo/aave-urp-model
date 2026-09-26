"""Static Aave v3 protocol constants and risk parameters for the ETL pipeline."""

from typing import Dict, List

STABLECOINS: List[str] = [
    "USDC",
    "USDT",
    "DAI",
    "FRAX",
    "LUSD",
    "GHO",
]

LST_ASSETS: List[str] = [
    "wstETH",
    "cbETH",
    "rETH",
]

PRICE_PROXIES: Dict[str, str] = {
    "ETH": "ETHUSDT",
    "WETH": "ETHUSDT",
    "wstETH": "ETHUSDT",
    "cbETH": "ETHUSDT",
    "rETH": "ETHUSDT",
    "WBTC": "BTCUSDT",
    "UNI": "UNIUSDT",
    "LINK": "LINKUSDT",
}

DEFAULT_LIQUIDATION_THRESHOLD: float = 0.80

LIQUIDATION_THRESHOLDS: Dict[str, float] = {
    "ETH": 0.825,
    "WETH": 0.825,
    "wstETH": 0.825,
    "cbETH": 0.825,
    "rETH": 0.825,
    "WBTC": 0.75,
    "UNI": 0.65,
    "LINK": 0.65,
    "USDC": DEFAULT_LIQUIDATION_THRESHOLD,
    "USDT": DEFAULT_LIQUIDATION_THRESHOLD,
    "DAI": DEFAULT_LIQUIDATION_THRESHOLD,
    "FRAX": DEFAULT_LIQUIDATION_THRESHOLD,
    "LUSD": DEFAULT_LIQUIDATION_THRESHOLD,
    "GHO": DEFAULT_LIQUIDATION_THRESHOLD,
    "DEFAULT": DEFAULT_LIQUIDATION_THRESHOLD,
}
