# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Extreme Value Theory: GPD Calibration of Collateral Tail Risk (Peaks Over Threshold)
#
# This notebook estimates the **99% Expected Shortfall** ($ES_{0.99}$) of daily price losses for the four main Aave collateral assets (**ETH, BTC, UNI, LINK**) using the **Peaks Over Threshold (POT)** approach and the **Generalized Pareto Distribution (GPD)**.
#
# Both valuation windows are processed in parallel, each covering roughly two years of Binance daily closes:
#
# - **Baseline scenario:** window ending 15 March 2024 (`data/01_raw/baseline_mar2024/`)
# - **Stress scenario:** window ending 5 August 2024 (`data/01_raw/stress_aug2024/`)
#
# Pipeline:
#
# 1. Continuously compounded returns $r_t = \ln P_t - \ln P_{t-1}$ and the loss variable $X_t = -r_t$ (the market's left tail becomes the right tail modelled by classical EVT).
# 2. Threshold diagnostics: **Mean Residual Life Plot (MRLP)** and **parameter stability plots** for the shape $\xi$ and modified scale $\sigma^*$.
# 3. Selection of the threshold $u^*$ inside the 85th-99th percentile band of the losses.
# 4. **Maximum Likelihood** GPD fit of the excesses $Y = X - u^*$ and verification of the heavy-tail (Fréchet) condition $\xi > 0$.
# 5. Goodness of fit: **Q-Q plot** and **Anderson-Darling** $A^2$ test with a parametric-bootstrap p-value.
# 6. Closed-form $VaR_{0.99}$ and $ES_{0.99}$, plus a **bootstrap 95% confidence interval** for $ES_{0.99}$.
# 7. One consolidated parameter table per scenario ($u^*$, $\xi$, $\sigma$, $ES_{0.99}$).
# 8. Extensions on the fitted tail: $ES_p$ at $p \in \{0.99, 0.995, 0.999\}$ (Section 16), the ES of the $h$-day loss by Monte Carlo (Section 17), and the **break-even price shocks** implied by the real Aave parameters (Sections 18-19).
#
# Unless stated otherwise, all risk measures are expressed in **daily log-loss units** (e.g. $0.10$ means a daily log-return of $-0.10$).
#
# This notebook is strictly for **exploratory analysis and stochastic validation**: it reads the raw price extracts and the Aave reserve parameters, and keeps every result in memory. The only files it writes are the manual test dumps of Sections 20-21 (`*_not_for_prod.csv`).

# %% [markdown]
# ## 1. Imports and configuration

# %%
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.ticker import PercentFormatter

sys.path.append(str(Path("..").resolve()))  # project root, so that ``src.utils`` is importable from notebooks/
from src.utils.protocol_params import get_param, load_reserve_params  # noqa: E402

try:
    from scipy import stats
    from scipy.stats import genpareto

    STATS_BACKEND: str = "scipy.stats"
except ImportError:
    # Windows Application Control can block a scipy DLL that scipy.stats needs (see src/utils/stats_compat.py).
    from src.utils import stats_compat as stats
    from src.utils.stats_compat import genpareto

    STATS_BACKEND = stats.BACKEND_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
    force=True,
)
logger = logging.getLogger("evt_gpd")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (10, 5),
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.frameon": True,
        "axes.spines.top": False,
    }
)

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

RAW_DATA_DIR = Path("../data/01_raw")
SCENARIO_DIRS: list[str] = ["baseline_mar2024", "stress_aug2024"]
SCENARIO_KEY: dict[str, str] = {
    "baseline_mar2024": "baseline",
    "stress_aug2024": "stress",
}
SCENARIOS: list[str] = [SCENARIO_KEY[folder] for folder in SCENARIO_DIRS]
ASSETS: list[str] = ["ETH", "BTC", "UNI", "LINK"]

CONFIDENCE_LEVEL: float = 0.99

THRESHOLD_BAND: tuple[float, float] = (0.85, 0.99)
BAND_LABEL: str = f"q{THRESHOLD_BAND[0] * 100:g}-q{THRESHOLD_BAND[1] * 100:g} band"
MRL_QUANTILE_GRID: np.ndarray = np.round(np.arange(0.50, 0.99 + 1e-9, 0.005), 4)
STABILITY_QUANTILE_GRID: np.ndarray = np.round(np.arange(0.60, 0.99 + 1e-9, 0.005), 4)
STABILITY_WINDOW: int = 5
MIN_EXCEEDANCES: int = 30

for grid_name, grid in (("MRL_QUANTILE_GRID", MRL_QUANTILE_GRID), ("STABILITY_QUANTILE_GRID", STABILITY_QUANTILE_GRID)):
    if grid.min() > THRESHOLD_BAND[0] or grid.max() < THRESHOLD_BAND[1] - 1e-9:
        raise ValueError(f"{grid_name} [{grid.min()}, {grid.max()}] does not cover THRESHOLD_BAND {THRESHOLD_BAND}")

# Manual overrides after visual inspection, as {(scenario, asset): loss quantile}, e.g. {("stress", "UNI"): 0.92}
THRESHOLD_OVERRIDES: dict[tuple[str, str], float] = {}

CI_LEVEL: float = 0.95
Z_CRIT: float = float(stats.norm.ppf(0.5 + CI_LEVEL / 2))
N_BOOTSTRAP_ES: int = 1000
N_BOOTSTRAP_AD: int = 999
RANDOM_SEED: int = 42
XI_ZERO_TOL: float = 1e-8
CDF_CLIP: float = 1e-12

SCENARIO_COLORS: dict[str, str] = {"baseline": "#1f77b4", "stress": "#d62728"}
RNG_STAGES: dict[str, int] = {"anderson_darling": 0, "es_bootstrap": 1, "horizon_mc": 2}


def make_rng(scenario_key: str, asset: str, stage: str) -> np.random.Generator:
    """Return a reproducible generator that does not depend on cell execution order."""
    return np.random.default_rng(
        [RANDOM_SEED, SCENARIOS.index(scenario_key), ASSETS.index(asset), RNG_STAGES[stage]]
    )


print(f"Scenarios: {SCENARIOS} | Assets: {ASSETS}")
print(f"Confidence level p = {CONFIDENCE_LEVEL} | Threshold band (loss quantiles) = {THRESHOLD_BAND}")
print(f"Bootstrap replicates: ES = {N_BOOTSTRAP_ES}, Anderson-Darling = {N_BOOTSTRAP_AD}")
print(f"Statistics backend: {STATS_BACKEND}")

# %% [markdown]
# ## 2. Market data ingestion
#
# For each scenario folder, the single long-format Binance extract `prices/binance_daily.csv` (columns `date`, `ticker`, `timestamp`, `open`, `high`, `low`, `close`, `volume`; one row per day and ticker, e.g. `ETHUSDT`) is loaded. Only the **close** price is kept: the rows are validated per ticker and then pivoted to one column per asset, which is split into one series per asset (ETH, BTC, UNI, LINK). Each series is validated before use:
#
# - required columns (`date`, `ticker`, `close`) are present, and every expected ticker (`{asset}USDT`) is in the file;
# - dates are unique per ticker and sorted;
# - close prices are strictly positive (a requirement of the logarithm);
# - calendar gaps are reported, because a log return across a gap would aggregate more than one day of risk.
#
# Intraday data are not used for EVT (see `docs/decisions.md`), and the per-ticker extracts (`binance_prices_*.csv`) are kept on disk but no longer read.

# %%
BINANCE_DAILY_FILE: str = "prices/binance_daily.csv"
BINANCE_QUOTE: str = "USDT"
REQUIRED_PRICE_COLUMNS: tuple[str, ...] = ("date", "ticker", "close")


def load_close_prices(scenario_dir: Path, assets: list[str]) -> dict[str, pd.Series]:
    """Load and validate the Binance daily close series of several assets from the long-format extract.

    Args:
        scenario_dir: Scenario folder under ``data/01_raw``.
        assets: Asset tickers without quote currency, e.g. ``["ETH", "BTC"]``.

    Returns:
        Close prices of each asset, indexed by calendar date and named after the asset.

    Raises:
        FileNotFoundError: If the CSV extract does not exist.
        ValueError: If required columns or tickers are missing, dates are duplicated within
            a ticker, or close prices are missing or non-positive.
    """
    csv_path = scenario_dir / BINANCE_DAILY_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"Price extract not found: {csv_path.resolve()}")

    frame = pd.read_csv(csv_path)
    missing_columns = set(REQUIRED_PRICE_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing_columns)}")

    ticker_to_asset = {f"{asset}{BINANCE_QUOTE}": asset for asset in assets}
    missing_tickers = sorted(set(ticker_to_asset) - set(frame["ticker"]))
    if missing_tickers:
        raise ValueError(f"{csv_path.name} is missing tickers: {missing_tickers}")

    frame = frame.loc[frame["ticker"].isin(ticker_to_asset), list(REQUIRED_PRICE_COLUMNS)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["asset"] = frame["ticker"].map(ticker_to_asset)
    frame = frame.sort_values(["asset", "date"])

    if frame.duplicated(["asset", "date"]).any():
        raise ValueError(f"{csv_path.name} contains duplicated dates")
    if frame["close"].isna().any() or (frame["close"] <= 0).any():
        raise ValueError(f"{csv_path.name} contains missing or non-positive close prices")

    for asset, group in frame.groupby("asset"):
        n_gaps = int((group["date"].diff().dt.days.dropna() != 1).sum())
        if n_gaps:
            logger.warning(
                "%s [%s]: %d calendar gaps detected; affected returns span several days", csv_path, asset, n_gaps
            )

    wide = frame.pivot(index="date", columns="asset", values="close").sort_index()
    if wide.isna().any().any():
        logger.warning("%s: tickers do not share the same calendar dates; each series keeps its own dates", csv_path)
    return {asset: wide[asset].dropna().rename(asset) for asset in assets}


close_prices: dict[str, dict[str, pd.Series]] = {}
for folder_name in SCENARIO_DIRS:
    scenario_key = SCENARIO_KEY[folder_name]
    scenario_dir = RAW_DATA_DIR / folder_name
    print(f"[{scenario_key}] {scenario_dir.resolve()}")
    close_prices[scenario_key] = load_close_prices(scenario_dir, ASSETS)

price_coverage_df = pd.DataFrame(
    [
        {
            "scenario": scenario_key,
            "asset": asset,
            "n_prices": series.size,
            "start": series.index.min().date(),
            "end": series.index.max().date(),
            "first_close": series.iloc[0],
            "last_close": series.iloc[-1],
        }
        for scenario_key, by_asset in close_prices.items()
        for asset, series in by_asset.items()
    ]
).set_index(["scenario", "asset"])

display(price_coverage_df)


# %% [markdown]
# ## 3. Log returns and the loss variable
#
# Daily continuously compounded returns:
#
# $$
# r_t = \ln(P_t) - \ln(P_{t-1})
# $$
#
# Classical EVT (Pickands-Balkema-de Haan theorem) describes the **right** tail of a distribution, whereas the credit risk of a collateralized loan lies in extreme price **drops** (the left tail of returns). The loss variable is therefore defined strictly as the negative log return:
#
# $$
# X_t = -r_t
# $$
#
# so that large positive values of $X_t$ are severe crashes. The table below reports the empirical moments and quantiles of $X_t$; positive skewness and large excess kurtosis are the first evidence of tails heavier than the Gaussian.

# %%
def compute_log_losses(close: pd.Series) -> pd.Series:
    """Return daily losses X_t = -(ln P_t - ln P_{t-1}) indexed by the date of P_t."""
    log_returns = np.log(close).diff().dropna()
    return (-log_returns).rename(close.name)


def describe_losses(loss: pd.Series) -> dict[str, object]:
    """Summarize the empirical loss distribution with an emphasis on tail indicators."""
    values = loss.to_numpy()
    return {
        "n_obs": values.size,
        "mean": values.mean(),
        "std": values.std(ddof=1),
        "skewness": stats.skew(values, bias=False),
        "excess_kurtosis": stats.kurtosis(values, fisher=True, bias=False),
        "q90": np.quantile(values, 0.90),
        "q95": np.quantile(values, 0.95),
        "q99": np.quantile(values, 0.99),
        "max_loss": values.max(),
        "max_loss_date": loss.idxmax().date(),
    }


losses: dict[str, dict[str, pd.Series]] = {
    scenario_key: {asset: compute_log_losses(series) for asset, series in by_asset.items()}
    for scenario_key, by_asset in close_prices.items()
}

loss_summary_df = pd.DataFrame(
    [
        {"scenario": scenario_key, "asset": asset, **describe_losses(loss)}
        for scenario_key, by_asset in losses.items()
        for asset, loss in by_asset.items()
    ]
).set_index(["scenario", "asset"])

display(loss_summary_df.style.format(precision=4).set_caption("Empirical distribution of daily log-losses X_t"))


# %% [markdown]
# ## 4. GPD estimation toolkit
#
# For a threshold $u$, the excesses $Y = X - u \mid X > u$ are approximated by the GPD:
#
# $$
# G_{\xi,\sigma}(y) = 1 - \left(1 + \frac{\xi\, y}{\sigma}\right)^{-1/\xi}, \qquad y \ge 0,\; \sigma > 0
# $$
#
# with the exponential limit $G_{0,\sigma}(y) = 1 - e^{-y/\sigma}$. `scipy.stats.genpareto` uses the same parameterization (shape `c` $= \xi$, `scale` $= \sigma$); the location is fixed at `floc=0` because the model is fitted to excesses.
#
# Helpers defined below:
#
# - `fit_gpd`: MLE of $(\xi, \sigma)$; returns `None` instead of raising when the optimizer fails, so bootstrap loops can discard failed replicates.
# - `gpd_asymptotic_cov`: inverse expected Fisher information (Hosking & Wallis, 1987), valid for $\xi > -0.5$:
#
# $$
# \operatorname{Cov}(\hat\xi, \hat\sigma) \approx \frac{1+\xi}{N_u}\begin{pmatrix} 1+\xi & -\sigma \\ -\sigma & 2\sigma^2 \end{pmatrix}
# $$
#
# - `modified_scale`: $\sigma^* = \sigma_u - \xi u$ with a delta-method standard error, $\operatorname{Var}(\hat\sigma^*) = \operatorname{Var}(\hat\sigma) + u^2 \operatorname{Var}(\hat\xi) - 2u\operatorname{Cov}(\hat\xi, \hat\sigma)$.
# - `xi_zero_lrt`: likelihood ratio test of $H_0: \xi = 0$ (exponential, Gumbel domain) against $H_1: \xi > 0$ (Fréchet domain), using the signed root $r = \operatorname{sign}(\hat\xi)\sqrt{2(\ell_1 - \ell_0)} \sim N(0,1)$ under $H_0$.

# %%
@dataclass(frozen=True)
class GPDFit:
    """Maximum-likelihood GPD fit on threshold excesses."""

    xi: float
    sigma: float
    n_exceedances: int
    log_likelihood: float


def get_exceedances(loss: pd.Series, threshold: float) -> pd.Series:
    """Return the excesses Y = X - u of the losses strictly above the threshold, keeping their dates."""
    return loss[loss > threshold] - threshold


def fit_gpd(excesses: np.ndarray) -> GPDFit | None:
    """Fit a GPD with location fixed at zero by maximum likelihood.

    Args:
        excesses: Non-negative threshold excesses.

    Returns:
        The fitted parameters, or ``None`` if the optimizer fails or returns
        non-finite estimates.

    Raises:
        ValueError: If any excess is negative.
    """
    excesses = np.asarray(excesses, dtype=float)
    if np.any(excesses < 0):
        raise ValueError("Threshold excesses must be non-negative")
    if excesses.size < 3:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xi, _, sigma = genpareto.fit(excesses, floc=0.0)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        logger.debug("GPD MLE failed on %d excesses: %s", excesses.size, exc)
        return None

    log_likelihood = float(np.sum(genpareto.logpdf(excesses, xi, loc=0.0, scale=sigma)))
    if not (np.isfinite(xi) and np.isfinite(sigma) and sigma > 0 and np.isfinite(log_likelihood)):
        return None
    return GPDFit(xi=float(xi), sigma=float(sigma), n_exceedances=int(excesses.size), log_likelihood=log_likelihood)


def gpd_asymptotic_cov(fit: GPDFit) -> np.ndarray:
    """Asymptotic covariance of (xi_hat, sigma_hat); NaN where MLE regularity fails (xi <= -0.5)."""
    if fit.xi <= -0.5:
        return np.full((2, 2), np.nan)
    factor = (1.0 + fit.xi) / fit.n_exceedances
    return factor * np.array([[1.0 + fit.xi, -fit.sigma], [-fit.sigma, 2.0 * fit.sigma**2]])


def modified_scale(fit: GPDFit, threshold: float) -> tuple[float, float]:
    """Return the threshold-invariant scale sigma* = sigma_u - xi * u and its delta-method standard error."""
    sigma_star = fit.sigma - fit.xi * threshold
    gradient = np.array([-threshold, 1.0])
    variance = float(gradient @ gpd_asymptotic_cov(fit) @ gradient)
    return sigma_star, float(np.sqrt(variance)) if variance >= 0 else np.nan


def xi_zero_lrt(excesses: np.ndarray, fit: GPDFit) -> tuple[float, float]:
    """Signed-root likelihood ratio test of H0: xi = 0 against H1: xi > 0.

    Returns:
        The signed-root statistic and its one-sided p-value.
    """
    excesses = np.asarray(excesses, dtype=float)
    n = excesses.size
    sigma_exponential = excesses.mean()
    log_likelihood_null = -n * np.log(sigma_exponential) - n
    deviance = max(2.0 * (fit.log_likelihood - log_likelihood_null), 0.0)
    signed_root = float(np.sign(fit.xi) * np.sqrt(deviance))
    return signed_root, float(stats.norm.sf(signed_root))


# %% [markdown]
# ## 5. Threshold diagnostics on a quantile grid
#
# Thresholds are indexed by **empirical loss quantiles** so that the grids are comparable across assets with different volatilities.
#
# **Mean Residual Life (mean excess) function.** If the excesses over some $u_0$ follow a GPD with $\xi < 1$, then for every $u > u_0$:
#
# $$
# e(u) = \mathbb{E}[X - u \mid X > u] = \frac{\sigma_{u_0} + \xi\,(u - u_0)}{1 - \xi}
# $$
#
# so the empirical mean excess $\hat e(u)$ must be approximately **linear** in $u$ above a valid threshold, with slope $\xi / (1 - \xi)$ (an upward slope indicates $\xi > 0$). Pointwise confidence bands are $\hat e(u) \pm z_{0.975}\, s_u / \sqrt{N_u}$. The MRL grid runs from the 50th to the 99th percentile.
#
# **Parameter stability.** Above a valid threshold, $\hat\xi(u)$ and the modified scale $\hat\sigma^*(u) = \hat\sigma_u - \hat\xi u$ must be approximately **constant** (up to sampling noise). The GPD is refitted on the 60th-99th percentile grid in steps of 0.5 percentile points, with Wald 95% intervals from Section 4.
#
# Both grids start well below the 85th-99th percentile selection band, so each plot shows how the curves behave before entering the band, and the configuration cell checks that the grids cover the band. At the upper end of the stability grid, the fits rest on fewer than 20 exceedances and their confidence bands widen sharply; those points are shown for completeness but carry little information.

# %%
def mean_residual_life(loss: pd.Series, quantile_grid: np.ndarray) -> pd.DataFrame:
    """Empirical mean excess function with pointwise normal confidence bands."""
    values = loss.to_numpy()
    rows: list[dict[str, float]] = []
    for quantile in quantile_grid:
        threshold = float(np.quantile(values, quantile))
        excesses = values[values > threshold] - threshold
        if excesses.size < 2:
            continue
        mean_excess = excesses.mean()
        standard_error = excesses.std(ddof=1) / np.sqrt(excesses.size)
        rows.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "n_exceedances": excesses.size,
                "mean_excess": mean_excess,
                "ci_low": mean_excess - Z_CRIT * standard_error,
                "ci_high": mean_excess + Z_CRIT * standard_error,
            }
        )
    return pd.DataFrame(rows)


def parameter_stability(loss: pd.Series, quantile_grid: np.ndarray) -> pd.DataFrame:
    """Refit the GPD over a threshold grid and report xi and sigma* with Wald confidence intervals."""
    rows: list[dict[str, float]] = []
    for quantile in quantile_grid:
        threshold = float(np.quantile(loss, quantile))
        excesses = get_exceedances(loss, threshold).to_numpy()
        fit = fit_gpd(excesses)
        if fit is None:
            logger.warning("%s: GPD fit failed at quantile %.3f; grid point skipped", loss.name, quantile)
            continue
        xi_se = float(np.sqrt(gpd_asymptotic_cov(fit)[0, 0]))
        sigma_star, sigma_star_se = modified_scale(fit, threshold)
        rows.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "n_exceedances": fit.n_exceedances,
                "xi": fit.xi,
                "xi_ci_low": fit.xi - Z_CRIT * xi_se,
                "xi_ci_high": fit.xi + Z_CRIT * xi_se,
                "sigma": fit.sigma,
                "sigma_star": sigma_star,
                "sigma_star_ci_low": sigma_star - Z_CRIT * sigma_star_se,
                "sigma_star_ci_high": sigma_star + Z_CRIT * sigma_star_se,
            }
        )
    return pd.DataFrame(rows)


mrl_tables: dict[str, dict[str, pd.DataFrame]] = {scenario_key: {} for scenario_key in SCENARIOS}
stability_tables: dict[str, dict[str, pd.DataFrame]] = {scenario_key: {} for scenario_key in SCENARIOS}

for scenario_key in SCENARIOS:
    for asset in ASSETS:
        loss = losses[scenario_key][asset]
        mrl_tables[scenario_key][asset] = mean_residual_life(loss, MRL_QUANTILE_GRID)
        stability_tables[scenario_key][asset] = parameter_stability(loss, STABILITY_QUANTILE_GRID)

print(f"MRL grid points: {MRL_QUANTILE_GRID.size} | stability grid points: {STABILITY_QUANTILE_GRID.size}")
print("Stability preview (baseline ETH, threshold band):")
preview = stability_tables["baseline"]["ETH"]
display(preview[preview["quantile"].between(THRESHOLD_BAND[0] - 1e-9, THRESHOLD_BAND[1] + 1e-9)].round(4))


# %% [markdown]
# ## 6. Threshold selection ($u^*$)
#
# Threshold choice is a bias-variance trade-off: a low $u$ violates the asymptotic GPD approximation (bias), a high $u$ leaves too few exceedances (variance). With about 730 daily observations per window, the 85th-99th percentile band spans from roughly 110 exceedances at $q_{0.85}$ down to roughly 8 at $q_{0.99}$.
#
# To make the visual criterion reproducible, $u^*$ is chosen by the following rule, applied to the stability grid of Section 5:
#
# 1. Restrict candidates to the band $[q_{0.85}, q_{0.99}]$ with at least `MIN_EXCEEDANCES` (30) exceedances. With $n \approx 730$, this floor caps the **effective** upper end of the candidate set at about the 96th percentile; quantiles above it remain visible in the plots but cannot be selected, because the MLE and its asymptotic standard errors are unreliable on so few points.
# 2. For each candidate, measure the **local instability** of $\hat\xi(u)$ as the standard deviation of $\hat\xi$ over a centred window of `STABILITY_WINDOW` grid points (±1 percentile point).
# 3. Keep only candidates with $\hat\xi > 0$ (Fréchet domain). If none exists in the band, all band candidates are kept, the selection is flagged `stability_fallback_xi_nonpositive`, and $\xi$ is **not** constrained (Section 9 explains why).
# 4. Select the candidate with the **flattest** $\hat\xi$ profile, i.e. the minimum local instability.
#
# After inspecting the plots in Sections 7-8, any threshold can be replaced through `THRESHOLD_OVERRIDES` (Section 1); the rest of the notebook updates accordingly.

# %%
@dataclass(frozen=True)
class ThresholdSelection:
    """Selected POT threshold for one asset in one scenario."""

    quantile: float
    threshold: float
    n_exceedances: int
    xi_local_std: float
    method: str


def select_threshold(
    loss: pd.Series,
    stability: pd.DataFrame,
    band: tuple[float, float],
    window: int,
    min_exceedances: int,
    override_quantile: float | None = None,
) -> ThresholdSelection:
    """Select u* as the flattest xi-stability point inside the quantile band, preferring xi > 0.

    Raises:
        ValueError: If the override quantile is outside (0, 1) or no grid point satisfies the band constraints.
    """
    if override_quantile is not None:
        if not 0.0 < override_quantile < 1.0:
            raise ValueError(f"Override quantile must lie in (0, 1), got {override_quantile}")
        threshold = float(np.quantile(loss, override_quantile))
        return ThresholdSelection(
            quantile=override_quantile,
            threshold=threshold,
            n_exceedances=int((loss > threshold).sum()),
            xi_local_std=np.nan,
            method="manual_override",
        )

    table = stability.sort_values("quantile").copy()
    table["xi_local_std"] = table["xi"].rolling(window, center=True, min_periods=window).std()
    is_candidate = (
        table["quantile"].between(band[0] - 1e-9, band[1] + 1e-9)
        & (table["n_exceedances"] >= min_exceedances)
        & table["xi_local_std"].notna()
    )
    candidates = table[is_candidate]
    if candidates.empty:
        raise ValueError(f"{loss.name}: no stability grid point satisfies the threshold band {band}")

    positive = candidates[candidates["xi"] > 0]
    if positive.empty:
        pool, method = candidates, "stability_fallback_xi_nonpositive"
    else:
        pool, method = positive, "stability_xi_positive"

    best = pool.loc[pool["xi_local_std"].idxmin()]
    return ThresholdSelection(
        quantile=float(best["quantile"]),
        threshold=float(best["threshold"]),
        n_exceedances=int(best["n_exceedances"]),
        xi_local_std=float(best["xi_local_std"]),
        method=method,
    )


threshold_selections: dict[str, dict[str, ThresholdSelection]] = {scenario_key: {} for scenario_key in SCENARIOS}
for scenario_key in SCENARIOS:
    for asset in ASSETS:
        selection = select_threshold(
            loss=losses[scenario_key][asset],
            stability=stability_tables[scenario_key][asset],
            band=THRESHOLD_BAND,
            window=STABILITY_WINDOW,
            min_exceedances=MIN_EXCEEDANCES,
            override_quantile=THRESHOLD_OVERRIDES.get((scenario_key, asset)),
        )
        if selection.method == "stability_fallback_xi_nonpositive":
            logger.warning(
                "[%s] %s: no threshold in the %s band yields xi > 0; flattest band point retained (q=%.3f)",
                scenario_key, asset, THRESHOLD_BAND, selection.quantile,
            )
        threshold_selections[scenario_key][asset] = selection

threshold_selection_df = pd.DataFrame(
    [
        {
            "scenario": scenario_key,
            "asset": asset,
            "threshold_quantile": selection.quantile,
            "u_star": selection.threshold,
            "n_exceedances": selection.n_exceedances,
            "xi_local_std": selection.xi_local_std,
            "selection_method": selection.method,
        }
        for scenario_key, by_asset in threshold_selections.items()
        for asset, selection in by_asset.items()
    ]
).set_index(["scenario", "asset"])

display(threshold_selection_df.style.format(precision=4).set_caption("Selected POT thresholds u*"))


# %% [markdown]
# ## 7. Mean Residual Life Plots
#
# Reading guide: the grey area is the 85th-99th percentile band of the losses and the dashed line is the selected $u^*$. A valid threshold is one above which $\hat e(u)$ is roughly linear inside its confidence band. The right end of each curve rests on very few exceedances and should be heavily discounted.

# %%
def plot_mean_residual_life(scenario_key: str) -> None:
    """Plot the MRLP of the four assets of one scenario with the threshold band and u*."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    color = SCENARIO_COLORS[scenario_key]
    for ax, asset in zip(axes.flat, ASSETS):
        mrl = mrl_tables[scenario_key][asset]
        selection = threshold_selections[scenario_key][asset]
        band_low, band_high = np.quantile(losses[scenario_key][asset], THRESHOLD_BAND)

        ax.plot(mrl["threshold"], mrl["mean_excess"], color=color, lw=1.6, label="Mean excess $\\hat{e}(u)$")
        ax.fill_between(mrl["threshold"], mrl["ci_low"], mrl["ci_high"], color=color, alpha=0.18, label="95% CI")
        ax.axvspan(band_low, band_high, color="grey", alpha=0.18, label=BAND_LABEL)
        ax.axvline(
            selection.threshold, color="black", ls="--", lw=1.2,
            label=f"$u^*$ = {selection.threshold:.4f} (q = {selection.quantile:.3f})",
        )
        ax.set(title=asset, xlabel="Threshold u (daily log-loss)", ylabel="Mean excess e(u)")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Mean Residual Life Plot - {scenario_key} scenario", fontsize=15, y=1.01)
    fig.tight_layout()
    plt.show()


for scenario_key in SCENARIOS:
    plot_mean_residual_life(scenario_key)


# %% [markdown]
# ## 8. Parameter stability plots
#
# Left column: shape $\hat\xi(u)$ with the zero line separating the Fréchet ($\xi > 0$) and Weibull ($\xi < 0$) domains. Right column: modified scale $\hat\sigma^*(u) = \hat\sigma_u - \hat\xi u$. Both should be flat (within their confidence bands) above a valid threshold.

# %%
def plot_parameter_stability(scenario_key: str) -> None:
    """Plot xi(u) and sigma*(u) with Wald confidence bands for the four assets of one scenario."""
    fig, axes = plt.subplots(len(ASSETS), 2, figsize=(13, 3.4 * len(ASSETS)), sharex="row")
    color = SCENARIO_COLORS[scenario_key]
    for row, asset in enumerate(ASSETS):
        stability = stability_tables[scenario_key][asset]
        selection = threshold_selections[scenario_key][asset]
        band_low, band_high = np.quantile(losses[scenario_key][asset], THRESHOLD_BAND)
        panels = (
            ("xi", "xi_ci_low", "xi_ci_high", "Shape $\\hat{\\xi}(u)$"),
            ("sigma_star", "sigma_star_ci_low", "sigma_star_ci_high", "Modified scale $\\hat{\\sigma}^*(u)$"),
        )
        for col, (estimate, ci_low, ci_high, label) in enumerate(panels):
            ax = axes[row, col]
            ax.plot(stability["threshold"], stability[estimate], "o-", color=color, ms=3, lw=1.3)
            ax.fill_between(stability["threshold"], stability[ci_low], stability[ci_high], color=color, alpha=0.18)
            ax.axvspan(band_low, band_high, color="grey", alpha=0.18)
            ax.axvline(selection.threshold, color="black", ls="--", lw=1.2)
            if estimate == "xi":
                ax.axhline(0.0, color="dimgrey", lw=0.9)
            ax.set(title=f"{asset} - {label}", xlabel="Threshold u (daily log-loss)", ylabel=label)

    fig.suptitle(
        f"GPD parameter stability - {scenario_key} scenario (grey: {BAND_LABEL}, dashed: $u^*$)",
        fontsize=15, y=1.005,
    )
    fig.tight_layout()
    plt.show()


for scenario_key in SCENARIOS:
    plot_parameter_stability(scenario_key)


# %% [markdown]
# ## 9. GPD fit at $u^*$ and verification of the heavy-tail condition
#
# The GPD is refitted by MLE on the excesses $Y = X - u^*$ of each asset. The **tail fraction** $\zeta_u = N_u / n$ is stored for the risk measures of Section 11.
#
# Verification of $\xi > 0$ (Fréchet domain, power-law tail) is reported at three levels:
#
# - `xi_positive`: sign of the point estimate;
# - `xi_ci_low`, `xi_ci_high`: Wald 95% interval;
# - `lrt_pvalue`: one-sided p-value of the signed-root likelihood ratio test of $H_0: \xi = 0$ against $H_1: \xi > 0$.
#
# The shape parameter is deliberately **not constrained** to $\xi > 0$ in the optimizer. If the likelihood peaks at $\xi \le 0$, imposing the constraint would place the estimate on the boundary ($\xi = 0$) and would hide, rather than fix, what the data indicate. Non-positive or non-significant estimates are therefore logged and kept visible in the tables. With 30-110 exceedances, the test has limited power, so a non-significant result means that the sample cannot distinguish a power-law tail from an exponential one; it is not evidence of a thin tail. A fit with $\hat\xi \ge 1$ raises an error, because the mean, and hence the ES, would be infinite.

# %%
@dataclass(frozen=True)
class TailModel:
    """Fitted POT/GPD tail model of one asset in one scenario."""

    scenario: str
    asset: str
    n_obs: int
    selection: ThresholdSelection
    excesses: pd.Series
    fit: GPDFit

    @property
    def zeta_u(self) -> float:
        """Empirical probability of exceeding the threshold, N_u / n."""
        return self.fit.n_exceedances / self.n_obs


tail_models: dict[str, dict[str, TailModel]] = {scenario_key: {} for scenario_key in SCENARIOS}
xi_rows: list[dict[str, object]] = []

for scenario_key in SCENARIOS:
    for asset in ASSETS:
        loss = losses[scenario_key][asset]
        selection = threshold_selections[scenario_key][asset]
        excesses = get_exceedances(loss, selection.threshold)
        fit = fit_gpd(excesses.to_numpy())
        if fit is None:
            raise RuntimeError(f"[{scenario_key}] {asset}: GPD MLE failed at u* = {selection.threshold:.5f}")
        if fit.xi >= 1.0:
            raise ValueError(f"[{scenario_key}] {asset}: xi = {fit.xi:.3f} >= 1 implies an infinite mean; ES is undefined")

        model = TailModel(scenario_key, asset, loss.size, selection, excesses, fit)
        tail_models[scenario_key][asset] = model

        cov = gpd_asymptotic_cov(fit)
        xi_se, sigma_se = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
        signed_root, lrt_pvalue = xi_zero_lrt(excesses.to_numpy(), fit)
        xi_positive = fit.xi > 0
        xi_significant = lrt_pvalue < 1 - CI_LEVEL

        if not xi_positive:
            logger.warning(
                "[%s] %s: xi = %.4f <= 0 at u* (q=%.3f); heavy-tail (Frechet) condition NOT met",
                scenario_key, asset, fit.xi, selection.quantile,
            )
        elif not xi_significant:
            logger.info(
                "[%s] %s: xi = %.4f > 0 but not significantly different from 0 (LRT p = %.3f)",
                scenario_key, asset, fit.xi, lrt_pvalue,
            )

        xi_rows.append(
            {
                "scenario": scenario_key,
                "asset": asset,
                "zeta_u": model.zeta_u,
                "xi": fit.xi,
                "xi_se": xi_se,
                "xi_ci_low": fit.xi - Z_CRIT * xi_se,
                "xi_ci_high": fit.xi + Z_CRIT * xi_se,
                "sigma": fit.sigma,
                "sigma_se": sigma_se,
                "log_likelihood": fit.log_likelihood,
                "lrt_signed_root": signed_root,
                "lrt_pvalue": lrt_pvalue,
                "xi_positive": xi_positive,
                "xi_significant": xi_significant,
                "tail_domain": "Frechet (heavy)" if xi_positive else "Gumbel/Weibull (light/bounded)",
            }
        )

xi_verification_df = pd.DataFrame(xi_rows).set_index(["scenario", "asset"])
display(
    xi_verification_df.join(threshold_selection_df[["u_star", "n_exceedances"]])
    .style.format(precision=4)
    .set_caption("GPD MLE at u* and verification of xi > 0")
)
print(
    f"xi > 0 in {int(xi_verification_df['xi_positive'].sum())}/{len(xi_verification_df)} fits; "
    f"significant at {1 - CI_LEVEL:.0%} in {int(xi_verification_df['xi_significant'].sum())}/{len(xi_verification_df)}"
)


# %% [markdown]
# ## 10. Goodness of fit: Q-Q plots and Anderson-Darling test
#
# **Q-Q plot.** The ordered empirical excesses $y_{(1)} \le \dots \le y_{(N_u)}$ are plotted against the theoretical GPD quantiles $G^{-1}_{\hat\xi,\hat\sigma}(p_i)$ at the plotting positions $p_i = (i - 0.5)/N_u$. Points on the 45° line indicate a good fit; systematic upward departures at the right end indicate an underestimated tail.
#
# **Anderson-Darling test.** The statistic weights discrepancies by $[F(1-F)]^{-1}$, which penalizes deviations in the tails far more than Kolmogorov-Smirnov:
#
# $$
# A^2 = -N_u - \frac{1}{N_u}\sum_{i=1}^{N_u} (2i - 1)\left[\ln G\!\left(y_{(i)}\right) + \ln\!\left(1 - G\!\left(y_{(N_u + 1 - i)}\right)\right)\right]
# $$
#
# Because $(\xi, \sigma)$ are estimated from the same data, the standard $A^2$ critical values (valid only for a fully specified null) would be far too conservative. The p-value is therefore obtained by a **parametric bootstrap** with re-estimation: simulate $N_u$ excesses from $G_{\hat\xi,\hat\sigma}$, refit the GPD, recompute $A^{2*}$, and set $p = \left(1 + \#\{A^{2*}_b \ge A^2\}\right) / (B + 1)$ with $B$ = `N_BOOTSTRAP_AD`. $H_0$ (the excesses follow a GPD) is not rejected when $p \ge 0.05$.

# %%
def anderson_darling_statistic(excesses: np.ndarray, xi: float, sigma: float) -> float:
    """Anderson-Darling A^2 of the excesses against a GPD(xi, sigma) with zero location."""
    ordered = np.sort(np.asarray(excesses, dtype=float))
    n = ordered.size
    cdf = np.clip(genpareto.cdf(ordered, xi, loc=0.0, scale=sigma), CDF_CLIP, 1.0 - CDF_CLIP)
    weights = 2.0 * np.arange(1, n + 1) - 1.0
    return float(-n - np.mean(weights * (np.log(cdf) + np.log1p(-cdf[::-1]))))


def anderson_darling_test(
    excesses: np.ndarray, fit: GPDFit, n_boot: int, rng: np.random.Generator
) -> dict[str, float]:
    """Anderson-Darling GPD test with a parametric-bootstrap p-value that accounts for estimated parameters."""
    excesses = np.asarray(excesses, dtype=float)
    a2_observed = anderson_darling_statistic(excesses, fit.xi, fit.sigma)

    a2_bootstrap = np.full(n_boot, np.nan)
    for b in range(n_boot):
        simulated = genpareto.rvs(fit.xi, loc=0.0, scale=fit.sigma, size=excesses.size, random_state=rng)
        refit = fit_gpd(simulated)
        if refit is not None:
            a2_bootstrap[b] = anderson_darling_statistic(simulated, refit.xi, refit.sigma)

    valid = a2_bootstrap[np.isfinite(a2_bootstrap)]
    if valid.size < 0.95 * n_boot:
        logger.warning("Anderson-Darling bootstrap: only %d/%d valid replicates", valid.size, n_boot)
    p_value = (1.0 + np.sum(valid >= a2_observed)) / (valid.size + 1.0)
    return {
        "ad_statistic": a2_observed,
        "ad_pvalue": float(p_value),
        "ad_critical_5pct": float(np.quantile(valid, 0.95)),
        "ad_n_valid": int(valid.size),
    }


gof_rows: list[dict[str, object]] = []
for scenario_key in SCENARIOS:
    for asset in ASSETS:
        model = tail_models[scenario_key][asset]
        result = anderson_darling_test(
            model.excesses.to_numpy(), model.fit, N_BOOTSTRAP_AD, make_rng(scenario_key, asset, "anderson_darling")
        )
        if result["ad_pvalue"] < 1 - CI_LEVEL:
            logger.warning(
                "[%s] %s: Anderson-Darling rejects the GPD at 5%% (A2 = %.3f, p = %.3f)",
                scenario_key, asset, result["ad_statistic"], result["ad_pvalue"],
            )
        gof_rows.append({"scenario": scenario_key, "asset": asset, **result})

gof_df = pd.DataFrame(gof_rows).set_index(["scenario", "asset"])
gof_df["gpd_not_rejected"] = gof_df["ad_pvalue"] >= 1 - CI_LEVEL
display(gof_df.style.format(precision=4).set_caption("Anderson-Darling goodness of fit (parametric bootstrap)"))


# %%
def plot_gpd_qq(scenario_key: str) -> None:
    """Q-Q plot of ordered empirical excesses against fitted GPD quantiles for one scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    color = SCENARIO_COLORS[scenario_key]
    for ax, asset in zip(axes.flat, ASSETS):
        model = tail_models[scenario_key][asset]
        empirical = np.sort(model.excesses.to_numpy())
        n = empirical.size
        plotting_positions = (np.arange(1, n + 1) - 0.5) / n
        theoretical = genpareto.ppf(plotting_positions, model.fit.xi, loc=0.0, scale=model.fit.sigma)
        upper = 1.05 * max(empirical.max(), theoretical.max())

        ax.scatter(theoretical, empirical, s=24, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.plot([0, upper], [0, upper], color="black", ls="--", lw=1)
        gof = gof_df.loc[(scenario_key, asset)]
        ax.text(
            0.04, 0.96,
            f"$A^2$ = {gof['ad_statistic']:.3f}\np-value = {gof['ad_pvalue']:.3f}\n$N_u$ = {n}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        ax.set(
            xlim=(0, upper), ylim=(0, upper),
            title=f"{asset}: $\\hat\\xi$ = {model.fit.xi:.3f}, $\\hat\\sigma$ = {model.fit.sigma:.4f}",
            xlabel="Theoretical GPD quantile", ylabel="Empirical excess (ordered)",
        )

    fig.suptitle(f"GPD Q-Q plots of threshold excesses - {scenario_key} scenario", fontsize=15, y=1.01)
    fig.tight_layout()
    plt.show()


for scenario_key in SCENARIOS:
    plot_gpd_qq(scenario_key)


# %% [markdown]
# ## 11. Value-at-Risk and Expected Shortfall
#
# Combining the empirical tail fraction $\zeta_u = N_u / n$ with the GPD for the excesses gives the tail estimator $\bar F(x) = \zeta_u \left(1 + \xi (x - u)/\sigma\right)^{-1/\xi}$ for $x > u$. Inverting at level $p$ (valid when $1 - p < \zeta_u$):
#
# $$
# VaR_p = u + \frac{\sigma}{\xi}\left[\left(\frac{1 - p}{\zeta_u}\right)^{-\xi} - 1\right] \qquad \left(\xi \to 0:\; VaR_p = u - \sigma \ln\frac{1 - p}{\zeta_u}\right)
# $$
#
# By the threshold-stability property, the excess over $VaR_p$ is again GPD with shape $\xi$ and scale $\sigma + \xi(VaR_p - u)$, whose mean is $\frac{\sigma + \xi (VaR_p - u)}{1 - \xi}$. Hence, for $\xi < 1$:
#
# $$
# ES_p = \mathbb{E}[X \mid X > VaR_p] = VaR_p + \frac{\sigma + \xi (VaR_p - u)}{1 - \xi} = \frac{VaR_p}{1 - \xi} + \frac{\sigma - \xi u}{1 - \xi}
# $$
#
# Two complementary figures are reported:
#
# - `es_price_drop`: the ES expressed as an **arithmetic price decline**, $\mathbb{E}[1 - e^{-X} \mid X > VaR_p] = 1 - e^{-VaR_p}\,\mathbb{E}[e^{-Z}]$ with $Z \sim GPD(\xi, \sigma + \xi(VaR_p - u))$, computed by numerical integration. This is the quantity to apply as a collateral price shock. Because of Jensen's inequality it is **not** equal to $1 - e^{-ES_p}$.
# - `var_empirical`, `es_empirical`: historical-simulation benchmarks. With $n \approx 730$, only about 7 observations lie beyond the empirical 99% quantile, which is exactly why a parametric tail model is needed.

# %%
def gpd_value_at_risk(threshold: float, xi: float, sigma: float, zeta_u: float, p: float) -> float:
    """POT tail-estimator VaR_p of the loss distribution.

    Raises:
        ValueError: If zeta_u is not in (0, 1) or p does not lie beyond the threshold (1 - p >= zeta_u).
    """
    if not 0.0 < zeta_u < 1.0:
        raise ValueError(f"zeta_u must lie in (0, 1), got {zeta_u}")
    tail_ratio = (1.0 - p) / zeta_u
    if tail_ratio >= 1.0:
        raise ValueError(f"p = {p} must exceed the threshold non-exceedance level 1 - zeta_u = {1.0 - zeta_u:.4f}")
    if abs(xi) < XI_ZERO_TOL:
        return float(threshold - sigma * np.log(tail_ratio))
    return float(threshold + sigma / xi * (tail_ratio ** (-xi) - 1.0))


def gpd_expected_shortfall(threshold: float, xi: float, sigma: float, zeta_u: float, p: float) -> float:
    """POT tail-estimator ES_p = E[X | X > VaR_p] of the loss distribution.

    Raises:
        ValueError: If xi >= 1 (infinite mean) or VaR_p is undefined.
    """
    if xi >= 1.0:
        raise ValueError(f"ES is infinite for xi >= 1 (xi = {xi:.4f})")
    value_at_risk = gpd_value_at_risk(threshold, xi, sigma, zeta_u, p)
    return float((value_at_risk + sigma - xi * threshold) / (1.0 - xi))


def gpd_expected_price_drop(threshold: float, xi: float, sigma: float, zeta_u: float, p: float) -> float:
    """Expected arithmetic price decline E[1 - exp(-X) | X > VaR_p] under the fitted tail."""
    value_at_risk = gpd_value_at_risk(threshold, xi, sigma, zeta_u, p)
    beyond_var_scale = sigma + xi * (value_at_risk - threshold)
    discount_factor = genpareto.expect(lambda z: np.exp(-z), args=(xi,), loc=0.0, scale=beyond_var_scale)
    return float(1.0 - np.exp(-value_at_risk) * discount_factor)


def empirical_tail_risk(loss: pd.Series, p: float) -> tuple[float, float, int]:
    """Historical-simulation VaR_p, ES_p and the number of observations at or beyond VaR_p."""
    values = loss.to_numpy()
    value_at_risk = float(np.quantile(values, p))
    tail = values[values >= value_at_risk]
    return value_at_risk, float(tail.mean()), int(tail.size)


risk_rows: list[dict[str, object]] = []
for scenario_key in SCENARIOS:
    for asset in ASSETS:
        model = tail_models[scenario_key][asset]
        params = (model.selection.threshold, model.fit.xi, model.fit.sigma, model.zeta_u, CONFIDENCE_LEVEL)
        var_empirical, es_empirical, n_tail_empirical = empirical_tail_risk(losses[scenario_key][asset], CONFIDENCE_LEVEL)
        risk_rows.append(
            {
                "scenario": scenario_key,
                "asset": asset,
                "var": gpd_value_at_risk(*params),
                "es": gpd_expected_shortfall(*params),
                "es_price_drop": gpd_expected_price_drop(*params),
                "var_empirical": var_empirical,
                "es_empirical": es_empirical,
                "n_tail_empirical": n_tail_empirical,
            }
        )

risk_df = pd.DataFrame(risk_rows).set_index(["scenario", "asset"])
display(
    risk_df.style.format(precision=4)
    .format({"es_price_drop": "{:.2%}"})
    .set_caption(f"GPD-POT risk measures at p = {CONFIDENCE_LEVEL} (daily log-loss units)")
)


# %% [markdown]
# ## 12. Bootstrap confidence interval for $ES_{0.99}$
#
# The point estimate of $ES_{0.99}$ extrapolates beyond the data and is sensitive to $\hat\xi$. Its sampling uncertainty is quantified by a **nonparametric bootstrap over the excesses** with $B$ = `N_BOOTSTRAP_ES` replicates:
#
# 1. Resample the $N_u$ excesses with replacement, holding $u^*$ and $\zeta_u$ fixed.
# 2. Refit the GPD by MLE, giving $(\xi^*_b, \sigma^*_b)$.
# 3. Recompute $ES^*_{0.99,b}$ with the closed form of Section 11.
#
# The 95% interval is the percentile interval $[q_{0.025}, q_{0.975}]$ of $\{ES^*_b\}$. Replicates with $\xi^*_b \ge 1$ (infinite ES) or failed fits are discarded and counted. The share of replicates with $\xi^*_b > 0$ (`boot_xi_positive_share`) provides a direct bootstrap measure of how firmly the data support a Fréchet tail. The histograms below are drawn over the central 99% of the replicates, because a few resamples with $\xi^*$ close to 1 produce extreme ES values that would otherwise flatten the plot.
#
# This interval is a **lower bound** on the true uncertainty: it ignores the randomness of $u^*$ and $\zeta_u$, and i.i.d. resampling ignores the volatility clustering of daily crypto returns.

# %%
def bootstrap_expected_shortfall(
    model: TailModel, p: float, n_boot: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Nonparametric bootstrap of (xi, sigma, ES_p) obtained by resampling the excesses at a fixed u* and zeta_u."""
    excesses = model.excesses.to_numpy()
    replicates = np.full((n_boot, 3), np.nan)
    for b in range(n_boot):
        resample = rng.choice(excesses, size=excesses.size, replace=True)
        fit = fit_gpd(resample)
        if fit is None:
            continue
        replicates[b, :2] = fit.xi, fit.sigma
        if fit.xi < 1.0:
            replicates[b, 2] = gpd_expected_shortfall(model.selection.threshold, fit.xi, fit.sigma, model.zeta_u, p)
    return pd.DataFrame(replicates, columns=["xi", "sigma", "es"])


bootstrap_replicates: dict[str, dict[str, pd.DataFrame]] = {scenario_key: {} for scenario_key in SCENARIOS}
bootstrap_rows: list[dict[str, object]] = []
alpha = 1.0 - CI_LEVEL

for scenario_key in SCENARIOS:
    for asset in ASSETS:
        replicates = bootstrap_expected_shortfall(
            tail_models[scenario_key][asset], CONFIDENCE_LEVEL, N_BOOTSTRAP_ES, make_rng(scenario_key, asset, "es_bootstrap")
        )
        bootstrap_replicates[scenario_key][asset] = replicates
        valid_es = replicates["es"].dropna()
        if valid_es.size < 0.95 * N_BOOTSTRAP_ES:
            logger.warning(
                "[%s] %s: only %d/%d valid ES bootstrap replicates", scenario_key, asset, valid_es.size, N_BOOTSTRAP_ES
            )
        bootstrap_rows.append(
            {
                "scenario": scenario_key,
                "asset": asset,
                "es_ci_low": valid_es.quantile(alpha / 2),
                "es_ci_high": valid_es.quantile(1.0 - alpha / 2),
                "es_boot_se": valid_es.std(ddof=1),
                "boot_xi_positive_share": float((replicates["xi"].dropna() > 0).mean()),
                "boot_n_valid": int(valid_es.size),
            }
        )

bootstrap_summary_df = pd.DataFrame(bootstrap_rows).set_index(["scenario", "asset"])
display(
    bootstrap_summary_df.join(risk_df[["es"]])
    .style.format(precision=4)
    .format({"boot_xi_positive_share": "{:.1%}"})
    .set_caption(f"Bootstrap {CI_LEVEL:.0%} percentile CI for ES (B = {N_BOOTSTRAP_ES})")
)


# %%
def plot_bootstrap_distribution(scenario_key: str) -> None:
    """Histogram of bootstrap ES replicates with the point estimate and the percentile CI."""
    fig, axes = plt.subplots(1, len(ASSETS), figsize=(4.2 * len(ASSETS), 4))
    color = SCENARIO_COLORS[scenario_key]
    for ax, asset in zip(axes, ASSETS):
        es_replicates = bootstrap_replicates[scenario_key][asset]["es"].dropna()
        summary = bootstrap_summary_df.loc[(scenario_key, asset)]
        point = risk_df.loc[(scenario_key, asset), "es"]

        view_low, view_high = es_replicates.quantile([0.005, 0.995])

        sns.histplot(es_replicates, bins=40, binrange=(view_low, view_high), color=color, alpha=0.55, ax=ax, stat="density")
        ax.axvline(point, color="black", lw=1.6, label=f"ES = {point:.4f}")
        ax.axvline(summary["es_ci_low"], color="black", ls="--", lw=1.1, label=f"{CI_LEVEL:.0%} CI")
        ax.axvline(summary["es_ci_high"], color="black", ls="--", lw=1.1)
        ax.set(
            title=asset,
            xlim=(view_low, view_high),
            xlabel=f"Bootstrap $ES_{{{CONFIDENCE_LEVEL}}}$ (log-loss)",
            ylabel="Density",
        )
        ax.legend(fontsize=8)

    fig.suptitle(f"Bootstrap distribution of $ES_{{{CONFIDENCE_LEVEL}}}$ - {scenario_key} scenario", fontsize=14, y=1.03)
    fig.tight_layout()
    plt.show()


for scenario_key in SCENARIOS:
    plot_bootstrap_distribution(scenario_key)

# %% [markdown]
# ## 13. Consolidated tail-parameter tables
#
# All outputs are merged into a single DataFrame per scenario, `evt_results[scenario]`, indexed by asset. The **headline view** shows the core tail parameters (threshold $u^*$, shape $\xi$, scale $\sigma$, $ES_{0.99}$ with its bootstrap CI). The **full view** adds the inference and goodness-of-fit diagnostics.

# %%
HEADLINE_LABELS: dict[str, str] = {
    "u_star": "Threshold u*",
    "threshold_quantile": "u* quantile",
    "n_exceedances": "N_u",
    "xi": "Shape ξ",
    "sigma": "Scale σ",
    "var": f"VaR {CONFIDENCE_LEVEL}",
    "es": f"ES {CONFIDENCE_LEVEL}",
    "es_ci_low": f"ES {CI_LEVEL:.0%} CI low",
    "es_ci_high": f"ES {CI_LEVEL:.0%} CI high",
    "es_price_drop": "ES as price drop",
    "xi_positive": "ξ > 0",
    "ad_pvalue": "AD p-value",
}


def build_scenario_table(scenario_key: str) -> pd.DataFrame:
    """Merge threshold, GPD fit, goodness-of-fit, risk and bootstrap outputs of one scenario."""
    components = [threshold_selection_df, xi_verification_df, gof_df, risk_df, bootstrap_summary_df]
    table = pd.concat([frame.xs(scenario_key, level="scenario") for frame in components], axis=1)
    if table.columns.duplicated().any():
        raise ValueError(f"Duplicated result columns: {table.columns[table.columns.duplicated()].tolist()}")
    table.insert(0, "confidence_level", CONFIDENCE_LEVEL)
    return table.loc[ASSETS]


def style_headline(table: pd.DataFrame, scenario_key: str) -> "pd.io.formats.style.Styler":
    """Format the headline parameter view of one scenario for display."""
    headline = table[list(HEADLINE_LABELS)].rename(columns=HEADLINE_LABELS)
    return (
        headline.style.format(precision=4)
        .format({HEADLINE_LABELS["threshold_quantile"]: "{:.3f}", HEADLINE_LABELS["es_price_drop"]: "{:.2%}"})
        .background_gradient(subset=[HEADLINE_LABELS["es"]], cmap="Reds")
        .set_caption(f"GPD-POT tail parameters - {scenario_key} scenario (daily log-loss units)")
        .set_table_styles([{"selector": "caption", "props": "font-size: 14px; font-weight: bold;"}])
    )


evt_results: dict[str, pd.DataFrame] = {scenario_key: build_scenario_table(scenario_key) for scenario_key in SCENARIOS}

for scenario_key in SCENARIOS:
    display(style_headline(evt_results[scenario_key], scenario_key))

# %%
for scenario_key in SCENARIOS:
    display(
        evt_results[scenario_key]
        .style.format(precision=4)
        .set_caption(f"Full EVT diagnostics - {scenario_key} scenario")
    )


# %% [markdown]
# ## 14. Cross-scenario comparison
#
# Left: $ES_{0.99}$ point estimates with their bootstrap 95% intervals. Right: $\hat\xi$ with Wald 95% intervals; estimates above the zero line belong to the Fréchet domain. Both windows cover about two years and overlap by roughly 19 months (August 2022 to March 2024), so differences between scenarios are driven by the non-overlapping months (March-August 2022 versus March-August 2024).

# %%
def plot_scenario_comparison() -> None:
    """Grouped comparison of ES (bootstrap CI) and xi (Wald CI) across assets and scenarios."""
    fig, (ax_es, ax_xi) = plt.subplots(1, 2, figsize=(15, 5.5))
    positions = np.arange(len(ASSETS))
    width = 0.38

    for offset, scenario_key in zip((-width / 2, width / 2), SCENARIOS):
        table = evt_results[scenario_key]
        color = SCENARIO_COLORS[scenario_key]
        es_error = np.vstack([table["es"] - table["es_ci_low"], table["es_ci_high"] - table["es"]])
        bars = ax_es.bar(positions + offset, table["es"], width, yerr=es_error, capsize=4, color=color, alpha=0.8, label=scenario_key)
        ax_es.bar_label(bars, labels=[f"{value:.3f}" for value in table["es"]], padding=3, fontsize=8, label_type="center", color="white")

        xi_error = np.vstack([table["xi"] - table["xi_ci_low"], table["xi_ci_high"] - table["xi"]])
        ax_xi.errorbar(positions + offset, table["xi"], yerr=xi_error, fmt="o", capsize=4, color=color, ms=7, label=scenario_key)

    ax_es.set(
        xticks=positions, xticklabels=ASSETS,
        ylabel=f"$ES_{{{CONFIDENCE_LEVEL}}}$ (daily log-loss)",
        title=f"$ES_{{{CONFIDENCE_LEVEL}}}$ with bootstrap {CI_LEVEL:.0%} CI",
    )
    ax_xi.axhline(0.0, color="dimgrey", lw=1)
    ax_xi.set(xticks=positions, xticklabels=ASSETS, ylabel="Shape $\\hat\\xi$", title=f"GPD shape with Wald {CI_LEVEL:.0%} CI")
    ax_es.legend()
    ax_xi.legend()
    fig.tight_layout()
    plt.show()


plot_scenario_comparison()

comparison_df = pd.concat(
    {scenario_key: evt_results[scenario_key][["u_star", "xi", "sigma", "es", "es_ci_low", "es_ci_high"]] for scenario_key in SCENARIOS},
    axis=1,
)
display(comparison_df.style.format(precision=4).set_caption("Side-by-side tail parameters"))

# %% [markdown]
# ## 15. Methodological notes
#
# - **Units.** $u^*$, $\sigma$, $VaR$ and $ES$ are daily log-losses. Use `es_price_drop` when a multiplicative collateral price shock $P_{t+1} = P_t\,(1 - \text{shock})$ is needed.
# - **Unconditional i.i.d. model.** The POT fit treats daily losses as independent. Crypto returns show volatility clustering, so exceedances arrive in clusters; this mainly affects the bootstrap width, which is likely understated. Natural extensions are runs declustering of the exceedances or the conditional GARCH-EVT approach of McNeil & Frey (2000).
# - **Small tail samples.** With 30-110 exceedances, $\hat\xi$ carries a standard error of roughly 0.10-0.20, so the sign of $\xi$ can flip between neighbouring thresholds. The stability rule, the LRT p-values and the bootstrap share of $\xi^* > 0$ should be read together, and $u^*$ can be fine-tuned through `THRESHOLD_OVERRIDES`.
# - **Threshold-selection uncertainty** is not propagated into the ES interval.

# %% [markdown]
# ## 16. Expected Shortfall at higher confidence levels ($p$ = 0.99, 0.995, 0.999)
#
# Sections 11-12 report $ES_{0.99}$. With the fitted model unchanged ($u^*$, $\hat\xi$, $\hat\sigma$, $\zeta_u$), this section reports, for $p \in \{0.99, 0.995, 0.999\}$, the daily $VaR_p$, the daily $ES_p$ (log-loss units) and the arithmetic price shock $\mathbb{E}[1 - e^{-X} \mid X > VaR_p]$, reusing `gpd_value_at_risk`, `gpd_expected_shortfall` and `gpd_expected_price_drop` of Section 11.
#
# The 95% interval of $ES_p$ is the bootstrap of Section 12 (`bootstrap_expected_shortfall`, same $B$ = `N_BOOTSTRAP_ES`, same random stream `es_bootstrap`). The $B$ replicates $(\xi^*_b, \sigma^*_b)$ stored in `bootstrap_replicates` are re-evaluated with the closed-form $ES_p$ at each level; this is identical to calling `bootstrap_expected_shortfall` again with the same stream for each $p$ (the resamples and refits do not depend on $p$), without repeating the MLE refits. The replicates are therefore paired across levels, and the $p = 0.99$ rows must reproduce Sections 11-12 exactly; the cell below checks it. The interval refers to `es` (log-loss units), not to the price shock.
#
# **$p = 0.999$ is an out-of-sample extrapolation.** The return period of a $p$-level daily loss is $1/(1-p)$ days: 100 days for $p$ = 0.99, 200 for 0.995 and **1,000 for 0.999, which exceeds the 730 observations of each window** (the expected number of sample days beyond $VaR_{0.999}$ is 0.73). The figure rests entirely on the GPD shape $\hat\xi$ extrapolated beyond the data and cannot be checked against history. The bootstrap interval only captures the sampling error of $(\hat\xi, \hat\sigma)$ **given** that the GPD form holds, so it understates the model risk at this level. The column `out_of_sample` is `True` when the return period exceeds the number of observations, and the rows are shaded.

# %%
ES_LEVELS: tuple[float, ...] = (0.99, 0.995, 0.999)
if CONFIDENCE_LEVEL not in ES_LEVELS:
    raise ValueError(f"ES_LEVELS {ES_LEVELS} must include CONFIDENCE_LEVEL = {CONFIDENCE_LEVEL}")


def return_period_days(p: float, horizon_days: int = 1) -> float:
    """Mean number of days between two exceedances of VaR_p of the h-day loss, h / (1 - p)."""
    return horizon_days / (1.0 - p)


def is_out_of_sample(p: float, n_obs: int, horizon_days: int = 1) -> bool:
    """Flag a risk level whose return period exceeds the number of daily observations of the window."""
    return return_period_days(p, horizon_days) > n_obs


def bootstrap_es_at_level(model: TailModel, replicates: pd.DataFrame, p: float) -> pd.Series:
    """ES_p of every bootstrap replicate of Section 12; failed fits and replicates with xi* >= 1 stay NaN."""
    es_at_level = pd.Series(np.nan, index=replicates.index, name="es")
    for b in replicates.index[replicates["es"].notna()]:
        es_at_level[b] = gpd_expected_shortfall(
            model.selection.threshold, replicates.at[b, "xi"], replicates.at[b, "sigma"], model.zeta_u, p
        )
    return es_at_level


ci_alpha = 1.0 - CI_LEVEL
multilevel_rows: list[dict[str, object]] = []
for scenario_key in SCENARIOS:
    for asset in ASSETS:
        model = tail_models[scenario_key][asset]
        for level in ES_LEVELS:
            params = (model.selection.threshold, model.fit.xi, model.fit.sigma, model.zeta_u, level)
            valid_es = bootstrap_es_at_level(model, bootstrap_replicates[scenario_key][asset], level).dropna()
            if valid_es.size < 0.95 * N_BOOTSTRAP_ES:
                logger.warning(
                    "[%s] %s p=%s: only %d/%d valid ES bootstrap replicates",
                    scenario_key, asset, level, valid_es.size, N_BOOTSTRAP_ES,
                )
            multilevel_rows.append(
                {
                    "scenario": scenario_key,
                    "asset": asset,
                    "confidence_level": level,
                    "var": gpd_value_at_risk(*params),
                    "es": gpd_expected_shortfall(*params),
                    "es_ci_low": valid_es.quantile(ci_alpha / 2),
                    "es_ci_high": valid_es.quantile(1.0 - ci_alpha / 2),
                    "es_price_drop": gpd_expected_price_drop(*params),
                    "return_period_days": return_period_days(level),
                    "n_obs": model.n_obs,
                    "out_of_sample": is_out_of_sample(level, model.n_obs),
                }
            )

multilevel_df = pd.DataFrame(multilevel_rows).set_index(["scenario", "asset", "confidence_level"])

# The p = 0.99 rows must reproduce Sections 11-12 (same model, same bootstrap stream).
for scenario_key in SCENARIOS:
    for asset in ASSETS:
        reference = pd.concat(
            [
                risk_df.loc[(scenario_key, asset), ["var", "es", "es_price_drop"]],
                bootstrap_summary_df.loc[(scenario_key, asset), ["es_ci_low", "es_ci_high"]],
            ]
        ).astype(float)
        current = multilevel_df.loc[(scenario_key, asset, CONFIDENCE_LEVEL), reference.index].astype(float)
        if not np.allclose(current, reference, rtol=1e-12, atol=0.0):
            raise ValueError(f"[{scenario_key}] {asset}: p = {CONFIDENCE_LEVEL} row differs from Sections 11-12")
print(f"p = {CONFIDENCE_LEVEL} rows reproduce Sections 11-12 (VaR, ES, price drop and bootstrap CI) for all scenarios and assets")

out_of_sample_levels = sorted(
    multilevel_df.index.get_level_values("confidence_level")[multilevel_df["out_of_sample"].to_numpy()].unique()
)
logger.warning(
    "%d/%d (scenario, asset, p) rows are out-of-sample extrapolations (return period > n_obs): p = %s",
    int(multilevel_df["out_of_sample"].sum()), len(multilevel_df), out_of_sample_levels,
)


def shade_out_of_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-wise CSS that shades the out-of-sample rows of a Styler."""
    css = np.where(frame["out_of_sample"].to_numpy()[:, None], "background-color: #fde0dd", "")
    return pd.DataFrame(np.broadcast_to(css, frame.shape), index=frame.index, columns=frame.columns)


display(
    multilevel_df.style.format(precision=4)
    .format({"es_price_drop": "{:.2%}", "return_period_days": "{:,.0f}"})
    .apply(shade_out_of_sample, axis=None)
    .set_caption(
        f"GPD-POT VaR_p, ES_p (daily log-loss) and ES price shock; {CI_LEVEL:.0%} bootstrap CI of ES (B = {N_BOOTSTRAP_ES}); "
        "shaded: out-of-sample extrapolation"
    )
)


# %% [markdown]
# ## 17. Expected Shortfall of the $h$-day loss (Monte Carlo)
#
# The collateral can keep falling between the liquidation trigger and the moment the position is closed. The loss over $h$ days is the sum of the daily log-losses, $X^{(h)} = \sum_{i=1}^{h} X_{t+i}$, and the matching arithmetic price shock is $1 - e^{-X^{(h)}}$. The sum of $h$ GPD-tailed variables has no closed form, so $VaR_p^{(h)}$, $ES_p^{(h)} = \mathbb{E}[X^{(h)} \mid X^{(h)} > VaR_p^{(h)}]$ and $\mathbb{E}[1 - e^{-X^{(h)}} \mid X^{(h)} > VaR_p^{(h)}]$ are obtained by simulation from the fitted daily model, for $h \in \{1, 2, 3, 5\}$ and $p \in \{0.99, 0.995\}$.
#
# **Daily model (semi-parametric, consistent with $\zeta_u$).** Each daily loss is drawn independently from the distribution that the POT estimator of Section 11 describes:
#
# - with probability $1 - \zeta_u$: a uniform draw from the **empirical** losses $X \le u^*$ (the body, which includes the days with gains);
# - with probability $\zeta_u$: $u^* + Y$ with $Y \sim GPD(\hat\xi, \hat\sigma)$ (the tail).
#
# For $h = 1$ and $1 - p < \zeta_u$, the $p$-quantile and the tail mean of this mixture are the closed forms of Section 11, so the simulation must reproduce them up to Monte Carlo error. The second cell below checks it (all $|z| < $ `MC_CHECK_MAX_Z`).
#
# **Simulation.** `N_MC_PATHS` paths of 5 daily losses per (scenario, asset) with a fixed seed (stream `horizon_mc` of `make_rng`); the sums for $h$ = 1, 2, 3, 5 are partial sums of the same paths (common random numbers, so the differences between horizons are not blurred by sampling noise). The Monte Carlo standard error of $ES$ uses $ES = VaR + \mathbb{E}[(X - VaR)^+]/(1-p)$: it is the sample standard deviation of $VaR + (X - VaR)^+/(1-p)$ divided by $\sqrt{N}$, which accounts for the estimation of the VaR. Both columns `*_mc_se` quantify only the simulation error, not the estimation error of $(\hat\xi, \hat\sigma)$ (that is the bootstrap of Section 16 for $h = 1$).
#
# **The i.i.d. assumption understates multi-day risk.** As in the rest of the notebook, the daily losses are independent. Real crypto returns show **volatility clustering**: a large loss is followed by large losses more often than an i.i.d. draw would suggest, which fattens the tail of the $h$-day sum. The simulated $ES^{(h)}$ for $h > 1$ should therefore be read as a lower reference, not as a conservative figure; the conditional GARCH-EVT alternative is pending (`docs/decisions.md`). In addition, the body is resampled from the same window, so its sample drift (a negative mean loss in a rising market) slightly lowers the $h$-day sums.
#
# The return period of the $h$-day loss is $h/(1-p)$ days, and the `out_of_sample` flag uses the same rule as Section 16 (return period > number of daily observations): for $p = 0.995$, $h = 5$ the return period is 1,000 days.

# %%
HORIZONS_DAYS: tuple[int, ...] = (1, 2, 3, 5)
HORIZON_LEVELS: tuple[float, ...] = (0.99, 0.995)
N_MC_PATHS: int = 1_000_000
MC_CHECK_MAX_Z: float = 4.0

if N_MC_PATHS < 200_000:
    raise ValueError(f"N_MC_PATHS = {N_MC_PATHS} is too small for stable ES estimates at p = {max(HORIZON_LEVELS)}")


def simulate_daily_losses(
    model: TailModel, loss: pd.Series, n_paths: int, n_days: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw i.i.d. daily losses from the empirical body below u* plus the fitted GPD tail above u*.

    Args:
        model: Fitted tail model; supplies u*, (xi, sigma) and zeta_u.
        loss: Daily log-losses of the window, whose values <= u* form the empirical body.
        n_paths: Number of simulated paths.
        n_days: Number of consecutive days per path.
        rng: Random generator.

    Returns:
        Array of shape (n_paths, n_days).

    Raises:
        ValueError: If the body and the exceedances do not add up to the sample size.
    """
    values = loss.to_numpy()
    threshold = model.selection.threshold
    body = values[values <= threshold]
    if body.size + model.fit.n_exceedances != values.size:
        raise ValueError(f"[{model.scenario}] {model.asset}: body and exceedances do not partition the sample")

    draws = rng.choice(body, size=(n_paths, n_days), replace=True)
    is_tail = rng.random((n_paths, n_days)) < model.zeta_u
    draws[is_tail] = threshold + genpareto.rvs(
        model.fit.xi, loc=0.0, scale=model.fit.sigma, size=int(is_tail.sum()), random_state=rng
    )
    return draws


def expected_shortfall_with_se(values: np.ndarray, p: float) -> tuple[float, float, float]:
    """Sample VaR_p, ES_p = E[X | X > VaR_p] and the Monte Carlo standard error of ES_p."""
    value_at_risk = float(np.quantile(values, p))
    expected_shortfall = float(values[values > value_at_risk].mean())
    ru_terms = value_at_risk + np.maximum(values - value_at_risk, 0.0) / (1.0 - p)
    return value_at_risk, expected_shortfall, float(ru_terms.std(ddof=1) / np.sqrt(values.size))


horizon_rows: list[dict[str, object]] = []
for scenario_key in SCENARIOS:
    for asset in ASSETS:
        model = tail_models[scenario_key][asset]
        daily_losses = simulate_daily_losses(
            model, losses[scenario_key][asset], N_MC_PATHS, max(HORIZONS_DAYS), make_rng(scenario_key, asset, "horizon_mc")
        )
        cumulative_losses = daily_losses.cumsum(axis=1)
        for horizon in HORIZONS_DAYS:
            horizon_loss = cumulative_losses[:, horizon - 1]
            horizon_price_shock = -np.expm1(-horizon_loss)  # 1 - exp(-X^(h))
            for level in HORIZON_LEVELS:
                value_at_risk, es_log, es_log_se = expected_shortfall_with_se(horizon_loss, level)
                _, es_shock, es_shock_se = expected_shortfall_with_se(horizon_price_shock, level)
                horizon_rows.append(
                    {
                        "scenario": scenario_key,
                        "asset": asset,
                        "confidence_level": level,
                        "horizon_days": horizon,
                        "var_log_loss": value_at_risk,
                        "es_log_loss": es_log,
                        "es_log_loss_mc_se": es_log_se,
                        "es_price_shock": es_shock,
                        "es_price_shock_mc_se": es_shock_se,
                        "n_paths": N_MC_PATHS,
                        "return_period_days": return_period_days(level, horizon),
                        "out_of_sample": is_out_of_sample(level, model.n_obs, horizon),
                    }
                )
        del daily_losses, cumulative_losses

horizon_df = pd.DataFrame(horizon_rows).set_index(["scenario", "asset", "confidence_level", "horizon_days"]).sort_index()
if min(HORIZONS_DAYS) != 1:
    raise ValueError("HORIZONS_DAYS must include h = 1, the base of es_ratio_vs_h1 and of the closed-form check")
horizon_df["es_ratio_vs_h1"] = horizon_df.groupby(level=["scenario", "asset", "confidence_level"])["es_log_loss"].transform(
    lambda es: es / es.iloc[0]  # rows are sorted by horizon, so iloc[0] is h = 1
)
print(f"Monte Carlo: {N_MC_PATHS:,} paths x {max(HORIZONS_DAYS)} days per (scenario, asset); {len(horizon_df)} result rows")


# %%
# Check: h = 1 must reproduce the closed forms of Sections 11 and 16 within Monte Carlo error.
consistency_rows: list[dict[str, object]] = []
for scenario_key in SCENARIOS:
    for asset in ASSETS:
        for level in HORIZON_LEVELS:
            simulated = horizon_df.loc[(scenario_key, asset, level, 1)]
            closed_form = multilevel_df.loc[(scenario_key, asset, level)]
            consistency_rows.append(
                {
                    "scenario": scenario_key,
                    "asset": asset,
                    "confidence_level": level,
                    "var_closed_form": closed_form["var"],
                    "var_mc": simulated["var_log_loss"],
                    "es_closed_form": closed_form["es"],
                    "es_mc": simulated["es_log_loss"],
                    "es_z": (simulated["es_log_loss"] - closed_form["es"]) / simulated["es_log_loss_mc_se"],
                    "shock_closed_form": closed_form["es_price_drop"],
                    "shock_mc": simulated["es_price_shock"],
                    "shock_z": (simulated["es_price_shock"] - closed_form["es_price_drop"]) / simulated["es_price_shock_mc_se"],
                }
            )

consistency_df = pd.DataFrame(consistency_rows).set_index(["scenario", "asset", "confidence_level"])
max_abs_z = float(consistency_df[["es_z", "shock_z"]].abs().to_numpy().max())
display(
    consistency_df.style.format(precision=4)
    .format({"shock_closed_form": "{:.3%}", "shock_mc": "{:.3%}", "es_z": "{:+.2f}", "shock_z": "{:+.2f}"})
    .set_caption("Check: h = 1 Monte Carlo versus closed form (z = difference / Monte Carlo standard error)")
)
if max_abs_z > MC_CHECK_MAX_Z:
    raise ValueError(f"h = 1 Monte Carlo does not reproduce the closed form: max |z| = {max_abs_z:.2f} > {MC_CHECK_MAX_Z}")
print(f"h = 1 Monte Carlo reproduces the closed form for all {len(consistency_df)} cases: max |z| = {max_abs_z:.2f} <= {MC_CHECK_MAX_Z}")


# %%
RESULT_ROW_ORDER = pd.MultiIndex.from_product([SCENARIOS, ASSETS], names=["scenario", "asset"])
for level in HORIZON_LEVELS:
    at_level = horizon_df.xs(level, level="confidence_level")
    display(
        at_level["es_price_shock"].unstack("horizon_days").reindex(RESULT_ROW_ORDER)
        .style.format("{:.2%}")
        .set_caption(f"ES price shock of the h-day loss, p = {level} (columns: horizon h in days)")
    )
    display(
        at_level["es_log_loss"].unstack("horizon_days").reindex(RESULT_ROW_ORDER)
        .style.format(precision=4)
        .set_caption(f"ES of the h-day log-loss, p = {level} (columns: horizon h in days)")
    )
    display(
        at_level["es_ratio_vs_h1"].unstack("horizon_days").reindex(RESULT_ROW_ORDER)
        .style.format(precision=2)
        .set_caption(f"ES(h) / ES(1), p = {level} (square-root-of-time reference: " + ", ".join(f"h={h}: {np.sqrt(h):.2f}" for h in HORIZONS_DAYS) + ")")
    )


# %% [markdown]
# ## 18. Break-even price shocks from the Aave parameters
#
# Consider a position liquidated at the instant its Health Factor reaches 1, i.e. $HF = C \cdot LT / D = 1$ and $C / D = 1 / LT$, where $C$ is the collateral value and $D$ the debt (stablecoins, so $D$ is constant in debt units). After a collateral price shock $s$ the protocol bears a loss only if the seized collateral cannot cover the repaid debt plus the liquidation bonus:
#
# $$
# C\,(1 - s) < D\,(1 + LB) \iff \frac{1 - s}{LT} < 1 + LB \iff s > 1 - LT\,(1 + LB)
# $$
#
# The two break-even shocks reported are therefore:
#
# - **with bonus**: $s^*_{LB} = 1 - LT\,(1 + LB)$, the smallest shock for which the LGD of `docs/decisions.md`, $\max\{0,\, 1 - \tfrac{HF_{entry}}{LT}\tfrac{1 - s}{1 + LB}\}$, is positive at $HF_{entry} = 1$;
# - **without bonus** ($LB = 0$): $s^*_{0} = 1 - LT$, the shock at which the collateral is worth less than the debt (the position is under water even if the liquidator received no bonus).
#
# Because $LB > 0$, $s^*_{LB} < s^*_0$: the bonus makes the loss start earlier.
#
# **Parameters.** $LT$ and $LB$ come from `src.utils.protocol_params.load_reserve_params(scenario, date)` at the valuation date $t$ of each scenario (baseline 2024-03-15, stress 2024-08-05), for `WETH`, `wstETH` (ETH shock, since a liquid-staking token is treated as ETH by decision), `WBTC`, `UNI` and `LINK`; the Binance window of each scenario must end on $t$. The base reserve parameters apply, not those of the ETH-correlated eMode category (LT 0.95, LB 0.01): in Aave v3, eMode restricts the borrowable assets to the same category, and the profiles borrow only stablecoins. The break-evens refer to a single-asset position; wallets with several collateral assets are weighted in the ECL notebook.

# %%
VALUATION_DATES: dict[str, str] = {"baseline": "2024-03-15", "stress": "2024-08-05"}
AAVE_SYMBOL_TO_ASSET: dict[str, str] = {"WETH": "ETH", "wstETH": "ETH", "WBTC": "BTC", "UNI": "UNI", "LINK": "LINK"}
BREAKEVEN_KEY_COLUMNS: list[str] = ["scenario", "asset", "aave_symbol"]


def breakeven_shocks(liquidation_threshold: float, liquidation_bonus: float) -> tuple[float, float]:
    """Minimum collateral price shock that causes a loss for a position liquidated at HF = 1.

    Args:
        liquidation_threshold: LT as a fraction, in (0, 1).
        liquidation_bonus: LB as a fraction above par, in [0, 1).

    Returns:
        ``(without_bonus, with_bonus)`` = ``(1 - LT, 1 - LT * (1 + LB))``.

    Raises:
        ValueError: If LT or LB are outside their valid range.
    """
    if not 0.0 < liquidation_threshold < 1.0:
        raise ValueError(f"liquidation_threshold must lie in (0, 1), got {liquidation_threshold}")
    if not 0.0 <= liquidation_bonus < 1.0:
        raise ValueError(f"liquidation_bonus must lie in [0, 1), got {liquidation_bonus}")
    return 1.0 - liquidation_threshold, 1.0 - liquidation_threshold * (1.0 + liquidation_bonus)


def build_breakeven_table() -> pd.DataFrame:
    """Read LT and LB at the valuation date of each scenario and compute the break-even shocks.

    Raises:
        ValueError: If the Binance window does not end on the valuation date or the parameters are not for that date.
        KeyError: If a reserve symbol is missing from the parameter table.
    """
    rows: list[dict[str, object]] = []
    for scenario_key in SCENARIOS:
        valuation_date = VALUATION_DATES[scenario_key]
        window_ends = {close_prices[scenario_key][asset].index.max().strftime("%Y-%m-%d") for asset in ASSETS}
        if window_ends != {valuation_date}:
            raise ValueError(f"[{scenario_key}] price windows end on {sorted(window_ends)}, expected {valuation_date}")

        params = load_reserve_params(scenario_key, valuation_date)
        if set(params["date"]) != {valuation_date}:
            raise ValueError(f"[{scenario_key}] reserve parameters are for {sorted(set(params['date']))}, expected {valuation_date}")

        for symbol, asset in AAVE_SYMBOL_TO_ASSET.items():
            if not bool(params.loc[params["symbol"] == symbol, "usage_as_collateral_enabled"].iloc[0]):
                logger.warning("[%s] %s is not enabled as collateral on %s", scenario_key, symbol, valuation_date)
            liquidation_threshold = get_param(params, symbol, "liquidation_threshold")
            liquidation_bonus = get_param(params, symbol, "liquidation_bonus")
            without_bonus, with_bonus = breakeven_shocks(liquidation_threshold, liquidation_bonus)
            rows.append(
                {
                    "scenario": scenario_key,
                    "asset": asset,
                    "aave_symbol": symbol,
                    "lt": liquidation_threshold,
                    "lb": liquidation_bonus,
                    "breakeven_no_bonus": without_bonus,
                    "breakeven_with_bonus": with_bonus,
                }
            )
    return pd.DataFrame(rows)


breakeven_df = build_breakeven_table().set_index(BREAKEVEN_KEY_COLUMNS)
display(
    breakeven_df.style.format(precision=4)
    .format({"breakeven_no_bonus": "{:.2%}", "breakeven_with_bonus": "{:.2%}"})
    .set_caption("Break-even price shocks for a position liquidated at HF = 1 (Aave v3 parameters at t)")
)


# %% [markdown]
# ## 19. ES price shocks versus break-even shocks
#
# The ES price shocks of Sections 16-17 are compared with the break-evens of Section 18 for every collateral symbol, using the closed form for $h = 1$ ($p$ = 0.99, 0.995, 0.999) and the Monte Carlo for $h = 2, 3, 5$ ($p$ = 0.99, 0.995). The tables mark the cells where the **ES price shock exceeds the break-even**, i.e. where the tail scenario that the ES describes would leave a loss for the protocol on a position liquidated at $HF = 1$:
#
# - **red**: the shock exceeds the break-even **without bonus** (the collateral is worth less than the debt, and the loss also holds for $LB$ = 0); it exceeds the break-even with bonus as well;
# - **orange**: the shock exceeds only the break-even **with bonus** (the loss is caused by the liquidation bonus);
# - **no colour**: no loss in the ES scenario.
#
# `breakeven_comparison_df` also carries `lgd_hf1`, the LGD of `docs/decisions.md` evaluated at $HF_{entry} = 1$ and the ES price shock. The ES price shock is an average over the tail beyond $VaR_p$, so a cell without colour does not rule out a loss for the worst days of that tail, and a coloured cell does not mean every day beyond $VaR_p$ produces a loss. The Monte Carlo rows inherit the i.i.d. limitation of Section 17, and the $p$ = 0.999 column is an extrapolation (Section 16).

# %%
SHOCK_GRID_COLUMNS: list[str] = [
    "scenario", "asset", "confidence_level", "horizon_days", "method",
    "es_log_loss", "es_price_shock", "ci_low", "ci_high", "out_of_sample_flag",
]
METHOD_ORDER: dict[str, int] = {"closed_form": 0, "monte_carlo": 1}


def build_shock_grid() -> pd.DataFrame:
    """Stack the closed-form (h = 1) and Monte Carlo (h >= 1) ES results into one long grid.

    ``ci_low`` / ``ci_high`` are the bootstrap percentile interval of ``es_log_loss`` of Section 16 and are NaN
    for the Monte Carlo rows.
    """
    closed_form = (
        multilevel_df.reset_index()
        .assign(horizon_days=1, method="closed_form")
        .rename(
            columns={
                "es": "es_log_loss", "es_price_drop": "es_price_shock",
                "es_ci_low": "ci_low", "es_ci_high": "ci_high", "out_of_sample": "out_of_sample_flag",
            }
        )
    )
    monte_carlo = (
        horizon_df.reset_index()
        .assign(method="monte_carlo", ci_low=np.nan, ci_high=np.nan)
        .rename(columns={"out_of_sample": "out_of_sample_flag"})
    )
    grid = pd.concat([closed_form[SHOCK_GRID_COLUMNS], monte_carlo[SHOCK_GRID_COLUMNS]], ignore_index=True)
    order = grid.assign(
        _scenario=grid["scenario"].map(SCENARIOS.index),
        _asset=grid["asset"].map(ASSETS.index),
        _method=grid["method"].map(METHOD_ORDER),
    ).sort_values(["_scenario", "_asset", "_method", "confidence_level", "horizon_days"], kind="stable")
    return order[SHOCK_GRID_COLUMNS].reset_index(drop=True)


shock_grid_df = build_shock_grid()

is_compared = (shock_grid_df["method"] == "closed_form") | (shock_grid_df["horizon_days"] > 1)
breakeven_comparison_df = shock_grid_df.loc[is_compared].merge(
    breakeven_df.reset_index(), on=["scenario", "asset"], how="inner"
)
breakeven_comparison_df["exceeds_no_bonus"] = breakeven_comparison_df["es_price_shock"] > breakeven_comparison_df["breakeven_no_bonus"]
breakeven_comparison_df["exceeds_with_bonus"] = breakeven_comparison_df["es_price_shock"] > breakeven_comparison_df["breakeven_with_bonus"]
breakeven_comparison_df["lgd_hf1"] = np.maximum(
    0.0,
    1.0 - (1.0 - breakeven_comparison_df["es_price_shock"])
    / (breakeven_comparison_df["lt"] * (1.0 + breakeven_comparison_df["lb"])),
)
if (breakeven_comparison_df["exceeds_no_bonus"] & ~breakeven_comparison_df["exceeds_with_bonus"]).any():
    raise ValueError("A shock above the no-bonus break-even must also exceed the with-bonus break-even")

COLOR_NO_BONUS: str = "background-color: #f4a6a6"
COLOR_WITH_BONUS: str = "background-color: #fbd9a3"


def comparison_wide_table(scenario_key: str) -> pd.DataFrame:
    """Break-even parameters and ES price shocks of one scenario, one row per Aave symbol."""
    rows = breakeven_comparison_df[breakeven_comparison_df["scenario"] == scenario_key]
    shocks = rows.pivot(index="aave_symbol", columns=["confidence_level", "horizon_days"], values="es_price_shock")
    shocks = shocks.reindex(list(AAVE_SYMBOL_TO_ASSET))
    shocks.columns = pd.MultiIndex.from_tuples([(f"ES p = {p:.1%}", f"h = {h}") for p, h in shocks.columns])
    parameters = breakeven_df.xs(scenario_key, level="scenario").droplevel("asset").reindex(list(AAVE_SYMBOL_TO_ASSET))
    parameters = parameters[["lt", "lb", "breakeven_no_bonus", "breakeven_with_bonus"]]
    parameters.columns = pd.MultiIndex.from_tuples(
        [("Break-even", "LT"), ("Break-even", "LB"), ("Break-even", "no bonus"), ("Break-even", "with bonus")]
    )
    return pd.concat([parameters, shocks], axis=1)


def style_comparison(table: pd.DataFrame, scenario_key: str) -> "pd.io.formats.style.Styler":
    """Format the comparison table and colour the ES shocks that exceed a break-even."""
    no_bonus = table[("Break-even", "no bonus")].to_numpy()[:, None]
    with_bonus = table[("Break-even", "with bonus")].to_numpy()[:, None]
    shock_columns = [column for column in table.columns if column[0] != "Break-even"]
    shocks = table[shock_columns].to_numpy()

    css = pd.DataFrame("", index=table.index, columns=table.columns)
    css[shock_columns] = np.where(shocks > no_bonus, COLOR_NO_BONUS, np.where(shocks > with_bonus, COLOR_WITH_BONUS, ""))
    formats = {column: "{:.1%}" for column in table.columns if column[0] != "Break-even" or column[1] in ("no bonus", "with bonus")}
    formats[("Break-even", "LT")] = "{:.2f}"
    formats[("Break-even", "LB")] = "{:.2f}"
    return (
        table.style.format(formats, na_rep="-")
        .apply(lambda _: css, axis=None)
        .set_caption(
            f"ES price shock vs break-even - {scenario_key} scenario "
            "(red: above break-even without bonus; orange: above break-even with bonus only)"
        )
    )


for scenario_key in SCENARIOS:
    display(style_comparison(comparison_wide_table(scenario_key), scenario_key))


# %%
def plot_shock_vs_horizon(scenario_key: str) -> None:
    """Plot the Monte Carlo ES price shock against the horizon, with the Aave break-even shocks of each asset."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    color = SCENARIO_COLORS[scenario_key]
    for ax, asset in zip(axes.flat, ASSETS):
        monte_carlo = shock_grid_df[
            (shock_grid_df["scenario"] == scenario_key) & (shock_grid_df["asset"] == asset) & (shock_grid_df["method"] == "monte_carlo")
        ]
        for level, line_style in zip(HORIZON_LEVELS, ("-", "--")):
            at_level = monte_carlo[monte_carlo["confidence_level"] == level]
            ax.plot(at_level["horizon_days"], at_level["es_price_shock"], marker="o", ls=line_style, color=color, lw=1.8, label=f"ES {level} price shock (MC)")
        for symbol, line_width in zip([s for s, a in AAVE_SYMBOL_TO_ASSET.items() if a == asset], (1.4, 0.9)):
            row = breakeven_df.loc[(scenario_key, asset, symbol)]
            ax.axhline(row["breakeven_with_bonus"], color="black", ls=":", lw=line_width, label=f"{symbol} break-even, with bonus")
            ax.axhline(row["breakeven_no_bonus"], color="dimgrey", ls="-.", lw=line_width, label=f"{symbol} break-even, no bonus")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set(title=asset, xticks=list(HORIZONS_DAYS), xlabel="Horizon h (days)", ylabel="Price shock")
        ax.legend(fontsize=7, loc="center right")

    fig.suptitle(f"ES price shock of the h-day loss vs break-even shocks - {scenario_key} scenario (i.i.d. daily losses)", fontsize=14, y=1.01)
    fig.tight_layout()
    plt.show()


for scenario_key in SCENARIOS:
    plot_shock_vs_horizon(scenario_key)


# %% [markdown]
# ## 20. Manual export of $ES_{0.99}$ collateral shocks (test dump, not for production)
#
# This section is the only part of the notebook that writes to disk. It dumps the parametric $ES_{0.99}$ shock of every collateral asset and market regime to `data/02_interim/notebook_export_evt_shocks_not_for_prod.csv`, in long format with one row per `(scenario, asset)`; scenarios are labelled by their raw-data folder (`baseline_mar2024`, `stress_aug2024`). The file name flags it as a manual test dump of this notebook: it is **not** the official artefact of the automated pipeline and must not be consumed in production.
#
# Exported columns:
#
# - `es_price_shock`: the $ES_{0.99}$ tail event expressed as an arithmetic price decline, $\mathbb{E}[1 - e^{-X} \mid X > VaR_{0.99}]$, i.e. the multiplicative shock $P_{t+1} = P_t\,(1 - \text{shock})$ to apply to the collateral price (Section 11).
# - `es_log_loss`, `es_log_loss_ci_low`, `es_log_loss_ci_high`: $ES_{0.99}$ in daily log-loss units with its bootstrap 95% percentile interval (Section 12).
# - `var_log_loss`, `u_star`, `threshold_quantile`, `n_exceedances`, `zeta_u`, `xi`, `sigma`, `ad_pvalue`: the POT/GPD fit behind each shock, kept for traceability.
# - `n_obs`, `window_start`, `window_end`: number of daily losses and calendar span of the price sample.
#
# Before writing, every row is checked for finite values, $\xi < 1$, the ordering $u^* < VaR_{0.99} < ES_{0.99}$, a well-ordered bootstrap interval, and the bounds $1 - e^{-VaR_{0.99}} \le$ `es_price_shock` $\le 1 - e^{-ES_{0.99}}$ (the upper bound follows from Jensen's inequality, because $1 - e^{-x}$ is concave). After writing, the file is read back and compared exactly with the in-memory table. The file is overwritten on every run, and the objects built in Sections 1-13 are only read, never modified.

# %%
INTERIM_DATA_DIR: Path = Path("../data/02_interim")
EVT_SHOCKS_EXPORT_PATH: Path = INTERIM_DATA_DIR / "notebook_export_evt_shocks_not_for_prod.csv"
SHOCK_TOLERANCE: float = 1e-9
SHOCK_KEY_COLUMNS: list[str] = ["scenario", "asset"]
SHOCK_COLUMN_MAP: dict[str, str] = {
    "confidence_level": "confidence_level",
    "es_price_drop": "es_price_shock",
    "es": "es_log_loss",
    "es_ci_low": "es_log_loss_ci_low",
    "es_ci_high": "es_log_loss_ci_high",
    "var": "var_log_loss",
    "u_star": "u_star",
    "threshold_quantile": "threshold_quantile",
    "n_exceedances": "n_exceedances",
    "zeta_u": "zeta_u",
    "xi": "xi",
    "sigma": "sigma",
    "ad_pvalue": "ad_pvalue",
}


def build_shock_export_table(
    results: dict[str, pd.DataFrame],
    prices: dict[str, dict[str, pd.Series]],
    loss_series: dict[str, dict[str, pd.Series]],
) -> pd.DataFrame:
    """Stack the ES_p collateral shocks of every scenario into one long table keyed by (scenario, asset).

    Args:
        results: Consolidated EVT tables of Section 13, keyed by scenario and indexed by asset.
        prices: Daily close prices of Section 2, used to record the sample window.
        loss_series: Daily log-losses of Section 3, used to record the sample size.

    Returns:
        One row per (scenario, asset), with scenarios labelled by their raw-data folder name.

    Raises:
        KeyError: If a scenario, a required column or an asset is missing from the inputs.
    """
    blocks: list[pd.DataFrame] = []
    for scenario_folder in SCENARIO_DIRS:
        key = SCENARIO_KEY[scenario_folder]
        if key not in results or key not in prices or key not in loss_series:
            raise KeyError(f"EVT inputs missing for scenario '{key}'")
        scenario_table = results[key]
        missing_columns = sorted(set(SHOCK_COLUMN_MAP) - set(scenario_table.columns))
        if missing_columns:
            raise KeyError(f"[{key}] EVT results are missing columns: {missing_columns}")
        missing_assets = sorted(set(ASSETS) - set(scenario_table.index))
        if missing_assets:
            raise KeyError(f"[{key}] EVT results are missing assets: {missing_assets}")

        block = scenario_table.loc[ASSETS, list(SHOCK_COLUMN_MAP)].rename(columns=SHOCK_COLUMN_MAP)
        block = block.rename_axis("asset").reset_index()
        block.insert(0, "scenario", scenario_folder)
        block["n_obs"] = [loss_series[key][name].size for name in ASSETS]
        block["window_start"] = [prices[key][name].index.min().strftime("%Y-%m-%d") for name in ASSETS]
        block["window_end"] = [prices[key][name].index.max().strftime("%Y-%m-%d") for name in ASSETS]
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def _failing_keys(shocks: pd.DataFrame, mask: pd.Series | np.ndarray) -> list[tuple[str, str]]:
    """Return the (scenario, asset) keys of the rows selected by a boolean mask."""
    return list(shocks.loc[mask, SHOCK_KEY_COLUMNS].itertuples(index=False, name=None))


def validate_shock_table(shocks: pd.DataFrame, tolerance: float = SHOCK_TOLERANCE) -> None:
    """Check that every exported shock is finite and consistent with the POT/GPD model.

    Raises:
        ValueError: If rows are missing or duplicated, values are non-finite or a consistency rule fails.
    """
    expected_rows = len(SCENARIO_DIRS) * len(ASSETS)
    if len(shocks) != expected_rows:
        raise ValueError(f"Expected {expected_rows} shock rows, found {len(shocks)}")
    if shocks.duplicated(SHOCK_KEY_COLUMNS).any():
        raise ValueError("Shock table contains duplicated (scenario, asset) keys")

    numeric_columns = list(SHOCK_COLUMN_MAP.values()) + ["n_obs"]
    is_finite = np.isfinite(shocks[numeric_columns].to_numpy(dtype=float)).all(axis=1)
    if not is_finite.all():
        raise ValueError(f"Non-finite shock parameters for: {_failing_keys(shocks, ~is_finite)}")

    rules: dict[str, pd.Series] = {
        f"confidence_level == {CONFIDENCE_LEVEL}": pd.Series(
            np.isclose(shocks["confidence_level"], CONFIDENCE_LEVEL), index=shocks.index
        ),
        "xi < 1 (finite ES)": shocks["xi"] < 1.0,
        "u* < VaR < ES": (shocks["u_star"] < shocks["var_log_loss"]) & (shocks["var_log_loss"] < shocks["es_log_loss"]),
        "ES CI low <= ES CI high": shocks["es_log_loss_ci_low"] <= shocks["es_log_loss_ci_high"],
        "0 < price shock < 1": shocks["es_price_shock"].between(0.0, 1.0, inclusive="neither"),
        "1 - exp(-VaR) <= price shock <= 1 - exp(-ES)": shocks["es_price_shock"].between(
            -np.expm1(-shocks["var_log_loss"]) - tolerance, -np.expm1(-shocks["es_log_loss"]) + tolerance
        ),
    }
    for rule, passed in rules.items():
        if not passed.all():
            raise ValueError(f"Shock validation failed ({rule}) for: {_failing_keys(shocks, ~passed)}")

    outside_ci = ~shocks["es_log_loss"].between(shocks["es_log_loss_ci_low"], shocks["es_log_loss_ci_high"])
    for scenario_label, asset_label in _failing_keys(shocks, outside_ci):
        logger.warning("[%s] %s: ES point estimate lies outside its bootstrap CI", scenario_label, asset_label)


def export_shock_table(shocks: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Write the shock table to CSV and verify an exact round trip.

    Returns:
        The table as reloaded from disk.

    Raises:
        OSError: If the file cannot be written or read back.
        ValueError: If the reloaded file differs from the in-memory table.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            logger.info("Overwriting existing notebook dump %s", output_path.resolve())
        shocks.to_csv(output_path, index=False)
        reloaded = pd.read_csv(output_path, float_precision="round_trip")
    except OSError as exc:
        raise OSError(f"Failed to write or reload {output_path.resolve()}: {exc}") from exc

    try:
        pd.testing.assert_frame_equal(reloaded, shocks, check_exact=True)
    except AssertionError as exc:
        raise ValueError(f"Round-trip verification failed for {output_path.resolve()}: {exc}") from exc
    return reloaded


evt_shocks_export_df = build_shock_export_table(evt_results, close_prices, losses)
validate_shock_table(evt_shocks_export_df)
evt_shocks_reloaded_df = export_shock_table(evt_shocks_export_df, EVT_SHOCKS_EXPORT_PATH)
logger.info(
    "ES %s collateral shocks written to %s (%d rows; manual notebook dump, NOT for production)",
    CONFIDENCE_LEVEL, EVT_SHOCKS_EXPORT_PATH.resolve(), len(evt_shocks_reloaded_df),
)

display(
    evt_shocks_reloaded_df.set_index(SHOCK_KEY_COLUMNS)
    .style.format(precision=4)
    .format({"es_price_shock": "{:.2%}", "confidence_level": "{:.2f}", "threshold_quantile": "{:.3f}"})
    .set_caption(f"ES {CONFIDENCE_LEVEL} collateral shocks reloaded from {EVT_SHOCKS_EXPORT_PATH.name}")
)


# %% [markdown]
# ## 21. Manual export of the ES shock grid and the break-even shocks (test dump, not for production)
#
# Two further manual test dumps follow the same pattern and carry the same disclaimer as Section 20: validate, write, reload and compare exactly with the in-memory table; not the official artefact of the automated pipeline; overwritten on every run. The Section 20 file is not modified. Scenarios are labelled by their raw-data folder.
#
# **`data/02_interim/evt_shock_grid_not_for_prod.csv`** (long format, one row per `(scenario, asset, confidence_level, horizon_days, method)`; 88 rows = 2 scenarios $\times$ 4 assets $\times$ (3 closed-form levels at $h = 1$ + 4 horizons $\times$ 2 levels by Monte Carlo)):
#
# - `method`: `closed_form` (Sections 11 and 16, $h = 1$, $p$ = 0.99, 0.995, 0.999) or `monte_carlo` (Section 17, $h$ = 1, 2, 3, 5, $p$ = 0.99, 0.995; the $h = 1$ rows repeat the closed-form check).
# - `es_log_loss`, `es_price_shock`: $ES_p$ of the $h$-day log-loss and the matching arithmetic price shock $\mathbb{E}[1 - e^{-X^{(h)}} \mid X^{(h)} > VaR_p^{(h)}]$.
# - `ci_low`, `ci_high`: bootstrap 95% percentile interval of **`es_log_loss`** (Section 16; it does not refer to `es_price_shock`); NaN for the Monte Carlo rows, where the simulation error is reported in Section 17 but the estimation error of $(\hat\xi, \hat\sigma)$ is not propagated.
# - `out_of_sample_flag`: `True` when the return period `horizon_days / (1 - confidence_level)` exceeds the number of daily observations of the window (730): $p$ = 0.999 at $h = 1$ and $p$ = 0.995 at $h = 5$. These values extrapolate beyond the data.
#
# **`data/02_interim/evt_breakeven_not_for_prod.csv`** (one row per `(scenario, asset, aave_symbol)`; 10 rows): `lt` and `lb` from `load_reserve_params` at $t$, and `breakeven_no_bonus` $= 1 - LT$, `breakeven_with_bonus` $= 1 - LT\,(1 + LB)$ (Section 18).
#
# Before writing, the grid is checked for the expected number of rows and unique keys, finite values, $0 <$ `es_price_shock` $< 1$ and `es_price_shock` $\le 1 - e^{-\text{es\_log\_loss}}$ (Jensen), a bootstrap interval present exactly for the closed-form rows and well ordered, strictly increasing ES in $p$, and an `out_of_sample_flag` consistent with its definition. The break-even table is checked for finite values, $LT \in (0,1)$, $LB \in [0,1)$, the two formulas and the ordering $0 <$ with bonus $\le$ no bonus $< 1$.

# %%
EVT_SHOCK_GRID_EXPORT_PATH: Path = INTERIM_DATA_DIR / "evt_shock_grid_not_for_prod.csv"
EVT_BREAKEVEN_EXPORT_PATH: Path = INTERIM_DATA_DIR / "evt_breakeven_not_for_prod.csv"
GRID_KEY_COLUMNS: list[str] = ["scenario", "asset", "confidence_level", "horizon_days", "method"]
BREAKEVEN_EXPORT_COLUMNS: list[str] = [
    "scenario", "asset", "aave_symbol", "lt", "lb", "breakeven_no_bonus", "breakeven_with_bonus",
]
SCENARIO_FOLDER: dict[str, str] = {key: folder for folder, key in SCENARIO_KEY.items()}


def build_shock_grid_export_table(grid: pd.DataFrame) -> pd.DataFrame:
    """Label the scenarios of the shock grid of Section 19 by their raw-data folder and fix the column dtypes.

    Raises:
        KeyError: If a required column is missing or a scenario has no folder.
    """
    missing_columns = sorted(set(SHOCK_GRID_COLUMNS) - set(grid.columns))
    if missing_columns:
        raise KeyError(f"Shock grid is missing columns: {missing_columns}")
    table = grid[SHOCK_GRID_COLUMNS].copy()
    unknown_scenarios = sorted(set(table["scenario"]) - set(SCENARIO_FOLDER))
    if unknown_scenarios:
        raise KeyError(f"Shock grid has scenarios without a raw-data folder: {unknown_scenarios}")
    table["scenario"] = table["scenario"].map(SCENARIO_FOLDER)
    table["horizon_days"] = table["horizon_days"].astype("int64")
    table["out_of_sample_flag"] = table["out_of_sample_flag"].astype(bool)
    return table.reset_index(drop=True)


def build_breakeven_export_table(breakeven: pd.DataFrame) -> pd.DataFrame:
    """Label the scenarios of the break-even table of Section 18 by their raw-data folder.

    Raises:
        KeyError: If a scenario has no raw-data folder.
    """
    table = breakeven.reset_index()[BREAKEVEN_EXPORT_COLUMNS].copy()
    unknown_scenarios = sorted(set(table["scenario"]) - set(SCENARIO_FOLDER))
    if unknown_scenarios:
        raise KeyError(f"Break-even table has scenarios without a raw-data folder: {unknown_scenarios}")
    table["scenario"] = table["scenario"].map(SCENARIO_FOLDER)
    return table


def _failing_rows(table: pd.DataFrame, key_columns: list[str], mask: pd.Series | np.ndarray) -> list[tuple]:
    """Return the key tuples of the rows selected by a boolean mask."""
    return list(table.loc[mask, key_columns].itertuples(index=False, name=None))


def validate_shock_grid_table(grid: pd.DataFrame, tolerance: float = SHOCK_TOLERANCE) -> None:
    """Check that every row of the ES shock grid is finite and consistent with the model and its definitions.

    Raises:
        ValueError: If rows are missing or duplicated, values are non-finite or a consistency rule fails.
    """
    expected_rows = len(SCENARIO_DIRS) * len(ASSETS) * (len(ES_LEVELS) + len(HORIZONS_DAYS) * len(HORIZON_LEVELS))
    if len(grid) != expected_rows:
        raise ValueError(f"Expected {expected_rows} shock-grid rows, found {len(grid)}")
    if grid.duplicated(GRID_KEY_COLUMNS).any():
        raise ValueError(f"Shock grid contains duplicated keys {GRID_KEY_COLUMNS}")
    unknown_methods = sorted(set(grid["method"]) - set(METHOD_ORDER))
    if unknown_methods:
        raise ValueError(f"Unknown methods in the shock grid: {unknown_methods}")

    value_columns = ["confidence_level", "horizon_days", "es_log_loss", "es_price_shock"]
    is_finite = np.isfinite(grid[value_columns].to_numpy(dtype=float)).all(axis=1)
    if not is_finite.all():
        raise ValueError(f"Non-finite shock-grid values for: {_failing_rows(grid, GRID_KEY_COLUMNS, ~is_finite)}")

    is_closed_form = grid["method"] == "closed_form"
    has_ci = grid["ci_low"].notna() & grid["ci_high"].notna()
    no_ci = grid["ci_low"].isna() & grid["ci_high"].isna()
    n_obs = np.array([losses[SCENARIO_KEY[scenario]][asset].size for scenario, asset in zip(grid["scenario"], grid["asset"])])
    expected_flag = grid["horizon_days"].to_numpy() / (1.0 - grid["confidence_level"].to_numpy()) > n_obs

    rules: dict[str, pd.Series] = {
        "0 < confidence_level < 1": grid["confidence_level"].between(0.0, 1.0, inclusive="neither"),
        "horizon_days >= 1": grid["horizon_days"] >= 1,
        "closed_form only at horizon_days == 1": ~is_closed_form | (grid["horizon_days"] == 1),
        "es_log_loss > 0": grid["es_log_loss"] > 0.0,
        "0 < price shock < 1": grid["es_price_shock"].between(0.0, 1.0, inclusive="neither"),
        "price shock <= 1 - exp(-ES) (Jensen)": grid["es_price_shock"] <= -np.expm1(-grid["es_log_loss"]) + tolerance,
        "CI present and ordered for closed_form, NaN for monte_carlo": (is_closed_form & has_ci & (grid["ci_low"] <= grid["ci_high"]))
        | (~is_closed_form & no_ci),
        "out_of_sample_flag == (horizon_days / (1 - p) > n_obs)": grid["out_of_sample_flag"] == pd.Series(expected_flag, index=grid.index),
    }
    for rule, passed in rules.items():
        if not passed.all():
            raise ValueError(f"Shock-grid validation failed ({rule}) for: {_failing_rows(grid, GRID_KEY_COLUMNS, ~passed)}")

    by_level = grid.sort_values(["scenario", "asset", "method", "horizon_days", "confidence_level"])
    group_keys = ["scenario", "asset", "method", "horizon_days"]
    for column in ("es_log_loss", "es_price_shock"):
        step = by_level.groupby(group_keys)[column].diff()
        not_increasing = step.notna() & (step <= 0.0)
        if not_increasing.any():
            raise ValueError(f"{column} must increase with confidence_level; failing: {_failing_rows(by_level, GRID_KEY_COLUMNS, not_increasing)}")

    outside_ci = is_closed_form & ~grid["es_log_loss"].between(grid["ci_low"], grid["ci_high"])
    for scenario_label, asset_label, level in grid.loc[outside_ci, ["scenario", "asset", "confidence_level"]].itertuples(index=False, name=None):
        logger.warning("[%s] %s p=%s: ES point estimate lies outside its bootstrap CI", scenario_label, asset_label, level)


def validate_breakeven_table(table: pd.DataFrame, tolerance: float = 1e-12) -> None:
    """Check that the break-even table is complete, finite and consistent with the two break-even formulas.

    Raises:
        ValueError: If rows are missing or duplicated, values are non-finite or a consistency rule fails.
    """
    expected_rows = len(SCENARIO_DIRS) * len(AAVE_SYMBOL_TO_ASSET)
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} break-even rows, found {len(table)}")
    if table.duplicated(BREAKEVEN_KEY_COLUMNS).any():
        raise ValueError(f"Break-even table contains duplicated keys {BREAKEVEN_KEY_COLUMNS}")

    value_columns = ["lt", "lb", "breakeven_no_bonus", "breakeven_with_bonus"]
    is_finite = np.isfinite(table[value_columns].to_numpy(dtype=float)).all(axis=1)
    if not is_finite.all():
        raise ValueError(f"Non-finite break-even values for: {_failing_rows(table, BREAKEVEN_KEY_COLUMNS, ~is_finite)}")

    rules: dict[str, pd.Series] = {
        "asset matches aave_symbol": table["asset"] == table["aave_symbol"].map(AAVE_SYMBOL_TO_ASSET),
        "0 < lt < 1": table["lt"].between(0.0, 1.0, inclusive="neither"),
        "0 <= lb < 1": table["lb"].between(0.0, 1.0, inclusive="left"),
        "breakeven_no_bonus == 1 - lt": pd.Series(
            np.isclose(table["breakeven_no_bonus"], 1.0 - table["lt"], rtol=0.0, atol=tolerance), index=table.index
        ),
        "breakeven_with_bonus == 1 - lt * (1 + lb)": pd.Series(
            np.isclose(table["breakeven_with_bonus"], 1.0 - table["lt"] * (1.0 + table["lb"]), rtol=0.0, atol=tolerance),
            index=table.index,
        ),
        "0 < with bonus <= no bonus < 1": (table["breakeven_with_bonus"] > 0.0)
        & (table["breakeven_with_bonus"] <= table["breakeven_no_bonus"])
        & (table["breakeven_no_bonus"] < 1.0),
    }
    for rule, passed in rules.items():
        if not passed.all():
            raise ValueError(f"Break-even validation failed ({rule}) for: {_failing_rows(table, BREAKEVEN_KEY_COLUMNS, ~passed)}")


evt_shock_grid_export_df = build_shock_grid_export_table(shock_grid_df)
validate_shock_grid_table(evt_shock_grid_export_df)
evt_shock_grid_reloaded_df = export_shock_table(evt_shock_grid_export_df, EVT_SHOCK_GRID_EXPORT_PATH)
logger.info(
    "ES shock grid written to %s (%d rows; manual notebook dump, NOT for production)",
    EVT_SHOCK_GRID_EXPORT_PATH.resolve(), len(evt_shock_grid_reloaded_df),
)

evt_breakeven_export_df = build_breakeven_export_table(breakeven_df)
validate_breakeven_table(evt_breakeven_export_df)
evt_breakeven_reloaded_df = export_shock_table(evt_breakeven_export_df, EVT_BREAKEVEN_EXPORT_PATH)
logger.info(
    "Break-even shocks written to %s (%d rows; manual notebook dump, NOT for production)",
    EVT_BREAKEVEN_EXPORT_PATH.resolve(), len(evt_breakeven_reloaded_df),
)

display(
    evt_shock_grid_reloaded_df.pivot_table(
        index=["scenario", "asset"],
        columns=["method", "confidence_level", "horizon_days"],
        values="es_price_shock",
    )
    .style.format("{:.2%}", na_rep="-")
    .set_caption(f"ES price shock reloaded from {EVT_SHOCK_GRID_EXPORT_PATH.name} (columns: method, p, h)")
)
display(
    evt_breakeven_reloaded_df.set_index(BREAKEVEN_KEY_COLUMNS)
    .style.format(precision=4)
    .format({"breakeven_no_bonus": "{:.2%}", "breakeven_with_bonus": "{:.2%}"})
    .set_caption(f"Break-even shocks reloaded from {EVT_BREAKEVEN_EXPORT_PATH.name}")
)
