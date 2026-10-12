"""Fallback for the few ``scipy.stats`` functions that the notebooks use.

Why this module exists
----------------------
On some Windows machines, Windows Application Control (Smart App Control / WDAC)
blocks the compiled scipy extension ``scipy.sparse.csgraph._shortest_path``
("DLL load failed ... Una directiva de Control de aplicaciones bloqueo este
archivo"). ``import scipy.stats`` needs that extension through
``scipy.stats._qmc``, so it raises ``ImportError`` even though
``scipy.special``, ``scipy.optimize``, ``scipy.integrate`` and ``scipy.linalg``
import fine. The system policy is not touched: this module re-implements only
what the notebooks need, using ``numpy``, ``scipy.special``, ``scipy.optimize``
and ``scipy.integrate``.

``scipy.stats`` is preferred whenever it can be imported. Notebooks select the
backend with::

    try:
        from scipy import stats
        from scipy.stats import genpareto
    except ImportError:
        from src.utils import stats_compat as stats
        from src.utils.stats_compat import genpareto

What is provided (names and signatures follow ``scipy.stats``)
--------------------------------------------------------------
* ``genpareto`` (scalar shape ``c``): ``logpdf``, ``pdf``, ``cdf``, ``ppf``,
  ``rvs``, ``expect`` and ``fit(data, floc=0.0)``.
* ``norm``: ``ppf``, ``sf``. ``chi2``: ``sf``.
* ``skew`` and ``kurtosis`` (sample-bias options as in scipy).

Fidelity to scipy
-----------------
``genpareto.fit`` reproduces ``scipy.stats.genpareto.fit(data, floc=0.0)`` step
by step: the same penalised negative log-likelihood, the starting point
``(c, scale) = (1, 1)`` that scipy derives for this distribution, and the same
``scipy.optimize.fmin`` call with its default tolerances (``xtol = ftol = 1e-4``).
Results therefore agree with scipy to numerical precision, including the small
convergence error that the loose ``fmin`` tolerance leaves in the estimates.
``polish=True`` adds tightly converged Nelder-Mead restarts for a sharper
maximum likelihood estimate. ``rvs`` draws ``U ~ Uniform(0, 1)`` from the given
generator and applies ``ppf``, as scipy does, so a seeded generator gives the
same sample. ``expect`` uses the same three-piece ``quad`` integration as scipy.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from scipy import integrate, optimize, special

logger = logging.getLogger(__name__)

BACKEND_NAME: str = "src.utils.stats_compat (numpy / scipy.special / scipy.optimize fallback)"

# Penalty weight scipy adds per observation that falls outside the support (log of float max, times 100).
_LOGXMAX: float = float(np.log(np.finfo(float).max))
_SQRT2: float = float(np.sqrt(2.0))
# MLE regularity (finite Fisher information) requires xi > -0.5; fits below it are returned but flagged.
_XI_REGULARITY_BOUND: float = -0.5


class FitError(RuntimeError):
    """Raised when a maximum likelihood fit ends outside the allowed parameter range."""


class _Norm:
    """Standard normal distribution (``scipy.stats.norm`` subset)."""

    @staticmethod
    def ppf(q: float | np.ndarray) -> float | np.ndarray:
        """Quantile function (inverse CDF)."""
        return special.ndtri(q)

    @staticmethod
    def sf(x: float | np.ndarray) -> float | np.ndarray:
        """Survival function ``P(Z > x)``, computed as ``erfc(x / sqrt(2)) / 2`` as scipy does."""
        return 0.5 * special.erfc(np.asarray(x, dtype=float) / _SQRT2)[()]


class _Chi2:
    """Chi-square distribution (``scipy.stats.chi2`` subset)."""

    @staticmethod
    def sf(x: float | np.ndarray, df: float) -> float | np.ndarray:
        """Survival function ``P(X > x)`` = regularised upper incomplete gamma ``Q(df / 2, x / 2)``."""
        if not df > 0:
            raise ValueError(f"df must be positive, got {df}")
        values = np.asarray(x, dtype=float)
        return np.where(values > 0.0, special.gammaincc(df / 2.0, np.maximum(values, 0.0) / 2.0), 1.0)[()]


class _GenPareto:
    """Generalised Pareto distribution with shape ``c`` (= xi), ``loc`` and ``scale`` (``scipy.stats.genpareto`` subset).

    Only a scalar shape ``c`` is supported. The support is ``[loc, inf)`` for
    ``c >= 0`` and ``[loc, loc - scale / c]`` for ``c < 0``.
    """

    # --- helpers ---------------------------------------------------------------------------------

    @staticmethod
    def _check(c: float, scale: float) -> None:
        if not np.isfinite(c):
            raise ValueError(f"shape c must be finite, got {c}")
        if not scale > 0:
            raise ValueError(f"scale must be positive, got {scale}")

    @staticmethod
    def _upper(c: float) -> float:
        """Upper end of the standardised support."""
        return -1.0 / c if c < 0 else np.inf

    @staticmethod
    def _std_logpdf(z: np.ndarray, c: float) -> np.ndarray:
        """Log-density of the standardised distribution on its support."""
        if c == 0:
            return -z
        return -special.xlog1py(c + 1.0, c * z) / c

    # --- public API ------------------------------------------------------------------------------

    def logpdf(self, x: float | np.ndarray, c: float, loc: float = 0.0, scale: float = 1.0) -> float | np.ndarray:
        """Log-density; ``-inf`` outside the support."""
        self._check(c, scale)
        z = (np.asarray(x, dtype=float) - loc) / scale
        out = np.full(z.shape, -np.inf)
        with np.errstate(invalid="ignore"):
            inside = (z >= 0.0) & (z <= self._upper(c))
        out[inside] = self._std_logpdf(z[inside], c) - np.log(scale)
        return out[()]

    def pdf(self, x: float | np.ndarray, c: float, loc: float = 0.0, scale: float = 1.0) -> float | np.ndarray:
        """Density; zero outside the support."""
        return np.exp(self.logpdf(x, c, loc, scale))

    def cdf(self, x: float | np.ndarray, c: float, loc: float = 0.0, scale: float = 1.0) -> float | np.ndarray:
        """Cumulative distribution function ``1 - (1 + c z)^(-1/c)`` with ``z = (x - loc) / scale``."""
        self._check(c, scale)
        z = (np.asarray(x, dtype=float) - loc) / scale
        upper = self._upper(c)
        out = np.zeros(z.shape)
        out[z >= upper] = 1.0
        inside = (z > 0.0) & (z < upper)
        out[inside] = -special.inv_boxcox1p(-z[inside], -c)
        return out[()]

    def ppf(self, q: float | np.ndarray, c: float, loc: float = 0.0, scale: float = 1.0) -> float | np.ndarray:
        """Quantile function ``loc + scale * ((1 - q)^(-c) - 1) / c``; ``nan`` for ``q`` outside [0, 1]."""
        self._check(c, scale)
        q = np.asarray(q, dtype=float)
        out = np.full(q.shape, np.nan)
        out[q == 0.0] = loc
        out[q == 1.0] = loc + scale * self._upper(c)
        inside = (q > 0.0) & (q < 1.0)
        out[inside] = loc + scale * (-special.boxcox1p(-q[inside], -c))
        return out[()]

    def rvs(
        self,
        c: float,
        loc: float = 0.0,
        scale: float = 1.0,
        size: int | tuple[int, ...] | None = None,
        random_state: np.random.Generator | None = None,
    ) -> float | np.ndarray:
        """Random variates by inverse transform of ``Uniform(0, 1)`` draws (the generic scipy method)."""
        self._check(c, scale)
        generator = random_state if random_state is not None else np.random.default_rng()
        uniforms = generator.uniform(size=size)
        return self.ppf(uniforms, c, 0.0, 1.0) * scale + loc

    def expect(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        args: tuple[float, ...] = (),
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> float:
        """Expected value of ``func(X)`` by numerical integration, split at the 5% and 95% quantiles as scipy does."""
        (c,) = args
        self._check(c, scale)

        def integrand(x: float, shape: float) -> float:
            return func(x) * self.pdf(x, shape, loc=loc, scale=scale)

        lower = loc
        upper = loc + self._upper(c) * scale
        cdf_lower, cdf_upper = self.cdf([lower, upper], c, loc=loc, scale=scale)
        span = cdf_upper - cdf_lower
        body_start, body_end = loc + self.ppf(cdf_lower + span * np.array([0.05, 0.95]), c, 0.0, 1.0) * scale
        total = sum(
            integrate.quad(integrand, a, b, args=(c,))[0]
            for a, b in ((lower, body_start), (body_start, body_end), (body_end, upper))
        )
        return float(total)

    # --- maximum likelihood ----------------------------------------------------------------------

    def _penalised_nnlf(self, theta: np.ndarray, data: np.ndarray) -> float:
        """Negative log-likelihood of ``data`` for ``theta = (c, scale)`` with ``loc = 0``, as scipy defines it.

        Observations outside the support are dropped and each one adds a large penalty.
        """
        c, scale = float(theta[0]), float(theta[1])
        if not np.isfinite(c) or scale <= 0:
            return np.inf
        z = data / scale
        with np.errstate(invalid="ignore"):
            inside = (z >= 0.0) & (z <= self._upper(c))
        n_bad = int((~inside).sum())
        with np.errstate(all="ignore"):
            terms = self._std_logpdf(z[inside], c)
        finite = np.isfinite(terms)
        n_bad += int((~finite).sum())
        return float(-terms[finite].sum() + n_bad * _LOGXMAX * 100.0 + data.size * np.log(scale))

    def fit(self, data: np.ndarray, floc: float = 0.0, polish: bool = False) -> tuple[float, float, float]:
        """Maximum likelihood fit of ``(c, loc, scale)`` with the location fixed at zero.

        Args:
            data: Non-negative sample (threshold excesses).
            floc: Fixed location; only ``0.0`` is supported.
            polish: If ``True``, refine the scipy-equivalent solution with tightly converged
                Nelder-Mead restarts (``xatol = 1e-10``, ``fatol = 1e-12``).

        Returns:
            ``(c, loc, scale)`` as ``scipy.stats.genpareto.fit`` does.

        Raises:
            NotImplementedError: If ``floc`` is not 0.
            ValueError: If the data contain non-finite values.
            FitError: If the optimiser ends outside the allowed range (``scale <= 0`` or a non-finite shape).
        """
        if floc != 0.0:
            raise NotImplementedError("Only floc = 0 (fit to threshold excesses) is implemented")
        sample = np.asarray(data, dtype=float).ravel()
        if not np.isfinite(sample).all():
            raise ValueError("The data contains non-finite values.")

        # scipy's start for genpareto with loc fixed: c = 1 and scale = 1 (see module docstring).
        with np.errstate(all="ignore"):
            solution = optimize.fmin(self._penalised_nnlf, np.array([1.0, 1.0]), args=(sample,), disp=0)
        if not self._is_valid(solution, sample):
            # scipy would fail here. Retry once from a data-driven start rather than discard the sample.
            logger.debug("fmin from (1, 1) ended outside the parameter range; retrying from a moment-based start")
            with np.errstate(all="ignore"):
                solution = optimize.fmin(self._penalised_nnlf, np.array([0.1, sample.mean()]), args=(sample,), disp=0)
        if not self._is_valid(solution, sample):
            raise FitError("Optimization converged to parameters that are outside the range allowed by the distribution.")

        if polish:
            solution = self._polish(solution, sample)
        c, scale = float(solution[0]), float(solution[1])
        if c <= _XI_REGULARITY_BOUND:
            logger.debug("GPD shape %.3f <= -0.5: MLE regularity conditions fail (infinite Fisher information)", c)
        return c, 0.0, scale

    def _is_valid(self, theta: np.ndarray, data: np.ndarray) -> bool:
        """Whether a solution is a finite shape, a positive scale and an unpenalised likelihood."""
        if not (np.isfinite(theta).all() and theta[1] > 0):
            return False
        return bool(self._penalised_nnlf(theta, data) < _LOGXMAX * 100.0)  # a larger value means out-of-support points

    def _polish(self, theta: np.ndarray, data: np.ndarray, max_restarts: int = 5) -> np.ndarray:
        """Restart Nelder-Mead with tight tolerances until the likelihood stops improving."""
        best, best_value = np.asarray(theta, dtype=float), self._penalised_nnlf(theta, data)
        for _ in range(max_restarts):
            with np.errstate(all="ignore"):
                result = optimize.minimize(
                    self._penalised_nnlf, best, args=(data,), method="Nelder-Mead",
                    options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000, "maxfev": 8000},
                )
            if not result.fun < best_value - 1e-13:
                break
            best, best_value = result.x, float(result.fun)
        return best


def skew(values: np.ndarray, bias: bool = True) -> float:
    """Sample skewness; with ``bias=False`` the adjusted Fisher-Pearson coefficient (``scipy.stats.skew``)."""
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    deviations = x - x.mean()
    m2 = np.mean(deviations**2)
    m3 = np.mean(deviations**3)
    if m2 == 0:
        return float("nan")
    result = m3 / m2**1.5
    if not bias and n > 2:
        result = np.sqrt((n - 1.0) * n) / (n - 2.0) * result
    return float(result)


def kurtosis(values: np.ndarray, fisher: bool = True, bias: bool = True) -> float:
    """Sample kurtosis; ``fisher=True`` returns the excess kurtosis and ``bias=False`` the unbiased estimator."""
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    deviations = x - x.mean()
    m2 = np.mean(deviations**2)
    m4 = np.mean(deviations**4)
    if m2 == 0:
        return float("nan")
    result = m4 / m2**2
    if not bias and n > 3:
        result = (1.0 / (n - 2) / (n - 3) * ((n**2 - 1.0) * m4 / m2**2 - 3.0 * (n - 1) ** 2)) + 3.0
    return float(result - 3.0 if fisher else result)


norm = _Norm()
chi2 = _Chi2()
genpareto = _GenPareto()

__all__ = ["BACKEND_NAME", "FitError", "chi2", "genpareto", "kurtosis", "norm", "skew"]
