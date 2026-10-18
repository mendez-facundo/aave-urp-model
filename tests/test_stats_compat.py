"""Plain-assert checks for src.utils.stats_compat (run directly, no pytest needed).

The reference values are closed forms or published constants, so the checks do not need
``scipy.stats`` (which cannot be imported on machines where Windows Application Control
blocks a scipy DLL). When ``scipy.stats`` is importable, the fallback is also compared
with it.

Run from the repository root with: venv/Scripts/python.exe tests/test_stats_compat.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import special  # noqa: E402

from src.utils import stats_compat as sc  # noqa: E402
from src.utils.stats_compat import genpareto  # noqa: E402


def check_close(actual: float, expected: float, tol: float, label: str) -> None:
    assert abs(actual - expected) <= tol, f"{label}: {actual!r} != {expected!r} (tol {tol})"


def test_gpd_closed_forms() -> None:
    # c = 0.5, scale = 2, x = 2: 1 + c * x / scale = 1.5, cdf = 1 - 1.5^(-2) and pdf = (1 / scale) * 1.5^(-3).
    check_close(genpareto.cdf(2.0, 0.5, loc=0.0, scale=2.0), 1.0 - 1.5**-2, 1e-14, "cdf c=0.5")
    check_close(genpareto.logpdf(2.0, 0.5, loc=0.0, scale=2.0), np.log(0.5 * 1.5**-3), 1e-14, "logpdf c=0.5")
    check_close(genpareto.pdf(2.0, 0.5, loc=0.0, scale=2.0), 0.5 * 1.5**-3, 1e-14, "pdf c=0.5")
    # c = 0 is the exponential distribution.
    check_close(genpareto.cdf(1.5, 0.0, scale=2.0), 1.0 - np.exp(-0.75), 1e-14, "cdf c=0")
    check_close(genpareto.logpdf(1.5, 0.0, scale=2.0), -np.log(2.0) - 0.75, 1e-14, "logpdf c=0")
    # c = -0.5, scale = 1: support [0, 2], cdf(x) = 1 - (1 - x / 2)^2.
    check_close(genpareto.cdf(1.0, -0.5), 0.75, 1e-14, "cdf c=-0.5")
    assert genpareto.cdf(2.5, -0.5) == 1.0 and genpareto.cdf(-1.0, -0.5) == 0.0 and genpareto.cdf(-1.0, 0.5) == 0.0
    assert genpareto.logpdf(2.5, -0.5) == -np.inf and genpareto.logpdf(-1.0, 0.5) == -np.inf
    # c = -1 is the uniform distribution on [0, 1].
    check_close(genpareto.cdf(0.3, -1.0), 0.3, 1e-14, "cdf c=-1")
    # Location and scale: the cdf only depends on (x - loc) / scale.
    check_close(genpareto.cdf(5.0, 0.2, loc=1.0, scale=2.0), genpareto.cdf(2.0, 0.2), 1e-14, "loc/scale")


def test_gpd_ppf_round_trip() -> None:
    probabilities = np.array([1e-6, 0.01, 0.25, 0.5, 0.9, 0.99, 0.999, 1.0 - 1e-9])
    for c in (-0.4, -1e-9, 0.0, 1e-9, 0.15, 0.5, 1.3):
        quantiles = genpareto.ppf(probabilities, c, loc=0.5, scale=0.03)
        np.testing.assert_allclose(genpareto.cdf(quantiles, c, loc=0.5, scale=0.03), probabilities, rtol=1e-9, atol=1e-12)
    assert genpareto.ppf(0.0, 0.3) == 0.0 and genpareto.ppf(1.0, 0.3) == np.inf
    check_close(genpareto.ppf(1.0, -0.5), 2.0, 1e-14, "ppf(1) c=-0.5")
    assert np.isnan(genpareto.ppf(1.5, 0.3)) and np.isnan(genpareto.ppf(-0.1, 0.3))
    # Closed form VaR of the POT estimator: ppf(1 - q) = ((q)^(-c) - 1) / c.
    check_close(genpareto.ppf(0.99, 0.2), (0.01**-0.2 - 1.0) / 0.2, 1e-12, "ppf closed form")


def test_gpd_rvs_matches_inverse_transform() -> None:
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    draws = genpareto.rvs(0.25, loc=0.0, scale=0.02, size=1000, random_state=rng_a)
    expected = genpareto.ppf(rng_b.uniform(size=1000), 0.25, loc=0.0, scale=0.02)
    assert draws.shape == (1000,) and np.array_equal(draws, expected)
    big = genpareto.rvs(0.2, scale=2.0, size=400_000, random_state=np.random.default_rng(1))
    check_close(float(big.mean()), 2.0 / (1.0 - 0.2), 0.02, "rvs mean")


def test_gpd_expect() -> None:
    check_close(genpareto.expect(lambda z: z, args=(0.5,), scale=2.0), 2.0 / (1.0 - 0.5), 1e-7, "mean c=0.5")
    check_close(genpareto.expect(lambda z: np.exp(-z), args=(0.0,), scale=1.0), 0.5, 1e-9, "E exp(-Z), exponential")
    check_close(genpareto.expect(lambda z: z, args=(-0.5,), scale=1.0), 1.0 / (1.0 + 0.5), 1e-9, "mean c=-0.5")
    # Laplace transform of GPD(c = 0.3, scale = 0.04) against a fine quantile-grid average.
    u = (np.arange(2_000_000) + 0.5) / 2_000_000
    reference = float(np.mean(np.exp(-genpareto.ppf(u, 0.3, scale=0.04))))
    check_close(genpareto.expect(lambda z: np.exp(-z), args=(0.3,), scale=0.04), reference, 1e-6, "Laplace transform")


def test_gpd_fit_recovers_parameters() -> None:
    for true_c, true_scale in ((0.2, 1.5), (0.0, 0.03), (-0.2, 2.0)):
        sample = genpareto.rvs(true_c, scale=true_scale, size=20_000, random_state=np.random.default_rng(11))
        c_hat, loc_hat, scale_hat = genpareto.fit(sample, floc=0.0)
        assert loc_hat == 0.0
        check_close(c_hat, true_c, 0.05, f"fit c (true {true_c})")
        assert abs(scale_hat / true_scale - 1.0) < 0.05, f"fit scale {scale_hat} vs {true_scale}"
        # The polished fit has a likelihood at least as high as the default (scipy-equivalent) fit.
        c_pol, _, scale_pol = genpareto.fit(sample, floc=0.0, polish=True)
        assert genpareto.logpdf(sample, c_pol, scale=scale_pol).sum() >= genpareto.logpdf(sample, c_hat, scale=scale_hat).sum() - 1e-9
        check_close(c_pol, c_hat, 5e-3, "polish vs default c")
    # Small sample, as in the POT application (about 80 excesses).
    small = genpareto.rvs(0.15, scale=0.025, size=80, random_state=np.random.default_rng(3))
    c_hat, _, scale_hat = genpareto.fit(small, floc=0.0)
    assert -0.6 < c_hat < 1.0 and 0.0 < scale_hat < 0.2
    try:
        genpareto.fit(small, floc=0.5)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("fit with floc != 0 must raise NotImplementedError")


def test_norm_and_chi2() -> None:
    check_close(sc.norm.ppf(0.975), 1.959963984540054, 1e-12, "norm.ppf(0.975)")
    check_close(sc.norm.ppf(0.5), 0.0, 1e-15, "norm.ppf(0.5)")
    check_close(sc.norm.sf(1.6448536269514722), 0.05, 1e-12, "norm.sf(1.645)")
    check_close(sc.norm.sf(0.0), 0.5, 1e-15, "norm.sf(0)")
    check_close(sc.norm.sf(-1.0) + sc.norm.sf(1.0), 1.0, 1e-15, "norm.sf symmetry")
    # chi2.sf: published 5% critical values and the incomplete-gamma identity.
    check_close(sc.chi2.sf(3.841458820694124, 1), 0.05, 1e-12, "chi2.sf df=1")
    check_close(sc.chi2.sf(5.991464547107979, 2), 0.05, 1e-12, "chi2.sf df=2")
    check_close(sc.chi2.sf(18.307038053275146, 10), 0.05, 1e-12, "chi2.sf df=10")
    check_close(sc.chi2.sf(0.0, 3), 1.0, 0.0, "chi2.sf(0)")
    for df in (1, 4, 9):
        for x in (0.3, 2.0, 11.0):
            check_close(sc.chi2.sf(x, df), float(special.gammaincc(df / 2.0, x / 2.0)), 1e-15, f"chi2.sf({x}, {df})")
    check_close(sc.chi2.sf(2.0, 2), np.exp(-1.0), 1e-14, "chi2.sf df=2 closed form exp(-x/2)")


def test_skew_and_kurtosis() -> None:
    check_close(sc.skew([1, 2, 3, 4, 5]), 0.0, 1e-15, "skew symmetric")
    check_close(sc.skew([2, 8, 0, 4, 1, 9, 9, 0]), 0.2650554122698573, 1e-12, "skew scipy doc value")
    check_close(sc.kurtosis([1, 2, 3, 4, 5]), -1.3, 1e-12, "excess kurtosis biased")
    check_close(sc.kurtosis([1, 2, 3, 4, 5], fisher=False), 1.7, 1e-12, "Pearson kurtosis")
    rng = np.random.default_rng(5)
    for values in (rng.standard_normal(500), rng.standard_t(4, size=800), rng.exponential(size=300)):
        series = pd.Series(values)  # pandas implements the unbiased G1 and G2 estimators
        check_close(sc.skew(values, bias=False), float(series.skew()), 1e-10, "unbiased skew vs pandas")
        check_close(sc.kurtosis(values, fisher=True, bias=False), float(series.kurt()), 1e-10, "unbiased kurtosis vs pandas")


def test_agreement_with_scipy_stats_when_available() -> None:
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        print("  (scipy.stats not importable here: comparison with the real backend skipped)")
        return
    rng = np.random.default_rng(2)
    sample = scipy_stats.genpareto.rvs(0.18, scale=0.03, size=90, random_state=rng)
    ours = genpareto.fit(sample, floc=0.0)
    theirs = scipy_stats.genpareto.fit(sample, floc=0.0)
    np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=1e-9)
    grid = np.linspace(0.0, 0.3, 7)
    np.testing.assert_allclose(genpareto.cdf(grid, 0.2, scale=0.04), scipy_stats.genpareto.cdf(grid, 0.2, scale=0.04), rtol=1e-12)
    np.testing.assert_allclose(genpareto.logpdf(grid, 0.2, scale=0.04), scipy_stats.genpareto.logpdf(grid, 0.2, scale=0.04), rtol=1e-12)
    check_close(
        genpareto.expect(lambda z: np.exp(-z), args=(0.2,), scale=0.04),
        float(scipy_stats.genpareto.expect(lambda z: np.exp(-z), args=(0.2,), scale=0.04)), 1e-12, "expect vs scipy",
    )
    check_close(sc.skew(sample, bias=False), float(scipy_stats.skew(sample, bias=False)), 1e-12, "skew vs scipy")
    check_close(sc.kurtosis(sample, bias=False), float(scipy_stats.kurtosis(sample, bias=False)), 1e-12, "kurtosis vs scipy")
    check_close(sc.norm.ppf(0.975), float(scipy_stats.norm.ppf(0.975)), 1e-15, "norm.ppf vs scipy")
    check_close(sc.chi2.sf(7.3, 4), float(scipy_stats.chi2.sf(7.3, 4)), 1e-14, "chi2.sf vs scipy")


TESTS: List[Tuple[str, Callable[[], None]]] = [
    (name, func) for name, func in sorted(globals().items()) if name.startswith("test_") and callable(func)
]


def main() -> int:
    failures = 0
    for name, func in TESTS:
        try:
            func()
        except Exception as exc:  # noqa: BLE001 - report every failing check, then exit non-zero
            failures += 1
            print(f"FAIL {name}: {exc!r}")
        else:
            print(f"ok   {name}")
    print(f"{len(TESTS) - failures}/{len(TESTS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
