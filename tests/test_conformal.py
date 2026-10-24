"""Conformal layer: the finite-sample correction, coverage on exchangeable data, and the
one thing ACI is actually allowed to promise."""

from __future__ import annotations

import numpy as np
import pytest

from cfd.conformal import adaptive, cqr, split
from cfd.conformal.split import conformal_quantile


def test_conformal_quantile_uses_the_n_plus_one_correction() -> None:
    s = np.arange(1, 11, dtype=float)  # n = 10
    # ceil(11 * 0.9) / 10 = 10/10 -> the maximum score, not the plain 90th percentile
    assert conformal_quantile(s, 0.10) == 10.0
    # ceil(11 * 0.5) / 10 = 6/10 -> the 6th order statistic
    assert conformal_quantile(s, 0.50) == 6.0


def test_conformal_quantile_is_infinite_when_n_is_too_small() -> None:
    assert conformal_quantile(np.array([1.0, 2.0]), 0.05) == float("inf")


def test_conformal_quantile_ignores_non_finite_scores() -> None:
    assert conformal_quantile(np.array([1.0, np.nan, 3.0, np.inf]), 0.5) <= 3.0


@pytest.fixture
def exchangeable():
    rng = np.random.default_rng(0)
    n_cal, n_test = 2000, 20000
    x = rng.uniform(0, 1, n_cal + n_test)
    sigma = 0.5 + 2.0 * x  # heteroscedastic: a fixed-width band is the wrong shape
    y = 10 * x + rng.normal(0, sigma)
    yhat = 10 * x
    z = 1.2816  # true 10th/90th normal quantile offsets -- a deliberately miscalibrated model
    qlo, qhi = yhat - 0.6 * z * sigma, yhat + 0.6 * z * sigma
    cal, test = slice(0, n_cal), slice(n_cal, None)
    return y, yhat, qlo, qhi, cal, test, sigma


def test_split_conformal_hits_nominal_on_exchangeable_data(exchangeable) -> None:
    y, yhat, _, _, cal, test, _ = exchangeable
    qhat = split.calibrate(y[cal], yhat[cal], alpha=0.10)
    lo, hi = split.interval(yhat[test], qhat, floor=None)
    cov = np.mean((y[test] >= lo) & (y[test] <= hi))
    assert cov == pytest.approx(0.90, abs=0.015)


def test_cqr_repairs_an_undercovering_quantile_model(exchangeable) -> None:
    y, _, qlo, qhi, cal, test, _ = exchangeable
    raw_cov = np.mean((y[test] >= qlo[test]) & (y[test] <= qhi[test]))
    assert raw_cov < 0.70  # the raw quantiles are badly too narrow by construction

    qhat = cqr.calibrate(y[cal], qlo[cal], qhi[cal], alpha=0.10)
    assert qhat > 0
    lo, hi = cqr.interval(qlo[test], qhi[test], qhat, floor=None)
    cov = np.mean((y[test] >= lo) & (y[test] <= hi))
    assert cov == pytest.approx(0.90, abs=0.015)


def test_cqr_keeps_the_width_adaptive(exchangeable) -> None:
    y, yhat, qlo, qhi, cal, test, sigma = exchangeable
    s_hat = split.calibrate(y[cal], yhat[cal], 0.10)
    c_hat = cqr.calibrate(y[cal], qlo[cal], qhi[cal], 0.10)
    s_lo, s_hi = split.interval(yhat[test], s_hat, floor=None)
    c_lo, c_hi = cqr.interval(qlo[test], qhi[test], c_hat, floor=None)
    assert np.std(s_hi - s_lo) == pytest.approx(0.0)  # one width everywhere
    assert np.std(c_hi - c_lo) > 0.5  # wide where sigma is large, tight where small
    assert np.corrcoef(c_hi - c_lo, sigma[test])[0, 1] > 0.99


def test_one_sided_bound_hits_tau(exchangeable) -> None:
    y, _, _, qhi, cal, test, _ = exchangeable
    for tau in (0.8, 0.9):
        qhat = cqr.calibrate_upper(y[cal], qhi[cal], tau)
        bound = cqr.upper_bound(qhi[test], qhat, floor=None)
        assert np.mean(y[test] <= bound) == pytest.approx(tau, abs=0.015)


def test_nominal_to_taus() -> None:
    assert cqr.nominal_to_taus(0.90) == (0.05, 0.95)
    assert cqr.nominal_to_taus(0.50) == (0.25, 0.75)


def test_aci_long_run_miscoverage_tracks_alpha_under_a_level_shift() -> None:
    """The only thing ACI promises: time-averaged miscoverage -> alpha, shift or no shift."""
    rng = np.random.default_rng(1)
    n_series, n_days, alpha = 200, 400, 0.10
    calib_scores = np.abs(rng.normal(0, 1, 5000))

    # after day 100 the world shifts up by 2 sigma; a static interval will under-cover
    y_by_day = [rng.normal(0 if t < 100 else 2.0, 1, n_series) for t in range(n_days)]
    zeros = [np.zeros(n_series) for _ in range(n_days)]
    fn = adaptive.cqr_interval_fn(calib_scores, zeros, zeros, floor=-np.inf)

    static = adaptive.run_aci(y_by_day, fn, alpha=alpha, gamma=0.0)
    aci = adaptive.run_aci(y_by_day, fn, alpha=alpha, gamma=0.05)

    static_miss = 1 - np.mean(static.coverage[100:])
    aci_miss = 1 - np.mean(aci.coverage[100:])
    assert static_miss > 0.3  # the static method breaks
    assert aci_miss == pytest.approx(alpha, abs=0.03)  # ACI's frequency recovers
    assert np.mean(aci.width[100:]) > np.mean(static.width[100:])  # ... by paying in width


def test_aci_rejects_a_zero_lag() -> None:
    with pytest.raises(ValueError):
        adaptive.run_aci([], lambda t, a: (0, 0), 0.1, 0.1, feedback_lag=0)
