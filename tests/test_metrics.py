from __future__ import annotations

import numpy as np
import pytest

from cfd.eval.metrics import (
    coverage,
    mase,
    mean_width,
    paired_bootstrap_ci,
    pinball_loss,
    rmsse,
)


def _train_series(n: int = 60) -> np.ndarray:
    """Seasonal + trending, so the MASE/RMSSE scale denominators are non-zero."""
    t = np.arange(n, dtype=float)
    return 10.0 + 5.0 * np.sin(2 * np.pi * t / 7.0) + 0.05 * t


def test_perfect_forecast_scores_zero() -> None:
    y = np.array([3.0, 5.0, 4.0, 6.0])
    train = _train_series()
    assert mase(y, y, train) == pytest.approx(0.0)
    assert rmsse(y, y, train) == pytest.approx(0.0)


def test_scale_free_metrics_return_nan_on_a_flat_history() -> None:
    """A constant history has no scale to divide by; nan is the honest answer."""
    flat = np.full(60, 4.0)
    y = np.array([4.0, 5.0])
    assert np.isnan(mase(y, y, flat))
    assert np.isnan(rmsse(y, y, flat))


def test_pinball_is_asymmetric_in_the_right_direction() -> None:
    y = np.array([10.0])
    # tau=0.9 penalises under-prediction ~9x more than over-prediction
    under = pinball_loss(y, np.array([9.0]), tau=0.9)
    over = pinball_loss(y, np.array([11.0]), tau=0.9)
    assert under == pytest.approx(0.9)
    assert over == pytest.approx(0.1)
    assert under > over


def test_pinball_at_median_is_half_absolute_error() -> None:
    y, q = np.array([10.0]), np.array([7.0])
    assert pinball_loss(y, q, tau=0.5) == pytest.approx(1.5)


def test_coverage_and_width() -> None:
    y = np.array([1.0, 5.0, 9.0, 12.0])
    lo = np.array([0.0, 4.0, 10.0, 11.0])
    hi = np.array([2.0, 6.0, 11.0, 13.0])
    assert coverage(y, lo, hi) == pytest.approx(0.75)
    assert mean_width(lo, hi) == pytest.approx(1.75)


def test_coverage_is_gameable_by_widening_hence_width_matters() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    assert coverage(y, np.full(500, -1e6), np.full(500, 1e6)) == pytest.approx(1.0)


def test_paired_bootstrap_detects_a_real_improvement() -> None:
    baseline = np.array([100.0, 104.0, 98.0, 102.0, 101.0])
    better = baseline - np.array([8.0, 9.0, 7.5, 8.5, 8.0])

    absolute = paired_bootstrap_ci(better, baseline, n_boot=2000, seed=1)
    assert absolute["point"] < 0
    assert absolute["hi"] < 0  # improvement excludes zero

    relative = paired_bootstrap_ci(better, baseline, n_boot=2000, seed=1, relative=True)
    assert relative["point"] == pytest.approx(0.081, abs=0.01)
    assert relative["lo"] > 0


def test_paired_bootstrap_finds_nothing_when_there_is_nothing() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(100, 5, 8)
    b = a + rng.normal(0, 5, 8)
    ci = paired_bootstrap_ci(a, b, n_boot=2000, seed=2)
    assert ci["lo"] < 0 < ci["hi"]


def test_paired_bootstrap_rejects_mismatched_input() -> None:
    with pytest.raises(ValueError, match="must match"):
        paired_bootstrap_ci(np.zeros(4), np.zeros(5))
