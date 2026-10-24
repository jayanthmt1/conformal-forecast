"""Newsvendor layer: the fractile, the interpolation, and the cost identity."""

from __future__ import annotations

import numpy as np
import pytest

from cfd.decision.newsvendor import critical_fractile, interpolate_quantile, newsvendor_cost


def test_critical_fractile() -> None:
    assert critical_fractile(1, 1) == 0.5
    assert critical_fractile(4, 1) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        critical_fractile(0, 1)


def test_interpolation_is_exact_at_the_grid_and_linear_between() -> None:
    taus = np.array([0.1, 0.5, 0.9])
    q = np.array([[0.0, 10.0, 20.0], [5.0, 5.0, 5.0]])
    assert interpolate_quantile(taus, q, 0.5).tolist() == [10.0, 5.0]
    assert interpolate_quantile(taus, q, 0.7).tolist() == [15.0, 5.0]


def test_interpolation_refuses_to_extrapolate() -> None:
    taus = np.array([0.1, 0.9])
    q = np.array([[0.0, 20.0]])
    assert interpolate_quantile(taus, q, 0.99).tolist() == [20.0]
    assert interpolate_quantile(taus, q, 0.01).tolist() == [0.0]


def test_cost_identity() -> None:
    y = np.array([10.0, 10.0])
    order = np.array([13.0, 6.0])
    holding, stockout = newsvendor_cost(y, order, p=4.0, h=1.0)
    assert holding.tolist() == [3.0, 0.0]
    assert stockout.tolist() == [0.0, 16.0]


def test_ordering_the_critical_fractile_beats_ordering_the_mean() -> None:
    """The whole thesis in one line: on skewed demand with p > h, the mean under-orders."""
    rng = np.random.default_rng(0)
    y = rng.lognormal(2.0, 0.8, 200_000)
    p, h = 4.0, 1.0
    tau = critical_fractile(p, h)
    cost_mean = sum(c.sum() for c in newsvendor_cost(y, np.full_like(y, y.mean()), p, h))
    cost_q = sum(c.sum() for c in newsvendor_cost(y, np.full_like(y, np.quantile(y, tau)), p, h))
    assert cost_q < 0.95 * cost_mean  # ~9% cheaper on this distribution; grows with p/h
