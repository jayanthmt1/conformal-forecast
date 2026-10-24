from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cfd.config import load_config
from cfd.data.features import build_features
from cfd.data.synth import make_synthetic_panel
from cfd.models.baselines import croston, forecast_panel, seasonal_naive
from cfd.splits import folds_from_config


def test_seasonal_naive_tiles_the_last_week() -> None:
    history = np.arange(30, dtype=float)
    out = seasonal_naive(history, horizon=10, season=7)
    assert out[:7].tolist() == history[-7:].tolist()
    assert out[7:].tolist() == history[-7:-4].tolist()


def test_seasonal_naive_survives_an_empty_history() -> None:
    assert seasonal_naive(np.array([]), horizon=5).tolist() == [0.0] * 5


def test_croston_recovers_the_demand_rate() -> None:
    # demand of 10 every 5th day, first sale on day 4 -> the initial interval estimate is
    # already correct, so the rate is exactly 10/5 = 2.0/day
    history = np.zeros(100)
    history[4::5] = 10.0
    out = croston(history, horizon=3, alpha=0.1, variant="classic")
    assert out.tolist() == pytest.approx([2.0, 2.0, 2.0], abs=1e-9)


def test_croston_sba_shrinks_the_classic_estimate() -> None:
    history = np.zeros(100)
    history[4::5] = 10.0
    classic = croston(history, horizon=1, alpha=0.1, variant="classic")[0]
    sba = croston(history, horizon=1, alpha=0.1, variant="sba")[0]
    assert sba == pytest.approx(classic * 0.95)


def test_croston_interval_estimate_lags_when_the_series_opens_with_a_sale() -> None:
    """A leading non-zero has no preceding gap, so the interval is seeded at 1 and
    smooths toward the truth. With alpha=0.1 that lag is still visible after 100 days --
    worth knowing before reading too much into Croston on short intermittent series."""
    history = np.zeros(100)
    history[::5] = 10.0
    rate = croston(history, horizon=1, alpha=0.1, variant="classic")[0]
    assert rate > 2.0
    assert rate == pytest.approx(2.24, abs=0.05)


def test_croston_rejects_an_unknown_variant() -> None:
    with pytest.raises(ValueError, match="unknown croston variant"):
        croston(np.array([0.0, 5.0, 0.0]), horizon=2, variant="nope")


def test_croston_is_zero_on_a_never_selling_item() -> None:
    assert croston(np.zeros(50), horizon=4).tolist() == [0.0] * 4


def test_forecast_panel_actually_uses_each_series_history() -> None:
    """Regression test.

    polars group_by yields tuple keys. Building the history lookup with `for sid, grp`
    instead of `for (sid,), grp` makes every lookup miss, so each series forecasts from an
    empty history and silently returns zeros. Every downstream metric then reports nan and
    the whole baseline table looks broken rather than wrong.
    """
    cfg = load_config()
    panel = make_synthetic_panel(n_series=6, n_days=600, seed=5)
    df, _ = build_features(panel, cfg)
    fold = folds_from_config(panel.get_column("date").to_list(), cfg)[0]

    preds = forecast_panel(fold.assign(df), model="seasonal_naive", horizon=28)

    assert not preds.is_empty()
    assert preds.get_column("yhat").null_count() == 0
    assert preds.get_column("yhat").sum() > 0, "all-zero forecasts mean history lookup missed"

    # each series must get its own forecast, not a shared one
    per_series = preds.group_by("series_id").agg(pl.col("yhat").mean().alias("m"))
    assert per_series.get_column("m").n_unique() > 1


def test_forecast_panel_covers_the_whole_test_window() -> None:
    cfg = load_config()
    panel = make_synthetic_panel(n_series=4, n_days=600, seed=6)
    df, _ = build_features(panel, cfg)
    fold = folds_from_config(panel.get_column("date").to_list(), cfg)[0]

    preds = forecast_panel(fold.assign(df), model="croston", horizon=28)
    counts = preds.group_by("series_id").agg(pl.len().alias("n")).get_column("n").to_list()
    assert set(counts) == {28}
