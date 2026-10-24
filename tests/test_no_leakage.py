"""The load-bearing test.

If a feature at time t depends on any observation after t, this fails. The check is a
perturbation argument rather than an inspection: corrupt every future target, rebuild the
features, and assert nothing at or before the cutoff moved.
"""

from __future__ import annotations

import polars as pl
import pytest

from cfd.config import load_config
from cfd.data.features import build_features, feature_columns
from cfd.data.synth import make_synthetic_panel


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    return make_synthetic_panel(n_series=8, n_days=400, seed=7)


def test_features_ignore_future_targets(panel: pl.DataFrame, cfg: dict) -> None:
    cutoff = panel.get_column("date").sort()[250]

    corrupted = panel.with_columns(
        pl.when(pl.col("date") > cutoff)
        .then(pl.col("y") * 1000 + 7)
        .otherwise(pl.col("y"))
        .alias("y")
    )

    clean_df, built = build_features(panel, cfg)
    dirty_df, _ = build_features(corrupted, cfg)
    cols = feature_columns(built)

    keys = ["series_id", "date"]
    clean = clean_df.filter(pl.col("date") <= cutoff).sort(keys).select(keys + cols)
    dirty = dirty_df.filter(pl.col("date") <= cutoff).sort(keys).select(keys + cols)

    assert clean.height > 0
    for col in cols:
        assert clean.get_column(col).equals(dirty.get_column(col)), (
            f"feature {col!r} changed when only future targets were perturbed "
            "-> it reads the future"
        )


def test_lag_alignment_is_exact(panel: pl.DataFrame, cfg: dict) -> None:
    df, _ = build_features(panel, cfg)
    one = df.filter(pl.col("series_id") == df.get_column("series_id")[0]).sort("date")

    y = one.get_column("y").to_list()
    for lag in cfg["features"]["lags"]:
        got = one.get_column(f"y_lag_{lag}").to_list()
        assert got[lag:] == y[:-lag], f"y_lag_{lag} is misaligned"
        assert all(v is None for v in got[:lag]), f"y_lag_{lag} should be null at the head"


def test_rolling_windows_exclude_the_current_row(panel: pl.DataFrame, cfg: dict) -> None:
    df, _ = build_features(panel, cfg)
    one = df.filter(pl.col("series_id") == df.get_column("series_id")[0]).sort("date")

    y = one.get_column("y").to_list()
    window = cfg["features"]["rolling_windows"][0]
    shift = cfg["features"]["origin_shift"]
    got = one.get_column(f"y_rmean_{window}").to_list()

    # at row i the window must cover y[i-shift-window+1 .. i-shift], never y[i]
    for i in range(shift + window, min(len(y), 120)):
        expected = sum(y[i - shift - window + 1 : i - shift + 1]) / window
        assert got[i] == pytest.approx(expected), f"y_rmean_{window} wrong at row {i}"


def test_no_feature_reaches_inside_the_forecast_window(panel: pl.DataFrame, cfg: dict) -> None:
    """Causality is not enough -- features must be known at the forecast ORIGIN.

    Corrupt the last `horizon` days and rebuild: a 28-day-ahead model must be unaffected
    at every date in that window, because at the origin none of it has been observed.
    """
    horizon = cfg["splits"]["horizon"]
    dates = panel.get_column("date").unique().sort()
    origin = dates[-horizon - 1]

    corrupted = panel.with_columns(
        pl.when(pl.col("date") > origin)
        .then(pl.col("y") * 1000 + 7)
        .otherwise(pl.col("y"))
        .alias("y")
    )

    clean_df, built = build_features(panel, cfg)
    dirty_df, _ = build_features(corrupted, cfg)
    cols = feature_columns(built)

    keys = ["series_id", "date"]
    window = pl.col("date") > origin
    clean = clean_df.filter(window).sort(keys).select(keys + cols)
    dirty = dirty_df.filter(window).sort(keys).select(keys + cols)

    assert clean.height > 0
    for col in cols:
        assert clean.get_column(col).equals(dirty.get_column(col)), (
            f"feature {col!r} moved when only the forecast window was perturbed -> it is "
            "causal but not origin-safe, so it cannot be used at this horizon"
        )


def test_lags_shorter_than_the_horizon_are_rejected(panel: pl.DataFrame, cfg: dict) -> None:
    bad = {**cfg, "features": {**cfg["features"], "lags": [1, 7, 28]}}
    with pytest.raises(ValueError, match="shorter than origin_shift"):
        build_features(panel, bad)


def test_origin_shift_must_match_the_horizon(panel: pl.DataFrame, cfg: dict) -> None:
    bad = {**cfg, "features": {**cfg["features"], "origin_shift": 7}}
    with pytest.raises(ValueError, match="must equal splits.horizon"):
        build_features(panel, bad)


def test_rolling_does_not_bleed_across_series(cfg: dict) -> None:
    small = make_synthetic_panel(n_series=3, n_days=120, seed=11)
    df, _ = build_features(small, cfg)
    window = cfg["features"]["rolling_windows"][0]

    heads = df.sort(["series_id", "date"]).group_by("series_id", maintain_order=True).head(1)
    assert heads.get_column(f"y_rmean_{window}").null_count() == heads.height, (
        "the first row of every series must have a null rolling mean; "
        "a value there means the window spilled in from the previous series"
    )
