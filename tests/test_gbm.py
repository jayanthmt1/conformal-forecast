"""The two load-bearing properties of models/gbm.py, plus the fold pipeline contract."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from cfd.config import load_config
from cfd.data.features import build_features
from cfd.data.synth import make_synthetic_panel
from cfd.models.gbm import carve_validation, enforce_monotone, quantile_crossing_rate
from cfd.pipeline import feature_cols_from_config, fit_predict_fold, quantile_cols
from cfd.splits import folds_from_config


def test_enforce_monotone_sorts_every_row() -> None:
    preds = {0.1: np.array([5.0, 1.0]), 0.5: np.array([2.0, 2.0]), 0.9: np.array([1.0, 3.0])}
    fixed = enforce_monotone(preds)
    assert fixed[0.1].tolist() == [1.0, 1.0]
    assert fixed[0.5].tolist() == [2.0, 2.0]
    assert fixed[0.9].tolist() == [5.0, 3.0]


def test_crossing_rate_counts_rows_that_needed_sorting() -> None:
    preds = {0.1: np.array([5.0, 1.0, 0.0]), 0.9: np.array([1.0, 3.0, 0.0])}
    assert quantile_crossing_rate(preds) == pytest.approx(1 / 3)


def test_carve_validation_takes_the_tail_and_never_overlaps() -> None:
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(100)]
    df = pl.DataFrame({"date": days, "y": range(100)}).with_columns(pl.col("date").cast(pl.Date))
    split = carve_validation(df, val_len=10)
    assert split.n_val == 10
    assert split.fit.get_column("date").max() < split.val.get_column("date").min()
    assert split.n_fit + split.n_val == 100


def test_carve_validation_is_a_no_op_at_zero() -> None:
    df = pl.DataFrame({"date": [date(2020, 1, 1)], "y": [1]})
    assert carve_validation(df, 0).val is None


@pytest.fixture(scope="module")
def small_fold_preds():
    cfg = load_config()
    cfg["model"] = {**cfg["model"], "num_boost_round": 20, "early_stopping_rounds": 5}
    cfg["model"]["quantiles"] = [0.1, 0.5, 0.9]
    panel = make_synthetic_panel(n_series=9, n_days=600, seed=1)
    df, _ = build_features(panel, cfg)
    folds = folds_from_config(df.get_column("date").to_list(), cfg)
    preds, diag = fit_predict_fold(df, folds[-1], cfg, feature_cols_from_config(cfg))
    return cfg, folds[-1], preds, diag


def test_fold_predictions_cover_calib_and_test_only(small_fold_preds) -> None:
    cfg, fold, preds, _ = small_fold_preds
    assert set(preds.get_column("split").unique()) == {"calib", "test"}
    assert preds.get_column("date").min() == fold.calib_start
    assert preds.get_column("date").max() == fold.test_end
    n_series = preds.get_column("series_id").n_unique()
    assert preds.height == n_series * (cfg["splits"]["calib_len"] + cfg["splits"]["horizon"])


def test_fold_quantiles_are_monotone_and_nonnegative(small_fold_preds) -> None:
    _, _, preds, _ = small_fold_preds
    cols = list(quantile_cols(preds).values())
    assert cols == ["q_0.1", "q_0.5", "q_0.9"]
    mat = preds.select(cols).to_numpy()
    assert (np.diff(mat, axis=1) >= 0).all()
    assert (mat >= 0).all()
    assert (preds.get_column("yhat").to_numpy() >= 0).all()


def test_fold_segments_are_attached(small_fold_preds) -> None:
    _, _, preds, _ = small_fold_preds
    assert preds.get_column("demand_seg").null_count() == 0
    assert preds.get_column("volume_seg").null_count() == 0
