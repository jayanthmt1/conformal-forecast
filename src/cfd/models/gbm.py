"""Global LightGBM: one point model and a set of quantile models across all series.

TWO THINGS HERE ARE LOAD-BEARING
--------------------------------
1. Early stopping never touches the calibration block. That block is reserved for
   conformal calibration, and selecting a model against it would defeat the purpose --
   the residuals it holds would no longer be the residuals of an unseen model. A slice
   off the END of train is carved out for early stopping instead.

2. Predicted quantiles are sorted per row. Independently-fit quantile models are under
   no constraint to be monotone, so the raw 5th percentile can exceed the raw 95th.
   Sorting is the standard cheap fix and it matters before any interval is formed.

Hierarchy codes enter as ordinary numeric splits rather than declared categoricals; the
cardinality is tiny here and it keeps the design matrix a single float array.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

TARGET = "y"


def design_matrix(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Float matrix with polars nulls as NaN, which LightGBM handles natively."""
    return df.select([pl.col(c).cast(pl.Float64) for c in feature_cols]).to_numpy()


@dataclass(frozen=True)
class FitSplit:
    """Train, minus a tail slice held out for early stopping."""

    fit: pl.DataFrame
    val: pl.DataFrame | None

    @property
    def n_fit(self) -> int:
        return self.fit.height

    @property
    def n_val(self) -> int:
        return 0 if self.val is None else self.val.height


def carve_validation(train_df: pl.DataFrame, val_len: int) -> FitSplit:
    """Hold out the last `val_len` days of train for early stopping.

    Deliberately NOT the calibration block -- see the module docstring.
    """
    if val_len <= 0:
        return FitSplit(train_df, None)

    last = train_df.get_column("date").max()
    val_start = last - timedelta(days=val_len - 1)
    fit = train_df.filter(pl.col("date") < val_start)
    val = train_df.filter(pl.col("date") >= val_start)

    if fit.is_empty() or val.is_empty():
        return FitSplit(train_df, None)
    return FitSplit(fit, val)


def _train_booster(
    split: FitSplit,
    feature_cols: list[str],
    params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> lgb.Booster:
    dtrain = lgb.Dataset(
        design_matrix(split.fit, feature_cols),
        label=split.fit.get_column(TARGET).to_numpy(),
        feature_name=list(feature_cols),
    )

    valid_sets, callbacks = [], [lgb.log_evaluation(0)]
    if split.val is not None:
        valid_sets = [
            lgb.Dataset(
                design_matrix(split.val, feature_cols),
                label=split.val.get_column(TARGET).to_numpy(),
                reference=dtrain,
                feature_name=list(feature_cols),
            )
        ]
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    return lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def _model_cfg(cfg: dict[str, Any]) -> tuple[dict[str, Any], int, int, int]:
    mcfg = cfg["model"]
    return (
        dict(mcfg["params"]),
        int(mcfg["num_boost_round"]),
        int(mcfg["early_stopping_rounds"]),
        int(mcfg["val_len"]),
    )


def train_point(
    train_df: pl.DataFrame, feature_cols: list[str], cfg: dict[str, Any]
) -> lgb.Booster:
    """Conditional-mean model -- this is what the 'order the point forecast' policy uses."""
    params, rounds, stop, val_len = _model_cfg(cfg)
    params["objective"] = cfg["model"]["point_objective"]
    params.setdefault("metric", "rmse")
    return _train_booster(carve_validation(train_df, val_len), feature_cols, params, rounds, stop)


def train_quantiles(
    train_df: pl.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
    quantiles: list[float] | None = None,
) -> dict[float, lgb.Booster]:
    """One booster per quantile level, fit with pinball loss."""
    params, rounds, stop, val_len = _model_cfg(cfg)
    taus = list(quantiles if quantiles is not None else cfg["model"]["quantiles"])
    split = carve_validation(train_df, val_len)

    models: dict[float, lgb.Booster] = {}
    for tau in taus:
        p = dict(params)
        p.update(objective="quantile", alpha=float(tau), metric="quantile")
        p.pop("tweedie_variance_power", None)
        models[float(tau)] = _train_booster(split, feature_cols, p, rounds, stop)
    return models


def predict(booster: lgb.Booster, df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Point prediction, floored at zero -- demand cannot be negative."""
    raw = booster.predict(design_matrix(df, feature_cols), num_iteration=booster.best_iteration)
    return np.maximum(np.asarray(raw, dtype=float), 0.0)


def enforce_monotone(preds: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Sort predictions across quantile levels, row by row, to remove crossings."""
    taus = sorted(preds)
    stacked = np.column_stack([preds[t] for t in taus])
    stacked = np.sort(stacked, axis=1)
    return {tau: stacked[:, i] for i, tau in enumerate(taus)}


def predict_quantiles(
    models: dict[float, lgb.Booster], df: pl.DataFrame, feature_cols: list[str]
) -> dict[float, np.ndarray]:
    raw = {tau: predict(model, df, feature_cols) for tau, model in models.items()}
    return enforce_monotone(raw)


def quantile_crossing_rate(preds: dict[float, np.ndarray]) -> float:
    """Share of rows where the raw quantile predictions were not already monotone.

    Worth reporting: it is the empirical answer to 'why not just use the model's own
    quantiles as an interval?'
    """
    taus = sorted(preds)
    stacked = np.column_stack([preds[t] for t in taus])
    return float(np.mean(np.any(np.diff(stacked, axis=1) < 0, axis=1)))
