"""One walk-forward fold, end to end: fit the global GBM on train, predict calib and test.

Every later stage (conformal calibration, the decision layer, the shift experiment) reads
the prediction frame this module produces instead of retraining. That is what makes those
stages cheap enough to iterate on, and it guarantees they all score the same forecasts.

PREDICTION FRAME
----------------
One row per (fold, split, series_id, date) for split in {calib, test}, with:
    y            realised demand
    yhat         conditional-mean forecast (tweedie point model)
    q_<tau>      quantile forecasts, monotone across tau after enforce_monotone
    volume_seg   series segment labels, derived from TRAIN rows of that fold only
    demand_seg
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cfd.data.features import add_segments, feature_columns, feature_groups_from_config
from cfd.models.gbm import (
    predict,
    predict_quantiles,
    quantile_crossing_rate,
    train_point,
    train_quantiles,
)
from cfd.splits import Fold

PRED_SPLITS = ("calib", "test")


def quantile_col(tau: float) -> str:
    return f"q_{tau:g}"


def quantile_cols(df: pl.DataFrame) -> dict[float, str]:
    """{tau: column} for every quantile column present, sorted by tau."""
    found = {float(c[2:]): c for c in df.columns if c.startswith("q_")}
    return dict(sorted(found.items()))


def feature_cols_from_config(cfg: dict[str, Any], groups: list[str] | None = None) -> list[str]:
    return feature_columns(feature_groups_from_config(cfg, groups))


def fit_predict_fold(
    df: pl.DataFrame,
    fold: Fold,
    cfg: dict[str, Any],
    feature_cols: list[str],
    with_quantiles: bool = True,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Fit on the fold's train rows; return predictions for its calib and test rows.

    Returns (prediction frame, diagnostics). Diagnostics carry the booster iteration counts
    and the raw quantile-crossing rate -- both worth reporting, neither worth a database.
    """
    labelled = fold.assign(df)
    train = labelled.filter(pl.col("split") == "train")
    held = labelled.filter(pl.col("split").is_in(PRED_SPLITS)).sort(["series_id", "date"])

    point = train_point(train, feature_cols, cfg)
    out = held.select("split", "date", "series_id", "y").with_columns(
        pl.lit(fold.idx, dtype=pl.Int32).alias("fold"),
        pl.Series("yhat", predict(point, held, feature_cols), dtype=pl.Float64),
    )
    diag: dict[str, Any] = {
        "fold": fold.idx,
        "point_best_iter": int(point.best_iteration or point.current_iteration()),
    }

    if with_quantiles:
        models = train_quantiles(train, feature_cols, cfg)
        raw = {tau: predict(m, held, feature_cols) for tau, m in models.items()}
        diag["quantile_crossing_rate"] = quantile_crossing_rate(raw)
        preds = predict_quantiles(models, held, feature_cols)
        out = out.with_columns(
            [pl.Series(quantile_col(tau), preds[tau], dtype=pl.Float64) for tau in sorted(preds)]
        )

    segs = add_segments(out.select("series_id").unique(), train.select("series_id", "y"))
    out = out.join(segs.select("series_id", "volume_seg", "demand_seg"), on="series_id", how="left")
    return out.select(["fold", "split", "date", "series_id", "y", "yhat", *_rest(out)]), diag


def _rest(df: pl.DataFrame) -> list[str]:
    fixed = {"fold", "split", "date", "series_id", "y", "yhat"}
    return [c for c in df.columns if c not in fixed]


def history_for(df: pl.DataFrame, fold: Fold) -> pl.DataFrame:
    """Rows a production run would have at the test origin: train plus calib."""
    return fold.assign(df).filter(pl.col("split").is_in(["train", "calib"]))


def pinball_table(preds: pl.DataFrame, split: str = "test") -> pl.DataFrame:
    """Mean pinball loss per (fold, tau) -- the quantile model's own accuracy metric."""
    sub = preds.filter(pl.col("split") == split)
    y = sub.get_column("y").to_numpy()
    rows = []
    for (fold,), grp in sub.group_by("fold", maintain_order=True):
        y = grp.get_column("y").to_numpy()
        for tau, col in quantile_cols(grp).items():
            q = grp.get_column(col).to_numpy()
            diff = y - q
            rows.append(
                {
                    "fold": fold,
                    "tau": tau,
                    "pinball": float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff))),
                }
            )
    return pl.DataFrame(rows).sort("fold", "tau")
