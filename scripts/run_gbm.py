"""Rungs 2-3 of the ladder: global LightGBM point model and multi-quantile model.

Scores the point model on the same per-series metrics as the baselines, reports the
quantile model's pinball loss, and -- the part everything downstream depends on -- writes
calib and test predictions for every fold to results/preds/gbm.parquet.
"""

from __future__ import annotations

import argparse
import time

import polars as pl

from cfd.config import load_config, resolve
from cfd.eval.scoring import aggregate, score_predictions, summarise
from cfd.eval.track import append_results
from cfd.pipeline import (
    feature_cols_from_config,
    fit_predict_fold,
    history_for,
    pinball_table,
)
from cfd.splits import folds_from_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = resolve(cfg["data"]["processed_path"])
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/prepare_data.py first")

    df = pl.read_parquet(path)
    feature_cols = feature_cols_from_config(cfg)
    folds = folds_from_config(df.get_column("date").to_list(), cfg)
    print(f"{len(feature_cols)} features, {len(folds)} folds, {df.height:,} rows")

    frames, scored, diags = [], [], []
    for fold in folds:
        t0 = time.time()
        preds, diag = fit_predict_fold(df, fold, cfg, feature_cols)
        frames.append(preds)
        diags.append(diag)

        test = preds.filter(pl.col("split") == "test")
        scored.append(score_predictions(test, history_for(df, fold), fold.idx, "lgbm_point"))
        print(
            f"fold {fold.idx}: {time.time() - t0:5.1f}s  "
            f"best_iter={diag['point_best_iter']}  "
            f"raw quantile crossing={diag['quantile_crossing_rate']:.1%}"
        )

    all_preds = pl.concat(frames)
    out = resolve(cfg["eval"]["preds_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    all_preds.write_parquet(out)
    print(f"\nwrote {out}  ({all_preds.height:,} rows)")

    per_series = pl.concat(scored)
    agg = aggregate(per_series)
    print("\nper-fold:")
    print(agg)
    print("\nacross folds:")
    print(summarise(agg))

    pin = pinball_table(all_preds)
    print("\nmean pinball loss by tau (test, averaged over folds):")
    print(pin.group_by("tau").agg(pl.col("pinball").mean().round(4)).sort("tau"))

    append_results(agg, cfg, cfg["eval"]["results_path"], extra={"stage": "gbm"})
    append_results(
        pin.with_columns(pl.lit("lgbm_quantile").alias("model")),
        cfg,
        cfg["eval"]["results_path"],
        extra={"stage": "gbm_pinball"},
    )
    append_results(pl.DataFrame(diags), cfg, cfg["eval"]["results_path"], extra={"stage": "gbm_diag"})


if __name__ == "__main__":
    main()
