"""Feature-group ablation: drop one named group at a time, refit the point model, rescore.

This is the honest answer to "which features mattered". It is preferred here over SHAP,
which on correlated lag features of a global model invites questions about TreeSHAP's
independence assumptions that cannot be won in an interview. Writes results/ablation.parquet.
"""

from __future__ import annotations

import argparse
import time

import polars as pl

from cfd.config import load_config, resolve
from cfd.eval.scoring import aggregate, score_predictions, summarise
from cfd.eval.track import append_results
from cfd.pipeline import feature_cols_from_config, fit_predict_fold, history_for
from cfd.splits import folds_from_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pl.read_parquet(resolve(cfg["data"]["processed_path"]))
    folds = folds_from_config(df.get_column("date").to_list(), cfg)
    groups = list(cfg["features"]["groups"])

    variants = {"full": groups}
    for g in groups:
        variants[f"-{g}"] = [x for x in groups if x != g]

    scored = []
    for name, keep in variants.items():
        cols = feature_cols_from_config(cfg, keep)
        t0 = time.time()
        for fold in folds:
            preds, _ = fit_predict_fold(df, fold, cfg, cols, with_quantiles=False)
            test = preds.filter(pl.col("split") == "test")
            scored.append(score_predictions(test, history_for(df, fold), fold.idx, name))
        print(f"{name:<12} {len(cols):>2} features  {time.time() - t0:5.1f}s")

    agg = aggregate(pl.concat(scored))
    summary = summarise(agg)
    full = summary.filter(pl.col("model") == "full").get_column("wrmsse")[0]
    summary = summary.with_columns(
        (pl.col("wrmsse") - full).round(4).alias("delta_wrmsse"),
        ((pl.col("wrmsse") - full) / full).round(4).alias("delta_rel"),
    ).sort("delta_wrmsse", descending=True)

    print("\nleave-one-group-out (positive delta = the group was helping):")
    print(summary)

    res_dir = resolve("results")
    agg.write_parquet(res_dir / "ablation.parquet")
    append_results(agg, cfg, cfg["eval"]["results_path"], extra={"stage": "ablation"})
    print(f"\nwrote {res_dir / 'ablation.parquet'}")


if __name__ == "__main__":
    main()
