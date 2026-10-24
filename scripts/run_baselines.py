"""Rungs 0-1 of the baseline ladder, evaluated across every walk-forward fold.

Produces the row that every later model must beat. Writes to results/results.parquet.
Scoring goes through cfd.eval.scoring so the GBM rows are directly comparable.
"""

from __future__ import annotations

import argparse

import polars as pl

from cfd.config import load_config, resolve
from cfd.eval.scoring import aggregate, score_predictions, summarise
from cfd.eval.track import append_results
from cfd.models.baselines import BASELINES, forecast_panel
from cfd.pipeline import history_for
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
    horizon = cfg["splits"]["horizon"]
    folds = folds_from_config(df.get_column("date").to_list(), cfg)

    scored = []
    for model in BASELINES:
        for fold in folds:
            preds = forecast_panel(fold.assign(df), model=model, horizon=horizon)
            scored.append(score_predictions(preds, history_for(df, fold), fold.idx, model))
    per_series = pl.concat(scored)

    agg = aggregate(per_series)
    print("\nper-fold:")
    print(agg)
    print("\nacross folds (mean +/- sd):")
    print(summarise(agg))

    print("\nmase by demand segment:")
    print(
        per_series.group_by("model", "demand_seg")
        .agg(pl.col("mase").mean().round(3).alias("mase"), pl.len().alias("n"))
        .sort("model", "demand_seg")
    )

    out = append_results(agg, cfg, cfg["eval"]["results_path"], extra={"stage": "baselines"})
    print(f"\nappended to {out}")


if __name__ == "__main__":
    main()
