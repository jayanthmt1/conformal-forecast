"""Rungs 0-1 of the baseline ladder, evaluated across every walk-forward fold.

Produces the row that every later model must beat. Writes to results/results.parquet.
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from cfd.config import load_config, resolve
from cfd.data.features import add_segments
from cfd.eval.metrics import mase, rmsse, wrmsse
from cfd.eval.track import append_results
from cfd.models.baselines import BASELINES, forecast_panel
from cfd.splits import Fold, folds_from_config


def score_fold(df: pl.DataFrame, fold: Fold, model: str, horizon: int) -> list[dict]:
    labelled = fold.assign(df)
    preds = forecast_panel(labelled, model=model, horizon=horizon)
    if preds.is_empty():
        return []

    history = labelled.filter(pl.col("split").is_in(["train", "calib"]))
    hist_map = {
        sid: grp.get_column("y").to_numpy()
        for (sid,), grp in history.sort(["series_id", "date"]).group_by(
            "series_id", maintain_order=True
        )
    }

    segs = add_segments(
        preds.select("series_id").unique(),
        history.select("series_id", "y"),
    )
    seg_map = {r["series_id"]: r for r in segs.iter_rows(named=True)}

    def nullable(value: float) -> float | None:
        """nan -> None so polars treats unscorable series as null, not as a poisoned mean."""
        return None if not np.isfinite(value) else float(value)

    rows = []
    for (sid,), grp in preds.group_by("series_id", maintain_order=True):
        y = grp.get_column("y").to_numpy()
        yhat = grp.get_column("yhat").to_numpy()
        train_y = hist_map[sid]
        seg = seg_map.get(sid, {})
        rows.append(
            {
                "fold": fold.idx,
                "model": model,
                "series_id": sid,
                "rmsse": nullable(rmsse(y, yhat, train_y)),
                "mase": nullable(mase(y, yhat, train_y, season=7)),
                "weight": float(train_y.sum()),
                "volume_seg": seg.get("volume_seg"),
                "demand_seg": seg.get("demand_seg"),
            }
        )
    return rows


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

    rows: list[dict] = []
    for model in BASELINES:
        for fold in folds:
            rows += score_fold(df, fold, model, horizon)

    per_series = pl.DataFrame(rows)

    summary = (
        per_series.group_by("model", "fold")
        .agg(
            pl.struct("rmsse", "weight")
            .map_batches(
                lambda s: pl.Series(
                    [
                        wrmsse(
                            np.array(s.struct.field("rmsse").to_list()),
                            np.array(s.struct.field("weight").to_list()),
                        )
                    ]
                ),
                returns_scalar=True,
            )
            .alias("wrmsse"),
            pl.col("mase").mean().alias("mase"),
            pl.len().alias("n_series"),
        )
        .sort("model", "fold")
    )

    print("\nper-fold:")
    print(summary)

    print("\nacross folds (mean +/- sd):")
    print(
        summary.group_by("model")
        .agg(
            pl.col("wrmsse").mean().round(4).alias("wrmsse_mean"),
            pl.col("wrmsse").std().round(4).alias("wrmsse_sd"),
            pl.col("mase").mean().round(4).alias("mase_mean"),
            pl.col("mase").std().round(4).alias("mase_sd"),
        )
        .sort("wrmsse_mean")
    )

    print("\nmase by demand segment:")
    print(
        per_series.group_by("model", "demand_seg")
        .agg(pl.col("mase").mean().round(3).alias("mase"), pl.len().alias("n"))
        .sort("model", "demand_seg")
    )

    out = append_results(summary, cfg, cfg["eval"]["results_path"], extra={"stage": "baselines"})
    print(f"\nappended to {out}")


if __name__ == "__main__":
    main()
