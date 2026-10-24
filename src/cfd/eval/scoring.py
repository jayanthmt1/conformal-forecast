"""Per-series scoring shared by every model script, so the ladder is comparable.

Baselines, the GBM, and the ablation all route through here: same metrics, same segment
labels, same handling of unscorable series. A model that looks better because it was
scored differently is the easiest way to fool yourself.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from cfd.data.features import add_segments
from cfd.eval.metrics import mase, rmsse, wrmsse


def _nullable(value: float) -> float | None:
    """nan -> None, so polars treats an unscorable series as null rather than poisoning
    every aggregate it touches."""
    return None if not np.isfinite(value) else float(value)


def score_predictions(
    preds: pl.DataFrame,
    history: pl.DataFrame,
    fold_idx: int,
    model: str,
    season: int = 7,
) -> pl.DataFrame:
    """Score one model on one fold. `preds` needs date/series_id/y/yhat."""
    if preds.is_empty():
        return pl.DataFrame()

    # group_by yields tuple keys -- unpack, or every lookup silently misses
    hist_map = {
        sid: grp.get_column("y").to_numpy()
        for (sid,), grp in history.sort(["series_id", "date"]).group_by(
            "series_id", maintain_order=True
        )
    }
    segs = add_segments(preds.select("series_id").unique(), history.select("series_id", "y"))
    seg_map = {row["series_id"]: row for row in segs.iter_rows(named=True)}

    rows = []
    for (sid,), grp in preds.group_by("series_id", maintain_order=True):
        y = grp.get_column("y").to_numpy()
        yhat = grp.get_column("yhat").to_numpy()
        train_y = hist_map[sid]
        seg = seg_map.get(sid, {})
        rows.append(
            {
                "fold": fold_idx,
                "model": model,
                "series_id": sid,
                "rmsse": _nullable(rmsse(y, yhat, train_y)),
                "mase": _nullable(mase(y, yhat, train_y, season=season)),
                "weight": float(train_y.sum()),
                "volume_seg": seg.get("volume_seg"),
                "demand_seg": seg.get("demand_seg"),
            }
        )
    return pl.DataFrame(rows)


def aggregate(per_series: pl.DataFrame) -> pl.DataFrame:
    """model x fold -> weighted RMSSE and mean MASE."""
    rows = []
    for (model, fold), grp in per_series.group_by(["model", "fold"], maintain_order=True):
        rows.append(
            {
                "model": model,
                "fold": fold,
                "wrmsse": wrmsse(
                    grp.get_column("rmsse").to_numpy(),
                    grp.get_column("weight").to_numpy(),
                ),
                "mase": grp.get_column("mase").mean(),
                "n_series": grp.height,
                "n_unscorable": int(grp.get_column("rmsse").null_count()),
            }
        )
    return pl.DataFrame(rows).sort("model", "fold")


def summarise(agg: pl.DataFrame) -> pl.DataFrame:
    """Across folds: mean and sd of each metric."""
    return (
        agg.group_by("model")
        .agg(
            pl.col("wrmsse").mean().round(4).alias("wrmsse"),
            pl.col("wrmsse").std().round(4).alias("wrmsse_sd"),
            pl.col("mase").mean().round(4).alias("mase"),
            pl.col("mase").std().round(4).alias("mase_sd"),
            pl.col("n_unscorable").sum().alias("unscorable"),
        )
        .sort("wrmsse")
    )
