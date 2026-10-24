"""Cost simulation of three ordering policies across a sweep of stockout/holding ratios.

    point       order the conditional-mean forecast         -- what everyone does
    quantile    order the raw model quantile at p/(p+h)     -- right idea, uncalibrated
    conformal   order the conformally calibrated quantile   -- one-sided CQR at p/(p+h)

The conformal policy is calibrated on the fold's calib block and evaluated on its test
block, exactly like the intervals, so the cost it reports is out-of-sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cfd.conformal import cqr
from cfd.decision.newsvendor import critical_fractile, interpolate_quantile, newsvendor_cost
from cfd.pipeline import quantile_cols

POLICIES = ("point", "quantile", "conformal")


def quantile_matrix(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = quantile_cols(df)
    taus = np.array(list(cols), dtype=float)
    return taus, df.select(list(cols.values())).to_numpy()


def order_quantities(
    calib: pl.DataFrame, test: pl.DataFrame, tau: float
) -> dict[str, np.ndarray]:
    """Order per test row under each policy at critical fractile `tau`."""
    taus, q_cal = quantile_matrix(calib)
    _, q_test = quantile_matrix(test)

    raw_test = interpolate_quantile(taus, q_test, tau)
    raw_cal = interpolate_quantile(taus, q_cal, tau)
    qhat = cqr.calibrate_upper(calib.get_column("y").to_numpy(), raw_cal, tau)

    return {
        "point": test.get_column("yhat").to_numpy(),
        "quantile": raw_test,
        "conformal": cqr.upper_bound(raw_test, qhat),
    }


def simulate_fold(
    calib: pl.DataFrame, test: pl.DataFrame, cost_ratios: list[float], h: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (fold-level rows, per-series rows) for every policy x cost ratio."""
    fold = int(test.get_column("fold")[0])
    y = test.get_column("y").to_numpy()
    series = test.get_column("series_id").to_numpy()
    segs = test.get_column("demand_seg").to_numpy()

    fold_rows, series_rows = [], []
    for ratio in cost_ratios:
        p = ratio * h
        tau = critical_fractile(p, h)
        for policy, order in order_quantities(calib, test, tau).items():
            holding, stockout = newsvendor_cost(y, order, p, h)
            total = holding + stockout
            fold_rows.append(
                {
                    "fold": fold,
                    "cost_ratio": float(ratio),
                    "tau": tau,
                    "policy": policy,
                    "cost": float(total.sum()),
                    "holding": float(holding.sum()),
                    "stockout": float(stockout.sum()),
                    "service_level": float(np.mean(y <= order)),
                    "mean_order": float(order.mean()),
                    "mean_demand": float(y.mean()),
                }
            )
            per = pl.DataFrame(
                {"series_id": series, "demand_seg": segs, "cost": total, "short": y > order}
            ).group_by("series_id", "demand_seg").agg(
                pl.col("cost").sum(), (1 - pl.col("short").mean()).alias("service_level")
            )
            series_rows += [
                {"fold": fold, "cost_ratio": float(ratio), "policy": policy, **r}
                for r in per.iter_rows(named=True)
            ]
    return fold_rows, series_rows


def relative_to_point(fold_rows: pl.DataFrame) -> pl.DataFrame:
    """Cost of each policy as a fraction of the point policy's, per fold x ratio."""
    point = fold_rows.filter(pl.col("policy") == "point").select(
        "fold", "cost_ratio", pl.col("cost").alias("point_cost")
    )
    return fold_rows.join(point, on=["fold", "cost_ratio"]).with_columns(
        (pl.col("cost") / pl.col("point_cost")).alias("cost_vs_point")
    )
