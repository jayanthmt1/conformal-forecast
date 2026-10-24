"""Stage [4]: calibrate intervals on each fold's calib block, measure them on its test block.

Three interval methods at every nominal level in configs/base.yaml:
    raw_quantile     the quantile model's own (a/2, 1-a/2) predictions, uncalibrated
    split_conformal  symmetric band around the point forecast
    cqr              conformalized quantile regression

Writes
    results/conformal.parquet         coverage + width by fold x method x level x segment
    results/conformal_daily.parquet   per-test-day coverage at the headline level
    results/preds/intervals.parquet   row-level intervals at the headline level
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import polars as pl

from cfd.config import load_config, resolve
from cfd.conformal import cqr, split
from cfd.eval.metrics import coverage, mean_width
from cfd.eval.track import append_results
from cfd.pipeline import quantile_col

SEGMENTS = ("demand_seg", "volume_seg")
EPOCH = date(1970, 1, 1)


def intervals_for_fold(
    calib: pl.DataFrame, test: pl.DataFrame, level: float
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """{method: (lower, upper, qhat)} on the test rows at one nominal level."""
    alpha = 1.0 - level
    lo_tau, hi_tau = cqr.nominal_to_taus(level)
    lo_col, hi_col = quantile_col(lo_tau), quantile_col(hi_tau)
    if lo_col not in test.columns or hi_col not in test.columns:
        raise SystemExit(
            f"nominal level {level} needs quantiles {lo_tau} and {hi_tau} in model.quantiles"
        )

    y_cal = calib.get_column("y").to_numpy()
    out = {}

    qlo, qhi = test.get_column(lo_col).to_numpy(), test.get_column(hi_col).to_numpy()
    out["raw_quantile"] = (qlo, qhi, 0.0)

    s_hat = split.calibrate(y_cal, calib.get_column("yhat").to_numpy(), alpha)
    out["split_conformal"] = (*split.interval(test.get_column("yhat").to_numpy(), s_hat), s_hat)

    c_hat = cqr.calibrate(
        y_cal, calib.get_column(lo_col).to_numpy(), calib.get_column(hi_col).to_numpy(), alpha
    )
    out["cqr"] = (*cqr.interval(qlo, qhi, c_hat), c_hat)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    preds_path = resolve(cfg["eval"]["preds_path"])
    if not preds_path.exists():
        raise SystemExit(f"{preds_path} not found -- run scripts/run_gbm.py first")
    preds = pl.read_parquet(preds_path)

    levels = cfg["conformal"]["nominal_levels"]
    headline = round(1.0 - cfg["conformal"]["alpha"], 6)
    if headline not in levels:
        raise SystemExit(f"1 - conformal.alpha = {headline} must be among nominal_levels")

    rows, daily, row_level = [], [], []
    for (fold,), grp in preds.group_by("fold", maintain_order=True):
        calib = grp.filter(pl.col("split") == "calib")
        test = grp.filter(pl.col("split") == "test").sort(["series_id", "date"])
        y = test.get_column("y").to_numpy()

        for level in levels:
            for method, (lower, upper, qhat) in intervals_for_fold(calib, test, level).items():
                base = {"fold": fold, "method": method, "nominal": level, "qhat": qhat}
                rows.append(
                    {
                        **base,
                        "segment": "all",
                        "segment_value": "all",
                        "coverage": coverage(y, lower, upper),
                        "width": mean_width(lower, upper),
                        "n": y.size,
                    }
                )
                for seg in SEGMENTS:
                    labels = test.get_column(seg).to_numpy()
                    for value in np.unique(labels):
                        m = labels == value
                        rows.append(
                            {
                                **base,
                                "segment": seg,
                                "segment_value": str(value),
                                "coverage": coverage(y[m], lower[m], upper[m]),
                                "width": mean_width(lower[m], upper[m]),
                                "n": int(m.sum()),
                            }
                        )

                if level == headline:
                    days = test.get_column("date").cast(pl.Int32).to_numpy()  # days since epoch
                    for i, d in enumerate(np.unique(days)):
                        m = days == d
                        daily.append(
                            {
                                "fold": fold,
                                "method": method,
                                "test_day": i + 1,
                                "date": EPOCH + timedelta(days=int(d)),
                                "coverage": coverage(y[m], lower[m], upper[m]),
                                "width": mean_width(lower[m], upper[m]),
                            }
                        )
                    row_level.append(
                        test.select("fold", "date", "series_id", "y", "yhat", "demand_seg").with_columns(
                            pl.lit(method).alias("method"),
                            pl.Series("lower", lower),
                            pl.Series("upper", upper),
                        )
                    )

    result = pl.DataFrame(rows)
    daily_df = pl.DataFrame(daily)
    intervals = pl.concat(row_level)

    res_dir = resolve("results")
    result.write_parquet(res_dir / "conformal.parquet")
    daily_df.write_parquet(res_dir / "conformal_daily.parquet")
    intervals.write_parquet(res_dir / "preds" / "intervals.parquet")

    marginal = result.filter(pl.col("segment") == "all")
    print("calibration curve -- empirical coverage vs nominal (mean over folds):")
    print(
        marginal.group_by("method", "nominal")
        .agg(
            pl.col("coverage").mean().round(3).alias("coverage"),
            pl.col("coverage").std().round(3).alias("coverage_sd"),
            pl.col("width").mean().round(2).alias("width"),
        )
        .sort("nominal", "method")
    )

    print(f"\nsegment-conditional coverage at nominal {headline}:")
    print(
        result.filter((pl.col("nominal") == headline) & (pl.col("segment") != "all"))
        .group_by("segment", "segment_value", "method")
        .agg(
            pl.col("coverage").mean().round(3).alias("coverage"),
            pl.col("width").mean().round(2).alias("width"),
            pl.col("n").sum().alias("n"),
        )
        .sort("segment", "segment_value", "method")
    )

    append_results(marginal, cfg, cfg["eval"]["results_path"], extra={"stage": "conformal"})
    print(f"\nwrote {res_dir / 'conformal.parquet'}, conformal_daily.parquet, preds/intervals.parquet")


if __name__ == "__main__":
    main()
