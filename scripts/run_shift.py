"""Controlled degradation experiment: does adaptive calibration earn its keep?

Each scenario perturbs realised demand in the TEST window from `onset_day` on, leaving
every forecast untouched. That isolates "the world moved" from "the model changed".
Static one-sided CQR [0, q + qhat] (calibrated once on the calib block) is compared with ACI
(the same construction, re-levelled daily from realised coverage) on coverage, width, and
newsvendor cost.

The expected shape of the result, which is the point of running it:
    no shift      ACI ~ CQR            adaptation is overhead with nothing to adapt to
    level shift   CQR under-covers,    ACI recovers the frequency by widening
    variance      CQR under-covers,    ACI recovers, at a width cost

Writes results/shift.parquet (summary) and results/shift_daily.parquet (traces).
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from cfd.config import load_config, resolve
from cfd.conformal import adaptive
from cfd.decision.newsvendor import critical_fractile, interpolate_quantile, newsvendor_cost
from cfd.decision.simulate import quantile_matrix
from cfd.pipeline import quantile_col

HEADLINE_RATIO = 4.0


def apply_shift(y: np.ndarray, day: np.ndarray, onset: int, spec: dict, rng) -> np.ndarray:
    """Perturb realised demand on test days >= onset. Demand stays a non-negative integer."""
    out = y.astype(float).copy()
    m = day >= onset
    if "multiply" in spec:
        out[m] *= float(spec["multiply"])
    if "lognormal_sigma" in spec:
        out[m] *= rng.lognormal(0.0, float(spec["lognormal_sigma"]), m.sum())
    return np.round(out)


def by_day(values: np.ndarray, day: np.ndarray) -> list[np.ndarray]:
    return [values[day == d] for d in np.unique(day)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    preds = pl.read_parquet(resolve(cfg["eval"]["preds_path"]))
    level = round(1.0 - cfg["conformal"]["alpha"], 6)
    alpha = cfg["conformal"]["alpha"]
    gamma = cfg["conformal"]["aci"]["gamma"]
    lag = cfg["conformal"]["aci"]["feedback_lag"]
    onset = int(cfg["shift"]["onset_day"])
    scenarios = cfg["shift"]["scenarios"]
    h = float(cfg["decision"]["holding_cost"])
    p = HEADLINE_RATIO * h
    tau = critical_fractile(p, h)

    up_col = quantile_col(round(level, 6))
    rng = np.random.default_rng(cfg["eval"]["seed"])

    rows, traces = [], []
    for (fold,), grp in preds.group_by("fold", maintain_order=True):
        calib = grp.filter(pl.col("split") == "calib")
        test = grp.filter(pl.col("split") == "test").sort(["date", "series_id"])

        # test-day index, 1-based, aligned with the sorted test rows
        dates = test.get_column("date").cast(pl.Int32).to_numpy()
        day = dates - dates.min() + 1
        y_true = test.get_column("y").to_numpy()
        yhat = test.get_column("yhat").to_numpy()

        # one-sided CQR [0, q_{level} + qhat]: on zero-inflated counts the central interval's
        # lower edge is pinned at 0 and its score distribution has an atom there, which leaves
        # ACI with no lever. The one-sided construction is also what the ordering decision uses.
        y_cal = calib.get_column("y").to_numpy()
        cal_scores = y_cal - calib.get_column(up_col).to_numpy()
        interval_fn = adaptive.upper_bound_fn(cal_scores, by_day(test.get_column(up_col).to_numpy(), day))

        # one-sided pieces for the ordering decision at the headline fractile
        taus, q_cal = quantile_matrix(calib)
        _, q_test = quantile_matrix(test)
        q_tau_cal = interpolate_quantile(taus, q_cal, tau)
        q_tau_test = interpolate_quantile(taus, q_test, tau)
        order_scores = y_cal - q_tau_cal
        order_fn = adaptive.upper_bound_fn(order_scores, by_day(q_tau_test, day))

        for scenario, spec in scenarios.items():
            y = apply_shift(y_true, day, onset, spec, rng)
            y_days = by_day(y, day)

            for method, g in (("cqr_static", 0.0), ("aci", gamma)):
                trace = adaptive.run_aci(y_days, interval_fn, alpha, g, feedback_lag=lag)
                order_trace = adaptive.run_aci(
                    y_days, order_fn, 1.0 - tau, g, feedback_lag=lag
                )
                order = np.concatenate(order_trace.upper)
                holding, stockout = newsvendor_cost(y, order, p, h)
                post = np.array([d >= onset for d in np.unique(day)])

                rows.append(
                    {
                        "fold": fold,
                        "scenario": scenario,
                        "method": method,
                        "coverage": float(np.mean(trace.coverage)),
                        "coverage_post_onset": float(np.mean(np.array(trace.coverage)[post])),
                        "width": float(np.mean(trace.width)),
                        "width_post_onset": float(np.mean(np.array(trace.width)[post])),
                        "cost": float((holding + stockout).sum()),
                        "service_level": float(np.mean(y <= order)),
                        "final_alpha": trace.alpha_t[-1],
                    }
                )
                traces += [
                    {
                        "fold": fold,
                        "scenario": scenario,
                        "method": method,
                        "test_day": t + 1,
                        "alpha_t": trace.alpha_t[t],
                        "coverage": trace.coverage[t],
                        "width": trace.width[t],
                    }
                    for t in range(len(y_days))
                ]

            # the point policy under the same shifted demand, for reference
            holding, stockout = newsvendor_cost(y, yhat, p, h)
            rows.append(
                {
                    "fold": fold,
                    "scenario": scenario,
                    "method": "point_policy",
                    "coverage": None,
                    "coverage_post_onset": None,
                    "width": None,
                    "width_post_onset": None,
                    "cost": float((holding + stockout).sum()),
                    "service_level": float(np.mean(y <= yhat)),
                    "final_alpha": None,
                }
            )

    summary = pl.DataFrame(rows)
    daily = pl.DataFrame(traces)
    res_dir = resolve("results")
    summary.write_parquet(res_dir / "shift.parquet")
    daily.write_parquet(res_dir / "shift_daily.parquet")

    print(f"nominal {level}, ACI gamma={gamma}, feedback lag={lag} day(s), shift from test day {onset}\n")
    print("mean over folds; coverage/width are post-onset, cost is at p/h = 4 over the whole window:")
    print(
        summary.group_by("scenario", "method")
        .agg(
            pl.col("coverage_post_onset").mean().round(3).alias("coverage"),
            pl.col("width_post_onset").mean().round(2).alias("width"),
            pl.col("cost").mean().round(0).alias("cost"),
            pl.col("service_level").mean().round(3).alias("service"),
        )
        .sort("scenario", "method")
    )
    print(f"\nwrote {res_dir / 'shift.parquet'}, shift_daily.parquet")


if __name__ == "__main__":
    main()
