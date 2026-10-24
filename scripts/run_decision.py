"""Stage [5]: newsvendor cost of point vs quantile vs conformal ordering, swept over p/h.

The headline number of the project comes out of here: the relative cost reduction of the
conformal policy over the point policy at the headline cost ratio, with a paired
bootstrap CI across walk-forward folds (and, as a finer-grained check, across series).

Writes results/decision.parquet and results/decision_series.parquet.
"""

from __future__ import annotations

import argparse

import polars as pl

from cfd.config import load_config, resolve
from cfd.decision import relative_to_point, simulate_fold
from cfd.eval.metrics import paired_bootstrap_ci
from cfd.eval.track import append_results

HEADLINE_RATIO = 4.0


def ci_rows(
    folds: pl.DataFrame, series: pl.DataFrame, ratios: list[float], n_boot: int, seed: int
) -> pl.DataFrame:
    rows = []
    for ratio in ratios:
        for policy in ("quantile", "conformal"):
            for unit, frame, keys in (
                ("fold", folds, ["fold"]),
                ("series", series, ["fold", "series_id"]),
            ):
                sub = frame.filter(pl.col("cost_ratio") == ratio)
                a = sub.filter(pl.col("policy") == policy).sort(keys).get_column("cost").to_numpy()
                b = sub.filter(pl.col("policy") == "point").sort(keys).get_column("cost").to_numpy()
                ci = paired_bootstrap_ci(a, b, n_boot=n_boot, seed=seed, relative=True)
                rows.append(
                    {
                        "cost_ratio": ratio,
                        "policy": policy,
                        "vs": "point",
                        "unit": unit,
                        "reduction": ci["point"],
                        "lo": ci["lo"],
                        "hi": ci["hi"],
                        "n_pairs": int(ci["n_pairs"]),
                    }
                )
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    preds_path = resolve(cfg["eval"]["preds_path"])
    if not preds_path.exists():
        raise SystemExit(f"{preds_path} not found -- run scripts/run_gbm.py first")
    preds = pl.read_parquet(preds_path)

    ratios = [float(r) for r in cfg["decision"]["cost_ratios"]]
    h = float(cfg["decision"]["holding_cost"])

    fold_rows, series_rows = [], []
    for (_,), grp in preds.group_by("fold", maintain_order=True):
        calib = grp.filter(pl.col("split") == "calib")
        test = grp.filter(pl.col("split") == "test").sort(["series_id", "date"])
        f, s = simulate_fold(calib, test, ratios, h)
        fold_rows += f
        series_rows += s

    folds = relative_to_point(pl.DataFrame(fold_rows))
    series = pl.DataFrame(series_rows)
    cis = ci_rows(folds, series, ratios, cfg["eval"]["n_boot"], cfg["eval"]["seed"])

    res_dir = resolve("results")
    folds.write_parquet(res_dir / "decision.parquet")
    series.write_parquet(res_dir / "decision_series.parquet")
    cis.write_parquet(res_dir / "decision_ci.parquet")

    print("cost relative to the point policy (mean over folds; 1.00 = same cost):")
    print(
        folds.pivot(on="policy", index="cost_ratio", values="cost_vs_point", aggregate_function="mean")
        .select("cost_ratio", "point", "quantile", "conformal")
        .with_columns(pl.col("quantile", "conformal").round(3))
        .sort("cost_ratio")
    )

    print("\nrealised service level (share of series-days with no stockout) vs target tau:")
    print(
        folds.pivot(on="policy", index=["cost_ratio", "tau"], values="service_level", aggregate_function="mean")
        .with_columns(pl.col("tau", "point", "quantile", "conformal").round(3))
        .sort("cost_ratio")
    )

    headline = cis.filter(pl.col("cost_ratio") == HEADLINE_RATIO)
    print(f"\npaired bootstrap 95% CI for cost reduction vs point at p/h = {HEADLINE_RATIO:g}:")
    for r in headline.iter_rows(named=True):
        print(
            f"  {r['policy']:<10} over {r['unit']:<6} (n={r['n_pairs']:>4}): "
            f"{r['reduction']:+.1%}  [{r['lo']:+.1%}, {r['hi']:+.1%}]"
        )

    append_results(folds, cfg, cfg["eval"]["results_path"], extra={"stage": "decision"})
    print(f"\nwrote {res_dir / 'decision.parquet'}, decision_series.parquet, decision_ci.parquet")


if __name__ == "__main__":
    main()
