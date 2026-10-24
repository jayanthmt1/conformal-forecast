"""Stage [7]: figures, markdown tables, and the README results block -- all from results/.

Re-running this after a new data run regenerates every number the README quotes, so the
write-up can never drift from the experiments. The README block between
<!-- results:start --> and <!-- results:end --> is owned by this script.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from cfd.config import load_config, resolve

HEADLINE_RATIO = 4.0
METHOD_LABEL = {
    "raw_quantile": "raw quantile model",
    "split_conformal": "split conformal",
    "cqr": "CQR",
    "cqr_static": "CQR (static)",
    "aci": "ACI",
    "point_policy": "point policy",
}
POLICY_LABEL = {"point": "point (mean)", "quantile": "raw quantile", "conformal": "conformal"}
COLORS = {"raw_quantile": "#9aa0a6", "split_conformal": "#e0a100", "cqr": "#1a73e8",
          "cqr_static": "#9aa0a6", "aci": "#1a73e8", "point": "#5f6368",
          "quantile": "#e0a100", "conformal": "#1a73e8"}


def md_table(df: pl.DataFrame, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.iter_rows(named=True):
        cells = []
        for c in cols:
            v = row[c]
            if v is None:
                cells.append("")
            elif c in fmt:
                cells.append(format(v, fmt[c]))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- figures


def fig_calibration(conf: pl.DataFrame, out: Path) -> None:
    marg = conf.filter(pl.col("segment") == "all")
    fig, (ax, axw) = plt.subplots(1, 2, figsize=(10, 4))
    ax.plot([0.45, 1.0], [0.45, 1.0], "k--", lw=1, label="nominal = empirical")
    for method in ("raw_quantile", "split_conformal", "cqr"):
        g = (
            marg.filter(pl.col("method") == method)
            .group_by("nominal")
            .agg(pl.col("coverage").mean(), pl.col("coverage").std().alias("sd"), pl.col("width").mean())
            .sort("nominal")
        )
        x = g.get_column("nominal").to_numpy()
        ax.errorbar(x, g.get_column("coverage").to_numpy(), yerr=g.get_column("sd").to_numpy(),
                    marker="o", capsize=3, label=METHOD_LABEL[method], color=COLORS[method])
        axw.plot(x, g.get_column("width").to_numpy(), marker="o", label=METHOD_LABEL[method],
                 color=COLORS[method])
    ax.set(xlabel="nominal coverage", ylabel="empirical coverage (test, mean ± sd over folds)",
           title="Calibration curve")
    ax.legend(loc="upper left", fontsize=8)
    axw.set(xlabel="nominal coverage", ylabel="mean interval width (units)", title="Width paid for it")
    axw.legend(fontsize=8)
    for a in (ax, axw):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "calibration_curve.png", dpi=150)
    plt.close(fig)


def fig_segments(conf: pl.DataFrame, level: float, out: Path) -> None:
    seg = conf.filter((pl.col("nominal") == level) & (pl.col("segment") != "all"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, segment in zip(axes, ("demand_seg", "volume_seg"), strict=True):
        g = (
            seg.filter(pl.col("segment") == segment)
            .group_by("segment_value", "method")
            .agg(pl.col("coverage").mean(), pl.col("width").mean())
        )
        values = sorted(g.get_column("segment_value").unique().to_list())
        methods = ("raw_quantile", "split_conformal", "cqr")
        w = 0.26
        for i, method in enumerate(methods):
            sub = g.filter(pl.col("method") == method).sort("segment_value")
            lookup = {r["segment_value"]: r for r in sub.iter_rows(named=True)}
            cov = [lookup[v]["coverage"] if v in lookup else np.nan for v in values]
            wid = [lookup[v]["width"] if v in lookup else np.nan for v in values]
            xs = np.arange(len(values)) + (i - 1) * w
            bars = ax.bar(xs, cov, width=w, label=METHOD_LABEL[method], color=COLORS[method])
            for b, wi in zip(bars, wid, strict=True):
                ax.text(b.get_x() + b.get_width() / 2, 0.02, f"w={wi:.0f}", ha="center",
                        va="bottom", fontsize=7, rotation=90, color="white")
        ax.axhline(level, color="k", ls="--", lw=1)
        ax.set_xticks(range(len(values)), values)
        ax.set(title=f"coverage by {segment.replace('_seg', ' segment')} at nominal {level:.0%}",
               ylim=(0, 1.05))
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("empirical coverage (mean over folds)")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "coverage_by_segment.png", dpi=150)
    plt.close(fig)


def fig_cost_sweep(dec: pl.DataFrame, out: Path) -> None:
    fig, (ax, axs) = plt.subplots(1, 2, figsize=(11, 4))
    ax.axhline(1.0, color=COLORS["point"], ls="--", lw=1.2, label=POLICY_LABEL["point"])
    for policy in ("quantile", "conformal"):
        g = (
            dec.filter(pl.col("policy") == policy)
            .group_by("cost_ratio")
            .agg(pl.col("cost_vs_point").mean().alias("m"), pl.col("cost_vs_point").min().alias("lo"),
                 pl.col("cost_vs_point").max().alias("hi"))
            .sort("cost_ratio")
        )
        x = g.get_column("cost_ratio").to_numpy()
        ax.plot(x, g.get_column("m").to_numpy(), marker="o", label=POLICY_LABEL[policy], color=COLORS[policy])
        ax.fill_between(x, g.get_column("lo").to_numpy(), g.get_column("hi").to_numpy(), alpha=0.15,
                        color=COLORS[policy])
    ax.set(xlabel="stockout / holding cost ratio  p/h", ylabel="total cost relative to point policy",
           title="Cost of ordering the mean (band = fold min/max)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    tau = dec.group_by("cost_ratio").agg(pl.col("tau").first()).sort("cost_ratio")
    axs.plot(tau.get_column("cost_ratio").to_numpy(), tau.get_column("tau").to_numpy(), "k--", lw=1,
             label="target  p/(p+h)")
    for policy in ("point", "quantile", "conformal"):
        g = (dec.filter(pl.col("policy") == policy).group_by("cost_ratio")
             .agg(pl.col("service_level").mean()).sort("cost_ratio"))
        axs.plot(g.get_column("cost_ratio").to_numpy(), g.get_column("service_level").to_numpy(),
                 marker="o", label=POLICY_LABEL[policy], color=COLORS[policy])
    axs.set(xlabel="p/h", ylabel="realised service level\n(share of series-days with no stockout)",
            title="Does the order hit the fractile it aimed at?")
    axs.legend(fontsize=8)
    axs.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cost_sweep.png", dpi=150)
    plt.close(fig)


def fig_shift(daily: pl.DataFrame, onset: int, level: float, out: Path) -> None:
    scenarios = daily.get_column("scenario").unique().sort().to_list()
    order = [s for s in ("none", "level_up", "level_down", "variance") if s in scenarios]
    order += [s for s in scenarios if s not in order]
    fig, axes = plt.subplots(2, len(order), figsize=(3.2 * len(order), 6), sharex=True)
    for j, scenario in enumerate(order):
        for method in ("cqr_static", "aci"):
            g = (daily.filter((pl.col("scenario") == scenario) & (pl.col("method") == method))
                 .group_by("test_day").agg(pl.col("coverage").mean(), pl.col("width").mean())
                 .sort("test_day"))
            x = g.get_column("test_day").to_numpy()
            axes[0, j].plot(x, g.get_column("coverage").to_numpy(), label=METHOD_LABEL[method],
                            color=COLORS[method])
            axes[1, j].plot(x, g.get_column("width").to_numpy(), label=METHOD_LABEL[method],
                            color=COLORS[method])
        axes[0, j].axhline(level, color="k", ls="--", lw=1)
        for ax in axes[:, j]:
            if scenario != "none":
                ax.axvline(onset - 0.5, color="red", ls=":", lw=1)
            ax.grid(alpha=0.3)
        axes[0, j].set(title=scenario.replace("_", " "), ylim=(0, 1.05))
        axes[1, j].set_xlabel("test day")
    axes[0, 0].set_ylabel("daily coverage (pooled over series)")
    axes[1, 0].set_ylabel("mean interval width")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Static CQR vs ACI under injected shift (red dotted = onset)", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "shift_coverage_over_time.png", dpi=150)
    plt.close(fig)


def fig_ablation(abl: pl.DataFrame, out: Path) -> None:
    g = abl.group_by("model").agg(pl.col("wrmsse").mean(), pl.col("wrmsse").std().alias("sd"))
    full = g.filter(pl.col("model") == "full").get_column("wrmsse")[0]
    g = g.filter(pl.col("model") != "full").with_columns((pl.col("wrmsse") - full).alias("delta")).sort("delta")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(g.get_column("model").to_list(), g.get_column("delta").to_numpy(),
            xerr=g.get_column("sd").to_numpy(), color=COLORS["cqr"], alpha=0.85, capsize=3)
    ax.axvline(0, color="k", lw=1)
    ax.set(xlabel=f"change in WRMSSE when the group is removed (full model = {full:.3f})",
           title="Leave-one-group-out ablation (positive = group was helping)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------ tables


def ladder_table(results: pl.DataFrame) -> pl.DataFrame:
    rows = results.filter(pl.col("stage").is_in(["baselines", "gbm", "ablation"]))
    rows = rows.filter(~((pl.col("stage") == "ablation") & (pl.col("model") != "full")))
    rows = rows.with_columns(pl.col("model").replace({"full": "lgbm_point"}))
    # the ablation re-fits the same point model; keep the gbm stage row when both exist
    rows = rows.filter(~((pl.col("stage") == "ablation") & pl.col("model").is_in(
        results.filter(pl.col("stage") == "gbm").get_column("model").unique().to_list())))
    return (
        rows.group_by("model")
        .agg(pl.col("wrmsse").mean().alias("WRMSSE"), pl.col("wrmsse").std().alias("±"),
             pl.col("mase").mean().alias("MASE"), pl.col("mase").std().alias("± "))
        .sort("WRMSSE", descending=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--no-readme", action="store_true", help="skip rewriting the README block")
    args = ap.parse_args()

    cfg = load_config(args.config)
    level = round(1.0 - cfg["conformal"]["alpha"], 6)
    res = resolve("results")
    figs = resolve(cfg["eval"]["figures_dir"])
    tabs = resolve(cfg["eval"]["tables_dir"])
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    results = pl.read_parquet(res / "results.parquet")
    conf = pl.read_parquet(res / "conformal.parquet")
    dec = pl.read_parquet(res / "decision.parquet")
    cis = pl.read_parquet(res / "decision_ci.parquet")
    preds = pl.read_parquet(resolve(cfg["eval"]["preds_path"]))
    shift = pl.read_parquet(res / "shift.parquet") if (res / "shift.parquet").exists() else None
    shift_daily = pl.read_parquet(res / "shift_daily.parquet") if shift is not None else None
    abl = pl.read_parquet(res / "ablation.parquet") if (res / "ablation.parquet").exists() else None

    fig_calibration(conf, figs)
    fig_segments(conf, level, figs)
    fig_cost_sweep(dec, figs)
    if shift_daily is not None:
        fig_shift(shift_daily, int(cfg["shift"]["onset_day"]), level, figs)
    if abl is not None:
        fig_ablation(abl, figs)

    # ---- tables
    ladder = ladder_table(results)
    calib = (
        conf.filter(pl.col("segment") == "all").group_by("nominal", "method")
        .agg(pl.col("coverage").mean().alias("coverage"), pl.col("width").mean().alias("width"))
        .with_columns(pl.col("method").replace(METHOD_LABEL)).sort("nominal", "method")
    )
    segs = (
        conf.filter((pl.col("nominal") == level) & (pl.col("segment") == "demand_seg"))
        .group_by("segment_value", "method")
        .agg(pl.col("coverage").mean().alias("coverage"), pl.col("width").mean().alias("width"))
        .with_columns(pl.col("method").replace(METHOD_LABEL))
        .rename({"segment_value": "demand segment"}).sort("demand segment", "method")
    )
    sweep = (
        dec.pivot(on="policy", index="cost_ratio", values="cost_vs_point", aggregate_function="mean")
        .select("cost_ratio", "quantile", "conformal").rename({"cost_ratio": "p/h"}).sort("p/h")
    )
    ci_tab = cis.filter(pl.col("cost_ratio") == HEADLINE_RATIO).select(
        pl.col("policy").replace(POLICY_LABEL), "unit", "n_pairs", "reduction", "lo", "hi")

    tables = {
        "ladder.md": md_table(ladder, {"WRMSSE": ".3f", "±": ".3f", "MASE": ".3f", "± ": ".3f"}),
        "calibration.md": md_table(calib, {"coverage": ".3f", "width": ".1f"}),
        "segments.md": md_table(segs, {"coverage": ".3f", "width": ".1f"}),
        "cost_sweep.md": md_table(sweep, {"quantile": ".3f", "conformal": ".3f"}),
        "decision_ci.md": md_table(ci_tab, {"reduction": "+.1%", "lo": "+.1%", "hi": "+.1%"}),
    }
    if shift is not None:
        sh = (
            shift.group_by("scenario", "method")
            .agg(pl.col("coverage_post_onset").mean().alias("coverage (post-onset)"),
                 pl.col("width_post_onset").mean().alias("width"),
                 pl.col("cost").mean().alias("cost @ p/h=4"),
                 pl.col("service_level").mean().alias("service level"))
            .with_columns(pl.col("method").replace(METHOD_LABEL)).sort("scenario", "method")
        )
        tables["shift.md"] = md_table(sh, {"coverage (post-onset)": ".3f", "width": ".1f",
                                           "cost @ p/h=4": ",.0f", "service level": ".3f"})
    if abl is not None:
        full = abl.filter(pl.col("model") == "full").get_column("wrmsse").mean()
        at = (abl.group_by("model").agg(pl.col("wrmsse").mean().alias("WRMSSE"))
              .with_columns((pl.col("WRMSSE") - full).alias("Δ vs full"))
              .rename({"model": "variant"}).sort("Δ vs full", descending=True))
        tables["ablation.md"] = md_table(at, {"WRMSSE": ".3f", "Δ vs full": "+.3f"})
    for name, text in tables.items():
        (tabs / name).write_text(text + "\n")

    # ---- headline
    head = cis.filter((pl.col("cost_ratio") == HEADLINE_RATIO) & (pl.col("policy") == "conformal")
                      & (pl.col("unit") == "fold")).row(0, named=True)
    cov = conf.filter((pl.col("segment") == "all") & (pl.col("nominal") == level)
                      & (pl.col("method") == "cqr")).get_column("coverage").mean()
    raw_cov = conf.filter((pl.col("segment") == "all") & (pl.col("nominal") == level)
                          & (pl.col("method") == "raw_quantile")).get_column("coverage").mean()
    n_series = preds.get_column("series_id").n_unique()
    n_folds = preds.get_column("fold").n_unique()
    source = cfg["data"]["source"]
    subset = (f"store {cfg['data']['store']}, top {cfg['data']['top_n_series']} series by volume"
              if source == "m5" else f"synthetic panel, {n_series} series")

    headline = (
        f"> **Result** — on {'M5 Walmart' if source == 'm5' else 'the synthetic'} data ({subset}), "
        f"ordering the conformally calibrated quantile reduced simulated inventory cost by "
        f"**{head['reduction']:.1%} (95% paired-bootstrap CI {head['lo']:.1%} to {head['hi']:.1%})** "
        f"versus ordering the mean forecast at a stockout/holding ratio of **{HEADLINE_RATIO:g}:1**, while "
        f"CQR intervals held **{cov:.1%} empirical coverage** against a {level:.0%} nominal target "
        f"(raw quantile model: {raw_cov:.1%}), across **{n_folds} walk-forward folds** and "
        f"**{n_series} series**."
    )

    block = "\n\n".join([
        "<!-- results:start -->",
        f"_Generated by `scripts/make_report.py` from `results/`; data source: `{source}`._",
        headline,
        "### Forecast accuracy ladder\n\n" + tables["ladder.md"],
        "### Calibration\n\n![calibration curve](reports/figures/calibration_curve.png)\n\n" + tables["calibration.md"],
        f"### Coverage by demand segment (nominal {level:.0%})\n\n![coverage by segment](reports/figures/coverage_by_segment.png)\n\n" + tables["segments.md"],
        "### Decision layer: cost relative to the point policy\n\n![cost sweep](reports/figures/cost_sweep.png)\n\n" + tables["cost_sweep.md"]
        + f"\n\nPaired bootstrap 95% CIs for the cost reduction vs the point policy at p/h = {HEADLINE_RATIO:g}:\n\n" + tables["decision_ci.md"],
    ] + ([
        "### Shift experiment: static CQR vs ACI\n\n![shift](reports/figures/shift_coverage_over_time.png)\n\n" + tables["shift.md"],
    ] if shift is not None else []) + ([
        "### Feature-group ablation\n\n![ablation](reports/figures/ablation.png)\n\n" + tables["ablation.md"],
    ] if abl is not None else []) + ["<!-- results:end -->"])

    (tabs / "results_block.md").write_text(block + "\n")
    print(headline)
    print(f"\nfigures -> {figs}\ntables  -> {tabs}")

    if not args.no_readme:
        readme = resolve("README.md")
        text = readme.read_text()
        pattern = re.compile(r"<!-- results:start -->.*?<!-- results:end -->", re.DOTALL)
        if pattern.search(text):
            readme.write_text(pattern.sub(lambda _: block, text))
            print("README results block updated")
        else:
            print("README has no results block markers; nothing rewritten")


if __name__ == "__main__":
    main()
