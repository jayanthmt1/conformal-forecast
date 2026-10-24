"""Single-page view of the results: the headline, the figures, and a per-series explorer.

    uv run --extra app streamlit run app/streamlit_app.py

Reads only from results/ and reports/ -- nothing is retrained here. The explorer recomputes
order quantities for one series from the stored quantile forecasts, which is cheap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfd.config import load_config, resolve
from cfd.decision.newsvendor import critical_fractile
from cfd.decision.simulate import order_quantities

st.set_page_config(page_title="Conformal demand forecasting", layout="wide")


@st.cache_data
def load():
    cfg = load_config()
    res = resolve("results")
    out = {
        "cfg": cfg,
        "preds": pl.read_parquet(resolve(cfg["eval"]["preds_path"])),
        "intervals": pl.read_parquet(res / "preds" / "intervals.parquet"),
        "decision": pl.read_parquet(res / "decision.parquet"),
        "ci": pl.read_parquet(res / "decision_ci.parquet"),
        "conformal": pl.read_parquet(res / "conformal.parquet"),
    }
    block = resolve(cfg["eval"]["tables_dir"]) / "results_block.md"
    out["headline"] = next(
        (line for line in block.read_text().splitlines() if line.startswith("> **Result**")), ""
    ) if block.exists() else ""
    return out


data = load()
cfg = data["cfg"]
figs = resolve(cfg["eval"]["figures_dir"])
level = round(1 - cfg["conformal"]["alpha"], 6)

st.title("Distribution-free demand forecasting → inventory decisions")
st.caption(
    "Prediction intervals with measured coverage, fed into a newsvendor ordering rule. "
    f"Data source: `{cfg['data']['source']}`."
)
if data["headline"]:
    st.markdown(data["headline"])

tab_overview, tab_explore, tab_method = st.tabs(["Results", "Series explorer", "Method"])

with tab_overview:
    c1, c2 = st.columns(2)
    c1.image(str(figs / "cost_sweep.png"), caption="Cost relative to ordering the mean, swept over p/h")
    c2.image(str(figs / "calibration_curve.png"), caption="Empirical vs nominal coverage")
    c1.image(str(figs / "coverage_by_segment.png"), caption=f"Coverage by segment at nominal {level:.0%}")
    if (figs / "shift_coverage_over_time.png").exists():
        c2.image(str(figs / "shift_coverage_over_time.png"), caption="Static CQR vs ACI under injected shift")
    if (figs / "ablation.png").exists():
        st.image(str(figs / "ablation.png"), caption="Leave-one-group-out ablation")

    st.subheader("Cost reduction vs point policy (paired bootstrap 95% CI)")
    st.dataframe(
        data["ci"].with_columns(
            pl.col("reduction", "lo", "hi").map_elements(lambda v: f"{v:+.1%}", return_dtype=pl.Utf8)
        ).to_pandas(),
        hide_index=True,
        width="stretch",
    )

with tab_explore:
    preds = data["preds"]
    left, right = st.columns([1, 3])
    with left:
        fold = st.selectbox("fold", sorted(preds.get_column("fold").unique().to_list()), index=0)
        fold_preds = preds.filter(pl.col("fold") == fold)
        segs = fold_preds.select("series_id", "demand_seg").unique().sort("series_id")
        seg_filter = st.multiselect(
            "demand segment", sorted(segs.get_column("demand_seg").unique().to_list()),
            default=sorted(segs.get_column("demand_seg").unique().to_list()),
        )
        choices = segs.filter(pl.col("demand_seg").is_in(seg_filter)).get_column("series_id").to_list()
        series = st.selectbox("series", choices)
        ratio = st.slider("stockout / holding cost ratio  p/h", 1.0, 10.0, 4.0, 0.5)
        tau = critical_fractile(ratio, 1.0)
        st.metric("critical fractile  p/(p+h)", f"{tau:.3f}")

    calib = fold_preds.filter(pl.col("split") == "calib")
    test = fold_preds.filter((pl.col("split") == "test") & (pl.col("series_id") == series)).sort("date")
    orders = order_quantities(calib, test, tau)
    y = test.get_column("y").to_numpy()

    iv = data["intervals"].filter(
        (pl.col("fold") == fold) & (pl.col("series_id") == series) & (pl.col("method") == "cqr")
    ).sort("date")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    x = test.get_column("date").to_list()
    if iv.height:
        ax.fill_between(x, iv.get_column("lower").to_numpy(), iv.get_column("upper").to_numpy(),
                        alpha=0.18, color="#1a73e8", label=f"CQR {level:.0%} interval")
    ax.plot(x, y, "k.-", label="realised demand")
    ax.plot(x, orders["point"], color="#5f6368", ls="--", label="order: point (mean)")
    ax.plot(x, orders["quantile"], color="#e0a100", label=f"order: raw quantile τ={tau:.2f}")
    ax.plot(x, orders["conformal"], color="#1a73e8", label="order: conformal quantile")
    ax.set(title=f"{series}  ·  fold {fold}  ·  test window", ylabel="units")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    with right:
        st.pyplot(fig, clear_figure=True)

        p, h = ratio, 1.0
        rows = []
        for name, q in orders.items():
            holding = h * np.maximum(q - y, 0).sum()
            short = p * np.maximum(y - q, 0).sum()
            rows.append({"policy": name, "holding cost": holding, "stockout cost": short,
                         "total": holding + short, "service level": float(np.mean(y <= q))})
        st.dataframe(pl.DataFrame(rows).to_pandas().round(2), hide_index=True, width="stretch")
        seg = test.get_column("demand_seg")[0]
        st.caption(f"segment: **{seg}** · {len(y)} test days · costs are for this series only")

with tab_method:
    st.markdown(
        f"""
**Forecaster.** Global LightGBM across all series: a Tweedie point model (the conditional mean) and
one pinball-loss model per quantile level. Lag and rolling features are shifted by the full
{cfg['splits']['horizon']}-day horizon, so every feature is known at the forecast origin.

**Calibration.** Each walk-forward fold has a calibration block strictly between train and test.
*Split conformal* widens the point forecast symmetrically; *CQR* shifts the quantile model's
(α/2, 1−α/2) predictions by a single scalar until calibration coverage hits the target. The width then
adapts to the series while the level is set by data the model never saw.

**What is and is not promised.** Split conformal and CQR give finite-sample *marginal* coverage under
exchangeability. Time series are not exchangeable, so coverage here is **measured** on held-out test
windows, not assumed. ACI gives *long-run time-averaged* coverage under arbitrary shift, and says
nothing about any single step, about conditional coverage, or about width — which is why width is
shown next to every coverage number.

**Decision.** With stockout cost *p* and holding cost *h*, expected cost is minimised by ordering the
*p/(p+h)* quantile of demand, not its mean. Three policies are simulated: order the mean, order the raw
model quantile, order the conformally calibrated quantile (one-sided CQR at that fractile).
"""
    )
