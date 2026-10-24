# Distribution-Free Probabilistic Demand Forecasting & Inventory Optimization

Retail demand forecasting where the output is a **prediction interval with distribution-free
coverage validity**, and that interval feeds a **cost-optimal ordering decision** instead of being
reported and ignored.

> **Result** — *pending; filled in once the experiments run.* Will read: calibrated interval
> forecasting reduced simulated inventory cost by **X% (95% CI: a–b%)** versus mean-based ordering
> at a stockout/holding ratio of **4:1**, while holding **Y% empirical coverage** against a 90%
> nominal target, across **5 walk-forward folds** and **N** SKU-store series.

## Why intervals instead of point forecasts

Two things are wrong with forecasting a single number and ordering that much:

1. **A point forecast discards uncertainty.** Two SKUs both forecast at 100 units are not
   equivalent if one's plausible range is 95–105 and the other's is 20–300.
2. **The training loss is not the business loss.** Models are fit with MSE/MAE, which are
   symmetric. Real cost is not: unsold stock costs holding `h` per unit, a stockout costs lost
   margin `p`. When `p ≠ h`, ordering the *mean* is provably the wrong decision.

The newsvendor optimum is a **quantile** of the demand distribution, not its mean:

```
q* = F⁻¹( p / (p + h) )
```

At `p/h = 4` the cost-optimal order is roughly the 80th percentile. A point forecast therefore
structurally under-orders whenever stockouts cost more than holding — and the gap widens with the
cost ratio. This project quantifies that gap.

## Method

| Stage | Approach |
|---|---|
| Forecasting | Baseline ladder: seasonal naive → Croston → LightGBM point → multi-quantile LightGBM |
| Uncertainty | Split conformal, then **Conformalized Quantile Regression** (adaptive interval width) |
| Robustness | Controlled distribution-shift experiment, then Adaptive Conformal Inference |
| Decision | Newsvendor critical fractile; cost simulated across `p/h` ∈ [1, 10] |
| Evidence | Walk-forward CV, calibration curves, segment-conditional coverage, paired bootstrap CIs |

### On what conformal prediction does and does not give you

Conformal prediction does **not** produce a calibrated predictive *distribution*. Split conformal
gives calibrated prediction *intervals* under exchangeability; CQR gives calibrated intervals built
around conditional quantile estimates. Since time-series observations are not generally
exchangeable, the vanilla finite-sample guarantee is not claimed to transfer unchanged — coverage
is measured empirically under temporal drift instead.

Adaptive Conformal Inference is not a repair for that. It provides *long-run time-averaged*
coverage: the empirical miscoverage frequency converges to α with no distributional assumptions. It
says nothing about coverage at an individual step, nothing about conditional coverage, and nothing
about interval **width** — width inflation is its known failure mode. Width is therefore reported
next to coverage throughout.

## Data

M5 Forecasting — Accuracy (Walmart): 30,490 item×store series, 1,941 days, plus calendar (holidays,
events, SNAP days) and prices. Scoped to one store / top-N series by volume; the subset is recorded
in `configs/base.yaml` rather than left implicit.

A synthetic M5-like generator (`cfd.data.synth`) ships with the repo so the pipeline, tests, and
baselines run with no download. It includes smooth, volatile, and intermittent regimes, which is
what the segment-conditional coverage analysis needs.

## Quickstart

```bash
uv sync --extra dev
uv run pytest -q                        # leakage, split, and metric tests
uv run python scripts/prepare_data.py   # panel -> features parquet (synthetic by default)
uv run python scripts/run_baselines.py  # seasonal naive + Croston across all folds
uv run python scripts/run_gbm.py        # LightGBM point + multi-quantile across all folds
```

To switch to real M5: put `calendar.csv`, `sell_prices.csv`, and `sales_train_evaluation.csv` in
`data/raw/`, then set `data.source: m5` in `configs/base.yaml`.

## The causality invariant

Every feature at time *t* is computable from data strictly before *t*. Rolling statistics are
`shift(1)`-then-roll, never roll-then-`shift`. This is enforced, not asserted: the test in
`tests/test_no_leakage.py` corrupts every future target and checks that no past feature moves.

```bash
uv run pytest tests/test_no_leakage.py -v
```

## Layout

```
src/cfd/
├── config.py             # YAML config + stable run hash
├── data/
│   ├── synth.py          # synthetic M5-like panel (3 demand regimes)
│   ├── ingest.py         # raw -> PANEL_SCHEMA long format
│   └── features.py       # named feature groups; segment labels
├── splits.py             # walk-forward train | calib | test
├── models/baselines.py   # seasonal naive, Croston (SBA)
├── models/gbm.py         # global LightGBM point + multi-quantile boosters
└── eval/
    ├── metrics.py        # MASE, RMSSE, WRMSSE, pinball, coverage, paired bootstrap
    └── track.py          # append-only results.parquet keyed by config hash
```

Reproducibility is YAML in git plus a config hash on every result row — no tracking server.

## Status

- [x] Panel ingest + synthetic generator
- [x] Named feature groups (ablation-ready)
- [x] Walk-forward splitter with dedicated calibration block
- [x] Leakage / alignment / metric tests
- [x] Baselines: seasonal naive, Croston
- [x] LightGBM point + multi-quantile
- [ ] CQR calibration; calibration curve; segment-conditional coverage
- [ ] Newsvendor decision layer + `p/h` cost sweep
- [ ] Paired bootstrap CIs on headline comparisons
- [ ] Degradation experiment → Adaptive Conformal Inference
- [ ] Streamlit view; FastAPI + Docker

## References

- Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free
  Uncertainty Quantification*
- Romano, Patterson & Candès, *Conformalized Quantile Regression* (NeurIPS 2019)
- Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift* (NeurIPS 2021)
- Xu & Xie, *Conformal Prediction Interval for Dynamic Time-Series* (ICML 2021)
- Ban & Rudin, *The Big Data Newsvendor: Practical Insights from Machine Learning*
  (Operations Research 2019)
- Bertsimas & Kallus, *From Predictive to Prescriptive Analytics* (Management Science 2020)
- Makridakis, Spiliotis & Assimakopoulos, *The M5 competition: Background, organization, and
  implementation* (IJF 2022)
