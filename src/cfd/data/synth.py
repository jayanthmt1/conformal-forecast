"""Synthetic M5-like panel.

Exists so the whole pipeline, the leakage test, and the baselines run today without a
Kaggle download. Deliberately includes the three regimes that matter for the
segment-conditional coverage analysis: smooth, volatile, and intermittent.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

CATEGORIES = ["FOODS", "HOUSEHOLD", "HOBBIES"]
REGIMES = ["smooth", "volatile", "intermittent"]


def make_synthetic_panel(
    n_series: int = 60,
    n_days: int = 900,
    seed: int = 42,
    start: date = date(2013, 1, 1),
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    dow = np.array([d.weekday() for d in dates])
    t = np.arange(n_days, dtype=float)

    frames = []
    for s in range(n_series):
        regime = REGIMES[s % len(REGIMES)]
        cat = CATEGORIES[s % len(CATEGORIES)]
        dept = f"{cat}_{1 + (s // len(CATEGORIES)) % 3}"
        item = f"{dept}_{s:03d}"
        store = "CA_1"

        base = {"smooth": 40.0, "volatile": 18.0, "intermittent": 1.2}[regime]
        phase = rng.integers(0, 7)
        weekly = 1.0 + 0.35 * np.sin(2 * np.pi * (dow + phase) / 7.0)
        trend = 1.0 + 0.0004 * t * rng.normal(1.0, 0.3)

        price = 3.0 + rng.normal(0.0, 0.4) + 0.3 * np.sin(2 * np.pi * t / 180.0)
        price = np.maximum(price, 0.8)
        price_effect = (price / price.mean()) ** -1.2

        snap = ((t % 30) < 10).astype(np.int8)
        event = (rng.random(n_days) < 0.02).astype(np.int8)

        mu = base * weekly * trend * price_effect * (1 + 0.15 * snap + 0.6 * event)
        if regime == "volatile":
            # sigma large enough that the series' CV actually clears the volatile
            # threshold in add_segments -- otherwise the three regimes aren't separable
            # and the segment-conditional coverage analysis has nothing to condition on.
            mu = mu * rng.lognormal(0.0, 0.9, n_days)
        y = rng.poisson(np.maximum(mu, 0.01))
        if regime == "intermittent":
            y = y * (rng.random(n_days) < 0.35)

        frames.append(
            pl.DataFrame(
                {
                    "date": dates,
                    "series_id": [f"{item}_{store}"] * n_days,
                    "item_id": [item] * n_days,
                    "dept_id": [dept] * n_days,
                    "cat_id": [cat] * n_days,
                    "store_id": [store] * n_days,
                    "y": y.astype(np.int64),
                    "sell_price": price.astype(np.float64),
                    "snap": snap,
                    "event": event,
                }
            )
        )

    return (
        pl.concat(frames)
        .with_columns(pl.col("date").cast(pl.Date))
        .sort(["series_id", "date"])
    )
