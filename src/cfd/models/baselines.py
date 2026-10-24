"""Baseline ladder rung 0 and 1.

Both are *origin* forecasts: fitted on history up to the forecast origin, then projected
`horizon` steps with no further observations. That is what a real 28-day-ahead ordering
decision looks like, and it is why these numbers are honest lower bounds rather than the
inflated ones you get from feeding y_lag_7 straight through the test window.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl


def seasonal_naive(history: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    """Repeat the last full season forward."""
    if history.size == 0:
        return np.zeros(horizon, dtype=float)
    season = min(season, history.size)
    last = np.asarray(history[-season:], dtype=float)
    reps = int(np.ceil(horizon / season))
    return np.tile(last, reps)[:horizon]


def croston(
    history: np.ndarray,
    horizon: int,
    alpha: float = 0.1,
    variant: str = "sba",
) -> np.ndarray:
    """Croston's method for intermittent demand; flat forecast.

    variant='classic' -> z/p, variant='sba' -> Syntetos-Boylan bias correction.
    """
    y = np.asarray(history, dtype=float)
    nz = np.flatnonzero(y > 0)
    if nz.size == 0:
        return np.zeros(horizon, dtype=float)

    sizes = y[nz]
    intervals = np.diff(np.concatenate(([-1], nz))).astype(float)

    z, p = sizes[0], intervals[0]
    for size, interval in zip(sizes[1:], intervals[1:], strict=True):
        z += alpha * (size - z)
        p += alpha * (interval - p)

    rate = z / max(p, 1e-9)
    if variant == "sba":
        rate *= 1.0 - alpha / 2.0
    elif variant != "classic":
        raise ValueError(f"unknown croston variant: {variant!r}")

    return np.full(horizon, rate, dtype=float)


BASELINES: dict[str, Callable[..., np.ndarray]] = {
    "seasonal_naive": seasonal_naive,
    "croston": croston,
}


def forecast_panel(
    df: pl.DataFrame,
    model: str,
    horizon: int,
    history_split: str = "train",
    target_split: str = "test",
    **kwargs,
) -> pl.DataFrame:
    """Apply a baseline per series.

    History is everything in `history_split` and `calib` up to the test origin, which is
    what a production run would legitimately have.
    """
    if model not in BASELINES:
        raise ValueError(f"unknown baseline {model!r}; have {sorted(BASELINES)}")
    fn = BASELINES[model]

    hist_splits = [history_split, "calib"] if history_split == "train" else [history_split]
    history = df.filter(pl.col("split").is_in(hist_splits)).sort(["series_id", "date"])
    target = df.filter(pl.col("split") == target_split).sort(["series_id", "date"])

    # group_by yields the key as a tuple -- unpack it, or every lookup below misses
    # and each series silently forecasts from an empty history.
    hist_map = {
        sid: grp.get_column("y").to_numpy()
        for (sid,), grp in history.group_by("series_id", maintain_order=True)
    }

    out = []
    for (sid,), grp in target.group_by("series_id", maintain_order=True):
        h = grp.height
        preds = fn(hist_map.get(sid, np.array([])), max(h, horizon), **kwargs)[:h]
        out.append(
            grp.select("date", "series_id", "y").with_columns(
                pl.Series("yhat", preds, dtype=pl.Float64),
                pl.lit(model).alias("model"),
            )
        )
    return pl.concat(out) if out else pl.DataFrame()
