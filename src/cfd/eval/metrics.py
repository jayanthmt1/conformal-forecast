"""Accuracy, calibration, and uncertainty-of-comparison metrics.

paired_bootstrap_ci is the one that turns "CQR was better" into a result: every headline
comparison in the README should be reported as a difference with an interval over folds,
never as two bare point numbers.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def _as_float(x) -> Array:
    return np.asarray(x, dtype=float)


def mase(y: Array, yhat: Array, y_train: Array, season: int = 7) -> float:
    """Mean absolute scaled error, scaled by in-sample seasonal-naive MAE."""
    y, yhat, y_train = _as_float(y), _as_float(yhat), _as_float(y_train)
    if y_train.size <= season:
        return float("nan")
    scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    if scale <= 0:
        return float("nan")
    return float(np.mean(np.abs(y - yhat)) / scale)


def rmsse(y: Array, yhat: Array, y_train: Array) -> float:
    """Root mean squared scaled error (M5's per-series accuracy metric)."""
    y, yhat, y_train = _as_float(y), _as_float(yhat), _as_float(y_train)
    if y_train.size < 2:
        return float("nan")
    scale = np.mean(np.diff(y_train) ** 2)
    if scale <= 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - yhat) ** 2) / scale))


def wrmsse(per_series_rmsse: Array, weights: Array) -> float:
    """Volume- or dollar-weighted RMSSE across series."""
    r, w = _as_float(per_series_rmsse), _as_float(weights)
    ok = np.isfinite(r) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.sum(r[ok] * w[ok]) / np.sum(w[ok]))


def pinball_loss(y: Array, q_pred: Array, tau: float) -> float:
    """Quantile (pinball) loss at level tau."""
    y, q_pred = _as_float(y), _as_float(q_pred)
    diff = y - q_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def coverage(y: Array, lower: Array, upper: Array) -> float:
    """Empirical marginal coverage of [lower, upper]."""
    y, lower, upper = _as_float(y), _as_float(lower), _as_float(upper)
    return float(np.mean((y >= lower) & (y <= upper)))


def mean_width(lower: Array, upper: Array) -> float:
    """Always report this next to coverage — coverage alone is gameable by widening."""
    return float(np.mean(_as_float(upper) - _as_float(lower)))


def paired_bootstrap_ci(
    a: Array,
    b: Array,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    relative: bool = False,
) -> dict[str, float]:
    """Paired bootstrap CI for the mean improvement of `a` over `b`.

    `a` and `b` are matched per-fold (or per-series) values of a cost/error metric, so a
    negative point estimate means `a` is better. With relative=True the statistic is the
    fractional reduction (b - a) / b, which is what you quote as "reduced cost by X%".
    """
    a, b = _as_float(a), _as_float(b)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must match: {a.shape} vs {b.shape}")
    n = a.size
    if n < 2:
        raise ValueError("need at least 2 paired observations")

    def stat(ia: Array, ib: Array) -> float:
        return float(np.mean((ib - ia) / ib)) if relative else float(np.mean(ia - ib))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = np.array([stat(a[i], b[i]) for i in idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])

    return {
        "point": stat(a, b),
        "lo": float(lo),
        "hi": float(hi),
        "n_pairs": float(n),
        "alpha": float(alpha),
    }
