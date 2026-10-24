"""The newsvendor: the cost-optimal order is a quantile of demand, not its mean.

With holding cost h per unsold unit and stockout cost p per unit short, expected cost
    E[ h (q - Y)^+ + p (Y - q)^+ ]
is minimised at the critical fractile
    q* = F^{-1}( p / (p + h) ).

At p/h = 1 that is the median; at p/h = 4 it is the 80th percentile. Ordering the mean is
only optimal when the cost structure is symmetric AND the demand distribution is, and
neither holds for retail demand. Everything in this package is about quantifying how
much that mistake costs.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def critical_fractile(p: float, h: float) -> float:
    if p <= 0 or h <= 0:
        raise ValueError("costs must be positive")
    return p / (p + h)


def interpolate_quantile(taus: Array, q_matrix: Array, tau: float) -> Array:
    """Row-wise linear interpolation of a predicted quantile function at level `tau`.

    taus      sorted quantile levels the model was fit at, shape (k,)
    q_matrix  monotone predictions, shape (n, k)

    Outside the fitted range the nearest fitted quantile is returned -- extrapolating a
    tail from two points is a guess dressed up as a number, so it is refused.
    """
    taus = np.asarray(taus, dtype=float)
    q = np.asarray(q_matrix, dtype=float)
    if q.ndim != 2 or q.shape[1] != taus.size:
        raise ValueError(f"q_matrix must be (n, {taus.size}); got {q.shape}")
    if np.any(np.diff(taus) <= 0):
        raise ValueError("taus must be strictly increasing")

    if tau <= taus[0]:
        return q[:, 0].copy()
    if tau >= taus[-1]:
        return q[:, -1].copy()

    j = int(np.searchsorted(taus, tau, side="right")) - 1
    w = (tau - taus[j]) / (taus[j + 1] - taus[j])
    return (1.0 - w) * q[:, j] + w * q[:, j + 1]


def newsvendor_cost(y: Array, order: Array, p: float, h: float) -> tuple[Array, Array]:
    """Per-row (holding, stockout) cost of ordering `order` against realised `y`."""
    y, order = np.asarray(y, dtype=float), np.asarray(order, dtype=float)
    holding = h * np.maximum(order - y, 0.0)
    stockout = p * np.maximum(y - order, 0.0)
    return holding, stockout
