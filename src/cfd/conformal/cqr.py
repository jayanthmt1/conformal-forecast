"""Conformalized Quantile Regression (Romano, Patterson & Candes, 2019).

    s_i  = max( qlo_i - y_i ,  y_i - qhi_i )   on the calibration block
    qhat = the ceil((n+1)(1-alpha))/n empirical quantile of {s_i}
    C(x) = [qlo(x) - qhat, qhi(x) + qhat]

The quantile model supplies the *shape* -- wide where it expects demand to be volatile,
tight where it expects it to be stable -- and the conformal step supplies the *level*,
shifting both edges by a single scalar until calibration coverage hits the target. A
negative qhat is legitimate: it means the raw quantiles were already too wide.

ONE-SIDED VARIANT
The newsvendor decision does not want an interval; it wants a single calibrated quantile
at the critical fractile tau. `calibrate_upper` applies the same idea to one edge:

    s_i  = y_i - q_i,   qhat = the ceil((n+1) tau)/n quantile of {s_i},   q(x) + qhat

so that P(Y <= q(x) + qhat) >= tau under exchangeability.
"""

from __future__ import annotations

import numpy as np

from cfd.conformal.split import conformal_quantile

Array = np.ndarray


def scores(y: Array, qlo: Array, qhi: Array) -> Array:
    y = np.asarray(y, dtype=float)
    return np.maximum(np.asarray(qlo, dtype=float) - y, y - np.asarray(qhi, dtype=float))


def calibrate(y_calib: Array, qlo_calib: Array, qhi_calib: Array, alpha: float) -> float:
    return conformal_quantile(scores(y_calib, qlo_calib, qhi_calib), alpha)


def interval(qlo: Array, qhi: Array, qhat: float, floor: float | None = 0.0) -> tuple[Array, Array]:
    lower = np.asarray(qlo, dtype=float) - qhat
    upper = np.asarray(qhi, dtype=float) + qhat
    if floor is not None:
        lower = np.maximum(lower, floor)
    # a large negative qhat can cross the edges; an empty interval is never useful
    upper = np.maximum(upper, lower)
    return lower, upper


def calibrate_upper(y_calib: Array, q_calib: Array, tau: float) -> float:
    """Shift for a one-sided bound q(x) + qhat with P(Y <= bound) >= tau."""
    s = np.asarray(y_calib, dtype=float) - np.asarray(q_calib, dtype=float)
    return conformal_quantile(s, 1.0 - tau)


def upper_bound(q: Array, qhat: float, floor: float | None = 0.0) -> Array:
    out = np.asarray(q, dtype=float) + qhat
    return out if floor is None else np.maximum(out, floor)


def nominal_to_taus(level: float) -> tuple[float, float]:
    """A central interval at nominal coverage `level` uses quantiles (a/2, 1 - a/2)."""
    a = 1.0 - level
    return round(a / 2, 6), round(1.0 - a / 2, 6)
