"""Split conformal prediction around a point forecast.

    s_i  = |y_i - yhat_i|                      on the calibration block
    qhat = the ceil((n+1)(1-alpha))/n empirical quantile of {s_i}
    C(x) = [yhat(x) - qhat, yhat(x) + qhat]

Under exchangeability, P(Y in C(X)) >= 1 - alpha, with no assumption on the model or the
noise. The interval has one width for every row, which is exactly its weakness: a volatile
SKU and a stable one get the same band. CQR (cqr.py) fixes that.

WHAT THIS DOES NOT PROMISE
Time-series rows are not exchangeable, so the finite-sample guarantee is not claimed to
transfer unchanged. Coverage is *measured* on the test block, never assumed.
"""

from __future__ import annotations

import math

import numpy as np

Array = np.ndarray


def conformal_quantile(scores: Array, alpha: float) -> float:
    """The finite-sample-corrected (1 - alpha) quantile of nonconformity scores.

    Uses ceil((n+1)(1-alpha))/n rather than (1-alpha): the +1 is what pays for the
    guarantee holding at every n, not just asymptotically. When that level exceeds 1 the
    only valid interval is the whole real line, which is reported as +inf rather than
    quietly clipped.
    """
    s = np.asarray(scores, dtype=float)
    s = np.sort(s[np.isfinite(s)])
    n = s.size
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * (1.0 - alpha))  # the k-th order statistic, 1-indexed
    if k > n:
        return float("inf")
    return float(s[max(k, 1) - 1])


def absolute_scores(y: Array, yhat: Array) -> Array:
    return np.abs(np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float))


def calibrate(y_calib: Array, yhat_calib: Array, alpha: float) -> float:
    """qhat for a symmetric interval at nominal coverage 1 - alpha."""
    return conformal_quantile(absolute_scores(y_calib, yhat_calib), alpha)


def interval(yhat: Array, qhat: float, floor: float | None = 0.0) -> tuple[Array, Array]:
    """[yhat - qhat, yhat + qhat], with the lower bound floored at `floor`.

    Flooring at zero is free for a non-negative target: it cannot lose coverage and it
    stops the reported width being inflated by a region demand cannot occupy.
    """
    yhat = np.asarray(yhat, dtype=float)
    lower, upper = yhat - qhat, yhat + qhat
    if floor is not None:
        lower = np.maximum(lower, floor)
    return lower, upper
