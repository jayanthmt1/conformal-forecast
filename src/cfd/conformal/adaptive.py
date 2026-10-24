"""Adaptive Conformal Inference (Gibbs & Candes, 2021).

    alpha_{t+1} = alpha_t + gamma * ( alpha - err_t ),     err_t = 1{ y_t not in C_t }

where C_t is the conformal interval built at miscoverage level alpha_t. When the model
has been under-covering, alpha_t falls and the interval widens; when it has been
over-covering, alpha_t rises and the interval tightens.

STATE THE GUARANTEE PRECISELY
ACI gives *long-run time-averaged* coverage: the empirical miscoverage frequency over the
stream converges to alpha with no distributional assumptions at all -- it holds under
arbitrary shift. It does NOT restore finite-sample marginal coverage at any individual
step, it says nothing about conditional coverage, and it says nothing about interval
width. The known failure mode is width inflating to hold the frequency on target, which
is why every table in this project reports width next to coverage.

HOW IT IS APPLIED HERE
One alpha_t is shared by all series and updated once per test day using that day's
pooled miscoverage rate across series. err_t is therefore a fraction in [0, 1] rather
than a single indicator, which is the natural panel analogue. `feedback_lag` delays the
update: a 28-day-ahead system learns a day's outcome long after forecasting it, and
lag=1 is the optimistic end of that range. alpha_t is clipped to [eps, 1 - eps]; at the
lower clip the interval is as wide as the calibration scores allow rather than infinite.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from cfd.conformal.split import conformal_quantile

Array = np.ndarray
IntervalFn = Callable[[int, float], tuple[Array, Array]]


@dataclass
class ACITrace:
    alpha_t: list[float] = field(default_factory=list)
    coverage: list[float] = field(default_factory=list)
    width: list[float] = field(default_factory=list)
    lower: list[Array] = field(default_factory=list)
    upper: list[Array] = field(default_factory=list)


def run_aci(
    y_by_day: Sequence[Array],
    interval_at: IntervalFn,
    alpha: float,
    gamma: float,
    feedback_lag: int = 1,
    eps: float = 1e-3,
) -> ACITrace:
    """Drive alpha_t through the stream.

    y_by_day[t]        realised demand for every series on test day t
    interval_at(t, a)  (lower, upper) for day t at miscoverage level a
    """
    if feedback_lag < 1:
        raise ValueError("feedback_lag must be >= 1: today's outcome is not known today")

    trace = ACITrace()
    a_t = float(alpha)
    errs: list[float] = []
    for t, y in enumerate(y_by_day):
        a_t = float(np.clip(a_t, eps, 1.0 - eps))
        lower, upper = interval_at(t, a_t)
        covered = (y >= lower) & (y <= upper)
        errs.append(float(1.0 - covered.mean()))

        trace.alpha_t.append(a_t)
        trace.coverage.append(float(covered.mean()))
        trace.width.append(float(np.mean(upper - lower)))
        trace.lower.append(lower)
        trace.upper.append(upper)

        feedback_idx = t - (feedback_lag - 1)
        if feedback_idx >= 0:
            a_t = a_t + gamma * (alpha - errs[feedback_idx])
    return trace


def cqr_interval_fn(
    calib_scores: Array, qlo_by_day: Sequence[Array], qhi_by_day: Sequence[Array], floor: float = 0.0
) -> IntervalFn:
    """CQR edges for day t at any alpha -- the static method, re-levelled on demand."""

    def fn(t: int, a: float) -> tuple[Array, Array]:
        qhat = conformal_quantile(calib_scores, a)
        lower = np.maximum(qlo_by_day[t] - qhat, floor)
        upper = np.maximum(qhi_by_day[t] + qhat, lower)
        return lower, upper

    return fn


def upper_bound_fn(
    calib_scores: Array, q_by_day: Sequence[Array], floor: float = 0.0
) -> IntervalFn:
    """One-sided bound for the ordering decision: lower is -inf... floored at zero demand."""

    def fn(t: int, a: float) -> tuple[Array, Array]:
        qhat = conformal_quantile(calib_scores, a)
        upper = np.maximum(q_by_day[t] + qhat, floor)
        return np.full_like(upper, floor), upper

    return fn
