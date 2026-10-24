from __future__ import annotations

from datetime import date, timedelta
from itertools import pairwise

import polars as pl
import pytest

from cfd.splits import Fold, make_folds

DATES = [date(2020, 1, 1) + timedelta(days=i) for i in range(900)]


def test_folds_are_strictly_ordered_in_time() -> None:
    for fold in make_folds(DATES, n_folds=5, horizon=28, calib_len=56, min_train_days=365):
        assert fold.train_end < fold.calib_start
        assert fold.calib_end < fold.test_start


def test_test_windows_are_consecutive_and_disjoint() -> None:
    folds = make_folds(DATES, n_folds=5, horizon=28, calib_len=56, min_train_days=365)
    for prev, nxt in pairwise(folds):
        assert prev.test_end < nxt.test_start
        assert (nxt.test_start - prev.test_end).days == 1


def test_train_window_expands() -> None:
    folds = make_folds(DATES, n_folds=5, horizon=28, calib_len=56, min_train_days=365)
    ends = [f.train_end for f in folds]
    assert ends == sorted(ends)
    assert ends[0] < ends[-1]


def test_horizon_length_is_respected() -> None:
    for fold in make_folds(DATES, n_folds=3, horizon=28, calib_len=56, min_train_days=365):
        assert (fold.test_end - fold.test_start).days + 1 == 28
        assert (fold.calib_end - fold.calib_start).days + 1 == 56


def test_raises_when_history_is_too_short() -> None:
    with pytest.raises(ValueError, match="need >="):
        make_folds(DATES[:200], n_folds=5, horizon=28, calib_len=56, min_train_days=365)


def test_non_monotonic_fold_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-monotonic"):
        Fold(
            idx=0,
            train_start=date(2020, 1, 1),
            train_end=date(2020, 6, 1),
            calib_start=date(2020, 5, 1),  # overlaps train
            calib_end=date(2020, 7, 1),
            test_start=date(2020, 8, 1),
            test_end=date(2020, 9, 1),
        )


def test_assign_labels_every_row_exactly_once() -> None:
    fold = make_folds(DATES, n_folds=1, horizon=28, calib_len=56, min_train_days=365)[0]
    df = pl.DataFrame({"date": DATES, "y": range(len(DATES))})
    labelled = fold.assign(df)

    counts = dict(
        labelled.group_by("split").agg(pl.len().alias("n")).iter_rows()  # type: ignore[arg-type]
    )
    assert counts["test"] == 28
    assert counts["calib"] == 56
    assert labelled.height == sum(counts.values())
