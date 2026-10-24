"""Walk-forward temporal splits with a dedicated calibration block.

    |------------- train -------------|-- calib --|-- test --|

The calibration block is mandatory and must sit strictly between train and test in
time: split conformal / CQR need a held-out set that the model never saw, and for a
time series it has to be the *most recent* such set. Train windows expand across folds.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl

SPLIT_NAMES = ("train", "calib", "test", "unused")


@dataclass(frozen=True)
class Fold:
    idx: int
    train_start: date
    train_end: date
    calib_start: date
    calib_end: date
    test_start: date
    test_end: date

    def __post_init__(self) -> None:
        if not (
            self.train_start <= self.train_end
            < self.calib_start <= self.calib_end
            < self.test_start <= self.test_end
        ):
            raise ValueError(f"fold {self.idx} has non-monotonic boundaries: {self}")

    def assign(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add a `split` column labelling each row train/calib/test/unused."""
        d = pl.col("date")
        return df.with_columns(
            pl.when((d >= self.train_start) & (d <= self.train_end))
            .then(pl.lit("train"))
            .when((d >= self.calib_start) & (d <= self.calib_end))
            .then(pl.lit("calib"))
            .when((d >= self.test_start) & (d <= self.test_end))
            .then(pl.lit("test"))
            .otherwise(pl.lit("unused"))
            .alias("split")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": self.idx,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "calib_start": self.calib_start,
            "calib_end": self.calib_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


def make_folds(
    dates: Iterable[date],
    n_folds: int = 5,
    horizon: int = 28,
    calib_len: int = 56,
    min_train_days: int = 365,
) -> list[Fold]:
    """Non-overlapping consecutive test windows, oldest fold first."""
    unique: Sequence[date] = sorted(set(dates))
    n = len(unique)

    needed = min_train_days + calib_len + n_folds * horizon
    if n < needed:
        raise ValueError(
            f"need >= {needed} distinct dates for {n_folds} folds "
            f"(horizon={horizon}, calib_len={calib_len}, min_train={min_train_days}); got {n}"
        )

    folds: list[Fold] = []
    for i in range(n_folds):
        test_hi = n - (n_folds - 1 - i) * horizon  # exclusive
        test_lo = test_hi - horizon
        calib_hi = test_lo
        calib_lo = calib_hi - calib_len
        train_hi = calib_lo
        folds.append(
            Fold(
                idx=i,
                train_start=unique[0],
                train_end=unique[train_hi - 1],
                calib_start=unique[calib_lo],
                calib_end=unique[calib_hi - 1],
                test_start=unique[test_lo],
                test_end=unique[test_hi - 1],
            )
        )
    return folds


def folds_from_config(dates: Iterable[date], cfg: dict[str, Any]) -> list[Fold]:
    s = cfg["splits"]
    return make_folds(
        dates,
        n_folds=s["n_folds"],
        horizon=s["horizon"],
        calib_len=s["calib_len"],
        min_train_days=s["min_train_days"],
    )
