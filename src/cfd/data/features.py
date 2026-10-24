"""Feature construction, organised into named groups so ablations are one-liners.

CAUSALITY INVARIANT
-------------------
Every feature at time t must be computable from data strictly before t (prices are an
exception: they are set in advance and known at t). Rolling statistics are therefore
always shift-then-roll, never roll-then-shift. tests/test_no_leakage.py enforces this
by perturbing all future targets and asserting no past feature moves.

ORIGIN SAFETY
-------------
Causality is necessary but not sufficient. A 28-day-ahead forecast made at origin T may
only use observations up to T, so y_lag_1 is illegal at test day 2 onward even though it
is causal. All target-derived features are therefore shifted by origin_shift == horizon,
which is why the lag set starts at 28 rather than 1.
"""

from __future__ import annotations

from typing import Any

import polars as pl

CALENDAR_COLS = ["dow", "month", "weekofyear", "dayofmonth", "snap", "event"]
PRICE_COLS = ["sell_price", "price_chg_7", "price_rel_28"]
HIERARCHY_COLS = ["dept_code", "cat_code", "store_code"]


def add_calendar(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("date").dt.weekday().cast(pl.Int8).alias("dow"),
        pl.col("date").dt.month().cast(pl.Int8).alias("month"),
        pl.col("date").dt.week().cast(pl.Int8).alias("weekofyear"),
        pl.col("date").dt.day().cast(pl.Int8).alias("dayofmonth"),
        pl.col("snap").cast(pl.Int8),
        pl.col("event").cast(pl.Int8),
    )


def add_price(df: pl.DataFrame) -> pl.DataFrame:
    """Price features relative to the series' own trailing window.

    Deliberately avoids dept-level means over the full panel: those would be computed
    across the test period, and while price is known-at-time, a statistic pooled over
    future rows is not something a production system would have.
    """
    price = pl.col("sell_price")
    return df.with_columns(
        (price / price.shift(7).over("series_id") - 1.0).alias("price_chg_7"),
        (price / price.shift(1).rolling_mean(28).over("series_id") - 1.0).alias("price_rel_28"),
    )


def add_lags(df: pl.DataFrame, lags: list[int]) -> pl.DataFrame:
    return df.with_columns(
        [pl.col("y").shift(lag).over("series_id").alias(f"y_lag_{lag}") for lag in lags]
    )


def add_rolling(df: pl.DataFrame, windows: list[int], shift: int = 1) -> pl.DataFrame:
    """Rolling statistics over y[t-shift-w+1 .. t-shift].

    shift=1 is the minimum for causality. shift=origin_shift is what a genuine h-step
    forecast requires, because at the forecast origin nothing inside the test window has
    been observed yet.
    """
    past = pl.col("y").shift(shift)  # shift first, then roll -> strictly past-only
    exprs: list[pl.Expr] = []
    for w in windows:
        exprs += [
            past.rolling_mean(w).over("series_id").alias(f"y_rmean_{w}"),
            past.rolling_std(w).over("series_id").alias(f"y_rstd_{w}"),
            past.rolling_max(w).over("series_id").alias(f"y_rmax_{w}"),
            (past.rolling_sum(w).over("series_id") == 0)
            .cast(pl.Int8)
            .alias(f"y_allzero_{w}"),
        ]
    return df.with_columns(exprs)


_HIERARCHY_SOURCES = (("dept_id", "dept_code"), ("cat_id", "cat_code"), ("store_id", "store_code"))


def add_hierarchy(df: pl.DataFrame) -> pl.DataFrame:
    """Integer codes for LightGBM's categorical handling.

    Codes come from the *sorted* unique values, not from Categorical.to_physical(): the
    latter assigns codes via polars' string cache, so the same dept could get a different
    code in a different run. That made feature values non-reproducible, which the leakage
    test caught.
    """
    exprs = []
    for col, alias in _HIERARCHY_SOURCES:
        levels = sorted(df.get_column(col).unique().to_list())
        mapping = {level: i for i, level in enumerate(levels)}
        exprs.append(pl.col(col).replace_strict(mapping, return_dtype=pl.Int32).alias(alias))
    return df.with_columns(exprs)


def feature_groups_from_config(
    cfg: dict[str, Any], groups: list[str] | None = None
) -> dict[str, list[str]]:
    """{group: [column, ...]} derived from config alone -- no data required.

    Lets a downstream script recover the group -> column map from a saved parquet without
    rebuilding features, which is the difference between a fast ablation and a slow one.
    """
    fcfg = cfg["features"]
    names = list(groups if groups is not None else fcfg["groups"])
    catalogue: dict[str, list[str]] = {
        "calendar": list(CALENDAR_COLS),
        "price": list(PRICE_COLS),
        "lags": [f"y_lag_{lag}" for lag in fcfg["lags"]],
        "rolling": [
            f"y_{stat}_{w}"
            for w in fcfg["rolling_windows"]
            for stat in ("rmean", "rstd", "rmax", "allzero")
        ],
        "hierarchy": list(HIERARCHY_COLS),
    }
    unknown = set(names) - set(catalogue)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    return {name: catalogue[name] for name in names}


def build_features(
    panel: pl.DataFrame, cfg: dict[str, Any], groups: list[str] | None = None
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Return (feature frame, {group_name: [column, ...]}).

    Pass `groups` to build a subset — that is the whole ablation mechanism.
    """
    fcfg = cfg["features"]
    groups = list(groups if groups is not None else fcfg["groups"])
    origin_shift = int(fcfg.get("origin_shift", 1))

    horizon = cfg.get("splits", {}).get("horizon")
    if horizon is not None and origin_shift != horizon:
        raise ValueError(
            f"features.origin_shift ({origin_shift}) must equal splits.horizon ({horizon}): "
            "a feature closer to t than the forecast horizon is not known at the origin"
        )
    too_recent = [lag for lag in fcfg["lags"] if lag < origin_shift]
    if too_recent:
        raise ValueError(
            f"lags {too_recent} are shorter than origin_shift={origin_shift}; predicting "
            f"{origin_shift} days ahead cannot use observations from inside the test window"
        )

    catalogue = feature_groups_from_config(cfg, groups)
    df = panel.sort(["series_id", "date"])
    built: dict[str, list[str]] = {}

    if "calendar" in groups:
        df = add_calendar(df)
        built["calendar"] = catalogue["calendar"]
    if "price" in groups:
        df = add_price(df)
        built["price"] = catalogue["price"]
    if "lags" in groups:
        lags = fcfg["lags"]
        df = add_lags(df, lags)
        built["lags"] = catalogue["lags"]
    if "rolling" in groups:
        windows = fcfg["rolling_windows"]
        df = add_rolling(df, windows, shift=origin_shift)
        built["rolling"] = catalogue["rolling"]
    if "hierarchy" in groups:
        df = add_hierarchy(df)
        built["hierarchy"] = catalogue["hierarchy"]

    unknown = set(groups) - set(built)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")

    return df, built


def feature_columns(built: dict[str, list[str]], groups: list[str] | None = None) -> list[str]:
    keys = list(built) if groups is None else groups
    return [col for key in keys for col in built[key]]


def add_segments(
    df: pl.DataFrame, train_only: pl.DataFrame, n_volume_bins: int = 3
) -> pl.DataFrame:
    """Attach volume tercile and intermittency class for conditional-coverage analysis.

    Segments are derived from `train_only` so the grouping itself never sees test data.
    """
    stats = train_only.group_by("series_id").agg(
        pl.col("y").mean().alias("mean_y"),
        pl.col("y").std().alias("std_y"),
        (pl.col("y") == 0).mean().alias("zero_share"),
    )
    stats = stats.with_columns(
        (pl.col("std_y") / (pl.col("mean_y") + 1e-9)).alias("cv"),
        pl.col("mean_y")
        .qcut(n_volume_bins, labels=["low", "mid", "high"], allow_duplicates=True)
        .cast(pl.Utf8)
        .alias("volume_seg"),
    ).with_columns(
        pl.when(pl.col("zero_share") > 0.30)
        .then(pl.lit("intermittent"))
        .when(pl.col("cv") > 1.0)
        .then(pl.lit("volatile"))
        .otherwise(pl.lit("smooth"))
        .alias("demand_seg")
    )
    return df.join(
        stats.select("series_id", "volume_seg", "demand_seg", "zero_share", "cv"),
        on="series_id",
        how="left",
    )
