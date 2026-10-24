"""Raw data -> one tidy long-format panel.

Every downstream stage consumes PANEL_SCHEMA and nothing else, so swapping M5 for
another dataset (Favorita, electricity load) means writing one loader here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from cfd.config import resolve
from cfd.data.synth import make_synthetic_panel

PANEL_SCHEMA = [
    "date",
    "series_id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "y",
    "sell_price",
    "snap",
    "event",
]


def load_m5(
    raw_dir: str | Path,
    store: str | None = None,
    top_n_series: int | None = None,
) -> pl.DataFrame:
    """Load the M5 Accuracy CSVs into PANEL_SCHEMA.

    Expects calendar.csv, sell_prices.csv, and sales_train_evaluation.csv
    (falls back to sales_train_validation.csv) in raw_dir.
    """
    raw = resolve(raw_dir)
    calendar = pl.read_csv(raw / "calendar.csv", try_parse_dates=True)
    prices = pl.read_csv(raw / "sell_prices.csv")

    sales_path = raw / "sales_train_evaluation.csv"
    if not sales_path.exists():
        sales_path = raw / "sales_train_validation.csv"
    sales = pl.read_csv(sales_path)

    if store is not None:
        sales = sales.filter(pl.col("store_id") == store)

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    long = sales.unpivot(
        on=day_cols, index=id_cols, variable_name="d", value_name="y"
    )

    if top_n_series is not None:
        keep = (
            long.group_by("id")
            .agg(pl.col("y").sum().alias("total"))
            .sort("total", descending=True)
            .head(top_n_series)
            .get_column("id")
        )
        long = long.filter(pl.col("id").is_in(keep))

    cal = calendar.select(
        "d", "date", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI", "event_name_1"
    )
    long = long.join(cal, on="d", how="left").with_columns(
        pl.when(pl.col("state_id") == "CA")
        .then(pl.col("snap_CA"))
        .when(pl.col("state_id") == "TX")
        .then(pl.col("snap_TX"))
        .otherwise(pl.col("snap_WI"))
        .cast(pl.Int8)
        .alias("snap"),
        pl.col("event_name_1").is_not_null().cast(pl.Int8).alias("event"),
    )

    long = long.join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    return (
        long.rename({"id": "series_id"})
        .with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("y").cast(pl.Int64),
            pl.col("sell_price").cast(pl.Float64),
        )
        .select(PANEL_SCHEMA)
        .sort(["series_id", "date"])
    )


def build_panel(cfg: dict[str, Any]) -> pl.DataFrame:
    """Dispatch on cfg['data']['source']."""
    dcfg = cfg["data"]
    source = dcfg["source"]
    if source == "synthetic":
        panel = make_synthetic_panel(**dcfg["synthetic"])
    elif source == "m5":
        panel = load_m5(dcfg["raw_dir"], dcfg.get("store"), dcfg.get("top_n_series"))
    else:
        raise ValueError(f"unknown data.source: {source!r}")

    missing = set(PANEL_SCHEMA) - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")
    return panel.select(PANEL_SCHEMA)
