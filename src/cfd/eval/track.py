"""Append-only experiment log.

Deliberately not MLflow. Every row carries the config hash, so "which run produced this
number" is answerable with a filter, and the config itself is in git.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from cfd.config import config_hash, resolve


def append_results(
    rows: list[dict[str, Any]] | pl.DataFrame,
    cfg: dict[str, Any],
    path: str | Path = "results/results.parquet",
    extra: dict[str, Any] | None = None,
) -> Path:
    df = pl.DataFrame(rows) if not isinstance(rows, pl.DataFrame) else rows
    if df.is_empty():
        raise ValueError("refusing to append an empty result set")

    df = df.with_columns(pl.lit(config_hash(cfg)).alias("config_hash"))
    for key, value in (extra or {}).items():
        df = df.with_columns(pl.lit(value).alias(key))

    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        df = pl.concat([pl.read_parquet(out), df], how="diagonal_relaxed")
    df.write_parquet(out)
    return out


def load_results(path: str | Path = "results/results.parquet") -> pl.DataFrame:
    out = resolve(path)
    return pl.read_parquet(out) if out.exists() else pl.DataFrame()
