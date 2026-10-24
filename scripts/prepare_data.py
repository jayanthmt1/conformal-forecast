"""Raw data -> features parquet. Idempotent."""

from __future__ import annotations

import argparse

import polars as pl

from cfd.config import config_hash, load_config, resolve
from cfd.data.features import build_features, feature_columns
from cfd.data.ingest import build_panel
from cfd.splits import folds_from_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"config_hash={config_hash(cfg)}  source={cfg['data']['source']}")

    panel = build_panel(cfg)
    print(
        f"panel: {panel.height:,} rows  "
        f"{panel.get_column('series_id').n_unique():,} series  "
        f"{panel.get_column('date').min()} -> {panel.get_column('date').max()}"
    )

    df, built = build_features(panel, cfg)
    cols = feature_columns(built)
    print(f"features: {len(cols)} columns across {len(built)} groups")
    for group, group_cols in built.items():
        print(f"  {group:<10} {len(group_cols):>2}  {', '.join(group_cols)}")

    folds = folds_from_config(panel.get_column("date").to_list(), cfg)
    print(f"\n{len(folds)} walk-forward folds:")
    for fold in folds:
        print(
            f"  fold {fold.idx}  train {fold.train_start}..{fold.train_end}  "
            f"calib {fold.calib_start}..{fold.calib_end}  "
            f"test {fold.test_start}..{fold.test_end}"
        )

    out = resolve(cfg["data"]["processed_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"\nwrote {out.relative_to(resolve('.'))}  ({df.height:,} rows)")

    zero_share = df.select((pl.col("y") == 0).mean()).item()
    print(f"zero share of target: {zero_share:.1%}")


if __name__ == "__main__":
    main()
