"""Config loading and run identity.

Reproducibility here is deliberately boring: YAML in git, plus a stable hash of the
resolved config used to key every row written to results.parquet. No tracking server.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/base.yaml"


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with resolve(path).open() as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict[str, Any]) -> str:
    """Short stable hash of a config dict — the run identity."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
