"""Configuration loading and path helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config and attach the root directory that relative paths resolve against."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(Path(root).resolve() if root else REPO_ROOT)
    return cfg


def resolve(cfg: dict[str, Any], relative: str | Path) -> Path:
    """Resolve a path from the config against the project root."""
    p = Path(relative)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def output_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    """Return (and create) the output folders."""
    base = resolve(cfg, "outputs")
    dirs = {name: base / name for name in ("reports", "figures", "tableau", "model")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
