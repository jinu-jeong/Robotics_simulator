"""Result-directory helpers: timestamped run folders, JSON metrics, config snapshots."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import save_yaml


def make_run_dir(base: str | Path, name: str | None = None, timestamp: bool = True) -> Path:
    base = Path(base)
    parts = []
    if name:
        parts.append(name)
    if timestamp:
        parts.append(time.strftime("%Y%m%d_%H%M%S"))
    run = base / "_".join(parts) if parts else base
    run.mkdir(parents=True, exist_ok=True)
    return run


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def snapshot_config(cfg: Mapping[str, Any], run_dir: str | Path, name: str = "config.yaml") -> None:
    save_yaml(cfg, Path(run_dir) / name)
