"""Minimal YAML configuration utilities.

All experiment / viewer parameters live in ``configs/*.yaml``; nothing is
hard-coded in scripts. This module keeps the dependency surface small: a
config is a plain nested ``dict`` with dotted-path access helpers.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict (empty file -> ``{}``)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    """Write a mapping to YAML (used to snapshot configs next to results)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False)


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` or an explicit path.

    ``load_config("viewer")`` and ``load_config("configs/viewer.yaml")`` are
    equivalent when run from the project root.
    """
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return load_yaml(p)
    candidate = CONFIG_DIR / f"{name_or_path}.yaml"
    if candidate.exists():
        return load_yaml(candidate)
    raise FileNotFoundError(f"Config not found: {name_or_path!r} (looked at {p} and {candidate})")


def deep_update(base: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out: dict[str, Any] = copy.deepcopy(dict(base))
    if not override:
        return out
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_path(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """``get_path(cfg, "window.width", 1280)``."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node
