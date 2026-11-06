"""Config loading with `inherit:` chains (docs/PLAN.md Section 4: all configs inherit base.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> DictConfig:
    """Load a YAML config, merging it over its `inherit:` parent (resolved relative to the file).

    `overrides` are `key=value` strings (dotted keys), applied last.
    """
    path = Path(path)
    cfg = OmegaConf.load(path)
    parent = cfg.pop("inherit", None)
    if parent is not None:
        cfg = OmegaConf.merge(load_config(path.parent / parent), cfg)
    overrides = list(overrides)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def repo_path(rel: str | Path, root: Path = REPO_ROOT) -> Path:
    """Resolve a config path (relative to the repo root) to an absolute path."""
    p = Path(rel)
    return p if p.is_absolute() else root / p
