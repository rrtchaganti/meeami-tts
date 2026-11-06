"""Test-split lock (docs/PLAN.md Section 2, "Enforcement"; CLAUDE.md).

Any code that opens audio of an L2 speaker must call `assert_audio_allowed(speaker, cfg)` first.
Test-speaker audio is refused unless the git tag `cfg.freeze_tag` exists.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from omegaconf import DictConfig

from tts.config import REPO_ROOT


class SplitLockedError(PermissionError):
    """Raised when test-speaker audio is requested before the freeze tag exists."""


def freeze_tag_exists(tag: str, repo: Path = REPO_ROOT) -> bool:
    r = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def speakers(cfg: DictConfig, split: str) -> list[str]:
    """All L2 speakers of `split` ('dev' or 'test'), in config order."""
    return [s for group in cfg.l2[split].values() for s in group]


def accent_of(cfg: DictConfig) -> dict[str, tuple[str, str]]:
    """speaker -> (split, accent)."""
    return {s: (split, acc) for split in ("dev", "test")
            for acc, group in cfg.l2[split].items() for s in group}


def assert_audio_allowed(speaker: str, cfg: DictConfig, repo: Path = REPO_ROOT) -> None:
    if speaker.upper() in {s.upper() for s in speakers(cfg, "test")} and not freeze_tag_exists(
        cfg.freeze_tag, repo
    ):
        raise SplitLockedError(
            f"refusing to read audio of test speaker {speaker}: git tag {cfg.freeze_tag!r} not found"
        )


def assert_split_allowed(split: str, cfg: DictConfig, repo: Path = REPO_ROOT) -> None:
    """For entry points like `evaluate.py --split test`."""
    if split == "test" and not freeze_tag_exists(cfg.freeze_tag, repo):
        raise SplitLockedError(f"split 'test' is locked: git tag {cfg.freeze_tag!r} not found")
