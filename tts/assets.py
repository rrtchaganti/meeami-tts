"""Pinned model/data assets: MFA pretrained models and NLTK corpora (docs/PLAN.md Section 3).

`python -m tts.assets --config configs/base.yaml` downloads anything missing and verifies sha256
against the pins in configs/base.yaml. A mismatch aborts (CLAUDE.md: never substitute silently).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import DictConfig

from tts.align.mfa import MFAError, model_path, run_in_mfa
from tts.config import REPO_ROOT, load_config, repo_path
from tts.splits import sha256


class AssetError(RuntimeError):
    pass


def ensure_mfa_models(cfg: DictConfig, root: Path = REPO_ROOT) -> dict[str, Path]:
    out = {}
    for kind in ("acoustic", "g2p"):
        spec = cfg.mfa[f"{kind}_model"]
        path = model_path(cfg, kind, root)
        if not path.exists() or sha256(path) != spec.sha256:
            r = run_in_mfa(cfg, ["mfa", "model", "download", kind, spec.name, "--version",
                                 spec.version, "--force"], root, timeout=3600)
            if r.returncode != 0:
                raise MFAError(f"mfa model download {kind} {spec.name} {spec.version} failed: "
                               f"{r.stderr[-2000:]}")
            got = sha256(path)
            if got != spec.sha256:
                raise AssetError(f"{kind} model {spec.name} {spec.version}: sha256 {got} != pinned "
                                 f"{spec.sha256}")
        out[kind] = path
    return out


def ensure_nltk_data(cfg: DictConfig, root: Path = REPO_ROOT) -> Path:
    import nltk

    d = repo_path(cfg.paths.nltk_data, root)
    for res in cfg.nltk.resources:
        path = d / res.path
        if not path.exists() or sha256(path) != res.sha256:
            if not nltk.download(res.id, download_dir=str(d), quiet=True):
                raise AssetError(f"nltk.download({res.id!r}) failed")
            got = sha256(path)
            if got != res.sha256:
                raise AssetError(f"NLTK {res.id}: sha256 {got} != pinned {res.sha256}")
    return d


def use_local_nltk(cfg: DictConfig, root: Path = REPO_ROOT) -> None:
    """Point NLTK at the project data and forbid downloads (g2p_en downloads on import)."""
    import nltk

    d = str(repo_path(cfg.paths.nltk_data, root))
    if d not in nltk.data.path:
        nltk.data.path.insert(0, d)

    def _no_download(*a, **k):
        raise AssetError(f"NLTK download attempted ({a}); run `make assets`")
    nltk.download = _no_download


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(repo_path(args.config))
    for kind, p in ensure_mfa_models(cfg).items():
        print(f"mfa {kind} model: {p.relative_to(REPO_ROOT)} (sha256 ok)")
    print(f"nltk data: {ensure_nltk_data(cfg).relative_to(REPO_ROOT)} (sha256 ok)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
