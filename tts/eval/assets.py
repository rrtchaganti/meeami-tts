"""Download / locate the pinned Section 8 evaluation models (`make assets-eval`).

HF models are snapshot-downloaded at the pinned revision into eval_models.root/<name>; UTMOS is
fetched through torch.hub at a pinned tag with TORCH_HOME inside the project. Nothing is
substituted: a missing model raises (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from omegaconf import DictConfig

from tts.config import REPO_ROOT, load_config, repo_path


def hf_dir(cfg: DictConfig, key: str, root: Path = REPO_ROOT) -> Path:
    spec = cfg.eval_models[key]
    return repo_path(cfg.eval_models.root, root) / spec.repo.replace("/", "__") / spec.revision


def ensure_hf(cfg: DictConfig, key: str, root: Path = REPO_ROOT) -> Path:
    from huggingface_hub import snapshot_download

    spec = cfg.eval_models[key]
    d = hf_dir(cfg, key, root)
    if not (d / ".complete").exists():
        snapshot_download(spec.repo, revision=spec.revision, local_dir=str(d))
        (d / ".complete").write_text(spec.revision + "\n")
    return d


def torch_home(cfg: DictConfig, root: Path = REPO_ROOT) -> Path:
    return repo_path(cfg.eval_models.root, root) / "torch_hub"


def load_utmos(cfg: DictConfig, root: Path = REPO_ROOT, device: str = "cpu"):
    import torch

    os.environ["TORCH_HOME"] = str(torch_home(cfg, root))
    torch.hub.set_dir(str(torch_home(cfg, root) / "hub"))
    spec = cfg.eval_models.utmos
    model = torch.hub.load(spec.hub_repo, spec.entry, trust_repo=True)
    return model.to(device).eval()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--only", nargs="*", default=["asr", "speaker", "accent", "utmos"])
    a = ap.parse_args(argv)
    cfg = load_config(repo_path(a.config))
    for k in a.only:
        if k == "utmos":
            load_utmos(cfg)
            print(f"utmos: {cfg.eval_models.utmos.hub_repo} {cfg.eval_models.utmos.entry} ok")
        else:
            print(f"{k}: {ensure_hf(cfg, k).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
