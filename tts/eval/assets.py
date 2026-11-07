"""Download / locate the pinned Section 8 evaluation models (`make assets-eval`).

HF models are snapshot-downloaded at the pinned revision into eval_models.root/<name>; UTMOS: model
code through torch.hub at a pinned tag (TORCH_HOME inside the project), weights from a
sha256-pinned file loaded with weights_only=True. Nothing is
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


class UtmosWeightsError(RuntimeError):
    pass


def utmos_weights(cfg: DictConfig, root: Path = REPO_ROOT, download: bool = True) -> Path:
    """Pinned UTMOS checkpoint: downloaded once from `weights_url`, sha256-checked every time.

    hubconf.py at the pinned tag fetches the release asset itself with check_hash=False and
    torch.hub.load_state_dict_from_url(weights_only=False) (a full pickle load), so the weights are
    fetched here instead and loaded with weights_only=True (D-66).
    """
    import urllib.request

    from tts.splits import sha256

    spec = cfg.eval_models.utmos
    path = repo_path(cfg.eval_models.root, root) / "utmos" / spec.weights_url.rsplit("/", 1)[-1]
    if not path.exists():
        if not download:
            raise UtmosWeightsError(f"UTMOS weights missing: {path} (run make assets-eval)")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(spec.weights_url, tmp)
        tmp.rename(path)
    got = sha256(path)
    if got != spec.weights_sha256:
        raise UtmosWeightsError(f"UTMOS weights {path.name}: sha256 {got} != pinned {spec.weights_sha256}")
    return path


def load_utmos(cfg: DictConfig, root: Path = REPO_ROOT, device: str = "cpu"):
    import torch

    os.environ["TORCH_HOME"] = str(torch_home(cfg, root))
    torch.hub.set_dir(str(torch_home(cfg, root) / "hub"))
    spec = cfg.eval_models.utmos
    w = utmos_weights(cfg, root)
    # model class only from the pinned tag; weights from the sha256-pinned file, tensors only
    model = torch.hub.load(spec.hub_repo, spec.entry, trust_repo=True, pretrained=False)
    model.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
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
