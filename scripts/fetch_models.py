#!/usr/bin/env python
"""Hosted checkpoints on the Hugging Face Hub (DECISIONS D-95).

    python scripts/fetch_models.py                 # `make models`: download what is missing
    python scripts/fetch_models.py --publish       # owner only: upload export/*/infer.pt + baselines

Fetch: every file listed in the repo's MANIFEST.sha256 at `hosting.revision` is downloaded to its
local path (vits_mfa/<name>/infer.pt -> export/<name>/infer.pt; baselines -> data/models/...;
lexicon/lexicon.txt -> data/lexicon/lexicon.txt, only if missing) and must match its sha256, otherwise the run stops (CLAUDE.md: never substitute silently). Files that
already exist with the right hash are skipped. A private repo needs `huggingface-cli login` or
HF_TOKEN.

Publish: creates the repo (private per `hosting.private`), writes MANIFEST.sha256 and uploads
everything in one commit, with README.md / LICENSES.md from docs/hf_card/. Prints the commit hash to
pin as `hosting.revision`.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from tts.config import REPO_ROOT, load_config, repo_path  # noqa: E402
from tts.splits import sha256  # noqa: E402


def file_map(cfg) -> dict[str, Path]:
    """repo path -> local path."""
    m = {f"vits_mfa/{n}/infer.pt": repo_path("export") / n / "infer.pt" for n in cfg.hosting.systems}
    m |= {f"vits_mfa/{n}/config.yaml": repo_path("export") / n / "config.yaml"
          for n in cfg.hosting.systems}
    m |= {k: repo_path(v) for k, v in cfg.hosting.baselines.items()}
    m |= {k: repo_path(v) for k, v in cfg.hosting.get("if_missing", {}).items()}
    return m


def parse_manifest(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if line.strip():
            h, p = line.split(maxsplit=1)
            out[p.strip()] = h
    return out


def fetch(cfg) -> int:
    from huggingface_hub import hf_hub_download

    h = cfg.hosting
    man = parse_manifest(Path(hf_hub_download(h.hf_repo, "MANIFEST.sha256",
                                              revision=h.revision)).read_text())
    local = file_map(cfg)
    keep = set(cfg.hosting.get("if_missing", {}))
    for rp, digest in man.items():
        dst = local.get(rp)
        if dst is None:
            print(f"[models] {rp}: not used by this config, skipped")
            continue
        if dst.exists() and sha256(dst) == digest:
            print(f"[models] {rp}: present")
            continue
        if dst.exists() and rp in keep:
            print(f"[models] {rp}: local copy differs from the hosted one (e.g. words added at "
                  "inference); kept")
            continue
        src = Path(hf_hub_download(h.hf_repo, rp, revision=h.revision))
        got = sha256(src)
        if got != digest:
            raise SystemExit(f"[models] {rp}: sha256 {got} != manifest {digest}; stopping")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"[models] {rp} -> {dst.relative_to(REPO_ROOT)}")
    return 0


def publish(cfg, card_dir: Path) -> int:
    from huggingface_hub import CommitOperationAdd, HfApi

    h = cfg.hosting
    files = file_map(cfg)
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"[publish] missing: {missing} (run make export-infer first)")
    manifest = "".join(f"{sha256(p)}  {rp}\n" for rp, p in sorted(files.items())
                       if not rp.endswith(".yaml"))
    api = HfApi()
    api.create_repo(h.hf_repo, repo_type="model", private=bool(h.private), exist_ok=True)
    ops = [CommitOperationAdd(rp, str(p)) for rp, p in sorted(files.items())]
    ops.append(CommitOperationAdd("MANIFEST.sha256", manifest.encode()))
    for name in ("README.md", "LICENSES.md"):
        ops.append(CommitOperationAdd(name, str(card_dir / name)))
    info = api.create_commit(h.hf_repo, ops, commit_message="meeami-tts checkpoints (D-95)")
    print(f"[publish] {info.commit_url}\n[publish] pin hosting.revision: {info.oid}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--card_dir", type=Path, default=REPO_ROOT / "docs" / "hf_card",
                    help="README.md + LICENSES.md for the repo")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.publish:
        return publish(cfg, a.card_dir)
    return fetch(cfg)


if __name__ == "__main__":
    sys.exit(main())
