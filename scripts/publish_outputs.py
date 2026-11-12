#!/usr/bin/env python
"""Every evaluated test-set output on the Hugging Face Hub (DECISIONS E-36).

    python scripts/publish_outputs.py              # `make outputs`: download + verify + unpack
    python scripts/publish_outputs.py --publish    # owner: tar results/<system>/<split>/ and upload

One uncompressed, deterministic tar per system (`hosting.outputs.members`, paths relative to
results/<system>/<split>/): out.wav + timings.json per utterance (wav/<id>/), the MFA TextGrid
measured on each output (align_out/), and the metric files. So every row of per_utt.csv can be
traced to the exact audio it was computed from. The ground-truth references are L2-ARCTIC
(CC BY-NC 4.0) and are not re-hosted; `reference` in each timings.json names the file.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from tts.config import REPO_ROOT, load_config, repo_path  # noqa: E402
from tts.splits import sha256  # noqa: E402


def repo_name(o, system: str) -> str:
    return f"outputs/{o.split}/{system}.tar"


def build_tar(src: Path, members, dst: Path) -> Path:
    """Deterministic tar: sorted paths, zero mtimes, no owner names."""
    files = []
    for m in members:
        p = src / m
        if p.is_dir():
            files += sorted(q for q in p.rglob("*") if q.is_file())
        elif p.is_file():
            files.append(p)
        else:
            raise SystemExit(f"[outputs] missing {p}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    def norm(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, 0, "", ""
        return ti
    with tarfile.open(dst, "w", format=tarfile.PAX_FORMAT) as tf:
        for f in files:
            tf.add(f, arcname=str(f.relative_to(src)), filter=norm)
    return dst


def publish(cfg) -> int:
    from huggingface_hub import CommitOperationAdd, HfApi

    h, o = cfg.hosting, cfg.hosting.outputs
    tars = {}
    for s in o.systems:
        t = build_tar(repo_path("results") / s / o.split, o.members,
                      repo_path(o.tar_dir) / o.split / f"{s}.tar")
        tars[repo_name(o, s)] = t
        print(f"[outputs] {t.relative_to(REPO_ROOT)} {t.stat().st_size / 1e6:.0f} MB")
    manifest = "".join(f"{sha256(p)}  {rp}\n" for rp, p in sorted(tars.items()))
    ops = [CommitOperationAdd(rp, str(p)) for rp, p in sorted(tars.items())]
    ops.append(CommitOperationAdd(f"outputs/{o.split}/MANIFEST.sha256", manifest.encode()))
    info = HfApi().create_commit(h.hf_repo, ops,
                                 commit_message=f"evaluated {o.split} outputs (E-36)")
    print(f"[outputs] {info.commit_url}\n[outputs] pin hosting.outputs.revision: {info.oid}")
    return 0


def fetch(cfg) -> int:
    from huggingface_hub import hf_hub_download

    from scripts.fetch_models import parse_manifest

    h, o = cfg.hosting, cfg.hosting.outputs
    if not o.revision:
        raise SystemExit("[outputs] hosting.outputs.revision is not pinned yet")
    man = parse_manifest(Path(hf_hub_download(h.hf_repo, f"outputs/{o.split}/MANIFEST.sha256",
                                              revision=o.revision)).read_text())
    for s in o.systems:
        rp = repo_name(o, s)
        dst = repo_path("results") / s / o.split
        if (dst / "wav").is_dir():
            print(f"[outputs] {s}: results/{s}/{o.split}/wav present, skipped")
            continue
        src = Path(hf_hub_download(h.hf_repo, rp, revision=o.revision))
        if sha256(src) != man[rp]:
            raise SystemExit(f"[outputs] {rp}: sha256 mismatch; stopping")
        with tarfile.open(src) as tf:
            tf.extractall(dst, filter="data")
        print(f"[outputs] {rp} -> results/{s}/{o.split}/")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    return publish(cfg) if a.publish else fetch(cfg)


if __name__ == "__main__":
    sys.exit(main())
