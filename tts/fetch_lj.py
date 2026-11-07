"""Download the official LJSpeech 1.1 archive on the GPU instance (`make lj-download`, D-69).

Used with the M1-only bundle (`make bundle-m1`), which leaves the 3.6 GB of LJ wavs out because the
laptop upload is slow. The archive goes to data/raw/, is extracted to data/raw/LJSpeech-1.1/ and
deleted. Integrity is not trusted to the download: `make check-data` compares every wav with the
sha256 recorded on the laptop (data/BUNDLE_SHA256.txt).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

from tts.config import load_config, repo_path


def extract(archive: Path, dest: Path) -> Path:
    """Extract LJSpeech-1.1/ from the archive into dest (members outside it are refused)."""
    with tarfile.open(archive, "r:*") as tf:
        members = tf.getmembers()
        bad = [m.name for m in members if not (m.name == "LJSpeech-1.1"
               or m.name.startswith("LJSpeech-1.1/")) or ".." in Path(m.name).parts
               or m.issym() or m.islnk()]
        if bad:
            raise ValueError(f"unexpected archive members, e.g. {bad[:3]}")
        tf.extractall(dest, members=members)
    return dest / "LJSpeech-1.1"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--keep_archive", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(repo_path(a.config))
    lj = repo_path(cfg.paths.raw_lj)
    url = str(cfg.paths.lj_archive_url)
    archive = lj.parent / url.rsplit("/", 1)[-1]
    lj.parent.mkdir(parents=True, exist_ok=True)
    wavs = lj / "wavs"
    if wavs.is_dir() and sum(1 for _ in wavs.glob("*.wav")) >= 13100:
        print(f"[lj] {wavs} already has 13,100 wavs; nothing to do")
        return 0
    print(f"[lj] downloading {url}")
    subprocess.run(["wget", "-c", "-O", str(archive), url], check=True)
    print(f"[lj] extracting {archive.name}")
    extract(archive, lj.parent)
    if not a.keep_archive:
        archive.unlink()
    print(f"[lj] {sum(1 for _ in wavs.glob('*.wav'))} wavs in {wavs}. Next: make check-data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
