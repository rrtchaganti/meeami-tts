"""`make check-data` (docs/PLAN.md Section 10 runbook step 1, T13): verify a training bundle.

    python -m tts.check_data [--root DIR] [--write-manifest] [--skip-hash]

--write-manifest records sha256 + size of every bundled file in data/BUNDLE_SHA256.txt (used by
`make bundle`). Without it: checks the recorded checksums and file counts, then every row of the
training manifests: audio exists, 22,050 Hz mono, frame count matches n_frames, and
sum(durations) == n_frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

BUNDLE_LIST = "data/BUNDLE_FILES.txt"
SHA_FILE = "data/BUNDLE_SHA256.txt"
TRAIN_MANIFESTS = ["lj_train", "lj_val", "lj_test", "aug_a_all", "aug_b_all", "aug_a", "aug_b"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root: Path, files: list[str]) -> None:
    with ThreadPoolExecutor(8) as ex:
        hashes = list(ex.map(lambda f: sha256(root / f), files))
    (root / SHA_FILE).write_text("".join(f"{h}  {f}\n" for h, f in zip(hashes, files)))


def check(root: Path, skip_hash: bool, sr: int = 22050, hop: int = 256) -> list[str]:
    errors = []
    sha = root / SHA_FILE
    if not sha.exists():
        return [f"{SHA_FILE} missing (not a bundle?)"]
    entries = [l.split("  ", 1) for l in sha.read_text().splitlines() if l.strip()]
    missing = [f for _, f in entries if not (root / f).exists()]
    if missing:
        errors.append(f"{len(missing)} bundled files missing, e.g. {missing[:3]}")
    if not skip_hash:
        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(lambda e: (e, sha256(root / e[1]) if (root / e[1]).exists() else None),
                              entries))
        bad = [e[1] for e, h in got if h is not None and h != e[0]]
        if bad:
            errors.append(f"{len(bad)} checksum mismatches, e.g. {bad[:3]}")
    print(f"files: {len(entries)} recorded, {len(missing)} missing"
          + ("" if skip_hash else ", checksums verified"))

    for m in TRAIN_MANIFESTS:
        p = root / "data" / "manifests" / f"{m}.jsonl"
        if not p.exists():
            print(f"manifest {m}: not present")
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        bad_sum = [r["id"] for r in rows if sum(r["durations"]) != r["n_frames"]]

        def audio_ok(r):
            try:
                info = sf.info(str(root / r["wav"]))
            except Exception:
                return f"{r['id']}: unreadable {r['wav']}"
            if info.samplerate != sr or info.channels != 1:
                return f"{r['id']}: {info.samplerate} Hz / {info.channels} ch"
            if info.frames // hop != r["n_frames"]:
                return f"{r['id']}: {info.frames // hop} frames != n_frames {r['n_frames']}"
            return None
        with ThreadPoolExecutor(16) as ex:
            bad_audio = [x for x in ex.map(audio_ok, rows) if x]
        print(f"manifest {m}: {len(rows)} rows; duration-sum errors {len(bad_sum)}; "
              f"audio errors {len(bad_audio)}")
        if bad_sum:
            errors.append(f"{m}: {len(bad_sum)} rows with sum(durations) != n_frames")
        if bad_audio:
            errors.append(f"{m}: {len(bad_audio)} audio problems, e.g. {bad_audio[:3]}")
    counts = {"lj_train": 12474, "lj_val": 100, "lj_test": 499}
    for m, n in counts.items():
        p = root / "data" / "manifests" / f"{m}.jsonl"
        if p.exists():
            k = sum(1 for l in p.read_text().splitlines() if l.strip())
            if k != n:
                errors.append(f"{m}: {k} rows, expected {n}")
    return errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--skip-hash", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if a.write_manifest:
        files = [l for l in (root / BUNDLE_LIST).read_text().splitlines() if l.strip()]
        write_manifest(root, files)
        print(f"wrote {SHA_FILE} for {len(files)} files")
        return 0
    errors = check(root, a.skip_hash)
    if errors:
        print("check-data: FAIL\n  " + "\n  ".join(errors))
        return 1
    print("check-data: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
