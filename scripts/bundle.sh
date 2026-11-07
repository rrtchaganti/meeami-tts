#!/usr/bin/env bash
# Build the GPU training bundle (docs/PLAN.md Section 10, T13; `make bundle`).
#
#   bash scripts/bundle.sh [OUT]        # default OUT = export/bundle.tar
#   BUNDLE_AUG=0 bash scripts/bundle.sh  # light bundle (~3.8 GB): no augmented audio; regenerate
#                                        # on the instance with `make aug-a aug-b` (Section 10)
#   BUNDLE_AUG=0 BUNDLE_LJWAV=0 ...      # M1-only bundle (`make bundle-m1`): no LJ wavs either;
#                                        # the instance runs `make lj-download` (D-69)
#
# Contents: LJSpeech wavs referenced by the manifests + metadata, VITS filelists, lexicon,
# lj_* and aug_* manifests, augmented FLACs (data/aug/{a,b}), the pretrained VITS generator, and
# data/BUNDLE_SHA256.txt. No L2-ARCTIC audio (not needed for training). After writing the tar it is
# unpacked into a temp directory together with a fresh `git archive` of HEAD and
# `python -m tts.check_data` must pass there (T13 acceptance).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-export/bundle.tar}"
CONDA="${CONDA:-conda}"
mkdir -p "$(dirname "$OUT")"

BUNDLE_AUG="${BUNDLE_AUG:-1}" python3 - <<'PY'
import json, os, pathlib
root = pathlib.Path(".")
files = set()
man = root / "data/manifests"
names = ["lj_train", "lj_val", "lj_test"]
if os.environ["BUNDLE_AUG"] == "1":
    names += ["aug_a_all", "aug_b_all", "aug_a", "aug_b"]
else:  # aug regeneration on the instance needs the LJ stats and the dev pace stats (in git)
    print("bundle: light (no augmented audio)")
for name in names:
    p = man / f"{name}.jsonl"
    if p.exists():
        files.add(str(p))
        for line in p.read_text().splitlines():
            if line.strip():
                files.add(json.loads(line)["wav"])
extras = ["data/manifests/overfit.jsonl", "data/raw/LJSpeech-1.1/metadata.csv",
          "data/lexicon/lexicon.txt", "data/models/vits/pretrained_ljs.pth"]
if os.environ["BUNDLE_AUG"] == "1":
    extras += ["data/manifests/aug_a_scores.jsonl", "data/manifests/aug_b_scores.jsonl"]
for extra in extras:
    if (root / extra).exists():
        files.add(extra)
files |= {str(p) for p in (root / "data/filelists/vits").glob("*.txt")}
missing = [f for f in files if not (root / f).exists()]
if missing:
    raise SystemExit(f"bundle: {len(missing)} files missing, e.g. {missing[:3]}")
(root / "data/BUNDLE_FILES.txt").write_text("\n".join(sorted(files)) + "\n")
print(f"bundle: {len(files)} files")
PY

$CONDA run -n tts python -m tts.check_data --write-manifest
echo "data/BUNDLE_FILES.txt" > /tmp/bundle_meta_$$.txt
echo "data/BUNDLE_SHA256.txt" >> /tmp/bundle_meta_$$.txt
LIST=data/BUNDLE_FILES.txt
if [ "${BUNDLE_LJWAV:-1}" = "0" ]; then   # LJ wavs stay listed + hashed, but are not packed (D-69)
  grep -v '^data/raw/LJSpeech-1.1/wavs/' data/BUNDLE_FILES.txt > /tmp/bundle_list_$$.txt
  LIST=/tmp/bundle_list_$$.txt
  echo "bundle: LJ wavs not packed (instance downloads them: make lj-download)"
fi
tar -cf "$OUT" -T "$LIST" -T /tmp/bundle_meta_$$.txt
rm -f /tmp/bundle_list_$$.txt
rm -f /tmp/bundle_meta_$$.txt
sha256sum "$OUT" > "$OUT.sha256"
du -h "$OUT"

# acceptance: fresh unpack + check-data in a temp directory
TMP="$(mktemp -d)"
trap 'rm -rf -- "$TMP"' EXIT
git archive HEAD | tar -x -C "$TMP"
tar -xf "$OUT" -C "$TMP"
if [ "${BUNDLE_LJWAV:-1}" = "0" ]; then   # stand-in for the instance download
  ln -s "$PWD/data/raw/LJSpeech-1.1/wavs" "$TMP/data/raw/LJSpeech-1.1/wavs"
fi
(cd "$TMP" && $CONDA run -n tts python -m tts.check_data --root .)
echo "bundle OK: $OUT ($(cut -d' ' -f1 "$OUT.sha256"))"
