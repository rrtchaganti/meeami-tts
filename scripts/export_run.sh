#!/usr/bin/env bash
# Export a finished run for download (runbook step 7).
#
#   bash scripts/export_run.sh <exp>     # e.g. m1 -> export/m1.tar.gz (+ .sha256)
#
# Packs runs/<exp>/{ckpt_last.pt, config.yaml, git_commit.txt, train.log, init_skipped_keys.txt,
# tb/} and results/profile_*.json, laid out as export/<exp>/... so that unpacking on the laptop
# (tar -xzf export/<exp>.tar.gz) gives export/<exp>/ckpt_last.pt as scripts/infer.py expects.
set -euo pipefail
cd "$(dirname "$0")/.."
EXP="${1:?usage: export_run.sh <exp>}"
RUN="runs/$EXP"
[ -f "$RUN/ckpt_last.pt" ] || { echo "no $RUN/ckpt_last.pt"; exit 1; }
STAGE="$(mktemp -d)"
trap 'rm -rf -- "$STAGE"' EXIT
mkdir -p "$STAGE/export/$EXP"
for f in ckpt_last.pt config.yaml git_commit.txt train.log init_skipped_keys.txt; do
  [ -e "$RUN/$f" ] && cp "$RUN/$f" "$STAGE/export/$EXP/"
done
[ -d "$RUN/tb" ] && cp -r "$RUN/tb" "$STAGE/export/$EXP/"
ls results/profile_*.json >/dev/null 2>&1 && cp results/profile_*.json "$STAGE/export/$EXP/"
mkdir -p export
tar -czf "export/$EXP.tar.gz" -C "$STAGE" "export/$EXP"
sha256sum "export/$EXP.tar.gz" > "export/$EXP.tar.gz.sha256"
du -h "export/$EXP.tar.gz"
cat "export/$EXP.tar.gz.sha256"
