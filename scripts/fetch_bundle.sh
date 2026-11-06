#!/usr/bin/env bash
# Fetch and unpack the training bundle on the GPU instance (runbook step 1).
#
#   bash scripts/fetch_bundle.sh <url-or-path-to-bundle.tar> [sha256]
#
# If the sha256 is omitted, <bundle>.sha256 next to it (URL or path) is used when available.
# Alternative: rsync -avP export/bundle.tar* user@host:~/meeami-tts/  then run this with the path.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:?usage: fetch_bundle.sh <url-or-path> [sha256]}"
WANT="${2:-}"
mkdir -p export
DST="export/bundle.tar"
case "$SRC" in
  http://*|https://*)
    wget -c -O "$DST" "$SRC"
    if [ -z "$WANT" ]; then WANT="$(wget -qO- "$SRC.sha256" 2>/dev/null | cut -d' ' -f1 || true)"; fi ;;
  *)
    if [ "$(realpath "$SRC")" != "$(realpath -m "$DST")" ]; then cp "$SRC" "$DST"; fi
    if [ -z "$WANT" ] && [ -f "$SRC.sha256" ]; then WANT="$(cut -d' ' -f1 "$SRC.sha256")"; fi ;;
esac
if [ -n "$WANT" ]; then
  GOT="$(sha256sum "$DST" | cut -d' ' -f1)"
  [ "$GOT" = "$WANT" ] || { echo "bundle sha256 mismatch: $GOT != $WANT"; exit 1; }
  echo "[fetch] sha256 ok"
else
  echo "[fetch] WARNING: no sha256 given; per-file checksums are still verified by make check-data"
fi
tar -xf "$DST"
echo "[fetch] unpacked. Next: make check-data"
