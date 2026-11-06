#!/usr/bin/env bash
# GPU instance setup (docs/PLAN.md Section 10 runbook step 1): the `tts` conda env from
# envs/tts.yml (MFA is not needed for training) and a CUDA check. Installs Miniconda into
# ~/miniconda3 if conda is missing. rubberband-cli is installed only with INSTALL_RUBBERBAND=1
# (needed only to regenerate augmentation on the instance: make aug-a aug-b).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v conda >/dev/null 2>&1; then
  if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
    echo "[setup] installing Miniconda to ~/miniconda3"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  fi
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
conda config --set solver libmamba >/dev/null 2>&1 || true

if conda env list | awk '{print $1}' | grep -qx tts; then
  conda env update -n tts -f envs/tts.yml --prune
else
  conda env create -n tts -f envs/tts.yml
fi

if [ "${INSTALL_RUBBERBAND:-0}" = "1" ]; then
  if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=""; fi
  $SUDO apt-get update -y && $SUDO apt-get install -y rubberband-cli ffmpeg sox
fi

mkdir -p logs runs export
conda run -n tts python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not visible in the tts env"
print("[setup] torch", torch.__version__, "CUDA", torch.version.cuda, torch.cuda.get_device_name(0),
      "bf16", torch.cuda.is_bf16_supported())
PY
echo "[setup] done. Next: bash scripts/fetch_bundle.sh <url|path> && make check-data"
