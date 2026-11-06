"""`make smoke` (T8): every training config for 200 steps at batch 4 (CLAUDE.md), a resume check
and a short benchmark. A/B use the unfiltered aug_*_all.jsonl when the UTMOS-filtered manifest
does not exist yet (logged; smoke only, never for real training)."""
from __future__ import annotations

import subprocess
import sys

from tts.config import REPO_ROOT

PY = [sys.executable, str(REPO_ROOT / "scripts" / "train.py")]
COMMON = ["--batch_size", "4", "train.ckpt_interval=100", "train.val_interval=200",
          "train.log_interval=50", "train.num_workers=4"]


def run(args: list[str]) -> None:
    print("$ train.py " + " ".join(args), flush=True)
    subprocess.run(PY + args, check=True, cwd=REPO_ROOT)


def main() -> int:
    for c in ["m1", "m1_pos", "abl_a", "abl_b"]:
        extra = []
        if c.startswith("abl_"):
            k = c[-1]
            filt = REPO_ROOT / "data" / "manifests" / f"aug_{k}.jsonl"
            if not filt.exists() or filt.stat().st_size == 0:
                print(f"smoke {c}: UTMOS-filtered {filt.name} missing; using aug_{k}_all.jsonl")
                extra = [f"train.aug_manifests=[data/manifests/aug_{k}_all.jsonl]"]
        print(f"=== smoke {c}", flush=True)
        run(["--config", f"configs/{c}.yaml", "--tag", f"smoke_{c}", "--max_steps", "200",
             *COMMON, *extra])
    print("=== resume check (smoke_m1: 200 -> 250)", flush=True)
    run(["--config", "configs/m1.yaml", "--tag", "smoke_m1", "--resume", "--max_steps", "250",
         *COMMON])
    print("=== benchmark", flush=True)
    run(["--config", "configs/m1.yaml", "--batch_size", "4", "--benchmark", "20",
         "train.num_workers=4"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
