"""Calling MFA from the tts env (docs/PLAN.md Section 3; CLAUDE.md: always `conda run -n mfa`).

MFA_ROOT_DIR points at `paths.mfa_root` so pretrained models live inside the project (offline,
bundleable). Alignment commands are added in T4.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from omegaconf import DictConfig

from tts.config import REPO_ROOT, repo_path


class MFAError(RuntimeError):
    pass


def conda_exe() -> str:
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not exe:
        raise MFAError("conda executable not found (CONDA_EXE unset and conda not on PATH)")
    return exe


def mfa_env(cfg: DictConfig, root: Path = REPO_ROOT) -> dict[str, str]:
    return {**os.environ, "MFA_ROOT_DIR": str(repo_path(cfg.paths.mfa_root, root))}


def run_in_mfa(cfg: DictConfig, args: list[str], root: Path = REPO_ROOT,
               timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run `args` inside the mfa conda env with the project MFA_ROOT_DIR."""
    cmd = [conda_exe(), "run", "-n", cfg.mfa.conda_env, *args]
    return subprocess.run(cmd, capture_output=True, text=True, env=mfa_env(cfg, root),
                          cwd=root, timeout=timeout)


def model_path(cfg: DictConfig, kind: str, root: Path = REPO_ROOT) -> Path:
    """Downloaded model zip, e.g. kind='acoustic' -> data/models/mfa/pretrained_models/acoustic/X.zip"""
    name = cfg.mfa[f"{kind}_model"].name
    return repo_path(cfg.paths.mfa_root, root) / "pretrained_models" / kind / f"{name}.zip"


_PHONES_SRC = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "from montreal_forced_aligner.models import AcousticModel\n"
    "print(json.dumps(sorted(AcousticModel(Path(sys.argv[1])).meta['phones'])))\n"
)


def acoustic_model_phones(cfg: DictConfig, root: Path = REPO_ROOT) -> set[str]:
    """Phone set of the pretrained acoustic model (Section 3 'fail fast' check)."""
    path = model_path(cfg, "acoustic", root)
    if not path.exists():
        raise MFAError(f"acoustic model not found at {path}; run `make mfa-models`")
    r = run_in_mfa(cfg, ["python", "-I", "-c", _PHONES_SRC, str(path)], root, timeout=600)
    if r.returncode != 0:
        raise MFAError(f"reading acoustic model phones failed: {r.stderr[-2000:]}")
    return set(json.loads(r.stdout.strip().splitlines()[-1]))
