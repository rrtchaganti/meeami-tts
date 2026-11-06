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
        raise MFAError(f"acoustic model not found at {path}; run `make assets`")
    r = run_in_mfa(cfg, ["python", "-I", "-c", _PHONES_SRC, str(path)], root, timeout=600)
    if r.returncode != 0:
        raise MFAError(f"reading acoustic model phones failed: {r.stderr[-2000:]}")
    return set(json.loads(r.stdout.strip().splitlines()[-1]))


# --- alignment (T4) ----------------------------------------------------------------------------
_PARAMS_SRC = (
    "import json\n"
    "from montreal_forced_aligner.alignment.pretrained import PretrainedAligner as P\n"
    "print(json.dumps(sorted(P.get_configuration_parameters())))\n"
)
_PARAM_CACHE: dict[str, set[str]] = {}


def passthrough_names(cfg: DictConfig, root: Path = REPO_ROOT) -> set[str]:
    """Option names MFA accepts as pass-through config (everything else is silently dropped)."""
    key = cfg.mfa.conda_env
    if key not in _PARAM_CACHE:
        r = run_in_mfa(cfg, ["python", "-I", "-c", _PARAMS_SRC], root, timeout=600)
        if r.returncode != 0:
            raise MFAError(f"reading MFA parameters failed: {r.stderr[-2000:]}")
        _PARAM_CACHE[key] = set(json.loads(r.stdout.strip().splitlines()[-1]))
    return _PARAM_CACHE[key]


def passthrough_args(cfg: DictConfig, root: Path = REPO_ROOT) -> list[str]:
    """`--beam 100 --retry_beam 400` from cfg.mfa.passthrough; unknown names raise (D-11)."""
    names = passthrough_names(cfg, root)
    unknown = sorted(set(cfg.mfa.passthrough) - names)
    if unknown:
        raise MFAError(f"MFA would silently ignore {unknown}: not in its parameter list")
    return [x for k, v in cfg.mfa.passthrough.items() for x in (f"--{k}", str(v))]


def align_corpus(cfg: DictConfig, corpus: Path, out: Path, single_speaker: bool,
                 root: Path = REPO_ROOT, log: Path | None = None) -> subprocess.CompletedProcess:
    """Section 3: `mfa align <corpus> <lexicon> english_us_arpa <out> [--single_speaker] -j N
    --beam 100 --retry_beam 400 --clean`."""
    args = ["mfa", "align", str(corpus), str(repo_path(cfg.paths.lexicon, root)),
            cfg.mfa.acoustic_model.name, str(out)]
    if single_speaker:
        args.append("--single_speaker")
    args += ["-j", str(cfg.mfa.num_jobs), *passthrough_args(cfg, root), "--clean",
             "--temporary_directory", str(repo_path(cfg.paths.mfa_tmp, root))]
    r = run_in_mfa(cfg, args, root)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"$ {' '.join(args)}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}\n")
    return r


def align_one(cfg: DictConfig, wav: Path, lab: Path, out: Path,
              root: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    """Section 3: `mfa align_one ref.wav ref.lab <lexicon> english_us_arpa ref.TextGrid
    --beam 100 --retry_beam 400` (used from T5/T10)."""
    args = ["mfa", "align_one", str(wav), str(lab), str(repo_path(cfg.paths.lexicon, root)),
            cfg.mfa.acoustic_model.name, str(out), *passthrough_args(cfg, root),
            "--temporary_directory", str(repo_path(cfg.paths.mfa_tmp, root))]
    return run_in_mfa(cfg, args, root)
