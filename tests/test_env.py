"""T1 environment checks (docs/PLAN.md Sections 3, 8, 10).

Run inside the `tts` env (`make test`). MFA is reached only via `conda run -n mfa`.
Nothing here downloads data or models.
"""
import importlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TTS_YML = ROOT / "envs" / "tts.yml"

# conda/pip distribution name -> import name. Every package in envs/tts.yml must appear here
# or in NOT_IMPORTABLE, so a package added to the env without a test fails loudly.
IMPORT_NAME = {
    "pytorch": "torch",
    "torchaudio": "torchaudio",
    "numpy": "numpy",
    "scipy": "scipy",
    "librosa": "librosa",
    "pysoundfile": "soundfile",
    "cmudict": "cmudict",
    "inflect": "inflect",
    "unidecode": "unidecode",
    "nltk": "nltk",
    "praatio": "praatio",
    "tensorboard": "tensorboard",
    "omegaconf": "omegaconf",
    "pyyaml": "yaml",
    "tqdm": "tqdm",
    "openai-whisper": "whisper",
    "jiwer": "jiwer",
    "huggingface_hub": "huggingface_hub",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "ptflops": "ptflops",
    "pytest": "pytest",
    "g2p_en": "g2p_en",
    "pyrubberband": "pyrubberband",
    "faster-whisper": "faster_whisper",
    "speechbrain": "speechbrain",
    "ctranslate2": "ctranslate2",
}
# Entries that are not Python packages; ffmpeg and cudnn are checked separately below.
NOT_IMPORTABLE = {"python", "pip", "cuda-version", "ffmpeg", "cudnn"}


def _dist_name(spec: str) -> str:
    return re.split(r"[<>=!~ \[]", spec.strip(), maxsplit=1)[0].lower()


def _tts_packages() -> list[str]:
    deps = yaml.safe_load(TTS_YML.read_text())["dependencies"]
    names = []
    for d in deps:
        if isinstance(d, dict):
            names += [_dist_name(p) for p in d.get("pip", [])]
        else:
            names.append(_dist_name(d))
    return [n for n in names if n not in NOT_IMPORTABLE]


@pytest.mark.parametrize("dist", _tts_packages())
def test_import(dist, monkeypatch):
    assert dist in IMPORT_NAME, f"{dist} is in envs/tts.yml but has no import mapping in test_env.py"
    if dist == "g2p_en":
        # g2p_en calls nltk.download() at import if its corpora are missing; T1 must not download.
        import nltk

        monkeypatch.setattr(nltk, "download", lambda *a, **k: False)
    importlib.import_module(IMPORT_NAME[dist])


def test_ctranslate2_installed():
    # faster-whisper's backend; not listed in the yml, pulled in by faster-whisper.
    importlib.import_module("ctranslate2")


def test_ffmpeg_binary():
    assert shutil.which("ffmpeg"), "ffmpeg not on PATH in the tts env"


# --- GPU -----------------------------------------------------------------------------------
def test_cuda_visible():
    import torch

    assert torch.cuda.is_available(), "CUDA not visible to torch in the tts env"
    x = torch.ones(4, device="cuda")
    assert float((x * 2).sum().cpu()) == 8.0


def test_bf16_supported():
    import torch

    assert torch.cuda.is_bf16_supported(), "bf16 autocast (PLAN Section 4) needs bf16 support"


def test_cudnn9():
    # torch and ctranslate2 (faster-whisper) must share cuDNN 9 (docs/DECISIONS.md D-4).
    import torch

    assert torch.backends.cudnn.is_available()
    assert torch.backends.cudnn.version() // 10000 == 9, torch.backends.cudnn.version()


def test_ctranslate2_cuda():
    import ctranslate2

    assert ctranslate2.get_cuda_device_count() > 0, "faster-whisper (ctranslate2) cannot see the GPU"


def test_mms_fa_available():
    # Attribute check only; does not download the MMS weights.
    import torchaudio

    assert hasattr(torchaudio.pipelines, "MMS_FA")
    assert hasattr(torchaudio.functional, "forced_align")


# --- Rubber Band (PLAN Section 6) ----------------------------------------------------------
def test_rubberband():
    exe = shutil.which("rubberband")
    assert exe, "rubberband-cli not on PATH"
    out = subprocess.run([exe, "--version"], capture_output=True, text=True)
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", out.stdout + out.stderr)
    assert m, f"cannot parse rubberband version: {out.stdout!r} {out.stderr!r}"
    assert (int(m.group(1)), int(m.group(2))) >= (3, 0), f"rubberband {m.group(0)} < 3.0"
    importlib.import_module("pyrubberband")


# --- MFA (PLAN Section 3) ------------------------------------------------------------------
def _conda() -> str:
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    assert exe, "conda executable not found (CONDA_EXE unset and conda not on PATH)"
    return exe


def _mfa(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_conda(), "run", "-n", "mfa", "mfa", *args], capture_output=True, text=True, timeout=300
    )


def test_mfa_version():
    r = _mfa("version")
    assert r.returncode == 0, f"mfa version failed: {r.stderr}"
    assert re.search(r"\b3\.\d+\.\d+", r.stdout), f"expected MFA 3.x, got {r.stdout!r}"


# Section 3 flags that MFA declares as click options (must appear in --help).
MFA_FLAGS = {
    "align": ["--single_speaker", "--clean", "-j"],
    "align_one": [],
}
# Section 3 decoder settings. MFA 3.4.3 does not list these in --help: they are pass-through
# config options (click ignore_unknown_options), and MFA silently DROPS unknown names, so they are
# checked by asking MFA's own parser instead (docs/DECISIONS.md E-2, D-11).
MFA_PASSTHROUGH = {"beam": 100, "retry_beam": 400}
# Positional signatures used in Section 3.
MFA_USAGE = {
    "align": ["CORPUS_DIRECTORY", "DICTIONARY_PATH", "ACOUSTIC_MODEL_PATH", "OUTPUT_DIRECTORY"],
    "align_one": ["SOUND_FILE_PATH", "TEXT_FILE_PATH", "DICTIONARY_PATH", "ACOUSTIC_MODEL_PATH",
                  "OUTPUT_PATH"],
}


def _mfa_help(cmd: str) -> str:
    # Wide terminal so rich-click does not wrap option names across lines.
    env = {**os.environ, "COLUMNS": "200", "TERM": "dumb"}
    r = subprocess.run([_conda(), "run", "-n", "mfa", "mfa", cmd, "--help"],
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, f"mfa {cmd} --help failed: {r.stderr}"
    return r.stdout


@pytest.mark.parametrize("cmd", sorted(MFA_USAGE))
def test_mfa_command_form(cmd):
    text = _mfa_help(cmd)
    usage = re.search(r"Usage:.*?(?:\n\s*\n|$)", text, re.S)
    assert usage, f"no usage line in mfa {cmd} --help"
    usage_line = " ".join(usage.group(0).split())
    for arg in MFA_USAGE[cmd]:
        assert arg in usage_line, f"mfa {cmd}: positional {arg} not in usage: {usage_line}"
    assert "deprecat" not in text.lower(), f"mfa {cmd} --help mentions deprecation:\n{text}"
    for flag in MFA_FLAGS[cmd]:
        assert re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text), (
            f"mfa {cmd} --help does not list {flag}"
        )


@pytest.mark.parametrize("cmd", sorted(MFA_USAGE))
def test_mfa_passthrough_flags(cmd):
    # Both commands must accept extra args, and MFA's parser must map our flags to the values.
    src = (
        "import json, sys\n"
        "from montreal_forced_aligner.command_line.mfa import mfa_cli\n"
        "from montreal_forced_aligner.alignment.pretrained import PretrainedAligner as P\n"
        f"c = mfa_cli.commands[{cmd!r}]\n"
        "cs = c.context_settings\n"
        "assert cs.get('ignore_unknown_options') and cs.get('allow_extra_args'), cs\n"
        "args = sys.argv[1:]\n"
        "print(json.dumps({k: v for k, v in P.parse_args(None, args).items() if 'beam' in k}))\n"
    )
    argv = [x for k, v in MFA_PASSTHROUGH.items() for x in (f"--{k}", str(v))]
    r = subprocess.run([_conda(), "run", "-n", "mfa", "python", "-I", "-c", src, *argv],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"MFA parser check failed for {cmd}: {r.stderr}"
    parsed = json.loads(r.stdout.strip().splitlines()[-1])
    assert parsed == MFA_PASSTHROUGH, f"mfa {cmd}: parsed {parsed}, expected {MFA_PASSTHROUGH}"
