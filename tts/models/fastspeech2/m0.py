"""M0: pretrained FastSpeech2 + HiFi-GAN with injected durations (docs/PLAN.md Section 4).

ming024/FastSpeech2 LJSpeech 900k checkpoint and its bundled HiFi-GAN LJSpeech generator (same
22,050 Hz / hop 256 frame rate as ours). Our tokens map to its symbols ("@" + ARPAbet; sil ->
"@sil", sp -> "@sp"); any unmapped symbol raises. Reference durations are passed as duration
targets to the variance adaptor (its training path); pitch and energy stay predicted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import DictConfig

from tts.config import REPO_ROOT, repo_path
from tts.text.symbols import SIL, SP

from .fastspeech2 import FastSpeech2
from .hifigan.models import Generator
from .symbols import SYMBOL_TO_ID

HERE = Path(__file__).resolve().parent


def map_tokens(tokens: list[str]) -> list[int]:
    out = []
    for t in tokens:
        s = "@sil" if t == SIL else "@sp" if t == SP else "@" + t
        if s not in SYMBOL_TO_ID:
            raise KeyError(f"token {t!r} has no FastSpeech2 symbol ({s!r})")
        out.append(SYMBOL_TO_ID[s])
    return out


def _safe_load(path: Path):
    """weights_only load; allowlists only the numpy scalar/dtype globals these 2021 checkpoints
    contain (no arbitrary code execution)."""
    from numpy._core import multiarray as ma  # numpy 2 name; pickles use numpy.core.*

    allowed = [(ma.scalar, "numpy.core.multiarray.scalar"), np.dtype] + \
        [type(np.dtype(t)) for t in ("float64", "int64", "float32")]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


class _AttrDict(dict):
    def __getattr__(self, k):
        return self[k]


class M0:
    hop = 256
    sample_rate = 22050

    def __init__(self, cfg: DictConfig, device: str = "cuda", root: Path = REPO_ROOT):
        m = cfg.m0
        pre = yaml.safe_load((HERE / "config/LJSpeech/preprocess.yaml").read_text())
        mod = yaml.safe_load((HERE / "config/LJSpeech/model.yaml").read_text())
        pre["path"]["preprocessed_path"] = str(HERE / "config/LJSpeech")  # stats.json lives here
        assert pre["preprocessing"]["audio"]["sampling_rate"] == self.sample_rate
        assert pre["preprocessing"]["stft"]["hop_length"] == self.hop
        self.device = torch.device(device)
        self.model = FastSpeech2(pre, mod).to(self.device).eval()
        ck = _safe_load(repo_path(m.fastspeech2_ckpt, root))
        self.model.load_state_dict(ck["model"])
        hcfg = _AttrDict(json.loads((HERE / "hifigan/config.json").read_text()))
        self.vocoder = Generator(hcfg).to(self.device).eval()
        vk = _safe_load(repo_path(m.hifigan_ckpt, root))
        self.vocoder.load_state_dict(vk["generator"])
        self.vocoder.remove_weight_norm()
        self.p_control = float(m.pitch_control)
        self.e_control = float(m.energy_control)

    @torch.no_grad()
    def synthesize(self, tokens: list[str], durations: list[int]) -> np.ndarray:
        """Audio (float32, 22.05 kHz) of exactly sum(durations) * 256 samples."""
        ids = torch.tensor([map_tokens(tokens)], device=self.device)
        d = torch.tensor([durations], device=self.device, dtype=torch.long)
        n = torch.tensor([ids.size(1)], device=self.device)
        total = int(sum(durations))
        out = self.model(torch.zeros(1, dtype=torch.long, device=self.device), ids, n, ids.size(1),
                         d_targets=d, p_control=self.p_control, e_control=self.e_control,
                         max_mel_len=total, mel_lens=torch.tensor([total], device=self.device))
        mel = out[1]  # postnet output [1, T, 80]
        wav = self.vocoder(mel.transpose(1, 2)).squeeze().float().cpu().numpy()
        n_samp = total * self.hop
        wav = wav[:n_samp] if len(wav) >= n_samp else np.pad(wav, (0, n_samp - len(wav)))
        return wav.astype(np.float32)

    def predict_log_durations(self, tokens: list[str]) -> np.ndarray:
        """FastSpeech2's own log(d + 1) duration prediction, converted to log frames."""
        ids = torch.tensor([map_tokens(tokens)], device=self.device)
        n = torch.tensor([ids.size(1)], device=self.device)
        with torch.no_grad():
            out = self.model(torch.zeros(1, dtype=torch.long, device=self.device), ids, n, ids.size(1))
        log_d1 = out[4][0].float().cpu().numpy()  # log(d + 1)
        return np.log(np.clip(np.exp(log_d1) - 1.0, 1.0, None))
