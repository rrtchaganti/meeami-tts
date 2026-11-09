"""TTS systems behind one interface for the inference pipeline (docs/PLAN.md Section 5).

`predict_log_durations(tokens)` -> native log durations (frames) for every token;
`synthesize(tokens, durations, noise_scale)` -> float32 audio of exactly sum(durations) * hop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from tts.config import repo_path
from tts.text.diph import merge_log_durations, split_tokens, to_ids_split
from tts.text.symbols import to_ids


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


class VitsSystem:
    """M1 / M1+pos / A / B: a VITS-MFA checkpoint from scripts/train.py."""

    name = "vits"

    def __init__(self, ckpt: str | Path, device: str = "cuda"):
        from tts.models.vits_mfa.models import build_generator

        ck = torch.load(repo_path(ckpt), map_location="cpu", weights_only=True)
        self.cfg = OmegaConf.create(ck["config"])
        self.device = torch.device(device)
        self.net = build_generator(self.cfg).to(self.device).eval()
        self.net.load_state_dict(ck["net_g"])
        self.net.dec.remove_weight_norm()
        self.hop = int(self.cfg.audio.hop_length)
        self.sample_rate = int(self.cfg.audio.sample_rate)
        self.step = int(ck.get("step", 0))
        self.diph = bool(self.cfg.model.get("diph_split", False))
        self.diph_frac = float(self.cfg.model.get("diph_first_frac", 0.5))

    def _x(self, tokens):
        ids = to_ids_split(tokens) if self.diph else to_ids(tokens)
        x = torch.tensor([ids], device=self.device)
        return x, torch.tensor([x.size(1)], device=self.device)

    def predict_log_durations(self, tokens: list[str]) -> np.ndarray:
        """Per input token; with diph_split the two parts are merged back (D-81)."""
        model_tokens, _, src = split_tokens(tokens) if self.diph else (tokens, None, None)
        x, n = self._x(model_tokens)
        logw, _ = self.net.predict_log_durations(x, n)
        _sync(self.device)
        logw = logw[0].float().cpu().numpy()
        return merge_log_durations(logw, src, len(tokens)) if self.diph else logw

    def synthesize(self, tokens, durations, noise_scale: float = 0.667) -> np.ndarray:
        if self.diph:
            tokens, durations, _ = split_tokens(tokens, durations, self.diph_frac)
        x, n = self._x(tokens)
        d = torch.tensor([durations], device=self.device)
        wav, _ = self.net.infer(x, n, d, noise_scale=noise_scale)
        _sync(self.device)
        y = wav[0, 0].float().cpu().numpy()
        n_samp = int(sum(durations)) * self.hop
        return y[:n_samp] if len(y) >= n_samp else np.pad(y, (0, n_samp - len(y)))


class M0System:
    name = "m0"

    def __init__(self, cfg: DictConfig, device: str = "cuda"):
        from tts.models.fastspeech2.m0 import M0

        self.m0 = M0(cfg, device)
        self.hop, self.sample_rate = M0.hop, M0.sample_rate
        self.device = device

    def predict_log_durations(self, tokens):
        return self.m0.predict_log_durations(tokens)

    def synthesize(self, tokens, durations, noise_scale: float = 0.667):
        y = self.m0.synthesize(tokens, durations)
        _sync(self.device)
        return y


def load_system(ckpt: str, cfg: DictConfig, device: str):
    """`--ckpt m0` selects M0; anything else is a VITS-MFA checkpoint path."""
    if ckpt == "m0":
        from tts.config import load_config, REPO_ROOT

        return M0System(load_config(REPO_ROOT / "configs" / "m0.yaml"), device)
    return VitsSystem(ckpt, device)
