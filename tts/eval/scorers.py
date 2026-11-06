"""Section 8 scorers: UTMOS (SpeechMOS utmos22_strong), ECAPA speaker embeddings
(speechbrain/spkrec-ecapa-voxceleb), CommonAccent accent ID (US-English probability).
Models are loaded from the pinned local snapshots (tts.eval.assets); none is substituted."""
from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig

from tts.audio import resample
from tts.eval.assets import hf_dir, load_utmos


class UTMOS:
    def __init__(self, cfg: DictConfig, device: str = "cuda"):
        self.model = load_utmos(cfg, device=device)
        self.device = device

    @torch.no_grad()
    def __call__(self, y: np.ndarray, sr: int) -> float:
        y16 = resample(y.astype(np.float32), sr, 16000)
        x = torch.from_numpy(y16).unsqueeze(0).to(self.device)
        return float(self.model(x, 16000).item())


def _speechbrain_model(cfg: DictConfig, key: str, device: str):
    from speechbrain.inference.classifiers import EncoderClassifier

    d = hf_dir(cfg, key)
    if not (d / ".complete").exists():
        raise FileNotFoundError(f"{cfg.eval_models[key].repo} missing at {d}; run make assets-eval")
    return EncoderClassifier.from_hparams(source=str(d), savedir=str(d / "_sb"),
                                          run_opts={"device": device})


class SpeakerEmbedder:
    def __init__(self, cfg: DictConfig, device: str = "cuda"):
        self.model = _speechbrain_model(cfg, "speaker", device)
        self.device = device

    @torch.no_grad()
    def __call__(self, y: np.ndarray, sr: int) -> np.ndarray:
        y16 = torch.from_numpy(resample(y.astype(np.float32), sr, 16000)).unsqueeze(0)
        e = self.model.encode_batch(y16.to(self.device)).squeeze().float().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class AccentID:
    def __init__(self, cfg: DictConfig, device: str = "cuda"):
        self.model = _speechbrain_model(cfg, "accent", device)
        self.us = cfg.eval_models.accent.us_label
        labels = self.model.hparams.label_encoder.ind2lab
        self.us_index = [i for i, l in labels.items() if l == self.us]
        if not self.us_index:
            raise ValueError(f"accent model has no {self.us!r} class: {sorted(labels.values())}")

    @torch.no_grad()
    def us_probability(self, y: np.ndarray, sr: int) -> float:
        y16 = torch.from_numpy(resample(y.astype(np.float32), sr, 16000)).unsqueeze(0)
        out = self.model.classify_batch(y16)
        logp = out[0].squeeze()
        p = torch.softmax(logp, -1) if logp.min() < 0 or logp.sum() > 1.001 else logp
        return float(p[self.us_index[0]])
