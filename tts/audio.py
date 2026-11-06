"""Audio I/O and the training spectrogram (docs/PLAN.md Sections 2-4).

`spectrogram` mirrors jaywalnut310/vits `mel_processing.spectrogram_torch`: reflect-pad by
(n_fft - hop) / 2 on both sides, `center=False`, magnitude with +1e-6 inside the sqrt. Section 3
requires the target frame count to come from this function itself, so `n_frames` calls it.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from omegaconf import DictConfig


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono float32 in [-1, 1] (same scale as VITS's int16 / 32768) and the file's sample rate.

    Multi-channel files are averaged to mono; nothing is trimmed (Section 5 step 1).
    """
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y.mean(axis=1), sr


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y
    return soxr.resample(y, sr_in, sr_out, quality="HQ").astype(np.float32)


def write_wav16(path: str | Path, y: np.ndarray, sr: int) -> None:
    """16-bit PCM WAV (MFA input, Section 2). Clips to [-1, 1) before quantization."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(y, -1.0, 32767 / 32768), sr, subtype="PCM_16")


@lru_cache(maxsize=4)
def _hann(win: int) -> torch.Tensor:
    return torch.hann_window(win)


def spectrogram(y: torch.Tensor, audio: DictConfig) -> torch.Tensor:
    """Linear magnitude spectrogram [..., n_fft/2+1, frames] (fp32), as in VITS."""
    n_fft, hop, win = int(audio.filter_length), int(audio.hop_length), int(audio.win_length)
    y = y.float()
    squeeze = y.dim() == 1
    if squeeze:
        y = y.unsqueeze(0)
    pad = (n_fft - hop) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop, win_length=win, window=_hann(win).to(y.device),
                      center=False, pad_mode="reflect", normalized=False, onesided=True,
                      return_complex=True)
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
    return spec[0] if squeeze else spec


def n_frames(y: np.ndarray | torch.Tensor, audio: DictConfig) -> int:
    """Number of training-spectrogram frames for waveform `y` (from `spectrogram` itself)."""
    t = torch.as_tensor(y)
    return int(spectrogram(t, audio).shape[-1])
