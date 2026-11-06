"""ASR for WER/CER (docs/PLAN.md Section 8): faster-whisper large-v3, Whisper's
EnglishTextNormalizer on both sides, jiwer for the metrics."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from omegaconf import DictConfig

from tts.audio import resample
from tts.eval.assets import hf_dir


@lru_cache(maxsize=1)
def _normalizer():
    from whisper.normalizers import EnglishTextNormalizer

    return EnglishTextNormalizer()


def normalize_for_wer(text: str) -> str:
    return " ".join(_normalizer()(text).split())


class ASR:
    def __init__(self, cfg: DictConfig, device: str = "cuda"):
        from faster_whisper import WhisperModel

        d = hf_dir(cfg, "asr")
        if not (d / ".complete").exists():
            raise FileNotFoundError(f"ASR model missing at {d}; run `make assets-eval`")
        spec = cfg.eval_models.asr
        ct = spec.compute_type if device == "cuda" else "int8"
        self.model = WhisperModel(str(d), device=device, compute_type=ct)
        self.beam = int(spec.beam_size)
        self.lang = spec.language

    def transcribe(self, y: np.ndarray, sr: int) -> str:
        y16 = resample(y.astype(np.float32), sr, 16000)
        segs, _ = self.model.transcribe(y16, language=self.lang, beam_size=self.beam,
                                        vad_filter=False, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()


def wer_cer(ref: str, hyp: str) -> tuple[float, float]:
    import jiwer

    r, h = normalize_for_wer(ref), normalize_for_wer(hyp)
    if not r:
        return (0.0 if not h else 1.0), (0.0 if not h else 1.0)
    return jiwer.wer(r, h), jiwer.cer(r, h)
