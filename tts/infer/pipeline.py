"""Inference pipeline (docs/PLAN.md Section 5): word timing and pauses from the reference,
native LJSpeech elasticity inside each word, exact output length.

Steps: 1 load reference (any format/rate, mono, no trim) -> 2 normalize, G2P + append OOVs,
16 kHz WAV + .lab -> 3 `mfa align_one` (fallback MMS_FA) -> 4 token timeline -> 5 native
durations -> 6 fit each word (elastic / proportional / phone_copy) -> 7 stretch cap -> 8 quantize
to round(N_ref * 22050 / sr_ref / 256) frames -> 9 synthesize, pad/trim to
round(N_ref * 22050 / sr_ref) samples -> 10 out.wav, timings.json, timing.png.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from omegaconf import DictConfig

from tts.align.mfa import align_one
from tts.align.textgrid import AlignmentError, _fill_zero_phones, parse, read_tiers
from tts.audio import resample, write_wav16
from tts.config import REPO_ROOT, repo_path
from tts.guard import assert_audio_allowed, speakers
from tts.text.lexicon import G2P, ensure_words, load_lexicon
from tts.text.normalize import Normalizer, words as norm_words
from tts.text.symbols import SIL, SP


class InferenceError(RuntimeError):
    pass


# --- step 1 --------------------------------------------------------------------------------------
def load_reference(path: str | Path, cfg: DictConfig) -> tuple[np.ndarray, int]:
    """Mono float32 and its native rate; any format libsndfile or ffmpeg can decode."""
    path = Path(path)
    l2 = repo_path(cfg.paths.raw_l2).resolve()
    try:
        rel = path.resolve().relative_to(l2)
        if rel.parts and rel.parts[0] in speakers(cfg, "test"):
            assert_audio_allowed(rel.parts[0], cfg)
    except ValueError:
        pass
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return y.mean(axis=1), int(sr)
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "ref.wav"
        r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
                            "-c:a", "pcm_f32le", str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            raise InferenceError(f"cannot decode {path.name}: {r.stderr.strip()[-200:]}")
        y, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
        return y.mean(axis=1), int(sr)


# --- steps 4-8 -----------------------------------------------------------------------------------
@dataclass
class Timeline:
    tokens: list[str]
    word_ids: list[int]
    frames: list[float]                         # per token (float until quantized)
    word_spans: list[tuple[int, int]] = field(default_factory=list)  # token index range per word
    capped: list[int] = field(default_factory=list)


def target_frames(n_ref: int, sr_ref: int, cfg: DictConfig) -> int:
    return int(round(n_ref * cfg.audio.sample_rate / sr_ref / cfg.audio.hop_length))


def target_samples(n_ref: int, sr_ref: int, cfg: DictConfig) -> int:
    return int(round(n_ref * cfg.audio.sample_rate / sr_ref))


def build_timeline(word_times: list[dict], word_phones: list[list[str]], total: int,
                   cfg: DictConfig) -> tuple[Timeline, list[tuple[int, int]]]:
    """Step 4. Returns the token skeleton and each word's [start, end) frame span."""
    fps = cfg.audio.sample_rate / cfg.audio.hop_length
    min_sp = int(cfg.align.min_sp_frames)
    spans = []
    prev_end = 0
    for w in word_times:
        s = min(max(int(round(w["start"] * fps)), prev_end), total)
        e = min(max(int(round(w["end"] * fps)), s), total)
        spans.append([s, e])
        prev_end = e
    tokens, wids, frames, wspans = [SIL], [-1], [float(spans[0][0])], []
    for i, ((s, e), ph) in enumerate(zip(spans, word_phones)):
        if i > 0:
            gap = s - spans[i - 1][1]
            if gap >= min_sp:
                tokens.append(SP), wids.append(-1), frames.append(float(gap))
            elif gap > 0:
                spans[i - 1][1] += gap  # short gap goes to the preceding word
        a = len(tokens)
        tokens += ph
        wids += [i] * len(ph)
        frames += [0.0] * len(ph)
        wspans.append((a, len(tokens)))
    tokens.append(SIL), wids.append(-1), frames.append(float(total - spans[-1][1]))
    return Timeline(tokens, wids, frames, wspans), [tuple(s) for s in spans]


def solve_word(logd: np.ndarray, sig: np.ndarray, D: float, z_range=(-5.0, 5.0),
               tol: float = 0.5) -> np.ndarray:
    """Step 6 (elastic): d_i = exp(log d_i + z sigma_i), one z by bisection, |sum - D| <= tol;
    then rescaled to hit D exactly (also covers z saturating at the bounds)."""
    def total(z):
        return float(np.exp(logd + z * sig).sum())
    a, b = z_range
    if total(a) >= D:
        z = a
    elif total(b) <= D:
        z = b
    else:
        z = 0.5 * (a + b)
        for _ in range(200):
            t = total(z)
            if abs(t - D) <= tol:
                break
            a, b = (a, z) if t > D else (z, b)
            z = 0.5 * (a + b)
    d = np.exp(logd + z * sig)
    return d * (D / d.sum())


def fit_words(tl: Timeline, spans, logd_hat: np.ndarray, sigma: np.ndarray, transfer: str,
              cap: float | None) -> Timeline:
    """Steps 6-7 for 'elastic' and 'proportional'. Inserts sp after capped words."""
    toks, wids, fr = list(tl.tokens), list(tl.word_ids), list(tl.frames)
    out_t, out_w, out_f, capped = [], [], [], []
    wi = 0
    i = 0
    n_words = len(tl.word_spans)
    while i < len(toks):
        if wi < n_words and i == tl.word_spans[wi][0]:
            a, b = tl.word_spans[wi]
            s, e = spans[wi]
            D = float(e - s)
            dhat = np.exp(logd_hat[a:b])
            extra = 0.0
            if cap is not None and D > cap * dhat.sum():
                extra = D - cap * dhat.sum()
                D = cap * dhat.sum()
                capped.append(wi)
            if transfer == "elastic":
                d = solve_word(logd_hat[a:b], sigma[a:b], D)
            elif transfer == "proportional":
                d = dhat * (D / dhat.sum())
            else:
                raise ValueError(transfer)
            out_t += toks[a:b]
            out_w += wids[a:b]
            out_f += d.tolist()
            if extra > 0:
                if b < len(toks) and toks[b] == SP:
                    fr[b] += extra
                else:
                    out_t.append(SP), out_w.append(-1), out_f.append(extra)
            i = b
            wi += 1
        else:
            out_t.append(toks[i]), out_w.append(wids[i]), out_f.append(fr[i])
            i += 1
    return Timeline(out_t, out_w, out_f, capped=capped)


def quantize(frames: list[float], tokens: list[str], total: int) -> list[int]:
    """Step 8: cumulative rounding to integer frames summing to `total`; >= 1 frame per phone."""
    cum = np.cumsum(frames)
    cum = cum * (total / cum[-1]) if cum[-1] > 0 else cum
    b = np.concatenate([[0], np.rint(cum).astype(int)])
    b[-1] = total
    d = np.diff(b).tolist()
    _fill_zero_phones(tokens, d)
    assert sum(d) == total and min(d) >= 0
    return [int(x) for x in d]


def word_times_from(tokens, word_ids, durs, words, fps) -> list[dict]:
    b = np.concatenate([[0], np.cumsum(durs)]) / fps
    out = {}
    for i, w in enumerate(word_ids):
        if w >= 0:
            if w not in out:
                out[w] = [b[i], b[i + 1]]
            out[w][1] = b[i + 1]
    return [{"word": words[w], "start": float(s), "end": float(e)} for w, (s, e) in sorted(out.items())]


# --- the pipeline ---------------------------------------------------------------------------------
class Pipeline:
    def __init__(self, cfg: DictConfig, system, root: Path = REPO_ROOT):
        self.cfg = cfg
        self.system = system
        self.root = root
        self.norm = Normalizer(cfg)
        self.lexicon_path = repo_path(cfg.paths.lexicon, root)
        st = json.loads((repo_path(cfg.paths.stats, root) / "phone_logdur.json").read_text())
        self.sigma = {p: v["sigma"] for p, v in st["phones"].items()}
        self._g2p = None

    def g2p(self, w):
        if self._g2p is None:
            self._g2p = G2P(self.cfg, self.root)
        return self._g2p(w)

    def align(self, y, sr, words, tmp: Path, aligner: str, n_frames: int,
              ref_textgrid: Path | None = None):
        """Step 3. Returns (word_times, parsed or None, aligner_used, reason).

        `ref_textgrid`: a TextGrid already produced by the same MFA model/lexicon/beam without
        speaker adaptation (batch pre-alignment, D-51); skips the align_one call."""
        if aligner == "mfa":
            wav16, lab, tg = tmp / "ref.wav", tmp / "ref.lab", tmp / "ref.TextGrid"
            if ref_textgrid is not None:
                tg = Path(ref_textgrid)
                r = None
            else:
                write_wav16(wav16, resample(y, sr, int(self.cfg.mfa.sample_rate)),
                            int(self.cfg.mfa.sample_rate))
                lab.write_text(" ".join(words) + "\n")
                r = align_one(self.cfg, wav16, lab, tg, self.root)
            reason = None
            if (r is not None and r.returncode != 0) or not tg.exists():
                reason = "mfa_failed"
            else:
                try:
                    wt, pt = read_tiers(tg)
                    p = parse(wt, pt, n_frames, self.cfg, words)
                    return ([{"word": w.word, "start": w.start, "end": w.end} for w in p.words],
                            p, "mfa", None)
                except AlignmentError as e:
                    reason = e.reason
            fallback = reason
        else:
            fallback = None
        from tts.eval.assets import torch_home
        from tts.infer.ctc import ctc_word_times

        dev = "cuda" if str(getattr(self.system, "device", "cpu")).startswith("cuda") else "cpu"
        wt = ctc_word_times(y, sr, words, str(torch_home(self.cfg, self.root)), dev,
                            float(self.cfg.eval.min_pause_s))
        return wt, None, "ctc", fallback

    def run(self, text: str, ref: str | Path, out_dir: str | Path, transfer: str | None = None,
            cap: float | None | str = "cfg", noise_scale: float | None = None,
            aligner: str | None = None, ref_textgrid: Path | None = None,
            plot: bool = True) -> dict:
        c = self.cfg.inference
        transfer = transfer or c.transfer
        cap = c.cap if cap == "cfg" else cap
        noise_scale = c.noise_scale if noise_scale is None else noise_scale
        aligner = aligner or c.aligner
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        lat = {}
        t0 = time.perf_counter()

        # 1-2
        y, sr = load_reference(ref, self.cfg)
        n_ref = len(y)
        normalized = self.norm(text)
        words = norm_words(normalized)
        if not words:
            raise InferenceError("text has no words after normalization")
        added = ensure_words(words, self.lexicon_path, self.g2p, self.cfg.lexicon.possessive_rule)
        lex = load_lexicon(self.lexicon_path)
        phones = [lex[w] for w in words]
        total = target_frames(n_ref, sr, self.cfg)
        n_out = target_samples(n_ref, sr, self.cfg)
        lat["normalize_g2p"] = time.perf_counter() - t0

        # 3
        t = time.perf_counter()
        with tempfile.TemporaryDirectory(dir=repo_path(self.cfg.paths.mfa_tmp, self.root)
                                         if repo_path(self.cfg.paths.mfa_tmp, self.root).exists()
                                         else None) as d:
            wt, parsed, used, fallback = self.align(y, sr, words, Path(d), aligner, total,
                                                    ref_textgrid)
        lat["align"] = time.perf_counter() - t

        # 4-8
        t = time.perf_counter()
        notes = []
        if transfer == "phone_copy" and parsed is None:
            notes.append("phone_copy needs MFA phone durations; used elastic (CTC fallback)")
            transfer_used = "elastic"
        else:
            transfer_used = transfer
        if transfer_used == "phone_copy":
            tokens, wids, durs = parsed.phones, parsed.word_ids, list(parsed.durations)
            capped = []
            logd = self.system.predict_log_durations(tokens)
        else:
            tl, spans = build_timeline(wt, phones, total, self.cfg)
            logd = self.system.predict_log_durations(tl.tokens)
            sig = np.array([self.sigma.get(p, 0.0) for p in tl.tokens])
            fitted = fit_words(tl, spans, logd, sig, transfer_used,
                               None if cap in (None, "none") else float(cap))
            tokens, wids = fitted.tokens, fitted.word_ids
            durs = quantize(fitted.frames, tokens, total)
            capped = fitted.capped
        lat["durations"] = time.perf_counter() - t

        speech = [i for i, w in enumerate(wids) if w >= 0]
        native = float(np.exp(np.asarray(logd)[speech]).sum()) if len(logd) == len(tokens) else None
        stretch = (float(sum(durs[i] for i in speech)) / native) if native else None

        # 9
        t = time.perf_counter()
        audio = self.system.synthesize(tokens, durs, noise_scale)
        lat["synthesis"] = time.perf_counter() - t
        audio = audio[:n_out] if len(audio) >= n_out else np.pad(audio, (0, n_out - len(audio)))
        out_sr = self.system.sample_rate

        # 10
        fps = self.cfg.audio.sample_rate / self.cfg.audio.hop_length
        out_words = word_times_from(tokens, wids, durs, words, fps)
        sf.write(out_dir / "out.wav", audio, out_sr, subtype="PCM_16")
        lat["total"] = time.perf_counter() - t0
        dur_out = n_out / out_sr
        info = {"text": text, "normalized": normalized, "reference": str(ref),
                "sr_ref": sr, "n_ref": n_ref, "ref_duration_s": n_ref / sr,
                "out_samples": int(len(audio)), "expected_samples": n_out,
                "length_error_samples": int(len(audio)) - n_out, "aligner": used,
                "mfa_fallback_reason": fallback, "transfer": transfer_used, "cap": cap,
                "noise_scale": noise_scale, "capped_words": [words[i] for i in capped],
                "stretch_ratio": stretch,
                "oov_added": added, "notes": notes, "ref_words": wt, "out_words": out_words,
                "tokens": tokens, "durations": durs,
                "latency_s": lat, "rtf_end_to_end": lat["total"] / dur_out,
                "rtf_model": (lat["durations"] + lat["synthesis"]) / dur_out}
        (out_dir / "timings.json").write_text(json.dumps(info, indent=1) + "\n")
        if plot:
            try:
                from tts.infer.plot import timing_plot

                timing_plot(y, sr, audio, out_sr, wt, out_words, out_dir / "timing.png")
            except Exception as e:  # plotting must never fail a synthesis
                info["notes"].append(f"timing.png failed: {e}")
        return info
