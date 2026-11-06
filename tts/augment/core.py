"""Pace-augmentation core shared by Ablations A and B (docs/PLAN.md Sections 6, 7).

Given an aligned LJ row and a plan (per-phone factors, sp lengthening, inserted pauses):
 1. one time-varying, pitch-preserving Rubber Band stretch (R3 "fine" engine, `--timemap`)
    anchored at the LJ phone boundaries; speech phones scaled by r_i, silences mapped 1:1;
 2. lengthened `sp` and inserted pauses are built by looping the utterance's own silence (room
    tone) with 10 ms crossfades, never digital zeros;
 3. durations from the target boundaries with the Section 3 cumulative rounding (and the >= 1
    frame rule, D-33); audio padded/trimmed to exactly n_frames * hop samples.
MFA is not re-run.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig

from tts.align.textgrid import _fill_zero_phones
from tts.text.symbols import SIL, SP

SILENCE = (SIL, SP)


@dataclass
class Plan:
    factors: list[float]                       # per token; 1.0 for sil
    sp_target: dict[int, float] = field(default_factory=dict)   # token idx -> target samples
    inserts: list[tuple[int, float]] = field(default_factory=list)  # (after token idx, samples)


def rng_for(seed: int, utt: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(utt.encode())])


def phone_sigmas(stats: dict, phones: list[str]) -> np.ndarray:
    ph = stats["phones"]
    return np.array([ph[p]["sigma"] if p in ph else 0.0 for p in phones])


def elastic_factors(z: float, sig: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.exp(z * sig), lo, hi)


def solve_z(durs: np.ndarray, sig: np.ndarray, target: float, lo: float, hi: float,
            z_range=(-5.0, 5.0), tol=1e-3) -> float:
    """Bisection for z with sum(d_i * clip(exp(z sigma_i))) = target (monotone in z)."""
    a, b = z_range

    def f(z):
        return float(np.sum(durs * elastic_factors(z, sig, lo, hi))) - target
    if f(a) >= 0:
        return a
    if f(b) <= 0:
        return b
    for _ in range(100):
        m = 0.5 * (a + b)
        if f(m) > 0:
            b = m
        else:
            a = m
        if b - a < tol:
            break
    return 0.5 * (a + b)


# --- room tone ---------------------------------------------------------------------------------
def _xfade_concat(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(a), len(b))
    if n == 0:
        return np.concatenate([a, b])
    t = np.linspace(0, math.pi / 2, n, dtype=np.float32)
    mid = a[-n:] * np.cos(t) + b[:n] * np.sin(t)  # equal-power
    return np.concatenate([a[:-n], mid, b[n:]])


def loop_tone(tone: np.ndarray, length: int, xfade: int) -> np.ndarray:
    """`length` samples made by looping `tone` with crossfades."""
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(tone) < 2 * xfade + 1:
        raise ValueError("room tone too short to loop")
    out = tone.copy()
    while len(out) < length:
        out = _xfade_concat(out, tone, xfade)
    return out[:length].astype(np.float32)


def room_tone(y: np.ndarray, bounds: np.ndarray, phones: list[str], min_len: int,
              win: int) -> np.ndarray:
    """Longest silence segment of the utterance; else its lowest-RMS window (never zeros)."""
    best = None
    for i, p in enumerate(phones):
        if p in SILENCE:
            s, e = int(bounds[i]), int(bounds[i + 1])
            if e - s >= min_len and (best is None or e - s > best[1] - best[0]):
                best = (s, e)
    if best is not None:
        return y[best[0]:best[1]].copy()
    hop = max(1, win // 4)
    starts = range(0, max(1, len(y) - win), hop)
    rms = [float(np.sqrt(np.mean(y[s:s + win] ** 2) + 1e-12)) for s in starts]
    s = list(starts)[int(np.argmin(rms))]
    return y[s:s + win].copy()


def lengthen(seg: np.ndarray, target: int, tone: np.ndarray, xf: int) -> np.ndarray:
    """Silence `seg` lengthened to `target` samples by looping itself (or the room tone if it is
    too short); starts with seg's first samples and ends with seg's last ones (no clicks)."""
    src = seg if len(seg) >= 2 * xf + 1 else tone
    if target <= len(seg):
        return seg
    head_len = target - len(seg) + xf
    looped = loop_tone(src, max(head_len, 2 * xf + 1), xf)[:head_len]
    if len(seg) >= xf:
        return _xfade_concat(looped, seg, xf)
    return np.concatenate([looped[: target - len(seg)], seg])


# --- render --------------------------------------------------------------------------------------
def render(y: np.ndarray, row: dict, plan: Plan, cfg: DictConfig, rbargs: dict) -> tuple:
    """Returns (audio, phones, durations, word_ids)."""
    import pyrubberband as pyrb

    a = cfg.augment
    sr, hop = int(cfg.audio.sample_rate), int(cfg.audio.hop_length)
    xf = int(round(a.crossfade_ms * sr / 1000))
    phones, durs, wids = list(row["phones"]), np.asarray(row["durations"]), list(row["word_ids"])
    src_b = np.concatenate([[0], np.cumsum(durs)]) * hop
    y = y[: int(src_b[-1])]
    if len(y) < src_b[-1]:
        y = np.pad(y, (0, int(src_b[-1]) - len(y)))
    tone = room_tone(y, src_b, phones, 2 * xf + 1, int(a.room_tone_window_ms * sr / 1000))

    # 1. speech stretch (silences 1:1)
    seg = np.diff(src_b).astype(np.float64)
    stretched = seg * np.where([p in SILENCE for p in phones], 1.0, plan.factors)
    mid_b = np.concatenate([[0.0], np.cumsum(stretched)])
    tmap = [(0, 0)]
    for k in range(1, len(src_b)):
        s, t = int(src_b[k]), int(round(mid_b[k]))
        if s > tmap[-1][0] and t > tmap[-1][1]:
            tmap.append((s, t))
    if tmap[-1][0] != len(y):
        tmap[-1] = (len(y), tmap[-1][1])
    ys = pyrb.timemap_stretch(y, sr, tmap, dict(rbargs)).astype(np.float32)
    total_mid = tmap[-1][1]
    ys = ys[:total_mid] if len(ys) >= total_mid else np.pad(ys, (0, total_mid - len(ys)))

    # 2. silences: lengthen sp by looping themselves; insert room-tone pauses
    inserts: dict[int, list[float]] = {}
    for j, L in plan.inserts:
        inserts.setdefault(j, []).append(L)
    pieces, out_ph, out_w, out_len, padded = [], [], [], [], []
    for k, p in enumerate(phones):
        s, e = int(round(mid_b[k])), int(round(mid_b[k + 1]))
        piece = ys[s:e]
        if p == SP and k in plan.sp_target and plan.sp_target[k] > len(piece):
            piece = lengthen(piece, int(round(plan.sp_target[k])), tone, xf)
        pieces.append(piece), out_ph.append(p), out_w.append(wids[k])
        out_len.append(len(piece)), padded.append(False)
        for L in inserts.get(k, []):
            n = int(round(L))
            pieces.append(loop_tone(tone, n + 2 * xf, xf)), out_ph.append(SP), out_w.append(-1)
            out_len.append(n), padded.append(True)
    # inserted pauses carry xf extra samples on each side, consumed by the two crossfades
    audio = np.zeros(0, dtype=np.float32)
    after_insert = False
    for piece, pad in zip(pieces, padded):
        if pad:
            audio = _xfade_concat(audio, piece, xf) if len(audio) >= xf else \
                np.concatenate([audio, piece[xf:]])
            after_insert = True
        elif after_insert and len(piece) >= xf:
            audio = _xfade_concat(audio, piece, xf)
            after_insert = False
        else:
            if after_insert:  # next piece too short to crossfade: drop the pad instead
                audio = audio[: len(audio) - xf]
                after_insert = False
            audio = np.concatenate([audio, piece])
    # 3. durations from target boundaries (cumulative rounding) and exact length
    tgt_b = np.concatenate([[0], np.cumsum(out_len)])
    fb = np.rint(tgt_b / hop).astype(int)
    d = list(np.diff(fb))
    _fill_zero_phones(out_ph, d)
    n_frames = int(sum(d))
    n = n_frames * hop
    audio = audio[:n] if len(audio) >= n else np.pad(audio, (0, n - len(audio)), mode="wrap")
    return np.clip(audio, -1.0, 1.0), out_ph, [int(x) for x in d], out_w
