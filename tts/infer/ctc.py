"""CTC word aligner: torchaudio MMS_FA (docs/PLAN.md Section 3 "Fallback aligner", Section 8).

Used when MFA fails / returns spn / leaves a word without an interval, and as the independent
second aligner for evaluation. The MMS_FA acoustic model (wav2vec2, torchaudio bundle) gives the
frame-level emissions; the alignment is CTC Viterbi forced alignment. conda-forge's torchaudio
2.8.0 is built without the `forced_align` C++ extension (E-15), so the identical algorithm
(Viterbi over the blank-interleaved target, transitions stay / advance / skip a blank between
distinct tokens) is implemented here in numpy (D-56). Returns word (start, end) in seconds.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

from tts.audio import resample


def forced_align(log_probs: np.ndarray, targets: list[int], blank: int = 0) -> np.ndarray:
    """CTC Viterbi alignment. log_probs [T, C]; returns, per frame, the index in `targets` it is
    aligned to, or -1 for blank. Raises if the targets cannot fit into T frames."""
    T = log_probs.shape[0]
    L = len(targets)
    ext = np.full(2 * L + 1, blank, dtype=np.int64)
    ext[1::2] = targets
    S = ext.size
    repeats = sum(1 for a, b in zip(targets, targets[1:]) if a == b)
    if T < L + repeats:
        raise ValueError(f"{L} tokens (+{repeats} repeats) do not fit into {T} frames")
    neg = -np.inf
    dp = np.full((T, S), neg)
    bp = np.zeros((T, S), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    dp[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        dp[0, 1] = log_probs[0, ext[1]]
    can_skip = np.zeros(S, dtype=bool)
    can_skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    for t in range(1, T):
        prev = dp[t - 1]
        stay = prev
        step = np.concatenate([[neg], prev[:-1]])
        skip = np.where(can_skip, np.concatenate([[neg, neg], prev[:-2]]), neg)
        cand = np.stack([stay, step, skip])
        arg = cand.argmax(0)
        dp[t] = cand[arg, np.arange(S)] + log_probs[t, ext]
        bp[t] = arg
    s = S - 1 if S == 1 or dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    if not np.isfinite(dp[T - 1, s]):
        raise ValueError("no valid CTC path")
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return np.where(path % 2 == 1, path // 2, -1)


@lru_cache(maxsize=2)
def _bundle(device: str, torch_home: str):
    os.environ["TORCH_HOME"] = torch_home
    torch.hub.set_dir(os.path.join(torch_home, "hub"))
    import torchaudio

    b = torchaudio.pipelines.MMS_FA
    return b, b.get_model(with_star=False).to(device).eval(), b.get_dict()


def ctc_word_times(y: np.ndarray, sr: int, words: list[str], torch_home: str,
                   device: str = "cpu", merge_gap_s: float = 0.1) -> list[dict]:
    """`merge_gap_s`: gaps shorter than this (not a pause by the Section 8 definition) are split
    at their midpoint; CTC's spiky emissions otherwise end words too early (D-57)."""
    b, model, dictionary = _bundle(device, torch_home)
    y16 = resample(y.astype(np.float32), sr, b.sample_rate)
    wav = torch.from_numpy(y16).unsqueeze(0).to(device)
    with torch.inference_mode():
        emission, _ = model(wav)
    lp = torch.log_softmax(emission[0].float(), -1).cpu().numpy()
    targets, owner = [], []
    for wi, w in enumerate(words):
        chars = [c for c in w.lower() if c in dictionary] or ["a"]
        targets += [dictionary[c] for c in chars]
        owner += [wi] * len(chars)
    path = forced_align(lp, targets, blank=dictionary.get("-", 0))
    sec = wav.size(1) / lp.shape[0] / b.sample_rate  # seconds per emission frame
    first = np.full(len(words), -1)
    last = np.full(len(words), -1)
    for t, k in enumerate(path):
        if k >= 0:
            wi = owner[k]
            if first[wi] < 0:
                first[wi] = t
            last[wi] = t
    out = [{"word": w, "start": float(first[i] * sec), "end": float((last[i] + 1) * sec)}
           for i, w in enumerate(words)]
    for a, b in zip(out, out[1:]):
        gap = b["start"] - a["end"]
        if 0 < gap < merge_gap_s:
            a["end"] = b["start"] = 0.5 * (a["end"] + b["start"])
    return out
