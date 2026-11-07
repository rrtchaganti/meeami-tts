"""Time-varying WSOLA time-scale modification (docs/DECISIONS.md D-67).

Replaces Rubber Band R3 for the augmented sets: on LJSpeech, R3 costs ~0.85 UTMOS even at
0.95x/1.05x, while WSOLA (the algorithm behind `sox tempo`) is near-lossless up to ~1.2x (D-67).

    timemap_stretch(y, sr, tmap, frame_ms, tolerance_ms)

`tmap` is the same sample-level anchor list pyrubberband.timemap_stretch takes:
[(src_0=0, tgt_0=0), ..., (len(y), T)], strictly increasing in both coordinates. Output frame k
(centre k*H in the output, H = frame/2, periodic Hann, 50% overlap) is copied from the input
around src(k*H) (piecewise-linear inverse map), shifted by up to +-tolerance to the position whose
waveform best continues the previous frame (normalized cross-correlation, Verhelst & Roelands
1993).
Output length is exactly T; pitch is preserved; each anchor lands within +-tolerance.
"""
from __future__ import annotations

import numpy as np


def timemap_stretch(y: np.ndarray, sr: int, tmap: list[tuple[int, int]], frame_ms: float = 40.0,
                    tolerance_ms: float = 10.0) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    src = np.array([p[0] for p in tmap], dtype=np.float64)
    tgt = np.array([p[1] for p in tmap], dtype=np.float64)
    if len(tmap) < 2 or np.any(np.diff(src) <= 0) or np.any(np.diff(tgt) <= 0):
        raise ValueError("tmap must be strictly increasing in source and target")
    T = int(tgt[-1])
    N = 2 * max(1, int(round(frame_ms * sr / 1000 / 2)))
    H, half = N // 2, N // 2
    tol = max(0, int(round(tolerance_ms * sr / 1000)))
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)   # periodic Hann: sums to 1 at H = N/2

    pad = half + tol + N
    xp = np.pad(y, (pad, pad))
    n_frames = int(np.ceil(T / H)) + 1
    out = np.zeros(n_frames * H + N)
    wsum = np.zeros_like(out)
    prev = None
    for k in range(n_frames):
        c = float(np.interp(k * H, tgt, src))                # nominal input centre
        start = int(round(c)) - half + pad                   # nominal input start in xp
        if prev is None or tol == 0:
            best = start
        else:
            cont = xp[prev + H: prev + H + N]                # natural continuation of last frame
            lo = max(start - tol, 0)
            hi = min(start + tol, len(xp) - N)
            region = xp[lo: hi + N]
            if np.dot(cont, cont) < 1e-12:
                best = start
            else:
                xc = np.correlate(region, cont, mode="valid")
                c2 = np.concatenate([[0.0], np.cumsum(region * region)])
                energy = np.sqrt(np.maximum(c2[N:] - c2[:-N], 1e-12))
                lags = np.arange(lo, lo + len(xc)) - start
                # normalized cross-correlation; tiny preference for the nominal position breaks
                # ties between pitch periods (exact identity for an identity map)
                score = xc / energy - 1e-3 * np.abs(lags) / max(tol, 1)
                best = lo + int(np.argmax(score))
        o = k * H                                            # output start = centre k*H - half (+half)
        out[o: o + N] += win * xp[best: best + N]
        wsum[o: o + N] += win
        prev = best
    res = out[half: half + T] / np.maximum(wsum[half: half + T], 1e-3)
    return res.astype(np.float32)
