"""Section 8 metric functions: word-boundary error, pause F1, bootstrap CIs, paired Wilcoxon."""
from __future__ import annotations

import difflib

import numpy as np


def match_words(ref: list[dict], hyp: list[dict]) -> list[tuple[int, int]]:
    a = [w["word"].lower() for w in ref]
    b = [w["word"].lower() for w in hyp]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return [(m.a + k, m.b + k) for m in sm.get_matching_blocks() for k in range(m.size)]


def boundary_errors(ref: list[dict], hyp: list[dict]) -> list[float]:
    """|start| and |end| errors (s) for every matched word (every word start and end)."""
    out = []
    for i, j in match_words(ref, hyp):
        out += [abs(ref[i]["start"] - hyp[j]["start"]), abs(ref[i]["end"] - hyp[j]["end"])]
    return out


def boundary_summary(errs: list[float]) -> dict:
    if not errs:
        return {"mae_ms": float("nan"), "within_20ms": float("nan"), "within_50ms": float("nan")}
    x = np.asarray(errs) * 1000
    eps = 1e-6  # ms; exact-tolerance boundaries count as within (float rounding)
    return {"mae_ms": float(x.mean()), "within_20ms": float((x <= 20 + eps).mean() * 100),
            "within_50ms": float((x <= 50 + eps).mean() * 100)}


def pauses(words: list[dict], min_pause: float) -> list[bool]:
    """Per word boundary k (between word k and k+1): silence >= min_pause."""
    return [words[k + 1]["start"] - words[k]["end"] >= min_pause for k in range(len(words) - 1)]


def pause_f1(ref: list[dict], hyp: list[dict], min_pause: float = 0.1) -> dict:
    """F1 of output pauses vs reference pauses at word-boundary level (boundaries whose two
    words are both matched)."""
    pairs = match_words(ref, hyp)
    idx = {i: j for i, j in pairs}
    tp = fp = fn = 0
    for k in range(len(ref) - 1):
        if k in idx and (k + 1) in idx and idx[k + 1] == idx[k] + 1:
            j = idx[k]
            r = ref[k + 1]["start"] - ref[k]["end"] >= min_pause
            h = hyp[j + 1]["start"] - hyp[j]["end"] >= min_pause
            tp += r and h
            fp += (not r) and h
            fn += r and not h
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    if tp + fp + fn == 0:
        f1 = 1.0  # no pauses on either side: perfect agreement
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def bootstrap_ci(x, n: int = 1000, seed: int = 1234, stat=np.nanmean) -> tuple[float, float, float]:
    """(estimate, lo, hi) with a 95% percentile bootstrap over utterances."""
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))],
                   dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([stat(x[rng.integers(0, x.size, x.size)]) for _ in range(n)])
    return float(stat(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def wilcoxon(a, b) -> dict:
    """Paired Wilcoxon signed-rank test over utterances present (non-NaN) in both."""
    from scipy.stats import wilcoxon as w

    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 5 or np.allclose(a[m], b[m]):
        return {"n": int(m.sum()), "statistic": float("nan"), "p": float("nan"),
                "median_diff": float(np.median(a[m] - b[m])) if m.any() else float("nan")}
    r = w(a[m], b[m])
    return {"n": int(m.sum()), "statistic": float(r.statistic), "p": float(r.pvalue),
            "median_diff": float(np.median(a[m] - b[m]))}
