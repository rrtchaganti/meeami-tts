"""`make pace-stats-b` -> stats/l2_dev_pace.json (docs/PLAN.md Section 7 "Pace statistics").

For each aligned dev word: rho_w = reference word frames / sum_p exp(mu_p) over its phones,
mu_p = LJ mean log duration (stats/phone_logdur.json). No trained model is needed.
Stored: empirical utterance mean log rho; per-word residuals around it with their lag-1
autocorrelation and variance; P(sp after a word); empirical sp length histogram (frames).
Only word-level ratios and pause statistics are used (never L2 phone timing inside words).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import date

import numpy as np

from tts.config import REPO_ROOT, load_config, repo_path
from tts.stats import read_jsonl
from tts.text.symbols import SIL, SP


def word_spans(row: dict) -> list[tuple[int, int]]:
    """[(first token idx, last token idx)] per word, in order."""
    spans: dict[int, list[int]] = {}
    for i, w in enumerate(row["word_ids"]):
        if w >= 0:
            spans.setdefault(w, [i, i])[1] = i
    return [tuple(spans[w]) for w in sorted(spans)]


def utterance_stats(row: dict, mu: dict) -> dict | None:
    spans = word_spans(row)
    if not spans:
        return None
    ph, d = row["phones"], row["durations"]
    log_rho = []
    for a, b in spans:
        frames = sum(d[a:b + 1])
        expect = sum(math.exp(mu[ph[i]]) for i in range(a, b + 1))
        log_rho.append(math.log(max(frames, 1) / expect))
    sp_after, sp_len = [], []
    for k, (a, b) in enumerate(spans[:-1]):
        nxt = b + 1
        has = nxt < len(ph) and ph[nxt] == SP
        sp_after.append(has)
        if has:
            sp_len.append(d[nxt])
    return {"log_rho": log_rho, "sp_after": sp_after, "sp_len": sp_len}


def build(rows: list[dict], mu: dict) -> dict:
    means, resid_all, lag_pairs, sp_after, sp_len = [], [], [], [], []
    n_words = 0
    for r in rows:
        if r.get("align_status") != "ok":
            continue
        u = utterance_stats(r, mu)
        if u is None:
            continue
        lr = np.asarray(u["log_rho"])
        m = float(lr.mean())
        means.append(m)
        res = lr - m
        resid_all += res.tolist()
        lag_pairs += list(zip(res[:-1].tolist(), res[1:].tolist()))
        sp_after += u["sp_after"]
        sp_len += u["sp_len"]
        n_words += len(lr)
    x = np.asarray([a for a, _ in lag_pairs])
    y = np.asarray([b for _, b in lag_pairs])
    phi = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 0.0
    hist = Counter(int(v) for v in sp_len)
    return {"n_utterances": len(means), "n_words": n_words,
            "utt_mean_log_rho": sorted(means),
            "resid_var": float(np.var(resid_all)), "resid_lag1": phi,
            "p_sp_after_word": float(np.mean(sp_after)) if sp_after else 0.0,
            "sp_len_hist": {str(k): v for k, v in sorted(hist.items())}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dev-accent pace statistics (make pace-stats-b)")
    ap.add_argument("--config", default="configs/splits.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(repo_path(a.config))
    lj = json.loads((repo_path(cfg.paths.stats) / "phone_logdur.json").read_text())
    mu = {p: v["mean"] for p, v in lj["phones"].items()}
    rows = read_jsonl(repo_path(cfg.paths.manifests) / "l2_dev.jsonl")
    assert all(r["split"] == "dev" for r in rows), "Ablation B may only use dev speakers"
    out = build(rows, mu)
    out = {"source": "data/manifests/l2_dev.jsonl (MFA, all scripted dev utterances)",
           "created": date.today().isoformat(), "accents": list(cfg.l2.dev.keys()), **out}
    dst = repo_path(cfg.paths.stats) / "l2_dev_pace.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")
    m = np.asarray(out["utt_mean_log_rho"])
    print(f"wrote {dst.relative_to(REPO_ROOT)}: {out['n_utterances']} utts, {out['n_words']} words; "
          f"utt mean log rho {m.mean():.3f} (sd {m.std():.3f}, rho x{math.exp(m.mean()):.2f}); "
          f"resid var {out['resid_var']:.3f}, lag-1 {out['resid_lag1']:.3f}; "
          f"P(sp after word) {out['p_sp_after_word']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
