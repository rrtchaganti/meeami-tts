"""Check that augmented audio follows its planned durations: MFA-align a sample of augmented
clips and compare MFA word boundaries with the planned (manifest) ones.

    python -m tts.augment.verify --manifest data/manifests/aug_a_all.jsonl --n 100
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys

import numpy as np

from tts.align.lj import _prep_one
from tts.align.mfa import align_corpus
from tts.align.textgrid import read_tiers
from tts.config import REPO_ROOT, load_config, repo_path
from tts.stats import read_jsonl


def planned_word_times(row: dict, hop: int, sr: int) -> list[tuple[str, float, float]]:
    b = np.concatenate([[0], np.cumsum(row["durations"])]) * hop / sr
    spans: dict[int, list[int]] = {}
    for i, w in enumerate(row["word_ids"]):
        if w >= 0:
            spans.setdefault(w, [i, i])[1] = i
    return [(row["words"][w], float(b[a]), float(b[e + 1])) for w, (a, e) in sorted(spans.items())]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(repo_path(a.config))
    rows = read_jsonl(repo_path(a.manifest))
    rows = random.Random(cfg.seed).sample(rows, min(a.n, len(rows)))
    name = repo_path(a.manifest).stem + "_verify"
    corpus = repo_path(cfg.paths.mfa_corpus) / name
    out = repo_path(cfg.paths.align) / name
    shutil.rmtree(corpus, ignore_errors=True)
    corpus.mkdir(parents=True)
    for r in rows:
        _prep_one((str(repo_path(r["wav"])), str(corpus / f"{r['id']}.wav"),
                   str(corpus / f"{r['id']}.lab"), " ".join(r["words"]), int(cfg.mfa.sample_rate)))
    res = align_corpus(cfg, corpus, out, single_speaker=True)
    if res.returncode != 0:
        print(res.stderr[-2000:], file=sys.stderr)
        return 1
    hop, sr = int(cfg.audio.hop_length), int(cfg.audio.sample_rate)
    errs, n_ok = [], 0
    for r in rows:
        tg = out / f"{r['id']}.TextGrid"
        if not tg.exists():
            continue
        words, _ = read_tiers(tg)
        mw = [w for w in words if w.label.strip()]
        pw = planned_word_times(r, hop, sr)
        if [w.label for w in mw] != [w for w, _, _ in pw]:
            continue
        n_ok += 1
        for m, (_, s, e) in zip(mw, pw):
            errs += [abs(m.start - s), abs(m.end - e)]
    x = np.asarray(errs) * 1000
    summary = {"manifest": a.manifest, "sampled": len(rows), "aligned_and_matched": n_ok,
               "boundaries": int(x.size), "mae_ms": float(x.mean()), "median_ms": float(np.median(x)),
               "within_20ms": float((x <= 20).mean() * 100), "within_50ms": float((x <= 50).mean() * 100)}
    dst = repo_path(cfg.paths.reports) / f"{name}.json"
    dst.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    shutil.rmtree(corpus, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
