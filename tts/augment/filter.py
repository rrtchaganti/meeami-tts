"""UTMOS filter for augmented sets (docs/PLAN.md Section 6 "Filtering and training").

Each augmented clip is scored with UTMOS and with ECAPA speaker similarity to its original; the
`augment.filter.drop_fraction` (5%) with the largest UTMOS drop (original - augmented) are
dropped. Writes data/manifests/aug_<kind>.jsonl (kept), aug_<kind>_scores.jsonl and
reports/aug_<kind>.md (with 20 random kept clips to listen to).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date

import numpy as np

from tts.audio import load_wav
from tts.config import REPO_ROOT, load_config, repo_path
from tts.splits import write_jsonl
from tts.stats import read_jsonl


def select(scores: list[dict], drop_fraction: float) -> tuple[list[dict], float]:
    """Keep all but the `drop_fraction` with the largest utmos_drop. Returns (kept, threshold)."""
    order = sorted(scores, key=lambda s: s["utmos_drop"])
    n_keep = len(order) - int(round(drop_fraction * len(order)))
    thr = order[n_keep - 1]["utmos_drop"] if n_keep else float("-inf")
    return order[:n_keep], thr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", choices=["a", "b"], required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    cfg = load_config(repo_path(a.config))
    from tts.eval.scorers import UTMOS, SpeakerEmbedder, cosine

    man = repo_path(cfg.paths.manifests)
    rows = read_jsonl(man / f"aug_{a.kind}_all.jsonl")
    src = {r["id"]: r for r in read_jsonl(man / "lj_train.jsonl")}
    mos, spk = UTMOS(cfg, a.device), SpeakerEmbedder(cfg, a.device)
    orig: dict[str, tuple[float, np.ndarray]] = {}
    scores = []
    for i, r in enumerate(rows, 1):
        sid = r["source_id"]
        if sid not in orig:
            y, sr = load_wav(repo_path(src[sid]["wav"]))
            orig[sid] = (mos(y, sr), spk(y, sr))
        y, sr = load_wav(repo_path(r["wav"]))
        m, e = mos(y, sr), spk(y, sr)
        scores.append({"id": r["id"], "source_id": sid, "utmos_orig": orig[sid][0], "utmos": m,
                       "utmos_drop": orig[sid][0] - m, "spk_sim": cosine(e, orig[sid][1]),
                       "stretch": r["n_frames"] / src[sid]["n_frames"]})
        if i % 2000 == 0:
            print(f"  scored {i}/{len(rows)}", flush=True)
    kept, thr = select(scores, float(cfg.augment.filter.drop_fraction))
    keep_ids = {s["id"] for s in kept}
    write_jsonl(man / f"aug_{a.kind}_scores.jsonl", scores, force=True)
    write_jsonl(man / f"aug_{a.kind}.jsonl", [r for r in rows if r["id"] in keep_ids], force=True)

    def stat(key, ss):
        x = np.asarray([s[key] for s in ss])
        return f"{x.mean():.3f} (sd {x.std():.3f}, p5 {np.percentile(x, 5):.3f}, p95 {np.percentile(x, 95):.3f})"
    listen = random.Random(cfg.seed).sample(kept, min(int(cfg.augment.filter.n_listen), len(kept)))
    rep = [f"# Augmented set {a.kind.upper()} filter (T9, {date.today().isoformat()})", "",
           f"- Clips: {len(rows)}; kept {len(kept)}; dropped {len(rows) - len(kept)} "
           f"(largest UTMOS drop, threshold drop > {thr:.3f})",
           f"- UTMOS original: {stat('utmos_orig', scores)}",
           f"- UTMOS augmented (all): {stat('utmos', scores)}; kept: {stat('utmos', kept)}",
           f"- UTMOS drop (all): {stat('utmos_drop', scores)}",
           f"- ECAPA speaker similarity to original (all): {stat('spk_sim', scores)}",
           f"- Length ratio augmented / original: {stat('stretch', scores)}", "",
           "## 20 random kept clips for listening", "", "| Clip | Stretch | UTMOS | Drop | Spk sim |",
           "| --- | --- | --- | --- | --- |"]
    by_id = {r["id"]: r for r in rows}
    for s in sorted(listen, key=lambda s: s["id"]):
        rep.append(f"| `{by_id[s['id']]['wav']}` | {s['stretch']:.2f} | {s['utmos']:.2f} | "
                   f"{s['utmos_drop']:+.2f} | {s['spk_sim']:.3f} |")
    (repo_path(cfg.paths.reports) / f"aug_{a.kind}.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep[:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
