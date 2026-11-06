"""`make stats` -> stats/phone_logdur.json (docs/PLAN.md Section 5 step 6, Sections 6-7).

Per phone: mean and std of log duration (natural log of frames) over LJ train, excluding `sil`,
`sp` and padding. Phones with fewer than `stats.min_count` occurrences use the median sigma.
Zero-frame tokens have no log duration; they are counted but excluded.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from tts.config import REPO_ROOT, load_config, repo_path
from tts.text.symbols import PHONES


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def phone_logdur(rows: list[dict], min_count: int) -> dict:
    logs: dict[str, list[float]] = defaultdict(list)
    zeros: dict[str, int] = defaultdict(int)
    for r in rows:
        if r.get("is_aug"):
            continue
        for ph, d in zip(r["phones"], r["durations"]):
            if ph not in PHONES:
                continue
            if d <= 0:
                zeros[ph] += 1
            else:
                logs[ph].append(math.log(d))
    phones = {}
    for ph in PHONES:
        x = np.asarray(logs.get(ph, []), dtype=np.float64)
        phones[ph] = {"count": int(x.size), "zero_frames": zeros.get(ph, 0),
                      "mean": float(x.mean()) if x.size else None,
                      "std": float(x.std(ddof=1)) if x.size > 1 else None}
    reliable = [v["std"] for v in phones.values() if v["count"] >= min_count and v["std"] is not None]
    if not reliable:
        raise ValueError("no phone has enough occurrences to estimate sigma")
    median_sigma = float(np.median(reliable))
    for v in phones.values():
        v["sigma_from_median"] = v["count"] < min_count or v["std"] is None
        v["sigma"] = median_sigma if v["sigma_from_median"] else v["std"]
    means = [v["mean"] for v in phones.values() if v["mean"] is not None]
    for v in phones.values():
        if v["mean"] is None:  # unseen phone: fall back to the median mean (flagged)
            v["mean"] = float(np.median(means))
            v["mean_from_median"] = True
    return {"median_sigma": median_sigma, "min_count": min_count, "phones": phones}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phone log-duration stats (make stats)")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg: DictConfig = load_config(repo_path(args.config), args.overrides)
    src = repo_path(cfg.paths.manifests) / "lj_train.jsonl"
    rows = read_jsonl(src)
    if not rows or rows[0].get("durations") is None:
        print(f"{src} has no durations; run `make align-lj` first", file=sys.stderr)
        return 1
    out = phone_logdur(rows, int(cfg.stats.min_count))
    out = {"source": str(src.relative_to(REPO_ROOT)), "n_utterances": len(rows),
           "unit": "log(frames), frame = hop_length / sample_rate",
           "hop_length": int(cfg.audio.hop_length), "sample_rate": int(cfg.audio.sample_rate),
           "created": date.today().isoformat(), **out}
    dst = repo_path(cfg.paths.stats) / "phone_logdur.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1) + "\n")
    ph = out["phones"]
    few = sorted(k for k, v in ph.items() if v["sigma_from_median"])
    print(f"wrote {dst.relative_to(REPO_ROOT)}: {len(rows)} utterances, "
          f"{sum(v['count'] for v in ph.values())} phone tokens, median sigma "
          f"{out['median_sigma']:.3f}; median-sigma phones ({len(few)}): {', '.join(few)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
