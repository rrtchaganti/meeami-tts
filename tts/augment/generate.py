"""`make aug-a` / `make aug-b`: augmented copies of LJ train (docs/PLAN.md Sections 6, 7).

    python -m tts.augment.generate --kind a|b [--limit N] [--config configs/base.yaml]

Writes data/aug/<kind>/<id>_s<seed>.flac and data/manifests/aug_<kind>_all.jsonl (unfiltered;
the UTMOS filter, tts.augment.filter, writes aug_<kind>.jsonl). Deterministic per (seed, id).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from omegaconf import DictConfig, OmegaConf

from tts.audio import load_wav
from tts.augment.core import Plan, elastic_factors, phone_sigmas, render, rng_for, solve_z
from tts.augment.pace_b import word_spans
from tts.config import REPO_ROOT, load_config, repo_path
from tts.splits import write_jsonl
from tts.stats import read_jsonl
from tts.text.normalize import words_with_punct
from tts.text.symbols import SIL, SP


def plan_a(row: dict, stats: dict, cfg: DictConfig, rng: np.random.Generator) -> tuple[Plan, dict]:
    a = cfg.augment.a
    lo, hi = cfg.augment.factor_clip
    sr, hop = int(cfg.audio.sample_rate), int(cfg.audio.hop_length)
    ph = row["phones"]
    sig = phone_sigmas(stats, ph)
    z_u = float(rng.uniform(*a.z_u))
    n_words = max(row["word_ids"]) + 1
    z_w = z_u + rng.normal(0.0, a.eps_std, size=n_words)
    factors = [1.0 if p in (SIL, SP) else float(elastic_factors(z_w[w], sig[i], lo, hi))
               for i, (p, w) in enumerate(zip(ph, row["word_ids"]))]
    plan = Plan(factors)
    for i, p in enumerate(ph):
        if p == SP:
            plan.sp_target[i] = row["durations"][i] * hop * float(rng.uniform(*a.sp_factor))
    punct = [f for _, f in words_with_punct(row["text"])]
    spans = word_spans(row)
    if len(punct) == len(spans):
        for w, (_, b) in enumerate(spans[:-1]):
            if punct[w] and ph[b + 1] != SP and rng.random() < a.punct_pause_prob:
                ms = float(rng.uniform(*a.punct_pause_ms))
                plan.inserts.append((b, ms * sr / 1000))
    return plan, {"z_u": z_u}


def plan_b(row: dict, stats: dict, pace: dict, cfg: DictConfig,
           rng: np.random.Generator) -> tuple[Plan, dict]:
    lo, hi = cfg.augment.factor_clip
    hop = int(cfg.audio.hop_length)
    ph, d = row["phones"], np.asarray(row["durations"], dtype=float)
    sig = phone_sigmas(stats, ph)
    mu = {p: v["mean"] for p, v in stats["phones"].items()}
    spans = word_spans(row)
    m_u = float(rng.choice(pace["utt_mean_log_rho"]))
    phi, var = float(pace["resid_lag1"]), float(pace["resid_var"])
    e = rng.normal(0.0, np.sqrt(var))
    factors = [1.0] * len(ph)
    for k, (a, b) in enumerate(spans):
        if k:
            e = phi * e + np.sqrt(max(1 - phi ** 2, 0.0) * var) * rng.normal()
        rho = float(np.exp(m_u + e))
        idx = np.arange(a, b + 1)
        target = rho * sum(np.exp(mu[ph[i]]) for i in idx)  # word frames at the L2 pace
        z = solve_z(d[idx], sig[idx], target, lo, hi)
        for i in idx:
            factors[i] = float(elastic_factors(z, sig[i], lo, hi))
    plan = Plan(factors)
    lens = np.array([int(k) for k in pace["sp_len_hist"]])
    probs = np.array(list(pace["sp_len_hist"].values()), dtype=float)
    probs /= probs.sum()
    for (_, b) in spans[:-1]:
        if rng.random() >= pace["p_sp_after_word"]:
            continue
        L = int(rng.choice(lens, p=probs)) * hop
        nxt = b + 1
        if ph[nxt] == SP:
            plan.sp_target[nxt] = max(d[nxt] * hop, L)  # lengthen (never shorten)
        else:
            plan.inserts.append((b, L))
    return plan, {"log_rho_u": m_u}


_G: dict = {}


def _init(cfg_yaml, stats, pace):
    _G["cfg"] = OmegaConf.create(cfg_yaml)
    _G["stats"], _G["pace"] = stats, pace


def _one(job):
    row, kind, seed, out_path = job
    cfg, stats, pace = _G["cfg"], _G["stats"], _G["pace"]
    rng = rng_for(seed, row["id"])
    plan, meta = plan_a(row, stats, cfg, rng) if kind == "a" else plan_b(row, stats, pace, cfg, rng)
    y, sr = load_wav(repo_path(row["wav"]))
    audio, ph, dur, wid = render(y, row, plan, cfg, OmegaConf.to_container(cfg.augment.rubberband_args))
    sf.write(out_path, audio, sr, subtype="PCM_16")
    rel = str(Path(out_path).relative_to(REPO_ROOT))
    new = dict(row, id=f"{row['id']}_{kind}{seed}", wav=rel, phones=ph, durations=dur,
               word_ids=wid, n_frames=int(sum(dur)), is_aug=True, source_id=row["id"],
               aug_kind=kind, aug_seed=seed, **meta)
    assert len(audio) == new["n_frames"] * int(cfg.audio.hop_length)
    return new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", choices=["a", "b"], required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(repo_path(a.config), a.overrides)
    stats = json.loads((repo_path(cfg.paths.stats) / "phone_logdur.json").read_text())
    pace = None
    if a.kind == "b":
        pace = json.loads(repo_path(cfg.augment.b.stats).read_text())
    rows = read_jsonl(repo_path(cfg.paths.manifests) / "lj_train.jsonl")
    if a.limit:
        rows = rows[: a.limit]
    out_dir = repo_path(cfg.augment.out_dir) / a.kind
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = cfg.augment.audio_format
    jobs = [(r, a.kind, int(s), str(out_dir / f"{r['id']}_s{s}.{ext}"))
            for s in cfg.augment.seeds for r in rows]
    t = time.time()
    out = []
    workers = int(cfg.augment.workers) or os.cpu_count()
    with Pool(workers, initializer=_init,
              initargs=(OmegaConf.to_yaml(cfg, resolve=True), stats, pace)) as pool:
        for i, r in enumerate(pool.imap(_one, jobs, chunksize=4), 1):
            out.append(r)
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)} ({(time.time() - t) / i:.2f} s/clip wall)", flush=True)
    secs = time.time() - t
    name = f"aug_{a.kind}_all" + ("_limit" if a.limit else "")
    write_jsonl(repo_path(cfg.paths.manifests) / f"{name}.jsonl", out, force=True)
    print(f"aug-{a.kind}: {len(out)} clips in {secs / 60:.1f} min "
          f"({secs / len(out):.3f} s/clip wall, {workers} workers) -> {name}.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
