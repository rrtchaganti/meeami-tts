#!/usr/bin/env python
"""Evaluation harness (docs/PLAN.md Section 8).

    python scripts/evaluate.py --system m0 --split dev
    python scripts/evaluate.py --system m1 --ckpt export/m1/ckpt_last.pt --split dev

Writes results/<system>/<split>/{per_utt.csv, summary.json} and results/<split>_comparison.csv
(paired Wilcoxon vs M1), plus plots in results/plots/. --split test refuses to run unless the git
tag freeze-v1 exists. Stages are cached (re-runs skip finished work).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

# scripts/ must not be importable (scripts/profile.py would shadow the stdlib `profile`)
sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np  # noqa: E402

from tts.config import REPO_ROOT, load_config, repo_path  # noqa: E402
from tts.guard import assert_split_allowed  # noqa: E402

METRICS = ["length_error", "wb_mae_ms", "wb_within20", "wb_within50", "wb_mae_ctc_ms", "pause_f1",
           "wer", "cer", "wer_ref", "us_prob", "us_prob_ref", "spk_sim_lj", "spk_sim_ref", "utmos",
           "stretch_ratio"]


def jl_read(p: Path) -> dict:
    if not p.exists():
        return {}
    return {d["id"]: d for d in (json.loads(x) for x in p.read_text().splitlines() if x.strip())}


def jl_append(p: Path, d: dict) -> None:
    with open(p, "a") as f:
        f.write(json.dumps(d) + "\n")


def words_from_textgrid(tg: Path, silence: set) -> list[dict]:
    from tts.align.textgrid import read_tiers

    w, _ = read_tiers(tg)
    return [{"word": x.label.lower(), "start": x.start, "end": x.end} for x in w
            if x.label.strip() and x.label not in silence and x.label != "spn"]


def manual_words(row: dict, cfg) -> list[dict]:
    if row.get("ref_words_manual"):
        return row["ref_words_manual"]
    from tts.align.l2 import manual_words as mw
    from tts.align.textgrid import read_tiers

    p = repo_path(cfg.paths.raw_l2) / row["speaker"] / cfg.l2.annotation_dir / f"{row['utt']}.TextGrid"
    return mw(read_tiers(p)[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", required=True)
    ap.add_argument("--ckpt", default=None, help="default: export/<system>/ckpt_last.pt (m0: none)")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=0, help="limit utterances (debug)")
    ap.add_argument("--tag", default=None, help="results name (default: system)")
    ap.add_argument("--no_utmos", action="store_true",
                    help="skip UTMOS (torch.hub SpeechMOS); recorded as NaN with a note")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(repo_path(a.config), a.overrides)
    assert_split_allowed(a.split, cfg)
    from tts.audio import load_wav
    from tts.eval.metrics import boundary_errors, boundary_summary, pause_f1
    from tts.infer.batch import prealign, run_batch, select_rows
    from tts.infer.pipeline import Pipeline
    from tts.infer.systems import load_system
    from tts.text.normalize import Normalizer, words

    name = a.tag or a.system
    ckpt = a.ckpt or (a.system if a.system in ("m0", "v0") else f"export/{a.system}/ckpt_last.pt")
    res = repo_path("results") / name / a.split
    wav_root = res / "wav"
    res.mkdir(parents=True, exist_ok=True)
    ref_cache = repo_path("results") / "_ref" / a.split
    ref_cache.mkdir(parents=True, exist_ok=True)
    rows = select_rows(cfg, a.split, a.n)
    print(f"[eval] system={name} ckpt={ckpt} split={a.split} utterances={len(rows)}", flush=True)
    notes = []

    # 1. synthesize (cached per utterance)
    todo = [r for r in rows if not (wav_root / r["id"] / "timings.json").exists()]
    if todo:
        pipe = Pipeline(cfg, load_system(ckpt, cfg, a.device))
        pre = prealign(cfg, todo, ref_cache / "_ref_align", pipe.g2p)
        run_batch(pipe, todo, wav_root, pre, plot=False)
        del pipe
    timings = {r["id"]: json.loads((wav_root / r["id"] / "timings.json").read_text())
               for r in rows if (wav_root / r["id"] / "timings.json").exists()}
    ok_rows = [r for r in rows if r["id"] in timings]

    # 2. MFA on the outputs with the input text (batch, no speaker adaptation = align_one)
    from tts.align.lj import _prep_one
    from tts.align.mfa import align_corpus

    norm = Normalizer(cfg)
    out_align = res / "align_out"
    need = [r for r in ok_rows if not (out_align / f"{r['id']}.TextGrid").exists()]
    if need:
        corpus = res / "_out_corpus"
        shutil.rmtree(corpus, ignore_errors=True)
        corpus.mkdir(parents=True)
        for r in need:
            _prep_one((str(wav_root / r["id"] / "out.wav"), str(corpus / f"{r['id']}.wav"),
                       str(corpus / f"{r['id']}.lab"), " ".join(words(norm(r["text"]))),
                       int(cfg.mfa.sample_rate)))
        rr = align_corpus(cfg, corpus, out_align, single_speaker=True, log=res / "mfa_out.log")
        if rr.returncode != 0:
            notes.append(f"MFA on outputs failed (exit {rr.returncode}); see mfa_out.log")
        shutil.rmtree(corpus, ignore_errors=True)
    silence = set(cfg.align.silence_labels)

    # 3. per-output scores (cached): CTC words, ASR, accent, speaker, UTMOS
    out_cache = res / "scores.jsonl"
    done = jl_read(out_cache)
    refd = jl_read(ref_cache / "ref_scores.jsonl")
    from tts.eval.asr import ASR, wer_cer
    from tts.eval.scorers import AccentID, SpeakerEmbedder, cosine
    from tts.eval.assets import torch_home
    from tts.infer.ctc import ctc_word_times

    utmos = None
    if a.no_utmos:
        notes.append("UTMOS skipped (--no_utmos: torch.hub SpeechMOS not loaded); utmos = NaN")
    else:
        try:
            from tts.eval.scorers import UTMOS

            utmos = UTMOS(cfg, a.device)
        except Exception as e:  # pinned model unavailable: report, never substitute (CLAUDE.md)
            notes.append(f"UTMOS unavailable ({type(e).__name__}: {str(e)[:120]}); utmos = NaN")
    asr, accent, spk = ASR(cfg, a.device), AccentID(cfg, a.device), SpeakerEmbedder(cfg, a.device)

    lj_emb_p = repo_path("results") / "_ref" / "lj_test_ecapa_mean.npy"
    lj_gt = repo_path("results") / "_ref" / "lj_test_gt.json"
    gt_needs_mos = utmos is not None and lj_gt.exists() and \
        json.loads(lj_gt.read_text()).get("utmos") is None
    if not lj_emb_p.exists() or not lj_gt.exists() or gt_needs_mos:
        from tts.models.vits_mfa.data import read_manifest

        lj_rows = sorted(read_manifest("data/manifests/lj_test.jsonl"), key=lambda r: r["id"])
        lj_rows = lj_rows[:: max(1, len(lj_rows) // 100)][:100]
        embs, wers, moss = [], [], []
        for r in lj_rows:
            y, sr = load_wav(repo_path(r["wav"]))
            embs.append(spk(y, sr))
            wers.append(wer_cer(r["text"], asr.transcribe(y, sr))[0])
            moss.append(utmos(y, sr) if utmos else float("nan"))
        m = np.mean(embs, 0)
        np.save(lj_emb_p, m / np.linalg.norm(m))
        lj_gt.write_text(json.dumps({"n": len(lj_rows), "wer": float(np.mean(wers)),
                                     "utmos": float(np.nanmean(moss)) if utmos else None}, indent=1))
    lj_emb = np.load(lj_emb_p)

    for r in rows:
        if r["id"] not in refd:
            y, sr = load_wav(repo_path(r["wav"]))
            hyp = asr.transcribe(y, sr)
            w, c = wer_cer(r["text"], hyp)
            e = spk(y, sr)
            d = {"id": r["id"], "asr": hyp, "wer": w, "cer": c, "us_prob": accent.us_probability(y, sr),
                 "emb": e.tolist()}
            jl_append(ref_cache / "ref_scores.jsonl", d)
            refd[r["id"]] = d
    th = str(torch_home(cfg))
    for i, r in enumerate(ok_rows, 1):
        if r["id"] in done:
            continue
        y, sr = load_wav(wav_root / r["id"] / "out.wav")
        hyp = asr.transcribe(y, sr)
        w, c = wer_cer(r["text"], hyp)
        e = spk(y, sr)
        try:
            ctc = ctc_word_times(y, sr, words(norm(r["text"])), th, a.device,
                                 float(cfg.eval.min_pause_s))
        except Exception as ex:
            ctc = None
            notes.append(f"CTC on output failed for {r['id']}: {ex}")
        d = {"id": r["id"], "asr": hyp, "wer": w, "cer": c,
             "us_prob": accent.us_probability(y, sr),
             "spk_sim_lj": cosine(e, lj_emb), "spk_sim_ref": cosine(e, np.asarray(refd[r["id"]]["emb"])),
             "utmos": utmos(y, sr) if utmos else None, "ctc_words": ctc}
        jl_append(out_cache, d)
        done[r["id"]] = d
        if i % 100 == 0:
            print(f"  scored {i}/{len(ok_rows)}", flush=True)

    if utmos is not None:  # fill UTMOS for outputs scored in an earlier run without it
        filled = 0
        for r in ok_rows:
            d = done.get(r["id"])
            if d is not None and d.get("utmos") is None:
                y, sr = load_wav(wav_root / r["id"] / "out.wav")
                d["utmos"] = utmos(y, sr)
                filled += 1
        if filled:
            out_cache.write_text("".join(json.dumps(done[r["id"]]) + "\n" for r in ok_rows
                                         if r["id"] in done))

    # 4. per-utterance table
    per, pooled_mfa, pooled_ctc = [], [], []
    for r in rows:
        t = timings.get(r["id"])
        if t is None:
            per.append({"id": r["id"], "speaker": r["speaker"], "accent": r["accent"], "status": "synthesis failed"})
            continue
        ref_w = manual_words(r, cfg)
        tg = out_align / f"{r['id']}.TextGrid"
        out_w = words_from_textgrid(tg, silence) if tg.exists() else []
        e_mfa = boundary_errors(ref_w, out_w) if out_w else []
        s = done.get(r["id"], {})
        e_ctc = boundary_errors(ref_w, s["ctc_words"]) if s.get("ctc_words") else []
        pooled_mfa += e_mfa
        pooled_ctc += e_ctc
        bs = boundary_summary(e_mfa)
        pf = pause_f1(ref_w, out_w, float(cfg.eval.min_pause_s)) if out_w else {"f1": float("nan")}
        per.append({"id": r["id"], "speaker": r["speaker"], "accent": r["accent"], "status": "ok",
                    "aligner": t["aligner"], "length_error": t["length_error_samples"],
                    "wb_mae_ms": bs["mae_ms"], "wb_within20": bs["within_20ms"],
                    "wb_within50": bs["within_50ms"],
                    "wb_mae_ctc_ms": boundary_summary(e_ctc)["mae_ms"], "pause_f1": pf["f1"],
                    "wer": s.get("wer"), "cer": s.get("cer"), "wer_ref": refd[r["id"]]["wer"],
                    "us_prob": s.get("us_prob"), "us_prob_ref": refd[r["id"]]["us_prob"],
                    "spk_sim_lj": s.get("spk_sim_lj"), "spk_sim_ref": s.get("spk_sim_ref"),
                    "utmos": s.get("utmos") if s.get("utmos") is not None else float("nan"),
                    "stretch_ratio": t.get("stretch_ratio"), "n_capped": len(t["capped_words"]),
                    "ref_duration_s": t["ref_duration_s"]})
    cols = ["id", "speaker", "accent", "status", "aligner", *METRICS, "n_capped", "ref_duration_s"]
    with open(res / "per_utt.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(per)

    # 5. summary with bootstrap CIs, by accent
    from tts.eval.metrics import bootstrap_ci

    groups = {"all": [p for p in per if p["status"] == "ok"]}
    for p in groups["all"]:
        groups.setdefault(p["accent"], []).append(p)
    summary = {"system": name, "ckpt": ckpt, "split": a.split, "n": len(rows),
               "n_ok": len(groups["all"]), "notes": sorted(set(notes)),
               "pooled_boundaries_mfa": boundary_summary(pooled_mfa),
               "pooled_boundaries_ctc": boundary_summary(pooled_ctc),
               "lj_test_ground_truth": json.loads(lj_gt.read_text()),
               "aligner_counts": {k: sum(p.get("aligner") == k for p in groups["all"])
                                  for k in ("mfa", "ctc")}, "metrics": {}}
    for g, ps in groups.items():
        summary["metrics"][g] = {}
        for m in METRICS:
            est, lo, hi = bootstrap_ci([p.get(m) for p in ps], int(cfg.eval.bootstrap), cfg.seed)
            summary["metrics"][g][m] = {"mean": est, "ci95": [lo, hi]}
    (res / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    # 6. paired Wilcoxon vs M1 for every system with results on this split
    from tts.eval.metrics import wilcoxon

    base = repo_path("results") / "m1" / a.split / "per_utt.csv"
    if base.exists():
        def load(p):
            with open(p) as f:
                return {d["id"]: d for d in csv.DictReader(f)}
        b = load(base)
        lines = []
        for sysdir in sorted(repo_path("results").iterdir()):
            p = sysdir / a.split / "per_utt.csv"
            if sysdir.name in ("m1", "_ref", "plots") or not p.exists():
                continue
            o = load(p)
            ids = sorted(set(o) & set(b))
            for acc in ["all"] + sorted({b[i]["accent"] for i in ids}):
                sel = [i for i in ids if acc == "all" or b[i]["accent"] == acc]
                for m in METRICS:
                    def num(d):
                        try:
                            return float(d[m])
                        except (ValueError, KeyError):
                            return float("nan")
                    r = wilcoxon([num(o[i]) for i in sel], [num(b[i]) for i in sel])
                    lines.append({"system": sysdir.name, "vs": "m1", "accent": acc, "metric": m, **r})
        with open(repo_path("results") / f"{a.split}_comparison.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["system", "vs", "accent", "metric", "n", "statistic",
                                               "p", "median_diff"])
            wr.writeheader()
            wr.writerows(lines)

    # 7. plots
    from tts.eval.plots import make_plots

    make_plots(cfg, a.split)
    s = summary["metrics"]["all"]
    print(f"[eval] {name}/{a.split}: n_ok {summary['n_ok']}/{summary['n']}; word-boundary MAE "
          f"{summary['pooled_boundaries_mfa']['mae_ms']:.1f} ms; length error mean "
          f"{s['length_error']['mean']:.2f}; WER {s['wer']['mean']:.3f} (ref {s['wer_ref']['mean']:.3f}); "
          f"US prob {s['us_prob']['mean']:.3f} (ref {s['us_prob_ref']['mean']:.3f}); spk sim LJ "
          f"{s['spk_sim_lj']['mean']:.3f} / ref {s['spk_sim_ref']['mean']:.3f}; notes: {summary['notes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
