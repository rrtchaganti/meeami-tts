"""T5 `make align-l2-dev` / `make align-dev-eval` (docs/PLAN.md Sections 3, 7).

align-l2-dev: all scripted utterances of the 12 dev speakers -> 16 kHz copies, one folder per
speaker -> `mfa align` with speaker adaptation (no --single_speaker) -> parsed with the Section 3
rules -> fills `l2_dev.jsonl` (words, phones, durations, word_ids, n_frames, ref_words_mfa,
ref_words_manual for annotated IDs, align_status).

align-dev-eval: MFA vs manual annotation (`annotation/`), word and phone boundaries, per accent:
mean absolute error and % within 20 / 50 ms -> reports/mfa_l2_dev.md, results/mfa_l2_dev.json.

Test speakers are never touched here (tts.guard).
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from tts.align.lj import _prep_one
from tts.align.mfa import align_corpus
from tts.align.textgrid import AlignmentError, Interval, parse, read_tiers
from tts.config import REPO_ROOT, load_config, repo_path
from tts.guard import accent_of, assert_audio_allowed, speakers
from tts.splits import write_jsonl
from tts.stats import read_jsonl
from tts.text.normalize import Normalizer, words


def ref_n_frames(n_samples: int, sr: int, cfg: DictConfig) -> int:
    """Section 5 step 8: round(N_ref * 22050 / sr_ref / 256)."""
    return int(round(n_samples * cfg.audio.sample_rate / sr / cfg.audio.hop_length))


def _dirs(cfg, root):
    return repo_path(cfg.paths.mfa_corpus, root) / "l2_dev", repo_path(cfg.paths.align, root) / "l2_dev"


# --- stage: prep / align -------------------------------------------------------------------
def prep(cfg: DictConfig, rows: list[dict], corpus: Path, root: Path = REPO_ROOT) -> int:
    norm = Normalizer(cfg)
    jobs = []
    for r in rows:
        assert_audio_allowed(r["speaker"], cfg, root)
        d = corpus / r["speaker"]
        d.mkdir(parents=True, exist_ok=True)
        jobs.append((str(repo_path(r["wav"], root)), str(d / f"{r['utt']}.wav"),
                     str(d / f"{r['utt']}.lab"), " ".join(words(norm(r["text"]))),
                     int(cfg.mfa.sample_rate)))
    with Pool(os.cpu_count()) as pool:
        for _ in pool.imap_unordered(_prep_one, jobs, chunksize=32):
            pass
    return len(jobs)


# --- manual annotation ---------------------------------------------------------------------
def canonical(label: str) -> str | None:
    """'DH,D,s' -> 'DH'; 'AH0' -> 'AH'; additions ('sil,AH0,a' / 'x,y,a') -> None; silence -> ''."""
    parts = [p.strip() for p in label.split(",")]
    if len(parts) >= 3 and parts[2].lower() == "a":
        return None
    p = parts[0].upper()
    if p in ("", "SIL", "SP", "SPN"):
        return ""
    return re.sub(r"\d", "", p).rstrip("*")


def manual_tiers(path: Path) -> tuple[list[Interval], list[Interval]]:
    return read_tiers(path)


def manual_words(words_tier: list[Interval]) -> list[dict]:
    return [{"word": w.label.lower(), "start": w.start, "end": w.end}
            for w in words_tier if w.label.strip() and w.label.lower() not in ("sil", "sp", "<sil>")]


def _match(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return [(blk.a + k, blk.b + k) for blk in sm.get_matching_blocks() for k in range(blk.size)]


def compare(mfa_words, mfa_phones, man_words_tier, man_phones_tier) -> dict:
    """Boundary errors (seconds) for matched words and matched phones within matched words."""
    mw = [w for w in mfa_words if w.label.strip() and w.label not in ("sil", "sp")]
    hw = [w for w in man_words_tier if w.label.strip() and w.label.lower() not in ("sil", "sp")]
    pairs = _match([w.label.lower() for w in mw], [w.label.lower() for w in hw])
    word_err, phone_err = [], []
    for i, j in pairs:
        a, b = mw[i], hw[j]
        word_err += [abs(a.start - b.start), abs(a.end - b.end)]
        pa = [p for p in mfa_phones if a.start - 1e-6 <= 0.5 * (p.start + p.end) <= a.end + 1e-6
              and p.label not in ("", "sil", "sp", "spn")]
        pb = [p for p in man_phones_tier if b.start - 1e-6 <= 0.5 * (p.start + p.end) <= b.end + 1e-6]
        pb = [p for p in pb if canonical(p.label)]  # drop silences and additions
        for k, m in _match([re.sub(r"\d", "", p.label) for p in pa],
                           [canonical(p.label) for p in pb]):
            phone_err += [abs(pa[k].start - pb[m].start), abs(pa[k].end - pb[m].end)]
    return {"word": word_err, "phone": phone_err, "n_words_mfa": len(mw), "n_words_manual": len(hw),
            "n_words_matched": len(pairs)}


def summarize(errs: list[float]) -> dict:
    if not errs:
        return {"n": 0, "mae_ms": None, "within_20ms": None, "within_50ms": None}
    x = np.asarray(errs) * 1000
    return {"n": int(x.size), "mae_ms": float(x.mean()), "median_ms": float(np.median(x)),
            "within_20ms": float((x <= 20).mean() * 100), "within_50ms": float((x <= 50).mean() * 100)}


# --- parse + fill manifest -----------------------------------------------------------------
def _parse_row(args):
    row, tg_path, man_path, cfg_yaml = args
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(cfg_yaml)
    norm = Normalizer(cfg)
    out = dict(row)
    expected = words(norm(row["text"]))
    out["words"] = expected
    nf = ref_n_frames(row["n_samples"], row["sr"], cfg)
    out["n_frames"] = nf
    man = None
    if man_path and Path(man_path).exists():
        mt = manual_tiers(Path(man_path))
        out["ref_words_manual"] = manual_words(mt[0])
        man = mt
    if not Path(tg_path).exists():
        out["align_status"] = "mfa_failed"
        return out, None
    try:
        wt, pt = read_tiers(tg_path)
        p = parse(wt, pt, nf, cfg, expected)
    except AlignmentError as e:
        out["align_status"] = e.reason
        return out, None
    out.update(phones=p.phones, durations=p.durations, word_ids=p.word_ids,
               ref_words_mfa=[{"word": w.word, "start": w.start, "end": w.end} for w in p.words],
               align_status="ok")
    cmp = compare(wt, pt, *man) if man is not None and row.get("in_eval_set") is not None else None
    return out, cmp


def parse_dev(cfg: DictConfig, rows: list[dict], root: Path = REPO_ROOT):
    from omegaconf import OmegaConf

    out_dir = _dirs(cfg, root)[1]
    l2 = repo_path(cfg.paths.raw_l2, root)
    y = OmegaConf.to_yaml(cfg, resolve=True)
    jobs = [(r, str(out_dir / r["speaker"] / f"{r['utt']}.TextGrid"),
             str(l2 / r["speaker"] / cfg.l2.annotation_dir / f"{r['utt']}.TextGrid"), y)
            for r in rows]
    with Pool(os.cpu_count()) as pool:
        return list(pool.imap(_parse_row, jobs, chunksize=32))


def report(cfg: DictConfig, results, root: Path = REPO_ROOT, mfa_secs=None) -> None:
    acc = {s: a for s, (_, a) in accent_of(cfg).items()}
    status = Counter(r["align_status"] for r, _ in results)
    per = defaultdict(lambda: {"word": [], "phone": [], "utts": 0, "matched": 0, "manual": 0})
    for r, c in results:
        if c is None:
            continue
        for key in (acc[r["speaker"]], "all"):
            per[key]["word"] += c["word"]
            per[key]["phone"] += c["phone"]
            per[key]["utts"] += 1
            per[key]["matched"] += c["n_words_matched"]
            per[key]["manual"] += c["n_words_manual"]
    summary = {k: {"utterances": v["utts"], "word_match_rate": v["matched"] / max(v["manual"], 1),
                   "word": summarize(v["word"]), "phone": summarize(v["phone"])}
               for k, v in per.items()}
    res_dir = repo_path("results", root)
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "mfa_l2_dev.json").write_text(json.dumps(summary, indent=1) + "\n")

    def f(x, fmt="{:.1f}"):
        return "-" if x is None else fmt.format(x)
    rep = [f"# MFA error on accented speech (T5, `make align-dev-eval`, {date.today().isoformat()})",
           "", f"- Dev speakers: {', '.join(speakers(cfg, 'dev'))}; all {len(results)} scripted "
           f"utterances aligned with speaker adaptation (`mfa align`, no --single_speaker, "
           + " ".join(f"--{k} {v}" for k, v in cfg.mfa.passthrough.items()) + ")"
           + (f"; MFA wall time {mfa_secs / 60:.1f} min" if mfa_secs else ""),
           "- Alignment status: " + ", ".join(f"{k} {v}" for k, v in status.most_common()),
           "- Compared on every utterance with a manual TextGrid (`annotation/`, ~150 per speaker). "
           "Words matched by label sequence; phones matched within matched words by canonical "
           "label (error tags and stress ignored; additions excluded). Interval times only.",
           "- Each matched word / phone contributes two boundaries (start, end).", "",
           "| Accent | Utts | Words matched | Word MAE (ms) | Word <=20 ms | Word <=50 ms | "
           "Phone MAE (ms) | Phone <=20 ms | Phone <=50 ms |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k in sorted(summary, key=lambda k: (k == "all", k)):
        s = summary[k]
        w, p = s["word"], s["phone"]
        rep.append(f"| {k} | {s['utterances']} | {100 * s['word_match_rate']:.1f}% | {f(w['mae_ms'])} | "
                   f"{f(w['within_20ms'])}% | {f(w['within_50ms'])}% | {f(p['mae_ms'])} | "
                   f"{f(p['within_20ms'])}% | {f(p['within_50ms'])}% |")
    rp = repo_path(cfg.paths.reports, root) / "mfa_l2_dev.md"
    rp.write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L2 dev alignment and MFA-vs-manual evaluation")
    ap.add_argument("--config", default="configs/splits.yaml")
    ap.add_argument("--stage", choices=["prep", "align", "parse", "all"], default="all")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(repo_path(args.config), args.overrides)
    root = REPO_ROOT
    man_path = repo_path(cfg.paths.manifests, root) / "l2_dev.jsonl"
    rows = read_jsonl(man_path)
    assert all(r["split"] == "dev" for r in rows)
    corpus, out_dir = _dirs(cfg, root)
    mfa_secs = None
    if args.stage in ("prep", "all"):
        t = time.time()
        print(f"prep: {prep(cfg, rows, corpus, root)} files ({time.time() - t:.0f}s)")
    if args.stage in ("align", "all"):
        t = time.time()
        log = repo_path(cfg.paths.align, root) / "l2_dev_mfa.log"
        r = align_corpus(cfg, corpus, out_dir, single_speaker=False, root=root, log=log)
        mfa_secs = time.time() - t
        print(f"mfa align: exit {r.returncode} in {mfa_secs / 60:.1f} min")
        if r.returncode != 0:
            print(r.stderr[-3000:], file=sys.stderr)
            return 1
    if args.stage in ("parse", "all"):
        results = parse_dev(cfg, rows, root)
        write_jsonl(man_path, [r for r, _ in results], force=True)
        report(cfg, results, root, mfa_secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
