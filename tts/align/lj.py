"""T4 `make align-lj`: LJSpeech -> MFA corpus -> alignment -> parsed, QC'd, filled manifests.

docs/PLAN.md Sections 2 (MFA input), 3 (commands, parsing rules, quality checks), 10 (manifests).

    python -m tts.align.lj --config configs/splits.yaml [--stage prep|align|parse|all] [--limit N]

`--limit N` runs a pilot on the first N train IDs into `<corpus>_pilot` dirs (manifests untouched).
Source of truth for IDs and text is the VITS filelists + metadata.csv (T2), never the manifests,
so re-running cannot lose rows that an earlier run dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import date
from multiprocessing import Pool
from pathlib import Path

from omegaconf import DictConfig

from tts.align.mfa import align_corpus
from tts.align.textgrid import AlignmentError, parse_file, qc_reason
from tts.audio import load_wav, n_frames, resample, write_wav16
from tts.config import REPO_ROOT, load_config, repo_path
from tts.splits import fetch_filelists, lj_row, parse_filelist, read_lj_metadata, write_jsonl
from tts.text.normalize import Normalizer, words


def lj_items(cfg: DictConfig, root: Path = REPO_ROOT) -> list[tuple[str, str, str]]:
    """[(id, split, metadata column-3 text)] in filelist order."""
    fl = fetch_filelists(cfg, root)
    meta = read_lj_metadata(repo_path(cfg.paths.raw_lj, root) / cfg.lj.metadata)
    return [(i, s, meta[i]) for s, p in fl.items() for i in parse_filelist(p)]


def _dirs(cfg: DictConfig, pilot: bool, root: Path) -> tuple[Path, Path]:
    suffix = "_pilot" if pilot else ""
    return (repo_path(cfg.paths.mfa_corpus, root) / f"lj{suffix}",
            repo_path(cfg.paths.align, root) / f"lj{suffix}")


# --- stage 1: MFA corpus -----------------------------------------------------------------------
def _prep_one(job):
    src, dst_wav, dst_lab, lab_text, sr_out = job
    y, sr = load_wav(src)
    write_wav16(dst_wav, resample(y, sr, sr_out), sr_out)
    Path(dst_lab).write_text(lab_text + "\n", encoding="utf-8")


def prep(cfg: DictConfig, items, corpus: Path, root: Path = REPO_ROOT, workers: int = 0) -> int:
    norm = Normalizer(cfg)
    lj_wavs = repo_path(cfg.paths.raw_lj, root) / cfg.lj.wav_dir
    corpus.mkdir(parents=True, exist_ok=True)
    jobs = [(str(lj_wavs / f"{i}.wav"), str(corpus / f"{i}.wav"), str(corpus / f"{i}.lab"),
             " ".join(words(norm(t))), int(cfg.mfa.sample_rate)) for i, _, t in items]
    with Pool(workers or os.cpu_count()) as pool:
        for _ in pool.imap_unordered(_prep_one, jobs, chunksize=32):
            pass
    return len(jobs)


# --- stage 3: parse + QC -----------------------------------------------------------------------
_CFG: DictConfig | None = None


def _init(cfg_yaml: str):
    global _CFG
    import torch
    from omegaconf import OmegaConf

    torch.set_num_threads(1)
    _CFG = OmegaConf.create(cfg_yaml)


def _parse_one(job):
    utt, wav, tg, expected = job
    cfg = _CFG
    y, sr = load_wav(wav)
    if sr != cfg.audio.sample_rate:
        return utt, None, "bad_sample_rate", None
    nf = n_frames(y, cfg.audio)
    if not Path(tg).exists():
        return utt, None, "mfa_failed", nf
    try:
        p = parse_file(tg, nf, cfg, expected)
    except AlignmentError as e:
        return utt, None, e.reason, nf
    reason = qc_reason(p, nf, cfg)
    return utt, p, reason, nf


def parse_all(cfg: DictConfig, items, out_dir: Path, root: Path = REPO_ROOT, workers: int = 0):
    from omegaconf import OmegaConf

    norm = Normalizer(cfg)
    lj_wavs = repo_path(cfg.paths.raw_lj, root) / cfg.lj.wav_dir
    texts = {i: norm(t) for i, _, t in items}
    jobs = [(i, str(lj_wavs / f"{i}.wav"), str(out_dir / f"{i}.TextGrid"), words(texts[i]))
            for i, _, _ in items]
    results = {}
    with Pool(workers or os.cpu_count(), initializer=_init,
              initargs=(OmegaConf.to_yaml(cfg, resolve=True),)) as pool:
        for utt, p, reason, nf in pool.imap_unordered(_parse_one, jobs, chunksize=32):
            results[utt] = (p, reason, nf)
    return results, texts


def write_outputs(cfg: DictConfig, items, results, texts, root: Path, mfa_secs: float | None):
    rows = {"train": [], "val": [], "test": []}
    reasons = Counter()
    dropped = []
    merged = n_sp = zero_phone = 0
    frame_diffs = Counter()
    for utt, split, _ in items:
        p, reason, nf = results[utt]
        if p is not None:
            frame_diffs[p.raw_frame_sum - nf] += 1
        if reason is not None:
            reasons[reason] += 1
            dropped.append((utt, split, reason))
            continue
        r = lj_row(utt, texts[utt], split, cfg)
        r.update(words=[w.word for w in p.words], phones=p.phones, durations=p.durations,
                 word_ids=p.word_ids, n_frames=nf)
        assert sum(p.durations) == nf
        merged += p.n_merged_short_sil
        n_sp += p.phones.count("sp")
        zero_phone += p.extra.get("zero_frame_fixed", 0)
        assert all(d >= 1 for ph, d in zip(p.phones, p.durations) if ph not in ("sil", "sp"))
        rows[split].append(r)

    man = repo_path(cfg.paths.manifests, root)
    for s, rs in rows.items():
        write_jsonl(man / f"lj_{s}.jsonl", rs, force=True)
    align_dir = repo_path(cfg.paths.align, root)
    (align_dir / "lj_dropped.tsv").write_text(
        "id\tsplit\treason\n" + "".join(f"{u}\t{s}\t{r}\n" for u, s, r in dropped))

    total = len(items)
    kept = total - len(dropped)
    frac = kept / total
    need = cfg.align.qc.min_kept_fraction
    rep = [f"# LJSpeech alignment (T4, `make align-lj`, {date.today().isoformat()})", "",
           f"- MFA {cfg.mfa.acoustic_model.name} {cfg.mfa.acoustic_model.version}, "
           f"`--single_speaker -j {cfg.mfa.num_jobs} "
           + " ".join(f"--{k} {v}" for k, v in cfg.mfa.passthrough.items()) + " --clean`"
           + (f"; MFA wall time {mfa_secs / 60:.1f} min" if mfa_secs else ""),
           f"- **Kept {kept} / {total} ({100 * frac:.2f}%)**; required >= {100 * need:.0f}% -> "
           + ("PASS" if frac >= need else "FAIL"),
           "- Kept per split: " + ", ".join(f"{s} {len(v)}" for s, v in rows.items()),
           "- Every kept row: sum(durations) == n_frames (training spectrogram frame count)", "",
           "| Drop reason | Files |", "| --- | --- |"]
    labels = {"mfa_failed": "MFA produced no TextGrid", "spn": "contains `spn`",
              "long_phone": f"non-silence phone > {cfg.align.qc.max_phone_sec} s",
              "frame_diff": f"|duration sum - n_frames| > {cfg.align.qc.max_frame_diff} "
                            "before adjustment",
              "word_mismatch": "aligned words differ from transcript"}
    for k, v in reasons.most_common():
        rep.append(f"| {labels.get(k, k)} | {v} |")
    if not reasons:
        rep.append("| (none) | 0 |")
    rep += ["", "## Details", "",
            f"- Internal silences >= {cfg.align.min_sp_frames} frames kept as `sp`: {n_sp}; "
            f"shorter ones merged into the preceding phone: {merged}",
            f"- Speech phones that rounded to 0 frames and took 1 frame from the longer neighbour "
            f"(kept rows): {zero_phone}",
            "- Raw duration sum minus n_frames (all parsed files): "
            + ", ".join(f"{k:+d}: {v}" for k, v in sorted(frame_diffs.items())),
            f"- Dropped IDs: `{cfg.paths.align}/lj_dropped.tsv`"]
    report = repo_path(cfg.paths.reports, root) / "align_lj.md"
    report.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    return frac >= need


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LJSpeech MFA alignment (make align-lj)")
    ap.add_argument("--config", default="configs/splits.yaml")
    ap.add_argument("--stage", choices=["prep", "align", "parse", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="pilot on the first N IDs")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(repo_path(args.config), args.overrides)
    root = REPO_ROOT
    items = lj_items(cfg, root)
    pilot = args.limit > 0
    if pilot:
        items = items[: args.limit]
    corpus, out_dir = _dirs(cfg, pilot, root)

    mfa_secs = None
    if args.stage in ("prep", "all"):
        t = time.time()
        print(f"prep: {prep(cfg, items, corpus, root)} files -> {corpus} ({time.time() - t:.0f}s)")
    if args.stage in ("align", "all"):
        t = time.time()
        log = repo_path(cfg.paths.align, root) / f"{out_dir.name}_mfa.log"
        r = align_corpus(cfg, corpus, out_dir, single_speaker=True, root=root, log=log)
        mfa_secs = time.time() - t
        print(f"mfa align: exit {r.returncode} in {mfa_secs / 60:.1f} min (log: {log})")
        if r.returncode != 0:
            print(r.stderr[-3000:], file=sys.stderr)
            return 1
    if args.stage in ("parse", "all"):
        results, texts = parse_all(cfg, items, out_dir, root)
        if pilot:
            c = Counter(reason for _, reason, _ in results.values())
            print(f"pilot parse: {len(results)} files, reasons {dict(c)}")
            return 0
        return 0 if write_outputs(cfg, items, results, texts, root, mfa_secs) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
