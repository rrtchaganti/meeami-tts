"""Run the inference pipeline over L2 reference rows (m0-demo, evaluation; Sections 5, 8).

    python -m tts.infer.batch --system m0|v0|<ckpt> --split dev [--n 5] [--eval_set] --out DIR

References are pre-aligned in one `mfa align --single_speaker` pass (same acoustic model, lexicon
and beam as `mfa align_one`, and like align_one without speaker adaptation; D-51), then each row
goes through Pipeline.run with that TextGrid. --split test requires the freeze tag.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from omegaconf import DictConfig

from tts.align.lj import _prep_one
from tts.align.mfa import align_corpus
from tts.config import REPO_ROOT, load_config, repo_path
from tts.guard import assert_split_allowed
from tts.stats import read_jsonl
from tts.text.lexicon import ensure_words
from tts.text.normalize import Normalizer, words


def select_rows(cfg: DictConfig, split: str, n: int = 0, eval_set: bool = True) -> list[dict]:
    assert_split_allowed(split, cfg)
    rows = read_jsonl(repo_path(cfg.paths.manifests) / f"l2_{split}.jsonl")
    if eval_set:
        rows = [r for r in rows if r.get("in_eval_set")]
    rows.sort(key=lambda r: (r["utt"], r["speaker"]))
    if n:  # round-robin over accents, evenly spaced within each (deterministic)
        by_acc: dict[str, list[dict]] = {}
        for r in rows:
            by_acc.setdefault(r["accent"], []).append(r)
        accs = sorted(by_acc)
        pick = []
        for i in range(min(n, len(rows))):
            g = by_acc[accs[i % len(accs)]]
            j = i // len(accs)
            per = -(-n // len(accs))
            pick.append(g[(j * len(g)) // per])
        rows = pick
    return rows


def prealign(cfg: DictConfig, rows: list[dict], work: Path, g2p) -> dict[str, Path]:
    norm = Normalizer(cfg)
    corpus, out = work / "corpus", work / "aligned"
    shutil.rmtree(corpus, ignore_errors=True)
    corpus.mkdir(parents=True)
    all_words = []
    for r in rows:
        w = words(norm(r["text"]))
        all_words += w
        _prep_one((str(repo_path(r["wav"])), str(corpus / f"{r['id']}.wav"),
                   str(corpus / f"{r['id']}.lab"), " ".join(w), int(cfg.mfa.sample_rate)))
    ensure_words(all_words, repo_path(cfg.paths.lexicon), g2p, cfg.lexicon.possessive_rule)
    res = align_corpus(cfg, corpus, out, single_speaker=True, log=work / "mfa.log")
    if res.returncode != 0:
        print(f"batch pre-alignment failed (exit {res.returncode}); rows fall back to align_one",
              file=sys.stderr)
        return {}
    shutil.rmtree(corpus, ignore_errors=True)
    return {r["id"]: out / f"{r['id']}.TextGrid" for r in rows}


def run_batch(pipe, rows: list[dict], out_root: Path, prealigned: dict[str, Path] | None = None,
              **kw) -> list[dict]:
    infos = []
    for r in rows:
        d = out_root / r["id"]
        try:
            info = pipe.run(r["text"], repo_path(r["wav"]), d,
                            ref_textgrid=(prealigned or {}).get(r["id"]), **kw)
            info.update(id=r["id"], speaker=r["speaker"], accent=r["accent"], status="ok")
            print(f"[ok] {r['id']}: aligner {info['aligner']}, length error "
                  f"{info['length_error_samples']}", flush=True)
        except Exception as e:
            info = {"id": r["id"], "speaker": r["speaker"], "accent": r["accent"],
                    "status": f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"}
            print(f"[fail] {r['id']}: {info['status']}", flush=True)
        infos.append(info)
    return infos


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", required=True, help="m0, v0 or a VITS-MFA checkpoint")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(repo_path(a.config), a.overrides)
    from tts.infer.pipeline import Pipeline
    from tts.infer.systems import load_system

    rows = select_rows(cfg, a.split, a.n)
    pipe = Pipeline(cfg, load_system(a.system, cfg, a.device))
    out = repo_path(a.out)
    pre = prealign(cfg, rows, out / "_ref_align", pipe.g2p)
    infos = run_batch(pipe, rows, out, pre)
    (out / "batch.json").write_text(json.dumps(
        [{k: v for k, v in i.items() if k not in ("tokens", "durations")} for i in infos], indent=1))
    ok = sum(i["status"] == "ok" for i in infos)
    print(f"{ok}/{len(infos)} ok -> {out}")
    return 0 if ok == len(infos) else 1


if __name__ == "__main__":
    sys.exit(main())
