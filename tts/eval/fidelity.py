"""Duration fidelity and aligner calibration (DECISIONS D-90).

The word-boundary MAE of Section 8 compares the output with the manual annotation of the
reference. It mixes three error sources: the timing the pipeline asked for (MFA on the reference,
the same for every system), how faithfully the model renders the durations it is given, and the
error of the aligner used on the output. This module separates the last two:

* **fidelity**: for every word, the signed difference between where the aligner finds it in the
  output and where the pipeline placed it (`out_words` in timings.json). Reported as median bias
  (start, end), MAE, and the MAE left after removing the median bias (spread). It is computed for
  MFA (`align_out/*.TextGrid`) and for CTC (`ctc_words` in scores.jsonl).
* **calibration** (`--calibrate`): the same signed statistics for each aligner on the natural
  dev reference recordings against the manual annotation. A judge's own bias on real speech is
  the zero point for reading its bias on synthetic speech.

    python -m tts.eval.fidelity --split dev --system v0 --system m1 [--calibrate]
Writes results/fidelity/<split>.json (and results/fidelity/calibration_dev.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tts.config import REPO_ROOT, load_config, repo_path
from tts.eval.metrics import match_words


def signed_errors(ref: list[dict], hyp: list[dict]) -> tuple[list[float], list[float]]:
    """Signed start and end differences hyp - ref (s) for every matched word."""
    st, en = [], []
    for i, j in match_words(ref, hyp):
        st.append(hyp[j]["start"] - ref[i]["start"])
        en.append(hyp[j]["end"] - ref[i]["end"])
    return st, en


def summarize(st: list[float], en: list[float]) -> dict:
    s, e = np.asarray(st) * 1000, np.asarray(en) * 1000
    if len(s) == 0:
        return {"n_words": 0}
    bs, be = float(np.median(s)), float(np.median(e))
    both = np.r_[s, e]
    return {"n_words": int(len(s)), "bias_start_ms": bs, "bias_end_ms": be,
            "mae_ms": float(np.abs(both).mean()),
            "mae_unbiased_ms": float(np.abs(np.r_[s - bs, e - be]).mean())}


def _textgrid_words(p: Path, silence: set) -> list[dict] | None:
    from tts.align.textgrid import read_tiers

    try:
        w, _ = read_tiers(p)
    except Exception:
        return None
    return [{"word": x.label.lower(), "start": x.start, "end": x.end} for x in w
            if x.label.strip() and x.label not in silence and x.label != "spn"]


def system_fidelity(res: Path, silence: set) -> dict:
    """MFA and CTC fidelity of one results/<system>/<split> directory."""
    ctc = {}
    sc = res / "scores.jsonl"
    if sc.exists():
        for line in sc.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                ctc[d["id"]] = d.get("ctc_words")
    acc = {"mfa": ([], []), "ctc": ([], [])}
    for tj in sorted((res / "wav").glob("*/timings.json")):
        uid = tj.parent.name
        intended = json.loads(tj.read_text())["out_words"]
        for k, hyp in (("mfa", _textgrid_words(res / "align_out" / f"{uid}.TextGrid", silence)),
                       ("ctc", ctc.get(uid))):
            if hyp:
                s, e = signed_errors(intended, hyp)
                acc[k][0].extend(s), acc[k][1].extend(e)
    return {k: summarize(*v) for k, v in acc.items()}


def calibrate(cfg, device: str) -> dict:
    """MFA (reference alignment used by the pipeline) and CTC on the natural dev references vs
    the manual annotation. Dev speakers only."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import evaluate as ev  # manual_words

    from tts.audio import load_wav
    from tts.eval.assets import torch_home
    from tts.infer.batch import select_rows
    from tts.infer.ctc import ctc_word_times
    from tts.text.normalize import Normalizer, words

    rows = select_rows(cfg, "dev", 0)
    norm, th = Normalizer(cfg), str(torch_home(cfg))
    ref_dir = repo_path(cfg.fidelity.ref_timings_from) / "dev" / "wav"
    acc = {"mfa": ([], []), "ctc": ([], [])}
    for r in rows:
        man = ev.manual_words(r, cfg)
        tj = ref_dir / r["id"] / "timings.json"
        mfa = json.loads(tj.read_text())["ref_words"] if tj.exists() else None
        y, sr = load_wav(repo_path(r["wav"]))
        ctc = ctc_word_times(y, sr, words(norm(r["text"])), th, device,
                             float(cfg.eval.min_pause_s))
        for k, hyp in (("mfa", mfa), ("ctc", ctc)):
            if hyp:
                s, e = signed_errors(man, hyp)
                acc[k][0].extend(s), acc[k][1].extend(e)
    return {k: summarize(*v) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", action="append", default=[])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "fidelity.yaml"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    from tts.guard import assert_split_allowed

    assert_split_allowed(a.split, cfg)
    out = repo_path(cfg.fidelity.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    silence = set(cfg.align.silence_labels)
    if a.system:
        res = {s: system_fidelity(repo_path("results") / s / a.split, silence) for s in a.system}
        (out / f"{a.split}.json").write_text(json.dumps(res, indent=1))
        for s, d in res.items():
            for k, v in d.items():
                if v.get("n_words"):
                    print(f"{a.split} {s:16s} {k}: bias start {v['bias_start_ms']:+6.1f} end "
                          f"{v['bias_end_ms']:+6.1f} | MAE {v['mae_ms']:5.1f} | "
                          f"MAE without bias {v['mae_unbiased_ms']:5.1f} ms (n={v['n_words']})")
    if a.calibrate:
        c = calibrate(cfg, a.device)
        (out / "calibration_dev.json").write_text(json.dumps(c, indent=1))
        for k, v in c.items():
            print(f"natural dev speech, {k} vs manual: bias start {v['bias_start_ms']:+6.1f} end "
                  f"{v['bias_end_ms']:+6.1f} | MAE {v['mae_ms']:5.1f} | "
                  f"MAE without bias {v['mae_unbiased_ms']:5.1f} ms (n={v['n_words']})")


if __name__ == "__main__":
    main()
