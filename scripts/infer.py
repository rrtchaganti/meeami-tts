#!/usr/bin/env python
"""Time-aligned TTS inference (docs/PLAN.md Sections 5, 9).

    python scripts/infer.py --text "..." --ref ref.wav --out out/ --ckpt export/m1/ckpt_last.pt
    python scripts/infer.py --pairs_dir their_files/ --out out/ --ckpt export/m1/ckpt_last.pt

--ckpt m0 runs the FastSpeech2 baseline. Options: --config configs/inference.yaml,
--aligner mfa|ctc, --transfer elastic|proportional|phone_copy, --cap, --noise_scale,
--device cuda|cpu. Per pair: out.wav, timings.json, timing.png; prints per-stage latency and RTF.
Any failure prints a one-line reason and continues with the next pair.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# scripts/ must not be importable (scripts/profile.py would shadow the stdlib `profile`)
sys.path[0] = str(Path(__file__).resolve().parents[1])

from tts.config import load_config, repo_path  # noqa: E402

AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma", ".aiff")


def pairs_in(d: Path):
    for txt in sorted(d.glob("*.txt")):
        audio = [p for p in d.glob(txt.stem + ".*") if p.suffix.lower() in AUDIO_EXT]
        yield txt.stem, (audio[0] if audio else None), txt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--text")
    ap.add_argument("--ref")
    ap.add_argument("--pairs_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True, help="VITS-MFA checkpoint, or 'm0'")
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--aligner", choices=["mfa", "ctc"])
    ap.add_argument("--transfer", choices=["elastic", "proportional", "phone_copy"])
    ap.add_argument("--cap", default="cfg", help="float, or 'none'")
    ap.add_argument("--noise_scale", type=float)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args(argv)
    if not a.pairs_dir and not (a.text and a.ref):
        ap.error("give --text and --ref, or --pairs_dir")
    cfg = load_config(repo_path(a.config))
    from tts.infer.pipeline import Pipeline
    from tts.infer.systems import load_system

    system = load_system(a.ckpt, cfg, a.device)
    pipe = Pipeline(cfg, system)
    cap = a.cap if a.cap in ("cfg", "none") else float(a.cap)
    cap = None if cap == "none" else cap
    out = Path(a.out)
    jobs = ([(Path(a.ref).stem, Path(a.ref), a.text)] if a.text else
            [(n, r, t) for n, r, t in pairs_in(Path(a.pairs_dir))])
    n_ok = 0
    for name, ref, text in jobs:
        try:
            if ref is None:
                raise FileNotFoundError("no audio file next to the .txt")
            if isinstance(text, Path):
                text = text.read_text(encoding="utf-8").strip()
            d = out / name if a.pairs_dir else out
            info = pipe.run(text, ref, d, a.transfer, cap, a.noise_scale, a.aligner)
            lat = info["latency_s"]
            print(f"[ok] {name}: {info['out_samples']} samples (error {info['length_error_samples']}), "
                  f"aligner {info['aligner']}"
                  + (f" (MFA fallback: {info['mfa_fallback_reason']})" if info["mfa_fallback_reason"] else "")
                  + " | " + " ".join(f"{k} {v:.2f}s" for k, v in lat.items())
                  + f" | RTF model {info['rtf_model']:.3f} end-to-end {info['rtf_end_to_end']:.2f}")
            n_ok += 1
        except Exception as e:  # one-line reason, continue (Section 9)
            print(f"[fail] {name}: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
            if "--debug" in sys.argv:
                traceback.print_exc()
    print(f"{n_ok}/{len(jobs)} pairs synthesized")
    return 0 if n_ok == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
