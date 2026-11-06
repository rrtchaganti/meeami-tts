"""Build the committed edge-case suite (docs/PLAN.md Section 9) from dev-speaker recordings.

    python -m tts.infer.edge_cases --build      # writes tests/edge_cases/<case>.{wav|mp3} + .txt
    python -m tts.infer.edge_cases --run --ckpt m0 [--device cuda]   # `make edge`

Only dev speakers are used (test audio is locked). Pass = every case produces output of exact
length, or a clean fallback with the reason logged (Section 9).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from tts.align.textgrid import read_tiers
from tts.config import REPO_ROOT, load_config, repo_path
from tts.guard import assert_audio_allowed

EDGE = REPO_ROOT / "tests" / "edge_cases"
L2 = "data/raw/l2arctic"

# (case, speaker, utt, text written for the TTS input)
TEXT_CASES = [
    ("norm_number", "BWC", "arctic_a0031", "They were 300 yards apart."),
    ("norm_currency", "BWC", "arctic_b0260", "It's worth $8."),
    ("norm_date", "BWC", "arctic_a0439", "At sea Wednesday March 18th, 1908."),
    ("norm_abbrev_name", "BWC", "arctic_b0326", "Our Mr. Howison will call upon you at your hotel."),
    ("norm_ordinal", "BWC", "arctic_a0003", "For the 20th time that evening the two men shook hands."),
    ("norm_oov_name", "BWC", "arctic_a0276",
     "Oolong Atoll was one hundred and forty miles in circumference."),
    ("norm_no_punct", "NJS", "arctic_a0005", None),  # transcript lower-cased, punctuation removed
]


def _src(cfg, spk, utt):
    assert_audio_allowed(spk, cfg)
    return repo_path(f"{L2}/{spk}/wav/{utt}.wav"), \
        repo_path(f"{L2}/{spk}/transcript/{utt}.txt").read_text().strip()


def _sox(*args):
    subprocess.run(["sox", *map(str, args)], check=True, capture_output=True)


def _ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *map(str, args)], check=True)


def build(cfg) -> list[dict]:
    if EDGE.exists():
        for p in EDGE.iterdir():
            if p.name != ".gitkeep":
                p.unlink()
    EDGE.mkdir(parents=True, exist_ok=True)
    cases = []

    def add(name, audio_path, text, kind, expect="exact_length", aligner=None):
        (EDGE / f"{name}.txt").write_text(text + "\n")
        c = {"case": name, "audio": audio_path.name, "kind": kind, "expect": expect}
        if aligner:
            c["aligner"] = aligner
        cases.append(c)

    for name, spk, utt, text in TEXT_CASES:
        wav, tr = _src(cfg, spk, utt)
        dst = EDGE / f"{name}.wav"
        shutil.copy(wav, dst)
        add(name, dst, text or tr.lower(), "text normalization")

    base, base_txt = _src(cfg, "NJS", "arctic_a0010")
    for name, fn in [
        ("fmt_mp3", lambda d: _ff("-i", base, "-b:a", "64k", d)),
        ("fmt_8k", lambda d: _sox(base, "-r", "8000", d)),
        ("fmt_48k", lambda d: _sox(base, "-r", "48000", d)),
        ("fmt_stereo", lambda d: _sox(base, "-c", "2", d)),
        ("fmt_clipped", lambda d: _sox("-G", base, d, "gain", "18")),
        ("fmt_noisy", None),
        ("fmt_reverb", lambda d: _sox(base, d, "reverb", "80", "50", "100")),
    ]:
        dst = EDGE / (f"{name}.mp3" if name == "fmt_mp3" else f"{name}.wav")
        if name == "fmt_noisy":
            y, sr = sf.read(base, dtype="float32")
            rng = np.random.default_rng(0)
            noise = rng.standard_normal(len(y)).astype(np.float32)
            snr = 10.0
            noise *= np.sqrt((y ** 2).mean() / (noise ** 2).mean() / 10 ** (snr / 10))
            sf.write(dst, np.clip(y + noise, -1, 1), sr, subtype="PCM_16")
        elif name == "fmt_clipped":
            _sox(base, dst, "gain", "18")  # hard clipping, sox warns
        else:
            fn(dst)
        add(name, dst, base_txt, "audio format")

    # timing: > 20 s (4 utterances of one speaker), long pauses, fast speech, one word
    parts = [_src(cfg, "ASI", f"arctic_a{k:04d}") for k in range(10, 17)]  # 7 utts, > 20 s
    ys, sr = [], None
    for w, _ in parts:
        y, sr = sf.read(w, dtype="float32")
        ys.append(y)
    long = np.concatenate(ys)
    sf.write(EDGE / "timing_over_20s.wav", long, sr, subtype="PCM_16")
    add("timing_over_20s", EDGE / "timing_over_20s.wav", " ".join(t for _, t in parts), "timing")
    assert len(long) / sr > 20, "over-20s case must exceed 20 s"

    y1, _ = sf.read(parts[0][0], dtype="float32")
    y2, _ = sf.read(parts[1][0], dtype="float32")
    tone = y1[: int(0.05 * sr)]  # the recording's own leading room tone
    gap = np.tile(tone, int(np.ceil(1.8 * sr / len(tone))))[: int(1.8 * sr)]
    sf.write(EDGE / "timing_long_pause.wav", np.concatenate([y1, gap, y2]), sr, subtype="PCM_16")
    add("timing_long_pause", EDGE / "timing_long_pause.wav", f"{parts[0][1]} {parts[1][1]}", "timing")

    _sox(parts[2][0], EDGE / "timing_fast.wav", "tempo", "-s", "1.6")
    add("timing_fast", EDGE / "timing_fast.wav", parts[2][1], "timing")

    # one word, cut with the manual annotation word tier (dev speaker)
    wav, _ = _src(cfg, "ASI", "arctic_a0003")
    words, _ = read_tiers(repo_path(f"{L2}/ASI/annotation/arctic_a0003.TextGrid"))
    w = [x for x in words if x.label.strip() == "evening"][0]
    y, sr = sf.read(wav, dtype="float32")
    pad = int(0.15 * sr)
    seg = y[max(0, int(w.start * sr) - pad): int(w.end * sr) + pad]
    sf.write(EDGE / "timing_one_word.wav", seg, sr, subtype="PCM_16")
    add("timing_one_word", EDGE / "timing_one_word.wav", "Evening.", "timing")

    # alignment failure: a transcript word that is not spoken; a false start
    wav, tr = _src(cfg, "NJS", "arctic_a0011")
    shutil.copy(wav, EDGE / "align_unspoken_word.wav")
    tw = tr.split()
    add("align_unspoken_word", EDGE / "align_unspoken_word.wav",
        " ".join(tw[:2] + ["certainly"] + tw[2:]), "alignment failure", "exact_length_or_fallback")
    wav, tr = _src(cfg, "NJS", "arctic_a0012")
    y, sr = sf.read(wav, dtype="float32")
    first = y[: int(0.45 * sr)]
    sil = np.tile(y[: int(0.04 * sr)], 10)
    sf.write(EDGE / "align_false_start.wav", np.concatenate([first, sil, y]), sr, subtype="PCM_16")
    add("align_false_start", EDGE / "align_false_start.wav", tr, "alignment failure",
        "exact_length_or_fallback")

    # the CTC (MMS_FA) aligner path, forced (the automatic MFA->CTC fallback is unit-tested)
    wav, tr = _src(cfg, "NJS", "arctic_a0013")
    shutil.copy(wav, EDGE / "align_ctc_path.wav")
    add("align_ctc_path", EDGE / "align_ctc_path.wav", tr, "alignment failure", aligner="ctc")

    (EDGE / "cases.json").write_text(json.dumps(cases, indent=1) + "\n")
    (EDGE / "SOURCE.md").write_text(
        "# Edge-case suite sources\n\nBuilt by `python -m tts.infer.edge_cases --build` from "
        "L2-ARCTIC v5 dev speakers (BWC, NJS, ASI; CC BY-NC 4.0), converted with sox/ffmpeg. "
        "No test-accent audio. See cases.json for the case list.\n")
    return cases


def run(cfg, ckpt: str, device: str, out: Path) -> int:
    from tts.infer.pipeline import Pipeline
    from tts.infer.systems import load_system

    cases = json.loads((EDGE / "cases.json").read_text())
    pipe = Pipeline(cfg, load_system(ckpt, cfg, device))
    results, n_pass = [], 0
    for c in cases:
        text = (EDGE / f"{c['case']}.txt").read_text().strip()
        r = {"case": c["case"], "kind": c["kind"]}
        try:
            info = pipe.run(text, EDGE / c["audio"], out / c["case"], aligner=c.get("aligner"))
            ok = info["length_error_samples"] == 0
            r.update(ok=ok, aligner=info["aligner"], fallback=info["mfa_fallback_reason"],
                     length_error=info["length_error_samples"], capped=info["capped_words"],
                     rtf=round(info["rtf_end_to_end"], 2))
        except Exception as e:
            r.update(ok=False, error=f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        n_pass += r["ok"]
        print(("[pass] " if r["ok"] else "[FAIL] ") + json.dumps(r), flush=True)
        results.append(r)
    (out / "edge_results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"edge: {n_pass}/{len(cases)} pass (device {device})")
    return 0 if n_pass == len(cases) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--ckpt", default="m0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="out/edge")
    ap.add_argument("--config", default="configs/inference.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(repo_path(a.config))
    if a.build:
        cases = build(cfg)
        print(f"built {len(cases)} cases in {EDGE.relative_to(REPO_ROOT)}")
    if a.run:
        return run(cfg, a.ckpt, a.device, repo_path(a.out) / a.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
