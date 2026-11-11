#!/usr/bin/env python
"""Live interview demo (DECISIONS D-96).

    python scripts/demo.py                                   # the headline sentence (live_demo.default_utt)
    python scripts/demo.py --utt YBAA_arctic_a0003 [--text "For the 20th time ..."]
    python scripts/demo.py --ref their.mp3 --text "..."      # any audio file + its transcript
    python scripts/demo.py --suite                           # rehearsal: every case in live_demo.suite

For one input, every system in `live_demo.systems` (main, V0 by default) runs the full pipeline
(scripts/infer.py's Pipeline: normalize -> lexicon -> MFA on the reference -> elastic fit ->
synthesis). MFA runs once on the reference and its TextGrid is shared by the systems (the
pipeline's `ref_textgrid`, same align_one call). Then MFA aligns every output with the same words,
exactly as `make eval` does (single speaker, no adaptation), so the table shows where each word
really is in the output, not only where the pipeline placed it. --no_measure skips that step.
The terminal shows the normalized words, the per-word table, the aligner, the length check and
the latency; each system is warmed up once at load so the first call is not a CUDA cold start.
out/live_demo/<name>/ holds ref.<ext>, <system>/out.wav, timings.json, timing.png and an
index.html to play them side by side. --suite runs every rehearsal case and writes
out/live_demo/REHEARSAL.md; the exit code is 0 only if every case behaved as `expect` says.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
import time
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from tts.config import REPO_ROOT, load_config, repo_path  # noqa: E402


def utt_ref(cfg, utt: str) -> tuple[Path, str]:
    """L2-ARCTIC utterance id (SPK_arctic_xNNNN) -> (wav path, transcript)."""
    spk, _, name = utt.partition("_")
    root = repo_path(cfg.paths.raw_l2) / spk
    wav, txt = root / "wav" / f"{name}.wav", root / "transcript" / f"{name}.txt"
    if not wav.exists():
        raise FileNotFoundError(f"no L2-ARCTIC utterance {utt} ({wav})")
    return wav, txt.read_text(encoding="utf-8").strip()


def ckpt_for(spec: str) -> str:
    """infer.pt falls back to the full training checkpoint when only that is present."""
    p = repo_path(spec)
    if spec.endswith("infer.pt") and not p.exists() and p.with_name("ckpt_last.pt").exists():
        return str(p.with_name("ckpt_last.pt"))
    return spec


class Demo:
    def __init__(self, cfg, device: str):
        from tts.guard import assert_split_allowed
        from tts.infer.pipeline import Pipeline
        from tts.infer.systems import load_system

        assert_split_allowed("test", cfg)   # test-speaker audio only after freeze-v1
        self.cfg = cfg
        self.pipes = {}
        for name, spec in cfg.live_demo.systems.items():
            t = time.perf_counter()
            self.pipes[name] = Pipeline(cfg, load_system(ckpt_for(spec), cfg, device))
            warm_up(self.pipes[name].system)
            print(f"[demo] loaded {name} ({ckpt_for(spec)}) + warm-up in "
                  f"{time.perf_counter() - t:.1f}s")

    def align_reference(self, ref: Path, text: str, out: Path) -> tuple[Path | None, float]:
        """One MFA align_one on the reference (the call Pipeline.align makes), shared by all
        systems. None when it fails: each pipeline then makes its own attempt and falls back to
        CTC with the reason logged."""
        from tts.align.mfa import align_one
        from tts.audio import resample, write_wav16
        from tts.infer.pipeline import load_reference
        from tts.text.lexicon import ensure_words
        from tts.text.normalize import words

        pipe = next(iter(self.pipes.values()))
        t = time.perf_counter()
        ws = words(pipe.norm(text))
        if not ws:
            return None, 0.0
        ensure_words(ws, pipe.lexicon_path, pipe.g2p, self.cfg.lexicon.possessive_rule)
        y, sr = load_reference(ref, self.cfg)
        d = out / "_ref_align"
        d.mkdir(parents=True, exist_ok=True)
        sr16 = int(self.cfg.mfa.sample_rate)
        write_wav16(d / "ref.wav", resample(y, sr, sr16), sr16)
        (d / "ref.lab").write_text(" ".join(ws) + "\n")
        tg = out / "ref.TextGrid"
        tg.unlink(missing_ok=True)
        r = align_one(self.cfg, d / "ref.wav", d / "ref.lab", tg)
        shutil.rmtree(d, ignore_errors=True)
        return (tg if r.returncode == 0 and tg.exists() else None), time.perf_counter() - t

    def measure(self, out: Path, words: list[str]) -> tuple[dict, float]:
        """MFA on every system's out.wav with the input words (as `make eval`): measured word
        times per system, {} for a system MFA could not align."""
        from tts.align.lj import _prep_one
        from tts.align.mfa import align_corpus
        from tts.align.textgrid import read_tiers

        t = time.perf_counter()
        corpus, aligned = out / "_out_corpus", out / "align_out"
        shutil.rmtree(corpus, ignore_errors=True)
        shutil.rmtree(aligned, ignore_errors=True)
        corpus.mkdir(parents=True)
        for name in self.pipes:
            _prep_one((str(out / name / "out.wav"), str(corpus / f"{name}.wav"),
                       str(corpus / f"{name}.lab"), " ".join(words), int(self.cfg.mfa.sample_rate)))
        align_corpus(self.cfg, corpus, aligned, single_speaker=True, log=out / "mfa_out.log")
        shutil.rmtree(corpus, ignore_errors=True)
        silence = set(self.cfg.align.silence_labels)
        got = {}
        for name in self.pipes:
            try:
                w, _ = read_tiers(aligned / f"{name}.TextGrid")
                got[name] = [{"word": x.label.lower(), "start": x.start, "end": x.end} for x in w
                             if x.label.strip() and x.label not in silence and x.label != "spn"]
            except Exception:
                got[name] = []
        return got, time.perf_counter() - t

    def run(self, ref: Path, text: str, out: Path, quiet: bool = False,
            measure: bool = True) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        ref_copy = out / f"ref{ref.suffix.lower()}"
        shutil.copyfile(ref, ref_copy)
        tg, t_align = self.align_reference(ref, text, out)
        infos = {}
        for name, pipe in self.pipes.items():
            # the laptop GPU clocks down during the ~13 s MFA wait; re-warm so the latency shown
            # is the model's, not the power state's (one warm process: RTF model ~0.02)
            warm_up(pipe.system)
            info = infos[name] = pipe.run(text, ref, out / name, ref_textgrid=tg)
            if tg is not None:   # the shared alignment's time is part of every system's latency
                info["latency_s"]["align"] += t_align
                info["latency_s"]["total"] += t_align
                info["rtf_end_to_end"] = info["latency_s"]["total"] / (
                    info["out_samples"] / pipe.system.sample_rate)
                info["notes"].append(f"reference aligned once by scripts/demo.py ({t_align:.2f}s, "
                                     "shared by all systems; included in align/total)")
                (out / name / "timings.json").write_text(json.dumps(info, indent=1) + "\n")
        measured, t_meas = ({}, 0.0)
        if measure:
            from tts.text.normalize import words
            measured, t_meas = self.measure(out, words(infos[next(iter(infos))]["normalized"]))
        if not quiet:
            report(text, infos, measured, t_meas)
        write_index(out, ref_copy, text, infos)
        return infos


def warm_up(system) -> None:
    """One short synthesis so the first real call is not a CUDA/cuDNN cold start."""
    import numpy as np

    # ~3 s of speech, twice: cuDNN picks its kernels per input shape on the first calls
    toks = ["sil"] + ["HH", "AH0", "L", "OW1", "sp"] * 8 + ["sil"]
    wid = [-1] + [w for i in range(8) for w in (i, i, i, i, -1)] + [-1]
    kw = {"word_ids": wid} if getattr(system, "uses_word_ids", False) else {}
    try:
        for _ in range(2):
            system.predict_log_durations(toks, **kw)
            system.synthesize(toks, np.array([10] + [6, 4, 6, 12, 5] * 8 + [10]), **kw)
    except Exception as e:  # warm-up must never stop the demo
        print(f"[demo] warm-up skipped: {type(e).__name__}: {e}")


def report(text: str, infos: dict, measured: dict, t_meas: float) -> None:
    from tts.eval.metrics import match_words

    first = next(iter(infos.values()))
    print(f"\ntext       : {text!r}\nnormalized : {first['normalized']}")
    if first["oov_added"]:
        print(f"g2p (new)  : {first['oov_added']}")
    names = list(infos)
    ref = first["ref_words"]
    # measured[n][j] matched to reference word i (same word sequence; match_words handles gaps)
    m = {n: dict(match_words(ref, measured.get(n) or [])) for n in names}
    cols = "".join(f" | {n + ' measured':>17} {'Δstart':>7} {'Δend':>6}" for n in names)
    head = f"{'word':12} {'reference (MFA)':>15}{cols}"
    print(head + "\n" + "-" * len(head))
    errs = {n: [] for n in names}
    for i, w in enumerate(ref):
        row = f"{w['word']:12} {w['start']:6.2f}-{w['end']:<6.2f}s "
        for n in names:
            j = m[n].get(i)
            if j is None:
                row += f" | {'(not found)':>17} {'':>7} {'':>6}"
                continue
            o = measured[n][j]
            ds, de = 1000 * (o["start"] - w["start"]), 1000 * (o["end"] - w["end"])
            errs[n] += [abs(ds), abs(de)]
            row += f" | {o['start']:6.2f}-{o['end']:<6.2f}s    {ds:+7.0f} {de:+6.0f}"
        print(row)
    if measured:
        print("Δ ms = measured output time - reference time (MFA on both, as `make eval`); "
              "mean |Δ|: " + ", ".join(f"{n} {sum(e) / len(e):.0f} ms" if e else f"{n} n/a"
                                      for n, e in errs.items())
              + f"  [output alignment {t_meas:.1f}s, not part of synthesis]")
    for n, info in infos.items():
        lat = info["latency_s"]
        print(f"{n:5}: aligner {info['aligner']}"
              + (f" (MFA failed: {info['mfa_fallback_reason']})" if info["mfa_fallback_reason"] else "")
              + f" | {info['out_samples']} samples, length error {info['length_error_samples']}"
              + f" | capped {info['capped_words'] or 'none'}"
              + f" | total {lat['total']:.2f}s (MFA {lat.get('align', 0):.2f}s), "
                f"RTF model {info['rtf_model']:.3f} end-to-end {info['rtf_end_to_end']:.2f}")


def write_index(out: Path, ref: Path, text: str, infos: dict) -> None:
    first = next(iter(infos.values()))
    rows = [f"<h2>{html.escape(text)}</h2><p>normalized: <code>{html.escape(first['normalized'])}"
            f"</code></p><h3>Reference</h3><audio controls src='{ref.name}'></audio>"]
    for n, info in infos.items():
        rows.append(f"<h3>{n} (aligner {info['aligner']}, length error "
                    f"{info['length_error_samples']} samples)</h3>"
                    f"<audio controls src='{n}/out.wav'></audio><br><img src='{n}/timing.png' width='900'>")
    (out / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>meeami-tts demo"
                                    "</title><body style='font-family:sans-serif'>" + "".join(rows)
                                    + "</body>")


def suite(demo: Demo, cfg, out_root: Path) -> int:
    lines = ["# Live-demo rehearsal (scripts/demo.py --suite)", "",
             "| # | input | text | expected | result | aligner | length error | MFA s | total s | note |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    n_bad = 0
    for k, c in enumerate(cfg.live_demo.suite, 1):
        name = c.get("utt") or Path(c.ref).stem
        res, al, le, mfa, tot, note = "ok", "", "", "", "", ""
        try:
            ref, _ = utt_ref(cfg, c.utt) if c.get("utt") else (repo_path(c.ref), None)
            if not ref.exists():
                raise FileNotFoundError(f"no audio file {c.ref}")
            infos = demo.run(ref, c.text, out_root / f"case{k:02d}_{name}", quiet=True,
                             measure=False)
            main = next(iter(infos.values()))
            al = ",".join(sorted({i["aligner"] for i in infos.values()}))
            le = ",".join(str(i["length_error_samples"]) for i in infos.values())
            mfa = f"{main['latency_s'].get('align', 0):.1f}"
            tot = f"{main['latency_s']['total']:.1f}"
            note = main["normalized"]
            exact = all(i["length_error_samples"] == 0 for i in infos.values())
            mfa_used = all(i["aligner"] == "mfa" for i in infos.values())
            good = exact and (mfa_used or c.expect == "ok_any_aligner") and c.expect != "refused"
            if main["mfa_fallback_reason"]:
                note += f" (MFA: {main['mfa_fallback_reason']})"
        except Exception as e:  # refusals are one line, like scripts/infer.py
            res, note = "refused", f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
            good = c.expect == "refused"
        n_bad += not good
        verdict = "PASS" if good else "FAIL"
        print(f"[{verdict}] case {k:2d} {name}: {res} {al} | {note[:110]}")
        lines.append(f"| {k} | {name} | `{c.text.strip()}` | {c.expect} | {res} ({verdict}) | {al} "
                     f"| {le} | {mfa} | {tot} | {note.replace('|', '/')} |")
    (out_root / "REHEARSAL.md").write_text("\n".join(lines) + "\n")
    print(f"{len(cfg.live_demo.suite) - n_bad}/{len(cfg.live_demo.suite)} cases as expected; "
          f"{out_root / 'REHEARSAL.md'}")
    return 1 if n_bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--utt", help="L2-ARCTIC utterance id, e.g. HKK_arctic_a0003")
    ap.add_argument("--ref", help="any audio file (needs --text)")
    ap.add_argument("--text", help="what the reference says (default: the ARCTIC transcript)")
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--no_measure", action="store_true", help="skip MFA on the outputs")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "live_demo.yaml"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    out_root = repo_path(cfg.live_demo.out_dir)
    demo = Demo(cfg, a.device)
    if a.suite:
        return suite(demo, cfg, out_root)
    if a.ref:
        if not a.text:
            ap.error("--ref needs --text (the transcript of the audio)")
        ref, text, name = Path(a.ref), a.text, Path(a.ref).stem
    else:
        utt = a.utt or cfg.live_demo.default_utt
        ref, transcript = utt_ref(cfg, utt)
        text, name = a.text or transcript, utt
    try:
        demo.run(ref, text, out_root / name, measure=not a.no_measure)
    except Exception as e:
        print(f"[fail] {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        return 1
    print(f"\nplay: {out_root / name / 'index.html'}  (WSL: explorer.exe \"$(wslpath -w "
          f"{out_root / name / 'index.html'})\")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
