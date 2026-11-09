#!/usr/bin/env python
"""Static demo page (docs/PLAN.md Section 9 "Deliverables"; `make demo`).

    python scripts/make_demo.py [--n_per_accent 2] [--out out/demo_site]

Systems: `demo.systems` (configs/base.yaml); `main_system` is marked. Samples: test accents only after the freeze tag,
otherwise dev accents (labelled "dev"); suitcase clips of allowed speakers and own recordings
(demo/own_recordings/*.wav + .txt) labelled "stress test". For each sample and system the pipeline
output (out.wav, timings.json, timing.png) is kept under out/demo_site/samples/ — these are also
the backup outputs for the live demo. index.html links reference and output audio and the plots,
and a results table from results/<system>/<split>/summary.json when present.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

# scripts/ must not be importable (scripts/profile.py would shadow the stdlib `profile`)
sys.path[0] = str(Path(__file__).resolve().parents[1])

from tts.config import REPO_ROOT, load_config, repo_path  # noqa: E402
from tts.guard import accent_of, freeze_tag_exists  # noqa: E402


def systems(cfg) -> list[tuple[str, str]]:
    """cfg.demo.systems in order: m0, or export/<name>/ckpt_last.pt when it exists."""
    out = []
    for name in cfg.demo.systems:
        if name == "m0":
            out.append(("m0", "m0"))
        elif (repo_path("export") / name / "ckpt_last.pt").exists():
            out.append((name, str(repo_path("export") / name / "ckpt_last.pt")))
        else:
            print(f"[demo] skipping {name}: no export/{name}/ckpt_last.pt")
    return out


def samples(cfg, n_per_accent: int) -> list[dict]:
    from tts.infer.batch import select_rows

    split = "test" if freeze_tag_exists(cfg.freeze_tag) else "dev"
    rows = select_rows(cfg, split, n_per_accent * 3)
    out = [{"id": r["id"], "label": split, "accent": r["accent"], "speaker": r["speaker"],
            "text": r["text"], "ref": repo_path(r["wav"])} for r in rows]
    acc = accent_of(cfg)
    suit = repo_path(cfg.paths.raw_l2) / cfg.l2.suitcase_dir
    for spk in cfg.demo.suitcase_speakers:
        split_s, accent = acc[spk]
        if split_s == "test" and not freeze_tag_exists(cfg.freeze_tag):
            continue
        wav = suit / "wav" / f"{spk.lower()}.wav"
        txt = suit / "transcript" / f"{spk.lower()}.txt"
        if wav.exists() and txt.exists():
            out.append({"id": f"suitcase_{spk}", "label": "stress test", "accent": accent,
                        "speaker": spk, "text": txt.read_text().strip(), "ref": wav})
    own = repo_path(cfg.demo.own_recordings)
    if own.exists():
        for txt in sorted(own.glob("*.txt")):
            wav = next((p for p in own.glob(txt.stem + ".*") if p.suffix != ".txt"), None)
            if wav:
                out.append({"id": f"own_{txt.stem}", "label": "stress test", "accent": "own",
                            "speaker": "own", "text": txt.read_text().strip(), "ref": wav})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--out", default="out/demo_site")
    ap.add_argument("--n_per_accent", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    cfg = load_config(repo_path(a.config))
    n = a.n_per_accent or int(cfg.demo.n_per_accent)
    site = repo_path(a.out)
    (site / "samples").mkdir(parents=True, exist_ok=True)
    from tts.infer.pipeline import Pipeline
    from tts.infer.systems import load_system

    smp = samples(cfg, n)
    syss = systems(cfg)
    status = {}
    for name, ckpt in syss:
        pipe = Pipeline(cfg, load_system(ckpt, cfg, a.device))
        for s in smp:
            d = site / "samples" / s["id"] / name
            if (d / "out.wav").exists():
                status[(s["id"], name)] = "ok"
                continue
            try:
                pipe.run(s["text"], s["ref"], d)
                status[(s["id"], name)] = "ok"
            except Exception as e:
                status[(s["id"], name)] = f"failed: {type(e).__name__}: {str(e)[:120]}"
                print(f"[demo] {s['id']}/{name}: {status[(s['id'], name)]}")
        del pipe
    for s in smp:
        dst = site / "samples" / s["id"] / ("reference" + s["ref"].suffix.lower())
        if not dst.exists():
            shutil.copy(s["ref"], dst)

    # results tables
    tables = []
    for split in ("dev", "test"):
        rows = []
        for name, _ in syss:
            p = repo_path("results") / name / split / "summary.json"
            if p.exists():
                m = json.loads(p.read_text())
                rows.append((name, m))
        if rows:
            keys = [("wb_mae_ms", "Word-boundary MAE (ms)"), ("length_error", "Length error"),
                    ("pause_f1", "Pause F1"), ("wer", "WER"), ("us_prob", "US prob."),
                    ("spk_sim_lj", "Spk sim LJ"), ("spk_sim_ref", "Spk sim ref"), ("utmos", "UTMOS")]
            h = ["<table><tr><th>System</th>" + "".join(f"<th>{t}</th>" for _, t in keys) + "</tr>"]
            for name, m in rows:
                cells = []
                for k, _ in keys:
                    v = m["metrics"]["all"][k]
                    mean, (lo, hi) = v["mean"], v["ci95"]
                    cells.append("<td>-</td>" if mean != mean else
                                 f"<td>{mean:.3f}<br><small>[{lo:.3f}, {hi:.3f}]</small></td>")
                lab = f"<b>{name}</b> (main)" if name == cfg.main_system else name
                h.append(f"<tr><td>{lab}</td>{''.join(cells)}</tr>")
            h.append("</table>")
            tables.append(f"<h2>Results ({split}, 95% bootstrap CI)</h2>" + "".join(h))

    plots = sorted((repo_path("results") / "plots").glob("*.png")) if \
        (repo_path("results") / "plots").exists() else []
    (site / "plots").mkdir(exist_ok=True)
    for p in plots:
        shutil.copy(p, site / "plots" / p.name)

    e = html.escape
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Time-aligned US-English TTS demo"
        "</title><style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:2em auto;"
        "padding:0 1em;color:#222}table{border-collapse:collapse;margin:1em 0}td,th{border:1px "
        "solid #ccc;padding:4px 8px;font-size:13px;vertical-align:top}.tag{display:inline-block;"
        "padding:1px 6px;border-radius:4px;background:#eee;font-size:12px}.stress{background:#fde2c8}"
        ".sample{border-top:1px solid #ddd;padding:1em 0}img{max-width:100%}audio{width:260px}"
        "</style></head><body>",
        "<h1>Time-aligned US-English TTS</h1><p>LJSpeech voice, US accent, on the word timing of "
        "a non-US reference recording. Ablation B used pace statistics from dev-accent speakers; "
        "all reported numbers are on held-out accents.</p>",
        *tables,
    ]
    if plots:
        parts.append("<h2>Plots</h2>" + "".join(f"<p><img src='plots/{p.name}'></p>" for p in plots))
    parts.append("<h2>Samples</h2>")
    for s in smp:
        cls = "tag stress" if s["label"] == "stress test" else "tag"
        ref = next((site / "samples" / s["id"]).glob("reference.*"))
        row = [f"<div class='sample'><h3>{e(s['id'])} <span class='{cls}'>{e(s['label'])}</span>"
               f"</h3><p>accent <b>{e(s['accent'])}</b>, speaker {e(s['speaker'])}</p>"
               f"<p><i>{e(s['text'][:600])}</i></p><table><tr><td>reference<br><audio controls "
               f"preload='none' src='samples/{e(s['id'])}/{ref.name}'></audio></td>"]
        for name, _ in syss:
            st = status.get((s["id"], name), "missing")
            if st == "ok":
                row.append(f"<td>{e(name)}<br><audio controls preload='none' "
                           f"src='samples/{e(s['id'])}/{e(name)}/out.wav'></audio></td>")
            else:
                row.append(f"<td>{e(name)}<br><small>{e(st)}</small></td>")
        row.append("</tr></table>")
        for name, _ in syss:
            png = site / "samples" / s["id"] / name / "timing.png"
            if png.exists():
                row.append(f"<details><summary>timing plot: {e(name)}</summary><img "
                           f"src='samples/{e(s['id'])}/{e(name)}/timing.png'></details>")
        row.append("</div>")
        parts.append("".join(row))
    parts.append("</body></html>")
    (site / "index.html").write_text("\n".join(parts))
    print(f"demo: {len(smp)} samples x {len(syss)} systems -> {site / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
