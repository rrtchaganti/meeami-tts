"""T11-T14 helpers: tune choice, check-data on a tiny bundle, timing plot, plots on empty input."""
import json
import math

import numpy as np
import soundfile as sf

from tts.check_data import BUNDLE_LIST, check, write_manifest
from tts.eval.tune import choose


def test_tune_choose():
    g = [{"transfer": "elastic", "aligner": "mfa", "wb_mae_ms": 20, "utmos": 3.0, "wer": .1, "cap": 2.0},
         {"transfer": "elastic", "aligner": "mfa", "wb_mae_ms": 24, "utmos": 3.5, "wer": .2, "cap": 1.6},
         {"transfer": "elastic", "aligner": "mfa", "wb_mae_ms": 40, "utmos": 4.0, "wer": .0, "cap": 1.4},
         {"transfer": "proportional", "aligner": "mfa", "wb_mae_ms": 1, "utmos": 5.0, "wer": 0, "cap": 2}]
    pick, crit, best = choose(g, "mfa", 5)
    assert pick["cap"] == 1.6 and crit == "UTMOS" and best == 20
    for r in g:
        r["utmos"] = math.nan
    pick, crit, _ = choose(g, "mfa", 5)
    assert pick["cap"] == 2.0 and crit.startswith("WER")


def test_check_data_tiny_bundle(tmp_path):
    (tmp_path / "data/manifests").mkdir(parents=True)
    (tmp_path / "data/aug").mkdir(parents=True)
    wav = tmp_path / "data/aug/x.flac"
    sf.write(wav, np.zeros(10 * 256, dtype="float32"), 22050, subtype="PCM_16")
    row = {"id": "x", "wav": "data/aug/x.flac", "durations": [4, 6], "n_frames": 10}
    (tmp_path / "data/manifests/aug_a_all.jsonl").write_text(json.dumps(row) + "\n")
    files = ["data/aug/x.flac", "data/manifests/aug_a_all.jsonl"]
    (tmp_path / BUNDLE_LIST).write_text("\n".join(files) + "\n")
    write_manifest(tmp_path, files)
    assert check(tmp_path, skip_hash=False) == []
    row["n_frames"] = 11
    (tmp_path / "data/manifests/aug_a_all.jsonl").write_text(json.dumps(row) + "\n")
    errs = check(tmp_path, skip_hash=False)
    assert any("checksum" in e for e in errs) and any("sum(durations)" in e for e in errs)


def test_timing_plot(tmp_path):
    from tts.infer.plot import timing_plot

    y = (np.random.default_rng(0).standard_normal(22050) * 0.05).astype("float32")
    w = [{"word": "a", "start": 0.1, "end": 0.4}, {"word": "b", "start": 0.5, "end": 0.9}]
    timing_plot(y, 22050, y, 22050, w, w, tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 1000


def test_make_plots_without_results(tmp_path, monkeypatch):
    import tts.eval.plots as P

    monkeypatch.setattr(P, "repo_path", lambda p: tmp_path / p)
    P.make_plots(None, "dev")  # nothing to plot: must not raise
    assert (tmp_path / "results" / "plots").exists()
