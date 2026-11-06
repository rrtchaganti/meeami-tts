"""T10: inference pipeline steps (Section 5) with a fake system; end-to-end exact length."""
import json

import numpy as np
import pytest
import soundfile as sf

from tts.config import REPO_ROOT, load_config, repo_path
from tts.infer.pipeline import (
    Pipeline, build_timeline, fit_words, load_reference, quantize, solve_word, target_frames,
    target_samples, word_times_from,
)

CFG = load_config(REPO_ROOT / "configs" / "inference.yaml")
FPS = 22050 / 256


def W(word, s_frames, e_frames):
    return {"word": word, "start": s_frames / FPS, "end": e_frames / FPS}


def test_targets():
    assert target_frames(44100, 44100, CFG) == round(22050 / 256)
    assert target_samples(48000 * 3, 48000, CFG) == 3 * 22050
    assert target_samples(8000, 8000, CFG) == 22050


def test_timeline_sil_sp_short_gap():
    # words at [10,20) [21,30) [40,50), total 60: gap 1 frame -> word 0; gap 10 -> sp
    wt = [W("a", 10, 20), W("b", 21, 30), W("c", 40, 50)]
    tl, spans = build_timeline(wt, [["AH0"], ["B", "IY1"], ["K"]], 60, CFG)
    assert tl.tokens == ["sil", "AH0", "B", "IY1", "sp", "K", "sil"]
    assert spans == [(10, 21), (21, 30), (40, 50)]
    assert tl.frames[0] == 10 and tl.frames[4] == 10 and tl.frames[-1] == 10
    assert tl.word_spans == [(1, 2), (2, 4), (5, 6)]


def test_solve_word_elastic():
    logd = np.log([5.0, 10.0])
    sig = np.array([0.2, 0.6])
    d = solve_word(logd, sig, 30.0)
    assert abs(d.sum() - 30.0) < 1e-6
    assert d[1] / 10 > d[0] / 5  # the more elastic phone stretches more


def test_fit_words_cap_inserts_sp():
    wt = [W("a", 0, 40)]
    tl, spans = build_timeline(wt, [["AH0", "B"]], 50, CFG)
    logd = np.log(np.array([1.0, 5.0, 5.0, 1.0]))
    sig = np.array([0.0, 0.5, 0.5, 0.0])
    f = fit_words(tl, spans, logd, sig, "elastic", cap=2.0)
    assert f.tokens == ["sil", "AH0", "B", "sp", "sil"] and f.capped == [0]
    assert abs(sum(f.frames[1:3]) - 20.0) < 1e-6 and abs(f.frames[3] - 20.0) < 1e-6
    p = fit_words(tl, spans, logd, sig, "proportional", cap=None)
    assert p.tokens == ["sil", "AH0", "B", "sil"] and abs(p.frames[1] - 20) < 1e-6


def test_quantize_exact_total_and_min_frame():
    d = quantize([0.0, 0.3, 10.2, 0.4, 5.1], ["sil", "AH0", "B", "IY1", "sil"], 16)
    assert sum(d) == 16 and min(d[1:4]) >= 1


def test_word_times_from():
    out = word_times_from(["sil", "AH0", "B", "sp", "K", "sil"], [-1, 0, 0, -1, 1, -1],
                          [2, 3, 4, 5, 6, 7], ["ab", "k"], FPS)
    assert out[0]["start"] == pytest.approx(2 / FPS) and out[0]["end"] == pytest.approx(9 / FPS)
    assert out[1]["start"] == pytest.approx(14 / FPS)


def test_load_reference_formats(tmp_path):
    y = (np.sin(np.linspace(0, 200, 48000)) * 0.3).astype(np.float32)
    st = np.stack([y, y * 0.5], 1)
    sf.write(tmp_path / "st.wav", st, 48000)
    m, sr = load_reference(tmp_path / "st.wav", CFG)
    assert sr == 48000 and m.ndim == 1 and len(m) == 48000
    import subprocess

    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(tmp_path / "st.wav"),
                    str(tmp_path / "st.mp3")], check=True)
    m3, sr3 = load_reference(tmp_path / "st.mp3", CFG)
    assert sr3 == 48000 and abs(len(m3) - 48000) < 3000


def test_test_speaker_reference_locked():
    from tts.guard import SplitLockedError, freeze_tag_exists

    if freeze_tag_exists(CFG.freeze_tag):
        pytest.skip("freeze tag exists")
    p = repo_path(CFG.paths.raw_l2) / "ABA" / "wav" / "arctic_a0001.wav"
    with pytest.raises(SplitLockedError):
        load_reference(p, CFG)


class FakeSystem:
    hop, sample_rate, device = 256, 22050, "cpu"

    def predict_log_durations(self, tokens):
        return np.full(len(tokens), np.log(6.0))

    def synthesize(self, tokens, durations, noise_scale=0.667):
        return (np.random.default_rng(0).standard_normal(sum(durations) * 256) * 0.01).astype(
            np.float32)


@pytest.mark.parametrize("sr", [8000, 44100])
def test_pipeline_exact_length_ctc_or_mfa(tmp_path, sr):
    p = repo_path("data/raw/l2arctic/ASI/wav/arctic_a0003.wav")
    if not p.exists() or not repo_path(CFG.paths.lexicon).exists():
        pytest.skip("data not prepared")
    y, sr0 = sf.read(p, dtype="float32")
    from tts.audio import resample

    ref = tmp_path / "ref.wav"
    sf.write(ref, resample(y, sr0, sr), sr)
    text = repo_path("data/raw/l2arctic/ASI/transcript/arctic_a0003.txt").read_text()
    info = Pipeline(CFG, FakeSystem()).run(text, ref, tmp_path / "out", aligner="mfa")
    n_ref = sf.info(ref).frames
    assert info["out_samples"] == round(n_ref * 22050 / sr) and info["length_error_samples"] == 0
    assert (tmp_path / "out" / "out.wav").exists() and (tmp_path / "out" / "timings.json").exists()
    d = json.loads((tmp_path / "out" / "timings.json").read_text())
    assert len(d["ref_words"]) == len(d["out_words"])


def test_mfa_failure_falls_back_to_ctc(tmp_path, monkeypatch):
    """Section 5 step 3: if align_one fails, MMS_FA word boundaries are used."""
    import subprocess

    import tts.infer.pipeline as P
    from tts.eval.assets import torch_home

    p = repo_path("data/raw/l2arctic/ASI/wav/arctic_a0003.wav")
    ckpt = torch_home(CFG) / "hub" / "checkpoints" / "model.pt"
    if not p.exists() or not ckpt.exists():
        pytest.skip("data or MMS_FA weights missing")
    monkeypatch.setattr(P, "align_one", lambda *a, **k: subprocess.CompletedProcess([], 1, "", "x"))
    text = repo_path("data/raw/l2arctic/ASI/transcript/arctic_a0003.txt").read_text()
    info = Pipeline(CFG, FakeSystem()).run(text, p, tmp_path / "o", plot=False)
    assert info["aligner"] == "ctc" and info["mfa_fallback_reason"] == "mfa_failed"
    assert info["length_error_samples"] == 0


def test_ctc_forced_align_viterbi():
    from tts.infer.ctc import forced_align

    # 6 frames, classes {0: blank, 1: a, 2: b}; best path: a a - b b -  (targets a b)
    lp = np.log(np.array([[.1, .8, .1], [.1, .8, .1], [.8, .1, .1],
                          [.1, .1, .8], [.1, .1, .8], [.8, .1, .1]]))
    assert forced_align(lp, [1, 2]).tolist() == [0, 0, -1, 1, 1, -1]
    # repeated token needs a blank between the two copies
    lp2 = np.log(np.full((3, 3), 1 / 3))
    p = forced_align(lp2, [1, 1])
    assert p.tolist() == [0, -1, 1]
    with pytest.raises(ValueError):
        forced_align(np.log(np.full((2, 3), 1 / 3)), [1, 1])
