"""T4: audio frame count, TextGrid parsing rules, QC, MFA flag validation, phone stats."""
import math

import numpy as np
import pytest
import torch
from praatio import textgrid as ptg

from tts.align import mfa
from tts.align.textgrid import AlignmentError, Interval, parse, parse_file, qc_reason
from tts.audio import n_frames, spectrogram
from tts.config import REPO_ROOT, load_config
from tts.stats import phone_logdur

CFG = load_config(REPO_ROOT / "configs" / "base.yaml")
FPS = 22050 / 256  # frames per second


def t(frames: float) -> float:
    """Seconds for a frame boundary."""
    return frames / FPS


# --- audio ---------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [400, 1000, 22050, 212893])  # >= 385: reflect pad of 384
def test_n_frames_from_spectrogram(n):
    y = np.random.default_rng(0).standard_normal(n).astype("float32") * 0.1
    spec = spectrogram(torch.from_numpy(y), CFG.audio)
    assert spec.shape == (513, n // 256) and n_frames(y, CFG.audio) == n // 256


# --- parsing -------------------------------------------------------------------------------
def W(a, b, w):
    return Interval(t(a), t(b), w)


def test_rules_sil_sp_merge_wordids():
    # sil 0-10 | "hi" HH 10-15 AY1 15-20 | gap 20-22 (2 frames: merge) | "yo" Y 22-30 OW1 30-40
    # | gap 40-50 (10 frames: sp) | "ok" OW1 50-60 | trailing sil 60-70
    words = [W(0, 10, ""), W(10, 20, "hi"), W(20, 22, ""), W(22, 40, "yo"), W(40, 50, ""),
             W(50, 60, "ok"), W(60, 70, "")]
    phones = [W(0, 10, "sil"), W(10, 15, "HH"), W(15, 20, "AY1"), W(20, 22, ""), W(22, 30, "Y"),
              W(30, 40, "OW1"), W(40, 50, "sp"), W(50, 60, "OW1"), W(60, 70, "")]
    p = parse(words, phones, 70, CFG, expected_words=["hi", "yo", "ok"])
    assert p.phones == ["sil", "HH", "AY1", "Y", "OW1", "sp", "OW1", "sil"]
    assert p.durations == [10, 5, 7, 8, 10, 10, 10, 10]  # 2-frame gap merged into AY1
    assert p.word_ids == [-1, 0, 0, 1, 1, -1, 2, -1]
    assert p.n_merged_short_sil == 1 and p.raw_frame_sum == 70
    assert [w.word for w in p.words] == ["hi", "yo", "ok"]


def test_adjacent_silences_collapse_and_edges_sil():
    # speech touches t=0 and the end: zero-frame sil tokens are still present
    words = [W(0, 10, "a"), W(10, 13, ""), W(13, 20, "b")]
    phones = [W(0, 10, "AH0"), W(10, 11, "sil"), W(11, 13, "sp"), W(13, 20, "B")]
    p = parse(words, phones, 20, CFG)
    assert p.phones == ["sil", "AH0", "sp", "B", "sil"]  # 1+2 frames -> 3 frames -> sp
    assert p.durations == [0, 10, 3, 7, 0]


def test_last_token_adjusted_to_n_frames():
    words = [W(0, 5, ""), W(5, 20, "a"), W(20, 30, "")]
    phones = [W(0, 5, ""), W(5, 20, "AH0"), W(20, 30, "")]
    p = parse(words, phones, 31, CFG)
    assert p.raw_frame_sum == 30 and p.durations[-1] == 11 and sum(p.durations) == 31
    p = parse(words, phones, 28, CFG)
    assert p.durations[-1] == 8 and sum(p.durations) == 28


def test_deficit_taken_from_speech_when_no_trailing_silence():
    words = [W(0, 20, "a")]
    phones = [W(0, 10, "AH0"), W(10, 20, "B")]
    p = parse(words, phones, 19, CFG)
    assert p.phones == ["sil", "AH0", "B", "sil"] and p.durations == [0, 10, 9, 0]


def test_cumulative_rounding():
    # boundaries at 1.4, 2.8, 4.2 frames -> round -> 1, 3, 4 (differences, not rounded lengths)
    words = [W(0, 1.4, ""), W(1.4, 4.2, "a")]
    phones = [W(0, 1.4, ""), W(1.4, 2.8, "AH0"), W(2.8, 4.2, "B")]
    p = parse(words, phones, 4, CFG)
    assert p.durations == [1, 2, 1, 0]


@pytest.mark.parametrize("label", ["spn"])
def test_spn_rejected(label):
    words = [W(0, 10, "a"), W(10, 20, "<unk>")]
    phones = [W(0, 10, "AH0"), W(10, 20, label)]
    with pytest.raises(AlignmentError) as e:
        parse(words, phones, 20, CFG)
    assert e.value.reason == "spn"


def test_word_mismatch_rejected():
    words = [W(0, 10, "a")]
    phones = [W(0, 10, "AH0")]
    with pytest.raises(AlignmentError) as e:
        parse(words, phones, 10, CFG, expected_words=["b"])
    assert e.value.reason == "word_mismatch"


def test_qc():
    words = [W(0, 60, "a")]
    long = parse(words, [W(0, 60, "AH0")], 60, CFG)  # 60 frames = 0.70 s
    assert qc_reason(long, 60, CFG) == "long_phone"
    short = parse([W(0, 20, "a")], [W(0, 20, "AH0")], 20, CFG)
    assert qc_reason(short, 20, CFG) is None
    assert qc_reason(short, 23, CFG) == "frame_diff"  # raw 20 vs 23: off by 3 > 2
    assert qc_reason(short, 22, CFG) is None


def test_parse_file_roundtrip(tmp_path):
    tg = ptg.Textgrid()
    tg.addTier(ptg.IntervalTier("words", [(t(5), t(20), "hi")], 0, t(30)))
    tg.addTier(ptg.IntervalTier("phones", [(t(5), t(12), "HH"), (t(12), t(20), "AY1")], 0, t(30)))
    path = tmp_path / "x.TextGrid"
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)
    p = parse_file(path, 30, CFG, ["hi"])
    assert p.phones == ["sil", "HH", "AY1", "sil"] and p.durations == [5, 7, 8, 10]


# --- MFA flag validation (D-11 follow-up) --------------------------------------------------
def test_passthrough_flags_validated(monkeypatch):
    monkeypatch.setattr(mfa, "passthrough_names", lambda cfg, root=None: {"beam", "retry_beam"})
    assert mfa.passthrough_args(CFG) == ["--beam", "100", "--retry_beam", "400"]
    bad = load_config(REPO_ROOT / "configs" / "base.yaml")
    bad.mfa.passthrough["retry-beam"] = 7  # the typo MFA would drop without error
    with pytest.raises(mfa.MFAError, match="silently ignore"):
        mfa.passthrough_args(bad)


def test_passthrough_names_from_real_mfa():
    names = mfa.passthrough_names(CFG)
    assert {"beam", "retry_beam"} <= names


# --- stats ---------------------------------------------------------------------------------
def test_phone_logdur():
    rows = [{"phones": ["sil", "AH0", "B", "sp", "AH0", "sil"], "durations": [3, 4, 2, 5, 8, 0]},
            {"phones": ["AH0", "B"], "durations": [0, 2], "is_aug": False},
            {"phones": ["AH0"], "durations": [100], "is_aug": True}]  # augmented: ignored
    out = phone_logdur(rows, min_count=2)
    ah = out["phones"]["AH0"]
    assert ah["count"] == 2 and ah["zero_frames"] == 1
    assert ah["mean"] == pytest.approx((math.log(4) + math.log(8)) / 2)
    assert ah["std"] == pytest.approx(np.std([math.log(4), math.log(8)], ddof=1))
    assert not ah["sigma_from_median"] and ah["sigma"] == ah["std"]
    b = out["phones"]["B"]
    assert b["count"] == 2 and b["std"] == 0.0
    zh = out["phones"]["ZH"]  # unseen
    assert zh["count"] == 0 and zh["sigma_from_median"] and zh["mean_from_median"]
    assert zh["sigma"] == out["median_sigma"]


def test_zero_frame_phone_gets_one_frame():
    # B spans 10.0-10.4 frames -> rounds to 0 frames; neighbours tie at 10, the previous one gives
    words = [W(0, 20, "ab")]
    phones = [W(0, 10.0, "AH0"), W(10.0, 10.4, "B"), W(10.4, 20, "IY1")]
    p = parse(words, phones, 20, CFG)
    assert p.phones == ["sil", "AH0", "B", "IY1", "sil"]
    assert p.durations == [0, 9, 1, 10, 0] and p.extra["zero_frame_fixed"] == 1


def test_zero_frame_phone_searches_beyond_neighbours():
    from tts.align.textgrid import _fill_zero_phones

    ph = ["sil", "AH0", "B", "IY1", "K", "sil"]
    d = [0, 1, 0, 1, 6, 0]
    assert _fill_zero_phones(ph, d) == 1
    assert d == [0, 1, 1, 1, 5, 0] and sum(d) == 8


def test_align_one_tmp_follows_lexicon(tmp_path):
    """E-35: MFA caches the parsed dictionary, so align_one's temp dir is keyed by the lexicon hash."""
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    cfg.paths.lexicon = str(tmp_path / "lex.txt")
    cfg.paths.mfa_tmp = str(tmp_path / "tmp")
    (tmp_path / "lex.txt").write_text("a\tAH0\n")
    first = mfa.align_one_tmp(cfg)
    assert first == mfa.align_one_tmp(cfg) and first.parent == tmp_path / "tmp"
    first.mkdir(parents=True)
    (tmp_path / "lex.txt").write_text("a\tAH0\nbcd\tB EH1 D\n")
    second = mfa.align_one_tmp(cfg)
    assert second != first and not first.exists()   # the stale cache is removed
