"""V0 baseline: original pretrained VITS driven by reference word timing (DECISIONS D-90)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from tts.config import REPO_ROOT, load_config, repo_path
from tts.models.vits_mas.v0 import (VITS_SYMBOLS, arpa_to_ipa, intersperse, ipa_line,
                                    split_frames, to_chars)
from tts.text.symbols import PHONES

CFG = load_config(REPO_ROOT / "configs" / "v0.yaml")


def test_symbol_table_is_the_original():
    assert len(VITS_SYMBOLS) == 178 and VITS_SYMBOLS[0] == "_" and VITS_SYMBOLS[16] == " "


def test_every_phone_maps_into_the_vits_symbols():
    for p in PHONES:
        assert all(c in VITS_SYMBOLS for c in arpa_to_ipa(p)), p


def test_espeak_conventions():
    assert arpa_to_ipa("AH0") == "ə" and arpa_to_ipa("AH1") == "ˈʌ"
    assert arpa_to_ipa("ER0") == "ɚ" and arpa_to_ipa("ER1") == "ˈɜː"
    assert arpa_to_ipa("IY0") == "i" and arpa_to_ipa("IY2") == "ˌiː"
    assert arpa_to_ipa("R") == "ɹ" and arpa_to_ipa("G") == "ɡ"


def _hello_world():
    toks = ["sil", "HH", "AH0", "L", "OW1", "sp", "W", "ER1", "L", "D", "B", "AY1", "sil"]
    wids = [-1, 0, 0, 0, 0, -1, 1, 1, 1, 1, 2, 2, -1]
    return toks, wids


def test_text_and_ownership():
    toks, wids = _hello_world()
    assert ipa_line(toks, wids) == "həlˈoʊ, wˈɜːld bˈaɪ."
    chars, owner = to_chars(toks, wids)
    ids, own = intersperse(chars, owner)
    assert len(ids) == 2 * len(chars) + 1 and ids[0] == 0 and ids[-1] == 0
    assert own[0] == 0 and set(own) == set(range(len(toks)))  # every token owns something
    last_space = len(chars) - 1 - chars[::-1].index(" ")
    assert owner[last_space] == toks.index("D")  # a word's space belongs to its last phone


def test_split_frames_keeps_token_totals():
    toks, wids = _hello_world()
    _, own = intersperse(*to_chars(toks, wids))
    rng = np.random.default_rng(0)
    d = rng.integers(0, 30, len(toks)).tolist()
    sub = split_frames(d, own, rng.uniform(0.1, 5, len(own)))
    assert min(sub) >= 0
    for i, di in enumerate(d):
        assert sum(s for s, o in zip(sub, own) if o == i) == di


@pytest.mark.skipif(not repo_path(CFG.pretrained.vits_lj.path).exists(),
                    reason="pretrained VITS not downloaded")
def test_v0_loads_and_keeps_exact_length():
    from tts.infer.systems import load_system

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = load_system("v0", CFG, dev)
    toks, wids = _hello_world()
    logd = s.predict_log_durations(toks, wids)
    assert logd.shape == (len(toks),) and np.isfinite(logd).all()
    assert np.allclose(logd, s.predict_log_durations(toks, wids))  # seeded SDP
    d = [3, 5, 4, 6, 12, 10, 4, 9, 5, 4, 5, 15, 8]
    y = s.synthesize(toks, d, 0.333, wids)
    assert len(y) == sum(d) * s.hop and np.abs(y).max() > 0.01


def test_blank_owner_before():
    toks, wids = _hello_world()
    chars, owner = to_chars(toks, wids)
    ids_a, own_a = intersperse(chars, owner, "after")
    ids_b, own_b = intersperse(chars, owner, "before")
    assert ids_a == ids_b
    # "before": the blank preceding each character belongs to that character's token
    for k, o in enumerate(owner):
        if k > 0:
            assert own_b[2 * k] == o
        assert own_b[2 * k + 1] == o
        assert own_a[2 * k + 1] == o and own_a[2 * k + 2] == o
    assert own_b[-1] == owner[-1] and own_a[0] == 0 and own_b[0] == 0
    for own in (own_a, own_b):  # every token, incl. the opening sil, owns a sub-token
        assert set(own) == set(range(len(toks)))
    with pytest.raises(ValueError):
        intersperse(chars, owner, "middle")
