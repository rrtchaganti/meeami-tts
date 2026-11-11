"""Inference-input pre-pass (D-94): stress cases and the no-change guarantee on corpus text."""
from pathlib import Path

import pytest

from tts.config import REPO_ROOT, load_config
from tts.text.lexicon import cmu_dict
from tts.text.normalize import Normalizer, words
from tts.text.stress import CASES, load_cases, run_case, verdict

CFG = load_config(REPO_ROOT / "configs" / "inference.yaml")


@pytest.fixture(scope="module")
def norm():
    return Normalizer(CFG)


@pytest.mark.parametrize("cat,text,expected", load_cases(CASES),
                         ids=lambda v: v if isinstance(v, str) and len(v) < 30 else None)
def test_stress_case(norm, cat, text, expected):
    r = run_case(norm, text, {}, cmu_dict())
    assert verdict(r, expected) in ("PASS", "G2P"), (text, r)


def test_pre_pass_is_deterministic_and_lexicon_independent(norm):
    # the lexicon grows at inference; the rules must not depend on it (GPU spelled every time)
    assert words(norm("a GPU")) == words(norm("a GPU")) == ["a", "g", "p", "u"]


def test_bare_title_needs_a_capitalised_name(norm):
    assert words(norm("Elm Dr is closed")) == ["elm", "dr", "is", "closed"]
    assert words(norm("Dr Smith")) == ["doctor", "smith"]


def test_corpus_word_sequences_unchanged():
    """The pre-pass may change no training/eval transcript except the LJ bare-"Dr" fix."""
    cfg = load_config(REPO_ROOT / "configs" / "splits.yaml")
    if not (REPO_ROOT / cfg.paths.raw_lj).exists():
        pytest.skip("raw corpora not present")
    from tts.text.lexicon import corpus_texts

    new = Normalizer(cfg)
    old = Normalizer(cfg)
    old.pre = None
    for src, texts in corpus_texts(cfg).items():
        changed = [t for t in texts if words(old(t)) != words(new(t))]
        if src == "lj":
            assert len(changed) == 1 and "Dr Paul C. Peters" in changed[0], changed[:5]
        else:
            assert changed == [], (src, changed[:5])
