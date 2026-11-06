"""T3: lexicon format, sources, OOV G2P, inference append; MFA phone-set agreement; assets."""
import pytest

from tts.config import REPO_ROOT, load_config, repo_path
from tts.text import lexicon as L
from tts.text.symbols import PHONE_SET

CFG = load_config(REPO_ROOT / "configs" / "splits.yaml")


def fake_g2p(word):
    return ["F", "EY1", "K"]


# --- format --------------------------------------------------------------------------------
def test_roundtrip(tmp_path):
    lex = {"zeta": ["Z", "EY1", "T", "AH0"], "alpha": ["AE1", "L", "F", "AH0"]}
    p = tmp_path / "lexicon.txt"
    p.write_text(L.format_lexicon(lex))
    assert p.read_text().splitlines()[0] == "alpha\tAE1 L F AH0"  # sorted, tab-separated
    assert L.load_lexicon(p) == lex


@pytest.mark.parametrize("line", [
    "word P1 P2",            # no tab
    "Word\tW ER1 D",         # not lowercase
    "word\tW ER D",          # vowel without stress digit
    "word\t",                # empty pronunciation
])
def test_load_rejects_bad_lines(tmp_path, line):
    p = tmp_path / "lexicon.txt"
    p.write_text(line + "\n")
    with pytest.raises(L.LexiconError):
        L.load_lexicon(p)


def test_load_rejects_duplicates(tmp_path):
    p = tmp_path / "lexicon.txt"
    p.write_text("read\tR IY1 D\nread\tR EH1 D\n")
    with pytest.raises(L.LexiconError, match="duplicate"):
        L.load_lexicon(p)


# --- sources -------------------------------------------------------------------------------
def test_cmudict_first_pronunciation():
    pron, src = L.pronounce("printing", fake_g2p)
    assert src == "cmudict" and pron == ["P", "R", "IH1", "N", "T", "IH0", "NG"]
    assert L.cmu_dict()["printing"][1] != pron  # a second variant exists and is not used


def test_oov_goes_to_g2p():
    pron, src = L.pronounce("meeamixq", fake_g2p)
    assert src == "g2p" and pron == ["F", "EY1", "K"]


@pytest.mark.parametrize("word,suffix", [
    ("companion's", ["Z"]), ("chemist's", ["S"]), ("cyrus's", ["IH0", "Z"]),
])
def test_possessive_rule(word, suffix):
    if word in L.cmu_dict():
        pytest.skip(f"{word} is in cmudict")
    pron, src = L.pronounce(word, fake_g2p)
    assert src == "cmudict+s"
    assert pron == list(L.cmu_dict()[word[:-2]][0]) + suffix
    assert L.pronounce(word, fake_g2p, possessive_rule=False)[1] == "g2p"


def test_build_reports_derived():
    lex, derived = L.build(["printing", "meeamixq", "printing"], fake_g2p)
    assert set(lex) == {"printing", "meeamixq"} and derived == {"meeamixq": "g2p"}


def test_real_g2p_en_offline():
    g = L.G2P(CFG)  # uses project NLTK data; any download attempt raises
    pron = g("meeami")
    assert pron and set(pron) <= PHONE_SET
    assert g("o'malleyx") == g("omalleyx")  # apostrophes dropped before the neural model


def test_ensure_words_appends_only_missing(tmp_path):
    p = tmp_path / "lexicon.txt"
    p.write_text("printing\tP R IH1 N T IH0 NG\n")
    assert L.ensure_words(["printing", "meeamixq", "meeamixq"], p, fake_g2p) == ["meeamixq"]
    assert L.ensure_words(["meeamixq"], p, fake_g2p) == []
    assert L.load_lexicon(p)["meeamixq"] == ["F", "EY1", "K"]


# --- MFA agreement and the built lexicon (need `make lexicon` assets) ----------------------
def test_mfa_phone_set_equals_symbols():
    from tts.align.mfa import acoustic_model_phones, model_path

    if not model_path(CFG, "acoustic").exists():
        pytest.skip("MFA acoustic model not downloaded (make assets)")
    assert acoustic_model_phones(CFG) == set(PHONE_SET)


def test_built_lexicon():
    p = repo_path(CFG.paths.lexicon)
    if not p.exists():
        pytest.skip("lexicon not built (make lexicon)")
    lex = L.load_lexicon(p)  # validates format, one pronunciation, phone set
    assert len(lex) > 10000
    assert {p for pr in lex.values() for p in pr} <= PHONE_SET


# --- assets --------------------------------------------------------------------------------
def test_use_local_nltk_forbids_downloads():
    import nltk

    from tts.assets import AssetError, use_local_nltk

    orig = nltk.download
    try:
        use_local_nltk(CFG)
        assert str(repo_path(CFG.paths.nltk_data)) == nltk.data.path[0]
        with pytest.raises(AssetError):
            nltk.download("punkt")
    finally:
        nltk.download = orig


def test_nltk_sha_mismatch(tmp_path, monkeypatch):
    import nltk

    from tts.assets import AssetError, ensure_nltk_data

    cfg = load_config(REPO_ROOT / "configs" / "base.yaml", [f"paths.nltk_data={tmp_path}"])
    cfg.nltk.resources[0].sha256 = "0" * 64
    (tmp_path / "taggers").mkdir()
    (tmp_path / "taggers" / "averaged_perceptron_tagger.zip").write_bytes(b"x")
    (tmp_path / "corpora").mkdir()
    monkeypatch.setattr(nltk, "download", lambda *a, **k: True)  # "downloads" nothing new
    with pytest.raises(AssetError, match="sha256"):
        ensure_nltk_data(cfg)
