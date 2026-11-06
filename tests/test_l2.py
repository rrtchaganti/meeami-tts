"""T5: manual-annotation parsing and MFA-vs-manual boundary comparison (Section 3)."""
from tts.align.l2 import canonical, compare, ref_n_frames, summarize
from tts.align.textgrid import Interval as I
from tts.config import REPO_ROOT, load_config

CFG = load_config(REPO_ROOT / "configs" / "base.yaml")


def test_canonical_labels():
    assert canonical("DH,D,s") == "DH" and canonical("AH0") == "AH"
    assert canonical("sil") == "" and canonical("") == ""
    assert canonical("sil,AH0,a") is None  # addition: not a canonical phone
    assert canonical("T,T*,s") == "T"


def test_compare_words_and_phones():
    mfa_w = [I(0, .1, ""), I(.1, .5, "the"), I(.5, .9, "cat")]
    mfa_p = [I(.1, .3, "DH"), I(.3, .5, "AH0"), I(.5, .6, "K"), I(.6, .8, "AE1"), I(.8, .9, "T")]
    man_w = [I(0, .12, ""), I(.12, .5, "the"), I(.5, .95, "cat")]
    man_p = [I(.12, .3, "DH,D,s"), I(.3, .5, "AH1"), I(.5, .6, "K"), I(.6, .62, "sil,AH0,a"),
             I(.62, .8, "AE1"), I(.8, .95, "T")]
    c = compare(mfa_w, mfa_p, man_w, man_p)
    assert c["n_words_matched"] == 2
    assert sorted(round(x, 3) for x in c["word"]) == [0.0, 0.0, 0.02, 0.05]
    assert len(c["phone"]) == 10  # 5 matched phones x 2 boundaries; the addition is excluded
    s = summarize(c["word"])
    assert s["n"] == 4 and abs(s["mae_ms"] - 17.5) < 1e-6 and s["within_20ms"] == 75.0


def test_ref_n_frames():
    assert ref_n_frames(44100, 44100, CFG) == round(22050 / 256)
