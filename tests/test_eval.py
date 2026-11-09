"""T11: metric functions (boundary errors, pause F1, bootstrap, Wilcoxon)."""
import numpy as np
import pytest

from tts.eval.metrics import (
    boundary_errors, boundary_summary, bootstrap_ci, match_words, pause_f1, wilcoxon,
)


def W(w, s, e):
    return {"word": w, "start": s, "end": e}


def test_boundaries_and_matching():
    ref = [W("a", 0.0, 0.5), W("b", 0.7, 1.0), W("c", 1.0, 1.5)]
    hyp = [W("a", 0.01, 0.5), W("x", 0.5, 0.6), W("b", 0.72, 1.0), W("c", 1.0, 1.6)]
    assert match_words(ref, hyp) == [(0, 0), (1, 2), (2, 3)]
    e = boundary_errors(ref, hyp)
    assert sorted(np.round(e, 3)) == [0.0, 0.0, 0.0, 0.01, 0.02, 0.1]
    s = boundary_summary(e)
    assert s["mae_ms"] == pytest.approx(130 / 6) and s["within_20ms"] == pytest.approx(5 / 6 * 100)


def test_pause_f1():
    ref = [W("a", 0, 0.5), W("b", 0.7, 1.0), W("c", 1.0, 1.5)]   # pause after a only
    hyp = [W("a", 0, 0.5), W("b", 0.65, 1.0), W("c", 1.2, 1.5)]  # pauses after a and b
    r = pause_f1(ref, hyp, 0.1)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0) and r["f1"] == pytest.approx(2 / 3)
    assert pause_f1(ref[1:], ref[1:])["f1"] == 1.0  # no pauses on either side


def test_bootstrap_and_wilcoxon():
    est, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, float("nan")], n=200)
    assert est == 2.5 and lo <= est <= hi
    a = np.arange(20, dtype=float)
    r = wilcoxon(a + 1.0, a)
    assert r["n"] == 20 and r["p"] < 0.001 and r["median_diff"] == 1.0


def test_utmos_weights_pinned(tmp_path):
    """UTMOS weights must match the pinned sha256; no download when download=False."""
    import hashlib

    from omegaconf import OmegaConf

    from tts.eval.assets import UtmosWeightsError, utmos_weights

    url = "https://example.invalid/rel/utmos.pt"
    f = tmp_path / "eval" / "utmos" / "utmos.pt"
    cfg = OmegaConf.create({"eval_models": {"root": "eval", "utmos": {
        "weights_url": url, "weights_sha256": "0" * 64}}})
    with pytest.raises(UtmosWeightsError, match="missing"):
        utmos_weights(cfg, tmp_path, download=False)
    f.parent.mkdir(parents=True)
    f.write_bytes(b"weights")
    with pytest.raises(UtmosWeightsError, match="sha256"):
        utmos_weights(cfg, tmp_path, download=False)
    cfg.eval_models.utmos.weights_sha256 = hashlib.sha256(b"weights").hexdigest()
    assert utmos_weights(cfg, tmp_path, download=False) == f


def test_phone_dynamics_stats():
    """D-76: inside/onset/thirds of within-phone spectral change; sil/sp and short phones skipped."""
    import numpy as np

    from tts.eval.phone_dynamics import frame_change, phone_stats, summarize
    m = np.zeros((2, 20))
    m[0, 10:] = 3.0                      # one step of size 3 at frame 10 (onset of phone 2)
    m[1, 13] = 4.0                       # a bump inside phone 2: change 4 at 13 and at 14
    dm = frame_change(m)
    assert dm[0] == 0 and dm[10] == 3 and dm[13] == 4 and dm[14] == 4
    s = phone_stats(dm, ["sil", "AA1", "B", "IY1"], [2, 8, 9, 1], min_frames=8)
    assert len(s["inside"]) == 2 and s["onset"] == [0.0, 3.0]
    assert s["inside"][1] == 1.0         # frames 11..18 of phone B: 8 / 8
    assert [t[1] for t in s["thirds"]] == [4 / 3, 4 / 3, 0.0]  # thirds 11-13, 14-16, 17-18
    r = summarize(s)
    assert r["n_phones"] == 2 and r["onset_over_inside"] == 3.0
