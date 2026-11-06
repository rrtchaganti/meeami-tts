"""T9: augmentation core (elastic factors, z solve, room tone, render) and filter selection."""
import numpy as np
import pytest

from tts.augment.core import Plan, elastic_factors, lengthen, loop_tone, render, solve_z
from tts.augment.filter import select
from tts.augment.pace_b import build, word_spans
from tts.config import REPO_ROOT, load_config

CFG = load_config(REPO_ROOT / "configs" / "base.yaml")


def test_solve_z_hits_target_and_clips():
    d = np.array([5.0, 10.0, 3.0])
    s = np.array([0.4, 0.6, 0.3])
    z = solve_z(d, s, 25.0, 0.6, 2.5)
    assert abs(np.sum(d * elastic_factors(z, s, 0.6, 2.5)) - 25.0) < 0.05
    assert solve_z(d, s, 1000.0, 0.6, 2.5) == 5.0  # unreachable: saturates at the clip
    assert elastic_factors(10.0, s, 0.6, 2.5).max() == 2.5


def test_loop_and_lengthen_never_zero():
    rng = np.random.default_rng(0)
    tone = (rng.standard_normal(2000) * 0.01).astype(np.float32)
    out = loop_tone(tone, 9000, 220)
    assert len(out) == 9000 and np.abs(out[-500:]).mean() > 0.002
    seg = (rng.standard_normal(1500) * 0.01).astype(np.float32)
    long = lengthen(seg, 6000, tone, 220)
    assert len(long) == 6000
    assert np.allclose(long[:100], seg[:100]) and np.allclose(long[-100:], seg[-100:])


def _row():
    # sil 10 | AH0 20 B 15 (word 0) | sp 10 | K 20 (word 1) | sil 5  -> 80 frames
    return {"id": "X", "phones": ["sil", "AH0", "B", "sp", "K", "sil"],
            "durations": [10, 20, 15, 10, 20, 5], "word_ids": [-1, 0, 0, -1, 1, -1],
            "words": ["ab", "k"], "n_frames": 80, "text": "ab, k"}


def test_render_lengths_and_durations():
    pytest.importorskip("pyrubberband")
    rng = np.random.default_rng(1)
    hop = 256
    y = (rng.standard_normal(80 * hop) * 0.05).astype(np.float32)
    plan = Plan([1.0, 2.0, 2.0, 1.0, 0.6, 1.0], sp_target={3: 25 * hop}, inserts=[(4, 8 * hop)])
    rb = {"--fine": ""}
    audio, ph, d, w = render(y, _row(), plan, CFG, rb)
    assert ph == ["sil", "AH0", "B", "sp", "K", "sp", "sil"] and w == [-1, 0, 0, -1, 1, -1, -1]
    assert d == [10, 40, 30, 25, 12, 8, 5] and len(audio) == sum(d) * hop
    assert np.abs(audio).max() > 0


def test_pace_stats_build():
    mu = {"AH0": np.log(10), "B": np.log(10), "K": np.log(10)}
    r = dict(_row(), align_status="ok")
    out = build([r], mu)
    # word 0: 35 frames / 20 expected; word 1: 20 / 10
    lr = [np.log(35 / 20), np.log(20 / 10)]
    assert out["utt_mean_log_rho"] == [pytest.approx(np.mean(lr))]
    assert out["p_sp_after_word"] == 1.0 and out["sp_len_hist"] == {"10": 1}
    assert word_spans(r) == [(1, 2), (4, 4)]


def test_filter_select_drops_largest():
    scores = [{"id": i, "utmos_drop": float(i)} for i in range(100)]
    kept, thr = select(scores, 0.05)
    assert len(kept) == 95 and max(s["id"] for s in kept) == 94 and thr == 94.0
