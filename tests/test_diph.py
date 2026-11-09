"""Diphthong split (tts/text/diph.py, DECISIONS D-81)."""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from tts.config import REPO_ROOT, load_config, repo_path
from tts.text.diph import (
    N_VOCAB_SPLIT, SECOND_TO_ID, expand_embedding, merge_log_durations, split_tokens,
    to_ids_split,
)
from tts.text.symbols import N_VOCAB, SYMBOL_TO_ID

TOKS = ["sil", "K", "R", "AY1", "sp", "HH", "AW1", "sil"]
DURS = [3, 4, 5, 17, 2, 6, 1, 0]


def test_split_tokens_durations_and_mapping():
    t, d, src = split_tokens(TOKS, DURS, 0.5)
    assert t == ["sil", "K", "R", "AY1", "AY1+", "sp", "HH", "AW1", "AW1+", "sil"]
    assert d == [3, 4, 5, 8, 9, 2, 6, 1, 0, 0] and sum(d) == sum(DURS)  # round(8.5) = 8
    assert src == [0, 1, 2, 3, 3, 4, 5, 6, 6, 7]
    assert split_tokens(["EY2"], [2], 0.5)[1] == [1, 1]
    assert split_tokens(["OY0"], [10], 0.3)[1] == [3, 7]
    assert split_tokens(["OW1"], [10], 0.99)[1] == [9, 1]  # each part keeps >= 1 when d >= 2
    t2, d2, _ = split_tokens(TOKS)
    assert t2 == t and d2 is None
    assert split_tokens(["AA1", "IY0"], [4, 5])[0] == ["AA1", "IY0"]  # monophthongs untouched


def test_ids_and_vocab():
    ids = to_ids_split(["AY1", "AY1+", "K"])
    assert ids[0] == SYMBOL_TO_ID["AY1"] and ids[2] == SYMBOL_TO_ID["K"]
    assert N_VOCAB <= ids[1] < N_VOCAB_SPLIT and N_VOCAB_SPLIT == 87
    assert sorted(SECOND_TO_ID.values()) == list(range(N_VOCAB, N_VOCAB_SPLIT))
    with pytest.raises(KeyError):
        to_ids_split(["AA1+"])


def test_merge_log_durations():
    _, _, src = split_tokens(TOKS)
    logw = np.log(np.array([3, 4, 5, 8, 9, 2, 6, 1, 1, 1], float))
    m = merge_log_durations(logw, src, len(TOKS))
    assert np.allclose(np.exp(m), [3, 4, 5, 17, 2, 6, 2, 1])


def test_expand_embedding_copies_base_rows():
    old = torch.randn(N_VOCAB, 8)
    new = expand_embedding(old)
    assert new.shape == (N_VOCAB_SPLIT, 8) and torch.equal(new[:N_VOCAB], old)
    assert torch.equal(new[SECOND_TO_ID["OW2+"]], old[SYMBOL_TO_ID["OW2"]])
    with pytest.raises(ValueError):
        expand_embedding(new)


def test_m1_pos_diph_config_differs_only_in_split():
    a = OmegaConf.to_container(load_config(REPO_ROOT / "configs" / "m1_pos.yaml"))
    b = OmegaConf.to_container(load_config(REPO_ROOT / "configs" / "m1_pos_diph.yaml"))
    assert (a.pop("exp"), b.pop("exp")) == ("m1_pos", "m1_pos_diph")
    assert (a["model"].pop("diph_split"), b["model"].pop("diph_split")) == (False, True)
    assert a == b and b["model"]["use_pos"] is True


def _cfg(diph):
    return load_config(REPO_ROOT / "configs" / "base.yaml",
                       ["train.precision=fp32", "model.use_pos=true",
                        f"model.diph_split={str(diph).lower()}"])


def test_dataset_split_and_branch_from_m1(tmp_path):
    from tts.models.vits_mfa.data import ManifestDataset, collate, read_manifest
    from tts.models.vits_mfa.trainer import build_state, init_from_branch, state_dict, train_step

    p = repo_path("data/manifests/lj_val.jsonl")
    if not p.exists():
        pytest.skip("LJ manifests not built")
    rows = read_manifest(p)[:2]
    ds = ManifestDataset(rows, _cfg(True))
    it = ds[0]
    t, d, _ = split_tokens(rows[0]["phones"], rows[0]["durations"])
    assert it["ids"].tolist() == to_ids_split(t) and it["dur"].tolist() == d
    assert int(it["dur"].sum()) == rows[0]["n_frames"]
    assert all(it["dmask"][j] == 0 for j in range(len(d)) if d[j] == 0)

    dev = torch.device("cpu")
    m1 = build_state(load_config(REPO_ROOT / "configs" / "base.yaml"), dev)
    st = build_state(_cfg(True), dev)
    assert st.net_g.enc_p.emb.weight.shape[0] == N_VOCAB_SPLIT
    kept = init_from_branch(st, state_dict(m1, ""))
    assert kept == ["enc_p.pos_emb.weight"]
    e_new, e_old = st.net_g.enc_p.emb.weight, m1.net_g.enc_p.emb.weight
    assert torch.equal(e_new[SECOND_TO_ID["AY1+"]], e_old[SYMBOL_TO_ID["AY1"]])
    out = train_step(st, collate([ds[0], ds[1]]), _cfg(True), dev)
    assert all(np.isfinite(v) for v in out.values())

    # the inference wrapper: per-input-token durations, exact length
    from tts.infer.systems import VitsSystem
    ck = tmp_path / "ck.pt"
    torch.save(state_dict(st, OmegaConf.to_yaml(_cfg(True))), ck)
    sysm = VitsSystem(ck, "cpu")
    logd = sysm.predict_log_durations(TOKS)
    assert logd.shape == (len(TOKS),) and np.isfinite(logd).all()
    y = sysm.synthesize(TOKS, DURS, noise_scale=0.333)
    assert len(y) == sum(DURS) * 256
