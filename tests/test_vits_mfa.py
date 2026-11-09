"""T7: VITS-MFA model (Section 4 changes 1-6): forward/backward, checkpoint loading, durations,
position bins, data loader, sampler. The one-utterance overfit is `make overfit` (slow)."""
import pytest
import torch

from tts.config import REPO_ROOT, load_config, repo_path
from tts.models.vits_mfa.data import ManifestDataset, MixedBucketSampler, collate, read_manifest
from tts.models.vits_mfa.models import (
    build_generator, duration_loss, duration_path, load_pretrained, position_bins,
)
from tts.models.vits_mfa.trainer import (
    build_state, init_from_branch, init_from_pretrained_vits, state_dict, train_step,
)
from tts.text.symbols import N_VOCAB, SYMBOL_TO_ID

CFG = load_config(REPO_ROOT / "configs" / "base.yaml",
                  ["train.batch_size=4", "train.precision=fp32"])
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_duration_path_is_one_hot_expansion():
    dur = torch.tensor([[2, 0, 3, 1]])
    x_mask = torch.ones(1, 1, 4)
    y_mask = torch.ones(1, 1, 6)
    attn = duration_path(dur, x_mask, y_mask)[0, 0]  # [t_y, t_x]
    assert attn.sum(0).tolist() == [2, 0, 3, 1] and attn.sum(1).tolist() == [1] * 6
    assert attn.argmax(1).tolist() == [0, 0, 2, 2, 2, 3]


def test_position_bins():
    bins = position_bins(torch.tensor([[2, 4]]), 6, 16)[0].tolist()
    # (k+0.5)/d: d=2 -> .25,.75 -> bins 4,12; d=4 -> .125,.375,.625,.875 -> 2,6,10,14
    assert bins == [4, 12, 2, 6, 10, 14]


def test_duration_loss_masks():
    logw = torch.zeros(1, 1, 3)
    dur = torch.tensor([[1, 100, 1]])
    assert duration_loss(logw, dur, torch.tensor([[1.0, 0.0, 1.0]])).item() == 0.0
    assert duration_loss(logw, dur, torch.ones(1, 3)).item() > 0


def test_n_vocab_and_no_mas():
    net = build_generator(CFG)
    assert net.enc_p.emb.num_embeddings == N_VOCAB == 72
    import tts.models.vits_mfa.models as m
    assert "monotonic_align" not in open(m.__file__).read().split('"""', 2)[2]


def test_pretrained_loads_only_expected_skips():
    path = repo_path(CFG.pretrained.vits_lj.path)
    if not path.exists():
        pytest.skip("pretrained VITS not downloaded")
    net = build_generator(CFG)
    sd = torch.load(path, map_location="cpu", weights_only=True)["model"]
    loaded, skipped, missing = load_pretrained(net, sd)
    skipped_keys = {s.split(":")[0] for s in skipped}
    # expected: the text embedding (new symbol set) and the stochastic duration predictor
    assert all(k == "enc_p.emb.weight" or k.startswith("dp.") for k in skipped_keys), skipped
    assert "enc_p.emb.weight" in skipped_keys
    # expected missing: the new embedding and the deterministic duration predictor
    assert all(k == "enc_p.emb.weight" or k.startswith("dp.") for k in missing), missing
    assert len(loaded) > 400


def _rows(n=4):
    p = repo_path("data/manifests/lj_val.jsonl")
    if not p.exists():
        pytest.skip("LJ manifests not built")
    rows = read_manifest(p)
    if rows[0].get("durations") is None:
        pytest.skip("LJ manifests not aligned")
    return sorted(rows, key=lambda r: r["n_frames"])[:n]


def test_dataset_item_consistency():
    ds = ManifestDataset(_rows(), CFG)
    it = ds[0]
    assert it["spec"].shape[1] == it["dur"].sum().item() == ds.rows[0]["n_frames"]
    assert it["wav"].numel() == it["spec"].shape[1] * 256
    assert it["dmask"][0] == 0 and it["ids"][0] == SYMBOL_TO_ID["sil"]


@pytest.mark.parametrize("use_pos", [False, True])
def test_forward_backward(use_pos):
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml",
                      ["train.precision=bf16", f"model.use_pos={str(use_pos).lower()}"])
    ds = ManifestDataset(_rows(2), cfg)
    batch = collate([ds[0], ds[1]])
    state = build_state(cfg, DEV)
    out = train_step(state, batch, cfg, DEV)
    assert all(torch.isfinite(torch.tensor(v)) for v in out.values()), out
    assert state.step == 1
    if use_pos:
        assert state.net_g.enc_p.pos_emb.weight.grad is not None


def test_infer_exact_length():
    net = build_generator(CFG).to(DEV).eval()
    x = torch.tensor([[1, 10, 20, 30, 1]], device=DEV)
    dur = torch.tensor([[3, 5, 0, 7, 2]], device=DEV)
    audio, y_mask = net.infer(x, torch.tensor([5], device=DEV), dur)
    assert audio.shape[-1] == 17 * 256 and int(y_mask.sum()) == 17


def test_init_from_pretrained_in_state():
    path = repo_path(CFG.pretrained.vits_lj.path)
    if not path.exists():
        pytest.skip("pretrained VITS not downloaded")
    state = build_state(CFG, torch.device("cpu"))
    info = init_from_pretrained_vits(state, str(path))
    assert info["loaded"] > 400


def test_mixed_sampler_ratio_and_coverage():
    base = list(range(100, 200))   # 100 base rows
    aug = list(range(50, 250))     # 200 aug rows
    s = MixedBucketSampler(base, aug, batch_size=10, aug_fraction=0.4, seed=1)
    batches = list(iter(s))
    assert len(batches) == len(s) == 100 // 6
    for b in batches:
        assert sum(i < 100 for i in b) == 6 and sum(i >= 100 for i in b) == 4
    plain = list(iter(MixedBucketSampler(base, [], batch_size=10)))
    assert sorted(i for b in plain for i in b) == list(range(100))


def test_init_from_branch_m1_to_pos():
    """D-76: an M1 checkpoint branches into M1+pos (pos_emb stays zero) and into M1 itself."""
    m1 = build_state(CFG, torch.device("cpu"))
    ck = state_dict(m1, "")
    pos_cfg = load_config(REPO_ROOT / "configs" / "m1_pos.yaml")
    pos = build_state(pos_cfg, torch.device("cpu"))
    assert init_from_branch(pos, ck) == ["enc_p.pos_emb.weight"]
    assert torch.count_nonzero(pos.net_g.enc_p.pos_emb.weight) == 0
    w = "enc_p.proj.weight"
    assert torch.equal(pos.net_g.state_dict()[w], m1.net_g.state_dict()[w])
    assert init_from_branch(build_state(CFG, torch.device("cpu")), ck) == []
    with pytest.raises(RuntimeError):  # an M1+pos checkpoint cannot go into M1
        init_from_branch(build_state(CFG, torch.device("cpu")), state_dict(pos, ""))


@pytest.mark.parametrize("x", ["a", "b"])
def test_abl_pos_configs_differ_only_in_use_pos(x):
    """D-78: abl_{a,b}_pos = abl_{a,b} + use_pos (and their own exp name)."""
    from omegaconf import OmegaConf
    base = OmegaConf.to_container(load_config(REPO_ROOT / "configs" / f"abl_{x}.yaml"))
    pos = OmegaConf.to_container(load_config(REPO_ROOT / "configs" / f"abl_{x}_pos.yaml"))
    assert pos.pop("exp") == f"abl_{x}_pos" and base.pop("exp") == f"abl_{x}"
    assert pos["model"].pop("use_pos") is True and base["model"].pop("use_pos") is False
    assert pos == base
