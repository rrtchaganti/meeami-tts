"""Training step and state for VITS-MFA (docs/PLAN.md Section 4 "Losses", "Hyperparameters").

L_G = c_mel * L_mel + c_kl * L_KL + c_dur * L_dur + L_adv + c_fm * L_fm  (45, 1, 1, 1, 1; VITS's
feature_loss already includes the factor 2). Least-squares GAN for the discriminators.
bf16 autocast for the networks; STFT, mel and every loss are computed in fp32.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from omegaconf import DictConfig

from tts.audio import mel_spectrogram, spec_to_mel
from tts.models.vits_mfa import commons
from tts.models.vits_mfa.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from tts.models.vits_mfa.models import (
    MultiPeriodDiscriminator, build_generator, duration_loss, load_pretrained,
)


@dataclass
class TrainState:
    net_g: torch.nn.Module
    net_d: torch.nn.Module
    optim_g: torch.optim.Optimizer
    optim_d: torch.optim.Optimizer
    sched_g: torch.optim.lr_scheduler.LRScheduler
    sched_d: torch.optim.lr_scheduler.LRScheduler
    step: int = 0
    epoch: int = 0


def build_state(cfg: DictConfig, device: torch.device) -> TrainState:
    net_g = build_generator(cfg).to(device)
    net_d = MultiPeriodDiscriminator(cfg.model.use_spectral_norm).to(device)
    t = cfg.train
    kw = dict(lr=t.learning_rate, betas=tuple(t.betas), eps=t.eps)
    optim_g = torch.optim.AdamW(net_g.parameters(), **kw)
    optim_d = torch.optim.AdamW(net_d.parameters(), **kw)
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=t.lr_decay)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=t.lr_decay)
    return TrainState(net_g, net_d, optim_g, optim_d, sched_g, sched_d)


def init_from_pretrained_vits(state: TrainState, path: str) -> dict:
    """Section 4 change 3: load matching generator weights; discriminators start from scratch."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt)
    loaded, skipped, missing = load_pretrained(state.net_g, sd)
    return {"loaded": len(loaded), "skipped": skipped, "missing": missing}


BRANCH_NEW_KEYS = {"enc_p.pos_emb.weight"}  # zero-initialized: M1 -> M1+pos starts as M1


def init_from_branch(state: TrainState, ckpt: dict) -> list[str]:
    """Branching design: generator + discriminator weights from one of our checkpoints; fresh
    optimizers and step 0. Only BRANCH_NEW_KEYS may be absent (kept at their init); any other
    missing or unexpected key raises. A 72-row text embedding is expanded for diph_split models
    (each X+ row copies X, D-81). Returns the keys left at init (D-76)."""
    sd = dict(ckpt["net_g"])
    k = "enc_p.emb.weight"
    own = state.net_g.state_dict()[k]
    if sd[k].shape != own.shape:  # M1 (72 symbols) -> diph_split model (87), D-81
        from tts.text.diph import expand_embedding
        sd[k] = expand_embedding(sd[k])
        if sd[k].shape != own.shape:
            raise RuntimeError(f"init_from: embedding {tuple(sd[k].shape)} != {tuple(own.shape)}")
    res = state.net_g.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if k not in BRANCH_NEW_KEYS] + list(res.unexpected_keys)
    if bad:
        raise RuntimeError(f"init_from: incompatible generator keys {bad}")
    state.net_d.load_state_dict(ckpt["net_d"])
    return list(res.missing_keys)


def state_dict(state: TrainState, cfg_yaml: str) -> dict:
    return {"net_g": state.net_g.state_dict(), "net_d": state.net_d.state_dict(),
            "optim_g": state.optim_g.state_dict(), "optim_d": state.optim_d.state_dict(),
            "sched_g": state.sched_g.state_dict(), "sched_d": state.sched_d.state_dict(),
            "step": state.step, "epoch": state.epoch, "config": cfg_yaml}


def load_state(state: TrainState, ckpt: dict, generator_only: bool = False) -> None:
    state.net_g.load_state_dict(ckpt["net_g"])
    if generator_only:
        return
    state.net_d.load_state_dict(ckpt["net_d"])
    state.optim_g.load_state_dict(ckpt["optim_g"])
    state.optim_d.load_state_dict(ckpt["optim_d"])
    state.sched_g.load_state_dict(ckpt["sched_g"])
    state.sched_d.load_state_dict(ckpt["sched_d"])
    state.step, state.epoch = ckpt["step"], ckpt["epoch"]


def _autocast(cfg: DictConfig, device: torch.device):
    if cfg.train.precision == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def set_decoder_frozen(state: TrainState, frozen: bool) -> None:
    """freeze_decoder_steps warm-up: only text encoder + duration predictor learn."""
    for name in ("dec", "flow", "enc_q"):
        for p in getattr(state.net_g, name).parameters():
            p.requires_grad_(not frozen)


def train_step(state: TrainState, batch: dict, cfg: DictConfig, device: torch.device) -> dict:
    t = cfg.train
    hop = int(cfg.audio.hop_length)
    seg = int(t.segment_size)
    x = batch["ids"].to(device, non_blocking=True)
    x_len = batch["x_lengths"].to(device)
    spec = batch["spec"].to(device, non_blocking=True)
    y_len = batch["y_lengths"].to(device)
    dur = batch["dur"].to(device)
    dmask = batch["dmask"].to(device)
    wav = batch["wav"].to(device, non_blocking=True)
    state.net_g.train(), state.net_d.train()

    with _autocast(cfg, device):
        y_hat, logw, _, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = \
            state.net_g(x, x_len, spec, y_len, dur)
    y_hat = y_hat.float()
    mel = spec_to_mel(spec, cfg.audio)
    y_mel = commons.slice_segments(mel, ids_slice, seg // hop)
    y_hat_mel = mel_spectrogram(y_hat.squeeze(1), cfg.audio)
    y = commons.slice_segments(wav, ids_slice * hop, seg)

    # discriminator
    with _autocast(cfg, device):
        y_d_r, y_d_g, _, _ = state.net_d(y, y_hat.detach())
    loss_d, _, _ = discriminator_loss([o.float() for o in y_d_r], [o.float() for o in y_d_g])
    state.optim_d.zero_grad(set_to_none=True)
    loss_d.backward()
    gn_d = commons.clip_grad_value_(state.net_d.parameters(), t.grad_clip)
    state.optim_d.step()

    # generator
    with _autocast(cfg, device):
        y_d_r, y_d_g, fmap_r, fmap_g = state.net_d(y, y_hat)
    loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * t.c_mel
    loss_kl = kl_loss(z_p.float(), logs_q.float(), m_p.float(), logs_p.float(), z_mask.float()) \
        * t.c_kl
    loss_dur = duration_loss(logw, dur, dmask * x_mask.squeeze(1)) * t.c_dur
    loss_fm = feature_loss([[f.float() for f in fs] for fs in fmap_r],
                           [[f.float() for f in fs] for fs in fmap_g]) * t.c_fm
    loss_adv, _ = generator_loss([o.float() for o in y_d_g])
    loss_g = loss_mel + loss_kl + loss_dur + loss_fm + loss_adv
    state.optim_g.zero_grad(set_to_none=True)
    loss_g.backward()
    gn_g = commons.clip_grad_value_(state.net_g.parameters(), t.grad_clip)
    state.optim_g.step()
    state.step += 1
    return {"loss/g_total": loss_g.item(), "loss/d": loss_d.item(), "loss/mel": loss_mel.item(),
            "loss/kl": loss_kl.item(), "loss/dur": loss_dur.item(), "loss/fm": loss_fm.item(),
            "loss/adv": loss_adv.item(), "grad/g": float(gn_g), "grad/d": float(gn_d)}


@torch.no_grad()
def val_losses(state: TrainState, batch: dict, cfg: DictConfig, device: torch.device) -> dict:
    """Teacher-forced val: duration loss + mel L1 of the full-length posterior reconstruction."""
    net = state.net_g
    net.eval()
    x = batch["ids"].to(device)
    x_len = batch["x_lengths"].to(device)
    spec = batch["spec"].to(device)
    y_len = batch["y_lengths"].to(device)
    dur = batch["dur"].to(device)
    dmask = batch["dmask"].to(device)
    with _autocast(cfg, device):
        h, x_mask = net.enc_p.encode(x, x_len)
        logw = net.dp(h, x_mask)
        z, m_q, logs_q, y_mask = net.enc_q(spec, y_len)
        y_hat = net.dec(z * y_mask)
    y_hat = y_hat.float().squeeze(1)
    mel = spec_to_mel(spec, cfg.audio)
    mel_hat = mel_spectrogram(y_hat, cfg.audio)[..., : mel.size(-1)]
    m = y_mask[..., : mel_hat.size(-1)].float()
    l_mel = (torch.abs(mel[..., : mel_hat.size(-1)] - mel_hat) * m).sum() / (m.sum() * mel.size(1))
    l_dur = duration_loss(logw, dur, dmask * x_mask.squeeze(1))
    return {"val/mel_l1": l_mel.item(), "val/dur": l_dur.item()}
