"""VITS with MFA durations (docs/PLAN.md Section 4, "M1: VITS with MFA durations").

Adapted from jaywalnut310/vits `models.py` @ 2e561ba (MIT, see LICENSE.vits). Changes:
 1. No MAS / monotonic_align: the alignment is a one-hot expansion of the MFA frame durations.
 2. add_blank is false (handled by the data: no blank tokens are interspersed).
 3. n_vocab = 72 (tts.text.symbols); the text embedding is re-initialized when loading the
    pretrained LJ generator (`load_pretrained`), every other shape-matching weight is loaded.
 4. Deterministic DurationPredictor (2 convs, 256 filters, kernel 3, dropout 0.5) on the detached
    text-encoder output, predicting log duration; MSE masked to exclude sil, padding and
    augmented rows (`duration_loss`).
 5. use_pos: expand the encoder hidden states by durations first, add a learned embedding of
    each frame's relative position in its phone (n_pos_bins bins of (k + 0.5) / d), then the
    1x1 projection to (m, logs). use_pos=false keeps the original order (project, then expand).
 6. diph_split (DECISIONS D-81, opt-in, PLAN deviation): n_vocab 87; diphthongs become two
    tokens (tts/text/diph.py).
Generator, posterior encoder, flow and discriminators are unchanged.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from . import attentions, commons, modules
from .commons import get_padding, init_weights


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size,
                                padding=kernel_size // 2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = torch.detach(x)
        x = self.drop(self.norm_1(torch.relu(self.conv_1(x * x_mask))))
        x = self.drop(self.norm_2(torch.relu(self.conv_2(x * x_mask))))
        return self.proj(x * x_mask) * x_mask  # log duration (frames)


class TextEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers,
                 kernel_size, p_dropout, use_pos=False, n_pos_bins=16):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.use_pos = use_pos
        self.n_pos_bins = n_pos_bins
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)
        self.encoder = attentions.Encoder(hidden_channels, filter_channels, n_heads, n_layers,
                                          kernel_size, p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        if use_pos:
            self.pos_emb = nn.Embedding(n_pos_bins, hidden_channels)
            nn.init.zeros_(self.pos_emb.weight)  # starts as the use_pos=false model

    def encode(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        return x, x_mask

    def prior(self, x, x_mask, attn, y_mask, pos_bins):
        """Frame-level prior (m_p, logs_p) [b, c, t_y] from token states x [b, h, t_x]."""
        if self.use_pos:
            h = torch.matmul(attn.squeeze(1), x.transpose(1, 2)).transpose(1, 2)  # [b, h, t_y]
            h = h + self.pos_emb(pos_bins).transpose(1, 2)
            stats = self.proj(h) * y_mask
        else:
            stats = self.proj(x) * x_mask
            stats = torch.matmul(attn.squeeze(1), stats.transpose(1, 2)).transpose(1, 2)
        return torch.split(stats, self.out_channels, dim=1)


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4,
                 gin_channels=0):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(
                channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, gin_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                              gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(
                upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock(ch, k, d))
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                r = self.resblocks[i * self.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        return torch.tanh(self.conv_post(x))

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        pad = (get_padding(kernel_size, 1), 0)
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=pad)),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=pad)),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=pad)),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=pad)),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=pad)),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for layer in self.convs:
            x = F.leaky_relu(layer(x), modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for layer in self.convs:
            x = F.leaky_relu(layer(x), modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(use_spectral_norm)]
            + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in [2, 3, 5, 7, 11]])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r), y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r), fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# --- duration utilities ----------------------------------------------------------------------
def duration_path(durations: torch.Tensor, x_mask: torch.Tensor, y_mask: torch.Tensor):
    """One-hot alignment [b, 1, t_y, t_x] from integer durations [b, t_x] (change 1)."""
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    return commons.generate_path(durations.unsqueeze(1).to(x_mask.dtype), attn_mask)


def position_bins(durations: torch.Tensor, t_y: int, n_bins: int) -> torch.Tensor:
    """Per frame: bin of (k + 0.5) / d, k = index of the frame inside its phone (change 5)."""
    b = durations.size(0)
    out = torch.zeros(b, t_y, dtype=torch.long, device=durations.device)
    for i in range(b):
        d = durations[i][durations[i] > 0].long()
        if d.numel() == 0:
            continue
        starts = torch.cumsum(d, 0) - d
        frame_d = torch.repeat_interleave(d, d)
        k = torch.arange(frame_d.numel(), device=d.device) - torch.repeat_interleave(starts, d)
        rel = (k.float() + 0.5) / frame_d.float()
        n = min(t_y, rel.numel())
        out[i, :n] = torch.clamp((rel[:n] * n_bins).long(), max=n_bins - 1)
    return out


def duration_loss(logw: torch.Tensor, durations: torch.Tensor, loss_mask: torch.Tensor):
    """MSE on log frames over tokens where loss_mask = 1 (change 4)."""
    target = torch.log(durations.float().clamp_min(1.0)).unsqueeze(1)
    m = loss_mask.unsqueeze(1).float()
    denom = m.sum().clamp_min(1.0)
    return torch.sum(((logw.float() - target) ** 2) * m) / denom


# --- synthesizer -----------------------------------------------------------------------------
class SynthesizerMFA(nn.Module):
    def __init__(self, n_vocab, spec_channels, segment_size, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock,
                 resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, dp_filter_channels=256,
                 dp_kernel_size=3, dp_dropout=0.5, use_pos=False, n_pos_bins=16, **_):
        super().__init__()
        self.segment_size = segment_size
        self.use_pos = use_pos
        self.n_pos_bins = n_pos_bins
        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels, filter_channels,
                                 n_heads, n_layers, kernel_size, p_dropout, use_pos, n_pos_bins)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes,
                             resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                             upsample_kernel_sizes)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4)
        self.dp = DurationPredictor(hidden_channels, dp_filter_channels, dp_kernel_size,
                                    dp_dropout)

    def forward(self, x, x_lengths, y, y_lengths, durations):
        """x [b, t_x] token ids, y [b, spec, t_y] linear spec, durations [b, t_x] frames."""
        h, x_mask = self.enc_p.encode(x, x_lengths)
        logw = self.dp(h, x_mask)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths)
        z_p = self.flow(z, y_mask)
        attn = duration_path(durations * x_mask.squeeze(1).long(), x_mask, y_mask)
        bins = position_bins(durations, y.size(2), self.n_pos_bins) if self.use_pos else None
        m_p, logs_p = self.enc_p.prior(h, x_mask, attn, y_mask, bins)
        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice)
        return o, logw, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.no_grad()
    def predict_log_durations(self, x, x_lengths):
        h, x_mask = self.enc_p.encode(x, x_lengths)
        return self.dp(h, x_mask).squeeze(1), x_mask.squeeze(1)

    @torch.no_grad()
    def infer(self, x, x_lengths, durations, noise_scale=0.667):
        """Section 5 step 9: encoder -> expand by given durations -> prior sample -> inverse flow
        -> decoder. Returns (audio [b, 1, t_y * hop], y_mask)."""
        h, x_mask = self.enc_p.encode(x, x_lengths)
        durations = durations.long() * x_mask.squeeze(1).long()
        y_lengths = durations.sum(1).clamp_min(1)
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn = duration_path(durations, x_mask, y_mask)
        bins = (position_bins(durations, y_mask.size(2), self.n_pos_bins)
                if self.use_pos else None)
        m_p, logs_p = self.enc_p.prior(h, x_mask, attn, y_mask, bins)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, reverse=True)
        return self.dec(z * y_mask), y_mask


def build_generator(cfg) -> SynthesizerMFA:
    from tts.text.diph import N_VOCAB_SPLIT
    from tts.text.symbols import N_VOCAB

    m = cfg.model
    return SynthesizerMFA(
        n_vocab=N_VOCAB_SPLIT if m.get("diph_split", False) else N_VOCAB, spec_channels=cfg.audio.filter_length // 2 + 1,
        segment_size=cfg.train.segment_size // cfg.audio.hop_length,
        inter_channels=m.inter_channels, hidden_channels=m.hidden_channels,
        filter_channels=m.filter_channels, n_heads=m.n_heads, n_layers=m.n_layers,
        kernel_size=m.kernel_size, p_dropout=m.p_dropout, resblock=str(m.resblock),
        resblock_kernel_sizes=list(m.resblock_kernel_sizes),
        resblock_dilation_sizes=[list(d) for d in m.resblock_dilation_sizes],
        upsample_rates=list(m.upsample_rates), upsample_initial_channel=m.upsample_initial_channel,
        upsample_kernel_sizes=list(m.upsample_kernel_sizes),
        dp_filter_channels=m.dp_filter_channels, dp_kernel_size=m.dp_kernel_size,
        dp_dropout=m.dp_dropout, use_pos=bool(m.use_pos), n_pos_bins=int(m.n_pos_bins))


def load_pretrained(model: nn.Module, state_dict: dict, reinit_prefixes=("enc_p.emb.",)):
    """Load every weight whose name and shape match, except `reinit_prefixes` (change 3).

    Returns (loaded, skipped) key lists; skipped entries are 'key: reason'.
    """
    own = model.state_dict()
    loaded, skipped = [], []
    new = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in reinit_prefixes):
            skipped.append(f"{k}: re-initialized (new symbol set)")
        elif k not in own:
            skipped.append(f"{k}: not in model")
        elif own[k].shape != v.shape:
            skipped.append(f"{k}: shape {tuple(v.shape)} != {tuple(own[k].shape)}")
        else:
            new[k] = v
            loaded.append(k)
    model.load_state_dict(new, strict=False)
    missing = [k for k in own if k not in new]
    return loaded, skipped, missing
