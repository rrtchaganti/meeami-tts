"""V0 baseline: the official pretrained LJSpeech VITS, as released (MAS-trained, stochastic duration
predictor, blank tokens, IPA characters), driven by the same reference word timing as every other
system (DECISIONS D-90, added after freeze-v1 as an inference-only baseline).

V0 is the system that M1 changes: M1 is this checkpoint fine-tuned with MFA durations instead of
MAS. V0 is not retrained. It plugs into the Section 5 pipeline through the same two calls as M0
and the VITS-MFA systems, at the level of our ARPAbet tokens:

* Input text. CLAUDE.md allows one pronunciation source, data/lexicon/lexicon.txt, so V0 does not
  run espeak. Each ARPAbet token is written in the IPA characters espeak en-us uses in the
  original VITS training filelists (`ARPA_IPA`). The stress mark goes directly before the vowel, as
  in those filelists. A space separates words, `sp` becomes ", ", the final `sil` becomes ".".
  As in `add_blank: true`, a blank is placed before, between and after the characters.
* Ownership. Every character belongs to one ARPAbet token, and so does the blank before it
  (`v0.blank_owner: before`, chosen on dev over "after", D-90). The leading blank belongs to the
  opening `sil`, the final blank to the closing `sil`. A word's trailing space belongs to its last
  phone.
* `predict_log_durations`: the stochastic duration predictor (SDP) gives a duration per character
  or blank. A token's native duration is the sum over the characters and blanks it owns.
* `synthesize`: each token's frames are split over its characters and blanks in proportion to the
  SDP durations, with cumulative rounding per token. This keeps every token's total exact, so the
  output length and the word boundaries match the other systems frame for frame.
The SDP noise is drawn from a generator seeded per call (`v0.sdp_seed`). Both calls therefore see
the same native durations for the same text.

StochasticDurationPredictor is from jaywalnut310/vits `models.py` @ 2e561ba (MIT, see
tts/models/vits_mfa/LICENSE.vits). Only the reverse (inference) path is kept; its noise is an
argument instead of an internal torch.randn.
"""
from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from tts.config import repo_path
from tts.models.vits_mfa import modules
from tts.text.symbols import SIL, SP

# jaywalnut310/vits text/symbols.py (178 symbols; index = embedding row of the pretrained model)
_PAD = "_"
_PUNCT = ';:,.!?¡¿—…"«»“” '
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_IPA = ("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
VITS_SYMBOLS = [_PAD] + list(_PUNCT) + list(_LETTERS) + list(_IPA)
_ID = {s: i for i, s in enumerate(VITS_SYMBOLS)}
BLANK = 0  # add_blank intersperses id 0

# espeak en-us conventions as found in the VITS LJSpeech .cleaned filelists
_VOWELS = {
    "AA": ("ɑː", "ɑː"), "AE": ("æ", "æ"), "AH": ("ə", "ʌ"), "AO": ("ɔː", "ɔː"),
    "AW": ("aʊ", "aʊ"), "AY": ("aɪ", "aɪ"), "EH": ("ɛ", "ɛ"), "ER": ("ɚ", "ɜː"),
    "EY": ("eɪ", "eɪ"), "IH": ("ɪ", "ɪ"), "IY": ("i", "iː"), "OW": ("oʊ", "oʊ"),
    "OY": ("ɔɪ", "ɔɪ"), "UH": ("ʊ", "ʊ"), "UW": ("uː", "uː"),
}  # (unstressed form, stressed form)
_CONS = {
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ",
    "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}
_STRESS = {"0": "", "1": "ˈ", "2": "ˌ"}


def arpa_to_ipa(tok: str) -> str:
    if tok in _CONS:
        return _CONS[tok]
    base, st = tok[:-1], tok[-1]
    if base not in _VOWELS or st not in _STRESS:
        raise KeyError(f"V0: no IPA for token {tok!r}")
    return _STRESS[st] + _VOWELS[base][0 if st == "0" else 1]


def to_chars(tokens: list[str], word_ids: list[int]) -> tuple[list[str], list[int]]:
    """IPA characters of the token sequence and, for each character, the index of its token."""
    chars, owner = [], []
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if t == SIL:
            s = "." if i == len(tokens) - 1 and i > 0 else ""
        elif t == SP:
            s = "," if nxt == SIL else ", "
        else:
            s = arpa_to_ipa(t)
            if nxt not in (None, SIL, SP) and word_ids[i + 1] != word_ids[i]:
                s += " "
        chars += list(s)
        owner += [i] * len(s)
    return chars, owner


def intersperse(chars: list[str], owner: list[int], blank_owner: str = "before"
                ) -> tuple[list[int], list[int]]:
    """add_blank: ids [_, c1, _, c2, _, ..., cn, _].

    blank_owner "after": each blank belongs to the character before it (the leading blank to
    token 0). "before": each blank belongs to the character after it (the final blank to the last
    token); the leading blank still belongs to token 0, which owns no character of its own.
    Chosen on dev (D-90): with "after", output words came out ~35 ms early."""
    if blank_owner not in ("after", "before"):
        raise ValueError(blank_owner)
    ids, own = [BLANK], [0]  # the leading blank always belongs to token 0 (the opening sil)
    for k, (c, o) in enumerate(zip(chars, owner)):
        if c not in _ID:
            raise KeyError(f"V0: character {c!r} not in the VITS symbol set")
        if blank_owner == "after":
            nb = o
        else:
            nb = owner[k + 1] if k + 1 < len(owner) else o
        ids += [_ID[c], BLANK]
        own += [o, nb]
    return ids, own


def split_frames(durations: list[int], own: list[int], w: np.ndarray) -> list[int]:
    """Each token's integer frames over the sub-tokens it owns, in proportion to w, by cumulative
    rounding inside the token (so every token keeps its exact total)."""
    own = np.asarray(own)
    out = np.zeros(len(own), dtype=int)
    for i, d in enumerate(durations):
        idx = np.flatnonzero(own == i)
        if len(idx) == 0:
            raise ValueError(f"V0: token {i} owns no characters")
        ww = np.maximum(w[idx], 1e-6)
        cum = np.rint(np.cumsum(ww) * (d / ww.sum())).astype(int)
        cum[-1] = d
        out[idx] = np.diff(np.concatenate([[0], cum]))
    return out.tolist()


class StochasticDurationPredictor(nn.Module):
    """Inference path of VITS's SDP (same parameter names as the checkpoint's `dp.*`)."""

    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4):
        super().__init__()
        filter_channels = in_channels  # as in the original
        self.log_flow = modules.Log()
        self.flows = nn.ModuleList([modules.ElementwiseAffine(2)])
        for _ in range(n_flows):
            self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())
        # posterior branch: training only, kept so the checkpoint loads strictly
        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3,
                                          p_dropout=p_dropout)
        self.post_flows = nn.ModuleList([modules.ElementwiseAffine(2)])
        for _ in range(4):
            self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)

    @torch.no_grad()
    def infer(self, x, x_mask, noise):
        """x [b, h, t] encoder states, noise [b, 2, t] (already scaled) -> log w [b, 1, t]."""
        x = self.proj(self.convs(self.pre(x), x_mask)) * x_mask
        flows = list(reversed(self.flows))
        flows = flows[:-2] + [flows[-1]]  # remove a useless vflow (original)
        z = noise * x_mask
        for flow in flows:
            z = flow(z, x_mask, g=x, reverse=True)
        return z[:, :1]


class V0:
    hop = 256
    sample_rate = 22050

    def __init__(self, cfg: DictConfig, device: str = "cuda"):
        from tts.models.vits_mfa.models import build_generator, load_pretrained

        self.cfg, self.device = cfg, torch.device(device)
        v = cfg.v0
        if cfg.model.use_pos or cfg.model.get("diph_split", False):
            raise ValueError("V0 needs the original encoder (use_pos and diph_split false)")
        self.hop, self.sample_rate = int(cfg.audio.hop_length), int(cfg.audio.sample_rate)
        sd = torch.load(repo_path(cfg.pretrained.vits_lj.path), map_location="cpu",
                        weights_only=True)["model"]
        net = build_generator(cfg)
        net.enc_p.emb = nn.Embedding(len(VITS_SYMBOLS), int(cfg.model.hidden_channels))
        if sd["enc_p.emb.weight"].shape[0] != len(VITS_SYMBOLS):
            raise ValueError("V0: symbol table does not match the checkpoint")
        _, _, missing = load_pretrained(net, sd, reinit_prefixes=("dp.",))
        bad = [k for k in missing if not k.startswith("dp.")]
        if bad:
            raise KeyError(f"V0: weights missing from the checkpoint: {bad[:5]}")
        net.dp = None  # M1's deterministic predictor is not part of V0
        self.net = net.to(self.device).eval()
        self.net.dec.remove_weight_norm()
        m = cfg.model
        self.sdp = StochasticDurationPredictor(int(m.hidden_channels), 192, 3, 0.5).eval()
        self.sdp.load_state_dict({k[3:]: t for k, t in sd.items() if k.startswith("dp.")})
        self.sdp.to(self.device)
        self.noise_w = float(v.sdp_noise_scale)
        self.seed = int(v.sdp_seed)
        self.blank_owner = str(v.blank_owner)

    def _sub(self, tokens, word_ids):
        if word_ids is None:
            raise ValueError("V0 needs word_ids (word boundaries become spaces)")
        chars, owner = to_chars(tokens, word_ids)
        ids, own = intersperse(chars, owner, self.blank_owner)
        x = torch.tensor([ids], device=self.device)
        return x, torch.tensor([len(ids)], device=self.device), own

    @torch.no_grad()
    def _sdp_w(self, x, n):
        h, x_mask = self.net.enc_p.encode(x, n)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        noise = torch.randn(1, 2, x.size(1), generator=g).to(self.device) * self.noise_w
        logw = self.sdp.infer(h, x_mask, noise)
        return torch.exp(logw)[0, 0].float().cpu().numpy()

    def predict_log_durations(self, tokens, word_ids=None) -> np.ndarray:
        x, n, own = self._sub(tokens, word_ids)
        w = self._sdp_w(x, n)
        tot = np.bincount(np.asarray(own), weights=w, minlength=len(tokens))
        return np.log(np.maximum(tot, 1e-3))

    def synthesize(self, tokens, durations, noise_scale: float = 0.667, word_ids=None):
        x, n, own = self._sub(tokens, word_ids)
        d = split_frames([int(v) for v in durations], own, self._sdp_w(x, n))
        wav, _ = self.net.infer(x, n, torch.tensor([d], device=self.device),
                                noise_scale=noise_scale)
        y = wav[0, 0].float().cpu().numpy()
        n_samp = int(sum(durations)) * self.hop
        return y[:n_samp] if len(y) >= n_samp else np.pad(y, (0, n_samp - len(y)))


def ipa_line(tokens: list[str], word_ids: list[int]) -> str:
    """Readable V0 input (for logs and tests)."""
    return "".join(to_chars(tokens, word_ids)[0])


__all__ = ["V0", "VITS_SYMBOLS", "arpa_to_ipa", "to_chars", "intersperse", "split_frames",
           "ipa_line"]
