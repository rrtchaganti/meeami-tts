"""Diphthong split at the model boundary (DECISIONS D-81; a PLAN Section 4 deviation, opt-in via
`model.diph_split`).

ARPAbet AW AY EY OW OY are single tokens, so each has one constant prior over all its frames, and
the glide must come from the flow alone (D-80). With the split, every diphthong token `X` becomes
`X` (first part) + `X+` (second part). The model learns two prior targets, as the pretrained VITS
had with its IPA input (`a`, `ɪ` plus blanks).

- Manifests, lexicon, MFA alignments, stats and the inference pipeline keep the 72-symbol tokens.
  The split happens only in the VITS dataset and `VitsSystem`.
- Durations: the first part gets max(1, round(d * first_frac)) frames, the second the rest (0 when
  d = 1; such a 0-frame part is excluded from the duration loss).
- Duration prediction for the pipeline: the per-part predictions are merged back per original token
  as log(exp(a) + exp(b)).
- New ids 72..86 follow the 72 base symbols. Their embeddings start as copies of the base token's,
  so a model branched from M1 starts with the same prior over the whole diphthong.
"""
from __future__ import annotations

import numpy as np
import torch

from tts.text.symbols import N_VOCAB, STRESSES, SYMBOL_TO_ID

DIPHTHONGS = ["AW", "AY", "EY", "OW", "OY"]
SPLIT = frozenset(d + s for d in DIPHTHONGS for s in STRESSES)
SECOND = [d + s + "+" for d in DIPHTHONGS for s in STRESSES]
SECOND_TO_ID = {t: N_VOCAB + i for i, t in enumerate(SECOND)}
N_VOCAB_SPLIT = N_VOCAB + len(SECOND)
assert len(SPLIT) == 15 and N_VOCAB_SPLIT == 87


def split_tokens(tokens: list[str], durations: list[int] | None = None, first_frac: float = 0.5):
    """-> (tokens', durations' or None, src) with src[j] = index of the original token."""
    out_t, out_d, src = [], [], []
    for i, t in enumerate(tokens):
        if t in SPLIT:
            out_t += [t, t + "+"]
            src += [i, i]
            if durations is not None:
                d = int(durations[i])
                a = d if d < 2 else min(d - 1, max(1, int(round(d * first_frac))))
                out_d += [a, d - a]
        else:
            out_t.append(t)
            src.append(i)
            if durations is not None:
                out_d.append(int(durations[i]))
    return out_t, (out_d if durations is not None else None), src


def to_ids_split(tokens: list[str]) -> list[int]:
    """Like symbols.to_ids, plus the second-part symbols; KeyError on anything unknown."""
    return [SECOND_TO_ID[t] if t.endswith("+") else SYMBOL_TO_ID[t] for t in tokens]


def merge_log_durations(logw: np.ndarray, src: list[int], n_tokens: int) -> np.ndarray:
    """Per-part log durations -> per original token: log of the summed frames."""
    out = np.full(n_tokens, -np.inf)
    for j, i in enumerate(src):
        out[i] = np.logaddexp(out[i], float(logw[j]))
    return out


def expand_embedding(old: torch.Tensor) -> torch.Tensor:
    """[72, h] base embedding -> [87, h]; each X+ row starts as a copy of X."""
    if old.size(0) != N_VOCAB:
        raise ValueError(f"expected {N_VOCAB} embedding rows, got {old.size(0)}")
    extra = old[[SYMBOL_TO_ID[t[:-1]] for t in SECOND]]
    return torch.cat([old, extra.clone()], 0)
