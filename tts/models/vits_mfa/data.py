"""Manifest dataset and samplers for VITS-MFA (docs/PLAN.md Section 4 change 6, Sections 6-7).

Rows come from data/manifests/*.jsonl (Section 10 schema). Each item: token ids, MFA frame
durations, linear spectrogram of the 22.05 kHz audio (tts.audio.spectrogram, fp32), waveform,
and the duration-loss mask (excludes sil, padding and augmented rows).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Sampler

from tts.audio import load_wav, spectrogram
from tts.config import REPO_ROOT, repo_path
from tts.text.symbols import SIL, SYMBOL_TO_ID, to_ids


def read_manifest(path: str | Path, root: Path = REPO_ROOT) -> list[dict]:
    with open(repo_path(path, root), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        if r.get("durations") is None:
            raise ValueError(f"{path}: row {r['id']} has no durations (run make align-lj / aug)")
        if sum(r["durations"]) != r["n_frames"]:
            raise ValueError(f"{path}: row {r['id']} durations sum != n_frames")
    return rows


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict], cfg: DictConfig, root: Path = REPO_ROOT):
        self.rows = rows
        self.audio = cfg.audio
        self.hop = int(cfg.audio.hop_length)
        self.sr = int(cfg.audio.sample_rate)
        self.root = root

    def __len__(self):
        return len(self.rows)

    def lengths(self) -> list[int]:
        return [r["n_frames"] for r in self.rows]

    def __getitem__(self, i):
        r = self.rows[i]
        y, sr = load_wav(repo_path(r["wav"], self.root))
        if sr != self.sr:
            raise ValueError(f"{r['wav']}: {sr} Hz, expected {self.sr}")
        wav = torch.from_numpy(y)
        spec = spectrogram(wav, self.audio)
        n = r["n_frames"]
        if spec.size(1) != n:
            raise ValueError(f"{r['id']}: spectrogram has {spec.size(1)} frames, manifest {n}")
        wav = wav[: n * self.hop]
        if wav.numel() < n * self.hop:
            wav = torch.nn.functional.pad(wav, (0, n * self.hop - wav.numel()))
        ids = torch.tensor(to_ids(r["phones"]), dtype=torch.long)
        dur = torch.tensor(r["durations"], dtype=torch.long)
        dmask = (ids != SYMBOL_TO_ID[SIL]).float() * (0.0 if r.get("is_aug") else 1.0)
        return {"id": r["id"], "ids": ids, "dur": dur, "dmask": dmask, "spec": spec, "wav": wav,
                "is_aug": bool(r.get("is_aug"))}


def collate(batch: list[dict]) -> dict:
    b = len(batch)
    tx = max(x["ids"].numel() for x in batch)
    ty = max(x["spec"].size(1) for x in batch)
    tw = max(x["wav"].numel() for x in batch)
    c = batch[0]["spec"].size(0)
    out = {"ids": torch.zeros(b, tx, dtype=torch.long), "dur": torch.zeros(b, tx, dtype=torch.long),
           "dmask": torch.zeros(b, tx), "spec": torch.zeros(b, c, ty), "wav": torch.zeros(b, 1, tw),
           "x_lengths": torch.zeros(b, dtype=torch.long), "y_lengths": torch.zeros(b, dtype=torch.long),
           "is_aug": torch.zeros(b, dtype=torch.bool), "id": [x["id"] for x in batch]}
    for i, x in enumerate(batch):
        n, m, w = x["ids"].numel(), x["spec"].size(1), x["wav"].numel()
        out["ids"][i, :n] = x["ids"]
        out["dur"][i, :n] = x["dur"]
        out["dmask"][i, :n] = x["dmask"]
        out["spec"][i, :, :m] = x["spec"]
        out["wav"][i, 0, :w] = x["wav"]
        out["x_lengths"][i], out["y_lengths"][i] = n, m
        out["is_aug"][i] = x["is_aug"]
    return out


class MixedBucketSampler(Sampler):
    """Length-bucketed batches; with aug_fraction f each batch holds round(bs*(1-f)) rows from the
    base pool and the rest from the augmented pool (Section 6: 60/40). One epoch = one pass over
    the base pool. Indices refer to the concatenated dataset [base..., aug...]."""

    def __init__(self, base_lengths, aug_lengths, batch_size, aug_fraction=0.0, seed=1234,
                 bucket_batches=50):
        self.base = np.asarray(base_lengths)
        self.aug = np.asarray(aug_lengths)
        self.bs = batch_size
        self.f = aug_fraction if len(self.aug) else 0.0
        self.n_aug = int(round(batch_size * self.f))
        self.n_base = batch_size - self.n_aug
        if self.n_base < 1:
            raise ValueError("aug_fraction leaves no base rows per batch")
        self.seed = seed
        self.epoch = 0
        self.bucket = bucket_batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _groups(self, lengths, k, rng, n_groups):
        idx = rng.permutation(len(lengths))
        groups = []
        chunk = k * self.bucket
        while len(groups) < n_groups:
            for s in range(0, len(idx), chunk):
                c = idx[s:s + chunk]
                c = c[np.argsort(lengths[c], kind="stable")]
                groups += [c[j:j + k] for j in range(0, len(c) - k + 1, k)]
            idx = rng.permutation(len(lengths))
        return groups[:n_groups]

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        n_batches = len(self.base) // self.n_base
        base = self._groups(self.base, self.n_base, rng, n_batches)
        batches = [list(g) for g in base]
        if self.n_aug:
            aug = self._groups(self.aug, self.n_aug, rng, n_batches)
            off = len(self.base)
            # pair by length rank so batch-mates have similar lengths
            base_order = np.argsort([self.base[g].mean() for g in base])
            aug_order = np.argsort([self.aug[g].mean() for g in aug])
            batches = [None] * n_batches
            for bi, ai in zip(base_order, aug_order):
                batches[bi] = list(base[bi]) + [int(x) + off for x in aug[ai]]
        order = rng.permutation(n_batches)
        return iter([[int(x) for x in batches[i]] for i in order])

    def __len__(self):
        return len(self.base) // self.n_base


def fixed_val_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    """The n fixed LJ val utterances logged as audio (deterministic by seed)."""
    rng = random.Random(seed)
    return sorted(rng.sample(rows, min(n, len(rows))), key=lambda r: r["id"])
