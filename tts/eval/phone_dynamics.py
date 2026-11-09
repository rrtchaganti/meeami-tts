"""Within-phone spectral movement: the phase-2 metric for M1+pos vs M1-cont (DECISIONS D-76).

On LJ val with the ground-truth MFA durations, each checkpoint resynthesizes every clip. Per
frame, spectral change = L2 distance between consecutive log-mel frames. For phones with at least
`min_frames` frames (no sil/sp), the mean change inside the phone (excluding its onset frame) is
reported overall and per third, along with the change at phone onsets. The LJ recordings give the
target. A flat per-phone prior shows as less movement inside phones and a larger
onset/inside ratio (word-ending investigation, D-74 section).

    python -m tts.eval.phone_dynamics --ckpt export/m1/ckpt_last.pt [--ckpt ...] [--device cuda]
Writes results/phone_dynamics/<name>.json (name = checkpoint's parent folder) and ground_truth.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tts.config import load_config, repo_path

PAUSE = {"sp", "sil"}


def frame_change(logmel: np.ndarray) -> np.ndarray:
    """[n_mels, T] -> [T]; element t = ||m[:, t] - m[:, t-1]||, element 0 = 0."""
    return np.r_[0.0, np.linalg.norm(np.diff(logmel, axis=1), axis=0)]


def phone_stats(dm: np.ndarray, phones: list[str], durations: list[int], min_frames: int) -> dict:
    """Per-phone lists: mean change inside (frames 1..d-1), per third, and at the onset frame."""
    out = {"inside": [], "onset": [], "thirds": [[], [], []]}
    d = np.asarray(durations, int)
    ends = np.cumsum(d)
    starts = ends - d
    for i, p in enumerate(phones):
        if p in PAUSE or d[i] < min_frames:
            continue
        a, b = int(starts[i]), int(ends[i])
        seg = dm[a + 1:b]
        out["inside"].append(float(seg.mean()))
        for t, part in enumerate(np.array_split(seg, 3)):
            out["thirds"][t].append(float(part.mean()))
        if a > 0:
            out["onset"].append(float(dm[a]))
    return out


def summarize(acc: dict) -> dict:
    ins, on = float(np.mean(acc["inside"])), float(np.mean(acc["onset"]))
    return {"n_phones": len(acc["inside"]), "inside": round(ins, 4), "onset": round(on, 4),
            "onset_over_inside": round(on / ins, 4),
            "inside_thirds": [round(float(np.mean(t)), 4) for t in acc["thirds"]]}


def _merge(acc: dict, s: dict) -> None:
    acc["inside"] += s["inside"]
    acc["onset"] += s["onset"]
    for t in range(3):
        acc["thirds"][t] += s["thirds"][t]


def main(argv=None) -> int:
    from tts.audio import load_wav, mel_spectrogram
    from tts.infer.systems import VitsSystem
    from tts.models.vits_mfa.data import read_manifest

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config", default="configs/inference.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(repo_path(a.config))
    pc = cfg.phone_dynamics
    rows = read_manifest(pc.manifest)
    out_dir = repo_path(pc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hop = int(cfg.audio.hop_length)

    def logmel(y):
        y = torch.from_numpy(np.asarray(y, np.float32))[None]
        return mel_spectrogram(y, cfg.audio)[0].numpy()

    def run(name, wav_of):
        acc = {"inside": [], "onset": [], "thirds": [[], [], []]}
        for r in rows:
            n = int(r["n_frames"])
            dm = frame_change(logmel(wav_of(r)[: n * hop])[:, :n])
            _merge(acc, phone_stats(dm, r["phones"], r["durations"], int(pc.min_frames)))
        res = {"name": name, "n_clips": len(rows), "noise_scale": float(pc.noise_scale),
               "seed": int(cfg.seed), **summarize(acc)}
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps(res), flush=True)
        return res

    gt = run("ground_truth", lambda r: load_wav(repo_path(r["wav"]))[0])
    for ck in a.ckpt:
        sysm = VitsSystem(repo_path(ck), a.device)
        torch.manual_seed(int(cfg.seed))
        res = run(Path(ck).parent.name,
                  lambda r: sysm.synthesize(r["phones"], r["durations"],
                                            noise_scale=float(pc.noise_scale)))
        print(f"{res['name']}: inside {res['inside'] / gt['inside']:.3f} x LJ, middle third "
              f"{res['inside_thirds'][1] / gt['inside_thirds'][1]:.3f} x LJ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
