"""Section 8 plot 1: reference and output words as bars on one time axis above both spectrograms."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _spec(y: np.ndarray, sr: int):
    import librosa

    S = librosa.feature.melspectrogram(y=y.astype(np.float32), sr=sr, n_fft=1024, hop_length=256,
                                       n_mels=80)
    return librosa.power_to_db(S + 1e-10, ref=np.max), len(y) / sr


def timing_plot(ref_y, ref_sr, out_y, out_sr, ref_words, out_words, path: Path, title: str = ""):
    fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True,
                           gridspec_kw={"height_ratios": [1.2, 2, 2]})
    for row, (words, color, label) in enumerate([(ref_words, "#4C72B0", "reference"),
                                                 (out_words, "#DD8452", "output")]):
        for w in words:
            ax[0].barh(1 - row, w["end"] - w["start"], left=w["start"], height=0.8, color=color,
                       edgecolor="white", alpha=0.85)
            ax[0].text(0.5 * (w["start"] + w["end"]), 1 - row, w["word"], ha="center",
                       va="center", fontsize=7, clip_on=True)
    ax[0].set_yticks([1, 0], ["reference", "output"])
    ax[0].set_title(title or "Word timing: reference vs output")
    for a, (y, sr, name) in zip(ax[1:], [(ref_y, ref_sr, "reference"), (out_y, out_sr, "output")]):
        S, dur = _spec(y, sr)
        a.imshow(S, origin="lower", aspect="auto", extent=[0, dur, 0, S.shape[0]], cmap="magma")
        a.set_ylabel(f"{name}\nmel bin")
        for w in ref_words:
            a.axvline(w["start"], color="#4C72B0", lw=0.6, alpha=0.7)
            a.axvline(w["end"], color="#4C72B0", lw=0.6, alpha=0.7, ls="--")
    ax[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
