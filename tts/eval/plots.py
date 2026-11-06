"""Section 8 plots (results/plots/). Each plot is drawn when its inputs exist, else skipped."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tts.config import repo_path  # noqa: E402

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _per_utt(split: str) -> dict[str, list[dict]]:
    out = {}
    root = repo_path("results")
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        p = d / split / "per_utt.csv"
        if d.is_dir() and p.exists():
            with open(p) as f:
                out[d.name] = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
    return out


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def plot_mfa_error(cfg, out: Path):
    p = repo_path("results") / "mfa_l2_dev.json"
    if not p.exists():
        return
    s = json.loads(p.read_text())
    accs = [a for a in sorted(s) if a != "all"] + ["all"]
    x = np.arange(len(accs))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (metric, label) in enumerate([("mae_ms", "MAE (ms)"), ("within_20ms", "% within 20 ms")]):
        ax[k].bar(x - 0.2, [s[a]["word"][metric] for a in accs], 0.4, label="word", color=COLORS[0])
        ax[k].bar(x + 0.2, [s[a]["phone"][metric] for a in accs], 0.4, label="phone", color=COLORS[1])
        ax[k].set_xticks(x, accs)
        ax[k].set_ylabel(label)
        ax[k].legend()
    fig.suptitle("Plot 3: MFA vs manual boundaries on dev accents")
    fig.tight_layout()
    fig.savefig(out / "03_mfa_error_dev.png", dpi=120)
    plt.close(fig)


def plot_per_accent(per: dict, split: str, out: Path):
    if not per:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    systems = list(per)
    accs = sorted({r["accent"] for rs in per.values() for r in rs})
    w = 0.8 / (len(systems) + 1)
    x = np.arange(len(accs))
    for k, (m_out, m_ref, label) in enumerate([("wer", "wer_ref", "WER"),
                                               ("us_prob", "us_prob_ref", "US-English probability")]):
        ref = [np.nanmean([_f(r[m_ref]) for r in per[systems[0]] if r["accent"] == a]) for a in accs]
        ax[k].bar(x - 0.4 + w / 2, ref, w, label="reference", color="#999999")
        for i, s in enumerate(systems):
            v = [np.nanmean([_f(r[m_out]) for r in per[s] if r["accent"] == a]) for a in accs]
            ax[k].bar(x - 0.4 + w * (i + 1.5), v, w, label=s, color=COLORS[i % len(COLORS)])
        ax[k].set_xticks(x, accs)
        ax[k].set_ylabel(label)
        ax[k].legend(fontsize=8)
    fig.suptitle(f"Plot 6: per-accent WER and US probability ({split})")
    fig.tight_layout()
    fig.savefig(out / f"06_per_accent_{split}.png", dpi=120)
    plt.close(fig)


def plot_vs_stretch(per: dict, split: str, out: Path):
    if not per:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    bins = np.array([0.0, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 10.0])
    centers = [0.7, 0.9, 1.1, 1.3, 1.55, 1.85, 2.3]
    for i, (s, rs) in enumerate(per.items()):
        st = np.array([_f(r["stretch_ratio"]) for r in rs])
        for k, m in enumerate(["utmos", "wer", "spk_sim_lj"]):
            v = np.array([_f(r[m]) for r in rs])
            ys = [np.nanmean(v[(st >= lo) & (st < hi)]) if np.any((st >= lo) & (st < hi)) else np.nan
                  for lo, hi in zip(bins[:-1], bins[1:])]
            ax[k].plot(centers, ys, "o-", label=s, color=COLORS[i % len(COLORS)])
    for k, m in enumerate(["UTMOS", "WER", "speaker sim. to LJ"]):
        ax[k].set_xlabel("stretch ratio (output / native)")
        ax[k].set_ylabel(m)
        ax[k].legend(fontsize=8)
    fig.suptitle(f"Plot 4: quality vs stretch ratio ({split})")
    fig.tight_layout()
    fig.savefig(out / f"04_vs_stretch_{split}.png", dpi=120)
    plt.close(fig)


def plot_boundaries(per: dict, split: str, out: Path):
    if not per:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, (s, rs) in enumerate(per.items()):
        for acc in sorted({r["accent"] for r in rs}):
            v = [_f(r["wb_mae_ms"]) for r in rs if r["accent"] == acc]
            ax[0].hist(v, bins=40, range=(0, 120), histtype="step", label=f"{s}/{acc}")
        mfa = np.array([_f(r["wb_mae_ms"]) for r in rs])
        ctc = np.array([_f(r["wb_mae_ctc_ms"]) for r in rs])
        ax[1].scatter(mfa, ctc, s=6, alpha=0.5, label=s, color=COLORS[i % len(COLORS)])
    lim = [0, 150]
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[0].set_xlabel("per-utterance word-boundary MAE (ms), MFA on output")
    ax[1].set_xlabel("MAE via MFA (ms)")
    ax[1].set_ylabel("MAE via MMS_FA (ms)")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    fig.suptitle(f"Plot 2: word-boundary error ({split})")
    fig.tight_layout()
    fig.savefig(out / f"02_boundaries_{split}.png", dpi=120)
    plt.close(fig)


def plot_tuning(out: Path):
    p = repo_path("results") / "tune" / "grid.json"
    if not p.exists():
        return
    g = json.loads(p.read_text())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    caps = [r for r in g if r["transfer"] == "elastic"]
    for ns in sorted({r["noise_scale"] for r in caps}):
        rs = [r for r in caps if r["noise_scale"] == ns]
        ax[0].plot([r["wb_mae_ms"] for r in rs], [r["utmos"] for r in rs], "o-", label=f"noise {ns}")
        for r in rs:
            ax[0].annotate(str(r["cap"]), (r["wb_mae_ms"], r["utmos"]), fontsize=7)
    ax[0].set_xlabel("word-boundary MAE (ms)")
    ax[0].set_ylabel("UTMOS")
    ax[0].legend()
    ax[0].set_title("Plot 5: stretch-cap trade-off (dev)")
    tv = {}
    for r in g:
        tv.setdefault(r["transfer"], []).append(r)
    names = sorted(tv)
    for k, m in enumerate(["us_prob", "wer", "utmos"]):
        ax[1].bar(np.arange(len(names)) + 0.27 * (k - 1), [np.nanmean([_f(x[m]) for x in tv[n]])
                                                           for n in names], 0.27, label=m)
    ax[1].set_xticks(range(len(names)), names)
    ax[1].legend()
    ax[1].set_title("Plot 7: transfer variants (dev)")
    fig.tight_layout()
    fig.savefig(out / "05_07_tuning.png", dpi=120)
    plt.close(fig)


def plot_efficiency(out: Path):
    ps = sorted(repo_path("results").glob("profile_*.json"))
    if not ps:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, p in enumerate(ps):
        d = json.loads(p.read_text())
        if i == 0 and "params" in d:
            ks = list(d["params"]["modules"])
            ax[0].barh(ks, [d["params"]["modules"][k] / 1e6 for k in ks], color=COLORS[0])
            ax[0].set_xlabel("parameters (M)")
        if "rtf_vs_length" in d:
            L = [x["seconds"] for x in d["rtf_vs_length"]]
            ax[1].plot(L, [x["rtf_model"] for x in d["rtf_vs_length"]], "o-", label=p.stem)
        if "latency_ms" in d:
            stages = list(d["latency_ms"])
            ax[2].bar(i, 0)
            bottom = 0
            for k, st in enumerate(stages):
                v = d["latency_ms"][st]["p50"]
                ax[2].bar(i, v, bottom=bottom, color=COLORS[k % len(COLORS)],
                          label=st if i == 0 else None)
                bottom += v
    ax[1].set_xlabel("output length (s)")
    ax[1].set_ylabel("RTF (model only)")
    ax[1].legend(fontsize=7)
    ax[2].set_xticks(range(len(ps)), [p.stem.replace("profile_", "") for p in ps])
    ax[2].set_ylabel("p50 latency (ms)")
    ax[2].legend(fontsize=7)
    fig.suptitle("Plot 9: efficiency")
    fig.tight_layout()
    fig.savefig(out / "09_efficiency.png", dpi=120)
    plt.close(fig)


def plot_training(out: Path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return
    runs = [d for d in sorted(repo_path("export").glob("*/tb")) if d.is_dir()] if \
        repo_path("export").exists() else []
    if not runs:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, tb in enumerate(runs):
        ea = EventAccumulator(str(tb))
        ea.Reload()
        for k, tag in enumerate(["loss/mel", "loss/kl", "loss/dur"]):
            if tag in ea.Tags().get("scalars", []):
                ev = ea.Scalars(tag)
                ax[k].plot([e.step for e in ev], [e.value for e in ev], label=tb.parent.name,
                           color=COLORS[i % len(COLORS)], lw=0.8)
                ax[k].set_title(tag)
    for a in ax:
        a.legend(fontsize=7)
        a.set_xlabel("step")
    fig.suptitle("Plot 10: training curves")
    fig.tight_layout()
    fig.savefig(out / "10_training.png", dpi=120)
    plt.close(fig)


def make_plots(cfg, split: str = "dev") -> None:
    out = repo_path("results") / "plots"
    out.mkdir(parents=True, exist_ok=True)
    per = _per_utt(split)
    for fn in (lambda: plot_mfa_error(cfg, out), lambda: plot_per_accent(per, split, out),
               lambda: plot_vs_stretch(per, split, out), lambda: plot_boundaries(per, split, out),
               lambda: plot_tuning(out), lambda: plot_efficiency(out), lambda: plot_training(out)):
        try:
            fn()
        except Exception as e:  # a missing/odd input must not abort evaluation
            print(f"[plots] skipped: {type(e).__name__}: {e}")
