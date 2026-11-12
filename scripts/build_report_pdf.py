"""Build the PDF project report (D-107, D-110): docs/meeami-tts_report.pdf.

Every number in a table or figure is read from committed result files (results/, samples/), never
typed in. Every abbreviation and technical name is explained where it first appears and in the
glossary (section "Terms and abbreviations"). Run in the `report` env (envs/report.yml): `make report-pdf`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tts.config import load_config  # noqa: E402


# --------------------------------------------------------------------------- data (pure, unit-tested)
def summary(system, split="test", root=ROOT):
    return json.loads((root / "results" / system / split / "summary.json").read_text())


def per_utt(system, split="test", root=ROOT):
    with open(root / "results" / system / split / "per_utt.csv") as f:
        return list(csv.DictReader(f))


def _num(x):
    return float(x) if x not in ("", None) else None


def quantile(values, q):
    v = sorted(values)
    if not v:
        return None
    pos = q * (len(v) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def mae_values(rows):
    return [float(r["wb_mae_ms"]) for r in rows if r["status"] == "ok" and r["wb_mae_ms"] != ""]


def stretch_bins(rows_by_sys, main, edges):
    """Mean per-reference MAE of every system, binned by the MAIN system's stretch ratio (as in REPORT §11)."""
    ratio = {r["id"]: _num(r["stretch_ratio"]) for r in rows_by_sys[main]}
    out = {}
    for s, rows in rows_by_sys.items():
        sums = [[0.0, 0] for _ in range(len(edges) - 1)]
        for r in rows:
            x, m = ratio.get(r["id"]), _num(r["wb_mae_ms"])
            if x is None or m is None:
                continue
            for i in range(len(edges) - 1):
                if edges[i] <= x < edges[i + 1]:
                    sums[i][0] += m
                    sums[i][1] += 1
        out[s] = [(a / n if n else None, n) for a, n in sums]
    return out


def paired(split, system, control, group="all", root=ROOT):
    """{metric: row} for one paired-test file (results/paired/<split>_<system>_vs_<control>.csv)."""
    with open(root / "results" / "paired" / f"{split}_{system}_vs_{control}.csv") as f:
        return {r["metric"]: r for r in csv.DictReader(f) if r["group"] == group}


def read_curves(path):
    """{tag: ([steps], [values])} from results/report_data/<run>_training_curves.csv."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            s, v = out.setdefault(r["tag"], ([], []))
            s.append(int(r["step"]))
            v.append(float(r["value"]))
    return out


def moving_average(v, k):
    out, acc = [], 0.0
    for i, x in enumerate(v):
        acc += x
        if i >= k:
            acc -= v[i - k]
        out.append(acc / min(i + 1, k))
    return out


def bin_label(lo, hi):
    if lo <= 0:
        return f"< {hi:g}"
    if hi >= 10:
        return f"> {lo:g}"
    return f"{lo:g}-{hi:g}"


def fmt_p(p):
    p = float(p)
    if p >= 0.001:
        return f"{p:.3f}"
    return f"{p:.0e}".replace("e-0", "e-")


# --------------------------------------------------------------------------- figures (matplotlib)
class Figures:
    def __init__(self, cfg):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt, self.cfg = plt, cfg
        self.dir = ROOT / cfg.fig_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 8.6,
            "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
            "axes.labelcolor": "#2b2a27", "xtick.color": "#52514e", "ytick.color": "#52514e",
            "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
            "legend.frameon": False, "figure.facecolor": "white"})
        self.col, self.lab = dict(cfg.colors), dict(cfg.labels)

    def save(self, fig, name):
        p = self.dir / f"{name}.png"
        fig.savefig(p, dpi=self.cfg.fig_dpi, bbox_inches="tight")
        self.plt.close(fig)
        return p

    def pipeline(self):
        plt = self.plt
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
        def box(x, y, w, h, text, fc="#f2f1ed", bold=False):
            ax.add_patch(plt.matplotlib.patches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#8a8984", lw=0.8))
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.9,
                    weight="bold" if bold else "normal", color="#1d1c1a")
        def arrow(x0, y0, x1, y1):
            ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.9))
        blue, orange = "#dce9f8", "#fbe3d8"
        box(1, 30, 17, 7, "reference recording\n(non-US speaker)", fc=orange)
        box(24, 30, 19, 7, "forced aligner MFA\n(fallback: CTC aligner)")
        box(49, 30, 22, 7, "start and end time of\nevery word, and pauses")
        box(1, 17, 17, 7, "input text", fc=blue)
        box(24, 17, 19, 7, "text normalizer\n→ phones from the lexicon")
        box(49, 17, 22, 7, "duration predictor\n→ native phone lengths d̂ᵢ")
        box(76, 16, 23, 21.5, "fit each word into its\nreference time span:\ndᵢ = d̂ᵢ · exp(z·σᵢ)\n(one z per word)\nat most 2.5× stretch,\nthe rest becomes a pause")
        box(1, 1.5, 70, 9, "VITS-MFA model: text encoder → each phone repeated for dᵢ frames\n→ sample (noise 0.333) → normalizing flow → HiFi-GAN decoder\n→ audio of exactly the reference length")
        box(77, 2, 22, 8, "output audio\n+ word timings", fc=blue, bold=True)
        arrow(18, 33.5, 24, 33.5); arrow(43, 33.5, 49, 33.5); arrow(71, 33.5, 76, 30)
        arrow(18, 20.5, 24, 20.5); arrow(43, 20.5, 49, 20.5); arrow(71, 20.5, 76, 22)
        arrow(88, 16, 70, 9); arrow(71, 6, 77, 6)
        ax.text(24, 13, "timing comes only from the reference;\nwords, voice and rhythm inside words only from the text and the model",
                fontsize=6.6, color="#52514e", style="italic", va="center")
        return self.save(fig, "pipeline")

    def headline(self, sample_dir):
        plt, np = self.plt, __import__("numpy")
        d = json.loads((sample_dir / "timings.json").read_text())
        rows = [("reference recording (non-US speaker); word times from the human annotation", sample_dir / "reference.wav",
                 d["annotation_words"], "#8a8984")]
        for s in (self.cfg.main, "v0"):
            sd = d["systems"][s]
            rows.append((f"{self.lab[s]}: mean word-boundary error {sd['metrics']['wb_mae_ms']:.1f} ms; "
                         "word times found by MFA in the output", sample_dir / f"{s}.wav", sd["measured_words"], self.col[s]))
        fig, axes = plt.subplots(len(rows), 1, figsize=(7.2, 4.4), sharex=True)
        dur = d["ref_duration_s"]
        for ax, (title, wav, words, c) in zip(axes, rows):
            with wave.open(str(wav)) as w:
                x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float)
                sr = w.getframerate()
            x /= max(1.0, abs(x).max())
            ax.plot(np.arange(len(x)) / sr, x, color=c, lw=0.35, alpha=0.75)
            for i, wd in enumerate(words):
                ax.axvspan(wd["start"], wd["end"], ymin=0.0, ymax=0.08, color=c, alpha=0.9 if i % 2 else 0.55, lw=0)
                ax.axvline(wd["start"], color="#2b2a27", lw=0.4, alpha=0.5)
                ax.text((wd["start"] + wd["end"]) / 2, 1.08, wd["word"], ha="center", va="bottom", fontsize=6.3)
            for wd in d["annotation_words"]:
                ax.axvline(wd["start"], color="#e34948", lw=0.5, ls=":", alpha=0.8)
            ax.set_ylim(-1.05, 1.55); ax.set_yticks([]); ax.grid(False)
            ax.set_title(title, loc="left", fontsize=7.4, pad=2)
            ax.set_xlim(0, dur)
        axes[-1].set_xlabel("time (seconds); red dotted lines = word starts in the human annotation of the reference")
        fig.tight_layout(h_pad=0.6)
        return self.save(fig, "headline")

    def cdf(self, rows_by_sys):
        plt, np = self.plt, __import__("numpy")
        fig, ax = plt.subplots(figsize=(7.0, 2.9))
        for s in self.cfg.core:
            v = np.sort(mae_values(rows_by_sys[s]))
            y = np.arange(1, len(v) + 1) / len(v)
            med = quantile(v.tolist(), 0.5)
            ax.plot(v, y, color=self.col[s], lw=2 if s == self.cfg.main else 1.4,
                    label=f"{self.lab[s]}: median {med:.1f} ms")
            ax.plot([med], [0.5], "o", ms=5, color=self.col[s], mec="white", mew=1.2)
        ax.axhline(0.5, color="#8a8984", lw=0.6, ls="--")
        ax.legend(loc="lower right", fontsize=7)
        ax.set_xlim(0, 100); ax.set_ylim(0, 1.0)
        ax.set_xlabel("mean word-boundary error of one reference sentence (ms)")
        ax.set_ylabel("share of the 1,248 sentences\nwith this error or less")
        return self.save(fig, "cdf")

    def per_accent(self, summaries):
        plt, np = self.plt, __import__("numpy")
        acc, core = list(self.cfg.accents), list(self.cfg.core)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
        for ax, (metric, scale, unit) in zip(axes, [("wb_mae_ms", 1, "mean word-boundary error (ms)"),
                                                    ("wer", 100, "word error rate (%)")]):
            w = 0.19
            for j, s in enumerate(core):
                m = [summaries[s]["metrics"][a][metric] for a in acc]
                mean = np.array([x["mean"] for x in m]) * scale
                lo = mean - np.array([x["ci95"][0] for x in m]) * scale
                hi = np.array([x["ci95"][1] for x in m]) * scale - mean
                xs = np.arange(len(acc)) + (j - 1.5) * (w + 0.015)
                ax.bar(xs, mean, w, color=self.col[s], label=self.lab[s], edgecolor="white", lw=0.8)
                ax.errorbar(xs, mean, yerr=[lo, hi], fmt="none", ecolor="#2b2a27", lw=0.7, capsize=1.5)
            ax.set_xticks(range(len(acc)), [a.capitalize() for a in acc])
            ax.set_ylabel(unit); ax.grid(axis="x", visible=False)
        axes[0].legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(1.1, 1.22))
        fig.tight_layout()
        return self.save(fig, "per_accent")

    def stretch(self, bins, edges, n_main):
        plt = self.plt
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        labels = [bin_label(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        xs = list(range(len(labels)))
        for s in self.cfg.core:
            ax.plot(xs, [b[0] for b in bins[s]], "-o", color=self.col[s], lw=2 if s == self.cfg.main else 1.4,
                    ms=4.5, mec="white", mew=1, label=self.lab[s])
        ax.set_xticks(xs, [f"{l}\n({n} sentences)" for l, n in zip(labels, n_main)], fontsize=7)
        ax.set_xlim(-0.3, len(xs) - 0.7)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
        ax.set_xlabel("stretch ratio: how long the reference makes the words, relative to LJ's natural length (1.0 = same)")
        ax.set_ylabel("mean word-boundary\nerror (ms)")
        return self.save(fig, "stretch")

    def tuning(self, grid, choice_name):
        plt = self.plt
        fig, ax = plt.subplots(figsize=(7.0, 2.9))
        shades = {0.333: "#2a78d6", 0.5: "#7fa9e3", 0.667: "#c3d6f2"}
        names = {"proportional": "proportional scaling", "phone_copy": "copy the reference's phone lengths"}
        for g in grid:
            if g["transfer"] == "elastic" and g["aligner"] == "mfa":
                ax.plot(g["wb_mae_ms"], g["utmos"], "o", ms=6, color=shades[g["noise_scale"]], mec="#2b2a27", mew=0.4)
                ax.text(g["wb_mae_ms"] + 0.12, g["utmos"] + 0.004, f"{g['cap'] or 'none'}", fontsize=5.6, color="#52514e")
            else:
                lab = names.get(g["transfer"], g["transfer"]) if g["aligner"] == "mfa" else "CTC aligner instead of MFA"
                ax.plot(g["wb_mae_ms"], g["utmos"], "s", ms=6, color="#b5b4ad", mec="#2b2a27", mew=0.4)
                ax.text(g["wb_mae_ms"] + 0.3, g["utmos"], f"{lab} (WER {100 * g['wer']:.1f}%)", fontsize=6.3, va="center")
            if g["name"] == choice_name:
                ax.plot(g["wb_mae_ms"], g["utmos"], "o", ms=13, mfc="none", mec="#e34948", mew=1.4)
                ax.annotate("chosen: elastic fit, cap 2.5, noise 0.333", (g["wb_mae_ms"], g["utmos"]),
                            (g["wb_mae_ms"] + 7, g["utmos"] - 0.035), fontsize=7, color="#e34948",
                            arrowprops=dict(arrowstyle="-", color="#e34948", lw=0.7))
        for ns, c in shades.items():
            ax.plot([], [], "o", color=c, mec="#2b2a27", mew=0.4, label=f"elastic fit, noise scale {ns} (number = stretch cap)")
        ax.plot([], [], "s", color="#b5b4ad", mec="#2b2a27", mew=0.4, label="other ways to transfer the timing")
        ax.legend(fontsize=6.5, loc="center right")
        ax.set_xlabel("mean word-boundary error (ms), 120 development references")
        ax.set_ylabel("UTMOS (predicted\nnaturalness, 1-5)")
        return self.save(fig, "tuning")

    def forest(self, comps):
        """comps: list of (label, {metric: row}). Δ = system − control; filled marker = Holm p < alpha."""
        plt = self.plt
        metrics = [("wb_mae_ms", 1, "word-boundary\nerror (ms)\nlower = better"),
                   ("wer", 100, "word error rate\n(% points)\nlower = better"),
                   ("utmos", 1, "UTMOS\nnaturalness\nhigher = better"),
                   ("us_prob", 1, "US-accent\nprobability\nhigher = better")]
        fig, axes = plt.subplots(1, 4, figsize=(7.3, 0.36 * len(comps) + 1.1), sharey=True)
        ys = list(range(len(comps)))[::-1]
        for ax, (m, sc, title) in zip(axes, metrics):
            for y, (lab, rows) in zip(ys, comps):
                r = rows[m]
                d = float(r["mean_diff"]) * sc
                sig = float(r["p_holm"]) < self.cfg.alpha
                c = "#2a78d6" if r["better"] == "True" else "#eb6834"
                ax.plot([0, d], [y, y], color=c, lw=1.2, alpha=0.6)
                ax.plot(d, y, "o", ms=5.5, color=c if sig else "white", mec=c, mew=1.3)
            ax.axvline(0, color="#2b2a27", lw=0.7)
            ax.set_title(title, fontsize=6.9); ax.grid(axis="y", visible=False)
            ax.tick_params(axis="x", labelsize=6.5)
        axes[0].set_yticks(ys, [c[0] for c in comps], fontsize=6.8)
        fig.text(0.01, -0.03, "filled dot = statistically significant (p < 0.05 after Holm correction); hollow = not "
                 "significant;\nblue = change in the better direction, orange = worse; 0 = no change", fontsize=6.6, color="#52514e")
        fig.suptitle("change from the control (system minus control)", fontsize=7.6, y=1.0)
        fig.tight_layout(w_pad=0.6)
        return self.save(fig, "forest")

    def fidelity(self, fid):
        plt, np = self.plt, __import__("numpy")
        core = list(self.cfg.core)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
        xs = np.arange(len(core))
        for k, key in enumerate(["bias_start_ms", "bias_end_ms"]):
            vals = [fid[s]["mfa"][key] for s in core]
            axes[0].bar(xs + (k - 0.5) * 0.36, vals, 0.34, color=[self.col[s] for s in core],
                        alpha=1.0 if k == 0 else 0.55, edgecolor="white")
            for x, v in zip(xs + (k - 0.5) * 0.36, vals):
                axes[0].text(x, v + (0.6 if v >= 0 else -0.6), f"{v:.1f}", ha="center",
                             va="bottom" if v >= 0 else "top", fontsize=5.8)
        axes[0].axhline(0, color="#2b2a27", lw=0.7); axes[0].set_ylim(-26, 4)
        axes[0].set_title("average offset of the words (ms); negative = early\ndark = word starts, light = word ends", fontsize=7.2)
        vals = [fid[s]["mfa"]["mae_unbiased_ms"] for s in core]
        axes[1].bar(xs, vals, 0.6, color=[self.col[s] for s in core], edgecolor="white")
        for x, v in zip(xs, vals):
            axes[1].text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=6.4)
        axes[1].set_title("scatter: average distance from the planned time\nafter removing that offset (ms)", fontsize=7.2)
        for ax in axes:
            ax.set_xticks(xs, [self.lab[s].replace(" (", "\n(") for s in core], fontsize=6.3)
            ax.grid(axis="x", visible=False)
        fig.tight_layout()
        return self.save(fig, "fidelity")

    def phone_dynamics(self, pd, gt):
        plt = self.plt
        order = [s for s in ["m1", "m1_cont", "m1_pos", "m1_pos_diph", "abl_a_pos", "abl_b_pos"] if s in pd]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
        r = [pd[s]["inside"] / gt["inside"] for s in order]
        o = [pd[s]["onset_over_inside"] for s in order]
        for ax, vals, ref, title in [(axes[0], r, 1.0, "sound change inside phones\n(LJ recordings = 1.00)"),
                                     (axes[1], o, gt["onset_over_inside"], "sound change at the start of a phone\n÷ sound change inside it")]:
            ax.barh(range(len(order)), vals, 0.6, color=[self.col.get(s, "#8a8984") for s in order], edgecolor="white")
            ax.axvline(ref, color="#e34948", lw=1, ls="--")
            ax.text(ref, -0.75, f" LJ recordings = {ref:.2f}", color="#e34948", fontsize=6.4)
            for y, v in enumerate(vals):
                ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=6.2)
            ax.set_yticks(range(len(order)), order, fontsize=6.6)
            ax.set_xlim(0, max(max(vals), ref) * 1.18); ax.set_title(title, fontsize=7.2)
            ax.grid(axis="y", visible=False); ax.set_ylim(len(order) - 0.5, -1.1)
        fig.tight_layout()
        return self.save(fig, "phone_dynamics")

    def training(self, curves):
        plt = self.plt
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.0), sharex=True)
        panels = [("val/mel_l1", "validation: spectrogram error (L1, lower is better)", 1, "#2a78d6"),
                  ("val/dur", "validation: duration error (log frames, lower is better)", 1, "#6250d6"),
                  ("loss/d", "training: discriminator loss (stays level = balanced)", 25, "#eb6834"),
                  ("loss/fm", "training: feature-matching loss", 25, "#1baf7a")]
        for ax, (tag, title, k, c) in zip(axes.flat, panels):
            s, v = curves[tag]
            if k > 1:
                ax.plot(s, v, color=c, lw=0.4, alpha=0.25)
                ax.plot(s, moving_average(v, k), color=c, lw=1.4)
            else:
                ax.plot(s, v, "-o", color=c, lw=1.2, ms=2.5)
            ax.set_title(title, fontsize=7.4, loc="left")
            ax.ticklabel_format(axis="x", style="plain")
        for ax in axes[1]:
            ax.set_xlabel("training step (batches of 32 clips)")
            ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
        fig.tight_layout()
        return self.save(fig, "training")

    def efficiency(self, profiles):
        plt, np = self.plt, __import__("numpy")
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.35, 1]})
        stages = [("normalize_g2p", "text to phones", "#8a8984"), ("text_encoder_durations", "text encoder + durations", "#2a78d6"),
                  ("flow_decoder", "flow + waveform decoder", "#1baf7a"), ("mfa_align_one", "aligning the reference (MFA)", "#eb6834")]
        names = list(profiles)
        for i, n in enumerate(names):
            left = 0.0
            for key, lab, c in stages:
                v = profiles[n]["latency_ms"][key]["p50"] / 1000
                axes[0].barh(i, v, 0.55, left=left, color=c, edgecolor="white", lw=0.6, label=lab if i == 0 else None)
                left += v
            axes[0].text(left + 0.05, i, f"{left:.2f} s", va="center", fontsize=6.6)
        axes[0].set_yticks(range(len(names)), names, fontsize=6.8); axes[0].invert_yaxis()
        axes[0].set_xlabel("time for one 3.7 s sentence (seconds, median)")
        axes[0].legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
        axes[0].grid(axis="y", visible=False)
        cols = ["#2a78d6", "#6250d6", "#eb6834"]
        for c, n in zip(cols, names):
            r = profiles[n]["rtf_vs_length"]
            axes[1].plot([x["seconds"] for x in r], [x["rtf_model"] for x in r], "-o", color=c, ms=3.5, label=n)
        axes[1].set_yscale("log"); axes[1].set_xlabel("length of the output (seconds)")
        axes[1].set_ylabel("real-time factor\n(model only)")
        axes[1].axhline(1, color="#8a8984", lw=0.6, ls="--")
        axes[1].legend(fontsize=6.2, loc="lower right", bbox_to_anchor=(1.0, 0.05))
        fig.tight_layout()
        return self.save(fig, "efficiency")

    def pace(self, rows):
        plt, np = self.plt, __import__("numpy")
        fig, ax = plt.subplots(figsize=(6.0, 2.4))
        bins = np.linspace(-0.45, 0.75, 41)
        for sp, c in (("dev", "#2a78d6"), ("test", "#eb6834")):
            v = np.array([float(r["utterance_mean_log_rho"]) for r in rows if r["split"] == sp])
            med = float(np.exp(np.median(v)))
            ax.hist(v, bins, histtype="step", lw=1.6, color=c, density=True,
                    label=f"{'development' if sp == 'dev' else 'test'} accents ({len(v):,} sentences): median {med:.2f}×")
            ax.axvline(np.median(v), color=c, lw=0.8, ls=":")
        ticks = [0.7, 0.8, 1.0, 1.25, 1.5, 2.0]
        ax.set_xticks([math.log(t) for t in ticks], [f"{t:g}×" for t in ticks])
        ax.axvline(0, color="#2b2a27", lw=0.7)
        ax.set_xlabel("pace of a reference sentence relative to LJ (above 1× = slower than LJ)")
        ax.set_ylabel("share of sentences\n(density)")
        ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
        return self.save(fig, "pace")


# --------------------------------------------------------------------------- glossary
GLOSSARY = [
    ("Systems compared", [
        ("V0", "The original pretrained VITS model for LJSpeech, used exactly as released (it learns its timing with MAS). "
               "The immediate baseline: it is the model that M1 starts from."),
        ("M0", "A pretrained FastSpeech2 model with a HiFi-GAN vocoder. The second baseline."),
        ("M1 (VITS-MFA)", "VITS fine-tuned with phone durations from MFA instead of MAS, so that durations can be set from outside."),
        ("m1_cont", "M1 trained 20,000 more steps with no change. The control for \"more training\"."),
        ("m1_pos", "M1 plus a position-in-phone embedding (see below)."),
        ("m1_pos_diph (main)", "m1_pos plus two-token diphthongs. The main system of this project."),
        ("abl_a_pos, abl_b_pos", "m1_pos trained with pace augmentation A or B: the two ablations."),
        ("ablation", "An experiment that adds or removes one component to measure what it contributes."),
        ("control", "The system that differs from the tested one in exactly one setting, so any difference is caused by that setting."),
    ]),
    ("Data and speech", [
        ("LJSpeech (LJ)", "A public dataset of 13,100 clips (about 24 h) read by one US female speaker. The target voice; the only training data."),
        ("L1, L2, L2-ARCTIC", "L1 = a speaker's native language; an L2 speaker speaks English as a second language. L2-ARCTIC is a public corpus "
                              "of 24 L2 speakers (6 native languages, 4 speakers each) reading the same sentences, with word and phone "
                              "boundaries corrected by hand for about 150 sentences per speaker."),
        ("reference", "The non-US recording whose timing the output must follow."),
        ("development (dev) / test accents", "Dev accents (Hindi, Mandarin, Spanish) were used for every decision. Test accents (Arabic, Korean, "
                                             "Vietnamese) were kept locked and evaluated once at the end, so the results show performance on unseen accents."),
        ("phone", "The smallest sound unit of a word; \"cry\" has three: K, R, AY."),
        ("ARPAbet", "The phone alphabet of CMUdict. A digit marks stress: 0 none, 1 primary, 2 secondary (AY1 = stressed \"eye\" vowel, AH0 = the unstressed \"uh\")."),
        ("diphthong", "A vowel that glides from one position to another: AY (\"cry\"), AW (\"how\"), EY (\"may\"), OW (\"go\"), OY (\"boy\")."),
        ("CMUdict", "The Carnegie Mellon University Pronouncing Dictionary of US English (about 134,000 words)."),
        ("lexicon", "This project's word → phones file (data/lexicon/lexicon.txt), the only source of pronunciations."),
        ("G2P, g2p_en", "Grapheme-to-phoneme: predicting the phones of a word from its spelling. g2p_en is a neural G2P model for English."),
        ("OOV", "Out of vocabulary: a word that is not in CMUdict."),
        ("IPA, espeak", "International Phonetic Alphabet. espeak is an open-source speech synthesizer whose IPA output the original VITS was trained on."),
        ("frame", "The time step of the model: 256 audio samples at 22,050 samples per second, i.e. 11.6 ms. Durations are counted in frames."),
        ("sil, sp, spn, pad", "Special tokens: silence at the start and end of a sentence; a pause between words; a word the aligner could not recognize; padding."),
        ("mel spectrogram", "A picture of audio over time and frequency, with frequencies on the mel scale (spaced like human hearing). Log-mel = its logarithm."),
        ("formants (F1, F2)", "Resonance frequencies of the mouth and throat. F1 and F2 identify a vowel, and their movement shows a diphthong's glide. "
                              "Not related to the F1 score of pauses."),
        ("LPC", "Linear predictive coding, a standard way to measure formants from audio."),
    ]),
    ("Models and methods", [
        ("TTS", "Text-to-speech."),
        ("VITS", "\"Variational Inference with adversarial learning for end-to-end Text-to-Speech\" (Kim et al., 2021): a TTS model that turns "
                 "phones directly into a waveform. Its parts: a text encoder (phones → hidden states), a prior (what each frame should look "
                 "like, predicted from the text), a posterior encoder (encodes real audio; used only in training), a normalizing flow (an "
                 "invertible network connecting the two) and a HiFi-GAN decoder (produces the waveform)."),
        ("MAS", "Monotonic alignment search: VITS's built-in method that, during training, decides which audio frames belong to which input token."),
        ("forced alignment, MFA", "Finding where each word and phone starts and ends in a recording, given its transcript. MFA = Montreal Forced "
                                  "Aligner (version 3.4.3, pretrained english_us_arpa model). Inside, it uses an HMM-GMM: a hidden Markov model with "
                                  "Gaussian-mixture sound models, the classic speech-recognition design."),
        ("CTC aligner, MMS_FA", "A second aligner built on Meta's MMS (Massively Multilingual Speech) recognition model, trained with CTC (connectionist temporal "
                                "classification). Used as a fallback when MFA fails, and as an independent second judge."),
        ("FastSpeech2", "A widely used TTS model that predicts each phone's duration, pitch and energy explicitly (in its \"variance adaptor\"); "
                        "a given duration can be passed in directly (d_targets)."),
        ("HiFi-GAN, vocoder", "A vocoder turns a spectrogram (or hidden features) into a waveform. HiFi-GAN is a fast neural vocoder trained as a GAN."),
        ("fine-tuning, warm start", "Continuing to train an already trained model on a changed task, starting from its weights."),
        ("duration predictor", "The part of the model that predicts how many frames each phone lasts."),
        ("d̂ᵢ, dᵢ, D, σᵢ, z", "d̂ᵢ: the model's natural (native) length of phone i. dᵢ: the length actually used. D: the word's length in the "
                             "reference, in frames. σᵢ: how much LJ naturally varies phone i (standard deviation of its log duration). z: one "
                             "number per word saying how much slower (z > 0) or faster (z < 0) than LJ, in units of σ."),
        ("elastic fit", "Filling a word's reference time span by scaling each phone by exp(z·σᵢ): phones LJ varies a lot stretch more, "
                        "stable phones (like stops) barely change."),
        ("stretch cap", "The most a word is stretched: 2.5× its natural length. Any extra time becomes a pause after the word."),
        ("stretch ratio", "Output speech length ÷ the model's natural length for the same words. 1.0 = LJ's own pace."),
        ("noise scale", "How much randomness is sampled from the VITS prior when generating (0.333 here; larger = more varied but less clear)."),
        ("position-in-phone embedding", "A learned vector added to every frame that tells it where it is inside its phone (16 steps from start to end)."),
        ("two-token diphthong", "Each diphthong is split into two model tokens (e.g. AY1 → AY1 + AY1+), so its start and end can sound different."),
        ("pace augmentation", "Training on extra copies of LJ clips whose words were made slower or faster. A: random paces. B: paces sampled "
                              "from the dev-accent speakers' statistics."),
        ("WSOLA, Rubber Band R3", "Ways to change audio speed without changing pitch. Rubber Band R3 (a phase vocoder) was planned; WSOLA "
                                  "(waveform-similarity overlap-add) was used because R3 damaged the audio."),
        ("ρ (rho)", "Pace of a reference: each word's length ÷ LJ's typical length for the same phones. 1.12 = 12% slower than LJ."),
        ("losses", "L_mel: spectrogram error. L_KL: KL divergence, which pulls the prior toward the encoded real audio. L_dur: duration error. "
                   "L_adv and L_fm: adversarial and feature-matching losses of the GAN."),
        ("GAN, discriminator", "Generative adversarial network: a second network (the discriminator) learns to tell real audio from generated "
                               "audio, and the generator learns to fool it."),
        ("bf16, fp32", "16-bit (bfloat16) and 32-bit floating-point number formats; bf16 is faster, fp32 more precise."),
        ("U(a, b), N(μ, s²), AR(1)", "A uniform random number between a and b; a normal (Gaussian) random number with mean μ and variance s²; "
                                     "a sequence where each value is correlated with the previous one."),
        ("typed-text pre-pass", "Rules that rewrite typed input (money, dates, URLs, letters…) into plain words before the normalizer."),
    ]),
    ("Metrics", [
        ("word-boundary error (MAE)", "Mean absolute error: for every word, how far (in ms) its start and its end in the output are from the "
                                      "human-annotated start and end in the reference, averaged. Lower is better."),
        ("within 20 ms / 50 ms", "The share of word starts and ends that are at most 20 (or 50) ms from the annotation."),
        ("length error", "Output length − reference length, in samples. Must be exactly 0."),
        ("pause F1", "How well the output's pauses (silences of 100 ms or more) match the reference's: the F1 score, which combines "
                     "precision (output pauses that are real) and recall (real pauses that the output has). 1 = perfect."),
        ("WER, CER", "Word (character) error rate: the share of words (characters) that an automatic speech recognizer gets wrong when "
                     "it transcribes the audio. Measures whether the words are said clearly. Lower is better."),
        ("ASR, Whisper", "Automatic speech recognition. Whisper large-v3 is OpenAI's ASR model (run through the faster-whisper implementation)."),
        ("US-accent probability", "The probability that a speech sample has a US accent, from an accent classifier (CommonAccent ECAPA, an "
                                  "ECAPA-TDNN, a time-delay neural network, trained to recognize English accents). 0 to 1."),
        ("speaker similarity", "Cosine similarity (−1 to 1) between voice embeddings from an ECAPA speaker model: to LJ (should be high) "
                               "and to the reference speaker (should be low)."),
        ("UTMOS", "An automatic predictor of how natural speech sounds, on the 1-5 MOS scale (MOS = mean opinion score, the average "
                  "rating listeners would give). LJ's own recordings score 4.36."),
        ("MUSHRA", "A standard listening-test format in which listeners rate several versions of the same sentence side by side."),
        ("bias, scatter (fidelity)", "Comparing where a word ends up in the output with where the pipeline placed it: bias = the average "
                                     "offset (negative = early); scatter = the remaining average distance once that offset is removed."),
    ]),
    ("Statistics", [
        ("95% CI", "Confidence interval: the range that contains the true average with 95% probability. Computed by bootstrap: re-drawing "
                   "the sentences at random 1,000 times and recomputing the average."),
        ("paired comparison", "Two systems speak the same sentences, so the difference is computed sentence by sentence, which removes "
                              "the variation between sentences."),
        ("p-value, p", "The probability of seeing a difference at least this large if the two systems were really equal. Small p = the "
                       "difference is unlikely to be chance. 1e-8 means 0.00000001."),
        ("Wilcoxon signed-rank test", "A standard test of whether paired differences are centred on zero, without assuming they follow a normal distribution."),
        ("Holm correction", "Each comparison tests 8 metrics, and with many tests some will look significant by pure chance. Holm's method "
                            "raises the p-values to correct for this. \"p (Holm)\" is the corrected value; a result counts as significant "
                            "when it is below 0.05."),
        ("significant, n.s.", "Significant: corrected p below 0.05. n.s.: not significant."),
        ("median, 90th percentile, tercile", "Median: half the values are below it. 90th percentile: 90% are below it. Terciles: three equal-sized groups."),
        ("Spearman correlation", "A measure (−1 to 1) of whether two quantities rise and fall together; near 0 = unrelated."),
    ]),
    ("Efficiency and project terms", [
        ("parameters (M)", "The number of learned weights in a network, in millions."),
        ("MACs, GMACs", "Multiply-accumulate operations, the basic unit of neural-network computation; GMACs = billions of them."),
        ("RTF", "Real-time factor: computing time ÷ length of the audio produced. Below 1 = faster than real time."),
        ("p50 latency", "The median time over repeated runs (p90: the time 90% of runs stay under)."),
        ("GPU, CPU, threads, CUDA", "Graphics processor (RTX 4060 laptop GPU; L40S = the rented data-center GPU used for training) and main "
                                    "processor; threads = CPU cores used; CUDA = NVIDIA's GPU computing platform."),
        ("git tag freeze-v1", "A label on the code version (commit 5889ecf) at which the models and experiment settings were fixed. "
                              "The test accents could only be read after it existed."),
        ("D-n, E-n", "Numbered entries of the project's decision log (D) and error log (E) in docs/DECISIONS.md."),
        ("HF Hub, sha256", "The Hugging Face Hub, where the trained models and all test outputs are hosted; sha256 = a fingerprint of a file used to check it is unchanged."),
        ("make, conda env", "make runs the project's named commands (e.g. make test); a conda env is an isolated, version-pinned set of software packages."),
    ]),
]


# --------------------------------------------------------------------------- PDF (reportlab)
def build_pdf(cfg, out):
    from matplotlib import get_data_path
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.fonts import addMapping
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate,
                                    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
    from reportlab.platypus.tableofcontents import TableOfContents

    # fonts: the DejaVu family shipped with matplotlib (covers →, ≥, σ, ×, −, ɪ ...), no system fonts needed
    ttf = Path(get_data_path()) / "fonts" / "ttf"
    for name, f in [("Serif", "DejaVuSerif"), ("Serif-B", "DejaVuSerif-Bold"), ("Serif-I", "DejaVuSerif-Italic"),
                    ("Serif-BI", "DejaVuSerif-BoldItalic"), ("Sans", "DejaVuSans"), ("Sans-B", "DejaVuSans-Bold"),
                    ("Sans-I", "DejaVuSans-Oblique"), ("Sans-BI", "DejaVuSans-BoldOblique"), ("Mono", "DejaVuSansMono")]:
        pdfmetrics.registerFont(TTFont(name, str(ttf / f"{f}.ttf")))
    for fam in ("Serif", "Sans"):
        addMapping(fam, 0, 0, fam); addMapping(fam, 1, 0, f"{fam}-B")
        addMapping(fam, 0, 1, f"{fam}-I"); addMapping(fam, 1, 1, f"{fam}-BI")

    INK, MUTED, ACCENT, RULE, TINT = (colors.HexColor(h) for h in ("#1d1c1a", "#52514e", "#2a78d6", "#d6d5cf", "#f2f1ed"))
    body = ParagraphStyle("body", fontName="Serif", fontSize=9.4, leading=13.2, textColor=INK, spaceAfter=5)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.6, textColor=MUTED)
    cap = ParagraphStyle("cap", parent=body, fontName="Sans", fontSize=7.8, leading=10.4, textColor=MUTED, spaceBefore=2, spaceAfter=9)
    h1 = ParagraphStyle("h1", fontName="Sans-B", fontSize=15, leading=19, textColor=INK, spaceBefore=10, spaceAfter=8, keepWithNext=1)
    h2 = ParagraphStyle("h2", fontName="Sans-B", fontSize=11.2, leading=14.5, textColor=ACCENT, spaceBefore=9, spaceAfter=4, keepWithNext=1)
    h3 = ParagraphStyle("h3", fontName="Sans-B", fontSize=9.4, leading=12.5, textColor=INK, spaceBefore=6, spaceAfter=2, keepWithNext=1)
    cell = ParagraphStyle("cell", fontName="Sans", fontSize=7.4, leading=9.4, textColor=INK)
    cellb = ParagraphStyle("cellb", parent=cell, fontName="Sans-B")
    eq = ParagraphStyle("eq", parent=body, fontName="Serif-I", alignment=TA_CENTER, spaceBefore=2, spaceAfter=7)
    W = A4[0] - 4.2 * cm

    class Doc(BaseDocTemplate):
        def afterFlowable(self, f):
            if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
                lvl = 0 if f.style.name == "h1" else 1
                key = f"h{id(f)}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(f.getPlainText(), key, level=lvl, closed=lvl > 0)
                self.notify("TOCEntry", (lvl, f.getPlainText(), self.page, key))

    def deco(canv, doc):
        canv.saveState()
        canv.setFont("Sans", 7); canv.setFillColor(MUTED)       # page number only, no running header
        canv.drawCentredString(A4[0] / 2, 1.1 * cm, str(doc.page))
        canv.restoreState()

    doc = Doc(str(out), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.9 * cm,
              bottomMargin=1.8 * cm, title="meeami-tts: Project Report", author="Chaganti Raja Ravi Teja",
              subject="Time-aligned US-English text-to-speech on the word timing of a non-US reference")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate("cover", [frame]), PageTemplate("main", [frame], onPage=deco)])

    S = []
    P = lambda t, st=body: S.append(Paragraph(t, st))  # noqa: E731
    def B(items):
        for t in items:
            S.append(Paragraph(t, bullet, bulletText="•"))
        S.append(Spacer(1, 3))
    def T(rows, widths, bold_rows=(), zebra=True, header=True, group_rows=()):
        data = [[Paragraph(str(c), cellb if (header and i == 0) or i in bold_rows or i in group_rows else cell) for c in r]
                for i, r in enumerate(rows)]
        t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if header else 0, hAlign="LEFT")
        st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.2),
              ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6), ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE)]
        if header:
            st += [("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK)]
        if zebra:
            st += [("BACKGROUND", (0, i), (-1, i), TINT) for i in range(2, len(rows), 2)]
        for i in bold_rows:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#e3edf9")))
        for i in group_rows:
            st += [("SPAN", (0, i), (-1, i)), ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#e3edf9")),
                   ("TOPPADDING", (0, i), (-1, i), 4)]
        t.setStyle(TableStyle(st))
        S.append(t); S.append(Spacer(1, 7))
    def F(path, caption, width=1.0):
        from PIL import Image as PImage
        w, h = PImage.open(path).size
        img = Image(str(path), width=width * W, height=width * W * h / w)
        S.append(KeepTogether([img, Paragraph(caption, cap)]))
    fig_no, tab_no = [0], [0]
    def fig_caption(text):
        fig_no[0] += 1
        return f"<b>Figure {fig_no[0]}.</b> {text}"
    def tab_caption(text):
        tab_no[0] += 1
        P(f"<b>Table {tab_no[0]}.</b> {text}", cap)

    # ---------------- data
    figs = Figures(cfg)
    systems, core, main = list(cfg.systems), list(cfg.core), cfg.main
    sums = {s: summary(s) for s in systems}
    rows = {s: per_utt(s) for s in systems}
    edges = list(cfg.stretch_bins)
    bins = stretch_bins({s: rows[s] for s in core}, main, edges)
    fid = json.loads((ROOT / "results/fidelity/test.json").read_text())
    calib = json.loads((ROOT / "results/fidelity/calibration_dev.json").read_text())
    pdyn = {p.stem: json.loads(p.read_text()) for p in (ROOT / "results/phone_dynamics").glob("*.json")}
    gt = pdyn.pop("ground_truth")
    grid = json.loads((ROOT / "results/tune/grid.json").read_text())
    choice = json.loads((ROOT / "results/tune/choice.json").read_text())
    mfa = json.loads((ROOT / cfg.data.mfa_error).read_text())
    curves = read_curves(ROOT / cfg.data.training)
    with open(ROOT / cfg.data.pace) as f:
        pace_rows = list(csv.DictReader(f))
    profiles = {k: json.loads((ROOT / v).read_text()) for k, v in cfg.data.profiles.items()}
    short = lambda s: cfg.labels[s].split(" (")[0]  # noqa: E731
    comps = [(f"{'dev' if sp == 'dev' else 'test'}: {short(a)} vs {short(b)}", paired(sp, a, b)) for sp, a, b in cfg.paired]
    M = lambda s, k, g="all": sums[s]["metrics"][g][k]["mean"]  # noqa: E731
    lab = dict(cfg.labels)
    n_main = [b[1] for b in bins[main]]
    lj = sums[main]["lj_test_ground_truth"]
    gpu = profiles[next(iter(profiles))]

    f_pipe, f_head = figs.pipeline(), figs.headline(ROOT / cfg.headline_sample)
    f_cdf, f_acc = figs.cdf(rows), figs.per_accent(sums)
    f_str, f_tun = figs.stretch(bins, edges, n_main), figs.tuning(grid, choice["choice"]["name"])
    f_for, f_fid, f_pd = figs.forest(comps), figs.fidelity(fid), figs.phone_dynamics(pdyn, gt)
    f_trn, f_eff, f_pace = figs.training(curves), figs.efficiency(profiles), figs.pace(pace_rows)
    head = json.loads((ROOT / cfg.headline_sample / "timings.json").read_text())

    # ---------------- cover
    S.append(Spacer(1, 2.2 * cm))
    S.append(Paragraph("meeami-tts", ParagraphStyle("t0", fontName="Sans-B", fontSize=30, leading=36, textColor=ACCENT)))
    S.append(Paragraph("Time-aligned US-English text-to-speech<br/>on the word timing of a non-US reference",
                       ParagraphStyle("t1", fontName="Sans", fontSize=16, leading=21, textColor=INK, spaceBefore=6)))
    S.append(Spacer(1, 0.5 * cm))
    S.append(Paragraph("Project report · Chaganti Raja Ravi Teja · November 2026",
                       ParagraphStyle("t2", fontName="Sans", fontSize=10, leading=14, textColor=MUTED)))
    S.append(Paragraph(f"Code: <a href='{cfg.links.github}' color='#2a78d6'>{cfg.links.github}</a><br/>"
                       f"Trained models and all test outputs: <a href='{cfg.links.hf}' color='#2a78d6'>{cfg.links.hf}</a>",
                       ParagraphStyle("t3", fontName="Sans", fontSize=8.5, leading=12, textColor=MUTED, spaceBefore=4)))
    S.append(Spacer(1, 0.8 * cm))
    tiles = [(f"{M(main, 'wb_mae_ms'):.1f} ms", f"average word-timing error\n(original VITS: {M('v0', 'wb_mae_ms'):.1f} ms)"),
             ("0", "samples of length difference\non all 1,248 test sentences"),
             (f"{100 * M(main, 'wer'):.1f}%", f"words misrecognized by ASR\n(in the non-US recordings: {100 * M(main, 'wer_ref'):.1f}%)"),
             (f"{M(main, 'utmos'):.2f} / 5", f"predicted naturalness (UTMOS)\n(LJSpeech recordings: {lj['utmos']:.2f})")]
    tstyle = ParagraphStyle("tv", fontName="Sans-B", fontSize=17, leading=21, textColor=ACCENT, alignment=TA_CENTER)
    tlab = ParagraphStyle("tl", fontName="Sans", fontSize=7.4, leading=9.6, textColor=MUTED, alignment=TA_CENTER)
    tt = Table([[Paragraph(v, tstyle) for v, _ in tiles], [Paragraph(l.replace("\n", "<br/>"), tlab) for _, l in tiles]],
               colWidths=[W / 4] * 4)
    tt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LINEBEFORE", (1, 0), (-1, -1), 2, colors.white),
                            ("TOPPADDING", (0, 0), (-1, 0), 10), ("BOTTOMPADDING", (0, -1), (-1, -1), 10)]))
    S.append(tt)
    S.append(Spacer(1, 0.4 * cm))
    S.append(Paragraph("Results of the main system on accents never used during development (Arabic, Korean, Vietnamese; "
                       "12 speakers × 104 sentences), evaluated once after the code was frozen. ASR = automatic speech "
                       "recognition. All terms are explained in the glossary after the contents.", cap))
    S.append(Spacer(1, 0.4 * cm))
    S.append(Paragraph("Abstract", h3))
    P("Given an English text and a recording of it by a non-US speaker, the system speaks the text in the voice and US accent "
      "of LJSpeech (a public dataset of one US female speaker) while following the recording's timing: the output is exactly as "
      "long as the recording, and every word starts and ends where the speaker said it, pauses included. The recording "
      "contributes timing only. "
      "The approach makes word durations an <i>input</i> to the model. VITS, a text-to-speech model that produces audio "
      "directly from phones, normally learns its own timing with a method called monotonic alignment search (MAS). We "
      "fine-tune it with phone durations measured by the Montreal Forced Aligner (MFA) instead, so durations can be set from "
      "outside. At synthesis, MFA finds the word boundaries in the recording, and each word's time span is filled with the "
      "model's own natural rhythm, stretched the way the LJSpeech speaker herself varies. "
      f"On {sums[main]['n']:,} test sentences from three unseen accents, words land on average {M(main, 'wb_mae_ms'):.1f} ms "
      f"from where a human annotator marked them in the recording (the aligner's own error on accented speech is 19.4 ms), "
      f"against {M('v0', 'wb_mae_ms'):.1f} ms for the original VITS given the same timing and {M('m0', 'wb_mae_ms'):.1f} ms "
      "for FastSpeech2, another widely used model. Two defects heard in listening were traced to their cause and fixed with "
      "minimal, controlled changes; two data-augmentation experiments did not help and are reported as negative results. "
      "Every choice was made on other accents before the test accents were opened.")
    S.append(NextPageTemplate("main")); S.append(PageBreak())

    # ---------------- contents + glossary
    S.append(Paragraph("Contents", ParagraphStyle("ct", parent=h1)))
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("toc0", fontName="Sans-B", fontSize=9.5, leading=14, leftIndent=0),
                       ParagraphStyle("toc1", fontName="Sans", fontSize=8.5, leading=11.5, leftIndent=14, textColor=MUTED)]
    S.append(toc)
    S.append(Spacer(1, 10))
    P("<b>How to read this report.</b> Every table and figure is generated from the result files in the repository by "
      "<font face='Mono'>scripts/build_report_pdf.py</font> (command: <font face='Mono'>make report-pdf</font>). "
      "References such as D-85 or E-35 point to numbered entries of the project's decision and error log, "
      "<font face='Mono'>docs/DECISIONS.md</font>. <font face='Mono'>docs/REPORT.md</font> is a longer version of this report "
      "that links every claim to the exact line of code behind it. The audio behind every claim is in "
      "<font face='Mono'>samples/</font>: exact copies of the files that were evaluated.", small)
    S.append(PageBreak())
    P("Terms and abbreviations", h1)
    P("Every short form, model name and technical term used in this report, in plain words. Each is also explained where it "
      "first appears.", small)
    grows, gidx = [["Term", "Meaning"]], []
    for group, items in GLOSSARY:
        gidx.append(len(grows)); grows.append([group, ""])
        grows += [[t, m] for t, m in items]
    T(grows, [0.24, 0.76], zebra=False, group_rows=tuple(gidx))

    # ================= 1 Introduction
    P("1. Introduction", h1)
    P("1.1 Task", h2)
    P("<b>Inputs:</b> an English text, and a recording of that text read by a <b>non-US</b> speaker (the <i>reference</i>). "
      "<b>Output:</b> speech that (1) says exactly the input words with a US accent, (2) is in the LJSpeech voice "
      "and (3) follows the reference's timing: the same total length, every word starting and ending at about the reference "
      "time, pauses included. The reference's voice, accent and pronunciation must not carry over into the output.")
    P("<b>Why it is hard.</b> Part of an accent lies in timing: speakers of English as a second language (L2 speakers) lengthen, "
      "shorten and pause differently from native speakers. The system must copy the timing of <i>words</i> from the reference "
      "but keep the timing of the <i>phones</i> (the individual sounds) inside each word native. It must also stay natural when "
      "a word has to be much slower or faster than the voice would normally say it.")
    P("1.2 Contributions", h2)
    B(["<b>Durations as an input.</b> VITS fine-tuned with MFA phone durations instead of MAS (we call it VITS-MFA), so a "
       "reference recording can dictate the timing. Compared with the unmodified VITS (system V0) given the same timing, the "
       f"average word-timing error falls from {M('v0', 'wb_mae_ms'):.1f} to {M(main, 'wb_mae_ms'):.1f} ms.",
       "<b>Word timing transferred with a native rhythm</b> (the \"elastic fit\" with a stretch limit and exact rounding): the "
       "output is exactly as long as the reference, by construction, for every sentence.",
       "<b>Two defects diagnosed and fixed</b>, each from a listening observation, each with a success rule written down "
       "in advance and a control model that differs in one setting only: a position-in-phone embedding (US-accent "
       "probability +0.09 on the development accents) and two-token diphthongs (\"cry\" heard as \"cra\"; test word error "
       "rate 4.4% → 2.9%).",
       "<b>Two pace-augmentation experiments</b> (training on artificially slowed and sped-up speech, A: random paces, "
       "B: paces of real non-US speakers), reported as negative results.",
       "<b>An evaluation on unseen accents</b>: the test accents are locked in code until the end, systems are compared "
       "sentence by sentence with corrected statistics, and extra measurements separate the model's own timing error "
       "from the aligner's measurement error."])
    P("1.3 Constraints", h2)
    T([["Constraint", "Consequence for the design"],
       ["Train on LJSpeech only", "Voice and accent come from one US speaker. The non-US recordings are used only to evaluate, to tune and for Ablation B's statistics."],
       ["Must work on accents not seen during development", "The non-US corpus (L2-ARCTIC) is split by accent; the test accents are locked in code until the git tag freeze-v1 (the frozen code version) exists."],
       ["One rented data-center GPU (NVIDIA L40S) for about 8 hours, plus an 8 GB laptop GPU", "The main model M1 is trained once on the whole budget; every later experiment continues from it for 20,000 training steps."],
       ["A live demonstration on unknown recordings", "A suite of difficult test cases, a second aligner as fallback, output length exact by construction, and rules for typed text."]],
      [0.32, 0.68])

    # ================= 2 Data
    P("2. Data and the split by accent", h1)
    T([["Dataset", "Use"],
       ["LJSpeech 1.1: 13,100 clips of one US female speaker, 22,050 samples per second", "Training, with the same training / validation / test lists as the original VITS (12,500 / 100 / 500), so the pretrained model never saw the validation or test clips. MFA aligned 13,073 of them (99.79%)."],
       ["L2-ARCTIC v5: 24 non-US speakers, 6 native languages × 4 speakers", "The reference recordings. Its hand-corrected word boundaries are the ground truth for timing."],
       ["Development (dev) accents: Hindi, Mandarin, Spanish", "Tuning, every experiment decision, Ablation B's pace statistics. 100 sentences × 12 speakers = 1,200 references."],
       ["Test accents: Arabic, Korean, Vietnamese", "Evaluated once, after the code was frozen. 104 sentences × 12 speakers = 1,248 references."]],
      [0.4, 0.6])
    tab_caption("Data. Only sentences with a hand annotation for every speaker of the split are evaluated.")
    P("<b>Why split by accent and not by sentence.</b> The claim to support is \"works for accents the system was never tuned "
      "on\"; splitting by sentence would let tuning hear every accent. <b>The lock is enforced in code:</b> reading a test "
      "speaker's audio raises an error unless the git tag <font face='Mono'>freeze-v1</font> exists, and a unit test fills "
      "fake test files with junk bytes so that any early read would fail.")

    # ================= 3 Method
    P("3. Method", h1)
    P("3.1 Design principle: one source for each property of the output", h2)
    T([["Output property", "Its only source", "Deliberately not taken from"],
       ["Words", "the input text, after normalization", "the reference audio or a transcript of it"],
       ["Pronunciation", "the project lexicon (first CMUdict pronunciation; a G2P model for unknown words)", "the reference's pronunciation"],
       ["Voice, accent, intonation", "the model, trained on LJSpeech only", "the reference (no accented audio is used for training)"],
       ["Word start and end times, pauses, total length", "the reference, aligned with MFA", "the model's own durations"],
       ["Timing of phones inside a word", "the model's natural durations, fitted to the word's time span", "the reference's phone timing, which carries its accent"]],
      [0.27, 0.40, 0.33])
    tab_caption("Nothing is averaged between the reference and the model, so each kind of failure has one place to look. Evidence "
                "for the last row: copying the reference's phone durations gave the worst word error rate of all settings tried "
                "(10.6% against 2.5%; §5.1). CMUdict = Carnegie Mellon University's pronouncing dictionary; G2P = a model that "
                "predicts pronunciation from spelling.")
    F(f_pipe, fig_caption("How an output is made. Timing enters only through the reference's word boundaries; inside each word the "
                          "model's natural rhythm is stretched the way LJ herself varies (§3.5). MFA and CTC are two forced aligners "
                          "(they find where each word is in a recording); d̂ᵢ, dᵢ, z and σᵢ are defined in §3.5; the text encoder, "
                          "normalizing flow and HiFi-GAN decoder are the parts of the VITS model (§3.4)."))

    P("3.2 Text front end", h2)
    B(["<b>One text normalizer</b> for the training text, the aligner's transcripts and the input at synthesis. It writes out "
       "currency, symbols, dates, times, abbreviations and numbers as words, in that order; any digit left over stops "
       "with an error instead of being silently dropped.",
       "<b>Lexicon:</b> 14,912 words with one pronunciation each, written in ARPAbet (the phone alphabet of CMUdict, e.g. "
       "\"cry\" = K R AY1): 13,640 from CMUdict, 1,168 predicted by g2p_en (a neural spelling-to-pronunciation model) for "
       "words not in CMUdict, and 104 possessives (<i>x's</i>) built by a rule, because g2p_en mangled them. Words first "
       "seen at synthesis are predicted and appended to the same file, so it stays the only source of pronunciations.",
       "<b>Model tokens:</b> 69 ARPAbet phones plus three special tokens (padding, silence at the ends, a pause between words) "
       "= 72. This is exactly MFA's phone set (checked by a test), so every token has a measured duration.",
       "<b>Typed-text rules</b> (added after the freeze for live input): money, times, dates, ratios, units, web addresses, "
       "e-mail addresses, Roman numerals, joined words such as CamelCase, abbreviations spelled as letters, and refusals "
       "for non-Latin script. 119 of 121 test inputs come out right; a test checks that the words of every corpus "
       "sentence are unchanged."])

    P("3.3 Forced alignment and its error on accented speech", h2)
    P("Forced alignment finds where each word and phone starts and ends in a recording whose text is known. MFA 3.4.3 (the "
      "Montreal Forced Aligner, with its pretrained US-English model and our lexicon) aligns three things with the same "
      "settings: LJSpeech (to get the training durations), the references (to get the timing to copy) and the model outputs "
      "(to measure their timing). Using one aligner everywhere means its conventions cancel out in comparisons. Its search "
      "is widened (beam 100, retry 400) and every setting is validated, because MFA silently ignores options it does not "
      "know. Boundaries are converted to frames (one frame = 11.6 ms) by rounding cumulative times, so rounding errors cannot "
      "add up; silences of at least 3 frames inside a sentence become pauses; every phone gets at least one frame.")
    T([["Development accent", "Word boundaries: mean error", "Words within 20 ms", "Phone boundaries: mean error", "Phones within 20 ms"]]
      + [[a.capitalize() if a != "all" else f"All ({mfa['all']['utterances']:,} sentences)",
          f"{mfa[a]['word']['mae_ms']:.1f} ms", f"{mfa[a]['word']['within_20ms']:.1f}%",
          f"{mfa[a]['phone']['mae_ms']:.1f} ms", f"{mfa[a]['phone']['within_20ms']:.1f}%"] for a in ("hindi", "mandarin", "spanish", "all")],
      [0.28, 0.18, 0.18, 0.18, 0.18], bold_rows=(4,))
    tab_caption("How far MFA's boundaries are from the hand-corrected boundaries of L2-ARCTIC (results/mfa_l2_dev.json). The "
                "system's timing metric aligns the output with MFA and compares it with the hand annotation of the reference, so "
                "about 19 ms of aligner error is part of every system's score: the main system's 23.9 ms is close to this "
                "floor. The fallback aligner (a CTC aligner built on Meta's MMS speech-recognition model) is less accurate, "
                "about 53 ms.")

    P("3.4 Models", h2)
    P("<b>VITS in brief.</b> VITS turns a sequence of phones directly into a waveform. A <i>text encoder</i> turns the phones "
      "into hidden states; each phone's state is repeated for as many frames as the phone lasts, giving the <i>prior</i> (what "
      "each frame should sound like); a <i>normalizing flow</i> (an invertible network) maps samples from the prior to "
      "acoustic features, and a <i>HiFi-GAN decoder</i> turns these into audio. In training, a <i>posterior encoder</i> "
      "encodes the real audio, and the model learns which frames belong to which phone with <i>MAS</i> (monotonic "
      "alignment search), searched again at every training step.")
    P("<b>V0, the immediate baseline</b> (used as is, not retrained): the official pretrained LJSpeech VITS exactly as released, "
      "with MAS, its random (stochastic) duration predictor and its input of IPA phonetic characters with a blank token "
      "between each pair. It is the model M1 is fine-tuned from, so comparing V0 with M1 measures the effect of replacing MAS. "
      "To give it the same pronunciations, the lexicon's phones are written in the IPA characters it was trained on; its "
      "own predicted durations give each phone's natural length, and the same timing fit drives it. Its one adapter "
      "choice (which phone a blank token belongs to) was made on the development accents (blank → next character: 40.7 ms "
      "against 50.8 ms).")
    P("<b>M0, the second baseline</b> (used as is): a pretrained FastSpeech2 model with a HiFi-GAN vocoder. FastSpeech2 predicts "
      "each phone's duration explicitly, so the reference durations can be passed straight in.")
    P("<b>M1, VITS-MFA</b>: VITS fine-tuned from the official checkpoint (553 matching weight tensors loaded) with six changes:")
    T([["#", "Change", "Reason"],
       ["1", "Remove MAS; frames are assigned to phones from MFA's durations", "Durations must be settable at synthesis. MAS learns its own alignment inside the model, which a reference cannot override."],
       ["2", "No blank tokens between phones", "MFA measures no duration for blanks."],
       ["3", "72 ARPAbet tokens instead of 178 IPA characters; the token embedding starts fresh", "The input alphabet changed."],
       ["4", "A deterministic duration predictor, trained on the log of MFA's frame counts", "Only the natural relative lengths inside a word are needed; random sampling would only add noise."],
       ["5", "Optional position-in-phone embedding (16 steps from start to end)", "Lets a frame's target change within a phone (§5.3)."],
       ["6", "Data loading from the project's file lists", "One format for all data."]],
      [0.05, 0.47, 0.48])
    tab_caption("The six changes from VITS to VITS-MFA. The posterior encoder, flow, HiFi-GAN decoder and the training losses are unchanged.")
    P("L<sub>G</sub> = 45·L<sub>mel</sub> + L<sub>KL</sub> + L<sub>dur</sub> + L<sub>adv</sub> + 2·L<sub>fm</sub>,   "
      "L<sub>dur</sub> = mean<sub>i</sub> (log d̂<sub>i</sub> − log d<sub>i</sub>)²", eq)
    P("The total training loss L<sub>G</sub> adds: the spectrogram error L<sub>mel</sub>; the KL divergence L<sub>KL</sub>, "
      "which pulls the prior toward the encoded real audio; the duration error L<sub>dur</sub>; and the two GAN losses "
      "(L<sub>adv</sub>: fool a discriminator network that tells real from generated audio; L<sub>fm</sub>: match the "
      "discriminator's internal features). In L<sub>dur</sub>, d̂<sub>i</sub> is the predicted and d<sub>i</sub> the measured "
      "length of phone i; taking logs makes the error relative (being 2× off costs the same for a short and a long phone). "
      "<b>Training:</b> 114,000 steps of 32 clips on one L40S GPU, about 8 hours, with 16-bit arithmetic (bf16) for the "
      "networks and 32-bit (fp32) for the losses. Later variants continue from M1 for 20,000 steps each with a fresh "
      "optimizer; new parts start at values that leave the model unchanged, so every variant <i>is</i> M1 at its first step.")
    F(f_trn, fig_caption("Training of M1. The validation duration error falls throughout. The validation spectrogram error is lowest "
                         "near 12,000 steps and then rises slightly while the GAN losses trade off: training and validation drift "
                         "together and the discriminator loss stays level, which indicates a balanced GAN rather than overfitting. "
                         "Faint lines: raw values; solid: average over 25 logged points. Validation = 100 held-out LJSpeech clips."))
    T([["System", "Difference from M1 (one setting each; a test checks this)"],
       ["m1_cont", "20,000 more steps, no change: the control for \"more training\""],
       ["m1_pos", "position-in-phone embedding switched on (it starts at zero, so the model starts as M1)"],
       ["abl_a_pos", "m1_pos + 40% of each batch from pace augmentation A (random paces)"],
       ["abl_b_pos", "m1_pos + 40% of each batch from pace augmentation B (paces of the development-accent speakers)"],
       ["m1_pos_diph (main)", "m1_pos + two-token diphthongs: each diphthong X becomes X and X+ (87 tokens)"]],
      [0.25, 0.75], bold_rows=(5,))
    tab_caption("The later variants. Same starting point, same number of steps, same fresh optimizer; each pair differs in one setting.")
    P("<b>What a fixed external alignment takes away.</b> With MAS, the model keeps adjusting which frames belong to which "
      "phone, using its own view of the sound. MFA's alignment is fixed before training by a classic statistical model (an "
      "HMM-GMM: hidden Markov model with Gaussian mixtures) that never sees the network's internal representation. The "
      "learning signal still reaches every part of the network, but that feedback is lost: boundaries MFA places wrongly stay "
      "wrong for all of training, and, without blank tokens, all frames of a phone share one target. Measured on LJSpeech, "
      f"every MFA-trained model has 14-16% less sound change inside phones than the real recordings (Figure {fig_no[0] + 1}). "
      "What it buys is control: these models put words where they are told to (§5.6).")
    F(f_pd, fig_caption("Sound change within phones, measured on 100 LJSpeech validation clips re-synthesized with their measured "
                        "durations: the average change between consecutive frames of the log-mel spectrogram (a picture of the "
                        "sound over time and frequency, on a hearing-like frequency scale). Left: inside phones, relative to the "
                        "real recordings. Right: change at the start of a phone divided by change inside it; the models pile their "
                        "change up at phone starts."))

    P("3.5 Synthesis: transferring the word timing", h2)
    P("For a word that lasts D frames in the reference, the duration predictor gives a natural length d̂<sub>i</sub> for each of "
      "its phones. All phones of the word are scaled with a single number z, found by bisection (repeated halving of an "
      "interval) so that the phones exactly fill the word:")
    P("d<sub>i</sub> = exp(log d̂<sub>i</sub> + z·σ<sub>i</sub>),   Σ<sub>i</sub> d<sub>i</sub> = D", eq)
    P("σ<sub>i</sub> is how much LJ naturally varies phone i (the standard deviation of its log duration in LJSpeech), so z "
      "says how much slower (z &gt; 0) or faster (z &lt; 0) than LJ the word is, in units of that variation. Phones LJ "
      "stretches a lot (vowels) absorb most of the change; phones she keeps steady (stops such as K) barely move. "
      "<i>Example</i>, \"cry\" at twice its natural length: scaling every phone equally gives K / R / AY ×2.0 each; the "
      "elastic fit (z ≈ 1.75) gives ×1.75 / ×2.4 / ×2.0.")
    B(["<b>Stretch cap.</b> A word is stretched to at most 2.5× its natural length; any extra time becomes a pause after it, and "
       "the next word still starts on time. This happens for 148 words in 128 of the 1,248 test sentences.",
       "<b>Exact length.</b> The phone lengths are rounded so that their running total, not each length separately, is "
       "rounded; every word boundary is then within half a frame of the plan, and the output is padded or trimmed to exactly "
       "the reference's number of samples. The length error is checked on every call, so it is 0 by construction.",
       "<b>Diphthongs are split only inside the model.</b> The main system splits each diphthong into two tokens just before "
       "the model and adds the two predicted lengths back together afterwards; the lexicon, the aligner and the timing fit "
       "never see the extra tokens."])

    # ================= 4 Evaluation
    P("4. How the systems are evaluated", h1)
    T([["Requirement", "Metric", "Definition"],
       ["Exact length", "Length error", "output length − reference length, in samples; must be 0"],
       ["Word timing", "Word-boundary error (MAE); share within 20 / 50 ms", "MAE = mean absolute error: for every word, how many ms its start and its end in the output are from the <b>hand-annotated</b> start and end in the reference (output times found by MFA), averaged"],
       ["Pauses", "Pause F1", "pause = a silence of at least 100 ms between two words. F1 combines precision (output pauses that are real) and recall (real pauses the output has); 1 = perfect"],
       ["Exact words", "WER, CER", "word / character error rate: the share of words / characters wrong when Whisper large-v3, OpenAI's speech recognizer, transcribes the output; text and transcript are normalized the same way"],
       ["US accent", "US-accent probability", "from CommonAccent ECAPA, a neural accent classifier; 0 to 1"],
       ["LJ voice", "Speaker similarity", "cosine similarity (−1 to 1) of voice embeddings from an ECAPA speaker model: to LJ (should be high), to the reference speaker (should be low)"],
       ["Naturalness", "UTMOS", "an automatic predictor of the mean opinion score (MOS) listeners would give, 1 to 5"],
       ["Robustness", "Stretch ratio", "output speech length ÷ the model's natural length for the same words (1.0 = LJ's own pace)"]],
      [0.15, 0.27, 0.58])
    tab_caption("Metrics. All evaluation models are pinned to exact versions, checked by file fingerprint (sha256), and loaded offline.")
    B(["<b>Statistics.</b> Averages come with 95% confidence intervals (CI): ranges that contain the true average with 95% "
       "probability, computed by bootstrap (re-drawing the sentences at random 1,000 times). Systems are compared "
       "<i>paired</i>: both speak the same sentences, so the difference is taken sentence by sentence. Each comparison uses "
       "the Wilcoxon signed-rank test, which asks whether these differences are centred on zero. Its p-value is the "
       "probability of a difference at least this large if the systems were really equal. Because 8 metrics are tested at "
       "once, some would look significant by chance; the Holm correction raises the p-values to account for that, and a "
       "difference is called significant only when the corrected p is below 0.05. Comparisons are made overall, per accent "
       "and per tercile (three equal groups) of how much the reference stretches LJ's pace.",
       "<b>Decision rules written before results.</b> The tuning rule and the success criteria of each experiment were "
       "written in the decision log before the experiment ran.",
       "<b>Extra measurements.</b> <i>Duration fidelity</i>: where a word ends up in the output against where the pipeline "
       "placed it; this isolates the model's own timing error from the error of aligning the reference. <i>Aligner "
       "calibration</i>: the same measurement for both aligners on real recordings. <i>Sound change within phones</i> "
       "(§3.4). <i>Formant probe</i>: the resonance frequencies F1 and F2 that identify a vowel, measured with linear "
       "predictive coding (LPC), to see whether a diphthong glides. <i>Pace ρ (rho)</i>: a reference word's length divided "
       "by LJ's typical length for the same phones."])
    P("<b>Caveats, stated rather than hidden.</b>", h3)
    B(["The accent classifier's US probability is close to 0 or 1 for single sentences and can flip on changes nobody hears, "
       "so it is only read as an average over many sentences.",
       "MFA judges systems that were trained on MFA durations, which may favour them. The second judge, the CTC aligner, places "
       f"word starts {calib['ctc']['bias_start_ms']:+.0f} ms late on real speech (MFA: {calib['mfa']['bias_start_ms']:+.0f} ms), "
       "so CTC numbers are only read against that offset, never as a ranking.",
       "UTMOS chose the settings of the audio-stretching tool used for augmentation (on LJSpeech only) and is also an evaluation metric.",
       "M1's development-accent numbers are slightly optimistic, because 120 of those 1,200 references were used for tuning. "
       "The test numbers have no such bias."])

    # ================= 5 Experiments and ablations
    P("5. Experiments and ablations", h1)
    P("Each later experiment started from a listening observation. The input was checked first; the defect was measured with "
      "a targeted measurement; its cause in the design was found; a minimal fix was tested against a control that differs "
      "in that one setting; and the same measurement plus all metrics were repeated. All decisions here were made on the "
      "development accents; an ablation is an experiment that adds or removes one component to measure its effect.", small)
    P("5.1 Tuning the synthesis settings (development accents)", h2)
    P("21 settings were tried on a fixed set of 120 development references: how the timing is transferred (the elastic fit, "
      "scaling every phone equally, or copying the reference's phone lengths), the stretch cap, the noise scale (how much "
      "randomness is sampled from the VITS prior; more is livelier but less clear) and the aligner. The rule, fixed before "
      "running: among elastic settings within 5 ms of the best timing error, take the one with the highest UTMOS. Chosen: "
      f"elastic fit, cap 2.5, noise scale 0.333, MFA ({choice['choice']['wb_mae_ms']:.1f} ms; best {choice['best_mae_ms']:.1f} ms; "
      f"UTMOS {choice['choice']['utmos']:.2f}; word error rate {100 * choice['choice']['wer']:.1f}%).")
    F(f_tun, fig_caption("The tuning grid. Each dot is one setting. The noise scale drives the word error rate (0.333: 2.5-3.5%; "
                         "0.667: 4.6-6.7%); a looser stretch cap lowers the timing error at no cost in naturalness; copying the "
                         "reference's phone lengths gives by far the worst word error rate, because the reference's phone timing "
                         "carries its accent. So only word boundaries are copied."))

    P("5.2 First comparison: M1 against M0 (development accents)", h2)
    T([["1,200 development references", "M0 (FastSpeech2)", "M1 (VITS-MFA)"],
       ["Word-boundary error", "31.3 ms", "<b>23.7 ms</b> (p ≈ 1e-140)"], ["Word boundaries within 20 ms", "55.4%", "<b>68.5%</b>"],
       ["Pause F1", "0.75", "<b>0.84</b>"], ["Word error rate", "10.0%", "<b>3.5%</b>"], ["UTMOS", "3.94", "<b>4.20</b>"],
       ["US-accent probability", "0.73", "0.72 (not significant)"]], [0.4, 0.3, 0.3])
    tab_caption("Both systems receive the same durations; M1 renders them more precisely. p ≈ 1e-140: the chance that the difference "
                "is accidental is about 10<super>−140</super>. US-accent probability did not improve, which set up the next experiments.")

    P("5.3 \"Words end sharply\" → position-in-phone embedding", h2)
    P("Measurements ruled out loudness, pitch and the duration predictor at word ends. What differed was the sound change "
      "<i>inside</i> phones: 14% less than LJ, 20% less in the middle third. The cause: without blank tokens, every frame of "
      "a phone has the same target. The fix gives each frame a learned vector for its position in the phone (frame k of a "
      "d-frame phone goes to step ⌊16(k + 0.5)/d⌋, from 0 at the start to 15 at the end). <b>Result against m1_cont</b> (the "
      "same 20,000 steps without the change): the sound change inside phones did not move (0.856× vs 0.858× LJ; the "
      "embedding's effect sits at the edges of phones, which this measurement leaves out), but the US-accent probability "
      "rose from 0.680 to 0.769 (p = 5e-31; higher in all three accents), with no significant change elsewhere. m1_pos "
      "became the base for the next experiments.")

    P("5.4 Pace augmentation, Ablations A and B (negative result)", h2)
    P("<b>Hypothesis:</b> training on LJSpeech clips that were made slower or faster teaches the model to stay natural when a "
      "reference forces unusual word lengths. <b>A (random paces):</b> each copy gets a pace drawn uniformly between −1.5 and "
      "+1.5 (in the units of z, §3.5), each word a small random deviation from it, and some pauses are lengthened or inserted "
      "with recorded room noise. <b>B (realistic paces):</b> word paces and pauses are drawn from the statistics of the "
      "development-accent speakers (each word's pace correlated with the previous word's; a pause after 22% of words); "
      "only word lengths are used, never their phone timing. Two copies of every clip, mixed 60/40 with the original clips; "
      "the duration loss ignores the copies; the 5% of copies that lost the most naturalness were removed.")
    P("<b>A change forced by measurement:</b> the planned audio-stretching tool, Rubber Band R3, damaged every clip (UTMOS 4.38 → "
      "2.92, about 1.2 points even when a clip changed length by less than 5%). It was replaced with WSOLA "
      "(waveform-similarity overlap-add, which changes speed without changing pitch), written for this project; the average "
      "UTMOS loss fell from 1.34 to 0.32, and re-aligning the stretched clips found word boundaries a median of about 7 ms "
      "from where they were planned.")
    T([["1,200 development references, compared with m1_pos", "m1_pos", "abl_a_pos", "abl_b_pos"],
       ["Word-boundary error", "24.16 ms", "24.04 ms (n.s.)", "24.21 ms (n.s.)"], ["Word error rate", "3.9%", "3.4% (n.s.)", "3.9% (n.s.)"],
       ["US-accent probability", "0.769", "<b>0.642</b> (p &lt; 1e-10)", "0.747 (n.s.)"],
       ["UTMOS", "4.18", "<b>4.07</b> (p &lt; 1e-10)", "<b>4.10</b> (p &lt; 1e-5)"]], [0.37, 0.18, 0.23, 0.22])
    tab_caption("n.s. = not significant after Holm correction. Neither ablation improved timing or word error rate, even for the "
                "most stretched third of the sentences; both cost naturalness, and B shows no gain even on the accents its "
                "statistics came from. The likely cause: 40% of each batch carried leftover stretching artifacts, which the model "
                "learned. A's paces are far wider than real speech, which explains its loss of US-accent probability.")

    P("5.5 \"cry\" heard as \"cra\" → two-token diphthong", h2)
    P("The lexicon gives \"cry\" → K R AY1 (correct), so the error was the model's. A diphthong is a vowel that glides, like "
      "the AY of \"cry\" moving from /a/ to /ɪ/. Measuring the formants (F1 and F2, the resonances that identify a vowel) "
      "showed the reference gliding (F2 up 285 Hz, F1 from 737 to 554 Hz) while M1, m1_cont and m1_pos held a flat /a/; "
      "across LJ's validation clips every model flattened diphthongs (average F2 movement 307-321 Hz against 396 Hz for LJ). "
      "<b>Cause: the token design.</b> The original VITS read \"cry\" as the IPA characters k, ɹ, a, ɪ with blanks between them, "
      "so a diphthong had two targets and a transition. Our ARPAbet input made it one token with one target. <b>Fix:</b> each "
      "diphthong (AW, AY, EY, OW, OY, at all stress levels) becomes two tokens, X and X+ (72 → 87 tokens). MFA gives one "
      "length for the diphthong, which is split evenly between the two; for the timing fit the two predicted lengths are "
      "added back together; X+ starts as a copy of X, so the model starts as m1_pos.")
    T([["1,200 development references, compared with m1_pos", "m1_pos", "m1_pos_diph", "p (Holm)"],
       ["Word-boundary error", "24.16 ms", "<b>23.48 ms</b>", "&lt; 1e-6"], ["Word error rate", "3.9%", "<b>2.7%</b>", "&lt; 1e-6"],
       ["Character error rate", "2.0%", "<b>1.2%</b>", "&lt; 1e-6"], ["UTMOS", "4.18", "<b>4.20</b>", "2e-4"],
       ["US-accent probability", "<b>0.769</b>", "0.738", "1e-4"], ["Average F2 movement in diphthongs (LJ: 396 Hz)", "321 Hz", "<b>389 Hz</b>", ""]],
      [0.4, 0.2, 0.2, 0.2])
    tab_caption("\"cry\" now glides (F1 864 → 528 Hz, F2 up 215 Hz). The one criterion not met, the US-accent probability, was "
                "investigated: its drop does not follow the number of diphthongs in a sentence (Spearman correlation −0.03, i.e. "
                "unrelated), sentences without any diphthong dropped too, the classifier flips both ways (91 large drops, 58 "
                "large rises), and the four largest drops sounded identical to the listener. Decision (D-85): m1_pos_diph is the "
                "main system. The code was then frozen and the test accents opened once.")

    P("5.6 The immediate baseline V0, and how faithfully each model renders its durations", h2)
    P("V0 is the same pretrained VITS before MAS was replaced, given the same word timing through the same pipeline. To see "
      "<i>why</i> it misses the timing, the duration-fidelity measurement compares where MFA finds each word in the output "
      "with where the pipeline placed it, which removes the error of aligning the reference.")
    F(f_fid, fig_caption("Duration fidelity on the test accents. Left: the average offset of word starts and ends from where they "
                         "were placed; V0's words come out about 20 ms early. Right: the remaining scatter once that offset is "
                         "removed; V0 is also the least precise. The fine-tuned systems put words almost exactly where they are "
                         "told. The CTC aligner, an independent second judge measured against its own offset on real speech, finds "
                         "the same early shift for V0."))
    P("The two aligners agree on the direction and size of V0's shift; the CTC aligner is too imprecise to rank the scatter, so "
      "that part rests on MFA. <b>The trade-off, stated plainly:</b> V0 is the better system on word error rate and UTMOS. "
      "Fine-tuning on MFA durations without blank tokens bought controllable timing and a more US-like accent, at a cost "
      "that the two-token diphthong already halved (word error rate above V0's: M1 +1.7 percentage points, main system +0.8).")
    F(f_for, fig_caption("Every comparison between a system and its control, as the average per-sentence difference "
                         "(system minus control). \"dev\" = development accents, \"test\" = test accents. A Holm-corrected p-value "
                         "accounts for the 8 metrics tested in each comparison (glossary); filled dots are differences that are "
                         "very unlikely to be chance."))

    # ================= 6 Results
    P("6. Results on the test accents", h1)
    P(f"Arabic, Korean and Vietnamese: 12 speakers, {sums[main]['n']:,} references, evaluated once after the code was frozen. "
      f"MFA aligned all {sums[main]['aligner_counts']['mfa']:,} references (the fallback aligner was never needed). The length "
      "error is 0 for every output of every system.")
    hdr = ["System", "Word-boundary error, ms [95% CI]", "Within 20 ms", "Pause F1", "Word error rate", "US accent probability", "UTMOS (1-5)", "Similarity to LJ voice"]
    trows = [hdr, ["Non-US references (input)", "", "", "", f"{100 * M(main, 'wer_ref'):.1f}%", f"{M(main, 'us_prob_ref'):.3f}", "", ""],
             ["LJSpeech recordings", "", "", "", f"{100 * lj['wer']:.1f}%", "", f"{lj['utmos']:.2f}", ""]]
    for s in systems:
        mm = sums[s]["metrics"]["all"]
        ci = mm["wb_mae_ms"]["ci95"]
        trows.append([lab[s], f"{mm['wb_mae_ms']['mean']:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]", f"{mm['wb_within20']['mean']:.1f}%",
                      f"{mm['pause_f1']['mean']:.3f}", f"{100 * mm['wer']['mean']:.1f}%", f"{mm['us_prob']['mean']:.3f}",
                      f"{mm['utmos']['mean']:.2f}", f"{mm['spk_sim_lj']['mean']:.3f}"])
    T(trows, [0.17, 0.19, 0.09, 0.08, 0.08, 0.13, 0.09, 0.17], bold_rows=(3 + systems.index(main),))
    tab_caption("Main results (results/&lt;system&gt;/test/summary.json). CI = confidence interval. Similarity to the reference "
                f"speaker's voice stays low ({M(main, 'spk_sim_ref'):.2f} for the main system): the reference's voice does not carry over.")
    pr = [["Compared pair (test)", "Word-boundary error (ms)", "Within 20 ms (% points)", "Word error rate (% points)", "UTMOS", "US accent probability"]]
    for sp, a, b in cfg.paired:
        if sp != "test":
            continue
        r = paired(sp, a, b)
        def c(m, sc=1, d=1):
            v = float(r[m]["mean_diff"]) * sc
            p = float(r[m]["p_holm"])
            return f"{v:+.{d}f}" + (f" (p {fmt_p(p)})" if p < cfg.alpha else " (n.s.)")
        pr.append([f"{short(a)} vs {short(b)}", c("wb_mae_ms", 1, 1), c("wb_within20", 1, 1), c("wer", 100, 1),
                   c("utmos", 1, 2), c("us_prob", 1, 3)])
    T(pr, [0.21, 0.17, 0.16, 0.16, 0.14, 0.16])
    tab_caption("Average change per sentence, first system minus second, on the 1,248 test references. p = Holm-corrected "
                "p-value of the Wilcoxon signed-rank test; n.s. = not significant (corrected p of 0.05 or more). "
                "Points = percentage points.")
    P("<b>Findings.</b>", h3)
    B([f"<b>Replacing MAS with MFA durations is what makes the output follow the reference timing:</b> main system against V0, "
       f"−16.5 ms word-boundary error and +28 percentage points of boundaries within 20 ms, in every accent and at every "
       f"amount of stretching; the cost is +0.8 points of word error rate and −0.12 UTMOS.",
       "<b>VITS-MFA beats FastSpeech2 on everything but accent</b> (M1 against M0: −8 ms timing error, word error rate 9.3% → 3.8%, UTMOS +0.23).",
       "<b>The two-token diphthong also helps on the test accents</b>, against its control: word error rate −1.5 points, timing error −0.7 ms, UTMOS +0.03.",
       "<b>Pace augmentation is a negative result on the test accents too:</b> no gain in timing or word error rate; UTMOS −0.11 (A), −0.09 (B).",
       "<b>The development conclusions held on the unseen accents</b>, whose pace was similar (pace figure at the end of this section)."])
    F(f_head, fig_caption(f"The headline sample, {head['id']} (\"{head['text']}\"). Top: the non-US reference with the "
                          "hand-annotated word times. Below: the main system and V0, each with the word times MFA finds in the "
                          "output. All three have the same length by construction; the main system's words sit on the reference's "
                          f"word starts, V0's drift early. Audio: samples/demo/arabic/{head['id']}/."))
    F(f_cdf, fig_caption("How the error is distributed over the test sentences: for each error value on the horizontal axis, the share of "
                         "sentences with that error or less (a cumulative distribution). A curve further left is better. The whole "
                         "distribution of the main system is shifted left of V0 and M0, not only its average. 90th percentiles "
                         "(90% of sentences are below): "
                         + ", ".join(f"{short(s)} {quantile(mae_values(rows[s]), 0.9):.1f} ms" for s in core) + "."))
    F(f_acc, fig_caption("Results per test accent. Error bars: 95% confidence intervals. The timing gain over V0 and M0 holds in "
                         "every accent; Korean speakers are the easiest for every system."))
    T([["Accent", "US-accent prob. of the references"] + [f"{short(s)}: timing error / word error rate" for s in ("v0", "m1", main)]
       + ["Main system: US-accent prob."]]
      + [[a.capitalize(), f"{M(main, 'us_prob_ref', a):.3f}"]
         + [f"{M(s, 'wb_mae_ms', a):.1f} ms / {100 * M(s, 'wer', a):.1f}%" for s in ("v0", "m1", main)]
         + [f"{M(main, 'us_prob', a):.3f}"] for a in cfg.accents],
      [0.13, 0.15, 0.18, 0.18, 0.2, 0.16])
    tab_caption("Per test accent. The US-accent probability rises from 0.41-0.60 for the references to 0.72-0.79 for the outputs.")
    F(f_str, fig_caption(f"Timing error against the stretch ratio (how much longer or shorter the reference makes the words than LJ "
                         f"would say them; sentences grouped by the main system's ratio, available for {sum(n_main):,} of them). "
                         "Every system's error rises when the reference is much slower or faster than LJ; the gap to V0 holds in "
                         "every group."))
    F(f_pace, fig_caption("Pace of the reference sentences relative to LJ (ρ, rho: each word's length divided by LJ's typical length "
                          "for the same phones, averaged per sentence). Both accent groups are about 12-16% slower than LJ with a "
                          "similar spread, so the development accents were a fair stand-in for the test accents."), 0.85)
    P("<b>Failures are shown, not hidden.</b> The worst sentence per accent is in samples/demo/ (e.g. THV_arctic_a0067: 148 ms "
      "for the main system and V0 alike). The second aligner (CTC) measured on the outputs gives 45.6-52.0 ms for all systems, "
      "but because it places word starts 38 ms late on real speech, V0's early words partly cancel that offset; its numbers "
      "are therefore read only against its own offset (§5.6), never as a ranking.")

    # ================= 7 Efficiency
    P("7. Efficiency", h1)
    macs = gpu["macs"]
    T([["Quantity", "Value (main system)"],
       ["Parameters used at synthesis", f"{gpu['params']['inference_path'] / 1e6:.1f} million: waveform decoder {gpu['params']['modules']['decoder'] / 1e6:.1f} M, "
        f"flow {gpu['params']['modules']['flow'] / 1e6:.1f} M, text encoder {gpu['params']['modules']['text_encoder'] / 1e6:.1f} M, "
        f"duration predictor {gpu['params']['modules']['duration_predictor'] / 1e6:.2f} M"],
       ["Parameters used only in training", f"{gpu['params']['training_only'] / 1e6:.1f} million: discriminators "
        f"{gpu['params']['modules']['discriminators (train only)'] / 1e6:.1f} M, posterior encoder {gpu['params']['modules']['posterior_encoder (train only)'] / 1e6:.1f} M"],
       ["Computation", f"{macs['1s']['gmacs_per_audio_second']:.1f} billion multiply-accumulate operations (GMACs) per second of audio; "
        f"{100 * macs['1s']['flow_decoder_gmacs'] / macs['1s']['total_gmacs']:.1f}% of it in the flow and decoder. Checked by hand for one "
        f"layer: {macs['hand_worked_decoder_conv_pre']['K']} × {macs['hand_worked_decoder_conv_pre']['C_in']} × {macs['hand_worked_decoder_conv_pre']['C_out']} "
        f"× {macs['hand_worked_decoder_conv_pre']['L_out']} = {macs['hand_worked_decoder_conv_pre']['macs'] / 1e6:.1f} million"],
       ["Real-time factor (computing time ÷ audio length), laptop GPU", f"model only {gpu['rtf']['model_only']:.3f} (about {1 / gpu['rtf']['model_only']:.0f}× faster than real time); "
        f"end to end {gpu['rtf']['end_to_end']:.2f}, dominated by aligning the reference"],
       ["Training speed (L40S GPU)", "0.229 s per step of 32 clips; 114,000 steps ≈ 7.3 hours"]], [0.33, 0.67])
    tab_caption("Measured with results/profile_*.json: 10 warm-up runs and 50 timed runs (the aligner: 1 + 5), median times, "
                "synchronized with the GPU. Operations counted with ptflops, a library that counts the operations of a network.")
    F(f_eff, fig_caption("Left: time to produce one 3.7-second sentence, split by stage, on the laptop GPU (NVIDIA RTX 4060) and on "
                         "the CPU with 12 or 1 threads (processor cores). Aligning the reference with MFA takes over 3 seconds "
                         "everywhere and dominates; on the GPU the model itself needs 45 ms. Right: the model's real-time factor "
                         "for outputs of different lengths (logarithmic scale; below the dashed line = faster than real time)."))

    # ================= 8 Robustness and live testing
    P("8. Robustness and live testing", h1)
    B(["<b>Difficult-input suite</b> (21 cases, all pass on GPU and CPU): digits, currency, dates, abbreviations, an unknown name; "
       "MP3 files, 8,000 and 48,000 samples per second, stereo, clipped, noisy and echoing audio; recordings over 20 s, a 1.8 s "
       "pause, 1.6× speed, a word cut off; a word in the text that is not spoken, a false start, and a forced use of the "
       "fallback aligner. Pass = exact length, or a clear fallback.",
       "<b>Typed text:</b> 121 test inputs (119 right; the two misses are guessed pronunciations of unknown names) and 12 "
       "end-to-end rehearsal cases, including inputs that must be refused.",
       "<b>Unit tests:</b> 317 pass on the author's machine; 309 pass and 9 are skipped on a fresh copy without the datasets.",
       "<b>Browser demo:</b> a visitor records or uploads a sentence and hears all seven systems side by side; 50 recorded "
       "sessions are kept in samples/browser_tests/."])
    P("Live testing of the browser demo found failures that read-aloud corpora never contain:", small)
    T([["Input", "What happened", "Cause", "Status"],
       ["abc, chaganti (words first seen at synthesis)", "MFA marked them unknown; the demo used the less accurate fallback aligner", "MFA reused an old copy of the lexicon it had cached", "<b>Fixed:</b> the cache is now tied to the lexicon's fingerprint"],
       ["bcd, xyz", "pronounced as an invented word (bcd → \"bed\")", "nothing marked them as letters", "<b>Fixed:</b> unknown lowercase strings of 3-5 letters without a vowel are spelled out"],
       ["the letter \"a\" in a, b, c", "aligned to only 30-80 ms; the output says \"uh\"", "one pronunciation for \"a\": AH0, the article", "<b>Open:</b> needs the letter-name reading (EY1 gave 220-330 ms)"],
       ["\"I\" in may I know you", "\"I\" got 30 ms, the aligner's minimum", "two vowels meet (EY then AY) with no consonant between", "<b>Open:</b> fine after a pause or at the start"],
       ["ijk; brrr, grr; pq", "invented word; wrongly spelled out; not spelled", "limits of the hand-written rule", "<b>Open:</b> needs a learned text normalizer"]],
      [0.21, 0.27, 0.25, 0.27])
    tab_caption("Live-testing findings (E-35). The test results are unaffected: every evaluated word was in the lexicon "
                "before MFA first cached it.")

    # ================= 9 Limitations
    P("9. Limitations and future work", h1)
    B(["<b>The accent classifier is a weak judge;</b> a listening test (for example MUSHRA, where listeners rate versions side "
       "by side) with accent ratings is the right next measurement.",
       "<b>Part of the accent lies in timing:</b> forcing LJ's voice onto a non-US speaker's word timing limits how American it can sound.",
       "<b>A fixed alignment flattens phones</b> (14-16% less sound change inside phones); V0 keeps a better word error rate and UTMOS.",
       "<b>No word-boundary marker</b> in the model's input (\"a name\" and \"an aim\" are the same phones); the diphthong is always split 50/50.",
       "<b>The ablations started late</b> (20,000 steps from M1), so they test late-stage augmentation only.",
       "<b>V0 needs an adapter</b> (lexicon phones written as IPA, a rule for blank tokens); a learned adapter might narrow its gap.",
       "<b>End-to-end speed is limited by the aligner</b> (3.3 s per reference against 45 ms for the model)."])
    P("<b>Future work, most promising first:</b> (1) let the model learn its alignment again (MAS) but only <i>inside</i> each "
      "word's span from MFA, restoring the feedback while keeping timing control; (2) mark word boundaries in the model's "
      "input; (3) learn where to split each diphthong; (4) split other gliding sounds (CH, JH, ER) the same way; (5) a "
      "listening test as the accent judge; (6) a faster, always-loaded aligner for live use; (7) letter names and "
      "vowel-only words for live input.")
    P("<b>Takeaway.</b> Make durations an input, copy the timing of words (not of phones), and check how sounds are tokenized "
      "before adding data: replacing MAS cut the word-timing error from 40.5 to 24 ms, and the largest later gain came from "
      "giving a diphthong two tokens, not from more training data.")

    # ================= 10 Reproducibility
    P("10. Reproducibility and deliverables", h1)
    T([["Step", "Command", "Result"],
       ["Software environments", "make env", "isolated, version-pinned conda environments tts and mfa (exact lists in envs/)"],
       ["Trained models", "make models", "the frozen models, downloaded from the Hugging Face Hub at a fixed revision and checked by file fingerprint (sha256)"],
       ["Synthesize", "make infer, make live-demo", "output audio of exact length, word timings, a timing plot"],
       ["All test outputs", "make outputs", "the 1,248 evaluated outputs of every system"],
       ["Tests", "make test, make text-stress, make rehearse", "309 unit tests pass on a fresh copy; 119 of 121 typed inputs; 12 of 12 rehearsal cases"],
       ["Browser demo", "make web-demo (and make web-tunnel to share it)", "record or upload, all seven systems side by side"],
       ["This report", "make report-pdf", "docs/meeami-tts_report.pdf, from the result files in the repository"]], [0.18, 0.3, 0.52])
    tab_caption("Checked from a fresh copy of the repository with no login and an empty download cache (D-104).")
    T([["Deliverable", "Where"],
       ["Code (frozen at git tag freeze-v1)", cfg.links.github],
       ["Detailed report linking every claim to its code", "docs/REPORT.md; decision and error log docs/DECISIONS.md"],
       ["Audio evidence: reference, hand annotation, all 7 outputs, word timings, 3 accents", "samples/demo/ (21 samples; each sample's timing error recomputed and checked against the evaluation, 147 of 147 match)"],
       ["Listening sets behind the experiment decisions; browser recordings", "samples/listening/, samples/browser_tests/"],
       ["Trained models and every test output", cfg.links.hf]], [0.45, 0.55])
    tab_caption("Where each deliverable is.")

    # ================= Appendix
    P("Appendix A. Deviations from the plan", h1)
    T([["The plan said", "Done instead", "Why"],
       ["Rubber Band R3 to stretch audio", "WSOLA (written for this project), same stretch plan", "R3 cost about 1.2-1.5 UTMOS on every clip; WSOLA 0.32"],
       ["torchaudio's forced-alignment function", "the same algorithm written in numpy, same MMS model", "the installed torchaudio package lacks that function"],
       ["Unknown words from g2p_en", "plus a rule for possessives", "g2p_en mangled 104 possessives"],
       ["Three GPU machines", "one L40S: M1, then 20,000-step continuations", "availability"],
       ["Speed measured on the data-center GPU", "laptop GPU and CPU only", "could not run on the rented machine"],
       ["72 model tokens", "87 for m1_pos_diph only", "diphthong flattening was diagnosed"],
       ["UTMOS loaded with its default loader", "weights pinned by fingerprint and loaded safely (weights only)", "the default loader ran unverified code"],
       ["Baseline M0 only", "plus V0, the original VITS with MAS (added after the freeze, synthesis only)", "the immediate baseline for replacing MAS"],
       ["Text normalizer as specified", "plus the typed-text rules (after the freeze)", "live input; corpus words unchanged (tested)"],
       ["MFA's single-file aligner with a fixed cache", "cache tied to the lexicon's fingerprint", "an old cached lexicon made new words unknown"]],
      [0.28, 0.38, 0.34])
    P("Appendix B. Claim → evidence", h1)
    T([["Claim", "Evidence (files in the repository)"],
       ["Output length exact for every output", "length_error = 0 in every results/*/test/per_utt.csv"],
       ["MFA's error on accented speech ≈ 19 ms", "results/mfa_l2_dev.json, reports/mfa_l2_dev.md"],
       ["Replacing MAS makes timing follow the reference", "results/paired/test_m1_pos_diph_vs_v0.csv; samples/demo/"],
       ["V0 renders durations early and scattered", "results/fidelity/test.json, calibration_dev.json"],
       ["A fixed alignment flattens phones", "results/phone_dynamics/*.json"],
       ["Cap 2.5 and noise scale 0.333 chosen by a rule fixed in advance", "results/tune/grid.json, choice.json"],
       ["The ablations give no gain", "results/paired/{dev,test}_abl_*_vs_m1_pos.csv"],
       ["The two-token diphthong improves timing, word error rate and UTMOS; \"cry\" glides", "results/paired/*_m1_pos_diph_vs_m1_pos.csv; samples/listening/p2c_diphthongs/"],
       ["Training curves; reference pace", "results/report_data/m1_training_curves.csv, pace_rho.csv"],
       ["The test accents were not read before the freeze", "git tag freeze-v1 → commit 5889ecf; tts/guard.py"],
       ["The hosted models are the frozen ones", "MANIFEST.sha256 on the Hugging Face Hub; scripts/export_infer.py --verify"]], [0.45, 0.55])

    doc.multiBuild(S)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "report_pdf.yaml"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config).report_pdf
    out = ROOT / (a.out or cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(cfg, out)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
