"""Build the PDF project report (D-107): docs/meeami-tts_report.pdf.

Every number in a table or figure is read from the committed result files (results/, samples/), never
typed in; the prose states the reasoning and points to docs/REPORT.md and docs/DECISIONS.md for detail.
Run in the `report` env (envs/report.yml): `make report-pdf`.
"""
from __future__ import annotations

import argparse
import csv
import json
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
            "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
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
        fig, ax = plt.subplots(figsize=(7.2, 2.9))
        ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
        def box(x, y, w, h, text, fc="#f2f1ed", ec="#8a8984", bold=False):
            ax.add_patch(plt.matplotlib.patches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2,
                    weight="bold" if bold else "normal", color="#1d1c1a")
        def arrow(x0, y0, x1, y1):
            ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.9))
        blue, orange = "#dce9f8", "#fbe3d8"
        box(1, 30, 17, 7, "reference.wav\n(non-US speaker)", fc=orange)
        box(24, 30, 19, 7, "MFA align_one\n(fallback: MMS_FA CTC)")
        box(49, 30, 22, 7, "word start / end times\n+ pauses ≥ 3 frames")
        box(1, 17, 17, 7, "input text", fc=blue)
        box(24, 17, 19, 7, "normalize + pre-pass\n→ lexicon phones")
        box(49, 17, 22, 7, "duration predictor\n→ native log d̂ᵢ")
        box(76, 16, 23, 21.5, "elastic fit per word\ndᵢ = exp(log d̂ᵢ + z·σᵢ)\none z per word\ncap 2.5× → rest is pause\ncumulative rounding\n= exact length", fc="#efeefa", bold=False)
        box(1, 2, 70, 8, "VITS-MFA: encoder → expand by dᵢ (+ position-in-phone) → prior (noise 0.333)\n→ inverse flow → HiFi-GAN decoder → pad / trim to exactly the reference length")
        box(77, 2, 22, 8, "out.wav · timings.json\ntiming.png", fc=blue, bold=True)
        arrow(18, 33.5, 24, 33.5); arrow(43, 33.5, 49, 33.5); arrow(71, 33.5, 77, 30)
        arrow(18, 20.5, 24, 20.5); arrow(43, 20.5, 49, 20.5); arrow(71, 20.5, 77, 22)
        arrow(88, 17, 70, 9); arrow(71, 6, 77, 6)
        ax.text(24, 13, "timing comes only from the reference;\nwords, voice and phone rhythm only from text + model",
                fontsize=6.8, color="#52514e", style="italic", va="center")
        return self.save(fig, "pipeline")

    def headline(self, sample_dir):
        plt, import_np = self.plt, __import__("numpy")
        d = json.loads((sample_dir / "timings.json").read_text())
        rows = [("reference (L2 speaker)", sample_dir / "reference.wav", d["annotation_words"], "#8a8984",
                 "human annotation")]
        for s in (self.cfg.main, "v0"):
            sd = d["systems"][s]
            rows.append((f"{self.lab[s]}: MAE {sd['metrics']['wb_mae_ms']:.1f} ms", sample_dir / f"{s}.wav",
                         sd["measured_words"], self.col[s], "MFA on the output"))
        fig, axes = plt.subplots(len(rows), 1, figsize=(7.2, 4.4), sharex=True)
        dur = d["ref_duration_s"]
        for ax, (title, wav, words, c, src) in zip(axes, rows):
            with wave.open(str(wav)) as w:
                x = import_np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float)
                sr = w.getframerate()
            x /= max(1.0, abs(x).max())
            t = import_np.arange(len(x)) / sr
            ax.plot(t, x, color=c, lw=0.35, alpha=0.75)
            for i, wd in enumerate(words):
                ax.axvspan(wd["start"], wd["end"], ymin=0.0, ymax=0.08, color=c, alpha=0.9 if i % 2 else 0.55, lw=0)
                ax.axvline(wd["start"], color="#2b2a27", lw=0.4, alpha=0.5)
                ax.text((wd["start"] + wd["end"]) / 2, 1.08, wd["word"], ha="center", va="bottom", fontsize=6.3)
            for wd in d["annotation_words"]:
                ax.axvline(wd["start"], color="#e34948", lw=0.5, ls=":", alpha=0.8)
            ax.set_ylim(-1.05, 1.55); ax.set_yticks([]); ax.grid(False)
            ax.set_title(f"{title}   (word times: {src})", loc="left", fontsize=7.8, pad=2)
            ax.set_xlim(0, dur)
        axes[-1].set_xlabel("time (s); red dotted lines = word starts in the human annotation of the reference")
        fig.tight_layout(h_pad=0.6)
        return self.save(fig, "headline")

    def cdf(self, rows_by_sys):
        plt, np = self.plt, __import__("numpy")
        fig, ax = plt.subplots(figsize=(7.0, 2.9))
        for s in self.cfg.core:
            v = np.sort(mae_values(rows_by_sys[s]))
            y = np.arange(1, len(v) + 1) / len(v)
            ax.plot(v, y, color=self.col[s], lw=2 if s == self.cfg.main else 1.4)
            med = quantile(v.tolist(), 0.5)
            ax.plot([med], [0.5], "o", ms=5, color=self.col[s], mec="white", mew=1.2)
            ax.lines[-2].set_label(f"{self.lab[s]}: median {med:.1f} ms")
        ax.axhline(0.5, color="#8a8984", lw=0.6, ls="--")
        ax.legend(loc="lower right", fontsize=7)
        ax.set_xlim(0, 100); ax.set_ylim(0, 1.0)
        ax.set_xlabel("per-reference word-boundary MAE (ms), test, 1,248 references")
        ax.set_ylabel("fraction of references ≤ x")
        return self.save(fig, "cdf")

    def per_accent(self, summaries):
        plt, np = self.plt, __import__("numpy")
        acc, core = list(self.cfg.accents), list(self.cfg.core)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
        for ax, (metric, scale, unit) in zip(axes, [("wb_mae_ms", 1, "word-boundary MAE (ms)"), ("wer", 100, "WER (%)")]):
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
            ys = [b[0] for b in bins[s]]
            ax.plot(xs, ys, "-o", color=self.col[s], lw=2 if s == self.cfg.main else 1.4, ms=4.5,
                    mec="white", mew=1, label=self.lab[s])
        ax.set_xticks(xs, [f"{l}\n(n={n})" for l, n in zip(labels, n_main)])
        ax.set_xlim(-0.3, len(xs) - 0.7); ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
        ax.set_xlabel("stretch ratio of the reference = main system's output speech frames ÷ its native frames")
        ax.set_ylabel("mean MAE (ms)")
        return self.save(fig, "stretch")

    def tuning(self, grid, choice_name):
        plt = self.plt
        fig, ax = plt.subplots(figsize=(7.0, 2.9))
        shades = {0.333: "#2a78d6", 0.5: "#7fa9e3", 0.667: "#c3d6f2"}
        for g in grid:
            if g["transfer"] == "elastic" and g["aligner"] == "mfa":
                ax.plot(g["wb_mae_ms"], g["utmos"], "o", ms=6, color=shades[g["noise_scale"]], mec="#2b2a27", mew=0.4)
                ax.text(g["wb_mae_ms"] + 0.12, g["utmos"] + 0.004, f"{g['cap'] or 'none'}", fontsize=5.6, color="#52514e")
            else:
                lab = g["transfer"] if g["aligner"] == "mfa" else "CTC aligner"
                ax.plot(g["wb_mae_ms"], g["utmos"], "s", ms=6, color="#b5b4ad", mec="#2b2a27", mew=0.4)
                ax.text(g["wb_mae_ms"] + 0.3, g["utmos"], f"{lab} (WER {100 * g['wer']:.1f}%)", fontsize=6.4, va="center")
            if g["name"] == choice_name:
                ax.plot(g["wb_mae_ms"], g["utmos"], "o", ms=13, mfc="none", mec="#e34948", mew=1.4)
                ax.annotate("chosen: elastic, cap 2.5, noise 0.333", (g["wb_mae_ms"], g["utmos"]),
                            (g["wb_mae_ms"] + 7, g["utmos"] - 0.035), fontsize=7, color="#e34948",
                            arrowprops=dict(arrowstyle="-", color="#e34948", lw=0.7))
        for ns, c in shades.items():
            ax.plot([], [], "o", color=c, mec="#2b2a27", mew=0.4, label=f"elastic, noise {ns} (label = stretch cap)")
        ax.plot([], [], "s", color="#b5b4ad", mec="#2b2a27", mew=0.4, label="other transfer / aligner")
        ax.legend(fontsize=6.5, loc="center right")
        ax.set_xlabel("word-boundary MAE (ms), 120 dev references"); ax.set_ylabel("UTMOS")
        return self.save(fig, "tuning")

    def forest(self, comps):
        """comps: list of (label, {metric: row}). Δ = system − control; filled marker = Holm p < alpha."""
        plt = self.plt
        metrics = [("wb_mae_ms", 1, "Δ word-boundary MAE (ms)\n(negative = better)"),
                   ("wer", 100, "Δ WER (points)\n(negative = better)"),
                   ("utmos", 1, "Δ UTMOS\n(positive = better)"),
                   ("us_prob", 1, "Δ US probability\n(positive = better)")]
        fig, axes = plt.subplots(1, 4, figsize=(7.3, 0.36 * len(comps) + 1.1), sharey=True)
        ys = list(range(len(comps)))[::-1]
        for ax, (m, sc, title) in zip(axes, metrics):
            for y, (lab, rows) in zip(ys, comps):
                r = rows[m]
                d = float(r["mean_diff"]) * sc
                sig = float(r["p_holm"]) < self.cfg.alpha
                good = r["better"] == "True"
                c = "#2a78d6" if good else "#eb6834"
                ax.plot([0, d], [y, y], color=c, lw=1.2, alpha=0.6)
                ax.plot(d, y, "o", ms=5.5, color=c if sig else "white", mec=c, mew=1.3)
            ax.axvline(0, color="#2b2a27", lw=0.7)
            ax.set_title(title, fontsize=7.2); ax.grid(axis="y", visible=False)
            ax.tick_params(axis="x", labelsize=6.5)
        axes[0].set_yticks(ys, [c[0] for c in comps], fontsize=6.8)
        fig.text(0.01, -0.02, "filled = significant after Holm correction (p < 0.05); hollow = not significant; "
                 "blue = in the better direction, orange = worse", fontsize=6.6, color="#52514e")
        fig.tight_layout(w_pad=0.6)
        return self.save(fig, "forest")

    def fidelity(self, fid):
        plt, np = self.plt, __import__("numpy")
        core = list(self.cfg.core)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
        xs = np.arange(len(core))
        for k, (key, lab) in enumerate([("bias_start_ms", "start"), ("bias_end_ms", "end")]):
            vals = [fid[s]["mfa"][key] for s in core]
            axes[0].bar(xs + (k - 0.5) * 0.36, vals, 0.34, color=[self.col[s] for s in core],
                        alpha=1.0 if k == 0 else 0.55, edgecolor="white")
            for x, v in zip(xs + (k - 0.5) * 0.36, vals):
                axes[0].text(x, v + (0.6 if v >= 0 else -0.6), f"{v:.1f}", ha="center",
                             va="bottom" if v >= 0 else "top", fontsize=5.8)
        axes[0].axhline(0, color="#2b2a27", lw=0.7); axes[0].set_ylim(-26, 4)
        axes[0].set_title("bias of rendered word times (ms)\nsolid = starts, light = ends; 0 = on time", fontsize=7.4)
        vals = [fid[s]["mfa"]["mae_unbiased_ms"] for s in core]
        axes[1].bar(xs, vals, 0.6, color=[self.col[s] for s in core], edgecolor="white")
        for x, v in zip(xs, vals):
            axes[1].text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=6.4)
        axes[1].set_title("scatter: MAE with the bias removed (ms)", fontsize=7.4)
        for ax in axes:
            ax.set_xticks(xs, [self.lab[s].replace(" (", "\n(") for s in core], fontsize=6.3)
            ax.grid(axis="x", visible=False)
        fig.tight_layout()
        return self.save(fig, "fidelity")

    def phone_dynamics(self, pd, gt):
        plt = self.plt
        order = [s for s in ["m1", "m1_cont", "m1_pos", "m1_pos_diph", "abl_a_pos", "abl_b_pos"] if s in pd]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2))
        r = [pd[s]["inside"] / gt["inside"] for s in order]
        o = [pd[s]["onset_over_inside"] for s in order]
        for ax, vals, ref, title in [(axes[0], r, 1.0, "movement inside phones ÷ LJ recordings"),
                                     (axes[1], o, gt["onset_over_inside"], "onset ÷ inside movement")]:
            ax.barh(range(len(order)), vals, 0.6, color=[self.col.get(s, "#8a8984") for s in order], edgecolor="white")
            ax.axvline(ref, color="#e34948", lw=1, ls="--")
            ax.text(ref, -0.75, f" LJ recordings = {ref:.2f}", color="#e34948", fontsize=6.4)
            for y, v in enumerate(vals):
                ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=6.2)
            ax.set_yticks(range(len(order)), order, fontsize=6.6)
            ax.set_xlim(0, max(max(vals), ref) * 1.18); ax.set_title(title, fontsize=7.4)
            ax.grid(axis="y", visible=False); ax.set_ylim(len(order) - 0.5, -1.1)
        fig.tight_layout()
        return self.save(fig, "phone_dynamics")


# --------------------------------------------------------------------------- PDF (reportlab)
def build_pdf(cfg, out):
    from matplotlib import get_data_path
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.fonts import addMapping
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
        canv.setFont("Sans", 7); canv.setFillColor(MUTED)
        canv.drawString(2.1 * cm, A4[1] - 1.25 * cm, "meeami-tts · time-aligned US-English TTS · project report")
        canv.drawRightString(A4[0] - 2.1 * cm, A4[1] - 1.25 * cm, "code frozen at git tag freeze-v1 (5889ecf)")
        canv.setStrokeColor(RULE); canv.setLineWidth(0.5)
        canv.line(2.1 * cm, A4[1] - 1.4 * cm, A4[0] - 2.1 * cm, A4[1] - 1.4 * cm)
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
    def T(rows, widths, bold_rows=(), align_right_from=1, zebra=True, header=True):
        data = [[Paragraph(str(c), cellb if (header and i == 0) or i in bold_rows else cell) for c in r]
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
        t.setStyle(TableStyle(st))
        S.append(t); S.append(Spacer(1, 7))
    def F(path, caption, width=1.0):
        from PIL import Image as PImage
        w, h = PImage.open(path).size
        img = Image(str(path), width=width * W, height=width * W * h / w)
        S.append(KeepTogether([img, Paragraph(caption, cap)]))
    fig_no = [0]
    def fig_caption(text):
        fig_no[0] += 1
        return f"<b>Figure {fig_no[0]}.</b> {text}"
    tab_no = [0]
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
    choice_name = choice["choice"]["name"]
    comps = [(f"{sp}: {cfg.labels[a].split(' (')[0]} − {cfg.labels[b].split(' (')[0]}", paired(sp, a, b))
             for sp, a, b in cfg.paired]
    M = lambda s, k, g="all": sums[s]["metrics"][g][k]["mean"]  # noqa: E731
    lab = dict(cfg.labels)
    n_main = [b[1] for b in bins[main]]
    lj = sums[main]["lj_test_ground_truth"]

    f_pipe, f_head = figs.pipeline(), figs.headline(ROOT / cfg.headline_sample)
    f_cdf, f_acc = figs.cdf(rows), figs.per_accent(sums)
    f_str, f_tun = figs.stretch(bins, edges, n_main), figs.tuning(grid, choice_name)
    f_for, f_fid, f_pd = figs.forest(comps), figs.fidelity(fid), figs.phone_dynamics(pdyn, gt)
    reuse = {k: ROOT / v for k, v in cfg.reuse_plots.items()}
    head = json.loads((ROOT / cfg.headline_sample / "timings.json").read_text())

    # ---------------- cover
    S.append(Spacer(1, 2.2 * cm))
    S.append(Paragraph("meeami-tts", ParagraphStyle("t0", fontName="Sans-B", fontSize=30, leading=36, textColor=ACCENT)))
    S.append(Paragraph("Time-aligned US-English text-to-speech<br/>on the word timing of a non-US reference",
                       ParagraphStyle("t1", fontName="Sans", fontSize=16, leading=21, textColor=INK, spaceBefore=6)))
    S.append(Spacer(1, 0.5 * cm))
    S.append(Paragraph("Project report · Chaganti Raja Ravi Teja · November 2026",
                       ParagraphStyle("t2", fontName="Sans", fontSize=10, leading=14, textColor=MUTED)))
    S.append(Paragraph(f"Code: <a href='{cfg.links.github}' color='#2a78d6'>{cfg.links.github}</a> (frozen at git tag "
                       f"<font face='Mono'>freeze-v1</font>, commit 5889ecf)<br/>Models and all test outputs: "
                       f"<a href='{cfg.links.hf}' color='#2a78d6'>{cfg.links.hf}</a>",
                       ParagraphStyle("t3", fontName="Sans", fontSize=8.5, leading=12, textColor=MUTED, spaceBefore=4)))
    S.append(Spacer(1, 0.8 * cm))
    tiles = [(f"{M(main, 'wb_mae_ms'):.1f} ms", f"word-boundary error\n(V0, original VITS: {M('v0', 'wb_mae_ms'):.1f} ms)"),
             ("0 samples", "length error on all\n1,248 test references"),
             (f"{100 * M(main, 'wer'):.1f}%", f"WER of the output\n(L2 references: {100 * M(main, 'wer_ref'):.1f}%)"),
             (f"{M(main, 'utmos'):.2f}", f"UTMOS naturalness\n(LJSpeech recordings: {lj['utmos']:.2f})")]
    tstyle = ParagraphStyle("tv", fontName="Sans-B", fontSize=17, leading=21, textColor=ACCENT, alignment=TA_CENTER)
    tlab = ParagraphStyle("tl", fontName="Sans", fontSize=7.4, leading=9.6, textColor=MUTED, alignment=TA_CENTER)
    tt = Table([[Paragraph(v, tstyle) for v, _ in tiles], [Paragraph(l.replace("\n", "<br/>"), tlab) for _, l in tiles]],
               colWidths=[W / 4] * 4)
    tt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LINEBEFORE", (1, 0), (-1, -1), 2, colors.white),
                            ("TOPPADDING", (0, 0), (-1, 0), 10), ("BOTTOMPADDING", (0, -1), (-1, -1), 10)]))
    S.append(tt)
    S.append(Spacer(1, 0.4 * cm))
    S.append(Paragraph("Held-out test accents (Arabic, Korean, Vietnamese; 12 speakers × 104 sentences), evaluated once "
                       "after the code freeze.", cap))
    S.append(Spacer(1, 0.4 * cm))
    S.append(Paragraph("Abstract", h3))
    P("Given an English text and a recording of it by a non-US speaker, the system speaks the text in the LJSpeech US "
      "voice and accent while following the recording's timing: the output is exactly as long as the reference and every "
      "word starts and ends where the speaker said it, pauses included. The reference contributes timing only. "
      "The approach makes durations an <i>input</i>: VITS is fine-tuned from the official LJSpeech checkpoint with "
      "monotonic alignment search (MAS) replaced by Montreal Forced Aligner (MFA) phone durations. At inference, MFA "
      "finds the reference's word boundaries, and each word's span is filled with the model's own native phone rhythm "
      "(one elastic factor per word, a 2.5× stretch cap, exact cumulative rounding). "
      f"On {sums[main]['n']:,} held-out test references, word boundaries land {M(main, 'wb_mae_ms'):.1f} ms from the human "
      f"annotation (MFA's own error on accented speech is 19.4 ms), against {M('v0', 'wb_mae_ms'):.1f} ms for the original "
      f"VITS given the same timing and {M('m0', 'wb_mae_ms'):.1f} ms for FastSpeech2. Two diagnosed defects were fixed "
      "with minimal, controlled changes (a position-in-phone embedding and a two-token diphthong); two pace-augmentation "
      "ablations are reported as negative results. All decisions were fixed on dev accents before the test accents were read.")
    S.append(NextPageTemplate("main")); S.append(PageBreak())

    # ---------------- contents
    S.append(Paragraph("Contents", ParagraphStyle("ct", parent=h1)))
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("toc0", fontName="Sans-B", fontSize=9.5, leading=14, leftIndent=0),
                       ParagraphStyle("toc1", fontName="Sans", fontSize=8.5, leading=11.5, leftIndent=14, textColor=MUTED)]
    S.append(toc)
    S.append(Spacer(1, 10))
    P("<b>How to read this report.</b> Tables and figures are generated from the committed result files by "
      "<font face='Mono'>scripts/build_report_pdf.py</font> (<font face='Mono'>make report-pdf</font>). "
      "<font face='Mono'>D-n</font> / <font face='Mono'>E-n</font> refer to entries of the decision and error log "
      "<font face='Mono'>docs/DECISIONS.md</font>; <font face='Mono'>docs/REPORT.md</font> holds the same material "
      "with a link to the exact line of code behind every claim. Audio for every claim is in "
      "<font face='Mono'>samples/</font> (byte-exact copies of the evaluated files).", small)
    S.append(PageBreak())

    # ================= 1 Introduction
    P("1. Introduction", h1)
    P("1.1 Task", h2)
    P("<b>Inputs:</b> an English text and a recording of that text read by a <b>non-US</b> speaker (the <i>reference</i>). "
      "<b>Output:</b> speech that (1) says exactly the input words with a US accent, (2) is in the LJSpeech voice "
      "(one US female speaker) and (3) follows the reference's timing: the same total length, every word starting "
      "and ending at about the reference time, pauses included. Voice, accent and pronunciation of the reference must not leak.")
    P("<b>Why it is hard.</b> Part of an accent lives in timing: L2 speakers lengthen, shorten and pause differently. "
      "The system must copy <i>word-level</i> timing from the reference while keeping <i>phone-level</i> timing inside "
      "each word native, and it must stay natural when a word is forced to be much slower or faster than the voice "
      "would say it.")
    P("1.2 Contributions", h2)
    B(["<b>Durations as an input.</b> VITS fine-tuned with MFA durations instead of MAS (VITS-MFA), so a reference can "
       "dictate timing; measured against the unmodified VITS (V0) given the same timing: "
       f"{M('v0', 'wb_mae_ms'):.1f} → {M(main, 'wb_mae_ms'):.1f} ms word-boundary error.",
       "<b>Word-timing transfer with native rhythm</b> (elastic fit, stretch cap, exact-length quantization): output "
       "length is exact for every reference by construction.",
       "<b>Two diagnosed fixes</b> from listening reports, each with a pre-stated success rule and a one-key control: a "
       "position-in-phone embedding (US probability +0.09 on dev) and a two-token diphthong (\"cry\" heard as \"cra\"; "
       "test WER 4.4% → 2.9%).",
       "<b>Two pace-augmentation ablations</b> (target-agnostic A, target-aware B) reported as negative results.",
       "<b>An accent-disjoint evaluation</b> with a test lock enforced in code, a paired-statistics protocol, and "
       "probes that separate the model's rendering error from the aligner's error."])
    P("1.3 Constraints", h2)
    T([["Constraint", "Consequence for the design"],
       ["Train on LJSpeech only", "Voice and accent come from one US speaker; L2 data is used only for evaluation, tuning and Ablation B's statistics."],
       ["Generalise to unseen accents", "L2-ARCTIC split by accent; test accents locked in code until git tag freeze-v1 exists."],
       ["One rented L40S GPU for ~8 h, an 8 GB laptop GPU", "M1 trained once on the full budget; every later experiment branches from it for 20k steps."],
       ["A live demo on unseen files", "Edge-case suite, CTC aligner fallback, exact output length by construction, typed-text pre-pass."]],
      [0.32, 0.68])

    # ================= 2 Data
    P("2. Data and the accent-disjoint split", h1)
    T([["Dataset", "Use"],
       ["LJSpeech 1.1: 13,100 clips, one US female speaker, 22.05 kHz", "Training, with the original VITS filelists (12,500 / 100 / 500), so the pretrained checkpoint never saw val or test. MFA kept 13,073 (99.79%)."],
       ["L2-ARCTIC v5: 24 speakers, 6 L1s × 4", "References for dev and test; its hand-corrected word tiers are the timing ground truth."],
       ["Dev accents: Hindi, Mandarin, Spanish", "Tuning, phase-2 decisions, Ablation B's pace statistics. 100 sentences × 12 speakers = 1,200 references."],
       ["Test accents: Arabic, Korean, Vietnamese", "Read once, after freeze-v1. 104 sentences × 12 speakers = 1,248 references."]],
      [0.4, 0.6])
    tab_caption("Data. Evaluation uses the utterances that have a manual annotation for every speaker of the split.")
    P("<b>Why split by accent.</b> The claim to support is \"works for accents the system was never tuned on\"; a "
      "sentence-level split would let tuning see every accent. <b>The lock is code, not convention:</b> reading a test "
      "speaker's audio raises <font face='Mono'>SplitLockedError</font> unless <font face='Mono'>git rev-parse "
      "refs/tags/freeze-v1</font> succeeds, and a unit test fills fake test WAVs with junk bytes so any premature read fails.")

    # ================= 3 Method
    P("3. Method", h1)
    P("3.1 Design principle: one source per output property", h2)
    T([["Output property", "Its only source", "Deliberately not taken from"],
       ["Words", "input text → normalizer", "the reference audio or an ASR transcript"],
       ["Pronunciation", "data/lexicon/lexicon.txt (first CMUdict entry; g2p_en for OOV)", "the reference's pronunciation"],
       ["Voice, accent, intonation", "the model, trained on LJSpeech only", "the reference (no accented audio in training)"],
       ["Word start/end times, pauses, length", "the reference, aligned with MFA", "the model's own durations"],
       ["Phone timing inside a word", "the model's native durations, fitted to the word span", "the reference's phone timing, which carries accent"]],
      [0.27, 0.40, 0.33])
    tab_caption("Nothing is averaged between reference and model, so each failure has one place to look. Evidence for the "
                "last row: copying the reference's phone durations gave the worst WER of the tuning grid (10.6% vs 2.5%; §5.1).")
    F(f_pipe, fig_caption("Inference pipeline. Timing enters only through the reference's word boundaries; inside each "
                          "word the model's native phone rhythm is stretched with LJ's own elasticity."))

    P("3.2 Text front end", h2)
    B(["<b>One normalizer</b> for training text, MFA transcripts and inference input (currency → symbols → unidecode → "
       "dates → times → abbreviations → numbers); any digit left over raises instead of being dropped.",
       "<b>Lexicon:</b> 14,912 words, one pronunciation each: 13,640 from CMUdict, 1,168 from g2p_en, 104 from a possessive "
       "rule (g2p_en mangled <i>x's</i> when <i>x</i> is in CMUdict). Unseen words at inference are G2P'd and appended to "
       "the same file, so it stays the only pronunciation source.",
       "<b>Symbols:</b> pad, sil, sp + 69 ARPAbet phones = 72, exactly MFA's english_us_arpa phone set (asserted by a "
       "test), so every model token has an MFA duration.",
       "<b>Typed-text pre-pass</b> (added after the freeze for live input): money, times, dates, ratios, units, URLs, "
       "e-mails, Roman numerals, CamelCase, initialisms and vowel-less letter strings, plus refusals for non-Latin "
       "script. 119 of 121 stress cases pass; a test asserts that every L2-ARCTIC transcript gives the same words as before."])

    P("3.3 Forced alignment and its error on accented speech", h2)
    P("MFA 3.4.3 with the pretrained english_us_arpa model and our lexicon aligns LJSpeech (training durations), the L2 "
      "references (the timing to copy) and the model outputs (the measured timing), with one set of parsing rules, so "
      "alignment conventions cancel in comparisons. Wide beam (100 / retry 400); MFA silently drops unknown flags, so the "
      "wrapper validates every flag. Boundaries are rounded on cumulative times (no drift), internal silences ≥ 3 frames "
      "become <font face='Mono'>sp</font>, and every speech phone gets ≥ 1 frame (382 phones had 0 frames otherwise).")
    T([["Dev accent", "Word MAE", "Word ≤ 20 ms", "Phone MAE", "Phone ≤ 20 ms"],
       ["Hindi", "18.2 ms", "76.8%", "12.3 ms", "89.0%"], ["Mandarin", "20.2 ms", "74.2%", "14.0 ms", "86.7%"],
       ["Spanish", "19.9 ms", "74.0%", "13.2 ms", "87.6%"], ["All (1,800 utterances)", "19.4 ms", "75.0%", "13.1 ms", "87.7%"]],
      [0.32, 0.17, 0.17, 0.17, 0.17], bold_rows=(4,))
    tab_caption("MFA against L2-ARCTIC's hand-corrected tiers (reports/mfa_l2_dev.md). The system metric aligns the output "
                "with MFA and compares with the manual reference annotation, so about 19 ms of MFA error is part of every "
                "system's score: 23.9 ms is close to that measurement floor. Fallback aligner: MMS_FA CTC (~53 ms MAE).")
    F(reuse["mfa_error"], fig_caption("MFA's word and phone boundary error on the dev accents against the manual annotation."), 0.9)

    P("3.4 Models", h2)
    P("<b>V0, the immediate baseline</b> (inference only): the official pretrained LJSpeech VITS exactly as released "
      "(MAS-learned alignment, stochastic duration predictor, blank tokens, IPA characters). It is the checkpoint M1 is "
      "fine-tuned from, so V0 vs M1 measures the MAS → MFA swap. Lexicon phones are written in espeak-style IPA; its "
      "predicted per-character durations give each phone's native rhythm, and the same word-timing fit drives it. Its one "
      "adapter choice (which phone owns a blank) was made on dev (blank → next character: 40.7 vs 50.8 ms).")
    P("<b>M0, the second baseline</b> (inference only): ming024's pretrained FastSpeech2 + HiFi-GAN; reference durations "
      "enter as d_targets through the variance adaptor's training path.")
    P("<b>M1, VITS-MFA</b>: jaywalnut310/vits fine-tuned from the official checkpoint (553 matching tensors loaded) with six changes:")
    T([["#", "Change", "Reason"],
       ["1", "Remove MAS; alignment = one-hot expansion of MFA durations", "Durations must be controllable at inference; MAS learns its own alignment, which a reference cannot override."],
       ["2", "No blank tokens", "Blanks have no MFA duration."],
       ["3", "72 ARPAbet symbols; text embedding re-initialized", "The pretrained model read 178 IPA characters."],
       ["4", "Deterministic duration predictor on the detached encoder, MSE on log frames, masked on sil", "Only native relative durations inside a word are needed; sampling would add random rhythm."],
       ["5", "Optional position-in-phone embedding (16 bins)", "Lets the prior change inside a phone (phase 2a)."],
       ["6", "Data loader on the project manifests", "One manifest format for all data."]],
      [0.05, 0.47, 0.48])
    tab_caption("PLAN changes from VITS to VITS-MFA. Posterior encoder, flow, HiFi-GAN decoder and the VITS losses are unchanged.")
    P("L<sub>G</sub> = 45·L<sub>mel</sub> + L<sub>KL</sub> + L<sub>dur</sub> + L<sub>adv</sub> + 2·L<sub>fm</sub>,   "
      "L<sub>dur</sub> = mean<sub>i</sub> (log d̂<sub>i</sub> − log d<sub>i</sub>)²", eq)
    P("Error on the log scale is relative (2× off costs the same for a 3-frame stop and a 20-frame vowel) and its minimizer "
      "is the geometric-mean duration, the natural centre for the multiplicative fit of §3.5. <b>Training:</b> 114,000 steps "
      "at batch 32 on one L40S (~8 h; bf16 networks, fp32 losses). val/dur fell throughout (0.134 → 0.094); val mel-L1 "
      "reached its minimum at 12k and rose slightly, the GAN trade-off rather than overfitting (train and val drifted "
      "together, discriminator balanced). Phase-2 variants branch from M1 for 20,000 steps each with a fresh optimizer; "
      "new parameters start so that every variant <i>is</i> M1 at step 0.")
    F(reuse["training"], fig_caption("M1 training curves (phase 1, 114k steps)."))
    T([["System", "Difference from M1 (one key each, asserted by tests)"],
       ["m1_cont", "+20k steps, no change: the control for \"more training\""],
       ["m1_pos", "use_pos: true (position-in-phone embedding, zero-initialized)"],
       ["abl_a_pos", "m1_pos + 40% Ablation A clips (target-agnostic pace augmentation)"],
       ["abl_b_pos", "m1_pos + 40% Ablation B clips (pace statistics of the dev accents)"],
       ["m1_pos_diph (main)", "m1_pos + diph_split: each diphthong X becomes X + X+ (87 symbols)"]],
      [0.25, 0.75], bold_rows=(5,))
    tab_caption("Phase-2 variants: one start point, one step count, one optimizer reset; each pair differs in one setting.")
    P("<b>What a fixed external alignment takes away.</b> MAS re-chooses the alignment every step from the model's own "
      "prior and flow output; MFA's alignment is fixed by an HMM-GMM that never sees the latent space. The gradients through "
      "the KL term are unchanged (MAS is not differentiable), but the feedback from latent space to alignment is lost: "
      "boundary errors stay in the KL for all of training, and with no blanks each phone has one constant prior over all its "
      "frames. Measured: every MFA-trained model moves 14-16% less inside phones than the LJ recordings (Figure "
      f"{fig_no[0] + 1}). What it buys is control: the fine-tuned systems render given durations about on time (§5.6).")
    F(f_pd, fig_caption("Phone dynamics on 100 LJ val clips resynthesized with ground-truth durations: mean log-mel "
                        "movement inside phones relative to the LJ recordings, and how much movement is piled up at phone onsets."))

    P("3.5 Inference: transferring word timing", h2)
    P("For a word of D reference frames, the duration predictor gives native durations d̂<sub>i</sub> for its phones, and "
      "every phone is scaled with one z per word, found by bisection:")
    P("d<sub>i</sub> = exp(log d̂<sub>i</sub> + z·σ<sub>i</sub>),   Σ<sub>i</sub> d<sub>i</sub> = D", eq)
    P("σ<sub>i</sub> is the phone's standard deviation of log duration in LJ train, so phones LJ naturally stretches "
      "(vowels) absorb most of the change and stops barely move: the word fills the reference's slot the way the LJ "
      "speaker would say it slower or faster. <i>Example</i>, \"cry\" at twice its native length: proportional scaling "
      "gives K/R/AY ×2.0 each; elastic gives ×1.75 / ×2.4 / ×2.0.")
    B(["<b>Stretch cap.</b> Beyond 2.5× native length a word is not stretched further; the excess becomes a pause and the "
       "next word still starts on time. It acts on 148 words in 128 of 1,248 test references.",
       "<b>Exact length.</b> Cumulative rounding to exactly round(N<sub>ref</sub>·22050/sr<sub>ref</sub>/256) frames "
       "(each word boundary within half a frame), then pad/trim to the exact sample count. Length error is an asserted "
       "invariant, not a statistic.",
       "<b>Diphthong split at the model boundary only.</b> m1_pos_diph splits each diphthong right before the model and "
       "merges the two predicted durations afterwards; lexicon, MFA and pipeline keep 72 symbols."])

    # ================= 4 Evaluation
    P("4. Evaluation methodology", h1)
    T([["Requirement", "Metric", "Definition"],
       ["Exact length", "Length error", "output samples − round(N_ref·22050/sr_ref); must be 0"],
       ["Word timing", "Word-boundary MAE; % within 20 / 50 ms", "every word's start and end; reference times from the <b>manual</b> annotation, output times from MFA on the output; words matched by sequence alignment"],
       ["Pauses", "Pause F1", "pause = silence ≥ 100 ms between two matched words; F1 over word boundaries"],
       ["Exact words", "WER, CER", "faster-whisper large-v3 vs input text, both through Whisper's EnglishTextNormalizer; also run on the references and LJ"],
       ["US accent", "US probability", "CommonAccent ECAPA accent-ID, softmax(30·cosine)"],
       ["LJ voice", "Speaker similarity", "ECAPA cosine to the mean of 100 LJ test clips (high) and to the reference speaker (low)"],
       ["Naturalness", "UTMOS", "SpeechMOS utmos22_strong; LJ recordings are the ceiling"],
       ["Robustness", "Stretch ratio", "speech frames ÷ the model's native frames, for binning"]],
      [0.17, 0.27, 0.56])
    tab_caption("Metrics. All evaluation models are pinned by version and sha256 and loaded offline.")
    B(["<b>Statistics.</b> 95% percentile bootstrap CIs over utterances (1,000 resamples). Paired Wilcoxon signed-rank tests "
       "of each system against its own control, overall, per accent and per stretch tercile; Holm-adjusted within each "
       "family of 8 metrics. Every system speaks the same references, so paired differences remove between-sentence variance.",
       "<b>Decision rules before results.</b> The tuning rule, the phase-2a winner rule and the diphthong success criteria "
       "were written in DECISIONS.md before each run.",
       "<b>Probes.</b> Duration fidelity (where a word lands in the output vs where the pipeline put it: separates the model's "
       "rendering error from the shared reference-alignment error), aligner calibration on natural dev speech, phone "
       "dynamics, an LPC formant probe for diphthong glides, and the reference pace ρ."])
    P("<b>Caveats, stated rather than hidden.</b>", h3)
    B(["Accent-ID US probability is near-binary per utterance (softmax ×30); it flips on inaudible changes, so it is read "
       "only as a population mean.",
       "MFA judges systems trained on MFA durations, which may favour them. CTC, the second judge, has a +38 ms start bias on "
       f"natural speech (calibration: MFA bias {calib['mfa']['bias_start_ms']:.1f} ms, CTC {calib['ctc']['bias_start_ms']:+.1f} ms), "
       "so CTC numbers are read against CTC's own zero point, never as a ranking.",
       "UTMOS chose the augmentation stretcher's parameters (on LJ only) and is also an evaluation metric.",
       "M1's dev numbers are slightly optimistic: 120 of 1,200 dev references were used for tuning. Test has no such bias."])

    # ================= 5 Experiments and ablations
    P("5. Experiments and ablations", h1)
    P("Each phase-2 round started from a listening observation, checked the input first, measured the defect with a "
      "targeted probe, found the design cause, made a minimal opt-in fix with a one-key control, and re-measured with the "
      "same probe plus the full metric set. All decisions below were made on dev.", small)
    P("5.1 Inference tuning (dev)", h2)
    P("21 settings on a fixed 120-utterance dev subset: transfer variant (elastic, proportional, phone_copy), stretch cap, "
      "prior noise scale and aligner. Rule, fixed before running: among elastic settings within 5 ms of the best MAE, take "
      "the highest UTMOS. Chosen: elastic, cap 2.5, noise 0.333, MFA (21.9 ms; best 21.5 ms; UTMOS 4.21; WER 2.5%).")
    F(f_tun, fig_caption("Tuning grid. The noise scale drives WER (0.333: 2.5-3.5%; 0.667: 4.6-6.7%), a looser cap lowers "
                         "MAE at no UTMOS cost, and copying the reference's phone durations (phone_copy) has by far the worst WER: the "
                         "reference's phone timing carries accent, so only word boundaries are copied."))

    P("5.2 Phase 1: M1 vs M0 (dev)", h2)
    T([["Dev, 1,200 references", "M0 (FastSpeech2)", "M1 (VITS-MFA)"],
       ["Word-boundary MAE", "31.3 ms", "<b>23.7 ms</b> (p ≈ 1e-140)"], ["Within 20 ms", "55.4%", "<b>68.5%</b>"],
       ["Pause F1", "0.75", "<b>0.84</b>"], ["WER", "10.0%", "<b>3.5%</b>"], ["UTMOS", "3.94", "<b>4.20</b>"],
       ["US probability", "0.73", "0.72 (n.s.)"]], [0.4, 0.3, 0.3])
    tab_caption("Both systems receive the same durations; M1 renders them more crisply. US probability did not improve, which set up phase 2.")

    P("5.3 Phase 2a: \"words end sharply\" → position-in-phone embedding", h2)
    P("Probes ruled out energy, pitch and the duration predictor at word ends; what differed was spectral movement "
      "<i>inside</i> phones (14% below LJ, 20% in the middle third). Cause: without blanks every frame of a phone shares one "
      "constant prior mean. Fix: frame k of a d-frame phone gets bin ⌊16(k + 0.5)/d⌋ and a learned vector is added before "
      "the prior projection. <b>Result vs m1_cont</b> (the same 20k steps without the change): within-phone movement did not "
      "move (0.856× vs 0.858× LJ; the learned effect sits at phone edges, which the probe excludes), but US probability "
      "rose 0.680 → 0.769 (p = 5e-31, higher in all three accents) with no significant change elsewhere. m1_pos became the "
      "base for the next rounds.")

    P("5.4 Phase 2b: pace augmentation, Ablations A and B (negative result)", h2)
    P("<b>Hypothesis:</b> training on time-stretched LJ teaches the model to stay natural at high stretch ratios. "
      "<b>A (target-agnostic):</b> utterance pace z<sub>u</sub> ~ U(−1.5, 1.5), word pace z<sub>w</sub> = z<sub>u</sub> + "
      "N(0, 0.5²), the same elastic factors as inference, longer and inserted pauses from room tone. <b>B (target-aware):</b> "
      "word paces and pauses sampled from dev-accent statistics (AR(1) residuals, P(pause) = 0.223); word-level ratios only. "
      "2 copies per LJ clip mixed 60/40; the duration loss is masked on augmented rows; the 5% worst clips by UTMOS drop removed.")
    P("<b>A forced deviation:</b> the planned Rubber Band R3 stretcher degraded every clip (UTMOS 4.38 → 2.92, ~1.2 even "
      "for &lt; 5% stretch), so a time-varying WSOLA stretcher with the same time map was written in numpy (mean UTMOS "
      "drop 1.336 → 0.321; re-alignment matched the planned boundaries to a median ~7 ms).")
    T([["Dev, paired vs m1_pos", "m1_pos", "abl_a_pos", "abl_b_pos"],
       ["Word-boundary MAE", "24.16 ms", "24.04 (n.s.)", "24.21 (n.s.)"], ["WER", "3.9%", "3.4% (n.s.)", "3.9% (n.s.)"],
       ["US probability", "0.769", "<b>0.642</b> (p &lt; 1e-10)", "0.747 (n.s.)"],
       ["UTMOS", "4.18", "<b>4.07</b> (p &lt; 1e-10)", "<b>4.10</b> (p &lt; 1e-5)"]], [0.31, 0.2, 0.25, 0.24])
    tab_caption("Neither ablation improved timing or WER in any stretch tercile, both cost UTMOS, and B shows no gain even on "
                "the accents its statistics came from. Likely cause: residual stretch artifacts in 40% of batches. A's pace range "
                "is far wider than real L2 pace, hence its US-probability loss.")

    P("5.5 Phase 2c: \"cry\" heard as \"cra\" → two-token diphthong", h2)
    P("The lexicon gives cry → K R AY1, so the error was the model's. An LPC probe showed the reference gliding to /ɪ/ "
      "(F2 +285 Hz, F1 737 → 554) while M1, m1_cont and m1_pos held an /a/; across LJ val every model flattened diphthongs "
      "(mean |ΔF2| 307-321 Hz vs 396 for LJ). <b>Cause: token design.</b> The pretrained VITS read \"cry\" as "
      "k _ ɹ _ ˈ _ a _ ɪ: a diphthong was two targets and a transition; ARPAbet made it one token with one constant prior. "
      "<b>Fix:</b> AW, AY, EY, OW, OY (× 3 stresses) become X + X+ (72 → 87 symbols); MFA's duration is split 50/50 and "
      "merged back as log(e<super>a</super> + e<super>b</super>) for timing; each X+ embedding starts as a copy of X.")
    T([["Dev, paired vs m1_pos", "m1_pos", "m1_pos_diph", "p (Holm)"],
       ["Word-boundary MAE", "24.16 ms", "<b>23.48 ms</b>", "&lt; 1e-6"], ["WER", "3.9%", "<b>2.7%</b>", "&lt; 1e-6"],
       ["CER", "2.0%", "<b>1.2%</b>", "&lt; 1e-6"], ["UTMOS", "4.18", "<b>4.20</b>", "2e-4"],
       ["US probability", "<b>0.769</b>", "0.738", "1e-4"], ["Mean |ΔF2| of diphthongs (LJ 396 Hz)", "321 Hz", "<b>389 Hz</b>", ""]],
      [0.4, 0.2, 0.2, 0.2])
    tab_caption("\"cry\" now glides (F1 864 → 528 Hz, F2 +215 Hz). The one unmet criterion, US probability, was investigated: "
                "it does not correlate with the number of diphthongs, sentences without any dropped too, the classifier flips "
                "both ways (91 large drops vs 58 rises), and the four largest drops sounded identical to the listener. "
                "Decision (D-85): m1_pos_diph is the main system; then the code was frozen and the test split read once.")

    P("5.6 The immediate baseline V0 and duration fidelity", h2)
    P("V0 is the same pretrained VITS before the MAS → MFA swap, given the same word timing through the same pipeline. "
      "To see <i>why</i> it misses the timing, the fidelity probe compares where MFA finds each word in the output with "
      "where the pipeline placed it, which removes the shared error of aligning the reference.")
    F(f_fid, fig_caption("Duration fidelity on test (MFA on the outputs vs the planned word times). V0 renders its given "
                         "durations ~20 ms early and scattered; the fine-tuned systems render them about on time. CTC, an independent "
                         "judge read against its own zero point, finds the same early shift for V0."))
    P("Two judges agree on the direction and size of V0's shift; CTC's own scatter is too large to rank the scatter, so "
      "that part rests on MFA. <b>The trade-off, stated plainly:</b> V0 is the stronger system on WER and UTMOS; "
      "fine-tuning on MFA durations without blanks bought controllable timing and a more US-like accent, at a cost the "
      "diphthong split already halved (M1 − V0: +1.7 WER points; main − V0: +0.8).")
    F(f_for, fig_caption("Every paired comparison of the project, as mean difference (system − control) per utterance."))

    # ================= 6 Results
    P("6. Results on the held-out test accents", h1)
    P(f"Arabic, Korean and Vietnamese, 12 speakers, {sums[main]['n']:,} references; read once after freeze-v1. All "
      f"{sums[main]['aligner_counts']['mfa']:,} references were aligned by MFA (no CTC fallback). Length error is 0 for "
      "every output of every system.")
    hdr = ["System", "Word MAE (ms) [95% CI]", "≤ 20 ms", "Pause F1", "WER", "US prob", "UTMOS", "Spk sim LJ"]
    trows = [hdr, ["L2 references", "", "", "", f"{100 * M(main, 'wer_ref'):.1f}%", f"{M(main, 'us_prob_ref'):.3f}", "", ""],
             ["LJSpeech recordings", "", "", "", f"{100 * lj['wer']:.1f}%", "", f"{lj['utmos']:.2f}", ""]]
    for s in systems:
        mm = sums[s]["metrics"]["all"]
        ci = mm["wb_mae_ms"]["ci95"]
        trows.append([lab[s], f"{mm['wb_mae_ms']['mean']:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]", f"{mm['wb_within20']['mean']:.1f}%",
                      f"{mm['pause_f1']['mean']:.3f}", f"{100 * mm['wer']['mean']:.1f}%", f"{mm['us_prob']['mean']:.3f}",
                      f"{mm['utmos']['mean']:.2f}", f"{mm['spk_sim_lj']['mean']:.3f}"])
    T(trows, [0.2, 0.2, 0.09, 0.1, 0.08, 0.1, 0.1, 0.13], bold_rows=(3 + systems.index(main),))
    tab_caption("Main results (results/&lt;system&gt;/test/summary.json). Speaker similarity to the reference speaker stays "
                f"low ({M(main, 'spk_sim_ref'):.2f} for the main system): the reference's voice does not leak.")
    pr = [["Paired comparison (test)", "Δ MAE (ms)", "Δ ≤ 20 ms (pt)", "Δ WER (pt)", "Δ UTMOS", "Δ US prob"]]
    for sp, a, b in cfg.paired:
        if sp != "test":
            continue
        r = paired(sp, a, b)
        def c(m, sc=1, d=1):
            v = float(r[m]["mean_diff"]) * sc
            p = float(r[m]["p_holm"])
            return f"{v:+.{d}f}" + (f" (p {fmt_p(p)})" if p < cfg.alpha else " (n.s.)")
        pr.append([f"{lab[a].split(' (')[0]} − {lab[b].split(' (')[0]}", c("wb_mae_ms", 1, 1), c("wb_within20", 1, 1),
                   c("wer", 100, 1), c("utmos", 1, 2), c("us_prob", 1, 3)])
    T(pr, [0.26, 0.15, 0.15, 0.15, 0.14, 0.15])
    tab_caption("Paired Wilcoxon tests on the 1,248 test references, Holm-adjusted (results/paired/test_*.csv).")
    P("<b>Findings.</b>", h3)
    B([f"<b>Replacing MAS with MFA durations is what makes the output follow the reference timing:</b> main vs V0, "
       f"−16.5 ms MAE and +28 points within 20 ms, in every accent and every stretch range; the cost is +0.8 WER points and "
       f"−0.12 UTMOS.",
       "<b>VITS-MFA beats FastSpeech2 on every axis but accent</b> (M1 vs M0: −8 ms MAE, WER 9.3% → 3.8%, UTMOS +0.23).",
       "<b>The diphthong split replicated on test</b> against its control: WER −1.5 points, MAE −0.7 ms, UTMOS +0.03.",
       "<b>Pace augmentation is a negative result on test too:</b> no timing or WER gain; UTMOS −0.11 (A), −0.09 (B).",
       "<b>Dev conclusions replicated on the unseen accents,</b> and the reference pace on dev (1.12× LJ) matched test (1.16×)."])
    F(f_head, fig_caption(f"The headline sample, {head['id']} (\"{head['text']}\"). Top: the L2 reference with its human "
                          "word annotation; below: the main system and V0, each with word times found by MFA on the output. Same "
                          "length by construction; the main system's words sit on the reference's word starts, V0's drift early. "
                          f"Audio: samples/demo/arabic/{head['id']}/."))
    F(f_cdf, fig_caption("Distribution of per-reference word-boundary MAE on test. The main system's whole distribution is "
                         "shifted left of V0 and M0, not just its mean; 90th percentiles: "
                         + ", ".join(f"{lab[s].split(' (')[0]} {quantile(mae_values(rows[s]), 0.9):.1f} ms" for s in core) + "."))
    F(f_acc, fig_caption("Per test accent, with 95% bootstrap CIs. The timing gain over V0 and M0 holds in every accent; "
                         "Korean speakers are easiest for every system."))
    T([["Accent", "Reference US prob"] + [f"{lab[s].split(' (')[0]} MAE / WER" for s in ("v0", "m1", main)] + ["main US prob"]]
      + [[a.capitalize(), f"{M(main, 'us_prob_ref', a):.3f}"]
         + [f"{M(s, 'wb_mae_ms', a):.1f} ms / {100 * M(s, 'wer', a):.1f}%" for s in ("v0", "m1", main)]
         + [f"{M(main, 'us_prob', a):.3f}"] for a in cfg.accents],
      [0.14, 0.15, 0.19, 0.19, 0.19, 0.14])
    tab_caption("Per accent, test. US probability rises from 0.41-0.60 for the references to 0.72-0.79 for the outputs.")
    F(f_str, fig_caption("Word-boundary MAE by stretch ratio (references binned by the main system's stretch ratio; "
                         f"{sum(n_main):,} references have one). Every system's error rises when the reference is far slower or "
                         "faster than LJ; the gap to V0 holds in every bin."))
    F(reuse["pace"], fig_caption("Reference pace relative to LJ on dev and test: dev was a fair stand-in for test."), 0.6)
    P("<b>Failures are shown, not hidden.</b> The worst reference per accent is in samples/demo/ (e.g. THV_arctic_a0067: "
      "148 ms for the main system and for V0 alike). The second-aligner cross-check (CTC on the outputs) gives 45.6-52.0 ms "
      "for all systems; its +38 ms start bias on real speech makes it unusable as a ranking (V0's early shift cancels part "
      "of that bias), which is why it is reported only against its own zero point (§5.6).")

    # ================= 7 Efficiency
    P("7. Efficiency", h1)
    T([["Quantity", "Value (main checkpoint)"],
       ["Inference parameters", "28.1 M: decoder 14.3 M, flow 7.1 M, text encoder 6.3 M, duration predictor 0.35 M"],
       ["Training-only parameters", "54.0 M: discriminators 46.7 M, posterior encoder 7.2 M"],
       ["Compute", "27.2 GMACs per second of audio; 99.7% in flow + decoder (hand-checked: dec.conv_pre = 59.2 MMACs/s)"],
       ["RTX 4060 laptop latency (p50, 3.7 s output)", "text encoder + durations 5.4 ms; flow + decoder 39.4 ms; MFA align_one 3.3 s"],
       ["RTX 4060 real-time factor", "model only 0.012 (80× faster than real time); end to end 0.90 (MFA dominates)"],
       ["CPU, 12 threads / 1 thread", "flow + decoder 0.78 s / 1.78 s (model RTF 0.21 / 0.48)"],
       ["Training (L40S)", "0.229 s/step at batch 32; 114k steps ≈ 7.3 h"]], [0.36, 0.64])
    tab_caption("results/profile_*.json (10 warm-up + 50 timed runs, CUDA-synchronized; MFA 1 + 5 runs). MACs counted with "
                "ptflops' aten backend (the module-hook backend reported ~0).")
    F(reuse["efficiency"], fig_caption("Latency and compute breakdown. End-to-end latency is aligner-bound, not model-bound."))

    # ================= 8 Robustness and live testing
    P("8. Robustness and live testing", h1)
    B(["<b>Edge-case suite</b> (21 cases, 21/21 on GPU and CPU): digits, currency, dates, abbreviations, an OOV name; MP3, "
       "8 / 48 kHz, stereo, clipped, noisy and reverberant audio; > 20 s, a 1.8 s pause, 1.6× tempo, a one-word cut; an "
       "unspoken word, a false start and a forced CTC path. Pass = exact length or a clean fallback.",
       "<b>Typed text:</b> 121-case stress set (119 pass; the 2 failures are G2P guesses for unknown names) and a 12-case "
       "end-to-end rehearsal including refusals.",
       "<b>Unit tests:</b> 308 pass on the author's machine; 300 pass, 9 skipped on a clean clone without datasets.",
       "<b>Browser demo:</b> visitors record or upload a sentence and get all seven systems side by side; 50 recorded "
       "sessions are kept in samples/browser_tests/."])
    P("Live testing of the browser demo exposed failures that read-speech corpora never contain:", small)
    T([["Input", "What happened", "Cause", "Status"],
       ["abc, chaganti (words first seen at inference)", "MFA returned spn; the demo fell back to CTC", "MFA reused a stale parsed copy of the lexicon", "<b>Fixed:</b> MFA's temp dir is keyed by the lexicon's hash"],
       ["bcd, xyz", "G2P invented a word (bcd → \"bed\")", "nothing marks a letter string", "<b>Fixed:</b> unknown vowel-less 3-5 letter tokens are spelled out"],
       ["letter \"a\" in a, b, c", "aligned to 30-80 ms; output says \"uh\"", "one pronunciation for \"a\": AH0, the article", "<b>Open:</b> needs a letter-name reading (EY1 gave 220-330 ms)"],
       ["\"I\" in may I know you", "\"I\" got 30 ms (MFA's minimum)", "vowel-to-vowel junction EY → AY", "<b>Open:</b> fine after a pause or sentence-initially"],
       ["ijk; brrr, grr; pq", "G2P word; wrongly spelled; not spelled", "limits of the regex heuristic", "<b>Open:</b> needs a learned normaliser"]],
      [0.21, 0.27, 0.25, 0.27])
    tab_caption("Live-testing findings (E-35). The test results are unaffected: every evaluation word was in the lexicon "
                "when MFA's cache was first built.")

    # ================= 9 Limitations
    P("9. Limitations and future work", h1)
    B(["<b>Accent-ID is a weak judge;</b> a MUSHRA-style listening test with accent ratings is the right next measurement.",
       "<b>Part of the accent lives in timing:</b> forcing LJ's voice onto L2 word timing caps how American it can sound.",
       "<b>Fixed alignments flatten phones</b> (14-16% less movement inside phones); V0 keeps better WER and UTMOS.",
       "<b>No word-boundary token</b> (\"a name\" = \"an aim\" at the encoder input); the 50/50 diphthong split is fixed.",
       "<b>The ablations were branched late</b> (20k steps from M1), so they test late-stage augmentation only.",
       "<b>V0 needs an adapter</b> (IPA from the lexicon, a blank-ownership rule); a learned adapter could narrow its gap.",
       "<b>End-to-end latency is MFA-bound</b> (3.3 s per reference vs 45 ms for the model)."])
    P("<b>Future work, by expected value:</b> (1) learn the alignment again with MAS <i>inside</i> MFA's word spans, "
      "restoring the alignment-latent feedback while keeping timing control; (2) a word-boundary flag on the encoder input; "
      "(3) learned diphthong split points; (4) splitting CH, JH, ER the same way; (5) a listening test as the accent judge; "
      "(6) a resident or faster aligner for live use; (7) letter names and vowel-only words for live input.")
    P("<b>Takeaway.</b> Make durations an input, copy word timing (not phone timing), and check the tokens before adding "
      "data: replacing MAS cut word error from 40.5 to 24 ms, and the largest later gain came from a two-token diphthong, "
      "not from more training data.")

    # ================= 10 Reproducibility
    P("10. Reproducibility and deliverables", h1)
    T([["Step", "Command", "Result"],
       ["Environments", "make env", "conda envs tts and mfa from pinned envs/*.yml (lock files in envs/lock/)"],
       ["Models", "make models", "frozen inference checkpoints and baselines from the HF Hub, pinned revision, sha256-checked"],
       ["Synthesize", "make infer / make live-demo", "out.wav (exact length), timings.json, timing.png"],
       ["Test outputs", "make outputs", "all 1,248 evaluated outputs per system into results/&lt;system&gt;/test/"],
       ["Tests", "make test; make text-stress; make rehearse", "300 pass on a clean clone; 119/121; 12/12"],
       ["Browser demo", "make web-demo (+ make web-tunnel)", "record / upload, all seven systems side by side"],
       ["This report", "make report-pdf", "docs/meeami-tts_report.pdf from the committed results"]], [0.18, 0.35, 0.47])
    tab_caption("Checked from a fresh clone with no login and an empty download cache (D-104).")
    T([["Deliverable", "Where"],
       ["Code (frozen at freeze-v1)", cfg.links.github],
       ["Detailed report with a code link for every claim", "docs/REPORT.md; decision and error log docs/DECISIONS.md"],
       ["Exact audio evidence: reference, annotation, all 7 outputs, word timings, ≥ 2 accents", "samples/demo/ (21 samples, 3 accents; MAE recomputed and checked, 147/147)"],
       ["Listening sets behind phase-2 decisions; browser recordings", "samples/listening/, samples/browser_tests/"],
       ["Checkpoints and every test output", cfg.links.hf],
       ["Slides", "slides/meeami-tts_v3.pptx"]], [0.45, 0.55])
    tab_caption("Where each deliverable lives.")

    # ================= Appendix
    P("Appendix A. Deviations from the plan", h1)
    T([["Plan said", "Done instead", "Why"],
       ["Rubber Band R3 time-stretching", "time-varying WSOLA (numpy), same time map", "R3 cost ~1.2-1.5 UTMOS on every clip; WSOLA 0.32"],
       ["torchaudio forced_align", "numpy CTC Viterbi, same MMS_FA model", "conda-forge torchaudio 2.8 lacks the extension"],
       ["OOV from g2p_en", "+ possessive rule", "g2p_en mangled 104 possessives"],
       ["Three GPU instances", "one L40S: M1, then 20k-step branches", "availability"],
       ["Data-center GPU profile", "laptop GPU and CPU only", "not runnable on the instance"],
       ["n_vocab = 72", "87 for m1_pos_diph only (opt-in)", "diagnosed diphthong flattening"],
       ["UTMOS via torch.hub defaults", "sha256-pinned weights, weights_only=True", "hub did a full pickle load of unhashed weights"],
       ["Baseline M0 only", "+ V0, the original VITS with MAS (after the freeze, inference only)", "the immediate baseline for the MAS → MFA swap"],
       ["Normalizer as specified", "+ typed-text pre-pass (after the freeze)", "live input; corpus words unchanged (tested)"],
       ["MFA align_one with a fixed temp dir", "temp dir keyed by the lexicon hash", "stale parsed lexicon made new words unknown"]],
      [0.28, 0.38, 0.34])
    P("Appendix B. Claim → evidence", h1)
    T([["Claim", "Evidence (committed files)"],
       ["Output length exact for every output", "length_error = 0 in every results/*/test/per_utt.csv"],
       ["MFA error on accented speech ≈ 19 ms", "reports/mfa_l2_dev.md"],
       ["MAS → MFA makes timing follow the reference", "results/paired/test_m1_pos_diph_vs_v0.csv; samples/demo/"],
       ["V0 renders durations early and scattered", "results/fidelity/test.json, calibration_dev.json"],
       ["Fixed alignments flatten phones", "results/phone_dynamics/*.json"],
       ["Cap 2.5 and noise 0.333 chosen by a pre-stated rule", "results/tune/grid.json, choice.json"],
       ["Ablations give no gain", "results/paired/{dev,test}_abl_*_vs_m1_pos.csv"],
       ["Diphthong split improves timing, WER, UTMOS; \"cry\" glides", "results/paired/*_m1_pos_diph_vs_m1_pos.csv; samples/listening/p2c_diphthongs/"],
       ["Test split not read before the freeze", "git tag freeze-v1 → 5889ecf; tts/guard.py"],
       ["Hosted checkpoints are the frozen ones", "MANIFEST.sha256 on the Hub; scripts/export_infer.py --verify"]], [0.45, 0.55])

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
