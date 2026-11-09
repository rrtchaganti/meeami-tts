"""Paired comparison of systems against a control on one split (DECISIONS D-82).

`scripts/evaluate.py` compares every system with M1 only. Phase-2 systems each have their own
control (D-79: the ablations vs m1_pos), so this compares per_utt.csv files directly:
paired Wilcoxon per metric, overall, per accent, and per bin of |log stretch_ratio| (PLAN
Section 6's hypothesis is about high stretch ratios, with no loss near 1.0x). p values are
Holm-adjusted within each (system, group) family of metrics.

    python -m tts.eval.paired --control m1_pos --system abl_a_pos --system abl_b_pos [--split dev]
Writes results/paired/<split>_<system>_vs_<control>.csv and prints a table.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tts.config import REPO_ROOT, load_config, repo_path
from tts.eval.metrics import wilcoxon


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p values; NaNs stay NaN and are not counted."""
    p = np.asarray(p, float)
    idx = [i for i in np.argsort(p) if not np.isnan(p[i])]
    out, run, m = np.full(len(p), np.nan), 0.0, len(idx)
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (m - k) * p[i]))
        out[i] = run
    return out.tolist()


def stretch_bins(ratio: pd.Series, quantiles: list[float]) -> pd.Series:
    """|log r| binned at the given quantiles -> labels 's0' (near 1.0x) ... 'sK' (furthest)."""
    x = np.abs(np.log(ratio.astype(float)))
    edges = np.r_[-np.inf, x.quantile(quantiles).to_numpy(), np.inf]
    lab = pd.cut(x, edges, labels=[f"s{i}" for i in range(len(edges) - 1)])
    return lab.astype(str).where(x.notna(), None)


def compare(sys_df: pd.DataFrame, ctl_df: pd.DataFrame, metrics: list[str], lower: list[str],
            quantiles: list[float]) -> pd.DataFrame:
    ok_s, ok_c = sys_df[sys_df.status == "ok"], ctl_df[ctl_df.status == "ok"]
    m = ok_s.merge(ok_c, on="id", suffixes=("", "_ctl"))
    m["sbin"] = stretch_bins(m["stretch_ratio_ctl"], quantiles)  # bin by the control's ratio
    groups = [("all", m)] + [(f"accent={a}", g) for a, g in m.groupby("accent")] + \
             [(f"stretch={b}", g) for b, g in m.groupby("sbin")]
    rows = []
    for gname, g in groups:
        fam = []
        for k in metrics:
            a, b = g[k].astype(float).to_numpy(), g[f"{k}_ctl"].astype(float).to_numpy()
            r = wilcoxon(a, b)
            d = np.nanmean(a - b)
            better = (d < 0) if k in lower else (d > 0)
            fam.append({"group": gname, "metric": k, "n": r["n"], "system": np.nanmean(a),
                        "control": np.nanmean(b), "mean_diff": d, "median_diff": r["median_diff"],
                        "p": r["p"], "better": bool(better)})
        for row, pa in zip(fam, holm([f["p"] for f in fam])):
            row["p_holm"] = pa
        rows += fam
    for gname, g in [(f"stretch={b}", g) for b, g in m.groupby("sbin")]:
        r = np.abs(np.log(g["stretch_ratio_ctl"].astype(float)))
        for row in rows:
            if row["group"] == gname:
                row["abs_log_ratio_range"] = f"{r.min():.3f}-{r.max():.3f}"
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--control", required=True)
    ap.add_argument("--system", action="append", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "inference.yaml"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    from tts.guard import assert_split_allowed
    assert_split_allowed(a.split, cfg)
    pc = cfg.paired
    res = repo_path("results")
    ctl = pd.read_csv(res / a.control / a.split / "per_utt.csv")
    out = repo_path(pc.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    for s in a.system:
        df = compare(pd.read_csv(res / s / a.split / "per_utt.csv"), ctl, list(pc.metrics),
                     list(pc.lower_is_better), list(pc.stretch_quantiles))
        df.to_csv(out / f"{a.split}_{s}_vs_{a.control}.csv", index=False)
        print(f"\n=== {s} vs {a.control} ({a.split}) ===")
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
