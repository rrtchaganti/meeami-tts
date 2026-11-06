"""`make tune` (docs/PLAN.md Section 5 "Parameters", Section 10 Days 3-5): dev accents only.

Grid over cap x noise_scale with transfer=elastic, plus the transfer variants (proportional,
phone_copy) and the CTC aligner at the default cap/noise, each evaluated with scripts/evaluate.py
on a fixed dev subset. Chooses (cap, noise_scale): among elastic settings whose word-boundary MAE
is within `tune.mae_tolerance_ms` of the best, the highest UTMOS (lowest WER if UTMOS is
unavailable). Writes the choice into configs/inference.yaml and the grid to results/tune/grid.json.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys

from tts.config import REPO_ROOT, load_config, repo_path


def choose(grid: list[dict], aligner: str, tol_ms: float) -> tuple[dict, str, float]:
    """Among elastic settings within tol_ms of the best word-boundary MAE: highest UTMOS, or
    lowest WER when UTMOS is unavailable. Returns (pick, criterion, best_mae)."""
    el = [r for r in grid if r["transfer"] == "elastic" and r["aligner"] == aligner]
    best = min(r["wb_mae_ms"] for r in el)
    ok = [r for r in el if r["wb_mae_ms"] <= best + tol_ms]
    have_mos = all(r["utmos"] is not None and not math.isnan(r["utmos"]) for r in ok)
    if have_mos:
        return max(ok, key=lambda r: r["utmos"]), "UTMOS", best
    return min(ok, key=lambda r: r["wer"]), "WER (UTMOS unavailable)", best


def _name(t, cap, ns, al):
    return f"{t}_cap{cap}_ns{ns}_{al}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--ckpt", default="export/m1/ckpt_last.pt")
    ap.add_argument("--system", default="m1")
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--dry_run", action="store_true", help="do not rewrite inference.yaml")
    a = ap.parse_args(argv)
    if a.split != "dev":
        raise SystemExit("tune runs on dev accents only (Section 5)")
    cfg = load_config(repo_path(a.config))
    g = cfg.inference.tuning_grid
    c = cfg.inference
    points = [("elastic", cap, ns, c.aligner) for cap in g.cap for ns in g.noise_scale]
    points += [(t, c.cap, c.noise_scale, c.aligner) for t in g.transfer if t != "elastic"]
    points += [("elastic", c.cap, c.noise_scale, al) for al in g.aligner if al != c.aligner]
    grid = []
    for t, cap, ns, al in points:
        name = _name(t, cap, ns, al)
        tag = f"tune/{name}"
        summ = repo_path("results") / tag / "dev" / "summary.json"
        if not summ.exists():
            cmd = [sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"), "--system", a.system,
                   "--ckpt", a.ckpt, "--split", "dev", "--n", str(cfg.tune.n_utts), "--tag", tag,
                   f"inference.transfer={t}", f"inference.cap={'null' if cap is None else cap}",
                   f"inference.noise_scale={ns}", f"inference.aligner={al}"]
            print("$ " + " ".join(cmd[1:]), flush=True)
            subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        s = json.loads(summ.read_text())
        m = s["metrics"]["all"]
        grid.append({"name": name, "transfer": t, "cap": cap, "noise_scale": ns, "aligner": al,
                     "wb_mae_ms": s["pooled_boundaries_mfa"]["mae_ms"],
                     "utmos": m["utmos"]["mean"], "wer": m["wer"]["mean"],
                     "us_prob": m["us_prob"]["mean"], "spk_sim_lj": m["spk_sim_lj"]["mean"]})
    out = repo_path("results") / "tune"
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.json").write_text(json.dumps(grid, indent=1) + "\n")

    pick, crit, best_mae = choose(grid, c.aligner, float(cfg.tune.mae_tolerance_ms))
    print(f"tune: chose cap={pick['cap']} noise_scale={pick['noise_scale']} by {crit} "
          f"(best MAE {best_mae:.1f} ms, tolerance {cfg.tune.mae_tolerance_ms} ms)")
    (out / "choice.json").write_text(json.dumps({"choice": pick, "criterion": crit,
                                                 "best_mae_ms": best_mae}, indent=1) + "\n")
    if not a.dry_run:
        p = repo_path(a.config)
        txt = p.read_text()
        cap_s = "null" if pick["cap"] is None else str(pick["cap"])
        txt = re.sub(r"(?m)^(  cap: )\S+(.*)$", rf"\g<1>{cap_s}\2  # tuned", txt, count=1)
        txt = re.sub(r"(?m)^(  noise_scale: )\S+(.*)$", rf"\g<1>{pick['noise_scale']}\2  # tuned",
                     txt, count=1)
        p.write_text(txt)
        print(f"wrote cap/noise_scale to {p.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
