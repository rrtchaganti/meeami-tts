"""Export the small derived tables the PDF report draws (D-110), so it builds from committed files.

- training curves of the phase-1 run, from its TensorBoard log (export/<run>/tb, laptop only);
- the reference pace rho per utterance (PLAN Section 7; tts.eval.plots.rho_by_split), which needs
  the evaluated word timings in results/<eval.rho_from>/<split>/wav (laptop or `make outputs`).
Run in the tts env: `make report-data`.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tts.config import load_config, repo_path  # noqa: E402


def write_rows(path: Path, header, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        n = 0
        for r in rows:
            w.writerow(r)
            n += 1
    return n


def training_rows(tb_dir: Path, tags):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags().get("scalars", []))
    for tag in tags:
        if tag not in have:
            raise SystemExit(f"tag {tag} not in {tb_dir}")
        for e in ea.Scalars(tag):
            yield tag, e.step, f"{e.value:.6g}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "report_pdf.yaml"))
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    rd = cfg.report_data
    out = repo_path(rd.out_dir)
    n = write_rows(out / f"{rd.training_run}_training_curves.csv", ["tag", "step", "value"],
                   training_rows(repo_path(rd.tb_dir), list(rd.tags)))
    print(f"training curves: {n} rows")
    from tts.eval.plots import rho_by_split
    r = rho_by_split(cfg, tuple(rd.pace_splits))
    n = write_rows(out / "pace_rho.csv", ["split", "utterance_mean_log_rho"],
                   ((sp, f"{v:.6f}") for sp, vals in r.items() for v in vals))
    print(f"pace: {n} rows ({', '.join(f'{k} {len(v)}' for k, v in r.items())})")


if __name__ == "__main__":
    main()
