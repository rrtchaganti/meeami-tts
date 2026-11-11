#!/usr/bin/env python
"""Inference-only checkpoints for hosting (DECISIONS D-95; `make export-infer`).

    python scripts/export_infer.py --name m1_pos_diph [--name ...] [--verify]

export/<name>/ckpt_last.pt (956 MB: generator, discriminators, optimizer states) ->
export/<name>/infer.pt (141 MB: `net_g`, `config`, `step` and the source file's sha256), which is
all `VitsSystem` reads. With --verify, both files are loaded and must give bit-identical
generator weights and an identical waveform for a fixed token/duration input on CPU.
Writes export/MANIFEST.sha256 (one line per infer.pt) for scripts/fetch_models.py.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

import torch  # noqa: E402

from tts.config import repo_path  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def export(name: str) -> Path:
    src = repo_path("export") / name / "ckpt_last.pt"
    ck = torch.load(src, map_location="cpu", weights_only=True)
    out = src.with_name("infer.pt")
    torch.save({"net_g": ck["net_g"], "config": ck["config"], "step": ck.get("step", 0),
                "source": {"file": f"export/{name}/ckpt_last.pt", "sha256": sha256(src)}}, out)
    return out


def verify(name: str) -> None:
    import numpy as np

    from tts.infer.systems import VitsSystem

    d = repo_path("export") / name
    a, b = VitsSystem(d / "ckpt_last.pt", "cpu"), VitsSystem(d / "infer.pt", "cpu")
    for k, v in a.net.state_dict().items():
        assert torch.equal(v, b.net.state_dict()[k]), k
    toks = ["sil", "HH", "AH0", "L", "OW1", "sp", "W", "ER1", "L", "D", "sil"]
    durs = np.array([10, 6, 4, 7, 15, 8, 6, 12, 6, 7, 10])
    ya = (torch.manual_seed(0), a.synthesize(toks, durs))[1]
    yb = (torch.manual_seed(0), b.synthesize(toks, durs))[1]
    assert np.array_equal(ya, yb), "waveforms differ"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    lines = []
    for n in a.name:
        out = export(n)
        if a.verify:
            verify(n)
        lines.append(f"{sha256(out)}  {n}/infer.pt")
        print(f"[export] {out} {out.stat().st_size / 1e6:.0f} MB" + (" (verified)" if a.verify else ""))
    (repo_path("export") / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
