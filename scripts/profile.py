#!/usr/bin/env python
"""Efficiency profile (docs/PLAN.md Section 8 "Efficiency metrics"; `make profile`).

    python scripts/profile.py --ckpt export/m1/ckpt_last.pt --device cuda|cpu [--threads N]

Parameters per module (inference path vs training-only), MACs (ptflops) for 1 s and 5 s outputs
as GMACs per second of audio plus one decoder layer worked by hand, per-stage latency
(normalize + G2P, MFA, text encoder + durations, flow + decoder; batch 1; warm-up + timed runs;
p50/p90; CUDA-synchronized) and RTF (model-only and end-to-end). -> results/profile_<device>.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

# scripts/ must not be importable (scripts/profile.py would shadow the stdlib `profile`)
sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tts.config import load_config, repo_path  # noqa: E402


def timed(fn, warm: int, runs: int, device: str) -> dict:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return {"p50": float(np.percentile(ts, 50)), "p90": float(np.percentile(ts, 90)),
            "mean": float(np.mean(ts)), "runs": runs, "warmup": warm}


def count(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all)")
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(repo_path(a.config))
    pc = cfg.profile
    dev = a.device
    if dev == "cpu":
        torch.set_num_threads(a.threads or os.cpu_count())
    from tts.infer.systems import VitsSystem
    from tts.models.vits_mfa.models import MultiPeriodDiscriminator
    from tts.text.symbols import to_ids

    sysm = VitsSystem(a.ckpt, dev)
    net = sysm.net
    hop, sr = sysm.hop, sysm.sample_rate
    fps = sr / hop
    out = {"ckpt": a.ckpt, "step": sysm.step, "device": dev,
           "threads": torch.get_num_threads() if dev == "cpu" else None,
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
           "cpu": platform.processor() or platform.machine(), "torch": torch.__version__}

    # parameters
    disc = MultiPeriodDiscriminator()
    mods = {"text_encoder": count(net.enc_p), "duration_predictor": count(net.dp),
            "flow": count(net.flow), "decoder": count(net.dec),
            "posterior_encoder (train only)": count(net.enc_q),
            "discriminators (train only)": count(disc)}
    out["params"] = {"modules": mods,
                     "inference_path": sum(v for k, v in mods.items() if "train only" not in k),
                     "training_only": sum(v for k, v in mods.items() if "train only" in k)}

    # MACs with ptflops, per stage, for 1 s and 5 s outputs
    from ptflops import get_model_complexity_info

    def make_inputs(seconds: float):
        frames = int(round(seconds * fps))
        n_tok = max(3, int(round(seconds * pc.tokens_per_second)))
        d = np.full(n_tok, frames // n_tok)
        d[-1] += frames - d.sum()
        x = torch.tensor([to_ids(["sil"] + ["AH0"] * (n_tok - 2) + ["sil"])], device=dev)
        return x, torch.tensor([n_tok], device=dev), torch.tensor([d.tolist()], device=dev), frames

    class Enc(torch.nn.Module):
        def forward(self, x, n):
            h, m = net.enc_p.encode(x, n)
            return net.dp(h, m)

    class FlowDec(torch.nn.Module):
        def forward(self, z, mask):
            return net.dec(net.flow(z, mask, reverse=True) * mask)

    macs = {}
    for seconds in (1.0, 5.0):
        x, n, d, frames = make_inputs(seconds)
        z = torch.randn(1, cfg.model.inter_channels, frames, device=dev)
        mask = torch.ones(1, 1, frames, device=dev)
        e, _ = get_model_complexity_info(Enc(), (1,), input_constructor=lambda _: {"x": x, "n": n},
                                         as_strings=False, print_per_layer_stat=False, verbose=False)
        f, _ = get_model_complexity_info(FlowDec(), (1,),
                                         input_constructor=lambda _: {"z": z, "mask": mask},
                                         as_strings=False, print_per_layer_stat=False, verbose=False)
        macs[f"{int(seconds)}s"] = {"text_encoder_dp_gmacs": e / 1e9, "flow_decoder_gmacs": f / 1e9,
                                    "total_gmacs": (e + f) / 1e9,
                                    "gmacs_per_audio_second": (e + f) / 1e9 / seconds}
    k, cin, cout = 7, int(cfg.model.inter_channels), int(cfg.model.upsample_initial_channel)
    L = int(round(fps))
    macs["hand_worked_decoder_conv_pre"] = {
        "layer": "dec.conv_pre Conv1d(192 -> 512, kernel 7) on 1 s of frames",
        "formula": "K x C_in x C_out x L_out", "K": k, "C_in": cin, "C_out": cout, "L_out": L,
        "macs": k * cin * cout * L, "gmacs": k * cin * cout * L / 1e9}
    out["macs"] = macs

    # latency per stage on a real reference (dev speaker)
    from tts.infer.pipeline import Pipeline, build_timeline, fit_words, quantize
    from tts.text.normalize import Normalizer, words

    pipe = Pipeline(cfg, sysm)
    ref = repo_path(pc.reference_wav)
    text = repo_path(pc.reference_txt).read_text().strip()
    info = pipe.run(text, ref, repo_path("results") / "_profile_tmp", plot=False)
    tokens, durs = info["tokens"], info["durations"]
    norm = Normalizer(cfg)
    lat = {"normalize_g2p": timed(lambda: pipe.lexicon_path.exists() and words(norm(text)), 10, 50,
                                  "cpu")}
    if pc.mfa_runs > 0:
        import tempfile

        from tts.audio import load_wav

        y, s0 = load_wav(ref)
        ws = words(norm(text))
        nf = info["durations"] and sum(info["durations"])

        def mfa():
            with tempfile.TemporaryDirectory() as dd:
                pipe.align(y, s0, ws, Path(dd), "mfa", nf)
        lat["mfa_align_one"] = timed(mfa, int(pc.mfa_warmup), int(pc.mfa_runs), "cpu")
    lat["text_encoder_durations"] = timed(lambda: sysm.predict_log_durations(tokens),
                                          int(pc.warmup), int(pc.runs), dev)
    lat["flow_decoder"] = timed(lambda: sysm.synthesize(tokens, durs, 0.667),
                                int(pc.warmup), int(pc.runs), dev)
    out["latency_ms"] = lat
    dur_out = sum(durs) * hop / sr
    model_ms = lat["text_encoder_durations"]["p50"] + lat["flow_decoder"]["p50"]
    e2e_ms = model_ms + lat["normalize_g2p"]["p50"] + lat.get("mfa_align_one", {"p50": 0})["p50"]
    out["reference"] = {"wav": pc.reference_wav, "output_seconds": dur_out}
    out["rtf"] = {"model_only": model_ms / 1000 / dur_out, "end_to_end": e2e_ms / 1000 / dur_out}

    rv = []
    for seconds in pc.rtf_lengths:
        x, n, d, frames = make_inputs(float(seconds))
        toks = ["sil"] + ["AH0"] * (x.size(1) - 2) + ["sil"]
        t = timed(lambda: sysm.synthesize(toks, d[0].tolist(), 0.667), 3, 10, dev)
        rv.append({"seconds": float(seconds), "p50_ms": t["p50"],
                   "rtf_model": t["p50"] / 1000 / float(seconds)})
    out["rtf_vs_length"] = rv
    name = a.tag or (f"profile_{dev}" + (f"_{torch.get_num_threads()}t" if dev == "cpu" else ""))
    dst = repo_path("results") / f"{name}.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"params": out["params"], "macs_1s": macs["1s"], "rtf": out["rtf"],
                      "latency_p50_ms": {k: round(v["p50"], 2) for k, v in lat.items()}}, indent=1))
    print(f"-> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
