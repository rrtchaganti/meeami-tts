"""T7 acceptance: after the one-utterance overfit, synthesize it with its MFA durations and check
intelligibility with the Section 8 ASR (WER). Writes <run>/overfit.wav and <run>/overfit.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from omegaconf import OmegaConf

from tts.config import repo_path
from tts.models.vits_mfa.data import ManifestDataset, read_manifest
from tts.models.vits_mfa.models import build_generator


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/overfit")
    ap.add_argument("--max_wer", type=float, default=0.25)
    a = ap.parse_args(argv)
    run = repo_path(a.run)
    ck = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=True)
    cfg = OmegaConf.create(ck["config"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_generator(cfg).to(dev).eval()
    net.load_state_dict(ck["net_g"])
    row = read_manifest(cfg.train.train_manifests[0])[0]
    it = ManifestDataset([row], cfg)[0]
    x = it["ids"].unsqueeze(0).to(dev)
    wav, _ = net.infer(x, torch.tensor([x.size(1)], device=dev), it["dur"].unsqueeze(0).to(dev))
    y = wav[0, 0].float().cpu().numpy()
    sr = int(cfg.audio.sample_rate)
    sf.write(run / "overfit.wav", y, sr)
    out = {"id": row["id"], "text": row["text"], "step": ck["step"], "samples": len(y),
           "expected_samples": row["n_frames"] * int(cfg.audio.hop_length)}
    try:
        from tts.eval.asr import ASR, wer_cer

        hyp = ASR(cfg, "cuda" if dev.type == "cuda" else "cpu").transcribe(y, sr)
        w, c = wer_cer(row["text"], hyp)
        out.update(asr=hyp, wer=w, cer=c, intelligible=w <= a.max_wer)
    except FileNotFoundError as e:
        out.update(asr=None, note=str(e))
    (run / "overfit.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0 if out.get("intelligible", True) else 1


if __name__ == "__main__":
    sys.exit(main())
