#!/usr/bin/env python
"""Train VITS-MFA (docs/PLAN.md Sections 4, 10 "Runs").

    python scripts/train.py --config configs/m1.yaml [--resume] [--init_from CKPT] [--tag NAME]
        [--max_steps N] [--batch_size N] [--benchmark N] [key=value ...]

runs/<exp>/: config.yaml (resolved), git_commit.txt, train.log, tb/, ckpt_<step>.pt, ckpt_last.pt
(keeps the last `train.keep_ckpts`). --init_from accepts the official VITS LJ generator (default,
from train.init_from) or one of our checkpoints (one-GPU branching, runbook step 5: weights only,
step counter restarts). --benchmark N times N steps and prints s/step and the steps that fit
train.budget_hours, then exits without writing checkpoints.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# scripts/ must not be importable (scripts/profile.py would shadow the stdlib `profile`)
sys.path[0] = str(Path(__file__).resolve().parents[1])

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader  # noqa: E402

from tts.config import REPO_ROOT, load_config, repo_path  # noqa: E402
from tts.models.vits_mfa.data import (  # noqa: E402
    ManifestDataset, MixedBucketSampler, collate, fixed_val_rows, read_manifest,
)
from tts.models.vits_mfa.trainer import (  # noqa: E402
    build_state, init_from_pretrained_vits, load_state, set_decoder_frozen, state_dict,
    train_step, val_losses,
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continue runs/<exp>/ckpt_last.pt")
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--tag", default=None, help="experiment name (default: config file stem)")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--benchmark", type=int, default=0)
    ap.add_argument("overrides", nargs="*")
    return ap.parse_args(argv)


def git_commit() -> str:
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                       text=True)
    dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "tts",
                            "scripts", "configs"], capture_output=True, text=True).stdout.strip()
    return r.stdout.strip() + (" (dirty)" if dirty else "")


def save(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def prune(run: Path, keep: int) -> None:
    ckpts = sorted(run.glob("ckpt_[0-9]*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for p in ckpts[:-keep]:
        p.unlink()


def main(argv=None) -> int:
    args = parse_args(argv)
    ov = list(args.overrides)
    if args.max_steps is not None:
        ov.append(f"train.max_steps={args.max_steps}")
    if args.batch_size is not None:
        ov.append(f"train.batch_size={args.batch_size}")
    cfg = load_config(repo_path(args.config), ov)
    exp = args.tag or cfg.get("exp") or Path(args.config).stem
    run = repo_path(cfg.train.runs_dir) / exp
    bench = args.benchmark > 0
    if not bench:
        run.mkdir(parents=True, exist_ok=True)
    t = cfg.train
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = bool(t.get("cudnn_benchmark", False))
        torch.backends.cuda.matmul.allow_tf32 = True

    handlers = [logging.StreamHandler(sys.stdout)]
    if not bench:
        handlers.append(logging.FileHandler(run / "train.log"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers,
                        force=True)
    log = logging.getLogger("train")
    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    if not bench:
        (run / "config.yaml").write_text(cfg_yaml)
        (run / "git_commit.txt").write_text(git_commit() + "\n")

    # data
    base_rows = [r for m in t.train_manifests for r in read_manifest(m)]
    aug_rows = [r for m in t.aug_manifests for r in read_manifest(m)]
    if float(t.aug_fraction) > 0 and not aug_rows:
        # never let an ablation silently train as M1 (empty / unfiltered-missing manifests)
        log.error(f"aug_fraction={t.aug_fraction} but {list(t.aug_manifests)} have no rows "
                  "(run the UTMOS filter: make aug-filter; see docs/HANDOFF.md)")
        return 3
    ds = ConcatDataset([ManifestDataset(base_rows, cfg), ManifestDataset(aug_rows, cfg)])
    sampler = MixedBucketSampler([r["n_frames"] for r in base_rows],
                                 [r["n_frames"] for r in aug_rows], t.batch_size,
                                 t.aug_fraction, seed=cfg.seed)
    nw = int(t.num_workers)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=nw,
                        pin_memory=dev.type == "cuda", persistent_workers=nw > 0,
                        prefetch_factor=4 if nw > 0 else None)
    val_rows = read_manifest(t.val_manifest)
    val_ds = ManifestDataset(val_rows, cfg)
    val_loader = DataLoader(val_ds, batch_size=min(8, len(val_rows)), collate_fn=collate)
    audio_rows = fixed_val_rows(val_rows, t.n_val_audio, cfg.seed)
    log.info(f"exp={exp} device={dev} base_rows={len(base_rows)} aug_rows={len(aug_rows)} "
             f"aug_fraction={t.aug_fraction} batch={t.batch_size} batches/epoch={len(sampler)}")

    # model
    state = build_state(cfg, dev)
    last = run / "ckpt_last.pt"
    if args.resume:
        if not last.exists():
            log.error(f"--resume: {last} not found")
            return 1
        load_state(state, torch.load(last, map_location=dev, weights_only=True))
        log.info(f"resumed from {last} at step {state.step}, epoch {state.epoch}")
    else:
        src = args.init_from or t.init_from
        if src:
            ck = torch.load(repo_path(src), map_location="cpu", weights_only=True)
            if "net_g" in ck:  # one of ours: weights only (branching design)
                state.net_g.load_state_dict(ck["net_g"])
                state.net_d.load_state_dict(ck["net_d"])
                log.info(f"init_from {src}: generator + discriminator weights (step reset)")
            else:
                info = init_from_pretrained_vits(state, str(repo_path(src)))
                log.info(f"init_from {src}: loaded {info['loaded']} tensors; skipped "
                         f"{len(info['skipped'])}; not initialized from checkpoint "
                         f"{len(info['missing'])}")
                if not bench:
                    (run / "init_skipped_keys.txt").write_text(
                        "\n".join(info["skipped"]) + "\n--- not in checkpoint\n"
                        + "\n".join(info["missing"]) + "\n")
            del ck

    tb = None
    if not bench:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(run / "tb"))

    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True
        log.info("signal received: saving and stopping")
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    def checkpoint():
        ck = state_dict(state, cfg_yaml)
        save(run / f"ckpt_{state.step}.pt", ck)
        save(last, ck)
        prune(run, int(t.keep_ckpts))
        log.info(f"saved ckpt_{state.step}.pt")

    @torch.no_grad()
    def validate():
        sums, n = {}, 0
        for b in val_loader:
            for k, v in val_losses(state, b, cfg, dev).items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        for k, v in sums.items():
            tb.add_scalar(k, v / n, state.step)
        state.net_g.eval()
        for r in audio_rows:
            it = val_ds[val_rows.index(r)]
            x = it["ids"].unsqueeze(0).to(dev)
            wav, _ = state.net_g.infer(x, torch.tensor([x.size(1)], device=dev),
                                       it["dur"].unsqueeze(0).to(dev))
            tb.add_audio(f"val/{r['id']}", wav[0].float().cpu(), state.step,
                         sample_rate=int(cfg.audio.sample_rate))
            if state.step == 0 or not getattr(validate, "gt_done", False):
                tb.add_audio(f"gt/{r['id']}", it["wav"].unsqueeze(0), state.step,
                             sample_rate=int(cfg.audio.sample_rate))
        validate.gt_done = True
        log.info(f"step {state.step} val " + " ".join(f"{k}={v / n:.4f}" for k, v in sums.items()))

    max_steps = int(t.max_steps)
    n_bench = args.benchmark
    times = []
    frozen = None
    t_last = time.time()
    while state.step < max_steps and not stop["flag"]:
        sampler.set_epoch(state.epoch)
        for batch in loader:
            want_frozen = state.step < int(t.freeze_decoder_steps)
            if want_frozen != frozen:
                set_decoder_frozen(state, want_frozen)
                frozen = want_frozen
            if dev.type == "cuda" and bench:
                torch.cuda.synchronize()
            t0 = time.time()
            out = train_step(state, batch, cfg, dev)
            if bench:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - t0)
                if len(times) >= n_bench:
                    break
                continue
            if not all(math.isfinite(v) for v in out.values()):
                log.error(f"non-finite loss at step {state.step}: {out}")
                checkpoint()
                return 2
            if state.step % int(t.log_interval) == 0:
                dt = (time.time() - t_last) / int(t.log_interval)
                t_last = time.time()
                for k, v in out.items():
                    tb.add_scalar(k, v, state.step)
                tb.add_scalar("lr", state.optim_g.param_groups[0]["lr"], state.step)
                log.info(f"step {state.step} epoch {state.epoch} {dt:.3f}s/step "
                         + " ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in out.items()))
            if state.step % int(t.val_interval) == 0:
                validate()
            if state.step % int(t.ckpt_interval) == 0:
                checkpoint()
            if state.step >= max_steps or stop["flag"]:
                break
        if bench and len(times) >= n_bench:
            break
        state.epoch += 1
        state.sched_g.step()
        state.sched_d.step()

    if bench:
        warm = min(10, len(times) // 5)
        s = sorted(times[warm:])
        mean = sum(s) / len(s)
        print(f"benchmark: {len(times)} steps (first {warm} excluded), batch {t.batch_size}, "
              f"{mean:.3f} s/step mean, {s[len(s) // 2]:.3f} median, "
              f"{torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else 0:.1f} GiB peak")
        for h in t.budget_hours:
            print(f"  {h:>4} h budget -> max_steps ~= {int(h * 3600 / mean):,}")
        return 0
    if not (run / f"ckpt_{state.step}.pt").exists():
        checkpoint()
    if state.step % int(t.val_interval) != 0:
        validate()
    tb.close()
    log.info(f"done at step {state.step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
