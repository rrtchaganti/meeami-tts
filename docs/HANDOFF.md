# Handoff — laptop work done, ready for the GPU run

Written at the end of the autonomous overnight run (2026-11-06). Details and reasons for every
choice are in [DECISIONS.md](DECISIONS.md) (D-n = decision, E-n = error/incident).

## 1. Status of the laptop tasks (PLAN.md Section 10)

| # | Task | Status | Acceptance result |
| --- | --- | --- | --- |
| T1 | Envs, Makefile, CLAUDE.md | done | `mfa version` 3.4.3; CUDA + bf16 visible in `tts` |
| T2 | Data, filelists, splits, manifests | done | 13,100 LJ clips; 24 L2 speakers; eval IDs dev 100 / test 104 |
| T3 | Normalizer, lexicon, symbols | done | 14,912 words; all 69 phones in MFA's set |
| T4 | LJ alignment, parser, stats | done | **13,073 / 13,100 kept (99.79%)**; sums = n_frames; stats written |
| T5 | Dev alignment + MFA-vs-manual | done | word MAE 19.4 ms, phone 13.1 ms (per accent in reports/mfa_l2_dev.md) |
| T6 | M0 with injected durations | done | 5 dev references: out.wav + timings.json + timing.png, exact length |
| T7 | VITS-MFA (changes 1-6) | done | fwd/bwd ok; only expected keys skipped; overfit ASR WER 5.6% |
| T8 | train.py, smoke | done | all 4 configs 200 steps @ batch 4; resume ok; 0.46 s/step |
| T9 | Aug A, pace stats B, aug B, filter | done | WSOLA (D-67); 2 x 23,701 kept of 24,948; UTMOS kept A 4.07 / B 3.66 (orig 4.38) |
| T10 | Inference pipeline, CLI, edge suite | done | edge 21/21 exact length on GPU and CPU; CTC fallback exercised |
| T11 | Evaluation harness | done | M0 on dev: see section 4 (UTMOS 3.94) |
| T12 | Profiling | done | `results/profile_cuda.json` from the smoke checkpoint (laptop) |
| T13 | GPU bundle + data check | done | `export/bundle.tar`; check-data PASS on a fresh unpack |
| T14 | Demo page | done | `out/demo_site/index.html` (M0 only until trained checkpoints exist) |

`make test`: **145 passed**.

## 2. Things only you can decide or do

### 2a. UTMOS and the A/B filter — resolved 2026-11-07

You reviewed `tarepan/SpeechMOS@v1.2.0`; its weights file is now downloaded separately,
sha256-pinned and loaded with `weights_only=True` (D-66). The first filter run showed that Rubber
Band R3 (the PLAN Section 6 stretcher) loses ~0.85-1.4 UTMOS on LJ speech even at mild ratios
(E-22); on your decision the augmented sets were regenerated with a time-varying WSOLA (D-67) and
filtered: `data/manifests/aug_a.jsonl`, `aug_b.jsonl` (23,701 clips each), reports in
`reports/aug_{a,b}.md` (R3 reports kept in `reports/r3/`). This is a deviation from Section 6 to
mention in the slides. Listening samples R3 vs WSOLA: `out/diag_stretch/` (laptop only).

### 2b. Upload size

`export/bundle.tar` is ~13 GB (LJ wavs 3.6 GB + augmented FLAC 9.2 GB). If your upload is slow,
use the light bundle (`make bundle BUNDLE_AUG=0`, ~3.8 GB) and regenerate augmentation on the
instance (section 3, step 1b; ~3 min per set on 12 cores plus MFA check and UTMOS filter;
WSOLA is numpy only, no `rubberband-cli` needed).

### 2c. max_steps

`train.max_steps` in `configs/base.yaml` is a placeholder (100,000). Set it from the 300-step
benchmark on the instance (section 3, step 2) — one value in base.yaml keeps M1, A and B identical.

## 3. GPU runbook (exact commands)

Per instance (L40/A100, CUDA image, 8+ cores, ~100 GB disk):

```bash
# 1. setup (~15 min)
git clone <repo-url> meeami-tts && cd meeami-tts && git checkout gpu-ready   # smoke-ok + WSOLA data/UTMOS fixes
bash scripts/setup_gpu.sh                         # tts env from envs/tts.yml + CUDA check
bash scripts/fetch_bundle.sh <url-or-path>/bundle.tar   # verifies bundle.tar.sha256 if present
make check-data                                   # sha256 of every file, counts, durations == frames
# 1b. only with the light bundle (BUNDLE_AUG=0): regenerate + verify + filter on the instance
#     (needs the mfa env and the eval models: make env-mfa assets assets-eval), then
#     make aug-a aug-b

# 2. benchmark (batch 32), then fix max_steps for all three runs
conda run -n tts python scripts/train.py --config configs/m1.yaml --benchmark 300
#    edit configs/base.yaml: train.max_steps: <value printed for your budget>

# 3. launch (one per instance; tmux survives a dropped SSH session)
mkdir -p logs
tmux new -d -s train 'conda run --no-capture-output -n tts python scripts/train.py --config configs/m1.yaml 2>&1 | tee logs/m1.log'
#    instance 2: configs/abl_a.yaml     instance 3: configs/abl_b.yaml
#    crashed/preempted: same command + --resume

# 4. monitor: ssh -L 6006:localhost:6006 user@host ; then on the instance
conda run -n tts tensorboard --logdir runs --port 6006
#    listen to val/* audio every ~10k steps; nvidia-smi

# 5. one-GPU fallback (branching design)
#    train m1 for most of the budget, then:
#    python scripts/train.py --config configs/m1.yaml    --init_from runs/m1/ckpt_last.pt --tag m1_cont
#    python scripts/train.py --config configs/abl_a.yaml --init_from runs/m1/ckpt_last.pt
#    python scripts/train.py --config configs/abl_b.yaml --init_from runs/m1/ckpt_last.pt

# 6. profile on the GPU before shutting down (needs MFA for the end-to-end stage: run
#    `make env-mfa assets` first, or pass profile.mfa_runs=0 to skip that stage)
make profile CKPT=runs/m1/ckpt_last.pt DEVICE=cuda

# 7. export and download, then terminate
bash scripts/export_run.sh m1     # export/m1.tar.gz (+ .sha256)
#    on the laptop: rsync -avP user@host:~/meeami-tts/export/ ./export/ ; sha256sum -c export/*.sha256
```

Laptop numbers for orientation: batch 4 runs at ~0.46 s/step on the RTX 4060 (2.8 GiB). Batch 32 on
an L40/A100 is expected around 0.5-1 s/step; the benchmark decides.

## 4. Results available now (dev only; test is locked)

| What | Result | Where |
| --- | --- | --- |
| LJ alignment QC | 13,073 / 13,100 kept (99.79%); all 27 drops = phone > 0.5 s | reports/align_lj.md |
| MFA vs manual (dev) | word MAE 19.4 ms (75.0% <= 20 ms, 93.8% <= 50 ms); phone 13.1 ms (87.7% / 96.7%) | reports/mfa_l2_dev.md |
| Dev pace (Ablation B) | words 1.17x LJ mean length; P(pause after word) 0.22 | stats/l2_dev_pace.json |
| Augmented sets (WSOLA) | MFA finds planned boundaries: A median 6.8 ms, B 7.2 ms (R3 was 17 / 23); UTMOS kept A 4.07, B 3.66 vs original 4.38 (R3 A 2.97) | reports/aug_*.md, reports/aug_*_all_verify.json |
| M0 on dev (1,200 refs) | length error 0; word-boundary MAE 31.2 ms (84.7% <= 50 ms); pause F1 0.75; WER 10.0% (refs 4.1%); US prob 0.73 (refs 0.30); spk sim LJ 0.76 / ref 0.05; UTMOS 3.94 [3.92, 3.96] (LJ test ground truth 4.36) | results/m0/dev/summary.json, results/plots/ |
| Overfit check (T7) | 500 steps, one utterance: ASR WER 5.6% | runs/overfit/overfit.json |
| Profile (4060, M1 architecture) | 28.1 M inference params; 27.2 GMACs/s audio; RTF model 0.015, end-to-end 1.35 (MFA ~5 s) | results/profile_cuda.json |
| Edge suite | 21/21 exact length, GPU and CPU (M0) | out/edge/{cuda,cpu}/edge_results.json |
| Bundle | export/bundle.tar 13 GB, sha256 1b283ea2…; check-data PASS on fresh unpack | export/bundle.tar.sha256 |

## 5. Back on the laptop (days 3-5)

```bash
for e in m1 abl_a abl_b; do tar -xzf export/$e.tar.gz; done   # -> export/<e>/ckpt_last.pt
make tune SPLIT=dev        # grid over cap x noise_scale; writes configs/inference.yaml
make eval-all SPLIT=dev
git tag freeze-v1          # G3: code, configs, checkpoints frozen
make eval-all SPLIT=test   # unlocks test audio
make profile CKPT=export/m1/ckpt_last.pt DEVICE=cuda
make demo
make edge CKPT=export/m1/ckpt_last.pt DEVICE=cuda && make edge CKPT=export/m1/ckpt_last.pt DEVICE=cpu
```

After `freeze-v1`, also fill the test manifest's manual word times if you want them in the
manifest (evaluation reads the annotation TextGrids directly, so this is optional).

## 6. Known limitations / things to watch

- `mfa align_one` costs ~10-15 s per reference (model loading); it dominates end-to-end latency in
  the live demo. Batch evaluation avoids it (D-53).
- CTC fallback word boundaries are coarser (~53 ms MAE vs ~21 ms for MFA on dev references).
- The laptop GPU (WSL) once failed with "CUDA error: unknown error" when several GPU jobs ran at
  once (E-16); run GPU jobs one at a time on the laptop.
- Ablation B's filter drops the most pause-heavy clips first (UTMOS penalises B's L2-like pauses,
  DECISIONS.md "Why B scores lower than A"); worth one sentence in the analysis.
- M1/A/B checkpoints do not exist yet, so `make demo`, `make eval-all` and `make tune` currently
  cover M0 only (they pick up `export/*/ckpt_last.pt` automatically).
