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
| T9 | Aug A, pace stats B, aug B, filter | **partial** | 2 x 24,948 clips, durations = frames; **UTMOS filter blocked** (see 2a) |
| T10 | Inference pipeline, CLI, edge suite | done | edge 21/21 exact length on GPU and CPU; CTC fallback exercised |
| T11 | Evaluation harness | done* | M0 on dev: see section 4 (*UTMOS = NaN, see 2a) |
| T12 | Profiling | done | `results/profile_cuda.json` from the smoke checkpoint (laptop) |
| T13 | GPU bundle + data check | done | `export/bundle.tar`; check-data PASS on a fresh unpack |
| T14 | Demo page | done | `out/demo_site/index.html` (M0 only until trained checkpoints exist) |

`make test`: **141 passed**.

## 2. Things only you can decide or do

### 2a. UTMOS (SpeechMOS) needs your permission — blocks the A/B filter and the MOS metric

The pinned model `tarepan/SpeechMOS:v1.2.0 utmos22_strong` is loaded with
`torch.hub.load(..., trust_repo=True)`, which downloads and *executes* code from that GitHub repo.
The session's safety policy refused to run it, and I did not work around that (no manual clone).
Consequences:

- **Ablation A/B training data are not filtered yet.** `data/manifests/aug_{a,b}_all.jsonl` hold all
  24,948 clips each; the filtered `aug_a.jsonl` / `aug_b.jsonl` (drop 5% largest UTMOS drop) do not
  exist, so `configs/abl_a.yaml` / `abl_b.yaml` refuse to train (missing file) — intentionally.
- UTMOS is NaN in evaluation results (`--no_utmos`), and `make tune` falls back to WER for its
  choice until UTMOS is available.

**To fix (laptop, ~40 min GPU):** look at https://github.com/tarepan/SpeechMOS (tag v1.2.0), then

```bash
make assets-eval          # downloads + loads UTMOS via torch.hub (TORCH_HOME=data/models/eval/torch_hub)
make aug-filter           # writes data/manifests/aug_a.jsonl, aug_b.jsonl, reports/aug_{a,b}.md
```

Then either rebuild the bundle (`make bundle`) or just copy the two small filtered manifests to the
instance (`data/manifests/aug_a.jsonl`, `aug_b.jsonl`; the audio in the bundle is unchanged).
`make eval ... ` without `--no_utmos` later backfills UTMOS into cached results.

If you decide *not* to filter, A/B can train on the unfiltered sets with
`train.aug_manifests=[data/manifests/aug_a_all.jsonl]` — a deviation from Section 6 that should be
written into DECISIONS.md and the slides.

### 2b. Upload size

`export/bundle.tar` is ~13 GB (LJ wavs 3.6 GB + augmented FLAC 9.2 GB). If your upload is slow,
use the light bundle (`make bundle BUNDLE_AUG=0`, ~3.8 GB) and regenerate augmentation on the
instance (section 3, step 1b; ~6 min per set on 12 cores, needs `rubberband-cli`).

### 2c. max_steps

`train.max_steps` in `configs/base.yaml` is a placeholder (100,000). Set it from the 300-step
benchmark on the instance (section 3, step 2) — one value in base.yaml keeps M1, A and B identical.

## 3. GPU runbook (exact commands)

Per instance (L40/A100, CUDA image, 8+ cores, ~100 GB disk):

```bash
# 1. setup (~15 min)
git clone <repo-url> meeami-tts && cd meeami-tts && git checkout smoke-ok
bash scripts/setup_gpu.sh                         # tts env from envs/tts.yml + CUDA check
bash scripts/fetch_bundle.sh <url-or-path>/bundle.tar   # verifies bundle.tar.sha256 if present
make check-data                                   # sha256 of every file, counts, durations == frames
#    copy the filtered manifests if they were made after bundling (2a):
#    rsync -avP data/manifests/aug_a.jsonl data/manifests/aug_b.jsonl user@host:~/meeami-tts/data/manifests/
# 1b. only with the light bundle: INSTALL_RUBBERBAND=1 bash scripts/setup_gpu.sh
#     conda run -n tts python -m tts.augment.generate --kind a   (and --kind b), then the filter

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
| Augmentation check | MFA finds planned boundaries: A median 17 ms, B 23 ms | reports/aug_*_all_verify.json |
| M0 on dev (1,200 refs) | length error 0; word-boundary MAE 31.2 ms (84.7% <= 50 ms); pause F1 0.75; WER 10.0% (refs 4.1%); US prob 0.73 (refs 0.30); spk sim LJ 0.76 / ref 0.05; UTMOS n/a | results/m0/dev/summary.json, results/plots/ |
| Overfit check (T7) | 500 steps, one utterance: ASR WER 5.6% | runs/overfit/overfit.json |
| Profile (4060, M1 architecture) | 28.1 M inference params; 27.2 GMACs/s audio; RTF model 0.015, end-to-end 1.35 (MFA ~5 s) | results/profile_cuda.json |
| Edge suite | 21/21 exact length, GPU and CPU (M0) | out/edge/{cuda,cpu}/edge_results.json |
| Bundle | export/bundle.tar 13 GB, sha256 1b283ea2…; check-data PASS on fresh unpack | export/bundle.tar.sha256 |

## 5. Back on the laptop (days 3-5)

```bash
for e in m1 abl_a abl_b; do tar -xzf export/$e.tar.gz; done   # -> export/<e>/ckpt_last.pt
make aug-filter            # if not done yet (2a) — only affects reports at this point
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
- `abl_a` / `abl_b` cannot start without the filtered manifests (2a) — by design.
- M1/A/B checkpoints do not exist yet, so `make demo`, `make eval-all` and `make tune` currently
  cover M0 only (they pick up `export/*/ckpt_last.pt` automatically).
