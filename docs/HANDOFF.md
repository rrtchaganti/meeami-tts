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
benchmark on the instance (section 3, phase 1 step 4) and commit it from the laptop.

## 3. GPU runbook — one L40, M1 first (D-68)

Only one L40 is available and **M1 is the priority**. Phase 1 trains M1 alone on the whole budget;
the ablations are decided after M1's dev results (phase 2, optional, possibly a later instance).
The three-instance plan of PLAN Section 10 is not used.

Instance: 1x L40, CUDA 12.x image (driver supports CUDA 12.9), 8+ cores, ~100 GB disk,
`git make wget tmux`.

### Phase 1 — M1

```bash
# 1. code + env (~15 min)
git clone <CLONE_URL> meeami-tts && cd meeami-tts && git checkout gpu-ready
bash scripts/setup_gpu.sh                    # must end "[setup] ... bf16 True"

# 2. bundle — from the laptop (WSL):
#    rsync -avP ~/meeami-tts/export/bundle.tar ~/meeami-tts/export/bundle.tar.sha256 user@HOST:~/meeami-tts/export/
bash scripts/fetch_bundle.sh export/bundle.tar   # checks the sha256, unpacks
make check-data                                  # must end "check-data: PASS"
rm export/bundle.tar                             # optional, after PASS

# 3. benchmark (~5 min): prints s/step and the max_steps that fit 5 / 8 / 10 h
conda run -n tts python scripts/train.py --config configs/m1.yaml --benchmark 300
```

4. **Set `max_steps` on the laptop** (one committed value, recorded in `runs/m1/git_commit.txt`).
   Budget = hours you pay for minus ~0.5 h for profiling and export:

   ```bash
   # laptop: configs/base.yaml -> train.max_steps: <steps for (H - 0.5) h>
   make test && git commit -am "max_steps <N> from L40 benchmark" && git push
   # instance:
   git checkout main && git pull
   ```

```bash
# 5. train (tmux survives a dropped SSH session)
mkdir -p logs
tmux new -d -s train 'conda run --no-capture-output -n tts python scripts/train.py --config configs/m1.yaml 2>&1 | tee logs/m1.log'
#    watch: tmux attach -t train  (detach: Ctrl-b d)
#    crash / preemption: same command + --resume   (checkpoints every 2k steps)

# 6. monitor: on the laptop  ssh -L 6006:localhost:6006 user@HOST  ; on the instance
conda run -n tts tensorboard --logdir runs --port 6006
#    open http://localhost:6006 ; losses, val/* audio every ~10k steps, nvidia-smi

# 6b. batch-size throughput test (after M1 finishes, GPU otherwise idle; ~10 min; writes nothing)
#     clips/s = batch / median s/step ; batch 32 baseline: 32 / 0.227 = ~141 clips/s
for b in 48 64 96; do
  conda run -n tts python scripts/train.py --config configs/m1.yaml --benchmark 300 --batch_size $b 2>&1 \
    | grep -E "^benchmark|out of memory" | tee -a logs/batch_bench.txt
done

# 7. profile before shutting down (MFA is not installed here -> skip that stage)
make profile CKPT=runs/m1/ckpt_last.pt DEVICE=cuda ARGS="profile.mfa_runs=0"

# 8. export -> laptop, verify, then terminate
bash scripts/export_run.sh m1                    # export/m1.tar.gz + .sha256
#    laptop: rsync -avP 'user@HOST:~/meeami-tts/export/m1.tar.gz*' ~/meeami-tts/export/
#            cd ~/meeami-tts && sha256sum -c export/m1.tar.gz.sha256      # OK, then terminate
```

`export/m1/ckpt_last.pt` keeps generator + discriminator, so phase 2 can start from it on any
later instance. Keep it safe.

### Phase 2 — ablations (only if decided after M1's dev results)

Branching design (PLAN Section 10 step 5): three equal-length continuations from the finished M1
checkpoint, run back to back. The fair comparison is then `m1_cont` vs `abl_a` vs `abl_b`
(same start, same extra steps, same fresh optimizer/LR schedule); `m1` itself is their common base.

```bash
# new instance: steps 1-3 of phase 1 (full bundle), then put the M1 checkpoint back:
#    laptop: rsync -avP ~/meeami-tts/export/m1.tar.gz user@HOST:~/meeami-tts/export/
tar -xzf export/m1.tar.gz                        # -> export/m1/ckpt_last.pt
# benchmark the slower config (augmented clips are ~1.2x longer):
conda run -n tts python scripts/train.py --config configs/abl_a.yaml --benchmark 300
# K = steps per branch, e.g. (H - 0.5) h / 3 ; same K for all three, given on the command line
tmux new -d -s branches 'B=export/m1/ckpt_last.pt; K=<K>; R="conda run --no-capture-output -n tts python scripts/train.py"
  $R --config configs/m1.yaml    --init_from $B --tag m1_cont --max_steps $K 2>&1 | tee logs/m1_cont.log &&
  $R --config configs/abl_a.yaml --init_from $B --max_steps $K             2>&1 | tee logs/abl_a.log &&
  $R --config configs/abl_b.yaml --init_from $B --max_steps $K             2>&1 | tee logs/abl_b.log'
#    a crashed branch: rerun just that line with --resume (and --tag m1_cont for the first)
for e in m1_cont abl_a abl_b; do bash scripts/export_run.sh $e; done
```

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
| Bundle | export/bundle.tar 13 GB, sha256 f1e2ff2d… (2026-11-07, WSOLA data + filtered manifests); check-data PASS on fresh unpack | export/bundle.tar.sha256 |

## 5. Back on the laptop

After phase 1 (M1 only):

```bash
tar -xzf export/m1.tar.gz                        # -> export/m1/ckpt_last.pt
make tune SPLIT=dev                              # cap x noise_scale grid; writes configs/inference.yaml
make eval SYSTEM=m1 SPLIT=dev                    # M1 vs M0 on dev: decide on phase 2 from this
make profile CKPT=export/m1/ckpt_last.pt DEVICE=cuda   # laptop end-to-end numbers (with MFA)
make edge CKPT=export/m1/ckpt_last.pt DEVICE=cuda && make edge CKPT=export/m1/ckpt_last.pt DEVICE=cpu
make demo
git add -A && git commit -m "M1 dev results + tuned inference config" && git push
```

**Do not tag `freeze-v1` yet** if phase 2 is still possible: after the tag, model code and
experiment configs must not change (CLAUDE.md), and the test accents get unlocked. Tag only once
the set of systems is final.

If phase 2 ran, the comparison model is `m1_cont`:

```bash
for e in m1_cont abl_a abl_b; do tar -xzf export/$e.tar.gz; done
make eval-all SPLIT=dev SYSTEMS="m0 m1 m1_cont abl_a abl_b"
```

When the systems are final:

```bash
git tag freeze-v1 && git push origin freeze-v1   # G3: code, configs, checkpoints frozen
make eval-all SPLIT=test SYSTEMS="<final systems>"   # unlocks test audio
make demo
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
