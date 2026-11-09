# meeami-tts — time-aligned US-English TTS

Speaks the input text in the LJSpeech voice with a US accent, on the word timing (word starts/ends,
pauses, total length) of a reference recording of the same text by a non-US speaker.

- Specification: [docs/PLAN.md](docs/PLAN.md) (code comments cite its sections).
- Every decision and error, with reasons: [docs/DECISIONS.md](docs/DECISIONS.md).
- Current status, blockers and the GPU runbook: **[docs/HANDOFF.md](docs/HANDOFF.md)**.
- Slides: [slides/meeami-tts.pptx](slides/meeami-tts.pptx); demo page: `make demo` -> `out/demo_site/index.html`.

| Output property | Source |
| --- | --- |
| Words, pronunciation | input text -> normalizer -> `data/lexicon/lexicon.txt` (cmudict first pron., g2p_en OOVs) |
| Voice, accent, intonation | model trained on LJSpeech only (main: `m1_pos_diph`, VITS with MFA durations; M0: FastSpeech2 baseline) |
| Word times, pauses, length | reference audio aligned with MFA `english_us_arpa` (MMS_FA CTC fallback) |

## Setup

Requirements: Linux/WSL2, conda (Miniconda), NVIDIA GPU + driver for CUDA 12.9, system
`rubberband-cli` >= 3.0 (only for `augment.stretcher: rubberband`), `ffmpeg`, `sox`, `git`, `make`. Datasets are not downloaded by the code:
LJSpeech 1.1 must be at `data/raw/LJSpeech-1.1/`, L2-ARCTIC v5 (access form) at `data/raw/l2arctic/`.

```bash
make env      # pinned conda envs: tts (everything) and mfa (Montreal Forced Aligner 3.4.3 only)
make test     # unit tests (environment, normalizer, lexicon, parser, model, pipeline, metrics...)
```

`envs/lock/*.full.yml` hold full transitive exports of the tested environments. MFA is always
called as `conda run -n mfa mfa ...` with `MFA_ROOT_DIR=data/models/mfa`.

## Data split

Train on LJSpeech only (original VITS filelists: 12,500 / 100 / 500; 13,073 kept after alignment
QC). L2-ARCTIC is split by accent: **dev** Hindi, Mandarin, Spanish (tuning, Ablation B pace
statistics); **test** Arabic, Korean, Vietnamese, locked until the git tag `freeze-v1`
(`tts/guard.py`: code reading test-speaker audio refuses without it). Evaluation set = utterances
with manual annotations for every speaker of the split: dev 100 x 12, test 104 x 12.

## Systems and results

| System | What it is |
| --- | --- |
| M0 | FastSpeech2 + HiFi-GAN (pretrained LJSpeech), durations injected |
| M1 | VITS fine-tuned with MFA durations instead of MAS (114k steps, D-70) |
| m1_pos | M1 + 20k steps with a position-in-phone embedding (D-76) |
| **m1_pos_diph** | **main system**: m1_pos with each diphthong split into two tokens (D-81) |
| abl_a_pos / abl_b_pos | m1_pos + 40% pace-augmented clips, LJ-derived (A) or dev-accent-derived (B) pace (D-79) |

Held-out test accents (Arabic, Korean, Vietnamese), 1,248 references, after `freeze-v1`
(full CIs in `results/<system>/test/summary.json`, paired tests in `results/paired/`, D-86):

| System | Word-boundary MAE (ms) | Pause F1 | WER | US prob | UTMOS | Spk sim LJ |
| --- | --- | --- | --- | --- | --- | --- |
| L2 references | | | 0.063 | 0.486 | | |
| M0 | 31.9 | 0.726 | 0.093 | 0.746 | 3.96 | 0.766 |
| M1 | 24.2 | 0.822 | 0.038 | 0.734 | 4.19 | 0.805 |
| m1_pos | 24.6 | 0.826 | 0.044 | 0.781 | 4.17 | 0.806 |
| **m1_pos_diph** | **23.9** | **0.835** | **0.029** | 0.762 | **4.20** | 0.800 |
| abl_a_pos | 24.7 | 0.825 | 0.039 | 0.665 | 4.06 | 0.801 |
| abl_b_pos | 24.7 | 0.826 | 0.045 | 0.767 | 4.08 | 0.796 |

Output length equals the reference length exactly for every system and reference. LJ test
ground truth: WER 0.019, UTMOS 4.36. Ablation B used pace statistics from dev-accent speakers;
all reported numbers are on held-out accents. Neither ablation improved timing; both lowered UTMOS.

## Commands in run order

```bash
# --- laptop, days 1-2 ----------------------------------------------------------------------
make env && make test
make data            # T2: checks, VITS filelists, splits, eval IDs, manifest skeletons
make lexicon         # T3: pinned MFA/NLTK assets + lexicon (all phones in MFA's phone set)
make align-lj        # T4: MFA on LJSpeech (CPU, ~15 min), QC, filled lj_* manifests
make stats           # T4: stats/phone_logdur.json
make align-l2-dev    # T5: MFA on all dev utterances (speaker adaptation) -> l2_dev.jsonl
make align-dev-eval  # T5: MFA vs manual boundaries per accent -> reports/mfa_l2_dev.md
make m0-demo         # T6: FastSpeech2 baseline on 5 dev references
make overfit         # T7: one-utterance overfit + ASR intelligibility check
make smoke           # T8: every config 200 steps at batch 4, resume, benchmark
make pace-stats-b    # T9: stats/l2_dev_pace.json (dev accents only)
make aug-a aug-b     # T9: augmented sets (time-varying WSOLA, D-67) + MFA verification + UTMOS filter
make edge            # T10: 21-case edge suite (CKPT=m0|path, DEVICE=cuda|cpu)
make assets-eval     # pinned Section 8 models (ASR, ECAPA, accent ID, UTMOS via torch.hub)
make eval SYSTEM=m0 SPLIT=dev   # T11 (ARGS=--no_utmos if UTMOS is not available)
make profile CKPT=runs/smoke_m1/ckpt_last.pt DEVICE=cuda   # T12 (laptop numbers)
make bundle          # T13: export/bundle.tar (+ check-data on a fresh unpack); BUNDLE_AUG=0 = light
git tag smoke-ok      # after T8; then `git tag gpu-ready` on the commit the GPU run uses

# --- GPU instance, day 2: one L40, M1 first, ablations optional later (docs/HANDOFF.md, D-68) ---
git clone <repo> meeami-tts && cd meeami-tts && git checkout gpu-ready
bash scripts/setup_gpu.sh
bash scripts/fetch_bundle.sh <url-or-path>/bundle.tar
make check-data
conda run -n tts python scripts/train.py --config configs/m1.yaml --benchmark 300
#   -> set train.max_steps in configs/base.yaml on the laptop, commit, push; git pull here
tmux new -d -s train 'conda run --no-capture-output -n tts python scripts/train.py --config configs/m1.yaml 2>&1 | tee logs/m1.log'
#   after a crash: add --resume. Ablations (optional): branching from export/m1, HANDOFF.md phase 2
bash scripts/export_run.sh m1        # -> export/m1.tar.gz ; rsync to the laptop

# --- GPU instance, phase 2: branches from M1, 20k steps each, fresh optimizer (HANDOFF.md) ---
for e in m1_cont m1_pos; do conda run -n tts python scripts/train.py --config configs/$e.yaml \
    --init_from runs/m1/ckpt_last.pt --max_steps 20000; done          # 2a: +pos vs plain continuation
for e in abl_a_pos abl_b_pos m1_pos_diph; do ... same command ...; done # 2b ablations, 2c diphthong split
for e in m1_pos m1_cont abl_a_pos abl_b_pos m1_pos_diph; do bash scripts/export_run.sh $e; done

# --- laptop, days 3-5 -----------------------------------------------------------------------
for e in m1 m1_pos m1_pos_diph abl_a_pos abl_b_pos; do sha256sum -c export/$e.tar.gz.sha256 && tar -xzf export/$e.tar.gz; done
make tune SPLIT=dev                  # writes cap / noise_scale into configs/inference.yaml (D-72)
make eval-all SPLIT=dev              # SYSTEMS = m0 m1 m1_pos m1_pos_diph abl_a_pos abl_b_pos
make phone-dyn PD_CKPTS="export/m1_pos/ckpt_last.pt ..."   # within-phone movement vs LJ (D-76)
make paired CONTROL=m1_pos SYSTEMS_CMP="abl_a_pos abl_b_pos m1_pos_diph"   # paired tests (D-82)
make edge CKPT=export/m1_pos_diph/ckpt_last.pt DEVICE=cuda   # and DEVICE=cpu: 21/21
git tag freeze-v1                    # code, configs and checkpoints frozen (results/checkpoints.sha256)
make eval-all SPLIT=test             # held-out accents; also draws plot 8 (dev vs test pace)
make paired CONTROL=m1_pos SYSTEMS_CMP="abl_a_pos abl_b_pos m1_pos_diph" SPLIT=test
make profile DEVICE=cuda ; make profile DEVICE=cpu ; make profile DEVICE=cpu ARGS="--threads 1"
make demo                            # out/demo_site/index.html (+ backup outputs per sample)
python scripts/infer.py --text "..." --ref ref.wav --out out/ --ckpt export/m1_pos_diph/ckpt_last.pt
```

## Inference

```bash
python scripts/infer.py --text "For the 20th time that evening..." --ref ref.wav --out out/ \
    --ckpt export/m1_pos_diph/ckpt_last.pt   # main system; --ckpt m0 for the FastSpeech2 baseline
python scripts/infer.py --pairs_dir their_files/ --out out/ --ckpt export/m1_pos_diph/ckpt_last.pt
```

Options: `--config configs/inference.yaml`, `--aligner mfa|ctc`,
`--transfer elastic|proportional|phone_copy`, `--cap <float|none>`, `--noise_scale`,
`--device cuda|cpu`. Per pair: `out.wav` (exactly round(N_ref·22050/sr_ref) samples),
`timings.json` (reference and intended output word times, aligner used, latency, RTF) and
`timing.png`. Any failure prints a one-line reason and continues with the next pair.

## Known failure modes

- `mfa align_one` takes ~10-15 s per reference (model load); it dominates end-to-end latency.
- If MFA fails, returns `spn` or misses a word, word boundaries come from the MMS_FA CTC aligner,
  which is coarser (~53 ms word-boundary MAE vs ~21 ms for MFA on dev references).
- Extreme stretches are capped (2.5x the native word duration, tuned on dev, D-72); the excess
  becomes a pause.
- Part of an accent lives in timing: speech forced onto an L2 reference's word timing sounds less American
  than it could (dev US prob: L2 references 0.30, main-system outputs 0.74).
- ARPAbet diphthongs are single tokens, so the plain VITS-MFA model flattened them ("cry" heard
  as "cra", D-80). The main system splits each diphthong into two tokens with a fixed 50/50
  duration split (D-81); short unstressed diphthongs get only 1-2 frames per half.
- The accent-ID US probability is near-binary per utterance and flips on inaudible changes (D-84);
  read it as a population average, not per sample.
- Pace augmentation (Ablations A/B) with a WSOLA stretcher lowered UTMOS and did not improve
  timing on dev (D-83): the stretched clips carry artifacts the model learns.
- `phone_copy` needs MFA phone durations; with the CTC fallback it runs as `elastic`.
- Reference audio is decoded with libsndfile, then ffmpeg; anything neither can read fails cleanly.

## Licences

VITS (jaywalnut310, MIT) and FastSpeech2/HiFi-GAN (ming024, MIT) code is vendored under
`tts/models/` with their licence files. L2-ARCTIC (CC BY-NC 4.0) derived clips in
`tests/edge_cases/` come from dev speakers only.
