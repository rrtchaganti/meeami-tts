# meeami-tts — time-aligned US-English TTS

Speaks the input text in the LJSpeech voice with a US accent, on the word timing (word starts/ends,
pauses, total length) of a reference recording of the same text by a non-US speaker.

- Specification: [docs/PLAN.md](docs/PLAN.md) (code comments cite its sections).
- Every decision and error, with reasons: [docs/DECISIONS.md](docs/DECISIONS.md).
- Current status, blockers and the GPU runbook: **[docs/HANDOFF.md](docs/HANDOFF.md)**.

| Output property | Source |
| --- | --- |
| Words, pronunciation | input text -> normalizer -> `data/lexicon/lexicon.txt` (cmudict first pron., g2p_en OOVs) |
| Voice, accent, intonation | model trained on LJSpeech only (M1: VITS with MFA durations; M0: FastSpeech2 baseline) |
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

# --- laptop, days 3-5 -----------------------------------------------------------------------
for e in m1 abl_a abl_b; do tar -xzf export/$e.tar.gz; done
make tune SPLIT=dev                  # writes cap / noise_scale into configs/inference.yaml
make eval-all SPLIT=dev
git tag freeze-v1                    # code, configs and checkpoints frozen
make eval-all SPLIT=test
make profile CKPT=export/m1/ckpt_last.pt DEVICE=cuda
make demo                            # out/demo_site/index.html (+ backup outputs)
python scripts/infer.py --text "..." --ref ref.wav --out out/ --ckpt export/m1/ckpt_last.pt
```

## Inference

```bash
python scripts/infer.py --text "For the 20th time that evening..." --ref ref.wav --out out/ \
    --ckpt export/m1/ckpt_last.pt            # or --ckpt m0 for the FastSpeech2 baseline
python scripts/infer.py --pairs_dir their_files/ --out out/ --ckpt export/m1/ckpt_last.pt
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
- Extreme stretches are capped (default 2x the native word duration); the excess becomes a pause.
- `phone_copy` needs MFA phone durations; with the CTC fallback it runs as `elastic`.
- Reference audio is decoded with libsndfile, then ffmpeg; anything neither can read fails cleanly.

## Licences

VITS (jaywalnut310, MIT) and FastSpeech2/HiFi-GAN (ming024, MIT) code is vendored under
`tts/models/` with their licence files. L2-ARCTIC (CC BY-NC 4.0) derived clips in
`tests/edge_cases/` come from dev speakers only.
