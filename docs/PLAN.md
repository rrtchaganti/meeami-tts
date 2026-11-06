# Meeami Round 2: Time-Aligned US-English TTS — Project Plan

Nov 6, 2026 · @CHAGANTI RAJA RAVI TEJA

## 1. Task summary

Build a single-speaker TTS that speaks the input text in the LJSpeech voice with a US accent, on the word timing of a reference recording of the same text by a non-US speaker. The reference contributes timing only.

| Output property | Source |
| --- | --- |
| Words | Input text |
| Pronunciation (phone sequence) | `lexicon.txt`: first CMUdict pronunciation, g2p\_en for OOV words |
| Voice, accent, intonation | The model, trained on LJSpeech only |
| Word start/end times, pauses, total length | Reference audio, via MFA |

Acceptance criteria from the assignment: (1) US accent, exactly the input words; (2) LJSpeech voice; (3) output length equals reference length, and each word starts and ends at about the reference time, pauses included.

Systems: M0 (FastSpeech2 baseline), M1 (VITS with MFA durations, main), Ablation A and Ablation B (pace augmentation), optional M1+pos. Details in Section 4.

This doc is the project spec. Export it to `docs/PLAN.md` in the repo; `CLAUDE.md` (Section 10) tells Claude Code to read the relevant section before each task. Code comments and task IDs refer to section numbers.

## 2. Data and splits

Train only on LJSpeech. Split L2-ARCTIC by accent: dev accents may be used for tuning and Ablation B; test accents are read only after the freeze tag.

### Datasets

| Dataset | Content | Path | Use |
| --- | --- | --- | --- |
| LJSpeech 1.1 | 13,100 clips, one US female speaker, 22.05 kHz mono; `metadata.csv` column 3 = normalized text | `data/raw/LJSpeech-1.1/` | Training |
| [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus-docs) v5 | 24 speakers, 4 per L1; \~1,130 read ARCTIC prompts each; 44.1 kHz; per speaker `wav/`, `transcript/`, `annotation/` (manual TextGrids, \~150 per speaker), `textgrid/` | `data/raw/l2arctic/<SPK>/` | Dev and test references. Downloaded manually after the access form |
| L2-ARCTIC suitcase corpus | Spontaneous speech, 22 speakers, manual TextGrids | `data/raw/l2arctic/suitcase_corpus/` | Stress-test demo only; not in metrics |

### Splits (`configs/splits.yaml`)

| Split | Content |
| --- | --- |
| LJ train / val / test | The original VITS repo's LJSpeech filelists (12,500 / 100 / 500), so the pretrained checkpoint never saw val or test |
| L2 dev | Hindi: ASI, RRBI, SVBI, TNI · Mandarin: BWC, LXC, NCC, TXHC · Spanish: EBVS, ERMS, MBMPS, NJS |
| L2 test | Arabic: ABA, SKA, YBAA, ZHAA · Korean: HJK, HKK, YDCK, YKWK · Vietnamese: HQTV, PNV, THV, TLV |

- **Evaluation set:** utterance IDs that have a manual annotation for every speaker in the split (expected \~100 common sentences; a few speakers have fewer files, so take the intersection and log the count). Test = 12 speakers × common IDs.
- **Dev tuning set:** the same common IDs for the dev speakers.
- **Ablation B pool:** all scripted utterances of dev speakers.
- **Enforcement:** `evaluate.py --split test` and any loader of test-speaker audio must refuse to run unless git tag `freeze-v1` exists.

### Preprocessing

- LJSpeech audio is already 22.05 kHz mono 16-bit; use as is.
- MFA input: 16 kHz mono 16-bit WAV copies under `data/mfa/`; text as lowercase words, no punctuation except apostrophes, one `.lab` per WAV.
- Output length always comes from the original reference file, never the resampled copy.
- One normalizer (`tts/text/normalize.py`: numbers, ordinals, dates, currency, abbreviations, symbols) used for training text, MFA `.lab` files and inference input.

## 3. Forced alignment (MFA)

Use MFA's pretrained `english_us_arpa` acoustic model with our own single-pronunciation lexicon. No aligner training. The same lexicon, model and parsing rules are used for LJSpeech, L2 references and model outputs.

### Environment and models

- `envs/mfa.yml`: `montreal-forced-aligner` from conda-forge, pinned to the exact 3.x version installed on Day 1. MFA 3.4 deprecated the current `align` / `align_one` signatures and 4.0 changes them ([docs](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html)); never upgrade mid-project.
- `mfa model download acoustic english_us_arpa` and `mfa model download g2p english_us_arpa`.
- Fail fast if any lexicon phone is missing from the acoustic model's phone set (`mfa model inspect acoustic english_us_arpa`).

### Lexicon (`data/lexicon/lexicon.txt`)

- One line per word: `word<TAB>P1 P2 ...`, lowercase word, ARPAbet phones with stress digits, exactly one pronunciation.
- Source: first pronunciation from the `cmudict` package; OOV words from g2p\_en.
- Built by `make lexicon` from LJSpeech + L2-ARCTIC text. At inference, OOVs in new text are G2P'd and appended before alignment.
- The TTS front end reads the same file. Never regenerate pronunciations elsewhere.

### Commands

```bash
# LJSpeech (single speaker; parallelize by utterance)
mfa align data/mfa/lj data/lexicon/lexicon.txt english_us_arpa data/align/lj \
    --single_speaker -j 8 --beam 100 --retry_beam 400 --clean
# L2 dev speakers (one folder per speaker; speaker adaptation on)
mfa align data/mfa/l2_dev data/lexicon/lexicon.txt english_us_arpa data/align/l2_dev \
    -j 8 --beam 100 --retry_beam 400 --clean
# Single reference at inference
mfa align_one ref.wav ref.lab data/lexicon/lexicon.txt english_us_arpa ref.TextGrid \
    --beam 100 --retry_beam 400
```

Python calls MFA through `conda run -n mfa ...`. The wide beam is deliberate: one public benchmark measured roughly half the word-boundary error versus the default beam for this model ([source](https://huggingface.co/nyralabs/nyra_forced_aligner_en)).

### TextGrid parsing rules (`tts/align/textgrid.py`)

1. Read with `praatio`. Phone labels `""`, `sil`, `sp` are silence; `spn` means a failed word (drop the file for LJSpeech; trigger the fallback at inference).
2. Silence at the start and end becomes `sil`. Internal silence becomes `sp` if it is at least 3 frames (\~35 ms); shorter silence is merged into the preceding phone. Identical rule at training and inference.
3. Frames: boundary frame = `round(t × 22050 / 256)` on cumulative times; durations are differences. Take the target frame count from the training spectrogram function itself, not a formula, and adjust the last token so durations sum to it.
4. Each phone token stores the index of its word; `sil` / `sp` store -1.

### Quality checks (`reports/align_lj.md`)

Drop LJSpeech files that MFA failed on, that contain `spn`, that have a non-silence phone over 0.5 s, or whose duration sum differs from the frame count by more than 2 before adjustment. Report counts per reason; expect at least 98% kept.

### MFA error on accented speech (`reports/mfa_l2_dev.md`)

Align the dev speakers' annotated utterances and compare with the manual TextGrids (`annotation/`): word tier for word boundaries, phone tier for phone boundaries (use interval times only; ignore error tags in the labels). Report mean absolute boundary error and % within 20 / 50 ms, for words and phones separately, per accent.

### Fallback aligner

torchaudio's `MMS_FA` pipeline gives word boundaries when MFA fails or returns `spn`, and serves as the independent aligner in Section 8.

## 4. Models and training

M0 is inference-only. M1, A and B are fine-tuned from the same pretrained VITS checkpoint, with identical hyperparameters and step counts; only their training data differs.

### Experiment matrix

| ID | Config | System | Training data |
| --- | --- | --- | --- |
| M0 | `m0.yaml` | Pretrained FastSpeech2 + HiFi-GAN, reference durations injected | None (inference only) |
| M1 | `m1.yaml` | VITS with MFA durations | LJ train |
| M1+pos | `m1_pos.yaml` | M1 with `use_pos: true` (optional) | LJ train |
| A | `abl_a.yaml` | M1 architecture | LJ train + augmented set A (Section 6) |
| B | `abl_b.yaml` | M1 architecture | LJ train + augmented set B (Section 7) |

All configs inherit `configs/base.yaml` and override only what differs. All systems share the lexicon, alignments, inference pipeline (Section 5) and evaluation (Section 8).

### M0: FastSpeech2 baseline

- Code and weights: ming024/FastSpeech2 pretrained LJSpeech checkpoint plus its bundled HiFi-GAN LJSpeech vocoder. Same frame rate as ours (22,050 Hz, hop 256).
- Map our tokens to its symbol set (ARPAbet with stress; `sil`, `sp` to its silence symbols). Fail on any unmapped symbol.
- Inject reference durations by passing them as duration targets to the variance adaptor (its training code path); pitch and energy stay predicted.

### M1: VITS with MFA durations

Base code: jaywalnut310/vits. Start weights: its official LJSpeech generator checkpoint. Initialize the discriminators from scratch if no discriminator checkpoint is available.

Required changes:

1. Remove MAS (the `monotonic_align` module and its Cython build). The alignment matrix is a one-hot expansion of the MFA frame durations.
2. Set `add_blank: false`; the original LJSpeech config intersperses blank tokens, which have no MFA duration.
3. Symbols (`tts/text/symbols.py`): `pad`, `sil`, `sp`, then 69 ARPAbet tokens (15 vowels × 3 stress levels + 24 consonants); `n_vocab` = 72. Re-initialize the text embedding; load every other weight whose shape matches and log skipped keys.
4. Replace the stochastic duration predictor with VITS's deterministic `DurationPredictor` (2 conv layers, 256 filters, kernel 3, dropout 0.5) on the detached text-encoder output, predicting log duration. Loss: MSE on log frames, masked to exclude `sil`, padding and augmented samples (`is_aug` in the manifest).
5. `use_pos` flag (M1+pos only): expand the text-encoder hidden states by durations first, add a learned embedding of each frame's relative position in its phone (16 bins of (k + 0.5) / d), then apply the 1×1 projection to μ and log σ. With `use_pos: false`, keep the original order (project, then expand).
6. Data loader reads the manifests defined in Section 10.

### Losses

```latex
L_G = 45\,L_{mel} + L_{KL} + L_{dur} + L_{adv} + 2\,L_{fm}
```

L\_mel is L1 on mel spectrograms; L\_KL ties the posterior to the duration-expanded prior; L\_dur is described above; L\_adv and L\_fm are the least-squares GAN and feature-matching losses. Discriminators use the least-squares GAN loss.

### Hyperparameters (`configs/base.yaml`)

| Key | Value |
| --- | --- |
| Audio | 22,050 Hz; n\_fft and window 1024; hop 256; 80 mels; linear spectrogram (513 bins) for the posterior encoder |
| Model | VITS LJSpeech defaults: hidden 192, filter 768, 2 heads, 6 encoder layers |
| Segment | 8192 samples |
| Batch | 32 on L40/A100; 4 for laptop smoke tests |
| Optimizer | AdamW, lr 1e-4, betas (0.8, 0.99), eps 1e-9, lr decay 0.999875 per epoch |
| Precision | bf16 autocast; STFT, mel and loss computations in fp32 |
| Checkpoints | Every 2,000 steps plus `ckpt_last.pt`; keep the last 3 |
| Validation | Every 2,000 steps: val losses and audio for 4 fixed LJ val utterances to TensorBoard |
| Steps | `max_steps` set from the 300-step benchmark to fit 5–10 hours; identical for M1, A, B |
| Warm-up option | `freeze_decoder_steps` (default 0; use \~3,000 if the new embedding settles slowly) |
| Seed | 1234 |

## 5. Inference pipeline

Copy word boundaries and pauses from the reference; inside each word, distribute time across phones with native LJSpeech elasticity. All systems use this pipeline (`tts/infer/`), configured by `configs/inference.yaml`.

### Steps

1. **Load the reference.** Any format and sample rate; convert to mono. Do not trim. Record N\_ref samples and sr\_ref.
2. **Prepare text.** Normalize (Section 2), tokenize to words, G2P and append OOVs to the lexicon, write a 16 kHz WAV and `.lab` for MFA.
3. **Align** with `mfa align_one` (Section 3). If it fails, returns `spn`, or leaves any transcript word without an interval, use `MMS_FA` word boundaries.
4. **Build the token timeline:** `sil` before the first word, each word's phones, `sp` for internal gaps of at least 3 frames (shorter gaps go to the preceding word), `sil` after the last word.
5. **Predict native durations** d̂ᵢ with the model's duration predictor on the full token sequence, `sp` tokens included.
6. **Fit each word** spanning D frames: dᵢ = exp(log d̂ᵢ + z·σᵢ), solving for one scalar z by bisection on \[−5, 5\] (tolerance 0.5 frame). σᵢ comes from `stats/phone_logdur.json`: per-phone mean and std of log duration over LJ train, excluding silence; phones with fewer than 50 occurrences use the median σ. Minimum 1 frame per phone.
7. **Stretch cap:** if D > c × Σd̂ᵢ, fit to c × Σd̂ᵢ and insert the remainder as `sp` after the word.
8. **Quantize** cumulative boundaries to frames so total frames match round(N\_ref × 22050 / sr\_ref / 256).
9. **Synthesize:** text encoder → expand by durations → sample prior with noise scale → inverse flow → decoder. Pad or trim to exactly round(N\_ref × 22050 / sr\_ref) samples.
10. **Write** `out.wav`, `timings.json` (reference word times, intended output word times, which aligner was used) and `timing.png` (Section 8, plot 1).

### Parameters (`configs/inference.yaml`)

| Key | Default | Tuning grid (dev accents only) |
| --- | --- | --- |
| `transfer` | `elastic` | `elastic`, `proportional`, `phone_copy` |
| `cap` | 2.0 | 1.4, 1.6, 1.8, 2.0, 2.5, none |
| `noise_scale` | 0.667 | 0.333, 0.5, 0.667 |
| `min_sp_frames` | 3 | fixed |
| `aligner` | `mfa` | `mfa`, `ctc` |

Transfer variants: `elastic` is step 6; `proportional` scales d̂ᵢ uniformly to fit D; `phone_copy` uses the reference's MFA phone durations directly (the assignment's suggested approach). Word boundaries are identical across variants by construction, so their comparison isolates accent, naturalness and intelligibility.

## 6. Ablation A: target-agnostic pace augmentation

Augmented copies of LJ train built from LJSpeech statistics alone; no accented data is used. Hypothesis: better naturalness and intelligibility at high stretch ratios, with no loss near 1.0×.

### Generation (`tts/augment/`, `make aug-a`)

1. **Elasticity:** σₚ from `stats/phone_logdur.json` (Section 5, step 6).
2. **Pace:** per copy, utterance pace z\_u \~ Uniform(−1.5, 1.5); per word, z\_w = z\_u + ε with ε \~ N(0, 0.5²).
3. **Phone factors:** rᵢ = exp(z\_w · σᵢ), clipped to \[0.6, 2.5\]. `sil` unchanged.
4. **Pauses:** multiply each `sp` by a factor drawn from Uniform(1, 3); at word boundaries followed by punctuation in the text but with no `sp`, insert a pause of 100–600 ms with probability 0.1. Build lengthened or inserted silence by looping that utterance's own silence (room tone) with 10 ms crossfades, never digital zeros.
5. **Stretch:** convert phone factors into a sample-level time map anchored at LJSpeech phone boundaries and apply a time-varying pitch-preserving stretch with Rubber Band (`rubberband-cli` ≥ 3.0 and `pyrubberband.timemap_stretch`, R3 "fine" engine). Pad or trim to the exact target length.
6. **Durations:** computed from the target boundaries with the same cumulative rounding as Section 3; MFA is not re-run.
7. **Copies:** 2 per LJ train utterance (seeds 1234, 1235), 22.05 kHz WAVs in `data/aug/a/`, rows in `data/manifests/aug_a.jsonl` with `is_aug: true`, `source_id` and the sampled z\_u.

### Filtering and training

- Score each augmented clip with UTMOS and with speaker similarity to its original (models from Section 8); drop the 5% with the largest UTMOS drop. Write the filter report to `reports/aug_a.md` with 20 random clips listed for listening.
- Training sampler: 60% LJ train rows, 40% augmented rows per batch.
- Duration loss is masked on augmented rows (Section 4, change 4) so the duration predictor keeps learning native durations.
- Same `max_steps` as M1. Run with and without `use_pos` only if a fourth GPU slot is available.

## 7. Ablation B: target-aware pace augmentation (dev split only)

Same generator as Ablation A, but word paces and pauses are sampled from statistics of dev-accent speech (Hindi, Mandarin, Spanish). Evaluated on the unseen test accents, so a gain means pace knowledge transfers across accents.

### Pace statistics (`make pace-stats-b` → `stats/l2_dev_pace.json`)

1. Align all scripted utterances of the 12 dev speakers (`make align-l2-dev`, Section 3 command).
2. For each word, ρ\_w = reference word frames ÷ Σ exp(μₚ) over its phones, with μₚ the LJ mean log duration from `stats/phone_logdur.json`. This uses no trained model, so B can be generated before M1 finishes.
3. Store: empirical distribution of utterance mean log ρ; per-word residuals around it with their lag-1 autocorrelation; probability of an `sp` after a word; empirical `sp` length histogram.

### Generation (`make aug-b`)

1. Per copy, sample an utterance mean log ρ from the empirical distribution, then word residuals as an AR(1) process with the fitted lag-1 correlation and residual variance.
2. Convert each word's ρ\_w to phone factors with the elastic rule: solve z\_w so the word hits ρ\_w, then rᵢ = exp(z\_w · σᵢ), clipped to \[0.6, 2.5\].
3. Pauses: at each word boundary, add or lengthen an `sp` with the empirical probability, length drawn from the empirical histogram, room tone as in Ablation A.
4. Everything else is identical to Ablation A: time-map stretching, durations, 2 copies (seeds 1234, 1235), `data/aug/b/`, `data/manifests/aug_b.jsonl`, UTMOS filter, 60/40 sampler, masked duration loss, same `max_steps`.

### Rules

- Use only word-level ratios and pause statistics. Never L2 phone-level timing inside words; that is part of the accent.
- Never read test-accent audio for this ablation.
- README and slides state: "Ablation B used pace statistics from dev-accent speakers; all reported numbers are on held-out accents."
- After `freeze-v1`, plot dev and test ρ distributions together (Section 8, plot 8).

## 8. Evaluation

Every system is evaluated on the same reference set (Section 2): dev for tuning, test once after `freeze-v1`. Per-utterance metrics go to `results/<system>/<split>/per_utt.csv`; summaries with bootstrap 95% CIs (1,000 resamples over utterances) go to `summary.json`; paired Wilcoxon signed-rank tests against M1 go to `results/<split>_comparison.csv`. Everything is broken down by accent.

### Evaluation models (pin versions in `envs/tts.yml`; T11 verifies each loads)

| Purpose | Model |
| --- | --- |
| ASR for WER/CER | faster-whisper `large-v3`; text normalized with Whisper's `EnglishTextNormalizer`; WER via `jiwer` |
| Predicted MOS | SpeechMOS `utmos22_strong` (torch.hub) |
| Speaker similarity | SpeechBrain `speechbrain/spkrec-ecapa-voxceleb` |
| Accent ID | SpeechBrain `Jzuluaga/accent-id-commonaccent_ecapa` (US-English class probability) |
| Second aligner | torchaudio `MMS_FA` |

If any model is unavailable, stop and report; never substitute a different model silently.

### Quality metrics

| Requirement | Metric | Definition |
| --- | --- | --- |
| Timing | Word-boundary error | Boundaries = every word start and end. Reference times from the manual annotation word tier; output times from `mfa align_one` on the output with the input text. Report mean absolute error (ms) and % within 20 / 50 ms. Repeat with `MMS_FA` on the output as a cross-check. |
| Timing | Length error | Output samples minus round(N\_ref × 22050 / sr\_ref); must be 0 |
| Timing | Pause F1 | A pause = silence ≥ 100 ms between two words; F1 of output pauses vs reference pauses at word-boundary level |
| Exact words | WER, CER | ASR on the output vs input text. Also on the reference audio and on LJ test ground truth, for comparison |
| US accent | US probability | Accent-ID on output vs reference |
| LJSpeech voice | Speaker similarity | Cosine of ECAPA embeddings: output vs the mean embedding of 100 LJ test clips (high), output vs the reference speaker (low) |
| Naturalness | Predicted MOS | UTMOS on every output, plus LJ test ground truth as the ceiling |

### Efficiency metrics (`make profile`)

| Metric | Definition |
| --- | --- |
| Parameters | Per module: inference path (text encoder, duration predictor, flow, decoder) and training-only (posterior encoder, discriminators) |
| MACs | ptflops or fvcore on the inference path for 1 s and 5 s outputs, reported as GMACs per second of audio. Include one decoder layer worked by hand (K × C\_in × C\_out × L\_out) |
| Latency | Per stage (normalize + G2P, MFA, text encoder + durations, flow + decoder), batch 1, 10 warm-up + 50 timed runs, p50 / p90, CUDA-synchronized. Devices: rented L40/A100, laptop RTX 4060, CPU with 1 thread and all threads |
| RTF | Processing time ÷ output duration, model-only and end-to-end (including MFA) |

### Plots (`results/plots/`)

1. Per-sample word timing: reference and output words as bars on one time axis above both spectrograms
2. Reference vs output boundary scatter with y = x, plus error histograms per accent
3. MFA error on accented speech: word vs phone boundaries, per accent (from `reports/mfa_l2_dev.md`)
4. UTMOS, WER and speaker similarity vs stretch ratio, one line per system
5. Stretch-cap tradeoff on dev: timing error vs UTMOS across the `cap` grid
6. Per-accent WER and US probability, reference vs output
7. Transfer variants: `phone_copy` vs `proportional` vs `elastic` on US probability, WER, UTMOS
8. Dev vs test ρ distributions (after freeze)
9. Efficiency: parameters by module, RTF vs utterance length, stacked latency breakdown
10. Training curves: mel, KL, duration losses per system

## 9. Demo, deliverables and live inference

The live demo runs one command on unseen files, on the laptop, so robustness matters more than polish.

### Inference command (`scripts/infer.py`)

```bash
python scripts/infer.py --text "..." --ref ref.wav --out out/ --ckpt export/m1/ckpt_last.pt      # single pair
python scripts/infer.py --pairs_dir their_files/ --out out/ --ckpt export/m1/ckpt_last.pt         # folder of x.wav + x.txt
```

Options: `--config configs/inference.yaml` (default), `--aligner mfa|ctc`, `--transfer elastic|proportional|phone_copy`, `--cap`, `--noise_scale`, `--device cuda|cpu`. Outputs per pair: `out.wav`, `timings.json`, `timing.png`; prints per-stage latency and RTF. Any failure prints a one-line reason and continues with the next pair.

### Edge-case suite (`tests/edge_cases/`, `make edge`)

Small committed set of WAV + TXT pairs, recorded or converted on the laptop:

| Case | Example |
| --- | --- |
| Text normalization | Numbers, dates, currency, abbreviations, a name, an OOV word, no punctuation |
| Audio formats | MP3, 8 kHz, 48 kHz, stereo, clipped, noisy, reverberant |
| Timing | Over 20 s, long internal pauses, very fast speech, one-word utterance |
| Alignment failure | Transcript with one word not spoken; a false start |

Pass = every case produces output of exact length, or a clean fallback with the reason logged.

### Deliverables

- [ ] Repository: README with setup, data split, and every command in run order; pinned environments; links to exported checkpoints
- [ ] Demo page (`make demo`, static HTML): per sample, accent, speaker, text, reference audio, output audio per system, timing plot. All 3 test accents; dev samples labelled "dev"; suitcase clips and your own recordings labelled "stress test"
- [ ] Results tables and plots (Section 8)
- [ ] Slides covering the four key questions
- [ ] Backup outputs for every demo sample, in case the live environment fails

### Slide outline

1. Problem and what comes from where
2. Data strategy and accent-disjoint split
3. Alignment and MFA error on accented speech
4. Architecture choice and VITS-MFA walkthrough
5. Training strategy and losses
6. Inference: word-level transfer, elastic redistribution, stretch cap
7. Ablations A and B; transfer variants
8. Quality results per accent with CIs
9. Params, MACs, latency, RTF
10. Limitations and future work
11. Live demo and code walkthrough

### Live-readiness checklist

- [ ] Fresh clone + `make env` works on a clean machine
- [ ] MFA models and all checkpoints available offline
- [ ] `make edge` passes on GPU and on CPU
- [ ] Known failure modes written in the README

## 10. Execution: laptop, GPU instance and Claude Code

Everything except training runs on the laptop (RTX 4060, 8 GB). GPU instances are rented once, on Day 2, for 5–10 hours of training plus GPU profiling, running commands already tested on the laptop. Claude Code writes the code one task at a time; each task ends with its acceptance check passing and a commit.

### CLAUDE.md (repo root)

```markdown
# CLAUDE.md
Read the docs/PLAN.md sections named in the task before writing code.
- Test speakers (ABA SKA YBAA ZHAA HJK HKK YDCK YKWK HQTV PNV THV TLV): never read their audio
  unless git tag freeze-v1 exists.
- data/lexicon/lexicon.txt is the only source of pronunciations.
- Two conda envs: `mfa` (MFA only, called via `conda run -n mfa`) and `tts` (everything else).
- All parameters live in configs/*.yaml (inheriting base.yaml); no hard-coded paths or hyperparameters.
- Every module gets a unit test in tests/ and a make target. Run `make test` before calling a task done.
- Laptop GPU has 8 GB: smoke runs use batch 4.
- After tag freeze-v1, never change model code or experiment configs.
- If a pinned package or model is unavailable, stop and report; never substitute silently.
- End each task with: files changed, commands run, acceptance-check result.
```

### Fixed conventions

**Repository layout**

```
meeami-tts/
  CLAUDE.md  Makefile  README.md  docs/PLAN.md
  envs/      mfa.yml  tts.yml
  configs/   base.yaml  splits.yaml  inference.yaml  m0.yaml  m1.yaml  m1_pos.yaml  abl_a.yaml  abl_b.yaml
  tts/       text/  align/  models/{vits_mfa,fastspeech2}/  augment/  infer/  eval/
  scripts/   prepare_data.py  train.py  infer.py  evaluate.py  profile.py  make_demo.py
             setup_gpu.sh  bundle.sh  fetch_bundle.sh  export_run.sh
  tests/     unit tests + edge_cases/
  data/      raw/  mfa/  align/  lexicon/  manifests/  aug/{a,b}/        (git-ignored)
  stats/     phone_logdur.json  l2_dev_pace.json
  reports/   alignment and augmentation reports
  runs/      <exp>/   training outputs (git-ignored)
  export/    <exp>/   downloaded checkpoints (git-ignored)
  results/   <system>/<split>/  plots/
```

**Manifests** (`data/manifests/{lj_train,lj_val,lj_test,aug_a,aug_b,l2_dev,l2_test}.jsonl`), one JSON object per line:

```json
{"id": "LJ001-0001", "wav": "data/raw/LJSpeech-1.1/wavs/LJ001-0001.wav", "text": "printing in the only sense ...",
 "words": ["printing", "in", "..."], "phones": ["sil", "P", "R", "IH1", "..."], "durations": [14, 6, 5, 7],
 "word_ids": [-1, 0, 0, 0], "n_frames": 812, "speaker": "LJ", "accent": "us", "split": "train",
 "is_aug": false, "source_id": null}
```

L2 rows add `ref_words_manual` and `ref_words_mfa`: lists of `{"word", "start", "end"}` in seconds, plus `sr` and `n_samples` of the original file.

**Runs:** `runs/<exp>/` holds `config.yaml` (resolved), `git_commit.txt`, `train.log`, `tb/`, `ckpt_<step>.pt`, `ckpt_last.pt`. `train.py` flags: `--config`, `--resume`, `--init_from <ckpt>`, `--tag <name>`, `--max_steps`, `--batch_size`, `--benchmark <n>`, plus `key=value` overrides.

### Laptop tasks (Days 1–2)

| # | Claude Code builds | Command | Done when |
| --- | --- | --- | --- |
| T1 | Pinned `mfa` and `tts` envs, Makefile skeleton, CLAUDE.md, docs/PLAN.md | `make env && make test` | `mfa version` works; CUDA visible in `tts` |
| T2 | Data download, VITS LJ filelists, `splits.yaml`, empty manifests | `make data` | 13,100 LJ clips; 24 L2 speakers in correct splits; common annotated IDs logged |
| T3 | Normalizer, lexicon builder, symbols | `make lexicon` | Unit tests pass for numbers, dates, currency, abbreviations, OOVs; all lexicon phones in MFA's phone set |
| T4 | LJ alignment, TextGrid parser, filled LJ manifests, phone stats | `make align-lj && make stats` (CPU, background) | `reports/align_lj.md` shows ≥ 98% kept; duration sums equal `n_frames`; `stats/phone_logdur.json` written |
| T5 | Dev alignment and MFA-vs-manual evaluation | `make align-l2-dev && make align-dev-eval` | `reports/mfa_l2_dev.md` with word and phone errors per accent |
| T6 | M0 with injected durations | `make m0-demo` | Output, `timings.json` and timing plot for 5 dev references |
| T7 | VITS-MFA model (Section 4 changes 1–6) | `pytest tests/test_vits_mfa.py` | Forward/backward pass; checkpoint loads with only expected keys skipped; one-utterance overfit (\~500 steps, batch 1) becomes intelligible |
| T8 | `train.py` (flags above), TensorBoard logging | `make smoke` | Every config trains 200 steps at batch 4; resume works; `--benchmark 20` prints s/step |
| T9 | Augmentation A, pace stats B, augmentation B, filter | `make aug-a && make pace-stats-b && make aug-b` | Duration sums equal frames for every augmented row; reports list 20 clips that sound clean |
| T10 | Inference pipeline and CLI (Section 5), edge-case suite | `pytest tests/test_infer.py && make edge` | Exact output length on every case; fallback path exercised |
| T11 | Evaluation harness, metrics, plots | `make eval SYSTEM=m0 SPLIT=dev` | All evaluation models load; metrics and plots produced for M0 on dev |
| T12 | Profiling | `make profile CKPT=... DEVICE=cuda` | Params, MACs, latency breakdown and RTF saved to `results/profile_<device>.json` |
| T13 | GPU bundle and data check | `make bundle` | `make check-data` passes on a fresh unpack in a temp directory |
| T14 | Demo page generator | `make demo` | Static page builds from `results/` and demo outputs |

Tag `smoke-ok` after T8. `pace-stats-b` needs MFA, so it always runs on the laptop. The stretching steps (`aug-a`, `aug-b`) can run on the instance instead if it has many more CPU cores; time 100 files on each machine first. T10–T12 and T14 continue while Day 2 training runs.

### GPU instance runbook (Day 2)

Rent three L40 or A100 instances, one run each, with a CUDA image, 8+ CPU cores and about 100 GB of disk. Paying for three GPUs for 8 hours costs about the same as one for 24, but finishes in a third of the time. With only one GPU, use the branching design in step 5.

1. **Set up the instance** (about 15 minutes):

   ```bash
   git clone <repo> meeami-tts && cd meeami-tts && git checkout smoke-ok
   bash scripts/setup_gpu.sh           # creates the tts env from envs/tts.yml; MFA is not needed for training
   bash scripts/fetch_bundle.sh <url>  # or: rsync -avP bundle.tar.gz user@host:~/meeami-tts/
   make check-data                     # checksums, file counts, duration sums
   # only if augmentation was not bundled: make aug-a aug-b  (setup_gpu.sh installs rubberband-cli; stats/ comes from the laptop)
   ```
2. **Benchmark before launching:**

   ```bash
   python scripts/train.py --config configs/m1.yaml --benchmark 300
   ```

   It prints seconds per step and the steps that fit your time budget. Set `max_steps` in all three configs to that number, so every run gets the same step count.
3. **Launch inside tmux,** so runs survive a dropped SSH connection:

   ```bash
   tmux new -d -s train 'python scripts/train.py --config configs/m1.yaml 2>&1 | tee logs/m1.log'
   # instance 2: configs/abl_a.yaml   instance 3: configs/abl_b.yaml
   ```
4. **Monitor.** Forward TensorBoard to the laptop with `ssh -L 6006:localhost:6006 user@host`, then open `localhost:6006`. Listen to validation audio every 10k steps and check `nvidia-smi`. If an instance dies, rerun the same command with `--resume`; checkpoints are saved every 2k steps.
5. **One-GPU branching design (fallback).** Train M1 for most of the budget, then branch three equal-length continuations from the same checkpoint and run them one after another:

   ```bash
   python scripts/train.py --config configs/m1.yaml        --init_from runs/m1/ckpt_last.pt --tag m1_cont
   python scripts/train.py --config configs/abl_a.yaml     --init_from runs/m1/ckpt_last.pt
   python scripts/train.py --config configs/abl_b.yaml     --init_from runs/m1/ckpt_last.pt
   ```
6. **Profile on the GPU before shutting down.** Data-center latency and RTF numbers can only be measured here:

   ```bash
   make profile CKPT=runs/m1/ckpt_last.pt DEVICE=cuda
   ```
7. **Export and download,** then terminate:

   ```bash
   bash scripts/export_run.sh m1      # last checkpoint, config, logs, TensorBoard events, profile report -> export/m1.tar.gz
   rsync -avP user@host:~/meeami-tts/export/ ./export/   # run on the laptop; verify checksums, then terminate the instance
   ```

### Back on the laptop (Days 3–5)

```bash
make tune SPLIT=dev                               # grid over cap and noise_scale (Section 5); writes chosen values to configs/inference.yaml
make eval-all SPLIT=dev                           # sanity check of all systems with the tuned settings
git tag freeze-v1                                 # G3: code, configs and checkpoints frozen
make eval-all SPLIT=test                          # M0, M1, A, B on test accents; all plots
make profile CKPT=export/m1/ckpt_last.pt DEVICE=cuda   # laptop GPU numbers, alongside the data-center ones
make demo                                         # builds the static demo page
python scripts/infer.py --text "..." --ref ref.wav --out out/ --ckpt export/m1/ckpt_last.pt   # live-demo command
```

The README documents every command above in the order they run, so a reviewer can reproduce the whole project from a fresh clone.

## 11. Timeline and risks

The plan fits 5 days: the laptop does all coding, testing and evaluation, and rented GPUs are used once, on Day 2, for 5–10 hours of training. If training needs longer than that, use shorter fine-tunes or the one-GPU branching design, and cut from the bottom of this priority order: M0, M1, duration-transfer variants, Ablation A, Ablation B, M1+pos.

&#91;embedded content: project roadmap · 6 phases, 3 gates\]

Gate G3 is the integrity line: every model, setting and threshold is fixed before test-accent audio is touched. GPU profiling runs on the instance before it is terminated (runbook step 6), so no instance is needed after Day 2.

### Risk register

| Risk | Fallback |
| --- | --- |
| L2-ARCTIC access is slow | Build and debug the pipeline on your own accented recordings; swap in L2-ARCTIC when it arrives |
| MFA install or version problems | Pinned environment or Docker; CTC aligner covers alignment meanwhile |
| VITS fine-tuning unstable or slow | M0 carries the demo; train in fp32 if mixed precision diverges; drop M1+pos first |
| Stretch artifacts hurt augmented models | UTMOS filter, smaller augmented fraction; report a negative result honestly, it is still a finding |
| Pretrained checkpoint's text embedding doesn't transfer | Re-initialize it and warm up with flow and decoder frozen for a few thousand steps |
| Interview-day files break the pipeline | Robustness checklist in Section 9, CTC fallback, pre-generated backup outputs |
