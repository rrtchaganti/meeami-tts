# meeami-tts: time-aligned US-English TTS

Give it a sentence and a recording of that sentence read by a **non-US speaker**. It says the
sentence:
- in the **LJSpeech voice** (one US female speaker);
- with a **US accent**;
- **on the reference's timing**: every word starts and ends at about the reference time, pauses
  are kept, and the output is **exactly** as long as the reference.

```bash
make infer REF=ref.wav TEXT="Will we ever forget it."
# -> out/infer/out.wav (exact reference length), timings.json, timing.png
```

## Assignment deliverables: where each one is

| Requirement | Where |
| --- | --- |
| **Demonstration:** input text, reference and output side by side, word timings of both, ≥ 2 accents | [`samples/`](samples/README.md): 21 held-out test samples in **3 accents** (Arabic, Korean, Vietnamese). Each sample has the text, the reference (ground truth), all 7 systems' outputs, a word-timing table (human annotation, MFA on the reference, MFA on the output) and `timing.png`. Live: `make live-demo`, browser demo `make web-demo` ([DEMO.md](docs/DEMO.md)) |
| Output audio samples | [`samples/`](samples/README.md): byte-exact copies of the evaluated outputs. All 1,248 test utterances × 7 systems are on the Hugging Face Hub (`make outputs`) |
| Q1: data strategy, architecture, training, losses | [REPORT §4](docs/REPORT.md#4-data-and-the-accent-disjoint-split) data, [§7](docs/REPORT.md#7-models) models, training and losses (§7.2), [§10](docs/REPORT.md#10-the-projects-journey-experiments-and-the-reasoning-behind-each) experiments |
| Q2: motivation and detailed architecture | [REPORT §3](docs/REPORT.md#3-design-principle-one-source-per-output-property), [§7](docs/REPORT.md#7-models), [§8](docs/REPORT.md#8-inference-pipeline-transferring-word-timing); [How it works](#how-it-works) |
| Q3: params, MACs, latency, RTF, plots | [REPORT §12](docs/REPORT.md#12-efficiency), [`results/profile_*.json`](results), [`results/plots/`](results/plots) (09_efficiency and 10 more analysis plots), [§11](docs/REPORT.md#11-final-results-on-held-out-accents) test results |
| Q4: code walkthrough, inference on the interviewers' files | [Repository layout](#repository-layout); `make infer REF=… TEXT=…`, `make live-demo`, `make web-demo` |
| Slides | [`slides/meeami-tts_v2.pptx`](slides/meeami-tts_v2.pptx) |
| Evidence for every claim and decision | [REPORT §16](docs/REPORT.md#16-claim--evidence-index) (claim → result file → code → samples), [`docs/DECISIONS.md`](docs/DECISIONS.md) (every decision and error) |

## Quick start: inference and demo (fresh clone, ~15 min, no dataset needed)

```bash
git clone https://github.com/rrtchaganti/meeami-tts.git && cd meeami-tts
make env            # conda envs `tts` + `mfa` from envs/*.yml (~3-5 min)
                    # names taken already? use others:  export TTS_ENV=tts2 MFA_ENV=mfa2  (then make env)
make models         # checkpoints, baseline weights and the frozen lexicon from the public HF Hub repo, sha256-checked (~1.3 GB, no login)
make assets         # pinned MFA english_us_arpa models + NLTK data (~1-2 min)
make test           # unit tests: ~300 pass; 9 skip because they need the training data
```

The datasets are needed only to retrain or re-evaluate (see "Setup"). The demo's reference clips
ship in [`demo/clips/`](demo/clips/SOURCE.md) and [`tests/edge_cases/`](tests/edge_cases/SOURCE.md).

Then run it:

```bash
# first check: an accented clip that ships with the repo (L2-ARCTIC dev speaker, tests/edge_cases/)
make infer REF=tests/edge_cases/norm_date.wav TEXT="At sea Wednesday March 18th, 1908."
#   -> "[ok] norm_date: ... (error 0), aligner mfa ..." and out/infer/{out.wav,timings.json,timing.png}

# any reference recording (any format ffmpeg reads) + exactly what it says
make infer REF=their.wav TEXT="what the recording says"
make infer REF=their.wav TEXT="..." SYS=v0         # the original VITS (MAS) baseline; SYS=m0: FastSpeech2

# the interview demo: main system vs V0 on a held-out-accent L2-ARCTIC sentence,
# with word times measured on the output (docs/DEMO.md)
make live-demo
make live-demo UTT=YBAA_arctic_a0003 TEXT="For the 20th time that evening, the two men shook hands."
make rehearse       # 12 end-to-end cases (typed text, refusals) -> out/live_demo/REHEARSAL.md
make text-stress    # 121 typed-text cases through the normalizer, no audio

# browser demo on this machine: record/upload + text -> all 7 systems side by side (docs/DEMO.md §6)
make web-demo       # prints http://localhost:7860/?key=...;  make web-tunnel  shares it over https
```

No GPU? Add `ARGS="--device cpu"` (same output, slower synthesis; MFA runs on the CPU either way).

| What | Where |
| --- | --- |
| Code, configs, tests, docs, slides, result tables and plots, demo reference clips | this repository |
| Inference checkpoints (5 systems, 141 MB each), the two baselines' weights, the frozen lexicon | Hugging Face `Raviteja546/meeami-tts` (`make models`, pinned by `hosting.revision` in [`configs/base.yaml`](configs/base.yaml)) |
| Exact evaluated outputs with references and word timings (21 test samples, listening sets, browser tests) | this repository, [`samples/`](samples/README.md) |
| Every evaluated test output (1,248 utterances × 7 systems, with measured TextGrids and metric files) | Hugging Face, `outputs/test/` (`make outputs`, ~1.4 GB, pinned by `hosting.outputs.revision`) |
| Datasets, full training checkpoints (with discriminators, 956 MB each), dev-set outputs | the author's machine only (git-ignored); hashes in [`results/checkpoints.sha256`](results/checkpoints.sha256) |

| Document | What it gives you |
| --- | --- |
| **[docs/REPORT.md](docs/REPORT.md)** | **The final report:** problem, approach, every decision with its reasoning and code links, all results |
| [docs/PLAN.md](docs/PLAN.md) | The original specification (code comments cite its sections) |
| [docs/DECISIONS.md](docs/DECISIONS.md) | The running log of every decision (D-n) and error (E-n), in order |
| [docs/HANDOFF.md](docs/HANDOFF.md) | Status log and the GPU-instance runbook |
| [slides/meeami-tts_v2.pptx](slides/meeami-tts_v2.pptx) | 23-slide presentation with speaker notes (current draft, D-91; v1: [meeami-tts.pptx](slides/meeami-tts.pptx)) |
| `make demo` → `out/demo_site/index.html` | Demo page: reference vs every system's output and timing plot |
| **[docs/DEMO.md](docs/DEMO.md)** | **Live demo runbook:** `make models`, `make live-demo`, `make rehearse`, what to say, what to do when something fails |

---

## What comes from where

The design rule: every output property has exactly **one** source. One input decides it, and no
other input may influence it, so nothing from the reference can leak into the voice or
pronunciation. It also means each failure has one place to look.

| Output property | Its only source | Deliberately **not** taken from | Code |
| --- | --- | --- | --- |
| Words | Input text → normalizer | the reference audio, an ASR transcript | [`tts/text/normalize.py`](tts/text/normalize.py) |
| Pronunciation | `data/lexicon/lexicon.txt`: first CMUdict pronunciation, g2p_en for OOV words | the reference's pronunciation | [`tts/text/lexicon.py`](tts/text/lexicon.py) |
| Voice, accent, intonation | The model, trained on LJSpeech only | the reference (no accented audio in training) | [`tts/models/vits_mfa/models.py`](tts/models/vits_mfa/models.py) |
| Word start/end times, pauses, total length | Reference audio aligned with MFA `english_us_arpa` (MMS_FA CTC fallback) | the model's own durations | [`tts/infer/pipeline.py`](tts/infer/pipeline.py) |
| Phone timing inside each word | The model's native LJ durations, fitted to the word's span | the reference's phone timing (it carries the accent) | [`tts/infer/pipeline.py`](tts/infer/pipeline.py#L112) |

## Results on held-out accents

The test accents are Arabic, Korean and Vietnamese (12 speakers, 1,248 references). They were
never used for tuning and were only readable after the git tag `freeze-v1`.

| System | What it is | Word-boundary MAE | Pause F1 | WER | US prob | UTMOS | Spk sim LJ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L2 references | the input recordings | | | 6.3% | 0.486 | | |
| **V0** | **immediate baseline:** the original pretrained VITS (MAS, stochastic durations), same word timing injected | 40.5 ms | 0.815 | **2.1%** | 0.649 | **4.31** | 0.802 |
| M0 | second baseline: FastSpeech2 + HiFi-GAN, reference durations injected | 31.9 ms | 0.726 | 9.3% | 0.746 | 3.96 | 0.766 |
| M1 | VITS fine-tuned with MFA durations instead of MAS | 24.2 ms | 0.822 | 3.8% | 0.734 | 4.19 | 0.805 |
| m1_pos | M1 + position-in-phone embedding | 24.6 ms | 0.826 | 4.4% | 0.781 | 4.17 | 0.806 |
| **m1_pos_diph** | **main system:** m1_pos with each diphthong split into two tokens | **23.9 ms** | **0.835** | **2.9%** | 0.762 | **4.20** | 0.800 |
| abl_a_pos | m1_pos + 40% pace-augmented clips (pace from LJ statistics) | 24.7 ms | 0.825 | 3.9% | 0.665 | 4.06 | 0.801 |
| abl_b_pos | m1_pos + 40% pace-augmented clips (pace from dev-accent statistics) | 24.7 ms | 0.826 | 4.5% | 0.767 | 4.08 | 0.796 |

- **The main comparison: V0 vs the main system.** V0 is the model before MAS was replaced with MFA
  durations, given the same reference timing. The main system's word boundaries are 16.5 ms closer
  (p ≈ 1e-192, every accent), and its US probability is 0.11 higher. The cost is WER +0.8 points and
  UTMOS −0.12. V0 renders the durations it is given ~20 ms early and scattered (Stage 15).
- **Length:** output length equals reference length for every system and every reference (length error 0).
- **The references' 6.3% WER** is Whisper on the L2 input recordings themselves (human speech, no
  model). It shows how hard accented speech is for ASR; it is a reference point, not a drawback.
  Output WERs are Whisper on the synthesized speech of the same sentences.
- **WER is computed on normalized text, on both sides:** Whisper's `EnglishTextNormalizer`
  lower-cases, strips punctuation, expands contractions and maps "twentieth" / "20th" to one form.
  No evaluation sentence contains digits or symbols.
- **Ceilings:** LJ ground truth scores WER 1.9% and UTMOS 4.36.
- **Timing floor:** MFA's own word-boundary error on accented speech is 19.4 ms.
- **Ablation B:** used pace statistics from dev-accent speakers; all reported numbers are on held-out accents.
- **Where the numbers live:**
  - 95% bootstrap CIs: `results/<system>/test/summary.json`.
  - Paired Holm-adjusted tests: [`results/paired/`](results/paired).
  - Reasoning: [REPORT §10–11](docs/REPORT.md#10-the-projects-journey-experiments-and-the-reasoning-behind-each).

---

## How it works

```
reference.wav ─► MFA forced alignment ─► word start/end times ─┐
text ─► normalizer ─► lexicon phones ─► token timeline (sil · phones · sp · sil)
                                                               │
     duration predictor ─► native phone durations ─► fit each word to its reference span:
        dᵢ = exp(log d̂ᵢ + z·σᵢ), one z per word ("elastic", LJ's own per-phone variability σᵢ)
        cap at 2.5× native; the excess becomes a pause
                                                               │
     cumulative rounding to exactly the reference frame count ─► VITS ─► out.wav
```

- **Word boundaries come from the reference; time inside each word is spread with LJSpeech's
  elasticity.** Copying the reference's phone durations instead was measured to hurt
  intelligibility (WER 10.6% vs 2.5% on dev): phone timing carries the accent.
- **The model** is VITS with
  [six changes](docs/REPORT.md#72-m1-vits-with-mfa-durations-vits-mfa):
  - MFA durations replace MAS;
  - no blank tokens;
  - 72 ARPAbet symbols;
  - a deterministic duration predictor;
  - an optional position-in-phone embedding;
  - (main system only) diphthongs split into two tokens.

| | Original VITS (V0) | VITS-MFA (ours) |
| --- | --- | --- |
| Alignment in training | MAS, searched every step | Fixed: one-hot expansion of MFA frame counts |
| Input tokens | 178 espeak IPA characters + punctuation, blank between every token | 72 ARPAbet tokens, no blanks (87 with the diphthong split) |
| Duration model | Stochastic flow (1.32 M params): a different rhythm per sample | Deterministic, 2 convs (0.35 M), MSE on log frames: one native value per phone |
| Weights | — | Warm start: every pretrained tensor whose name and shape match (553) |

- **Why fixed alignments have a cost** ([REPORT §7.4](docs/REPORT.md#74-theory-what-a-fixed-external-alignment-takes-away)):
  - **Not lost:** gradients. MAS is not differentiable, so the KL loss still trains the flow and
    the prior encoder by the same paths.
  - **Lost:** MAS re-chose the alignment every step in the flow's output space, using the
    model's own prior. MFA's alignment never sees them.
  - **Result:** with no blanks, each phone gets one constant prior. Every MFA-trained model moves
    14–16% less inside phones than LJ, and "cry" flattened to "cra".
  - **What it buys:** durations the model actually follows (V0: −21 ms bias, 27.7 ms scatter;
    main: ~0 bias, 11 ms).

---

## Setup

**Requirements:**
- Linux or WSL2, with conda (Miniconda or Miniforge), `git` and `make`.
- An NVIDIA GPU with a driver for CUDA 12.9 is recommended. Every inference command also runs with
  `--device cpu`.
- `ffmpeg` comes with the `tts` env. `sox` and `rubberband-cli` ≥ 3.0 are needed only to rebuild
  the augmented training data.

**Datasets** are not downloaded by the code, and inference does not need them: `make models` also
brings the frozen lexicon, and the demo clips are in the repo. To rebuild or retrain (`make data`,
`make lexicon`, ...):
- LJSpeech 1.1 → `data/raw/LJSpeech-1.1/` (`make lj-download` fetches the official archive);
- L2-ARCTIC v5 (access form) → `data/raw/l2arctic/`.

```bash
make env      # two pinned conda envs: tts (everything) and mfa (Montreal Forced Aligner 3.4.3 only);
              # re-running it updates existing envs in place (--prune). Other names:
              # TTS_ENV=... MFA_ENV=... on every make call, or exported once
make models   # inference checkpoints + baseline weights + the frozen lexicon (Hugging Face, sha256-checked;
              # the lexicon only if missing, since inference appends new words to it)
make assets   # pinned MFA models + NLTK corpora (sha256-checked)
make lexicon  # rebuild data/lexicon/lexicon.txt from the corpus texts (needs the datasets; D-98)
make test     # unit tests: 308 pass with the full data (author's machine); 300 pass and 9 skip
              # on a clean clone without datasets (D-100, rechecked in D-104)
make help     # every target
```

- **Exact environments:** `envs/lock/*.full.yml` hold full transitive exports of the tested envs.
- **MFA** is always called as `conda run -n mfa mfa ...` with `MFA_ROOT_DIR=data/models/mfa`.
- **Checkpoints:** `make models` writes `export/<system>/infer.pt`, holding the generator weights
  and config only. That is everything inference reads, and it is bit-identical in output to the
  full training checkpoint `export/<system>/ckpt_last.pt` (D-95), which exists only on the
  author's machine. Every command takes either file, and the Makefile uses `ckpt_last.pt` when
  present, otherwise `infer.pt`. Training-checkpoint hashes:
  [`results/checkpoints.sha256`](results/checkpoints.sha256).
- **Configuration:** every parameter lives in [`configs/`](configs). Each config inherits
  `base.yaml` and overrides only what differs.

## Inference

```bash
make infer REF=ref.wav TEXT="..."                       # main system -> out/infer/
make infer REF=ref.wav TEXT="..." SYS=v0 OUT=out/v0     # SYS: m1_pos_diph (default), v0, m0, m1, m1_pos, abl_a_pos, abl_b_pos
# the same through the script, plus a folder of x.wav + x.txt pairs:
python scripts/infer.py --text "..." --ref ref.wav --out out/ --ckpt export/m1_pos_diph/infer.pt
python scripts/infer.py --pairs_dir their_files/ --out out/ --ckpt export/m1_pos_diph/infer.pt
python scripts/infer.py --text "..." --ref ref.wav --out out/ --ckpt v0     # or m0
```

(`python scripts/...` needs `conda activate tts` first; the `make` targets activate it themselves.)

**Options** (`python scripts/infer.py`, or `make infer ... ARGS="..."`):
- `--aligner mfa|ctc`
- `--transfer elastic|proportional|phone_copy`
- `--cap <float|none>`
- `--noise_scale`
- `--device cuda|cpu`
- `--config` (default `configs/inference.yaml`; the tuned values are elastic, cap 2.5, noise
  0.333, MFA)

**The text must be what the recording says.** The normalizer changes only the form (`20th`,
`$5`, `Dr.`, `12/03`, `10km`, `GPU` are read as a US speaker would; D-94). Words that were not
spoken are not detected: MFA forces them into the audio (docs/DEMO.md §3).

**Outputs for each pair:**
- `out.wav`: exactly round(N_ref·22050/sr_ref) samples.
- `timings.json`: normalized text, reference and output word times, aligner used, per-stage
  latency, RTF.
- `timing.png`: reference and output words over both spectrograms.

A failing pair prints a one-line reason and the run continues.

---

## The project, stage by stage

Each stage below lists its method, the code that implements it, the make target that runs it, and
how it was checked. The stages are in the order they were built. Full reasoning is in
[docs/REPORT.md](docs/REPORT.md); the decision IDs point into
[docs/DECISIONS.md](docs/DECISIONS.md).

### Stage 1: Environments (T1)

- **Method:**
  - Two pinned conda envs: `tts` and `mfa`.
  - conda-forge only.
  - PyTorch/torchaudio 2.8.0 + CUDA 12.9. torchaudio 2.9 removed APIs SpeechBrain and MMS_FA
    need (D-4).
- **Code:** [`envs/`](envs), [`tests/test_env.py`](tests/test_env.py).
- **Command:** `make env && make test`.
- **Checks:**
  - Every package imports and CUDA is visible.
  - The MFA command form is as PLAN §3 expects, with no deprecation notice.

### Stage 2: Data and the accent-disjoint split (T2)

- **Method:**
  - **LJ filelists:** the original VITS filelists (12,500 / 100 / 500), fetched by hash.
  - **L2-ARCTIC split by accent:** 24 speakers, 4 per first language (2 male, 2 female). Whole
    accents go to one side, so every test accent is unseen:
    - dev = Hindi, Mandarin, Spanish (all tuning and decisions);
    - test = Arabic, Korean, Vietnamese (read once, after the freeze).
  - **Evaluation set:** the IDs that are manually annotated for every speaker of the split.
    That gives dev 100 × 12 and test 104 × 12.
  - **Test lock:** test-speaker audio can't be read until `freeze-v1` exists.
- **Code:** [`tts/splits.py`](tts/splits.py), [`tts/guard.py`](tts/guard.py), [`configs/splits.yaml`](configs/splits.yaml).
- **Command:** `make data`.
- **Check:** [`reports/data_splits.md`](reports/data_splits.md).

### Stage 3: Text front end (T3)

- **Method:**
  - **One normalizer** for training text, MFA labels and inference input: numbers, ordinals,
    dates, currency, abbreviations, symbols.
  - **Lexicon:** one pronunciation per word. CMUdict first, then the possessive rule (D-26),
    then g2p_en.
  - **Symbols:** 72 (pad, sil, sp + 69 ARPAbet), equal to MFA's phone set.
- **Code:** [`tts/text/normalize.py`](tts/text/normalize.py), [`tts/text/lexicon.py`](tts/text/lexicon.py), [`tts/text/symbols.py`](tts/text/symbols.py).
- **Command:** `make lexicon`.
- **Check:** 14,912 words, every phone in MFA's set ([`reports/lexicon.md`](reports/lexicon.md)).

### Stage 4: Forced alignment and phone statistics (T4, T5)

- **Method:** MFA `english_us_arpa` with a wide beam (100/400), plus the TextGrid parsing rules:
  - silence ≥ 3 frames → `sp`;
  - cumulative frame rounding;
  - ≥ 1 frame per phone.

  The phone statistics are each phone's mean and σ of log duration over LJ train.
- **Code:** [`tts/align/mfa.py`](tts/align/mfa.py), [`tts/align/textgrid.py`](tts/align/textgrid.py), [`tts/align/lj.py`](tts/align/lj.py), [`tts/align/l2.py`](tts/align/l2.py), [`tts/stats.py`](tts/stats.py).
- **Commands:** `make align-lj`, `make stats`, `make align-l2-dev`, `make align-dev-eval`.
- **Checks:**
  - 13,073 / 13,100 LJ files kept ([`reports/align_lj.md`](reports/align_lj.md)).
  - **MFA vs manual annotation on accented dev speech:** word MAE 19.4 ms, phone MAE 13.1 ms
    ([`reports/mfa_l2_dev.md`](reports/mfa_l2_dev.md), plot 3). The ground truth is L2-ARCTIC's
    hand-corrected word and phone tiers (~150 utterances per speaker).

### Stage 5: M0, the FastSpeech2 baseline (T6)

M0 was the planned baseline. The immediate baseline for the MAS → MFA swap, the original VITS
(V0), was added after the freeze (Stage 15).

- **Method:** pretrained FastSpeech2 + HiFi-GAN. The reference durations are passed as
  `d_targets` to its variance adaptor.
- **Code:** [`tts/models/fastspeech2/m0.py`](tts/models/fastspeech2/m0.py).
- **Command:** `make m0-demo`.

### Stage 6: VITS-MFA model and trainer (T7, T8)

- **Method:** VITS with MAS removed. The alignment is a one-hot expansion of the MFA durations.
  - **Training:**
    - no blanks;
    - a re-initialized 72-symbol embedding;
    - a deterministic log-duration predictor on the detached encoder output;
    - loss 45·mel + KL + dur + adv + 2·fm;
    - bf16 with fp32 losses.
  - **Why deterministic durations:** the reference fixes each word's total, so the model only
    has to say how LJ would share it among the phones, which is one value per phone, not a sample.
    The log scale makes errors relative, and the minimizer is the geometric-mean duration.
  - **Options:**
    - `use_pos` (a position-in-phone embedding): frame k of a d-frame phone gets bin
      ⌊16·(k + 0.5)/d⌋, and a learned vector per bin is added before the prior projection.
      Zero-initialized, so it starts as M1;
    - one-GPU branching (`--init_from` one of our own checkpoints).
- **Code:** [`tts/models/vits_mfa/models.py`](tts/models/vits_mfa/models.py), [`trainer.py`](tts/models/vits_mfa/trainer.py), [`data.py`](tts/models/vits_mfa/data.py), [`scripts/train.py`](scripts/train.py).
- **Commands:** `make overfit`, `make smoke`.
- **Checks:**
  - **Overfit:** one utterance for 500 steps came out intelligible (WER 5.6%).
  - **Smoke:** every config ran 200 steps at batch 4, and resume worked.

### Stage 7: Pace augmentation for the ablations (T9)

- **Method:**
  - **Set A:** pace sampled from LJ statistics (target-agnostic).
  - **Set B:** word pace and pauses sampled from dev-accent statistics (target-aware, word level
    only).
  - **Phone factors:** exp(z·σᵢ), clipped to [0.6, 2.5].
  - **Pauses:** built from looped room tone.
  - **Stretcher:** time-varying WSOLA (it replaced Rubber Band R3, which cost about 1.4 UTMOS;
    D-67).
  - **Filter:** the 5% of clips with the largest UTMOS drop are removed.
- **Code:** [`tts/augment/`](tts/augment): `generate.py`, `pace_b.py`, `core.py`, `wsola.py`, `verify.py`, `filter.py`.
- **Commands:** `make pace-stats-b`, `make aug-a aug-b`, `make aug-filter`.
- **Check:** MFA re-alignment matches the planned boundaries to a median of about 7 ms
  ([`reports/aug_a.md`](reports/aug_a.md), [`aug_b.md`](reports/aug_b.md)).

### Stage 8: Inference pipeline and edge cases (T10)

- **Method:** the ten-step pipeline under [How it works](#how-it-works). Elastic word fitting by
  bisection, a stretch cap, cumulative rounding to the exact frame count, and pad/trim to the exact
  sample count.
  - **Fallback:** the MMS_FA CTC aligner, a numpy Viterbi (D-56).
  - **Fit variants** for a word of reference length D:
    - `elastic` (used): dᵢ = exp(log d̂ᵢ + z·σᵢ), one z per word;
    - `proportional`: dᵢ = d̂ᵢ·D/Σd̂;
    - `phone_copy`: the reference's own phone durations.

    Example, "cry" at twice LJ's length: proportional gives K/R/AY ×2.0 each; elastic gives
    ×1.75 / ×2.4 / ×2.0. The stop stretches least.
  - **Why the 2.5× cap:** L2 readers hesitate, and stretching phones 3–4× gives drawled vowels the
    model never saw. With the cap, the excess becomes a pause and the next word still starts on
    time.
    - **Dev cap sweep, MAE:** 26.1 (1.4×), 22.9 (2.0×), 21.9 (2.5×), 21.5 ms (none).
    - **UTMOS:** best at 2.5×.
    - **On test:** it acts on 10% of references.
- **Code:** [`tts/infer/pipeline.py`](tts/infer/pipeline.py), [`systems.py`](tts/infer/systems.py), [`ctc.py`](tts/infer/ctc.py), [`edge_cases.py`](tts/infer/edge_cases.py), [`scripts/infer.py`](scripts/infer.py).
- **Command:** `make edge` (`CKPT=... DEVICE=cuda|cpu`).
- **Check:** 21 edge cases covering text normalization, audio formats, extreme timing and
  alignment failures. **21/21 pass on GPU and CPU.**

### Stage 9: Evaluation harness and metrics (T11)

`scripts/evaluate.py` synthesizes every reference of a split and aligns the outputs with MFA. It
then scores every output and writes:
- `per_utt.csv`;
- `summary.json` (95% bootstrap CIs, per accent);
- paired Wilcoxon tests against M1;
- the plots.

| Metric | Definition | Code |
| --- | --- | --- |
| Length error | Output samples − round(N_ref·22050/sr_ref), must be 0 | [`pipeline.py`](tts/infer/pipeline.py#L333-L334) |
| Word-boundary MAE, % within 20 / 50 ms | Every word start and end. Output words come from MFA on the output; reference words come from the **manual** annotation. | [`metrics.py`](tts/eval/metrics.py#L16-L30) |
| Word-boundary MAE (CTC) | The same, with MMS_FA on the output (second-aligner cross-check; CTC has a +38 ms start bias on real speech, so it is not a ranking, Stage 15) | [`evaluate.py`](scripts/evaluate.py#L188-L193) |
| Pause F1 | Pauses (silence ≥ 100 ms) at word boundaries, output vs reference | [`metrics.py`](tts/eval/metrics.py#L38) |
| WER, CER | faster-whisper large-v3 + Whisper text normalizer + jiwer | [`asr.py`](tts/eval/asr.py#L45) |
| US probability | CommonAccent ECAPA accent ID, softmax(30·cosine), US class | [`scorers.py`](tts/eval/scorers.py#L55-L70) |
| Speaker similarity | ECAPA cosine to the mean of 100 LJ test clips, and to the reference speaker | [`scorers.py`](tts/eval/scorers.py#L39), [`evaluate.py`](scripts/evaluate.py#L149-L168) |
| UTMOS | SpeechMOS `utmos22_strong`, sha256-pinned weights | [`scorers.py`](tts/eval/scorers.py#L14), [`tts/eval/assets.py`](tts/eval/assets.py) |
| CIs, paired test | Bootstrap (1,000 resamples), Wilcoxon signed-rank | [`metrics.py`](tts/eval/metrics.py#L61-L83) |

- **Commands:**
  - `make assets-eval` (pinned evaluation models);
  - `make eval SYSTEM=m0 SPLIT=dev`.
- **Plots** (in [`results/plots/`](results/plots), from [`tts/eval/plots.py`](tts/eval/plots.py)):
  - 02 boundary scatter;
  - 03 MFA error;
  - 04 metrics vs stretch;
  - 05/07 tuning and transfer variants;
  - 06 per accent;
  - 08 dev vs test pace;
  - 09 efficiency;
  - 10 training curves.

### Stage 10: Profiling, GPU bundle and demo (T12–T14)

- **Profiling** ([`scripts/profile.py`](scripts/profile.py), `make profile`):
  - parameters per module;
  - MACs, using ptflops' `aten` backend plus one layer worked by hand;
  - p50/p90 latency per stage;
  - RTF.
- **Bundle** (`make bundle` / `bundle-m1`, then `make check-data` on a fresh unpack): every
  training file, sha256-verified.
- **Demo page** ([`scripts/make_demo.py`](scripts/make_demo.py), `make demo`): the per-sample
  outputs double as backup outputs for the live demo.

### Stage 11: Phase 1, training M1 on one L40S (D-70)

- **Method:**
  - A 300-step benchmark (0.229 s/step) set `max_steps` to 114,000 for an 8 h budget, at batch 32.
  - Training ran in tmux, with resume after an interruption.
  - Runbook: [docs/HANDOFF.md](docs/HANDOFF.md).
- **Commands:**
  - `python scripts/train.py --config configs/m1.yaml`;
  - `bash scripts/export_run.sh m1`.

### Stage 12: Tuning and the M1 dev results (D-72, D-73)

- **Method:** a grid over transfer variant × cap × noise scale × aligner, on 120 dev references.
  The rule was fixed in advance: highest UTMOS among settings within 5 ms of the best MAE
  ([`tts/eval/tune.py`](tts/eval/tune.py)).
- **Chosen:** elastic, cap 2.5, noise 0.333, MFA.
- **Command:** `make tune SPLIT=dev`, which writes the chosen values into
  `configs/inference.yaml`.
- **Result (M1 vs M0 on dev):**
  - MAE 31.3 → 23.7 ms;
  - WER 10.0 → 3.5%;
  - UTMOS 3.94 → 4.20;
  - US probability level.

### Stage 13: Phase 2, three experiments driven by listening

Every phase-2 system branches from M1 for 20,000 steps with a fresh optimizer, so each pair differs
in one setting.

| Phase | Observation | Probe added | Experiment | Outcome |
| --- | --- | --- | --- | --- |
| 2a (D-74 to D-78) | "Words end sharply": within-phone spectral movement 14% below LJ | **Phone dynamics** ([`tts/eval/phone_dynamics.py`](tts/eval/phone_dynamics.py), `make phone-dyn`) | `m1_pos` vs `m1_cont` | US prob 0.769 vs 0.680 (p = 5e-31); m1_pos becomes the base |
| 2b (D-79, D-82, D-83) | Do pace-augmented clips help at high stretch? | **Paired test against each system's own control, by accent and stretch tercile, Holm-adjusted** ([`tts/eval/paired.py`](tts/eval/paired.py), `make paired`) | `abl_a_pos`, `abl_b_pos` vs `m1_pos` | No gain at any stretch; UTMOS −0.08 to −0.11: a **negative result** |
| 2c (D-80, D-81, D-84) | "cry" heard as "cra": diphthongs flattened (LPC formant probe: AY/AW glides missing) | Diphthong glide probe (formants, D-80) | `m1_pos_diph` (each diphthong → two tokens, [`tts/text/diph.py`](tts/text/diph.py)) vs `m1_pos` | Glide restored; MAE −0.7 ms, WER 3.9 → 2.7%, UTMOS +0.02 → **main system** |

Full story and numbers:
[REPORT §10](docs/REPORT.md#10-the-projects-journey-experiments-and-the-reasoning-behind-each).

### Stage 14: Freeze and the test evaluation (D-85, D-86)

- **Freeze:** the final system set was fixed, the checkpoint hashes were committed, and
  `git tag freeze-v1` was placed on `5889ecf`.
- **Only then** were the test accents evaluated, once, for all six systems. Then came the paired
  tests, plot 8 (dev vs test pace: ×1.12 vs ×1.16 LJ), the profiles, the clean demo build and the
  slides.

### Stage 15: The immediate baseline V0, added after the freeze (D-90)

- **Why:** M1 replaces MAS with MFA durations, so the system to beat is the original VITS with
  MAS, not only FastSpeech2. V0 is the official pretrained LJSpeech VITS (the checkpoint M1 was
  fine-tuned from), unchanged and inference only. No frozen system or config was touched.
- **Method** ([`tts/models/vits_mas/v0.py`](tts/models/vits_mas/v0.py),
  [`configs/v0.yaml`](configs/v0.yaml)):
  - lexicon phones are written in espeak-style IPA, with blanks interspersed;
  - the stochastic duration predictor gives native durations per character, summed per phone;
  - each phone's frames are split back over its characters at synthesis;
  - the same pipeline and settings as every system.
  - One adapter choice, which character a blank belongs to, was made on dev (50.8 vs 40.7 ms).
- **New probe: duration fidelity and aligner calibration**
  ([`tts/eval/fidelity.py`](tts/eval/fidelity.py), `make fidelity`):
  - Fidelity is how far each output word lands from where the pipeline placed it (bias, MAE, and
    MAE without the bias).
  - Calibration measures each aligner's own bias on real speech.
  - Test: V0 −21 / −15 ms (starts / ends), 27.7 ms scatter; main system −1 / 0 ms, 11.2 ms.
  - CTC's own start bias on real speech is +38 ms; MFA has none.
- **Commands:** `make v0-demo`; `make eval SYSTEM=v0 SPLIT=dev|test`;
  `make paired CONTROL=v0 SYSTEMS_CMP="m1 m1_pos_diph m0" SPLIT=test`;
  `make fidelity SPLIT=dev` (also calibrates) and `SPLIT=test`.
- **Result:** see the results table and
  [REPORT §10.6](docs/REPORT.md#106-after-the-freeze-the-immediate-baseline-v0-d-90).

---

## Commands in run order

```bash
# --- laptop: build and verify (T1-T14) -------------------------------------------------------
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
make aug-a aug-b     # T9: augmented sets (time-varying WSOLA, D-67) + MFA verification
make aug-filter      # T9: UTMOS filter -> aug_a.jsonl / aug_b.jsonl
make edge            # T10: 21-case edge suite (CKPT=m0|path, DEVICE=cuda|cpu)
make assets-eval     # T11: pinned evaluation models (ASR, ECAPA, accent ID, UTMOS)
make eval SYSTEM=m0 SPLIT=dev
make profile DEVICE=cuda      # T12
make bundle-m1                # T13: small M1 bundle (+ check-data on a fresh unpack); `make bundle` = full
git tag smoke-ok; git tag gpu-ready

# --- GPU instance, phase 1: M1 (docs/HANDOFF.md) ---------------------------------------------
git clone <repo> meeami-tts && cd meeami-tts && git checkout gpu-ready
bash scripts/setup_gpu.sh && bash scripts/fetch_bundle.sh <url-or-path> && make lj-download && make check-data
conda run -n tts python scripts/train.py --config configs/m1.yaml --benchmark 300   # -> train.max_steps
tmux new -d -s train 'conda run --no-capture-output -n tts python scripts/train.py --config configs/m1.yaml 2>&1 | tee logs/m1.log'
bash scripts/export_run.sh m1          # -> export/m1.tar.gz ; rsync to the laptop

# --- GPU instance, phase 2: branches from M1, 20k steps each, fresh optimizer ------------------
B=runs/m1/ckpt_last.pt
python scripts/train.py --config configs/m1.yaml        --init_from $B --max_steps 20000 --tag m1_cont   # 2a control
python scripts/train.py --config configs/m1_pos.yaml    --init_from $B --max_steps 20000                 # 2a
python scripts/train.py --config configs/abl_a_pos.yaml --init_from $B --max_steps 20000                 # 2b
python scripts/train.py --config configs/abl_b_pos.yaml --init_from $B --max_steps 20000                 # 2b
python scripts/train.py --config configs/m1_pos_diph.yaml --init_from $B --max_steps 20000               # 2c
for e in m1_cont m1_pos abl_a_pos abl_b_pos m1_pos_diph; do bash scripts/export_run.sh $e; done

# --- laptop: tune, dev evaluation, freeze, test ------------------------------------------------
for e in m1 m1_pos m1_pos_diph abl_a_pos abl_b_pos; do sha256sum -c export/$e.tar.gz.sha256 && tar -xzf export/$e.tar.gz; done
make tune SPLIT=dev                  # writes cap / noise_scale into configs/inference.yaml (D-72)
make eval-all SPLIT=dev              # SYSTEMS = m0 m1 m1_pos m1_pos_diph abl_a_pos abl_b_pos
make phone-dyn PD_CKPTS="export/m1/ckpt_last.pt export/m1_pos/ckpt_last.pt ..."   # D-76
make paired CONTROL=m1_pos SYSTEMS_CMP="abl_a_pos abl_b_pos m1_pos_diph"           # D-82
make edge CKPT=export/m1_pos_diph/ckpt_last.pt DEVICE=cuda   # and DEVICE=cpu: 21/21
git tag freeze-v1                    # code, configs and checkpoints frozen (results/checkpoints.sha256)
make eval-all SPLIT=test             # held-out accents; also draws plot 8 (dev vs test pace)
make paired CONTROL=m1_pos SYSTEMS_CMP="abl_a_pos abl_b_pos m1_pos_diph" SPLIT=test
make profile DEVICE=cuda ; make profile DEVICE=cpu ; make profile DEVICE=cpu ARGS="--threads 1"
make demo                            # out/demo_site/index.html (+ backup outputs per sample)
# --- after the freeze: immediate baseline V0 (D-90; new files only) ----------------------------
make eval SYSTEM=v0 SPLIT=dev ; make eval SYSTEM=v0 SPLIT=test
make paired CONTROL=v0 SYSTEMS_CMP="m1 m1_pos_diph m0" SPLIT=test
make fidelity SPLIT=dev ; make fidelity SPLIT=test   # duration fidelity + aligner calibration
# --- live demo (D-94..D-96; docs/DEMO.md) ------------------------------------------------------
make infer REF=their.wav TEXT="..."  # one reference -> out/infer/ (SYS=v0|m0|<system>)
make models                          # export/<system>/infer.pt + baseline weights from the HF Hub (sha256-checked)
make text-stress                     # 121 typed-text cases through the normalizer (TEXT="..." for one input)
make rehearse                        # 12 end-to-end cases -> out/live_demo/REHEARSAL.md
make live-demo                       # main vs V0 on one test-accent sentence, measured word times
make live-demo UTT=YBAA_arctic_a0003 TEXT="For the 20th time that evening, the two men shook hands."
make live-demo REF=their.mp3 TEXT="what the recording says"
```

On the GPU instance, prefix the training commands with `conda run -n tts`, inside tmux.
[HANDOFF.md](docs/HANDOFF.md) has the exact launch lines.

---

## Repository layout

```
configs/    base.yaml (all shared parameters) · splits.yaml · inference.yaml (tuned) · one yaml per system
tts/
  text/       normalize.py · input_rules.py (typed-input pre-pass, D-94) · stress.py · lexicon.py · symbols.py · diph.py
  align/      mfa.py (MFA wrapper) · textgrid.py (parsing rules) · lj.py · l2.py (MFA vs manual)
  models/     vits_mfa/ (VITS-MFA: models, trainer, data) · fastspeech2/ (M0, vendored) · vits_mas/ (V0 adapter)
  augment/    generate.py (A/B plans) · pace_b.py · core.py · wsola.py · verify.py · filter.py
  infer/      pipeline.py (Section 5 steps 1-10) · systems.py · ctc.py · edge_cases.py · batch.py
  eval/       metrics.py · scorers.py · asr.py · paired.py · phone_dynamics.py · fidelity.py · tune.py · plots.py
  guard.py    test-split lock          splits.py · stats.py · audio.py · config.py
scripts/    train.py · infer.py · evaluate.py · profile.py · make_demo.py · demo.py (live demo) · web_demo.py (browser demo) · export_infer.py · fetch_models.py (HF Hub) · export_samples.py · publish_outputs.py · GPU bundle/setup/export
tests/      unit tests (one per module) + edge_cases/ (21 committed WAV/TXT cases)
stats/      phone_logdur.json (LJ per-phone log-duration mean, σ) · l2_dev_pace.json (Ablation B)
reports/    data splits, lexicon, LJ alignment QC, MFA error on accented speech, augmentation
results/    <system>/<split>/{per_utt.csv, summary.json} · paired/ · phone_dynamics/ · fidelity/ · tune/ · plots/ · profile_*.json · _rejected/
docs/       REPORT.md · PLAN.md · DECISIONS.md · HANDOFF.md
samples/    exact evaluated outputs + references + word timings (make samples, E-36): demo/ · listening/ · browser_tests/
slides/     meeami-tts_v2.pptx + build_v2.js (current) · meeami-tts.pptx + build.js (v1)
data/ runs/ export/ out/   git-ignored (datasets, training runs, checkpoints, demo)
```

## Known failure modes

- **MFA latency:** `mfa align_one` takes 3–15 s per reference, mostly model loading. It
  dominates end-to-end latency (model RTF 0.012, end-to-end 0.90 on the laptop GPU).
- **CTC fallback:** if MFA fails, returns `spn` or misses a word, word boundaries come from the
  MMS_FA CTC aligner. It is coarser: about 53 ms word-boundary MAE, against about 21 ms for MFA on
  dev references.
- **Stretch cap:** extreme stretches are capped at 2.5× the native word duration (tuned on dev,
  D-72); the excess becomes a pause.
- **Timing carries accent:** speech forced onto an L2 reference's word timing sounds less
  American than it could. Dev US probability: L2 references 0.30, main-system outputs 0.74.
- **Fixed diphthong split:** the main system splits each diphthong 50/50 (D-81). Short
  unstressed diphthongs get only 1–2 frames per half.
- **Noisy accent ID:** the US probability is near-binary per utterance and flips on inaudible
  changes (D-84). Read it as a population average, not per sample.
- **Augmentation hurt:** pace augmentation (Ablations A/B) lowered UTMOS and did not improve
  timing (D-83, D-86).
- **`phone_copy` without MFA:** it needs MFA phone durations; with the CTC fallback it runs as
  `elastic`.
- **Undecodable audio:** reference audio is decoded with libsndfile, then ffmpeg. Anything
  neither can read fails cleanly.
- **Spoken letters and "I" (E-35, [REPORT §15](docs/REPORT.md)):**
  - The letter `a` (`a, b, c`) is read and aligned as the article "uh" (AH0). Its alignment
    collapses to about 30 ms.
  - `I` right after a vowel (`may I know you`) can collapse to 30 ms, because MFA cannot place
    an EY→AY boundary.
  - **Fixed, MFA dictionary cache:**
    - **Symptom:** words new to the lexicon (`abc`, names) fell back to CTC.
    - **Cause:** MFA keeps a converted copy of `lexicon.txt` in its temporary directory and
      did not notice words that G2P added later.
    - **Fix:** that directory is now keyed by the lexicon's hash, so it is rebuilt whenever
      the file changes. This is a bug fix, not memorisation: only "word → phones" is cached,
      never audio, text or results.
    - **Limits:**
      - A wrong G2P guess stays in the lexicon until edited.
      - The cache saves only about 1.2 s per request (4.8 s vs 6.0 s).
      - It assumes one alignment at a time (the demo has a single worker).
  - **Heuristic, vowel-less letter strings:**
    - **Rule:** a lowercase token of 3–5 letters with no a/e/i/o/u, not in CMUdict, is spelled
      out (`bcd` → `b c d`). It is a regex rule, not learned.
    - **Fails on:**
      - `ijk` (has a vowel, so G2P guesses a word);
      - `brrr`, `grr`, `pfft` (spelled out);
      - 2-letter strings such as `pq` (left alone, so that corpus transcripts stay unchanged).
- **Abbreviation periods:** title abbreviations need their period, as in LJSpeech (`Dr.`, not
  `Dr`). Without it, `Dr` is read as "drive" (D-74).

## Licences

- **Vendored code:** VITS (jaywalnut310, MIT) and FastSpeech2/HiFi-GAN (ming024, MIT) live under
  `tts/models/`, with their licence files.
- **Edge-case clips:** the L2-ARCTIC (CC BY-NC 4.0) clips in `tests/edge_cases/` come from dev
  speakers only.
- **Demo clips:** `demo/clips/` holds 8 unmodified L2-ARCTIC test-speaker utterances (CC BY-NC
  4.0, attribution in [`demo/clips/SOURCE.md`](demo/clips/SOURCE.md)). They were added after
  `freeze-v1`.
- **Checkpoints** (Hugging Face `Raviteja546/meeami-tts`): fine-tuned from the MIT-licensed
  pretrained VITS on LJSpeech (public domain). `abl_b_pos` follows pace statistics from
  L2-ARCTIC, so treat it as non-commercial. The baseline weights are unmodified mirrors with
  their MIT licences ([`docs/hf_card/LICENSES.md`](docs/hf_card/LICENSES.md)).
- **Samples and hosted test outputs** (`samples/`, Hugging Face `outputs/test/`): the references
  and the owner's browser recordings are as recorded. The synthesized outputs follow L2-ARCTIC
  texts and word timings (CC BY-NC 4.0), so use them for **non-commercial research only**.
