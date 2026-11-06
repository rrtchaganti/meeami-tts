# meeami-tts

Time-aligned US-English TTS: speaks input text in the LJSpeech voice on the word timing of a
non-US reference recording. The full specification is in [docs/PLAN.md](docs/PLAN.md); decisions
and errors are logged in [docs/DECISIONS.md](docs/DECISIONS.md).

## Setup (T1)

Requires Miniconda/conda on `PATH` (or `make CONDA=/path/to/conda ...`), an NVIDIA GPU with a
CUDA 12.9-capable driver, and system `rubberband-cli` >= 3.0, `ffmpeg`, `sox`.

```bash
make env    # creates/updates the `tts` and `mfa` conda envs from envs/*.yml (pinned)
make test   # environment checks: CUDA, bf16, ctranslate2 GPU, Rubber Band, MFA CLI form
```

- `tts` env: everything except alignment. `mfa` env: Montreal Forced Aligner 3.4.3 only, always
  called as `conda run -n mfa mfa ...`.
- `envs/lock/*.full.yml` are full transitive exports of the tested envs (`make env-export`).

## Commands in run order

Filled in as tasks land (docs/PLAN.md Section 10). Targets not listed here still print
"not implemented yet".

```bash
make data   # T2: checks LJSpeech + L2-ARCTIC, fetches pinned VITS filelists, writes
            #     data/manifests/*.jsonl, data/splits/*_eval_ids.txt, reports/data_splits.md
make assets # T3: pinned MFA english_us_arpa models + NLTK data (sha256-checked), under data/models/
make lexicon  # T3: data/lexicon/lexicon.txt (cmudict first pron., g2p_en OOVs), reports/lexicon.md
```

Datasets are not downloaded by `make data`: LJSpeech 1.1 must be at `data/raw/LJSpeech-1.1/`
and L2-ARCTIC v5 (manual access form) at `data/raw/l2arctic/`. Test-accent audio is locked
until the `freeze-v1` git tag exists (`tts/guard.py`).
