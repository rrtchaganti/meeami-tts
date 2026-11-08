# Decisions and error log

Running record of decisions (D-n) and errors/incidents (E-n), newest task last. Each entry names the
task, the PLAN.md section it relates to, and the outcome. Entries are never edited after the fact
except to add a "Resolved:" line.

## T1 — environments, repo layout, Makefile skeleton (Sections 3, 8, 10)

### Decisions

- **D-1 CLAUDE.md check.** Repo-root CLAUDE.md is byte-identical to the Section 10 block
  (checked with `diff`). No changes.
- **D-2 Branch.** Repo had no commits on `master`; renamed to `main` (`git branch -m main`) at the user's request.
- **D-3 Channels.** Both env files use `conda-forge` + `nodefaults` only. Packages come from
  conda-forge when available; pip only for packages absent there (checked with
  `conda search -c conda-forge --override-channels`): `g2p_en`, `pyrubberband`, `faster-whisper`
  (pulls `ctranslate2`), `speechbrain`.
- **D-4 PyTorch / torchaudio 2.8.0, CUDA 12.9 build.** conda-forge offers pytorch up to 2.14 and
  torchaudio up to 2.11, but torchaudio 2.9 removed APIs the plan depends on: `forced_align`
  (backing `MMS_FA`, Section 3 fallback + Section 8 second aligner) and `torchaudio.load`
  (used by SpeechBrain). 2.8.0 is the newest release that still has them (deprecated, with warnings).
  `cuda-version=12.9` matches the available `cuda129` builds; ctranslate2 ≥ 4.5 needs CUDA 12 + cuDNN 9.
- **D-5 SpeechBrain 1.0.x.** Provides `speechbrain.inference` used to load the ECAPA speaker
  model and the CommonAccent accent-ID model (Section 8).
- **D-6 Extra packages beyond the task list:** numpy, scipy (Wilcoxon, bootstrap), librosa (mel
  filters as in VITS), soundfile, ffmpeg (portable to the GPU box), inflect, unidecode (VITS /
  FastSpeech2 text code, normalizer), nltk (g2p_en), pyyaml, tqdm, huggingface_hub, pandas
  (per-utterance CSVs), matplotlib (plots). Excluded: Cython (MAS is removed, Section 4),
  fvcore (ptflops covers MACs, Section 8).
- **D-7 MFA version.** `montreal-forced-aligner>=3,<4`, newest 3.x on conda-forge, pinned exactly
  after install (user approved). Section 3 notes 3.4 deprecates the current `align`/`align_one`
  signatures; `tests/test_env.py::test_mfa_command_form` checks the Section 3 flags and positional
  form and fails on any deprecation notice in `--help`.
- **D-8 No downloads in T1.** `g2p_en` calls `nltk.download()` on import when its corpora are
  missing; the import test patches `nltk.download` to a no-op so T1 downloads nothing. The NLTK
  tagger/cmudict data must be fetched in T3 (lexicon task) and recorded there.
- **D-9 .gitignore.** Ignores `data/ runs/ export/ logs/ out/`, bundles, all audio extensions,
  TensorBoard events; re-includes `tests/edge_cases/**` because Section 9 commits that suite.
- **D-10 Makefile.** `CONDA ?= conda` (overridable, no hard-coded path). `make env` creates or
  updates (`--prune`) each env, so it is safe to re-run. All other Section 10 targets are stubs.

- **D-11 MFA beam flags (option A, user standing instruction: take the recommended option).**
  Keep the Section 3 commands verbatim. `test_mfa_command_form` checks the positional form, absence
  of deprecation text, and the declared options `--single_speaker --clean -j` (`--help` rendered at
  `COLUMNS=200` so rich-click does not wrap names). New `test_mfa_passthrough_flags` asserts both
  commands accept extra args and that `PretrainedAligner.parse_args` returns
  `{beam: 100, retry_beam: 400}`. **Follow-up for T4:** the Python MFA wrapper must reject any
  pass-through flag not in `PretrainedAligner.get_configuration_parameters()` so typos fail loudly.
  Rejected: B (`--config_path` yaml; diverges from Section 3 text), C (older MFA; same mechanism
  across 3.x, loses fixes).
- **D-12 Pins.** Exact versions from the installed envs written into `envs/tts.yml` (incl.
  `cudnn=9.10.2.21` and `ctranslate2==4.8.5`, previously transitive, because GPU compatibility
  depends on them) and `envs/mfa.yml` (`montreal-forced-aligner=3.4.3`; resolves python 3.14.10,
  kalpy 0.10.5, kaldi 5.5.1172). pytorch/torchaudio keep a `cuda129*` build pattern so the CPU build
  can never be selected. Full transitive exports, with the machine-specific `prefix:` line removed,
  are committed as `envs/lock/{tts,mfa}.full.yml`.
- **D-13 README.** Placeholder with setup commands; command run order is filled in per task.

### Errors / incidents

- **E-1 `PackagesNotFoundError: soundfile` (first `make env`).** The conda-forge package is named
  `pysoundfile` (same library, imports as `soundfile`). This is a naming fix, not a substitution.
  Resolved: renamed in `envs/tts.yml` and in the import map in `tests/test_env.py`. Re-checked
  every other conda name with `conda search`; all resolve.
- **E-2 `--beam` / `--retry_beam` not in MFA 3.4.3 `--help` (first `make test`: 34 passed, 2 failed).**
  `test_mfa_command_form[align]` and `[align_one]` failed: neither help text lists `--beam` or
  `--retry_beam`. Everything else matches Section 3: positional forms
  (`align CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY`,
  `align_one SOUND_FILE_PATH TEXT_FILE_PATH DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_PATH`) are
  unchanged, no deprecation text appears, and `--single_speaker`, `--clean`, `-j` are listed.
  Source inspection (`montreal_forced_aligner/command_line/align.py`, `align_one.py`, `abc.py`):
  both commands use `ignore_unknown_options=True, allow_extra_args=True`; extra `--name value`
  pairs go through `PretrainedAligner.parse_args`, which keeps them only if `name` is in
  `get_configuration_parameters()`, and **silently drops unknown names**. `beam` (default 10) and
  `retry_beam` (default 40) are parameters of `alignment/mixins.py`, so the Section 3 flags are
  accepted as pass-through config options, but a typo would be ignored without error.
  Per user instruction, T1 stopped here (no pinning, no commit) pending a decision.
  Resolved: option A chosen (D-11). Confirmed directly that MFA's parser maps
  `--beam 100 --retry_beam 400` correctly and silently drops the typo `--retry-beam 7`.
- **E-3 `test_import[cudnn]` failed after pinning `cudnn`.** Intended guard: every yml package needs an
  import mapping. cuDNN is a native library, not a Python module. Resolved: added to
  `NOT_IMPORTABLE`; new `test_cudnn9` checks `torch.backends.cudnn.version()` is 9.x.
- **E-4 `conda env update --dry-run` unsupported (conda 24.11.1).** Verified the pinned yml files
  instead by re-running `make env` (update + `--prune`): no conda changes, all pip requirements
  "already satisfied", then `make test` 40/40.
- **Known warnings (not errors, left as is):** (a) pytest 9.2 `PytestRemovedIn10Warning` from the
  anyio pytest plugin (`CallSpec2` alias), harmless until pytest 10; (b) torchaudio 2.8 deprecation
  warning on `list_audio_backends` from SpeechBrain, expected per D-4 (APIs are removed in 2.9, which
  is why 2.8 is pinned).

## T2 — data and splits (Sections 2, 10)

### Decisions

- **D-14 Filelists are fetched by `make data`, not committed.** Source pinned to jaywalnut310/vits
  commit `2e561ba` (repo head, 2021-06-14); URL template, commit, line counts and sha256 of each
  file live in `configs/splits.yaml`. Downloaded once to `data/filelists/vits/` (git-ignored) and
  re-verified on every run; a hash mismatch aborts. Verified: 12,500 / 100 / 500 IDs, disjoint,
  union equals the 13,100 `metadata.csv` IDs. This is the only network access in T2.
- **D-15 Config inheritance.** `tts/config.py:load_config` resolves an `inherit:` key relative to the
  config file and merges with OmegaConf, then applies `key=value` overrides. `configs/base.yaml`
  holds only seed, freeze tag, paths and the sample rate so far; later tasks extend it.
- **D-16 Library modules outside the Section 10 package list.** `tts/config.py`, `tts/guard.py`,
  `tts/splits.py` sit at the package top level because they serve every subpackage. Scripts add the
  repo root to `sys.path` themselves so `python scripts/<x>.py` works without `make` (Section 9 demo).
- **D-17 Test-split lock (`tts/guard.py`).** `assert_audio_allowed(speaker, cfg)` raises
  `SplitLockedError` for test speakers unless `git rev-parse refs/tags/freeze-v1` succeeds;
  `assert_split_allowed('test', cfg)` is for `evaluate.py --split test`. The exception was first
  named `TestSplitLocked`, renamed because pytest would try to collect a `Test*` class.
  A unit test checks the test-speaker list in `splits.yaml` equals the one in CLAUDE.md.
- **D-18 What T2 reads for test speakers.** Directory listings (file names) and transcript `.txt`
  text only; no audio bytes and no wav headers. `sr` / `n_samples` in `l2_test.jsonl` are null until
  the freeze tag exists. The unit test fills fake test wavs with junk bytes so any read would fail.
- **D-19 Evaluation set = intersection of `annotation/` file names per split** (Section 2).
  Result: dev 100 IDs (× 12 = 1,200), test 104 IDs (× 12 = 1,248). 99 IDs are annotated by all 24
  speakers; each split adds a few its own speakers share. Only gap: YBAA lacks `arctic_a0094`
  (149 annotations), which is therefore excluded from test. Lists in `data/splits/{dev,test}_eval_ids.txt`;
  counts in `reports/data_splits.md`. Note: the 99 shared IDs mean dev tuning and test use the same
  sentences read by different accents; this is inherent to L2-ARCTIC and acceptable because the split
  is by accent (speaker), not by text.
- **D-20 Manifest skeletons.** Section 10 schema; fields that need T3/T4/T5 (`words`, `phones`,
  `durations`, `word_ids`, `n_frames`, `ref_words_*`) are `null`. LJ `text` is `metadata.csv`
  column 3 as-is (T3's normalizer is applied later). L2 rows add `utt` (ARCTIC ID) and
  `in_eval_set` (bool) beyond the plan's schema, so loaders can select the evaluation set and the
  Ablation B pool (all dev rows) from one file. `l2_dev` 13,450 rows, `l2_test` 13,417, `aug_a` and
  `aug_b` empty. `make data` never overwrites a non-empty manifest (later tasks fill them) unless
  run with `make data ARGS=--force`.
- **D-21 LJ audio check.** Every one of the 13,100 headers is checked for 22,050 Hz / mono / PCM_16
  (Section 2 "use as is"); all pass.

### Errors / incidents

- **E-5 Misleading report line (first `make data`).** The report flagged every speaker as having
  "fewer annotations than the 300-ID union", because speakers were annotated on different 150-ID
  subsets. Resolved: the report now lists, per speaker, the IDs it alone keeps out of its split's
  common set (only YBAA / `arctic_a0094`).

## T3 — normalizer, lexicon, symbols (Sections 2, 3, 4 change 3)

### Decisions

- **D-22 MFA models pinned and kept in the project.** `MFA_ROOT_DIR=data/models/mfa` (git-ignored,
  bundled in T13) so models are offline and independent of `~/Documents/MFA`. Pins in
  `configs/base.yaml`: acoustic `english_us_arpa` v3.0.0 (sha256 `d35ce271…`; internal version
  string `3.0.0a2.dev0+g940e907`, trained 2024-02-10) and G2P `english_us_arpa` v2.0.0a
  (`0ef6a3b2…`). Verified that `mfa model download … --version <pin>` into an empty root reproduces
  both checksums. `make assets` downloads only when missing and aborts on any hash mismatch.
- **D-23 MFA's G2P model is downloaded (Section 3 lists it) but not used.** Pronunciations come only
  from `lexicon.txt` (cmudict + g2p_en), per CLAUDE.md and Section 3.
- **D-24 Phone set.** The acoustic model's phone set is exactly the 69 ARPAbet tokens of
  `tts/text/symbols.py` (15 vowels × 3 stresses + 24 consonants); `pad, sil, sp` + 69 = 72 =
  `n_vocab`. `test_mfa_phone_set_equals_symbols` asserts equality, and `make lexicon` fails fast if
  any lexicon phone is outside the model's set.
- **D-25 g2p_en usage.** OOV words use `G2p().predict(letters)` (the neural model) directly, not
  `G2p()(text)`, which would redo its own number expansion, POS tagging and nltk-cmudict lookup.
  Apostrophes are dropped before prediction (its grapheme set is a–z). g2p_en still needs the NLTK
  `cmudict` corpus to construct and checks for `averaged_perceptron_tagger` on import, so both are
  pinned by sha256 in `data/models/nltk_data`; `use_local_nltk` points NLTK there and makes any
  download attempt raise.
- **D-26 Possessive rule (deviation from the literal "OOV from g2p_en"; recommended option taken).**
  141 OOVs were possessives; g2p_en mangled many whose base is in cmudict (e.g. `companion's` →
  `K AH0 M P EY1 N IY0 AH0 N Z`). Now `x's` with `x` in cmudict gets cmudict(x) + `IH0 Z`
  (after S Z SH ZH CH JH) / `S` (after P T K F TH) / `Z`. Affects 104 words; flag
  `lexicon.possessive_rule` in `configs/base.yaml` turns it off. Still built only in
  `tts/text/lexicon.py`, so "the lexicon is the only source" holds.
- **D-27 Lexicon contents.** Words from LJ `metadata.csv` column 3, all 24 L2 speakers' transcripts
  (text only; no audio, consistent with the test lock) and the 22 suitcase transcripts, each passed
  through the normalizer. Result: 14,912 words; 13,640 cmudict, 104 possessive rule, 1,168 g2p_en
  (LJ 1,235→mostly rare/archaic words and names; L2 28; suitcase 9). All 69 phones used. Full OOV
  list with pronunciations in `reports/lexicon.md` for review. Build is deterministic
  (lexicon sha256 `0969f40c…` on two runs). `ensure_words()` appends inference OOVs (Section 3).
- **D-28 Normalizer design (`tts/text/normalize.py`).** Rule order: currency (before `unidecode`,
  which would turn `£` into "PS") → symbols → `unidecode` → dates → times → abbreviations /
  initialisms → numbers (decades, ordinals, %, fractions, `#`, minus, ranges, decimals, years,
  cardinals). Tables (abbreviations, phrases, symbols, currency, `year_range` [1100, 2099]) live in
  `configs/base.yaml`. Output keeps `. , ? ! ; :` attached to the preceding word so Ablation A can
  find punctuation-followed boundaries (`words_with_punct`); `words()` gives `.lab`/lexicon tokens
  (`[a-z]+('[a-z]+)*`). Any digit left after normalization raises `NormalizeError` rather than
  being dropped. Conventions chosen: no "and" in numbers ("one hundred twenty"); bare 4-digit
  numbers in range are years ("1500 men" → "fifteen hundred men"); `St.` = saint before a
  capitalised word, else street; dotted initialisms are spelled ("U.S." → "u s"); all-caps words
  without dots are left to the lexicon (cmudict has e.g. "fbi").
- **D-29 Sentence-final periods after abbreviations.** A period that is both an abbreviation dot
  and a sentence end is kept for `a.m./p.m.`, `etc.`, and dotted initialisms when followed by a
  capital or end of text; for titles (`mr.`, `dr.` …) and `e.g./i.e.` only at end of text, because
  they are normally followed by a capitalised name/word.
- **D-30 Lexicon CLI as a module.** `python -m tts.text.lexicon` and `python -m tts.assets` instead
  of new files in `scripts/`, keeping `scripts/` to the Section 10 list.

### Errors / incidents

- **E-6 `AttributeError: 'str' object has no attribute 'stem'`** when reading the acoustic model in
  MFA 3.4.3: `AcousticModel()` needs a `pathlib.Path`, not a string. Resolved in `tts/align/mfa.py`.
- **E-7 Abbreviation dots swallowed sentence ends** ("at 7am." lost its period; first normalizer
  draft). Resolved by D-29.
- **E-8 Test failures on first run (30 passed, 2 failed):** (a) "e.g. Tom" gained a stray period from
  the capital-letter rule, fixed by D-29's end-of-text-only rule for `e.g./i.e.`; (b) an OmegaConf
  dotted override into a list index (`nltk.resources.0.sha256=…`) raises "Cannot merge incompatible
  container types"; the test now edits the loaded config instead. Then `make test` 84/84.

## T4 — LJ alignment, TextGrid parser, filled LJ manifests, phone stats (Sections 3, 4, 5)

### Decisions

- **D-31 Pilot first.** 50-file pilot (`python -m tts.align.lj --limit 50`, separate `_pilot` dirs)
  before the full run: 0.5 min, 50/50 parsed. Checked MFA's per-job logs
  (`data/mfa_tmp/.../alignment/log/align.*.log`) show `'beam': 100` and `'retry_beam': 400`, i.e. the
  pass-through flags really take effect (closes the D-11 follow-up together with D-32).
- **D-32 MFA wrapper (`tts/align/mfa.py`).** `align_corpus` / `align_one` build the Section 3 commands
  from config (`mfa.num_jobs`, `mfa.passthrough`), with `--temporary_directory data/mfa_tmp` so
  MFA's scratch stays out of the bundled model dir. `passthrough_args` asks MFA for
  `PretrainedAligner.get_configuration_parameters()` and raises on any name MFA would silently drop.
- **D-33 Every speech phone gets >= 1 frame.** Cumulative rounding (rule 3) left 382 speech phones
  (in kept rows) at 0 frames: MFA's 10 ms phones are shorter than one 11.6 ms frame. A 0-frame phone
  breaks the log-duration loss (VITS uses log(d + 1e-6) = -13.8) and the stats. The parser now gives
  such a phone 1 frame taken from its longer neighbour (ties: the previous one); sums are unchanged,
  boundaries move by <= 1 frame. Applied in the shared parser, so training, references and outputs
  are treated identically. Reported in `reports/align_lj.md`.
- **D-34 Leading/trailing `sil` always present.** LJ clips are trimmed, so speech often starts at
  t = 0; the token sequence still starts and ends with `sil` (possibly 0 frames) to match Section 5
  step 4, which always places `sil` before the first and after the last word.
- **D-35 Frame count.** `tts/audio.py:spectrogram` mirrors VITS `spectrogram_torch` (reflect pad
  (n_fft - hop)/2, `center=False`), giving `N // 256` frames; `n_frames` calls it on the real
  waveform as Section 3 requires. The cumulative-rounding sum `round(N/256)` was exactly +1 above it
  for all 13,100 files, absorbed by the last token. Inputs shorter than 385 samples cannot be padded
  (same as VITS); irrelevant for real audio.
- **D-36 Word check.** Besides the plan's QC reasons, a file is dropped if MFA's word sequence differs
  from the normalized transcript (`word_mismatch`); none occurred.
- **D-37 Manifests from the source of truth.** `make align-lj` rebuilds `lj_{train,val,test}.jsonl`
  from the VITS filelists + metadata (not from the manifests), so re-runs can't lose rows; dropped
  files are removed from the manifests and listed in `data/align/lj_dropped.tsv`. `text` is now the
  normalized text; `words` is the aligned word list.
- **D-38 Stats.** `stats/phone_logdur.json` (committed): natural log of frames over the 12,474 kept
  LJ train rows, 858,034 phone tokens, median sigma 0.434; AW0, OY0, OY2, UH0 have < 50 occurrences
  and use the median sigma (flagged `sigma_from_median`). Augmented rows are excluded by design.

### Results

- `reports/align_lj.md`: **13,073 / 13,100 kept (99.79%)**, train 12,474 / val 100 / test 499;
  all 27 drops are "non-silence phone > 0.5 s". MFA wall time 14.8 min (`-j 8`, 8 GB RAM was enough).
  31,650 `sp` tokens; 1,710 short silences merged.

### Errors / incidents

- **E-9 Two T4 test bugs:** a 256-sample input is shorter than the 384-sample reflect pad (test now
  uses >= 400 samples); `+key=value` is Hydra syntax that OmegaConf ignores, so the bad-flag test
  did not inject the typo (now set directly). Then the D-33 tie-break test expected the wrong
  neighbour; fixed the expectation (code was right).

## Autonomous run (2026-11-06 evening): T5-T14

The user granted a free hand to finish all laptop tasks overnight. Decisions follow the
"recommended option" rule; hard stops (test audio, unavailable pinned models) are reported in
docs/HANDOFF.md instead of being worked around.

## T5 — dev alignment and MFA-vs-manual evaluation (Sections 3, 7)

- **D-39 Scope.** `make align-l2-dev` aligns *all* 13,450 scripted dev utterances (needed for the
  Ablation B pool, Section 7) with speaker adaptation (one folder per speaker, no
  `--single_speaker`), parses them with the Section 3 rules (`n_frames` = round(N_ref·22050/sr/256),
  Section 5 step 8) and fills `l2_dev.jsonl` (`words`, `phones`, `durations`, `word_ids`,
  `n_frames`, `ref_words_mfa`, `ref_words_manual` for the 1,800 annotated IDs, `align_status`).
  `l2_test.jsonl` is untouched (test lock); its manual words are filled after freeze-v1.
- **D-40 Comparison method.** Words are matched by label sequence (difflib); phones are matched
  within matched words by canonical label: first field of L2-ARCTIC's `canon,perceived,tag`
  labels, stress removed, additions (`…,a`) excluded. Interval times only (Section 3). Each matched
  unit contributes start and end boundaries.
- **Result** (`reports/mfa_l2_dev.md`, `results/mfa_l2_dev.json`): 13,450/13,450 aligned; on the
  1,800 annotated utterances word MAE 19.4 ms (61.4% <= 20 ms, 92.4% <= 50 ms), phone MAE 13.1 ms
  (77.5% / 96.1%) [superseded: see "Correction to the T5 numbers" below — 75.0% / 93.8% and
  87.7% / 96.7%]; Hindi 18.2, Mandarin 20.2, Spanish 19.9 ms word MAE. MFA wall time 10.1 min.

## T6 — M0: FastSpeech2 + HiFi-GAN with injected durations (Section 4)

- **D-41 Weights from the official Google Drive folders via `gdown==5.2.0`** (new pinned pip
  package, added to envs/tts.yml). ming024/FastSpeech2 @ d4e79eb, `LJSpeech_900000.zip` ->
  `900000.pth.tar` (sha256 `8e248625…`); HiFi-GAN `generator_LJSpeech.pth.tar` from the repo's
  bundled zip (`bb4b0cb7…`). Unused AISHELL3/LibriTTS checkpoints from the same folder deleted.
- **D-42 Vendored inference code** (`tts/models/fastspeech2/`, MIT licences kept): transformer,
  FastSpeech2, variance adaptor, HiFi-GAN; the symbol table is reproduced in the original order so
  embedding indices match; module-global `device` replaced by the input's device. Mapping: phone
  -> `@`+phone, `sil` -> `@sil`, `sp` -> `@sp`; unknown symbols raise. Durations are passed as
  `d_targets` (training path); pitch/energy predicted. Output padded/trimmed to sum(d)·256.
- **D-43 Safe checkpoint loading everywhere.** All checkpoints load with `weights_only=True`. The
  2021 FastSpeech2/HiFi-GAN pickles contain numpy scalars; only `numpy.core.multiarray.scalar`
  (registered under its numpy-1 name) and numpy dtypes are allow-listed (`m0._safe_load`).
  (E-11: a first `weights_only=False` probe of the VITS checkpoint was refused by the session's
  safety classifier; the safe loader was the right call anyway.)

## T7 — VITS-MFA (Section 4 changes 1-6)

- **D-44 Pretrained VITS** from the official Drive folder (`pretrained_ljs.pth`, sha256
  `c94fb49d…`, pinned in base.yaml). It is generator-only: discriminators start from scratch (plan
  allows). Loading: 553 tensors loaded; skipped = `enc_p.emb.weight` (178 -> 72 symbols) + the 284
  stochastic-duration-predictor tensors; not initialized = new embedding + 10 deterministic-DP
  tensors (`runs/<exp>/init_skipped_keys.txt`). `test_pretrained_loads_only_expected_skips`
  asserts exactly this.
- **D-45 Code layout.** VITS `commons/modules/attentions/transforms/losses` vendored unchanged
  (relative imports only); `models.py` rewritten with changes 1-5; `data.py` (manifests, spec via
  `tts.audio.spectrogram`, collate, `MixedBucketSampler` 60/40); `trainer.py` (one train step,
  bf16 autocast with fp32 STFT/mel/losses, LSGAN, TensorBoard val). `use_pos` embedding is
  zero-initialized so M1+pos starts as M1.
- **D-46 Overfit check (acceptance).** `make overfit`: LJ019-0272 (a median-length ordinary
  sentence; the first pick was a chapter heading, re-picked), 500 steps, batch 1, from the
  pretrained generator. Duration loss 0.65 -> 0.013, val mel L1 0.40 -> 0.31; faster-whisper
  large-v3 transcribes "prisons were still far from inflicting the three punishments, hard
  labor, hard fare, and a hard bed." -> WER 5.6% (only the initial "the" missed). Exact length.
- **E-10 cudnn.benchmark = 13x slowdown.** With variable-length batches cuDNN re-tunes for every
  new shape: 6.2 s/step at batch 4 vs 0.457 s/step with it off. `train.cudnn_benchmark: false`.
  (The first 8 s/step measurement was also starved by MFA using all 12 cores.)

## T9 — pace augmentation A/B (Sections 6, 7)

- **D-47 One Rubber Band call per copy.** Speech phones get r_i via a `--timemap` anchored at every
  LJ phone boundary (R3 `--fine`, pitch preserved); silences are mapped 1:1 in that pass and then
  lengthened by looping their own audio (crossfaded so each lengthened silence starts and ends on
  its original samples — no clicks); inserted pauses loop the utterance's longest silence (room
  tone; fallback: lowest-RMS 50 ms window). Never digital zeros.
- **D-48 FLAC instead of WAV for augmented audio** (lossless 16-bit, 22.05 kHz): halves the GPU
  bundle; soundfile reads it transparently. Plan said WAV; content is identical.
- **D-49 Ablation B interpretation.** Target word length = rho_w x sum_p exp(mu_p) (rho is defined
  against the LJ means, Section 7 step 2); z_w solved by bisection under the [0.6, 2.5] clip
  (saturates when unreachable). Pauses: at each word boundary with P(sp)=0.223, length from the
  empirical histogram; an existing LJ sp is lengthened to max(existing, sampled), never shortened.
- **Pace stats B** (`stats/l2_dev_pace.json`): 13,450 utts, 119,472 words; utterance mean log rho
  0.159 (sd 0.186; dev speakers ~17% slower than LJ means); residual var 0.093, lag-1 -0.105;
  P(sp after word) 0.223. Uses only word ratios and pauses (Section 7 rules).
- **Results.** A: 24,948 clips in 6.2 min; B: 24,948 in 5.8 min (12 cores). Independent check
  (`tts.augment.verify`: MFA re-alignment of 100 random clips vs the planned boundaries): A median
  17.2 ms / MAE 28.6 ms (84% <= 50 ms); B median 23.0 / MAE 37.2 ms (77%) — of the order of MFA's
  own error, so the stretched audio follows its durations.
- **E-12 Augmentation crash** ("cannot give phone 2 (IY1) a frame"): a 1-frame phone compressed by
  0.6 between two 1-frame neighbours. The >= 1 frame rule now takes a frame from the *nearest*
  token that can spare one (identical result whenever a neighbour can spare one, so T4 manifests are
  unchanged); A regenerated, B unaffected (it never hit the case).
- **BLOCKED — UTMOS filter (E-13).** Loading `tarepan/SpeechMOS:v1.2.0` via
  `torch.hub.load(..., trust_repo=True)` executes code downloaded from GitHub and was refused by the
  session's safety classifier. Not worked around (no manual clone/import). `tts/augment/filter.py`
  is written and unit-tested; it must be run by the user (`make aug-filter`), see HANDOFF.md. Until
  then the filtered `aug_a.jsonl` / `aug_b.jsonl` do not exist and `abl_a`/`abl_b` refuse to train.

## T8 — train.py, smoke (Sections 4, 10)

- **D-50 train.py** implements the Section 10 flags (`--config --resume --init_from --tag --max_steps
  --batch_size --benchmark` + overrides). `--init_from` accepts the official VITS generator (Section
  4 change 3 path) or one of our checkpoints (generator + discriminator weights, step reset: the
  one-GPU branching design). Checkpoints every 2,000 steps + `ckpt_last.pt` (atomic writes), keep 3;
  SIGTERM/SIGINT saves and stops; non-finite loss saves and exits 2; lr decay per epoch.
  `train.max_steps` in base.yaml is a placeholder (100k) that the runbook overrides from the
  300-step benchmark so M1/A/B share one value.
- **Smoke result** (`make smoke`): m1, m1_pos, abl_a, abl_b each 200 steps at batch 4; resume
  200 -> 250 works; `--benchmark 20`: 0.463 s/step, 2.8 GiB peak on the RTX 4060 (batch 4).
  A/B smoke used the *unfiltered* `aug_*_all.jsonl` because the UTMOS filter is blocked (E-13);
  smoke only — real A/B training requires the filtered manifests.
- **E-16 `abl_b` smoke died with "CUDA error: unknown error"** while other GPU jobs (CTC, ASR, M0)
  shared the 8 GB laptop GPU; rerun alone it passed. WSL GPU instability under contention.
- **E-17 `scripts/profile.py` shadowed the stdlib `profile` module** (`module 'profile' has no
  attribute 'run'` inside torch) because Python puts the script's directory first on sys.path.
  Every script now replaces sys.path[0] with the repo root.
- **D-51 smoke-ok tag placement.** The runbook checks out `smoke-ok` and then runs
  `scripts/setup_gpu.sh` / `fetch_bundle.sh`, so the tag is placed after T13 (bundle scripts), not
  right after T8; no model/training code changes after the tag.

## T10 — inference pipeline, CLI, edge cases (Sections 5, 9)

- **D-52 Pipeline** (`tts/infer/pipeline.py`) follows Section 5 steps 1-10 literally; elastic fit
  by bisection on [-5, 5] (0.5-frame tolerance), then rescaled to hit D exactly; the cap inserts
  (or extends) an `sp` after the word; cumulative rounding + >= 1 frame per phone; output
  padded/trimmed to round(N_ref·22050/sr_ref). `timings.json` also records the stretch ratio
  (speech frames / native predicted frames) for plot 4. `phone_copy` needs MFA phone durations;
  with the CTC fallback it degrades to elastic and says so in `notes`.
- **D-53 Batch pre-alignment for evaluation.** `mfa align_one` costs ~14 s per reference (model
  load); batches (m0-demo, evaluate, tune) pre-align references and outputs in one
  `mfa align --single_speaker` pass — same acoustic model, lexicon and beam, and like align_one no
  speaker adaptation — and hand the TextGrid to the pipeline. Single-file inference (the live demo)
  still calls align_one.
- **E-14 `align_one` writes `<eps>` for word-tier silence** (corpus `align` writes ""), so every
  align_one result failed the word check and fell back to CTC. `<eps>` added to
  `align.silence_labels`.
- **E-15 conda-forge torchaudio 2.8.0 lacks the forced-alignment extension**
  (`forced_align ... Requires alignment extension`). **D-56:** keep the pinned MMS_FA acoustic model;
  implement CTC Viterbi forced alignment in numpy (`tts/infer/ctc.py:forced_align`, unit-tested).
  Same model and algorithm, no environment change the night before the GPU run. (D-4's reason for
  pinning torchaudio 2.8 was partly moot; 2.8 is still right for SpeechBrain's torchaudio.load.)
- **D-57 CTC word ends.** CTC emissions are spiky, so raw word ends come early. Gaps shorter than
  the Section 8 pause threshold (100 ms) are split at their midpoint (not a pause by definition).
  On 30 dev references vs manual: CTC MAE 58.1 -> 52.8 ms (MFA: 21.0 ms). CTC stays the fallback.
- **D-58 Edge suite** (`tests/edge_cases/`, 21 cases, 7.7 MB, committed; built by
  `make edge-build` from dev speakers BWC/NJS/ASI only): text normalization (digits, currency,
  date, abbreviation + name, ordinal, OOV name, no punctuation) by writing the *digit form* of what
  the ARCTIC speaker says; formats (MP3, 8 kHz, 48 kHz, stereo, clipped, noisy 10 dB SNR, reverb);
  timing (> 20 s, 1.8 s internal pause of room tone, 1.6x tempo, one word cut with the manual
  word tier); alignment (an unspoken transcript word, a false start, forced CTC path). The
  automatic MFA->CTC fallback is covered by `test_mfa_failure_falls_back_to_ctc`.
  **Result: 21/21 pass on GPU** (exact length). (CPU run: see HANDOFF.md.)

## T11 — evaluation harness (Section 8)

- **D-59 `scripts/evaluate.py`** caches every stage per utterance (synthesis, MFA-on-output,
  scores) so re-runs and tuning reuse work; references' ASR/accent/ECAPA are cached once per split
  in `results/_ref/`. Output word times: MFA (batch `--single_speaker`, D-53) on the output with the
  input text; MMS_FA cross-check. Pause F1 at word-boundary level over boundaries whose two words
  are matched; "no pauses on either side" counts as F1 = 1. Speaker similarity: output vs the mean
  ECAPA embedding of 100 LJ test clips (every 5th of the 499) and vs the reference utterance.
  Bootstrap 95% CIs (1,000 resamples, seed 1234); paired Wilcoxon vs M1 for every system.
- **D-60 Tolerance comparisons use a 1 µs epsilon** (|0.70 - 0.72| = 0.020000000000000018 s must
  count as "within 20 ms"); also applied to the T5 summary.
- **UTMOS** cannot be loaded in this session (E-13). `--no_utmos` makes this explicit: UTMOS =
  NaN with a note in summary.json; a later run without the flag backfills UTMOS into the cached
  scores (outputs and LJ ground truth). No other MOS model is substituted.

## T12-T14

- **D-61 Profiling** (`scripts/profile.py`): parameters per module (inference path vs training
  only), ptflops MACs for 1 s / 5 s outputs (text encoder + DP, flow + decoder) and a hand-worked
  `dec.conv_pre` (7 x 192 x 512 x 86 = 59.2 MMACs per second of audio); latency per stage with
  10 warm-up + 50 timed runs except MFA align_one (1 + 5: ~14 s per call); RTF model-only and
  end-to-end; RTF vs output length (1-20 s).
- **D-62 Bundle** (`make bundle`): uncompressed tar (audio does not compress) of exactly the files
  the training manifests reference + metadata, filelists, lexicon, pretrained VITS; per-file sha256
  in `data/BUNDLE_SHA256.txt`; then a fresh `git archive HEAD` + the tar are unpacked in a temp dir
  and `python -m tts.check_data` must pass (T13 acceptance). `BUNDLE_AUG=0` makes a ~3.8 GB light
  bundle (no augmented audio; regenerate on the instance, Section 10 allows it). No L2 audio.
- **D-63 Demo** (`make demo`): static `out/demo_site/index.html` with results tables (95% CIs),
  plots, and per sample the reference and every available system's output + timing plot; dev
  samples labelled "dev" until freeze-v1; dev-speaker suitcase clips and `demo/own_recordings/`
  labelled "stress test". The per-sample outputs double as the backup outputs.
- **D-64 Tune** (`make tune`, dev only): elastic cap x noise_scale grid + transfer variants + CTC
  aligner on a fixed 120-utterance dev subset; choice = highest UTMOS among settings within 5 ms of
  the best word-boundary MAE (lowest WER if UTMOS is unavailable); written into
  configs/inference.yaml (`# tuned`).

### T11 incidents

- **E-18 SpeechBrain models tried to download from the Hub and crashed**
  (`hf_hub_download() got an unexpected keyword argument 'use_auth_token'`): the accent model card's
  `hyperparams.yaml` sets `pretrained_path: Jzuluaga/accent-id-commonaccent_ecapa`, and speechbrain
  1.0.3 still passes `use_auth_token`, which huggingface_hub 1.x removed. Fixed by loading with
  `overrides={"pretrained_path": <pinned local snapshot>}` — fully offline, same pinned files, no
  package change. (Latent incompatibility speechbrain 1.0.3 / huggingface_hub 1.29 noted; nothing in
  the project calls the Hub through speechbrain any more.)
- **D-65 US probability = softmax(30 x cosine).** `classify_batch` of the CommonAccent ECAPA model
  returns additive-angular-margin cosine scores (sum ~20), not probabilities; my first heuristic
  produced 0.09 for LJSpeech. The training recipe (JuanPZuluaga/accent-recog-slt2022 @ a2ff115,
  `train_ecapa_tdnn.yaml`: AdditiveAngularMargin margin 0.2, **scale 30**) gives the logit scale;
  `eval_models.accent.score_scale: 30`. Sanity check: LJ001-0001 0.90; dev ASI (Hindi) 0.25, BWC
  (Mandarin) 0.16, NJS (Spanish) 0.93; an M0 output 0.98. Predicted labels were already right
  (LJ -> us, ASI -> indian), only the probability scaling changed.

### T13 result and incidents

- **Bundle** (`make bundle`, 8.7 min): `export/bundle.tar` 13 GB (13,767,915,520 bytes), 62,983 files,
  sha256 `1b283ea2d7ca8d452fef5daf0ad2b55e692d745013a282732cbedc1ead953bf1`. Fresh unpack + `git archive
  HEAD` in a temp dir: **check-data PASS** (all checksums; lj_train 12,474 / val 100 / test 499 and
  aug_a_all / aug_b_all 24,948 rows each, 0 duration-sum or audio errors).
- **E-19 Silent-ablation hazard found by check-data's output.** The bundle carries the empty
  `aug_a.jsonl` / `aug_b.jsonl` skeletons from T2; with them `abl_a` would have trained with
  aug_fraction silently forced to 0 (the sampler drops the augmented share when the pool is empty),
  i.e. as a second M1. `train.py` now exits (code 3) when aug_fraction > 0 and the aug manifests
  have no rows. Copy the UTMOS-filtered manifests over them (HANDOFF 2a) before training A/B.
- **E-20 "CUDA failed with error unknown error" in ctranslate2** during the M0 dev evaluation while
  the bundle build saturated disk and CPU — the same WSL GPU instability as E-16. Re-run alone;
  stage caching meant no work was lost.
- **E-21 ptflops counted ~0 MACs** with its module-hook backend (0.001 GMACs encoder, 0 flow+decoder);
  the `aten` backend counts the actual ops: 27.2 GMACs per second of audio (flow+decoder 27.1, text
  encoder + DP 0.08). `scripts/profile.py` now passes `backend="aten"`.
- **Correction to the T5 numbers (D-60 effect).** Both MFA and the manual annotations sit on a 10 ms
  grid, so many errors are exactly 20 or 50 ms and float rounding had excluded them. After the epsilon
  fix (`make align-dev-eval` re-run): word <= 20 ms **75.0%** (was reported 61.4%), <= 50 ms 93.8%;
  phone <= 20 ms 87.7%, <= 50 ms 96.7%. MAEs unchanged (word 19.4 ms, phone 13.1 ms).

## End-of-night results (dev only; test locked)

- **M0 on dev** (`make eval SYSTEM=m0 SPLIT=dev ARGS=--no_utmos`, 1,200/1,200 ok, all MFA-aligned):
  length error 0 on every utterance; word-boundary MAE vs manual 31.2 ms (55.4% <= 20 ms, 84.7%
  <= 50 ms; MMS_FA cross-check 50.7 ms); pause F1 0.748; WER 10.0% (references 4.1%, LJ test ground
  truth 1.9%); US probability 0.731 (references 0.298); speaker similarity to LJ 0.763, to the
  reference speaker 0.051; UTMOS not measured (E-13). Per accent in results/m0/dev/summary.json.
- **Profile** (laptop RTX 4060, smoke checkpoint = M1 architecture): 28.1 M inference parameters
  (decoder 14.3 M, flow 7.1 M, text encoder 6.3 M, DP 0.35 M), 54.0 M training-only; 27.2 GMACs/s
  of audio; p50 latency text encoder + durations 9 ms, flow + decoder 46 ms, `mfa align_one` ~5 s;
  RTF model-only 0.015, end-to-end 1.35 (MFA-bound); RTF 0.027 (1 s) -> 0.011 (20 s).
- **Edge suite** 21/21 on GPU and CPU (M0). **Demo page** built for M0 (9 samples).

## 2026-11-07 — UTMOS weights pinned and loaded safely (follow-up to E-13)

- **D-66 UTMOS weights: sha256 pin + `weights_only=True`.** The user reviewed
  `tarepan/SpeechMOS@v1.2.0` and pointed out that `hubconf.py` downloads the weights from the
  *v1.0.0* release asset (`utmos22_strong_step7459_v1.pt`). Checked: it calls
  `torch.hub.load_state_dict_from_url(...)` with defaults `check_hash=False` and, in torch 2.8,
  `weights_only=False` (a full pickle load). So the tag pins the code but not the weights, and the
  weights load would be able to run code. Change: `tts/eval/assets.py::utmos_weights` downloads that
  exact URL to `data/models/eval/utmos/`, checks sha256
  `38aa51ab79e2a4e09a1449758a4b37e9cbb2e8235a49662a732d33a9ba1e9bff` (411,179,119 bytes, downloaded
  2026-11-07), and `load_utmos` builds the model with `torch.hub.load(..., pretrained=False)` and
  loads the state dict with `torch.load(weights_only=True)`. Verified the file loads that way: plain
  OrderedDict, 225 tensors, 102.8 M parameters. Same model, same weights — no substitution. The
  repo's Python code (hubconf.py, speechmos/utmos22) still runs via torch.hub; that is what the user
  reviewed. Test: `tests/test_eval.py::test_utmos_weights_pinned`.

## 2026-11-07 — Augmentation stretcher: Rubber Band R3 -> WSOLA (deviation from PLAN Section 6)

- **E-22 Rubber Band R3 degrades LJ speech even at mild ratios.** The first UTMOS filter run on set
  A (R3 data) gave originals 4.376 vs augmented 2.917 (mean drop 1.459, p95 2.211); the drop was
  ~1.2 even for clips whose total length changed < 5%. Controlled test on 6 LJ train clips, uniform
  stretch (UTMOS, length factor 0.7 / 0.95 / 1.05 / 1.2 / 1.5 / 2.0; original 4.43):
  R3 `--fine` 3.28 / 3.61 / 3.49 / 3.38 / 3.08 / 2.48; R3 + `--window-short` 3.09 / 3.48 / 3.39 /
  3.35 / 3.02 / 2.42; R2 (default engine) at 1.2x 2.21; `sox tempo -s` (WSOLA) 4.23 / 4.43 / 4.42 /
  4.39 / 4.17 / 3.22. An identity map passes through unchanged, so the loss is the phase-vocoder
  processing itself, not our time map / room tone / FLAC path. Since every clip is affected, the 5%
  filter cannot remove it; training A/B on it would mostly measure the stretcher's artifacts.
  Listening samples: `out/diag_stretch/` (not in git). R3 reports kept in `reports/r3/`.
- **D-67 Use time-varying WSOLA (user decision, 2026-11-07).** Options put to the user: switch to
  WSOLA / keep R3 as planned and report it / build both. The user chose WSOLA.
  `tts/augment/wsola.py`: WSOLA with the same sample-level time map R3 received (anchors at every
  LJ phone boundary; speech phones x r_i, silences 1:1), periodic Hann, 50% overlap, normalized
  cross-correlation search with a tiny bias to zero lag (identity map = exact copy). sox itself
  only does constant tempo, so a time-varying implementation was needed; numpy only, no new package.
  Parameters chosen by UTMOS on the same 6 clips x 5 ratios: frame 15 ms, tolerance 5 ms
  (0.7 / 0.95 / 1.2 / 1.5 / 2.0: 4.26 / 4.43 / 4.40 / 4.31 / 3.83 — at or above sox everywhere; larger
  frames/tolerances were worse, e.g. 40 ms: 3.88 / 4.40 / 4.14 / 3.54 / 2.65). Held-out check through
  the real A pipeline (first 30 LJ train utterances x 2 seeds, mean length factor 1.16): mean UTMOS
  drop R3 1.336 -> WSOLA 0.321 (p95 2.04 -> 0.66), better in every stretch bin. ~0.03 s per clip.
  Config: `augment.stretcher: wsola`, `augment.wsola: {frame_ms: 15, tolerance_ms: 5}`;
  `stretcher: rubberband` still reproduces the PLAN Section 6 data. Tests:
  `tests/test_augment.py` (identity, length, pitch, RMS, time-varying anchor, render with both).
  Everything downstream is unchanged (plans, seeds, durations, MFA verification, UTMOS filter,
  sampler, `max_steps`). For the slides: this is a deviation from Section 6, with the numbers above
  as the reason. Caveat: the parameters were picked with UTMOS, which is also an evaluation metric;
  the selection used LJ ground-truth clips only (no system outputs, no L2 audio).
- **Regenerated sets (WSOLA, 2026-11-07).** `make aug-a aug-b`: generation 2.5 / 3.7 min (12
  workers); MFA check of 100 clips vs planned word boundaries, median 6.8 / 7.2 ms (R3: 17 / 23 ms);
  UTMOS filter, 23,701 of 24,948 kept in each. A: augmented 4.039 (kept 4.070) vs original 4.376,
  drop threshold 0.777 (R3: 2.917, threshold 2.211). B: augmented 3.620 (kept 3.660), threshold 1.324.
- **Why B scores lower than A (no change made).** B's extra drop follows its inserted pauses (L2
  pace statistics, P(pause after word) 0.22): mean 0.176 inserted pauses per word in B vs 0.001 in A;
  corr(UTMOS drop, pauses per word) 0.50 in B. B clips with no inserted pause drop 0.38 (A overall
  0.33); 0.1-0.2 pauses/word 0.72; > 0.3 1.01. So UTMOS is penalising the L2-like pause pattern B is
  designed to contain, not the stretcher. The 5% filter is applied as specified (it removes the most
  pause-heavy clips first; noted as a possible bias of Ablation B toward fewer pauses).

## 2026-11-07 — GPU plan: one L40, M1 first

- **D-68 One L40; M1 is the priority, ablations decided afterwards (user decision).** Only a single
  L40 is available. Phase 1 trains M1 alone on the whole budget (`train.max_steps` from the
  300-step benchmark, committed from the laptop). The ablations are decided after M1's dev results;
  if they run, they use PLAN Section 10's one-GPU branching design: three equal-length
  continuations from the finished M1 checkpoint (`m1_cont`, `abl_a`, `abl_b`, same `--max_steps K`
  on the command line, fresh optimizer/LR schedule each), so the ablation comparison is
  `m1_cont` vs `abl_a` vs `abl_b`, not `m1`. Consequences: (1) with branching, the ablations measure
  pace augmentation introduced late in fine-tuning, not from the start — to be stated in the
  write-up; (2) `freeze-v1` (and with it the test evaluation) waits until the set of systems is
  final, because experiment configs cannot change after it. Runbook: docs/HANDOFF.md section 3.

## 2026-11-07 — GPU instance setup

- **E-23 `CondaToSNonInteractiveError` on the instance.** Recent Miniconda installers configure
  Anaconda's `pkgs/main` and `pkgs/r` channels and refuse to run non-interactively until their Terms
  of Service are accepted. Our envs never use them (conda-forge + `nodefaults`), so rather than
  accepting the ToS on the user's behalf, `scripts/setup_gpu.sh` now installs Miniforge
  (conda-forge only) when no conda exists, and always sets the channels to conda-forge only
  (`--remove channels defaults`, strict priority). Package pins are unchanged.
- **D-69 M1-only bundle; the instance downloads LJSpeech itself.** The laptop upload runs at ~1 MB/s
  (13 GB = ~3.7 h). M1 needs neither the augmented audio (9.2 GB) nor our copy of the LJ wavs
  (3.6 GB, the untouched official files). `make bundle-m1` (`BUNDLE_AUG=0 BUNDLE_LJWAV=0`) packs
  manifests, filelists, lexicon, metadata and the pretrained VITS generator (~0.3 GB) while still
  listing and hashing every LJ wav in data/BUNDLE_SHA256.txt; on the instance `make lj-download`
  fetches the official archive (`paths.lj_archive_url`), extracts LJSpeech-1.1/ (members outside it
  refused) and `make check-data` verifies every wav against the laptop's sha256. The full 13 GB
  bundle is unchanged (lists kept as data/BUNDLE_*.full.txt); for phase 2 the augmented audio can be
  regenerated on the instance or uploaded separately.
- **E-24 pip conflict on the NVIDIA PyTorch image.** `conda env create` failed in the pip step:
  "speechbrain 1.0.3 depends on torch>=1.9; The user requested (constraint)
  torch==2.8.0a0+5228986c39.nv25.6". The NGC-style image sets a global pip constraint pinning its
  own torch build, which pip also applied inside our separate conda env. `setup_gpu.sh` now unsets
  `PIP_CONSTRAINT` (and index overrides) and sets `PIP_CONFIG_FILE=/dev/null` before creating the
  env, so pip installs exactly envs/tts.yml against the env's conda-forge torch 2.8.0. No pin changed.
- **E-25 setup_gpu.sh CUDA check never ran.** It was `conda run -n tts python - <<'PY'`; `conda run`
  does not forward stdin, so python got an empty script and exited 0 without printing. Now
  `conda run --no-capture-output -n tts python -c "..."` (fails the script if CUDA is not visible).
- **D-70 M1 budget: 8 h on the L40S, `max_steps` 114,000; `num_workers` 16.** Benchmark on the instance
  (`--benchmark 300`, batch 32): 0.229 s/step mean (0.227 median), 9.8 GiB peak of 46 GB. The user chose
  8 h; 7.5 h of training (0.5 h kept for profiling + export) = ~117,900 steps, minus ~3% for validation
  and checkpointing every 2k steps -> 114,000. Batch stays 32 (PLAN Section 4): spare memory is not
  spare speed, and a larger batch would change the experiment. The instance has 60 CPUs, so
  `num_workers` 8 -> 16 so loading never stalls the GPU; it changes loader throughput only, not batch
  contents or order.
- **D-71 Batch-size throughput test planned after M1 (user request).** The L40S runs batch 32 at
  0.229 s/step with 88-96% GPU utilisation and 11.8 of 46 GB used. The user wants to measure whether a
  bigger batch gives more clips per second. It is measured after M1 finishes (sharing the GPU with
  the run would slow M1 and invalidate the numbers): `--benchmark 300 --batch_size {48,64,96}`
  (HANDOFF phase 1 step 6b; benchmark mode writes no files). M1 itself stays at batch 32 (PLAN
  Section 4). Using another batch later (e.g. the phase-2 branches) is an experimental change: all
  branches would use the same batch, with the learning rate decided and logged before the run.
  **Result (L40S, M1 paused at a checkpoint and resumed afterwards):** median s/step and clips/s:
  batch 32 0.227 / 141; 48 0.301 / 159 (+13%); **64 0.381 / 168 (+19%)**; 96 0.603 / 159 (+13%);
  peak memory 9.8 / 13.7 / 17.6 / 25.5 GiB. Throughput peaks at 64 and falls at 96: the GPU is near
  its compute limit. M1 continues at batch 32 (a mid-run change would mix two settings; per hour,
  batch 64 makes 40% fewer updates, so the gain in learning per hour is unproven). Proposed for the
  optional phase-2 branches: batch 64 with learning rate 2e-4 (the original VITS LJ setting,
  64 / 2e-4), the same for `m1_cont`, `abl_a` and `abl_b`; to be confirmed before that run. Resuming
  restarts the sampler's order for the current epoch; no practical effect on M1.
- **E-26 `make profile` could not run on the GPU instance (M1 end, 2026-11-07).** The runbook's
  `ARGS="profile.mfa_runs=0"` failed: `scripts/profile.py` accepts no key=value overrides
  ("unrecognized arguments"). Even with that fixed, its latency stage runs the full pipeline on
  `profile.reference_wav` (an L2 dev clip) with MFA, and bundle-m1 ships neither L2 audio nor the
  `mfa` env. Decision: skip profiling on the instance; export M1 and profile on the laptop (RTX 4060
  and CPU) per HANDOFF section 5. Consequence: no L40S latency row in the efficiency table (PLAN
  Section 8 lists L40/A100); the training benchmark (0.229 s/step, D-70) is the only L40S number.
  If phase 2 runs, the L40S profile can be added then by copying the reference clip + transcript.
  HANDOFF step 7 and README updated.

## Phase 1 results (M1 on one L40S, then the laptop; 2026-11-07/08)

- **M1 training run.** 114,000 steps at batch 32 (commit 8d1253c, clean tree), finished
  2026-11-07 19:03 instance time. One planned pause for the batch test (D-71), resumed from
  `ckpt_last.pt`. Losses: `val/dur` 0.134 (2k) -> 0.094 (114k), falling throughout; train KL
  3.24 -> ~2.7; `val/mel_l1` lowest 0.354 at 12k, 0.377 at 114k. The rise in reconstruction L1 is
  read as the GAN trade-off, not overfitting: the discriminators start from scratch, `fm`
  0.6 -> 3-4 and discriminator grad norm grow after ~45k while `loss_d` stays 2.2-3.0 and `adv`
  1.5-2.8 (balanced), and train and val mel drift together. Learning rate left unchanged (decay
  0.999875 per epoch ends at ~0.96x; same scheme as VITS, Light Speed, Coqui), which keeps M1/A/B
  hyperparameters identical (PLAN Section 4). Note for any future change: `ExponentialLR` stores
  `gamma` in the checkpoint, so a `train.lr_decay=` override has no effect on `--resume`.
  Logging quirk: train.log shortens keys, so `loss/d` and `grad/d` both print as `d=` (first one
  is the loss).
- **D-72 Tuned inference setting: elastic, cap 2.5, noise_scale 0.333, MFA** (`make tune`,
  21 settings on the fixed 120-clip dev subset; rule D-64). Best MAE 21.5 ms (no cap); chosen
  21.9 ms with the highest UTMOS 4.21, WER 2.5%. Patterns: noise_scale drives WER (0.333: 2.5-3.5%,
  0.667: 4.6-6.7%); a looser cap lowers MAE (26.1 ms at 1.4 -> 21.5 ms uncapped) with no UTMOS
  cost; `phone_copy` has the worst WER (10.6%) with slightly higher US prob (0.77) and lower
  speaker similarity (0.78), which supports copying word boundaries only; CTC aligner MAE 44.5 ms,
  WER 12.7%. Written to configs/inference.yaml (`# tuned`); grid in results/tune/grid.json.
- **D-73 M1 dev results (1,200 references, tuned setting) vs M0** (paired Wilcoxon,
  results/dev_comparison.csv): word-boundary MAE 31.3 -> 23.7 ms (p ~ 1e-140); within 20 ms
  55.4% -> 68.5%; pause F1 0.75 -> 0.84; WER 10.0% -> 3.5% (reference audio 4.1%); speaker
  similarity to LJ 0.76 -> 0.80; UTMOS 3.94 -> 4.20 (LJ ground truth 4.36); US probability
  0.73 vs 0.72, no difference (p = 0.61); length error 0 for both. Holds per accent (MAE: Hindi
  22.0, Spanish 23.6, Mandarin 25.4 ms). Caveat: tuning used 120 of these references, so M1's dev
  numbers are slightly optimistic. Laptop profile with the M1 checkpoint: model RTF 0.015,
  end-to-end 1.43 (MFA 5.3 s per reference). Edge suite 21/21 on GPU and CPU. Demo rebuilt with
  M0 + M1. Not tagged `freeze-v1`: the phase-2 decision (ablations) is the user's.

### Word-ending investigation (user listening report, 2026-11-08)

- **Report:** in the TensorBoard validation clips, M1's words "end sharply" while LJ "extends"
  the end of each word; the accent or style sounds not quite US.
- **Measurements** (ad-hoc probes, not repo modules: 100 LJ val clips, ground-truth MFA durations,
  M1 at noise 0.667 and 0.333, vs the LJ recordings):
  - **Energy envelope at word ends:** same as LJ. Before a pause both fade ~16-17 dB over the last
    8 frames; same dip between connected words.
  - **Pitch and voicing over the last 12 frames before a pause:** same shape (M1 ~0.4 st lower).
  - **Duration predictor:** keeps LJ's word-final lengthening (pre-pausal final phones GT 13.07
    vs predicted 13.36 frames; other phones within 3%).
  - **Accent ID on the same sentences:** M1 US prob 0.87-0.90 vs LJ recordings 0.74.
  - **What differs:** spectral movement *inside* phones (log-mel frame-to-frame distance, phones
    >= 8 frames): LJ 4.73, M1 4.05 (-14%); middle third 4.29 vs 3.41 (-20%). Phone-onset steps
    are the same (6.35), so the onset/inside ratio is 1.34 vs 1.57. M1 holds each phone steadier
    and then jumps.
- **Reading:** this points at the prior, not the lexicon, MFA or durations. With
  `use_pos: false`, every frame of a phone gets the same prior mean (project, then expand). The
  original VITS softened this with blank tokens between phones (`add_blank`), which PLAN Section 4
  change 2 removed because blanks have no MFA duration. M1+pos (change 5, `configs/m1_pos.yaml`)
  is the PLAN's own remedy. Its position embedding is zero-initialized, so it can branch from the
  M1 checkpoint. Not tested: the encoder has no word-boundary token (the original VITS input had
  spaces and punctuation), so word-final vs word-internal phones are only distinguishable from
  context.
- **Proposed (user to decide):** phase 2 starts with `m1_pos` and `m1_cont` branched from M1 for
  the same K steps. Compare them on within-phone movement, listening and dev eval; the ablations
  then use the winning architecture.
