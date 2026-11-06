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
