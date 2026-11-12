# meeami-tts: Final Project Report

**Time-aligned US-English text-to-speech on the word timing of a non-US reference**

Chaganti Raja Ravi Teja · November 2026 · code frozen at git tag `freeze-v1` (commit `5889ecf`)

How to read this report:
- Each claim points to the code that implements it and to the result file that measures it.
- Line links point to the frozen code. The one exception is V0, the immediate baseline (the original VITS with MAS). It was added after the freeze as new files (§7.0, §10.6, D-90), and its links point to those files.
- **A 16-page PDF version** with figures is [meeami-tts_report.pdf](meeami-tts_report.pdf) (`make report-pdf`; every table and figure is generated from the committed results).
- `D-n` and `E-n` are entries in [DECISIONS.md](DECISIONS.md), the running log of every decision and error.
- The original specification is [PLAN.md](PLAN.md). Section 13 lists every deviation from it.
- **After the freeze**, only inference tooling and evidence were added, never model code or experiment configs: the typed-text pre-pass (§5.1), hosting, the live and browser demos, `samples/` and the outsider check (§14.1), and the live-testing findings (§15). The test numbers in §11 are unchanged by all of them.

---

## Contents

1. [Problem statement](#1-problem-statement)
2. [Results in one page](#2-results-in-one-page)
3. [Design principle: one source per output property](#3-design-principle-one-source-per-output-property)
4. [Data and the accent-disjoint split](#4-data-and-the-accent-disjoint-split)
5. [Text front end: normalizer and lexicon](#5-text-front-end-normalizer-and-lexicon)
6. [Forced alignment (MFA) and its error on accented speech](#6-forced-alignment-mfa-and-its-error-on-accented-speech)
7. [Models](#7-models)
8. [Inference pipeline: transferring word timing](#8-inference-pipeline-transferring-word-timing)
9. [Evaluation methodology](#9-evaluation-methodology)
10. [The project's journey: experiments and the reasoning behind each](#10-the-projects-journey-experiments-and-the-reasoning-behind-each)
11. [Final results on held-out accents](#11-final-results-on-held-out-accents)
12. [Efficiency](#12-efficiency)
13. [Deviations from the plan](#13-deviations-from-the-plan)
14. [Engineering for integrity and robustness](#14-engineering-for-integrity-and-robustness)
15. [Limitations and future work](#15-limitations-and-future-work)
16. [Claim → evidence index](#16-claim--evidence-index)

---

## 1. Problem statement

**Task.** The system takes two inputs:
- an English text;
- a recording of that text read by a **non-US** speaker (the *reference*).

It must output speech that:
1. says **exactly the input words** with a **US accent**;
2. is in the **LJSpeech voice** (one US female speaker);
3. follows the reference's **timing**: the output is exactly as long as the reference, and each word starts and ends at about the reference time, pauses included.

The reference contributes **timing only**. Its voice, accent and pronunciation must not leak into the output.

**Why this is hard.** Part of an accent lives in timing: L2 speakers lengthen, shorten and pause differently from native speakers. So the system has to keep two things apart:
- **Word-level timing,** which it must copy from the reference.
- **Phone-level timing inside each word,** which carries accent and must stay native.

It also has to keep sounding natural when a word is forced to be much slower or faster than the voice would say it.

**Constraints that shaped the design:**

| Constraint | Consequence |
| --- | --- |
| Train on LJSpeech only | The voice and accent come from one US speaker. The L2 data is used only for evaluation, tuning and one ablation. |
| Evaluation must measure generalisation to unseen accents | L2-ARCTIC is split **by accent**. The test accents (Arabic, Korean, Vietnamese) are locked in code until the git tag `freeze-v1` exists (§4, §14). |
| One rented GPU (L40S) for about 8 h, plus an 8 GB laptop GPU | M1 is trained once on the full budget. Every later experiment branches from it for 20k steps (§10). |
| Deadline 2026-11-11 | The plan's priority order decided what to cut. |
| A live demo on unseen files | An edge-case suite, a CTC aligner fallback, and output length that is exact by construction (§8, §14). |

**Acceptance criteria and their metrics:**

| Requirement | Metric (§9) | Final result, main system, test accents |
| --- | --- | --- |
| Exact words | WER/CER by Whisper large-v3, both texts normalized the same way (§9.1) | WER 2.9%. For comparison: Whisper on the L2 input recordings themselves, 6.3%; on LJSpeech recordings, 1.9% |
| US accent | Accent-ID US probability | 0.76 (references 0.49) |
| LJSpeech voice | ECAPA cosine to LJ | 0.80 (to the reference speaker: 0.11) |
| Exact length | Output samples − expected samples | **0 on all 1,248 references** |
| Word timing | Word-boundary MAE against the manual annotation | 23.9 ms. MFA's own error on accented speech is 19.4 ms (§6). |
| Pauses | Pause F1 | 0.835 |
| Naturalness | UTMOS | 4.20 (LJ ground truth 4.36) |

---

## 2. Results in one page

**Main system:** `m1_pos_diph`. It is VITS fine-tuned with MFA durations instead of monotonic alignment search (MAS), with two additions:
- a position-in-phone embedding;
- each diphthong split into two tokens.

**Held-out test accents** (Arabic, Korean, Vietnamese; 12 speakers × 104 sentences = 1,248 references, evaluated once, after `freeze-v1`; D-86. V0 was added after the freeze, D-90):

| System | Word-boundary MAE (ms) | Within 20 ms | Pause F1 | WER | US prob | UTMOS | Spk sim LJ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L2 references | | | | 0.063 | 0.486 | | |
| **V0: original VITS with MAS** (immediate baseline) | 40.5 | 41.2% | 0.815 | **0.021** | 0.649 | **4.31** | 0.802 |
| M0: FastSpeech2 (second baseline) | 31.9 | 55.3% | 0.726 | 0.093 | 0.746 | 3.96 | 0.766 |
| M1: VITS-MFA | 24.2 | 69.3% | 0.822 | 0.038 | 0.734 | 4.19 | 0.805 |
| m1_pos | 24.6 | 68.3% | 0.826 | 0.044 | 0.781 | 4.17 | 0.806 |
| **m1_pos_diph (main)** | **23.9** | **69.6%** | **0.835** | **0.029** | 0.762 | **4.20** | 0.800 |
| abl_a_pos | 24.7 | 68.3% | 0.825 | 0.039 | 0.665 | 4.06 | 0.801 |
| abl_b_pos | 24.7 | 68.3% | 0.826 | 0.045 | 0.767 | 4.08 | 0.796 |

Sources: [`results/<system>/test/summary.json`](../results) (means with 95% bootstrap CIs) and [`results/paired/`](../results/paired) (paired tests).

**Six findings:**
1. **Replacing MAS with MFA durations is what makes the output follow the reference timing.** V0 is the same pretrained VITS before the swap, given the same word timing. Against V0, the main system's word boundaries are 16.5 ms closer (40.5 → 23.9 ms; within 20 ms 41% → 70%; p ≈ 1e-192). The gain holds in every accent and every stretch range. V0 renders its given durations about 20 ms early and scattered; the fine-tuned systems render them on time (§10.6). **The cost:** against V0, WER is +0.8 points and UTMOS −0.12. US probability is +0.11.
2. **VITS with MFA durations also beats the FastSpeech2 baseline on every axis but accent.** Against M0: −8 ms word-boundary error, WER 9.3% → 3.8%, UTMOS +0.23. US probability is level.
3. **Splitting diphthongs fixed a diagnosed defect.** It came from a listening report ("cry" heard as "cra", §10.5). It also lowered test WER from 4.4% to 2.9% and improved timing, against its control.
4. **Pace augmentation (ablations A and B) is a negative result.** Neither improved timing or WER in any stretch range, and both lowered UTMOS.
5. **Output length is exact for every system and reference.** This holds by construction, not by tuning (§8).
6. **The dev conclusions replicated on the unseen test accents.** The test split was read only after every model and setting was fixed.

---

## 3. Design principle: one source per output property

The pipeline has one rule: every property of the output has exactly one source. "Exactly one source" means that one input decides the property and no other input may influence it. Nothing is averaged or mixed between the reference and the model, so no property can leak into another.

| Output property | Its only source | Deliberately **not** taken from | Where in code |
| --- | --- | --- | --- |
| Words | Input text → normalizer | the reference audio, an ASR transcript | [`normalize.py#L247`](../tts/text/normalize.py#L247) |
| Pronunciation (phone sequence) | `data/lexicon/lexicon.txt`: first CMUdict pronunciation; g2p_en for OOV words | the reference's pronunciation | [`lexicon.py#L102`](../tts/text/lexicon.py#L102) |
| Voice, accent, intonation | The model, trained on LJSpeech only | the reference: no accented audio is ever used for training | [`models.py#L274`](../tts/models/vits_mfa/models.py#L274) |
| Word start/end times, pauses, total length | The reference audio, aligned with MFA | the model's own durations | [`pipeline.py#L216`](../tts/infer/pipeline.py#L216) |
| Phone timing inside each word | The model's native LJ durations, fitted to the word's span (§8) | the reference's phone timing, which carries the L2 accent | [`pipeline.py#L112`](../tts/infer/pipeline.py#L112) |

**Why this matters:**
- **Each failure has one place to look.** A wrong word is a text or lexicon problem, a wrong voice is a model problem, and a mistimed word is an alignment or fitting problem.
- **The pronunciation never comes from the reference.** The reference's phone durations and its mispronunciations are part of its accent.
- **The pipeline copies word boundaries only.** Inside each word it redistributes time with LJSpeech's own elasticity (§8).
- **Evidence:** the transfer-variant test (D-72). Copying the reference's *phone* durations (`phone_copy`, the assignment's suggested approach) gave the worst WER: 10.6%, against 2.5% for `elastic`.

---

## 4. Data and the accent-disjoint split

| Dataset | Use |
| --- | --- |
| LJSpeech 1.1: 13,100 clips, one US female speaker, 22.05 kHz | Training. The original VITS filelists (12,500 / 100 / 500), so the pretrained checkpoint never saw val or test. |
| L2-ARCTIC v5: 24 speakers, 6 L1s × 4 (2 male, 2 female each) | References for dev and test |
| L2-ARCTIC suitcase corpus | Stress-test demo clips only, not in metrics |

**Split by accent, not by sentence** ([`configs/splits.yaml`](../configs/splits.yaml)):
- **Dev:** Hindi, Mandarin, Spanish. Used for tuning and for Ablation B's pace statistics.
- **Test:** Arabic, Korean, Vietnamese. Untouched until `freeze-v1`.

**Reasoning:** the claim to support is "works for accents the system was never tuned on". A sentence-level split would let the tuning see every accent.

**Evaluation set:** the utterances that have a manual annotation for *every* speaker of the split ([`splits.py#L159`](../tts/splits.py#L159)). That gives:
- dev: 100 IDs × 12 speakers = 1,200 references;
- test: 104 × 12 = 1,248. Only one gap: YBAA lacks `arctic_a0094` (D-19).

Dev and test share 99 sentence texts. That is inherent to L2-ARCTIC and acceptable, because the split is by speaker and accent (D-19).

**The test lock is enforced in code, not by convention** ([`guard.py#L39`](../tts/guard.py#L39), [`guard.py#L48`](../tts/guard.py#L48)):
- Any attempt to read a test speaker's audio raises `SplitLockedError` unless `git rev-parse refs/tags/freeze-v1` succeeds. So does `evaluate.py --split test`.
- The inference pipeline checks every reference path against the lock ([`pipeline.py#L40`](../tts/infer/pipeline.py#L40)).
- Before the freeze, `make data` read only file names and transcripts of test speakers. A unit test fills fake test WAVs with junk bytes, so any read would fail (D-18).

**LJ quality control:**
- MFA aligned all 13,100 files. 13,073 are kept (99.79%); all 27 drops have a speech phone longer than 0.5 s ([`textgrid.py#L195`](../tts/align/textgrid.py#L195), [`reports/align_lj.md`](../reports/align_lj.md)).
- The header of every LJ file was checked for 22,050 Hz / mono / PCM_16 (D-21).

---

## 5. Text front end: normalizer and lexicon

**One normalizer for everything.** It is [`tts/text/normalize.py`](../tts/text/normalize.py), and the same code processes:
- the training text;
- the MFA `.lab` files;
- the inference input.

A mismatch between those three would make the aligner and the model disagree on the words.

**Normalizer rule order** (D-28): currency → symbols → `unidecode` → dates → times → abbreviations → numbers.
- **Why currency comes first:** `unidecode` would turn `£` into "PS".
- **Conventions:** no "and" in numbers; bare 4-digit numbers in [1100, 2099] are years; dotted initialisms are spelled out.
- **Fails loudly:** any digit left after normalization raises `NormalizeError` instead of being dropped.

**Robustness audit** (D-74):
- Token sequences of all 13,100 LJ texts are unchanged under case changes, quotes, extra punctuation, curly apostrophes and extra whitespace.
- One bug was found and fixed: a backtick used as an apostrophe split words in 8.8% of LJ texts (test `test_punctuation_and_case_do_not_change_words`).

**Lexicon** ([`tts/text/lexicon.py`](../tts/text/lexicon.py)). Each word has exactly one pronunciation:
- the first CMUdict entry ([`lexicon.py#L102`](../tts/text/lexicon.py#L102));
- otherwise g2p_en's neural model. Its `predict()` is called directly, so g2p_en does not redo its own number expansion (D-25).

One deliberate deviation is the **possessive rule** ([`lexicon.py#L95`](../tts/text/lexicon.py#L95), D-26):
- g2p_en mangled possessives whose base word is in CMUdict (`companion's` → `K AH0 M P EY1 N IY0 AH0 N Z`).
- So `x's` gets CMUdict(x) plus /ɪz/, /s/ or /z/ by the last phone. This affects 104 words, and a config flag turns it off.

**Lexicon contents:**
- 14,912 words: 13,640 from CMUdict, 104 from the possessive rule and 1,168 from g2p_en. The full OOV list is in [`reports/lexicon.md`](../reports/lexicon.md).
- The build is deterministic (same sha256 on two runs).
- At inference, unseen words are G2P'd and appended to the same file ([`lexicon.py#L130`](../tts/text/lexicon.py#L130)). So "the lexicon is the only source of pronunciations" stays true.

**Symbols** ([`symbols.py#L15-L21`](../tts/text/symbols.py#L15-L21)):
- `pad`, `sil`, `sp` + 69 ARPAbet phones (15 vowels × 3 stresses + 24 consonants) = 72 symbols.
- This is **exactly** MFA's `english_us_arpa` phone set, asserted by `test_mfa_phone_set_equals_symbols` (D-24). Every model token therefore has an MFA duration.

### 5.1 Typed-text pre-pass, added after the freeze (D-94, E-35)

The corpus texts are clean read prose. An interviewer types anything. So a rule-based pre-pass ([`tts/text/input_rules.py`](../tts/text/input_rules.py), tables in `text.input_rules` of `configs/base.yaml`) runs in front of the frozen normalizer.

**What it handles:**
- **Money:** $, €, £, ₹/Rs/INR, ¥, with Indian digit grouping.
- **Times, dates and durations:** `3:30pm`, M/D dates against fractions, "the 5th of", h/m durations.
- **Numbers in context:** ratios and scores (`3-2`), multipliers (`1.5x`), versions, decades, °C/°F and units, Roman numerals in context.
- **Web text:** URLs and e-mails, phone numbers.
- **Abbreviations and casing:** bare titles (`Dr`, `St` before a capitalized word), weekdays, AD/BC, CamelCase splitting.
- **Spelling out:** unknown all-caps tokens (`GPU`), and unknown lowercase tokens of 3+ letters without a vowel (`bcd`, E-35).
- **Refusals:** non-Latin script, and text that leaves no words, each with a one-line reason.

**Fixed against the growing lexicon.** "Known word" means CMUdict plus a `keep_caps` list, not `lexicon.txt`. The lexicon grows at inference, so the rule's output must not depend on what earlier requests added.

**Checks:**
- **Stress set:** 121 typed-text cases in [`tests/data/text_cases.tsv`](../tests/data/text_cases.tsv), run by `make text-stress`. 119 pass; the other 2 are G2P guesses for unknown names. Before the pre-pass, 58 of 117 failed.
- **End-to-end rehearsal:** 12/12 cases, each an L2 reference plus typed text, including the refusals (`make rehearse`).
- **Regression:** `test_corpus_word_sequences_unchanged` asserts that every L2-ARCTIC transcript gives the same words as before. LJ has one intended change, the "Dr Paul" title bug fix. So the evaluation in §11 is unaffected.

**Known limits** (§15):
- one pronunciation per word (homographs such as "read" and "lead");
- US date order;
- the letter "a" is read as the article;
- the spelling rule misses letter strings with a vowel (`ijk`) and wrongly spells interjections that are not in CMUdict (`brrr`).

---

## 6. Forced alignment (MFA) and its error on accented speech

**Aligner:** MFA 3.4.3 with the pretrained `english_us_arpa` acoustic model and our single-pronunciation lexicon. The aligner is not trained.

**One aligner setup everywhere.** The same aligner, lexicon and parsing rules are used for LJSpeech (the training durations), the L2 references (the timing to copy) and the model outputs (the measured timing). So alignment conventions cancel in the comparisons.

**Wide beam** (`--beam 100 --retry_beam 400`):
- MFA drops unknown pass-through flags **silently** (E-2).
- So [`mfa.py#L87`](../tts/align/mfa.py#L87) asks MFA for its accepted parameter names and raises on any typo.
- MFA's per-job logs confirm the beam takes effect (D-31).

**TextGrid parsing rules** ([`textgrid.py#L74`](../tts/align/textgrid.py#L74)). The same code is used for training, references and outputs:
1. `""`, `sil`, `sp` (and `<eps>` from `align_one`, E-14) are silence. `spn` means a failed word.
2. Leading and trailing silence → `sil`. An internal silence of ≥ 3 frames (~35 ms) → `sp`. A shorter one is merged into the preceding phone ([`textgrid.py#L134-L137`](../tts/align/textgrid.py#L134-L137)).
3. Boundaries are rounded on **cumulative** times (`round(t·22050/256)`), and durations are their differences ([`textgrid.py#L70`](../tts/align/textgrid.py#L70)).
   - **Why:** rounding each duration separately would let errors accumulate along the utterance.
   - **Matching the spectrogram:** the target frame count comes from the training spectrogram function itself ([`audio.py#L61`](../tts/audio.py#L61)). The last token absorbs the difference, which was exactly +1 for all 13,100 files (D-35).
4. **Every speech phone gets ≥ 1 frame** ([`textgrid.py#L165`](../tts/align/textgrid.py#L165), D-33). MFA's 10 ms phones are shorter than one 11.6 ms frame, which left 382 phones at 0 frames. A 0-frame phone breaks the log-duration loss (log(0 + 1e-6) = −13.8).

**How accurate is MFA on accented speech?** ([`l2.py#L92`](../tts/align/l2.py#L92), [`reports/mfa_l2_dev.md`](../reports/mfa_l2_dev.md), plot 3).
- **The ground truth:** L2-ARCTIC ships hand-corrected word and phone tiers for about 150 utterances per speaker (`annotation/*.TextGrid`). The phone labels carry error tags.
- **Matching:** words are matched by label sequence. Phones are matched inside matched words by canonical label, with error tags and stress ignored and added phones excluded. Only interval times are used.
- MFA was compared with the manual annotations on all 1,800 annotated dev utterances:

| Accent | Word MAE | Word ≤ 20 ms | Phone MAE | Phone ≤ 20 ms |
| --- | --- | --- | --- | --- |
| Hindi | 18.2 ms | 76.8% | 12.3 ms | 89.0% |
| Mandarin | 20.2 ms | 74.2% | 14.0 ms | 86.7% |
| Spanish | 19.9 ms | 74.0% | 13.2 ms | 87.6% |
| **All** | **19.4 ms** | **75.0%** | **13.1 ms** | **87.7%** |

**How to read this:**
- The system's word-boundary error is measured by MFA on the output against the *manual* reference. Two MFA errors of about 19 ms each therefore enter the metric.
- So the main system's 23.9 ms on test is close to the measurement floor, not 24 ms of mistimed speech.
- **The "within 20 ms" figures were corrected once** (D-60). MFA and the annotations both sit on a 10 ms grid, and float rounding had excluded errors of exactly 20 ms. All tolerance checks now use a 1 µs epsilon ([`metrics.py#L28`](../tts/eval/metrics.py#L28)).

**Fallback aligner** ([`ctc.py#L21`](../tts/infer/ctc.py#L21)): torchaudio's `MMS_FA` model with a CTC Viterbi forced alignment written in numpy.
- **Why numpy:** conda-forge's torchaudio 2.8 lacks the alignment extension (E-15, D-56). The pinned model and the algorithm are the same.
- **Word ends:** CTC emissions are spiky, so gaps shorter than a pause are split at their midpoint ([`ctc.py#L69`](../tts/infer/ctc.py#L69), D-57).
- **Accuracy:** about 53 ms MAE, against 21 ms for MFA, so it is only a fallback.

---

## 7. Models

### 7.0 V0: the original VITS with MAS (immediate baseline, inference only; added after the freeze)

- **What it is:** the official pretrained LJSpeech VITS checkpoint, exactly as released, and not retrained. It is the checkpoint M1 is fine-tuned from (§7.2). So V0 is the system before the MAS → MFA swap, and V0 vs M1 measures what the swap did. It has MAS-learned alignment, the stochastic duration predictor (SDP), blank tokens and IPA character input ([`v0.py`](../tts/models/vits_mas/v0.py), [`configs/v0.yaml`](../configs/v0.yaml), D-90).
- **Same pipeline:** V0 runs through the same pipeline and settings (§8) as every other system. It gets the same word timing and the same exact output length.
- **Input text** ([`v0.py#L51-L89`](../tts/models/vits_mas/v0.py#L51-L89)):
  - The original model reads espeak IPA. The lexicon is the only allowed pronunciation source, so each ARPAbet phone is written in the IPA characters espeak en-us uses in the VITS training lists (for example `AH0` → ə, `ER1` → ˈɜː, `R` → ɹ).
  - The stress mark goes before the vowel, as in those lists.
  - Words are separated by spaces; `sp` becomes ", " and the final `sil` becomes ".".
- **Durations** ([`v0.py#L115`](../tts/models/vits_mas/v0.py#L115), [`v0.py#L216-L230`](../tts/models/vits_mas/v0.py#L216-L230)):
  - The SDP gives each character and blank a native duration (noise 0.8, VITS's own default, seeded).
  - A phone's native duration is the sum over the characters and blanks it owns.
  - At synthesis, each phone's frames are split back over its characters in the same proportions, with exact totals.
- **The one adapter choice** ([`v0.py#L92`](../tts/models/vits_mas/v0.py#L92)) is which phone a blank token belongs to. It was chosen on dev. When each blank was credited to the character *before* it, words came out about 35 ms early (dev MAE 50.8 ms). Crediting it to the character *after* it gave 40.7 ms, and that is the setting used (E-32).
- **Why have it:** the question "did replacing MAS help?" needs the model before the replacement, under the same conditions. FastSpeech2 (M0) answers a different question.

**How V0 is driven, step by step.** MAS runs only during training, to find the alignment the model learns from; at inference every VITS takes durations from its duration predictor. V0's predicted durations are replaced by durations built from the reference's **word** boundaries, not from MFA phone boundaries:
1. **Reference → word spans.** MFA aligns the L2 recording, and only word starts, ends and pauses are kept, as for every system. Each word gets a span of D frames.
2. **Text → V0's input.** Lexicon phones are written in espeak-style IPA characters, with a blank token between every character, as V0 was trained.
3. **V0's own rhythm inside a word.** V0's stochastic duration predictor gives a duration to every character and blank; these are summed per phone into native phone durations.
4. **The same fit as every system** (elastic, one z per word, cap 2.5×, §8) spreads D over the word's phones. The total comes from the reference; the split comes from V0.
5. **Back to V0's tokens.** Each phone's frames are split over its characters and blanks in V0's predicted proportions, with exact totals. So word boundaries and output length match the other systems frame for frame.
6. **Synthesis.** These per-token durations replace the predictor's output when the prior is expanded; then the inverse flow and the decoder run as usual.

**Space vs pause.** Both the space and the comma are tokens, but they mean different things:
- **The rule:** the token timeline has an `sp` token only where the reference has a silence of at least 3 frames (~35 ms) between two words (`align.min_sp_frames`, [`pipeline.py#L83`](../tts/infer/pipeline.py#L83)). A shorter gap goes to the preceding word. V0 writes `sp` as ", ", because in its LJSpeech training text a comma is where the speaker pauses. A plain space marks a word boundary with no pause.
- **Why words are usually not separated by silence:** in connected speech, words mostly run into each other. A stop's closure silence is part of the stop, not a gap between words. In the dev references, 78.9% of word junctions have no silence at all; the other 21.1% have one of at least 3 frames (12.2% at least 100 ms).
- **Two thresholds:** the 3-frame rule decides the tokens. The pause F1 metric counts only silences of at least 100 ms (§9.1).

**Where V0's 40.5 ms comes from.** The metric aligns V0's output with MFA and compares each word's start and end with the manual annotation of the reference. Two errors add up:
- **MFA's own error on the reference,** about 19–20 ms (§6). It is the same for every system, which is why even the main system is at 23.9 ms.
- **How faithfully the model renders the durations it is given.** This is where V0 fails: its words land 21 ms early on average, with 27.7 ms scatter around that shift. The main system lands about 0 ms off, with 11.2 ms scatter (§10.6).

**Why giving V0 the durations is not enough.** MAS learned, inside the model, which tokens carry which sound. A blank, for example, may carry the start of the next sound or the tail of the previous one. Imposed durations have to guess that mapping. Crediting each blank to the character before or after it moved the error from 52.0 to 40.5 ms, and the better choice still leaves a 21 ms early shift. The fine-tuned systems were trained on MFA durations, so for them "these frames are this phone" holds by construction. This tests V0 with the best adapter chosen on dev, not every possible adapter (§15).

### 7.1 M0: FastSpeech2 + HiFi-GAN baseline (inference only)

- **Model:** ming024's pretrained LJSpeech FastSpeech2 and its bundled HiFi-GAN, at the same frame rate as ours ([`tts/models/fastspeech2/m0.py`](../tts/models/fastspeech2/m0.py)).
- **Duration injection:** reference durations are passed as `d_targets` to the variance adaptor, the model's training-time code path ([`m0.py#L78-L85`](../tts/models/fastspeech2/m0.py#L78-L85)). Pitch and energy stay predicted.
- **Symbol mapping:** our tokens map onto its symbol table, kept in the original order. An unknown symbol raises ([`m0.py#L28`](../tts/models/fastspeech2/m0.py#L28)).
- **Why have it:** it is a strong, standard non-autoregressive baseline that accepts durations directly. It also gave a working demo before any training. It is the second baseline; V0 (§7.0) is the immediate one.

### 7.2 M1: VITS with MFA durations (VITS-MFA)

**Base:** jaywalnut310/vits and its official LJSpeech generator checkpoint. VITS was chosen because it is end-to-end: there is no separate vocoder, so no mel mismatch. It also has a high-quality pretrained LJ checkpoint to fine-tune from.

**The six PLAN changes** (header of [`models.py`](../tts/models/vits_mfa/models.py)):

| # | Change | Code | Reason |
| --- | --- | --- | --- |
| 1 | Remove MAS; the alignment is a one-hot expansion of MFA durations | [`models.py#L242`](../tts/models/vits_mfa/models.py#L242) | The durations must be **controllable** at inference. MAS learns its own alignment, which a reference could not override. Measured after the freeze: V0 renders given durations ~20 ms early and scattered (§10.6). |
| 2 | `add_blank: false` | [`data.py#L60-L73`](../tts/models/vits_mfa/data.py#L60-L73) | Blank tokens have no MFA duration. |
| 3 | 72 symbols; text embedding re-initialized, every other matching weight loaded | [`models.py#L348`](../tts/models/vits_mfa/models.py#L348) | The pretrained model used espeak IPA characters (178 symbols). |
| 4 | Deterministic duration predictor on the **detached** encoder output, MSE on log frames, masked on `sil`, padding and augmented rows | [`models.py#L32-L47`](../tts/models/vits_mfa/models.py#L32-L47), [`models.py#L265`](../tts/models/vits_mfa/models.py#L265), [`data.py#L71`](../tts/models/vits_mfa/data.py#L71) | Durations come from the reference, so the predictor only supplies *native relative* durations inside a word (§8). It needs to be deterministic, not stochastic. Detaching keeps it from distorting the encoder. |
| 5 | `use_pos`: expand the hidden states by duration, add a position-in-phone embedding, then project | [`models.py#L74-L83`](../tts/models/vits_mfa/models.py#L74-L83), [`models.py#L248`](../tts/models/vits_mfa/models.py#L248) | Lets the prior change inside a phone (§10.3). |
| 6 | Data loader reads the project manifests | [`data.py#L35`](../tts/models/vits_mfa/data.py#L35) | One manifest format for all data |

**Original VITS (V0) vs VITS-MFA at a glance:**

| | Original VITS (V0) | VITS-MFA (ours) |
| --- | --- | --- |
| Alignment in training | MAS, searched every step | Fixed one-hot expansion of MFA frame counts per phone |
| Input tokens | 178: espeak IPA characters + punctuation, a blank between every token | 72: 69 ARPAbet phones + `sil`, `sp`, `pad`, no blanks; 87 with the diphthong split |
| Duration model | Stochastic, flow-based ([`v0.py#L131`](../tts/models/vits_mas/v0.py#L131)), 1.32 M parameters, trained on MAS durations | Deterministic, two convolutions, 0.35 M parameters, MSE on log frames |
| Prior inside a phone | One constant mean and scale per token | + position-in-phone embedding (`m1_pos` onward) |
| Weights | — | Warm start: every pretrained tensor whose name and shape match (553) is loaded |
| Unchanged | Posterior encoder, flow, HiFi-GAN decoder, losses | Same, + L_dur |

**Why a deterministic duration predictor** (change 4, [`models.py#L32-L47`](../tts/models/vits_mfa/models.py#L32-L47), [`models.py#L265`](../tts/models/vits_mfa/models.py#L265)):
- **What the original does:** VITS's stochastic duration predictor is a normalizing flow conditioned on the text states. It learns the whole distribution of MAS durations through a variational lower bound. At inference it draws noise (scale 0.8), so the same sentence gets a different rhythm each time.
- **What the pipeline needs:** the reference fixes each word's total length. The model only has to say how LJ would share that length among the word's phones, which is one native value per phone, not a sample. Sampling would only add random rhythm inside a word, which the fit then squeezes back into the span.
- **The loss:** L_dur = mean over phones of (log d̂ᵢ − log dᵢ)², with dᵢ the MFA frame count. Error on the log scale is relative: being 2× off costs the same for a 3-frame stop and a 20-frame vowel. The minimizer is the typical (geometric-mean) duration, which is the natural centre for the multiplicative elastic fit (§8).
- **Masking and detaching:** the loss ignores `sil` (its length comes from the reference), padding and augmented clips. The input is detached, as in VITS, so the duration loss cannot distort the text encoder.

**Loading the pretrained checkpoint** (D-44). "Warm start" means the fine-tune starts from the pretrained LJSpeech VITS wherever the architecture still matches:
- 553 tensors loaded.
- Skipped: the 178-symbol embedding and the 284 stochastic-duration-predictor tensors.
- `test_pretrained_loads_only_expected_skips` asserts exactly this list, so an unexpected skip would fail the tests.
- The checkpoint is generator-only, so the discriminators start from scratch.

**Losses** ([`trainer.py#L111-L156`](../tts/models/vits_mfa/trainer.py#L111-L156)):

L_G = 45·L_mel + L_KL + L_dur + L_adv + 2·L_fm

- L_mel: L1 on mel spectrograms.
- L_KL: ties the posterior to the duration-expanded prior.
- L_adv: least-squares GAN loss. The factor 2 on L_fm is inside VITS's `feature_loss`.
- Precision: bf16 autocast for the networks; STFT, mel and every loss in fp32.

**Training (phase 1)** (D-70):
- 114,000 steps at batch 32 on one L40S, about 8 h.
- **Choosing max_steps:** a 300-step benchmark measured 0.229 s/step. Of the 8 h, 7.5 h went to training (0.5 h kept for export), minus about 3% for validation and checkpointing.
- **Fixes on the way:**
  - `cudnn.benchmark` was turned off: with variable-length batches it re-tunes for every shape and was 13× slower (E-10).
  - `train.py` refuses to train an ablation whose augmented manifest is empty (exit code 3, [`train.py#L109-L113`](../scripts/train.py#L109-L113)). Otherwise `abl_a` would have silently trained as a second M1 (E-19).

**Reading the training curves** (plot 10, D-70 notes):
- `val/dur` fell throughout (0.134 → 0.094).
- `val/mel_l1` reached its minimum at 12k (0.354) and ended at 0.377. This is the GAN trade-off, not overfitting:
  - feature-matching and adversarial losses rose while the discriminator loss stayed balanced (2.2–3.0);
  - train and val mel drifted together.
- The learning rate was left at VITS's schedule, so all systems keep identical hyperparameters.

**Batch-size test** (D-71, at the user's request): throughput peaked at batch 64 (+19% clips/s) and fell at 96. M1 stayed at batch 32. Changing it mid-run would mix two settings, and batch 64 makes 40% fewer updates per hour.

### 7.3 Phase-2 variants (all branch from the M1 checkpoint)

| System | Config | Difference from M1 |
| --- | --- | --- |
| m1_cont | [`m1.yaml`](../configs/m1.yaml) with `--tag m1_cont` | +20k steps with no change: the control for "more training" |
| m1_pos | [`m1_pos.yaml`](../configs/m1_pos.yaml) | `use_pos: true` |
| abl_a_pos | [`abl_a_pos.yaml`](../configs/abl_a_pos.yaml) | m1_pos + 40% Ablation A clips |
| abl_b_pos | [`abl_b_pos.yaml`](../configs/abl_b_pos.yaml) | m1_pos + 40% Ablation B clips |
| **m1_pos_diph** | [`m1_pos_diph.yaml`](../configs/m1_pos_diph.yaml) | m1_pos + `diph_split: true` (87 symbols) |

**Branching rules** ([`trainer.py#L58`](../tts/models/vits_mfa/trainer.py#L58)): generator and discriminator weights come from M1, with a fresh optimizer, step 0 and 20,000 steps each. So every phase-2 system shares one start point, one step count and one optimizer reset, and each pair differs in one setting.
- The tests assert that each config differs from its control in exactly one key: `test_abl_pos_configs_differ_only_in_use_pos` and `test_m1_pos_diph_config_differs_only_in_split`.
- The new parameters start so that each model **starts as M1**:
  - the position embedding is zero-initialized ([`models.py#L65`](../tts/models/vits_mfa/models.py#L65));
  - each new diphthong-second-half embedding is a copy of its base token's ([`diph.py#L65`](../tts/text/diph.py#L65)).
- **Check:** a 20-step smoke run from M1 gave val mel L1 0.381 (m1_pos) and 0.379 (m1_pos_diph), against M1's 0.377.

### 7.4 Theory: what a fixed external alignment takes away

VITS trains its prior through the KL term ([`losses.py#L46`](../tts/models/vits_mfa/losses.py#L46)), in the flow's output space:

L_KL = log q(z | x) − log N( f(z) ; μ_A(c), σ_A(c) )

Here μ and σ are the text-encoder prior, copied onto frames by the alignment A.
- **MAS (original VITS):** A* = argmax_A Σₜ log N( f(zₜ) ; μ_A(t), σ_A(t) ). The alignment is re-chosen every step, by the model's own prior and flow output. Alignment and latent space adapt to each other, in a hard-EM loop.
- **MFA (ours):** A is fixed before training by an HMM–GMM acoustic model on MFCCs. It never sees f(z), μ or σ.

**Not lost: the gradients.** MAS is not differentiable. So the KL still trains the flow and the prior encoder through exactly the same paths; only the value of A changed (D-75 audit). Replacing MAS does not remove the loss contribution through the flow or the prior encoder.

**Lost: the feedback from the latent space to the alignment.**
1. **Boundary mismatch is irreducible.** Frames that MFA assigns to the wrong phone stay in the KL for the whole of training, and the prior learns the average of the frames it is given. Its targets blur.
2. **No blanks.** With one token per phone, each phone has one constant prior over all its frames. In the original, transitions could sit on blank tokens; here they have nowhere to go, and the flow must create all movement inside a phone.
3. **Frozen errors and conventions.** MFA's errors are frozen into the training targets, and the duration predictor learns MFA's segmentation conventions, not the model's.

**Evidence** ([`results/phone_dynamics/`](../results/phone_dynamics), D-76): on 100 LJ val clips resynthesized with ground-truth MFA durations, every MFA-trained model moves 14–16% less inside phones than the LJ recordings: M1 0.84×, m1_cont 0.86×, m1_pos 0.86×, m1_pos_diph 0.86×. Their change is piled up at phone onsets: onset/inside ratio 1.61 against 1.34 for LJ. The audible case is the flattened diphthong ("cry" heard as "cra", §10.5). The split fixed that case; the position embedding nudged the average only slightly (§10.3). V0 is not on this probe: driven with phone durations, its boundaries move (§10.6), which would confound the measure.

**What it buys.** Controllable durations: the MFA-trained systems render given durations with ~1 ms bias and ~11 ms scatter, against V0's −21 ms and 27.7 ms (§10.6).

**The natural next step** is to get the feedback loop back without losing control: learn the alignment with MAS again, but only *inside* MFA's word spans, with the word boundaries fixed (§15).

---

## 8. Inference pipeline: transferring word timing

All systems use one pipeline, [`tts/infer/pipeline.py`](../tts/infer/pipeline.py), configured by [`configs/inference.yaml`](../configs/inference.yaml).

```
reference.wav ─► load (any format, mono, no trim) ─► MFA align_one (fallback: MMS_FA CTC)
                                                        │ word start/end times
text ─► normalize ─► lexicon phones ─► token timeline: sil · word phones · sp (gaps ≥ 3 frames) · sil
                                                        │
                           duration predictor ─► native log-durations d̂ᵢ
                                                        │
     for each word of D frames: dᵢ = exp(log d̂ᵢ + z·σᵢ), one z per word (bisection)
                 stretch cap: if D > 2.5·Σd̂ᵢ, fit 2.5·Σd̂ᵢ and put the rest into an sp
                                                        │
            cumulative rounding to exactly round(N_ref·22050/sr_ref/256) frames, ≥ 1 frame/phone
                                                        │
       VITS: encoder → expand → prior sample (noise 0.333) → inverse flow → decoder → pad/trim
                                                        ▼
                 out.wav (exact length) · timings.json · timing.png
```

| Step | Code | What it does, and why |
| --- | --- | --- |
| 1. Load the reference | [`pipeline.py#L40`](../tts/infer/pipeline.py#L40) | libsndfile, then ffmpeg for anything else. **No trimming**: the reference's leading and trailing silence is part of its timing. |
| 3. Align | [`pipeline.py#L216`](../tts/infer/pipeline.py#L216) | Falls back to CTC if MFA fails, returns `spn`, or the word sequence differs from the transcript |
| 4. Token timeline | [`pipeline.py#L83`](../tts/infer/pipeline.py#L83) | Same ≥ 3-frame `sp` rule as training; a shorter gap goes to the preceding word |
| 5–6. Elastic fit | [`pipeline.py#L112`](../tts/infer/pipeline.py#L112) | Explained below |
| 7. Stretch cap | [`pipeline.py#L150-L153`](../tts/infer/pipeline.py#L150-L153) | Explained below |
| 8. Quantize | [`pipeline.py#L176`](../tts/infer/pipeline.py#L176) | Explained below |
| 9. Synthesize | [`systems.py#L58`](../tts/infer/systems.py#L58), [`models.py#L313`](../tts/models/vits_mfa/models.py#L313) | Pad or trim to exactly round(N_ref·22050/sr_ref) samples ([`pipeline.py#L322`](../tts/infer/pipeline.py#L322)) |

**Step 5–6, the elastic fit.** The duration predictor gives a native duration d̂ᵢ for each phone. To fit a word of D frames, every phone is scaled by exp(z·σᵢ), with **one z per word**. σᵢ is that phone's standard deviation of log duration in LJ train ([`stats.py#L29`](../tts/stats.py#L29), [`stats/phone_logdur.json`](../stats/phone_logdur.json)).
- So phones that LJ naturally stretches a lot (vowels, high σ) absorb most of the change.
- Phones LJ keeps steady (stops, low σ) barely move.
- This is "native elasticity": the word fills the reference's slot the way the LJ speaker would say it slower or faster.

**The three fit variants** for a word of reference length D frames ([`pipeline.py#L154-L157`](../tts/infer/pipeline.py#L154-L157), [`pipeline.py#L299-L302`](../tts/infer/pipeline.py#L299-L302)):

| Variant | Rule | What it keeps |
| --- | --- | --- |
| `elastic` (used) | dᵢ = exp(log d̂ᵢ + z·σᵢ), one z per word so that Σ dᵢ = D | LJ's rhythm, stretched the way LJ itself varies |
| `proportional` | dᵢ = d̂ᵢ · D / Σ d̂ | LJ's proportions, every phone scaled by the same factor |
| `phone_copy` | dᵢ = the reference's own MFA phone durations | the L2 speaker's phone timing (and with it, accent) |

**Worked example** ("cry", from the LJ averages in [`stats/phone_logdur.json`](../stats/phone_logdur.json)):
- **Native durations:** K 8.7, R 5.4, AY1 12.6 frames, 26.7 in total. σ is 0.32, 0.51 and 0.40.
- **Reference span twice that** (D = 53.3 frames):
  - `proportional` gives 17.3 / 10.9 / 25.1 (all ×2.0);
  - `elastic` gives 15.2 / 13.1 / 25.1 (×1.75 / ×2.4 / ×2.0).
- So the stop K stretches least and the highly variable R most, as LJ would when speaking slowly.

"Keeping LJ's rhythm" means exactly this: the relative phone durations come from LJ, and only the word's total comes from the reference.

**Step 7, the stretch cap.** Beyond 2.5× its native length a word is not stretched further; the excess becomes a pause after it, and the next word still starts on time.
- **What it prevents:** L2 readers are slower than LJ (median ×1.12–1.16, plot 8) and hesitate, so some words last 3–4× what LJ would need. Stretching every phone that far gives drawled, held vowels far outside what the model saw in training. With the cap, a slow word sounds like a word followed by a pause.
- **How 2.5 was chosen** (dev tuning grid, elastic, noise 0.333, 120 dev references, [`results/tune/grid.json`](../results/tune/grid.json)):

| Cap | 1.4× | 1.6× | 1.8× | 2.0× | **2.5×** | none |
| --- | --- | --- | --- | --- | --- | --- |
| Word-boundary MAE (ms) | 26.1 | 23.8 | 23.0 | 22.9 | **21.9** | 21.5 |
| UTMOS | 4.200 | 4.183 | 4.194 | 4.189 | **4.215** | 4.207 |

  - A tight cap ends words early and costs timing.
  - 2.5× is 0.4 ms behind no cap and has the best UTMOS, which is what the pre-stated rule picks.
- **How often it acts:** on test it caps 148 words in 128 of 1,248 references (10%; `n_capped` in `results/m1_pos_diph/test/per_utt.csv`).

**Step 8, quantize.** Cumulative rounding keeps every word boundary within half a frame and makes the total **exactly** the reference length. That is why length error is 0 for every output. It is an invariant, asserted on every call, not a statistic.

**Transfer variants and tuning** (D-72, [`tune.py#L21`](../tts/eval/tune.py#L21), plots 5 and 7):
- 21 settings were run on a fixed 120-utterance dev subset.
- **Rule** (fixed before running): among `elastic` settings within 5 ms of the best word-boundary MAE, take the highest UTMOS.
- **Chosen:** `elastic`, cap 2.5, noise 0.333, MFA aligner. It gave 21.9 ms MAE (best 21.5 ms), UTMOS 4.21 and WER 2.5%.
- **What the grid showed:**
  - The noise scale drives WER: 0.333 gave 2.5–3.5%, against 4.6–6.7% at 0.667.
  - A looser cap lowers MAE at no UTMOS cost.
  - `phone_copy` had the worst WER (10.6%) and lower LJ similarity. This is evidence that the reference's phone timing carries accent, and it supports copying word boundaries only.
  - The CTC aligner gave 44.5 ms MAE and 12.7% WER.

**Diphthong split at the model boundary.** For `m1_pos_diph`, [`VitsSystem`](../tts/infer/systems.py#L24) splits each diphthong right before the model and merges the two predicted durations back afterwards ([`systems.py#L49-L66`](../tts/infer/systems.py#L49-L66)). The pipeline, the lexicon and MFA never see the extra symbols.

**Batch evaluation alignment** (D-53):
- `mfa align_one` costs about 14 s per call, mostly model loading.
- So evaluation pre-aligns all references and all outputs in one `mfa align --single_speaker` pass, with the same model, lexicon and beam, and no speaker adaptation (like `align_one`).
- The live demo still calls `align_one`.

---

## 9. Evaluation methodology

All metrics are computed by [`scripts/evaluate.py`](../scripts/evaluate.py). It caches each stage per utterance, so re-runs and tuning reuse work, and writes:
- `results/<system>/<split>/per_utt.csv`;
- `summary.json`, overall and per accent, with 95% bootstrap CIs;
- the plots in [`results/plots/`](../results/plots).

The evaluation models are pinned by version and hash, and loaded offline from local snapshots. None is substituted (§14).

### 9.1 Metrics, definitions and code

| Requirement | Metric | Definition | Code |
| --- | --- | --- | --- |
| Timing | **Length error** | Output samples − round(N_ref·22050/sr_ref); must be 0 | [`pipeline.py#L333-L334`](../tts/infer/pipeline.py#L333-L334) |
| Timing | **Word-boundary MAE, % within 20/50 ms** | Each word's start and end. Reference times come from the **manual** annotation word tier ([`evaluate.py#L53`](../scripts/evaluate.py#L53)). Output times come from MFA on the output with the input text ([`evaluate.py#L107-L125`](../scripts/evaluate.py#L107-L125)). Words are matched by sequence alignment. | [`metrics.py#L16`](../tts/eval/metrics.py#L16), [`metrics.py#L24`](../tts/eval/metrics.py#L24) |
| Timing | **Word-boundary MAE (CTC)** | Same, with MMS_FA times on the output: a second-aligner cross-check. CTC has its own +38 ms start bias on real speech, so it checks the direction of a shift, not the ranking of systems (§9.3). | [`evaluate.py#L188-L193`](../scripts/evaluate.py#L188-L193) |
| Timing | **Pause F1** | A pause is a silence ≥ 100 ms between two matched words. F1 of output vs reference pauses per word boundary; no pauses on either side counts as F1 = 1. | [`metrics.py#L38`](../tts/eval/metrics.py#L38) |
| Exact words | **WER, CER** | faster-whisper large-v3 transcript vs the input text, both through Whisper's `EnglishTextNormalizer`, scored by `jiwer`. Also run on the references and on LJ test ground truth. | [`asr.py#L45`](../tts/eval/asr.py#L45) |
| US accent | **US probability** | CommonAccent ECAPA accent ID. softmax(30·cosine) over the classes; the 30 is the training recipe's angular-margin scale (D-65). | [`scorers.py#L55-L70`](../tts/eval/scorers.py#L55-L70) |
| LJ voice | **Speaker similarity** | ECAPA cosine, output vs the mean embedding of 100 LJ test clips (should be high) and vs the reference speaker (should be low) | [`scorers.py#L39`](../tts/eval/scorers.py#L39), [`evaluate.py#L149-L168`](../scripts/evaluate.py#L149-L168) |
| Naturalness | **UTMOS** | SpeechMOS `utmos22_strong` on every output; LJ test ground truth is the ceiling | [`scorers.py#L14`](../tts/eval/scorers.py#L14) |
| Stretch | **Stretch ratio** | Speech frames ÷ the model's native predicted frames; used for binning | [`pipeline.py#L314-L316`](../tts/infer/pipeline.py#L314-L316) |

**WER normalization, in detail** ([`asr.py#L14-L22`](../tts/eval/asr.py#L14-L22), [`asr.py#L45`](../tts/eval/asr.py#L45)):
- Whisper's `EnglishTextNormalizer` runs on **both** the input text and the ASR transcript. It lower-cases, removes punctuation, expands contractions, maps spelled-out numbers and digits to one form, and maps British to US spelling.
- Example: "For the twentieth time, that evening!" and "For the 20th time that evening." both become "for the 20th time that evening".
- No dev or test evaluation sentence contains a digit or symbol (L2-ARCTIC writes numbers out). So number formats can only differ on the ASR side, where the normalizer handles them.
- Known gap: currency is not equalized ("$5.50" vs "five dollars fifty"). No evaluation sentence contains currency.

**Reading the references' 6.3% WER.** It is Whisper on the L2 input recordings themselves: human speech, no model involved. It measures how hard accented speech is for ASR, so it is a reference point, not a drawback of the system. The outputs (2.9%) are in a US voice and accent, so they are easier to recognize; the LJ recordings score 1.9%. Do not confuse it with output WERs such as phase 2c's 4.4% → 2.9%, which are Whisper on the synthesized speech of the same sentences.

**Probes added during the project:**

| Probe | What it measures | Code |
| --- | --- | --- |
| **Phone dynamics** (D-76) | Mean L2 distance between consecutive log-mel frames *inside* phones of ≥ 8 frames (overall and by thirds) vs at phone onsets, for LJ val resynthesized with ground-truth durations against the LJ recordings | [`phone_dynamics.py#L27-L49`](../tts/eval/phone_dynamics.py#L27-L49) |
| **Duration fidelity** (D-90, `make fidelity`) | For each word, the signed difference between where the aligner finds it in the output and where the pipeline placed it. Reported as bias (start, end), MAE, and MAE with the bias removed, for MFA and CTC. This separates the model's rendering error from the shared reference-alignment error. | [`fidelity.py#L31-L48`](../tts/eval/fidelity.py#L31-L48), [`fidelity.py#L62`](../tts/eval/fidelity.py#L62) |
| **Aligner calibration** (D-90, `make fidelity SPLIT=dev`) | The same statistics for MFA and CTC on the natural dev recordings against the manual annotation: each judge's own zero point | [`fidelity.py#L83`](../tts/eval/fidelity.py#L83) |
| **Diphthong glide** (D-80) | LPC formants (order 12, 11.025 kHz). F1/F2 change between 20–35% and 65–80% of each diphthong ≥ 8 frames. | ad-hoc probe script (session scratchpad, not a repo module); numbers in D-80 and D-84 |
| **Pace ρ** (plot 8, D-85) | Per word: reference frames ÷ Σ exp(μₚ) over its lexicon phones, with μₚ the LJ mean log duration; per utterance: the mean log ρ | [`plots.py#L225-L276`](../tts/eval/plots.py#L225-L276) |

### 9.2 Statistics

- **Confidence intervals:** 95% percentile bootstrap over utterances, 1,000 resamples, seed 1234 ([`metrics.py#L61`](../tts/eval/metrics.py#L61)).
- **Paired tests:** Wilcoxon signed-rank over utterances present in both systems ([`metrics.py#L72`](../tts/eval/metrics.py#L72)).
  - `evaluate.py` compares every system with M1 ([`evaluate.py#L270-L299`](../scripts/evaluate.py#L270-L299)).
  - [`paired.py#L42`](../tts/eval/paired.py#L42) compares each system with **its own control** (D-82). Groups: overall, per accent, and per tercile of |log stretch ratio|. The terciles test PLAN §6's hypothesis "a gain at high stretch, no loss near 1.0×".
  - Holm-adjusted within each group's family of 8 metrics ([`paired.py#L23`](../tts/eval/paired.py#L23)).
- **Why paired:** every system speaks the same 1,248 references, so per-utterance differences remove the large between-sentence variance.

### 9.3 Caveats about the metrics (stated, not hidden)

- **The accent-ID US probability is near-binary per utterance**, because of the softmax ×30. It flips on changes nobody can hear (D-84, §10.5), so read it as a population mean, never per sample.
- **UTMOS chose the WSOLA stretcher's parameters** (D-67), on LJ ground-truth clips only, while UTMOS is also an evaluation metric.
- **CTC is a weak second judge** (D-90). On natural dev speech it places word starts 38 ms late and has 47.8 ms MAE. MFA has no bias and 20.0 ms MAE ([`results/fidelity/calibration_dev.json`](../results/fidelity/calibration_dev.json)). A system that renders words *early* can therefore score a *lower* CTC MAE. V0 does: 45.6 ms against 49.1 for M1. CTC MAE is read only against its own zero point (§10.6).
- **MFA judges systems trained on MFA durations.** M1 and its branches learned MFA's segmentation of LJ, and MFA then measures them. That may favour them. CTC confirms V0's early shift independently, but it is too coarse to confirm the difference in scatter (§10.6).
- **M1's dev numbers are slightly optimistic,** because 120 of the 1,200 dev references were used for tuning (D-73). The test numbers have no such bias.

---

## 10. The project's journey: experiments and the reasoning behind each

The project ran in stages:
- **Laptop build (T1–T14):** environments, data, text, alignment, M0, the model code, training and smoke tests, augmentation, inference, evaluation, profiling, bundle, demo.
- **One GPU run:** M1.
- **Three phase-2 rounds:** each was triggered by a listening observation and decided by a pre-stated rule.
- **The freeze**, then the single test evaluation.
- **After the freeze:** V0, the immediate baseline, was added at review (§10.6).

### 10.1 Build and verification (T1–T14)

Each task ended with an acceptance check and a commit. Highlights:
- **T7 overfit check** (D-46, `make overfit`). 500 steps on one utterance took the duration loss from 0.65 to 0.013. Whisper transcribed the output at 5.6% WER (it missed only the initial "the"). This proves the MFA-duration path trains before spending GPU hours.
- **T8 smoke** (D-50): every config ran 200 steps at batch 4, and resume worked.
- **T10 edge suite** (D-58): 21 cases, all passing (§14).
- **T11 evaluation harness** (D-59): first run on M0.
- **T13 bundle** (D-62): a fresh unpack plus `make check-data` reproduces every file hash.

### 10.2 Phase 1: M1 vs M0 on dev (D-73)

The baseline result on 1,200 dev references (tuned setting, paired Wilcoxon):

| | M0 | M1 |
| --- | --- | --- |
| Word-boundary MAE | 31.3 ms | **23.7 ms** (p ≈ 1e-140) |
| Within 20 ms | 55.4% | **68.5%** |
| Pause F1 | 0.75 | **0.84** |
| WER | 10.0% | **3.5%** |
| UTMOS | 3.94 | **4.20** |
| US probability | 0.73 | 0.72 (n.s.) |

**Reasoning:**
- **Timing.** M1 is closer to the reference timing although both systems get the *same* durations. The likely reason: M1 renders its durations more crisply, which gives the aligner clearer boundaries to find. This was not tested separately.
- **US probability.** It did not improve, and that set up the next question.

### 10.3 Phase 2a: "words end sharply". Position-in-phone embedding (D-74 to D-78)

**Observation (user listening):** M1's words end abruptly, while LJ "extends" its word endings.

**Investigation.** Ad-hoc probes on 100 LJ val clips with ground-truth durations ruled out the obvious suspects:
- **Energy at word ends:** same fade as LJ.
- **Pitch over the last 12 frames:** same shape.
- **Duration predictor:** keeps LJ's pre-pausal lengthening (13.36 vs 13.07 frames).

What *did* differ was spectral movement **inside** phones: 14% below LJ, and 20% below in the middle third. M1 holds each phone steady, then jumps.

**Root-cause reasoning** (D-75 audit of the MAS → MFA swap):
- **The gradient paths are unchanged.** MAS is non-differentiable, so swapping it changes only the alignment's value.
- **The real difference is representational.** The original VITS put a blank token between every pair of phones, so its prior could change partway through a phone. Without blanks, every frame of a phone shares **one constant prior mean** (project, then expand: [`models.py#L80-L82`](../tts/models/vits_mfa/models.py#L80-L82)), and the flow must create all movement inside a phone.

**Experiment:**
- **Systems:** `m1_pos` (PLAN change 5: frames get a position-in-phone embedding *before* the projection, [`models.py#L76-L79`](../tts/models/vits_mfa/models.py#L76-L79)) against `m1_cont` (the same 20k extra steps without the change).
- **How the embedding works:** frame k of a phone of d frames gets bin ⌊16·(k + 0.5)/d⌋, from 0 at the phone's start to 15 at its end ([`models.py#L248`](../tts/models/vits_mfa/models.py#L248)). A learned 192-dim vector per bin is added to the phone's encoder state before it is projected to the prior mean and scale. So the prior can differ at the start, middle and end of a phone. The table is zero-initialized, so at step 0 the model is exactly M1.
- **Rule, fixed before running:** m1_pos wins if within-phone movement moves toward LJ without losing MAE, WER or UTMOS.

**Result:**
- **Within-phone movement** did not move: 0.856× LJ vs 0.858×.
- **US probability** rose by a large, consistent margin: 0.769 vs 0.680 (p = 5e-31). It was higher in all three accents and on 66% of utterances.
- **Everything else** showed no significant difference.
- **Why the two disagree:** the embedding's learned effect sits at phone *edges* (bins 0, 1, 15), which the within-phone metric excludes by design.

**Decision:** m1_pos became the base for the ablations. It was picked on the metric that matches the actual complaint, together with the user's listening.

### 10.4 Phase 2b: pace augmentation, Ablations A and B (D-47 to D-49, D-67, D-79, D-83)

**Hypothesis (PLAN §6):** training on time-stretched LJ copies teaches the model to stay natural and intelligible at high stretch ratios, with no loss near 1.0×.

**Ablation A (target-agnostic)** ([`generate.py#L32`](../tts/augment/generate.py#L32)):
- **Pace:** per copy, an utterance pace z_u ~ U(−1.5, 1.5); per word, z_w = z_u + N(0, 0.5²).
- **Phone factors:** rᵢ = clip(exp(z_w·σᵢ), 0.6, 2.5) ([`core.py#L43`](../tts/augment/core.py#L43)). These are the same elastic factors as inference, so the augmentation looks like what the model will be asked to do.
- **Pauses:** existing pauses are lengthened 1–3×. Pauses are also inserted after punctuation with probability 0.1, built from looped **room tone** ([`core.py#L91`](../tts/augment/core.py#L91)), never digital zeros.

**Ablation B (target-aware)** ([`generate.py#L57`](../tts/augment/generate.py#L57)). Word paces and pauses are sampled from **dev-accent** statistics ([`pace_b.py#L34-L81`](../tts/augment/pace_b.py#L34-L81), [`stats/l2_dev_pace.json`](../stats/l2_dev_pace.json)):
- an empirical utterance mean log ρ (dev speakers are about 17% slower than LJ's means);
- AR(1) word residuals (lag-1 −0.105);
- P(pause after a word) = 0.223, with lengths from the empirical histogram.

**Rules for B:**
- Only word-level ratios are used, never L2 phone timing inside words (that is accent).
- Test accents are never read.
- **Evaluated on held-out accents only:** a gain would mean that pace knowledge transfers across accents.

**Both sets:**
- 2 copies per LJ train utterance, mixed 60/40 into batches ([`data.py#L100`](../tts/models/vits_mfa/data.py#L100)).
- The duration loss is masked on augmented rows, so the predictor keeps learning native durations.
- The 5% of clips with the largest UTMOS drop are filtered out ([`filter.py#L24`](../tts/augment/filter.py#L24)).

**A deviation forced by measurement: Rubber Band R3 → WSOLA** (E-22, D-67).
- **The problem:** the planned Rubber Band R3 stretcher degraded LJ even at mild ratios. Mean UTMOS fell from 4.38 to 2.92, about 1.2 even for clips that changed length by less than 5%.
- **The cause:** an identity time map passed through unchanged, so the degradation was the phase vocoder itself.
- **Why filtering could not fix it:** every clip was affected, so the 5% filter could not remove it.
- **The fix:** we implemented a time-varying WSOLA stretcher in numpy ([`wsola.py#L21`](../tts/augment/wsola.py#L21)) that takes the same time map. It is pitch-preserving, with normalized cross-correlation.
- **Result:** the mean UTMOS drop fell from 1.336 to 0.321 on held-out LJ clips, better in every stretch bin.
- **Verification:** MFA re-alignment of 100 clips per set matched the planned boundaries to a median of about 7 ms (`tts/augment/verify.py`).

**Result (dev, paired against m1_pos, Holm-adjusted):**

| Metric | m1_pos | abl_a_pos | abl_b_pos |
| --- | --- | --- | --- |
| Word-boundary MAE | 24.16 ms | 24.04 (n.s.) | 24.21 (n.s.) |
| WER | 3.9% | 3.4% (n.s.) | 3.9% (n.s.) |
| US probability | 0.769 | **0.642** (p < 1e-10) | 0.747 (n.s.) |
| UTMOS | 4.18 | **4.07** (p < 1e-10) | **4.10** (p < 1e-5) |

**Reading:**
- **The hypothesis is not supported.** Even in the furthest stretch tercile, MAE, pause F1 and WER differences are not significant, and the UTMOS cost is the same in every tercile.
- **B fails on its most favourable split.** Its pace statistics come from the dev accents, yet it shows no timing gain there.
- **Likely cause:** 40% of the training batches carry residual stretch artifacts, which the model learns.
- **Why A loses more US probability:** its U(−1.5, 1.5) pace range is far wider than real L2 pace.
- **Decision:** both are reported as **negative results** and still evaluated on test. The PLAN's claim is about held-out accents.

### 10.5 Phase 2c: "cry" heard as "cra". The diphthong split (D-80, D-81, D-84)

**Observation (user listening):** "cry" sounded like "cra".

**Rule out the input first:** the lexicon gives `cry → K R AY1`, and the token sequence matches. The error was the model's.

**Measure it.** An LPC formant probe compared F1/F2 early vs late in each diphthong.

On the "cry" AY1 (reference `TNI_arctic_a0137`):
- **The reference glides to /ɪ/:** F2 1641 → 1926 Hz (+285), F1 737 → 554.
- **M1, m1_cont and m1_pos do not:** F2 −59 / −228 / +55 Hz, and F1 never falls. A held /a/.

Across LJ val (median F2 change), every model flattens diphthongs, AW worst:

| Diphthong | LJ | M1 | m1_pos |
| --- | --- | --- | --- |
| AY | +90 Hz | −8 | +139 |
| AW | −316 Hz | +17 | −49 |
| Mean \|ΔF2\| | 396 Hz | 307 | 321 |

**Root cause: token design.**
- The pretrained VITS read espeak IPA one character at a time, with blanks: "cry" = `k _ ɹ _ ˈ _ a _ ɪ`. A diphthong was **two targets and a transition token**.
- Our ARPAbet input makes AY **one token with one constant prior** over all its frames.
- So the MAS → MFA swap silently changed the *input representation* the pretrained model relied on.

**Fix** ([`tts/text/diph.py`](../tts/text/diph.py)):
- **Split:** each of AW, AY, EY, OW, OY (× 3 stresses) becomes `X` + `X+` (72 → 87 symbols) ([`diph.py#L33`](../tts/text/diph.py#L33)).
- **Durations:** MFA gives no boundary inside a diphthong, so its duration is split 50/50, each part ≥ 1 frame when d ≥ 2. A 0-frame second part (d = 1) is masked out of the duration loss ([`data.py#L73`](../tts/models/vits_mfa/data.py#L73)).
- **Duration prediction:** the two parts are merged back as log(exp a + exp b) ([`diph.py#L57`](../tts/text/diph.py#L57)). Timing control is unchanged.
- **Initialization:** each `X+` embedding copies `X` ([`diph.py#L65`](../tts/text/diph.py#L65), [`trainer.py#L64-L70`](../tts/models/vits_mfa/trainer.py#L64-L70)), so the model starts as M1. `X+` is a real token, not a dummy copy: after the copy it trains to its own target, the end of the glide (for AY, toward /ɪ/). Example: "cry" `K R AY1` → `K R AY1 AY1+`.
- **Scope:** the split lives **only at the model boundary**, in the VITS dataset and `VitsSystem`. Lexicon, MFA, manifests, statistics and the pipeline keep 72 symbols. The change is opt-in (`model.diph_split`) and leaves every other system untouched.
- **Tests:** 6 in [`tests/test_diph.py`](../tests/test_diph.py) (split and merge arithmetic, ids, embedding copy, one-key config difference, a branch plus train step, exact output length).

**Success criteria, fixed before training** (D-81), compared with m1_pos:
- the glide moves toward LJ;
- US probability does not drop;
- MAE, WER and UTMOS are not worse;
- listening agrees.

**Result** (dev, paired against m1_pos):

| Metric | m1_pos | m1_pos_diph | p (Holm) |
| --- | --- | --- | --- |
| Word-boundary MAE | 24.16 ms | **23.48** | < 1e-6 |
| WER | 3.9% | **2.7%** | < 1e-6 |
| CER | 2.0% | **1.2%** | < 1e-6 |
| UTMOS | 4.18 | **4.20** | 2e-4 |
| US probability | **0.769** | 0.738 | 1e-4 |

- **Glides:**
  - mean |ΔF2| is 389 Hz (LJ 396);
  - "cry" now glides: F1 864 → 528 Hz, F2 +215 Hz, close to the reference's F1 737 → 554, F2 +285.
- **MAE and WER:** the gains hold in every accent and every stretch tercile, largest at high stretch.

**The one unmet criterion, US probability (−0.03), was investigated rather than ignored:**
- The drop does not correlate with how many diphthongs a sentence has (Spearman −0.03, p 0.30). Sentences with **no** diphthongs dropped too (−0.037).
- The classifier flips both ways: 91 large drops against 58 large rises. The mean drop comes from a few 0.98 → 0.01 flips.
- The user listened to the four largest drops and heard **no difference**.
- **Conclusion:** classifier flips, not an accent change.

**Decision (D-85):** m1_pos_diph is the main system. Code, configs and checkpoint hashes ([`results/checkpoints.sha256`](../results/checkpoints.sha256)) were frozen at `freeze-v1`. Only then was the test split read.

**The method:**
1. listen;
2. check the input;
3. measure with a targeted probe;
4. find the design cause;
5. make a minimal, opt-in fix with a fair control;
6. re-measure with the same probe and the full metric set.

---

### 10.6 After the freeze: the immediate baseline V0 (D-90)

**The question raised at review:** M1 replaces MAS with MFA durations, so its immediate baseline is the original VITS with MAS, not FastSpeech2. V0 (§7.0) is that checkpoint, given the same word timing through the same pipeline. It is inference only, so it could be added after the freeze without touching any frozen system.

**Protocol:**
- V0's one free choice (blank ownership) was made on dev.
- Test was then run once with that choice.
- The first chained run had already evaluated test with the rejected choice. Both test results are kept: 52.0 ms rejected, 40.5 ms chosen (E-32). The choice changes no conclusion.

**Result** (test; paired vs V0, Holm; [`results/paired/test_*_vs_v0.csv`](../results/paired)):

| Main system − V0 | Mean difference | p (Holm) |
| --- | --- | --- |
| Word-boundary MAE | −16.5 ms (40.5 → 23.9) | 5e-192 |
| Within 20 ms | +28.4 pt | 1e-185 |
| US probability | +0.113 | 8e-20 |
| WER | +0.8 pt (2.1% → 2.9%) | 0.004 |
| UTMOS | −0.12 (4.31 → 4.20) | 3e-28 |
| Pause F1, speaker similarity | n.s. | |

- **Consistent across conditions:** the timing gain holds in all three accents (−15.9 / −17.4 / −16.2 ms). It grows with the stretch ratio (−15.9, −16.4, −17.4 ms by tercile). Dev shows the same pattern (−17.2 ms).
- **M1 vs V0:** the same picture. MAE −16.3 ms; WER +1.7 pt; UTMOS −0.12; US probability +0.086.

**Why V0 misses the timing: duration fidelity** (test, MFA on the output vs where the pipeline put each word; [`results/fidelity/test.json`](../results/fidelity/test.json)):

| System | Bias, starts / ends | MAE | MAE without the bias |
| --- | --- | --- | --- |
| V0 | −21.3 / −15.1 ms | 32.6 ms | 27.7 ms |
| M0 (FastSpeech2) | +0.4 / −0.5 ms | 19.9 ms | 19.9 ms |
| M1 | −1.3 / −0.4 ms | 11.5 ms | 11.4 ms |
| main | −1.0 / 0.0 ms | 11.3 ms | 11.2 ms |

- **V0's words are both early and scattered.** The MAS-trained model's internal alignment does not put a word's sound where its given frames are. Its blanks have no defined phonetic owner, and no adapter choice removes that: the rejected choice was worse, at −36 ms.
- **The fine-tuned systems render their durations almost exactly where they were placed.** This is direct evidence for PLAN change 1: with MFA durations, the reference timing becomes controllable.

**Is the judge fair?** (aligner calibration on natural dev speech, [`results/fidelity/calibration_dev.json`](../results/fidelity/calibration_dev.json))
- MFA: bias 0.0 / 0.0 ms, MAE 20.0 ms.
- CTC: bias +38.2 / −11.9 ms, MAE 47.8 ms.
- Against CTC's own zero point, CTC also finds V0 early: starts +5.3 against +38.2 for natural speech, ends −25.7 against −11.9. It finds M1 much closer: +26.4 / −14.6.
- So two independent judges agree on the direction and rough size of V0's shift.
- CTC's own scatter (40 ms) is too large to rank the systems' scatter. That part of the evidence rests on MFA, which M1 was trained on (§9.3).

**The trade-off, stated plainly:** V0 is the stronger system on WER and UTMOS. Fine-tuning on MFA durations without blanks bought:
- controllable timing;
- a more US-like accent (+0.11 US probability, a population mean; §9.3).

It cost about 0.8 points of WER and 0.12 UTMOS. The diphthong split (§10.5) already won back half of M1's WER gap to V0: M1 is +1.7 points, the main system +0.8.

---

## 11. Final results on held-out accents

The table is in §2. Paired tests on test (D-86, [`results/paired/`](../results/paired)):

- **m1_pos_diph vs V0 (the immediate baseline; D-90, §10.6):** MAE −16.5 ms; within 20 ms +28.4 pt; US prob +0.113; WER +0.8 pt; UTMOS −0.12. All significant after Holm. Pause F1 and speaker similarity are not significant.

- **m1_pos_diph vs m1_pos (its control):**
  - MAE −0.70 ms; WER −1.5 pt; CER −0.7 pt; UTMOS +0.03; within 20 ms +1.2 pt; pause F1 +0.009 (p 0.04). All significant after Holm.
  - US prob −0.019 (p 0.02). This is the same pattern as dev, read the same way.
  - The MAE and WER gains hold in all three accents.
- **m1_pos_diph vs M1:** WER −0.9 pt (p < 1e-6). MAE, pause F1, UTMOS and US prob (+0.027) are not significant.
- **Ablations vs m1_pos:**
  - no timing or WER gain;
  - UTMOS −0.11 (A) and −0.09 (B);
  - US prob −0.12 (A).
  - Ablation B's dev pace statistics did **not** turn into a gain on unseen accents.

**Per accent, test** ([`results/*/test/summary.json`](../results); plot 6):

| Accent | Reference US prob | V0 MAE / WER | M1 MAE / WER | m1_pos_diph MAE / WER | m1_pos_diph US prob |
| --- | --- | --- | --- | --- | --- |
| Arabic | 0.455 | 40.5 ms / 2.6% | 24.9 ms / 4.3% | 24.6 ms / 3.4% | 0.721 |
| Korean | 0.595 | 39.3 ms / 1.5% | 22.2 ms / 3.3% | 21.8 ms / 2.2% | 0.793 |
| Vietnamese | 0.406 | 41.6 ms / 2.2% | 25.4 ms / 4.0% | 25.4 ms / 3.2% | 0.771 |

**Distribution and stretch** (from `results/<system>/test/per_utt.csv`):

| | V0 | M0 | M1 | m1_pos_diph |
| --- | --- | --- | --- | --- |
| Median per-reference MAE | 38.0 ms | 29.4 ms | 21.9 ms | **21.5 ms** |
| 90th percentile (linear interpolation) | 56.8 ms | 45.6 ms | 36.3 ms | **35.5 ms** |
| Word boundaries within 20 ms / 50 ms | 41.2% / 75.8% | 55.3% / 84.3% | 69.3% / 90.3% | **69.6% / 90.4%** |
| MAE at stretch < 0.8 / 0.9–1.1 / > 1.2 | 35.2 / 39.8–40.1 / 42.4 | 30.7 / 30.8–30.9 / 32.7 | 26.0 / 23.0–23.3 / 26.1 | 24.7 / 22.8 / 26.1 |

- **Where the mass is:** 76% of the main system's references fall between 10 and 30 ms. V0's mode is 30–40 ms.
- **Stretch:** the stretch ratio is the main system's output speech frames over its native frames, available for 1,132 references. Every system's error rises when the reference is far slower or faster than LJ. The gap to V0 holds in every bin.
- **Worst cases:** the worst reference per accent is shown in `samples/demo/` (e.g. THV_arctic_a0067: 148 ms for the main system and for V0 alike), so failures are not hidden.

**Pace check (plot 8,** [`08_rho_dev_test.png`](../results/plots/08_rho_dev_test.png)**):**
- The median reference is 1.12× LJ's pace on dev and 1.16× on test, with a similar spread.
- So dev was a fair stand-in for test.

**Other test facts:**
- **Aligner:** all 1,248 test references were aligned by MFA; no CTC fallback was needed.
- **Second-aligner cross-check:** CTC on the outputs gives MAE 49.5 ms for the main system, 52.0 ms for M0, 49.1–49.5 ms for the VITS-MFA systems and 45.6 ms for V0. CTC's own +38 ms start bias on real speech makes these numbers unreliable as a ranking. V0's *early* shift cancels part of that bias, which is why V0 scores lowest. Read against CTC's zero point, CTC agrees with MFA (§10.6, §9.3). (An earlier version of this report read the CTC MAE as separating M0 from the VITS systems; that reading is withdrawn, D-90.)

---

## 12. Efficiency

Main checkpoint, from [`scripts/profile.py`](../scripts/profile.py) → [`results/profile_*.json`](../results):

| Quantity | Value |
| --- | --- |
| Inference parameters | 28.1 M: decoder 14.3 M, flow 7.1 M, text encoder 6.3 M, duration predictor 0.35 M |
| Training-only parameters | 54.0 M: discriminators 46.7 M, posterior encoder 7.2 M |
| Compute | 27.2 GMACs per second of audio; 99.7% of it in flow + decoder |
| Hand-checked layer | `dec.conv_pre`: 7 × 192 × 512 × 86 = 59.2 MMACs per second of audio |
| RTX 4060 laptop latency (p50, 3.7 s output) | text encoder + durations 5.4 ms; flow + decoder 39.4 ms; `mfa align_one` 3.3 s |
| RTX 4060 RTF | model-only 0.012; end to end 0.90 (MFA dominates) |
| CPU, 12 threads / 1 thread | flow + decoder 0.78 s / 1.78 s (model RTF 0.21 / 0.48) |

- **MAC counting:** with ptflops' module-hook backend the count came out as about 0 MACs, so the `aten` backend is used (E-21).
- **Latency method:** 10 warm-up + 50 timed runs, CUDA-synchronized. MFA was timed with 1 + 5 runs, because each call costs seconds.
- **Data-center GPU:** there is no L40S latency row. Profiling could not run on the instance (E-26), and the training benchmark (0.229 s/step) is the only L40S number.

---

## 13. Deviations from the plan

| PLAN said | Done instead | Why | Ref |
| --- | --- | --- | --- |
| Rubber Band R3 time-stretching | Time-varying WSOLA (numpy) with the same time map | R3 cost about 1.2–1.5 UTMOS on *every* clip; WSOLA 0.32 | E-22, D-67 |
| Augmented audio as WAV | FLAC (lossless) | Halves the GPU bundle; identical content | D-48 |
| `torchaudio.functional.forced_align` | numpy CTC Viterbi, same MMS_FA model | conda-forge torchaudio 2.8 lacks the extension | E-15, D-56 |
| "OOV from g2p_en" | + possessive rule for `x's` with x in CMUdict | g2p_en mangled 104 possessives | D-26 |
| Three GPU instances | One L40S: M1 first, then 20k-step branches | Availability | D-68, D-76 |
| Ablations trained from the pretrained model | Branched from M1 late in fine-tuning | One-GPU branching design (PLAN §10 step 5); stated in the write-up | D-68 |
| Data-center GPU profile | Laptop GPU and CPU only | Not runnable on the instance | E-26 |
| n_vocab = 72 | 87 for `m1_pos_diph` only (opt-in) | Diagnosed diphthong flattening | D-80, D-81 |
| UTMOS via `torch.hub` defaults | sha256-pinned weights loaded with `weights_only=True` | The hub path did a full pickle load of unhashed weights | E-13, D-66 |
| Baselines: M0 (FastSpeech2) only | + V0, the original pretrained VITS with MAS, inference only, added after the freeze | The immediate baseline for the MAS → MFA swap (raised at review). No frozen system or config changed. | D-90 |
| (V0) the original VITS reads espeak IPA | Lexicon pronunciations written in espeak-style IPA | `lexicon.txt` is the only pronunciation source (CLAUDE.md) | D-90 |
| Bundle includes LJ WAVs | M1 bundle without them; the instance downloads LJ, verified against the laptop's hashes | 1 MB/s upload | D-69 |
| Normalizer as specified | + a typed-text pre-pass in front of it, after the freeze | Interviewer input (money, URLs, letters…); corpus words unchanged, checked by a test | D-94, E-35 |
| `mfa align_one` with a fixed temporary directory | Temporary directory keyed by the lexicon's hash | MFA reused a stale parsed dictionary, so words added at inference were unknown (CTC fallback) | E-35 |
| Demo page as the audio deliverable | `samples/`: byte-exact copies of the evaluated outputs | The re-synthesized demo audio differs from the evaluated files (sampling noise) | E-36, D-103 |

---

## 14. Engineering for integrity and robustness

**Integrity:**
- **Test lock in code.** `tts/guard.py` blocks test-speaker audio until `freeze-v1` exists (§4).
- **Freeze discipline.** After the tag, no model code or experiment config changed. Checkpoint hashes are committed. V0 and the fidelity probe (D-90) were added after the tag as new files, with new configs. The only shared code change is the pipeline passing word positions to systems that ask for them, which only V0 does. Every other system gets the identical call.
- **Decision rules before results.** The tuning rule (D-64), the phase-2a winner rule (D-76) and the diphthong success criteria (D-81) were written down before each run.
- **Nothing silently substituted:**
  - pinned environments (`envs/*.yml`, plus full lock exports in `envs/lock/`);
  - pinned model hashes (MFA models, VITS, FastSpeech2, HiFi-GAN, UTMOS);
  - every model loads offline from local snapshots.
  - When a pinned item failed, work stopped and the issue was reported (E-13, E-18).
- **Safe loading.** Every checkpoint loads with `weights_only=True`. The 2021 FastSpeech2 pickles get a minimal numpy allow-list (D-43).

**Robustness:**
- **Edge-case suite** ([`tests/edge_cases/`](../tests/edge_cases), [`edge_cases.py`](../tts/infer/edge_cases.py), D-58). 21 cases:
  - text normalization (digits, currency, dates, abbreviations, an OOV name, no punctuation);
  - audio formats (MP3, 8 kHz, 48 kHz, stereo, clipped, noisy, reverberant);
  - timing (> 20 s, a 1.8 s pause, 1.6× tempo, a one-word cut);
  - alignment failures (an unspoken word, a false start, a forced CTC path).
  - Pass means exact length or a clean fallback. **21/21 on GPU and CPU** with the main checkpoint.
- **Failure handling.** Any failure in `scripts/infer.py --pairs_dir` prints a one-line reason and continues with the next pair.
- **Unit tests.** `make test` gives 313 passed, 1 skipped on the author's machine. That is 158 at the freeze; V0 and the fidelity probe added 8, and the post-freeze tooling of §14.1 and the PDF report builder added the rest. On a clean clone without the datasets it gives 305 passed, 9 skipped. The skip on the full machine is the pre-freeze lock test, which skips by design once the tag exists. Every module has a test and a make target.
- **Backup outputs for the live demo.** The demo page ([`scripts/make_demo.py`](../scripts/make_demo.py)) stores every sample's outputs and timing plots. It was rebuilt from an empty folder after the freeze (D-88). Its audio is re-synthesized, so it differs from the evaluated files by sampling noise. The evidence is therefore `samples/` (§14.1), not the demo page.

### 14.1 After the freeze: hosting, demos and evidence (D-95..D-105)

None of this changes a model or an experiment config.

**Hosting** ([`scripts/fetch_models.py`](../scripts/fetch_models.py), D-95/D-97/D-100):
- **What:** a public Hugging Face repository, `Raviteja546/meeami-tts`. It holds inference-only checkpoints (`infer.pt`: generator weights only; 141 MB instead of 956 MB; bit-identical output, `scripts/export_infer.py --verify`), the two baselines' weights and the frozen lexicon.
- **How it is fetched:** `make models` downloads them at a pinned revision and checks every file against `MANIFEST.sha256`.

**Live demo** ([`scripts/demo.py`](../scripts/demo.py), `configs/live_demo.yaml`, D-96):
- main vs V0 on a held-out L2-ARCTIC clip, with word times measured on the outputs;
- `make rehearse` runs 12 end-to-end cases.

**Browser demo** ([`scripts/web_demo.py`](../scripts/web_demo.py), D-101/D-102, E-37):
- **What a visitor does:** records or uploads a sentence, types it, and gets all seven systems side by side.
- **Server:** a stdlib server bound to localhost with an access key, and one job at a time.
- **Sharing:** a Cloudflare quick tunnel gives https, which browsers require for the microphone.
- **What testing it found:** recording real users exposed the failures in §15.

**Exact evidence** (`samples/`, [`scripts/export_samples.py`](../scripts/export_samples.py), E-36/D-103):
- **demo/:** 21 test samples in 3 accents. Each has the reference, the human annotation, all 7 outputs, word timings and a plot.
- **Selection:** fixed before listening. The headline prompt for every speaker, plus the best, median and worst reference per accent.
- **Self-check:** the export recomputes every sample's MAE from the committed files and stops unless it equals `per_utt.csv`; 147 of 147 matched.
- **listening/:** the listening sets behind phase-2 decisions, byte-identical to the dev evaluation outputs.
- **browser_tests/:** all 50 browser tests by the author.
- **On the Hub:** all 1,248 test outputs per system (`make outputs`, a pinned revision, a sha256 manifest).

**Outsider check** (D-100, D-104):
- **Setup:** a fresh clone, no login, an empty download cache.
- **Result:** `make models`, `assets`, `infer`, `rehearse`, `text-stress`, `test` and `outputs` all pass. The README's quick start is that sequence.

---

## 15. Limitations and future work

**Limitations:**
- **Accent ID is a weak judge.** It is near-binary per utterance, so small US-probability differences need listening tests to interpret. A MUSHRA-style listening test would be the right next measurement.
- **Part of the accent lives in timing.** Forcing the LJ voice onto an L2 reference's word timing caps how American it can sound. Dev US probability: references 0.30, outputs 0.74.
- **The 50/50 diphthong split is fixed.** MFA gives no internal boundary, and short unstressed diphthongs get only 1–2 frames per half.
- **No word-boundary token.** The encoder input is a bare phone stream ("a name" = "an aim"), so word-final phones are distinguishable only through context and `sp` (D-75c).
- **End-to-end latency is MFA-bound.** It takes 3.3 s per reference on the laptop, while the model itself runs 80× faster than real time.
- **V0 needs an adapter.** It reads lexicon pronunciations in espeak-style IPA instead of espeak's own output (function words keep full stress), and it needs a blank-ownership rule. Even with the better rule, a −21 ms bias remains. An adapter that learned to place V0's blanks could narrow the timing gap. It would still have to fight MAS's model-internal alignment.
- **Fixed alignments flatten phones** (§7.4): every MFA-trained model moves 14–16% less inside phones than LJ. V0 keeps slightly better WER and UTMOS (§10.6).
- **The ablations were branched late (20k steps from M1),** so they test late-stage augmentation, not augmentation from the start of fine-tuning.
- **The front end has one pronunciation per word,** and read-speech corpora never contain spelled letters. Live input does: the letter "a" and "I" after a vowel fail (table below), and homographs get one reading.

**Found in live testing of the browser demo (2026-11-12, E-35).** Spoken letters and the word "I", recorded by the owner through the browser demo. Each row was re-aligned with MFA after the fixes.

| Input (as typed) | What happened | Cause | Status |
|---|---|---|---|
| `abc`, `chaganti`, `lakshmi` (words first seen at inference) | MFA returned `<unk>`/`spn`, so the demo used the coarser CTC fallback | MFA cached its parsed copy of the lexicon in `data/mfa_tmp` and never saw words that G2P added later | **Fixed**: `align_one` uses a cache keyed by the lexicon's hash. Re-aligned with MFA: `abc` 0.54–1.32 s, `chaganti` 1.47–2.13 s |
| `bcd`, `xyz` (letters run together, no vowel) | G2P made up a word: `bcd` → B EH1 D ("bed") | Nothing told G2P that the token is a letter string | **Fixed**: unknown lowercase tokens of 3+ letters with no vowel are spelled out (`b c d`), like all-caps `GPU` |
| letter `a` in `a, b, c` / `g,a,c` / `have a,b,c` | `a` aligned to 30–80 ms. The spoken "ay" went to the silence or the next letter, and the output says "uh" | The lexicon has one pronunciation for `a`: AH0, the article. Checked: with an extra EY1 entry in a scratch lexicon, `a` aligned to 220–330 ms | **Open**: needs a letter-context rule plus a letter-name pronunciation, used by both MFA and synthesis |
| `I` in `may I know you` (2 of 3 recordings) | `i` got 30 ms (MFA's 3-frame minimum); `may` took the whole EY+AY stretch, so the output nearly drops "I". Measured error was 50–84 ms, against about 8 ms typical | Vowel-to-vowel junction (EY1→AY1) with no consonant, so MFA cannot place the boundary. The pronunciation (AY1) is right | **Open**: fine when there is a pause (3rd recording: 0.36 s), sentence-initially (`I would like…`: 0.16 s) and in `i, b, c` (0.27 s) |
| `a e i o u`, `b c d`, `g, i, c`, `p q` | correct | (`a` alone between pauses has no neighbour to lose its vowel to) | — |

The held-out results (§11) are unaffected. Every evaluation word was in the lexicon when MFA's cache was first built. The words added at inference (hi, ravi, dala, chaganti, lakshmi, abc, bcd, lexicon, normalized) all came from demo sessions.

**The two changes, and where they fail.**

*Change 1, MFA dictionary cache (a bug fix, not memorisation).*
- **What the cache is.** `mfa align_one` converts `lexicon.txt` (about 15k "word → phones" lines) into its own internal form and keeps that in its temporary directory. It holds no audio, no typed text and no results.
- **The bug.** MFA reused the converted copy even after G2P had appended new words to `lexicon.txt`, so every word added after the first run was unknown to MFA.
- **The fix.** The directory name now includes a hash of the lexicon. Any change to the file gives a fresh directory, which MFA rebuilds from the current file. Older directories are deleted.
- **Nothing is learned or remembered from earlier requests.** What persists is by design and was already so: a G2P guess is appended to the lexicon and reused for that word later, so a word is pronounced the same way every time.
- **The cache buys little.** Measured on one 4.4 s reference: 4.8 s with the cache, 6.0 s rebuilt from scratch.

Where it fails or could:
- **A wrong G2P guess is kept.** It stays in the lexicon until someone edits that line (e.g. `bcd` → "bed" is still in the local lexicon; the rule below now bypasses it).
- **The first request with a new word** pays the ~1.2 s rebuild.
- **Concurrent alignments are not supported.** Deleting old directories assumes one alignment at a time. The browser demo has a single worker; two parallel `align_one` runs with different lexicon versions could delete each other's directory.
- **A simpler alternative** with no cache at all is to rebuild on every call, at about +1.2 s per request.

*Change 2, spelling vowel-less letter strings (a hand-written heuristic).*
- **The rule.** A lowercase token of 3–5 letters that has no a/e/i/o/u and is not in CMUdict is spelled letter by letter (`bcd` → `b c d`), like the existing all-caps rule (`GPU`).
- **No model decides this.** It is a regex in `tts/text/input_rules.py`.

Probed edge cases:

| Input | Result | Right? |
|---|---|---|
| `bcd`, `xyz`, `dbms` | spelled | yes |
| `abd`, `bc`, `abc` | letters | yes, but only because CMUdict lists them |
| `hmm`, `shh`, `rhythm`, `mrs` | kept as words | yes (in CMUdict) |
| `ijk` | G2P invents a word | **no**: it contains a vowel, so the rule does not fire |
| `brrr`, `grr`, `pfft` | spelled "b r r r", "g r r", "p f f t" | **no**: interjections that are not in CMUdict |
| `pq` (2 letters) | not spelled | **no**: 2-letter tokens are excluded, because spelling them changed the L2 false start `cl` in a corpus transcript |

The robust alternative is a learned normaliser that decides from context whether to spell a token or read it as a word.

**Future work, ranked by expected value:**
1. Learn the alignment again inside MFA's word spans: MAS within each word, with the word boundaries fixed. This restores the alignment–latent feedback (§7.4) and keeps timing control.
2. A word-boundary or word-final flag on the encoder input (D-75c).
3. Learned diphthong split points, e.g. from formant trajectories, instead of 50/50.
4. Splitting other dynamic phones (CH, JH, ER) the same way.
5. A listening test (MOS or MUSHRA, plus accent ratings) to replace the classifier as the accent judge.
6. A faster aligner for live use: keep MFA loaded in memory, or use a stronger CTC aligner.
7. Letter names and vowel-to-vowel junctions (E-35): read a single letter in a letter list as its name (`a` → EY1), and give MFA a minimum duration for vowel-only words such as `I` and `a`.

**Takeaway.** Make durations an input, copy word timing (not phone timing), and check the tokens before adding data. Replacing MAS cut word error from 40.5 to 24 ms; the biggest later gain came from a two-token diphthong, not from more training data.

---

## 16. Claim → evidence index

**Audio evidence** ([`samples/`](../samples/README.md), E-36): every output there is a byte-exact copy of the evaluated file (`samples/SOURCES.tsv`, `MANIFEST.sha256`). For each demo sample the word-boundary MAE is recomputed from the committed TextGrids against the L2-ARCTIC human annotation and checked against `per_utt.csv` when the folder is built. All 1,248 test outputs per system are on the Hugging Face Hub (`make outputs`, pinned revision `e6bf851`).

| Claim needing audio | Samples |
| --- | --- |
| Time-aligned output on ≥ 2 held-out accents; length error 0 | [`samples/demo/`](../samples/demo): 3 accents, headline prompt × 12 speakers + best/median/worst per accent |
| Main system vs V0 (−16.5 ms) | each demo sample's README (main and V0 MAE side by side) |
| "Words end sharply" → m1_pos (§10.3) | [`samples/listening/p2a_word_endings/`](../samples/listening/p2a_word_endings) |
| "cry" heard as "cra" → diphthong split; US-prob drops inaudible (§10.5) | [`samples/listening/p2c_diphthongs/`](../samples/listening/p2c_diphthongs) (`cry_*`, `us_drop_*`) |
| Spoken letters, "I", MFA cache (§15, E-35) | [`samples/browser_tests/`](../samples/browser_tests) (the owner's recordings, before/after the fix) |


| Claim | Evidence | Code |
| --- | --- | --- |
| Output length is exact for every output | `length_error` column = 0 in every `results/*/test/per_utt.csv` | [`pipeline.py#L176`](../tts/infer/pipeline.py#L176), [`pipeline.py#L322`](../tts/infer/pipeline.py#L322) |
| MFA error on accented speech is about 19 ms (words) | [`reports/mfa_l2_dev.md`](../reports/mfa_l2_dev.md), plot 3 | [`l2.py#L92`](../tts/align/l2.py#L92) |
| Replacing MAS with MFA makes timing follow the reference (vs V0: −16.5 ms MAE, every accent) at a cost of +0.8 pt WER and −0.12 UTMOS | `results/paired/test_m1_pos_diph_vs_v0.csv`, `results/v0/test/summary.json` | [`v0.py`](../tts/models/vits_mas/v0.py), [`paired.py`](../tts/eval/paired.py) |
| V0 renders its given durations ~20 ms early and scattered; the MFA-trained systems do not | `results/fidelity/test.json` | [`fidelity.py#L62`](../tts/eval/fidelity.py#L62) |
| CTC has a +38 ms start bias on natural speech, so its MAE is not a ranking | `results/fidelity/calibration_dev.json` | [`fidelity.py#L83`](../tts/eval/fidelity.py#L83) |
| MFA-trained models move 14–16% less inside phones than LJ (fixed-alignment cost, §7.4) | `results/phone_dynamics/*.json` | [`phone_dynamics.py#L32`](../tts/eval/phone_dynamics.py#L32) |
| The 2.5× cap costs 0.4 ms against no cap, has the best UTMOS, and acts on 10% of test references | `results/tune/grid.json`, `n_capped` in `results/m1_pos_diph/test/per_utt.csv` | [`pipeline.py#L150-L153`](../tts/infer/pipeline.py#L150-L153) |
| M1 beats M0 on timing, WER and UTMOS | `results/dev_comparison.csv`, `results/test_comparison.csv` | [`evaluate.py#L270`](../scripts/evaluate.py#L270) |
| Copying phone durations hurts intelligibility | `results/tune/grid.json`, plot 7 | [`pipeline.py#L299-L302`](../tts/infer/pipeline.py#L299-L302) |
| The position embedding does not restore within-phone movement | `results/phone_dynamics/*.json` | [`phone_dynamics.py#L32`](../tts/eval/phone_dynamics.py#L32) |
| m1_pos raises US probability over m1_cont | D-78 (dev per-utterance CSVs) | [`metrics.py#L72`](../tts/eval/metrics.py#L72) |
| Ablations give no gain at any stretch | `results/paired/{dev,test}_abl_*_vs_m1_pos.csv` | [`paired.py#L42`](../tts/eval/paired.py#L42) |
| Rubber Band R3 degraded the augmented clips | `reports/r3/`, `reports/aug_a.md` | [`wsola.py#L21`](../tts/augment/wsola.py#L21) |
| The diphthong split improves timing, WER and UTMOS | `results/paired/{dev,test}_m1_pos_diph_vs_m1_pos.csv` | [`diph.py`](../tts/text/diph.py) |
| Dev pace is representative of test | [`results/plots/08_rho_dev_test.png`](../results/plots/08_rho_dev_test.png) | [`plots.py#L225`](../tts/eval/plots.py#L225) |
| The test split was not read before the freeze | `git tag freeze-v1` → 5889ecf; test results committed after it | [`guard.py#L39`](../tts/guard.py#L39) |
| Efficiency numbers | `results/profile_cuda.json`, `profile_cpu_12t.json`, `profile_cpu_1t.json` | [`profile.py`](../scripts/profile.py) |
| Typed text: 119/121 stress cases, corpus words unchanged | `make text-stress`, [`tests/data/text_cases.tsv`](../tests/data/text_cases.tsv), `test_corpus_word_sequences_unchanged` | [`input_rules.py`](../tts/text/input_rules.py) |
| Error distribution and stretch analysis (§11) | `results/<system>/test/per_utt.csv` (`wb_mae_ms`, `wb_within20/50`, `stretch_ratio`) | [`evaluate.py`](../scripts/evaluate.py) |
| Hosted checkpoints are the frozen ones | `MANIFEST.sha256` on the Hub; `scripts/export_infer.py --verify` (bit-identical output) | [`fetch_models.py`](../scripts/fetch_models.py) |
| Runs from a clean clone without login | D-104 (models, assets, infer, rehearse 12/12, stress 119/121, 300 tests, outputs) | `Makefile` quick-start targets |
