# Live demo runbook (D-96)

What to run in front of the interviewers, what to say, and what to do when something goes wrong.
Everything here runs the frozen main system; nothing changes a model or an experiment config.

## 0. Before the interview (once, ~10 min)

```bash
make models            # export/<system>/infer.pt + baseline weights from the HF repo, sha256-checked (D-95)
make test              # must be green
make text-stress       # 117 typed-text cases through the normalizer: expect "G2P 2 | PASS 115"
make rehearse          # 12 end-to-end cases -> out/live_demo/REHEARSAL.md: expect "12/12 cases as expected"
make live-demo         # the headline sentence; leaves a backup copy in out/live_demo/HKK_arctic_a0003/
```

Close other GPU programs. The first `make live-demo` after a reboot is slower (MFA and CUDA
caches), so run it once before the call.

## 1. The headline run (~1 min)

```bash
make live-demo                                   # HKK (Korean), "For the twentieth time that evening ..."
make live-demo UTT=YBAA_arctic_a0003             # same sentence, Arabic speaker
make live-demo UTT=HQTV_arctic_a0003             # same sentence, Vietnamese speaker
```

All 12 test speakers read this sentence, and the main system's error on it (24.8 ms) equals its
overall test error (23.9 ms), so it is a typical case, not a best case. The test speakers were
never used for training or tuning.

What the terminal shows, and what to say:

1. **`normalized`**: the words the model will say. Punctuation and case never reach the model.
2. **The word table.** "reference (MFA)" is where each word is in their recording. "measured" is
   where MFA finds the same word in our output, aligned exactly as `make eval` does. Δ is the
   difference. The main system lands within ~10 ms. V0, the original MAS VITS given the same
   timing, is consistently ~20–40 ms early. That is the D-90 finding: a MAS-trained model does
   not render durations it is given faithfully.
3. **`length error 0`**: the output has exactly the reference's length, in samples.
4. **Latency**: model RTF ~0.03–0.06 on the laptop GPU. MFA is ~13 s of the ~15 s total; that is
   the cost of an HMM aligner started per request (REPORT §12). The output alignment (~25 s) is
   only for the table and not part of synthesis.
5. Open the page: `explorer.exe "$(wslpath -w out/live_demo/HKK_arctic_a0003/index.html)"`.
   Play the reference, then main, then V0, and point at `timing.png` (the word boundaries of the
   reference and the output on one time axis).

Note that this live table compares with **MFA on the reference**. The headline 23.9 ms compares
with the **manual** annotation, so it also includes MFA's own error on accented speech (19.4 ms,
REPORT §6).

## 2. Their text and their audio

```bash
make live-demo UTT=SKA_arctic_a0031 TEXT="They were 300 yards apart."
make live-demo REF=/path/their.mp3 TEXT="what the recording says"     # any format ffmpeg reads
```

- **The text must be what the recording says.** The normalizer only changes the form
  (`300` → "three hundred", `Dr.` → "doctor", `12/03` → "december third"). If they type
  words the speaker did not say, MFA may fail, and the pipeline falls back to CTC. The
  `aligner` field says which one ran, and the length is still exact.
- To show how typed text is read without any audio: `make text-stress TEXT="Dr Smith's GPU
  costs ₹1,500 on 12/03."` prints the words and each word's phones with their source
  (lexicon / cmudict / g2p).

## 3. When something goes wrong

| What you see | What it means | What to say or do |
| --- | --- | --- |
| `aligner ctc (MFA failed: ...)` | MFA could not align the text to the audio (text mismatch, noise, very long silence) | "CTC fallback, logged in timings.json; the timing is a bit worse but the length is still exact." Check `normalized` against what the speaker said |
| They typed a word the speaker never said, and it "worked" | MFA does not fail on an inserted word. In rehearsal case 9 it gave the unspoken "old" 90 ms, taken from "two" and "men". The output says it in that slot | "A mismatch is not detected: MFA forces every transcript word into the audio. Detection would need a confidence check (MFA log-likelihood, or CTC agreement) — future work." |
| `[fail] NormalizeError: non-Latin letters` | non-English script | "The model has an English phone set, so it refuses rather than guessing." |
| `[fail] InferenceError: text has no words` | only punctuation or emoji | Ask for a sentence |
| `[fail] InferenceError: cannot decode ...` | the audio file cannot be read | Convert it, or use a different file |
| `g2p (new): [...]` | words not in the lexicon (names, brands) | "g2p_en guessed these and appended them to the lexicon, so later runs use the same pronunciation." |
| A word sounds wrong | a homograph (`read`, `live`) uses CMUdict's first entry; or a g2p guess | A known limitation: one pronunciation per word (CMUdict's first, by design: one lexicon everywhere), no POS tagging; fix it with a lexicon line |
| GPU out of memory | another program is using the GPU | `ARGS="--device cpu"` (slower synthesis, same output) |

Backup if the live run fails: the outputs of the last `make rehearse` and `make live-demo` in
`out/live_demo/`, and `make demo` → `out/demo_site/index.html` (every system, every accent).

## 4. What the rehearsal covers (configs/live_demo.yaml)

The rehearsal types every text the way an interviewer might: digits (`20th`, `300`, `10 days`,
`2 hrs`), `&`, curly quotes and dashes, all caps, tabs and extra spaces, trailing `...`/`?`/`:`.
The words stay the ones the speaker said. It also covers:
- an extra word the speaker never said. MFA squeezed it into 90 ms without complaint, so a
  mismatch is not detected; the length stays exact (see §3);
- non-Latin script, punctuation-only text and a missing file, each of which must be refused in
  one line.

`out/live_demo/REHEARSAL.md` has the result of the last run.

## 5. The normalizer stress set (`make text-stress`, tests/data/text_cases.tsv)

The set has 117 cases in 12 groups: numbers, dates, times, money, units, symbols, abbreviations,
initialisms, names and Roman numerals, punctuation, and inputs that must be refused. Before D-94,
58 failed: `12/03` was read as "twelve thirds", `Dr Smith` as "drive Smith", `10km` as
"ten km", `GPU` as a g2p guess, `World War II` as "ii", and Devanagari was transliterated into
consonants. The D-94 pre-pass fixes them. Two cases are correct but go to g2p: "lakh" and "py"
from PyTorch. The pre-pass changes none of the 26,867 L2-ARCTIC transcripts and one of the 13,100
LJ transcripts (a bare "Dr" that the old rules read as "drive"), so the frozen results stand.
