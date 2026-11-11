# Live demo runbook (D-96)

What to run in front of the interviewers, what to say, and what to do when something goes wrong.
Everything here runs the frozen main system; nothing changes a model or an experiment config.

## 0a. On a fresh machine (~15 min, no dataset needed; tested from a clean clone: D-100)

```bash
git clone git@github.com:rrtchaganti/meeami-tts.git && cd meeami-tts
make env                 # envs `tts` + `mfa`; if those names exist: export TTS_ENV=tts2 MFA_ENV=mfa2 first
hf auth login            # the model repo Raviteja546/meeami-tts is private
make models              # 5 checkpoints + 3 baseline weights + the frozen lexicon, sha256-checked
make assets              # pinned MFA english_us_arpa models + NLTK data
make test                # ~284 passed, ~9 skipped (the skips need the training data)
```

The demo's 8 reference clips ship in `demo/clips/` (L2-ARCTIC test speakers, CC BY-NC 4.0), so
`make live-demo` and `make rehearse` run without the dataset. Then continue with §0.

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

## 6. Browser demo hosted on this laptop (D-101)

The visitor records or uploads a sentence in their browser, types what it says, and gets the
outputs of **all seven systems** side by side: main, M1, M1+pos, both ablations, V0 and M0. Each
system shows its audio, timing plot, aligner, length error and, optionally, the word-time error
measured with MFA.

```bash
make web-demo            # terminal 1: loads all 7 systems (~15 s), prints http://localhost:7860/?key=XXXX
make web-tunnel          # terminal 2: prints https://<random>.trycloudflare.com
# send the visitor:  https://<random>.trycloudflare.com/?key=XXXX
```

- **Why a tunnel:** WSL2 sits behind NAT (172.17.x), so other machines cannot reach it directly.
  Browsers also allow the **microphone only on https:// or localhost**. A Cloudflare quick tunnel
  gives an https URL with no account and no router or Windows-firewall changes.
- **Install cloudflared once, without sudo:**
  ```bash
  mkdir -p ~/.local/bin && curl -L -o ~/.local/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    && chmod +x ~/.local/bin/cloudflared   # ~/.local/bin must be on PATH
  ```
- **First seconds of a tunnel:** cloudflared may log `ERR Register tunnel error ... Tunnel not
  found` a few times, and the URL answers 530, until it prints `Registered tunnel connection`
  (~10 s in the test). The `WRN ... ping_group_range` / ICMP lines are harmless: ping is not
  needed.
- **Access key:** every request needs `?key=`. It is random at each start, or fixed with
  `MEEAMI_DEMO_KEY=... make web-demo`. Without the key the page and the API answer 403. The URL
  changes on every `make web-tunnel`. Stop both terminals (Ctrl+C) after the session.
- **What one request costs:**
  - MFA on the reference: ~5–10 s;
  - all 7 syntheses: < 1 s together on the laptop GPU;
  - with "measure word times": +20–40 s for MFA on the 7 outputs.

  Requests queue and run one at a time. The page shows how many jobs are ahead.
- **Limits** (`configs/web_demo.yaml`): 30 s of audio, 20 MB per upload, 400 characters of text.
  Any format ffmpeg reads works; browser recordings arrive as webm/opus.
- **Result links:** after a run the address bar holds `&job=<id>`, so the page can be reloaded or
  shared. Results stay in `out/web_demo/<id>/` (the last 50 are kept).
- **Keep the laptop awake** and on power during the session. The GPU holds all 7 models in
  ~1.5 GB.
- **Same text rule as everywhere:** the text must be what the recording says. An extra word is
  not detected (§3).
