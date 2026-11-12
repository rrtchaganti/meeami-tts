# Samples: the exact audio behind the reported results

Built by `make samples` ([`scripts/export_samples.py`](../scripts/export_samples.py), [`configs/samples.yaml`](../configs/samples.yaml)). Nothing here was re-synthesized: each output is a byte-exact copy of the file the metrics were computed on ([`SOURCES.tsv`](SOURCES.tsv) gives every file's origin, [`MANIFEST.sha256`](MANIFEST.sha256) its hash). References are unmodified L2-ARCTIC recordings (CC BY-NC 4.0).

## 1. Demonstration on held-out accents (test split)

Selection, fixed in the config before listening: the headline prompt `arctic_a0003` for all 12 test speakers, plus the main system's best, median and worst utterance per accent by word-boundary MAE. So failures are shown, not only successes. Each folder has the input text, the reference, all 7 systems' outputs, word timings of the reference and every output (`timings.json`, `timing.png`) and a README with the per-utterance metrics from `results/<system>/test/per_utt.csv`.

| Accent | Sample | Selected as | Text | Main MAE (ms) | V0 MAE (ms) |
|---|---|---|---|---|---|
| arabic | [`ABA_arctic_a0003`](demo/arabic/ABA_arctic_a0003/) | headline prompt | For the twentieth time that evening the two men shook hands | 47.3 | 52.3 |
| arabic | [`SKA_arctic_a0003`](demo/arabic/SKA_arctic_a0003/) | headline prompt | For the twentieth time that evening the two men shook hands | 20.2 | 48.8 |
| arabic | [`YBAA_arctic_a0003`](demo/arabic/YBAA_arctic_a0003/) | headline prompt | For the twentieth time that evening the two men shook hands | 23.6 | 41.4 |
| arabic | [`ZHAA_arctic_a0003`](demo/arabic/ZHAA_arctic_a0003/) | headline prompt | For the twentieth time that evening the two men shook hands | 25.0 | 33.2 |
| arabic | [`ABA_arctic_a0018`](demo/arabic/ABA_arctic_a0018/) | best main-system MAE (6.0 ms) | There was a change now | 6.0 | 17.0 |
| arabic | [`YBAA_arctic_a0122`](demo/arabic/YBAA_arctic_a0122/) | median main-system MAE (22.2 ms) | Two years ago I gave up civilization for this | 22.2 | 35.0 |
| arabic | [`ABA_arctic_a0048`](demo/arabic/ABA_arctic_a0048/) | worst main-system MAE (131.4 ms) | He looked at the handkerchief more closely | 131.4 | 142.1 |
| korean | [`HJK_arctic_a0003`](demo/korean/HJK_arctic_a0003/) | headline prompt | For the twentieth time that evening the two men shook hands | 19.5 | 33.2 |
| korean | [`HKK_arctic_a0003`](demo/korean/HKK_arctic_a0003/) | headline prompt | For the twentieth time that evening the two men shook hands | 24.1 | 31.4 |
| korean | [`YDCK_arctic_a0003`](demo/korean/YDCK_arctic_a0003/) | headline prompt | For the twentieth time that evening the two men shook hands | 25.0 | 37.7 |
| korean | [`YKWK_arctic_a0003`](demo/korean/YKWK_arctic_a0003/) | headline prompt | For the twentieth time that evening the two men shook hands | 16.4 | 32.3 |
| korean | [`HJK_arctic_a0053`](demo/korean/HJK_arctic_a0053/) | best main-system MAE (3.7 ms) | Suddenly his fingers closed tightly over the handkerchief | 3.8 | 31.9 |
| korean | [`HJK_arctic_a0039`](demo/korean/HJK_arctic_a0039/) | median main-system MAE (20.5 ms) | The ship should be in within a week or ten days | 20.5 | 30.5 |
| korean | [`HJK_arctic_a0028`](demo/korean/HJK_arctic_a0028/) | worst main-system MAE (113.3 ms) | Robbery bribery fraud | 113.3 | 110.0 |
| vietnamese | [`HQTV_arctic_a0003`](demo/vietnamese/HQTV_arctic_a0003/) | headline prompt | For the twentieth time that evening the two men shook hands | 25.9 | 44.5 |
| vietnamese | [`PNV_arctic_a0003`](demo/vietnamese/PNV_arctic_a0003/) | headline prompt | For the twentieth time that evening the two men shook hands | 15.9 | 32.7 |
| vietnamese | [`THV_arctic_a0003`](demo/vietnamese/THV_arctic_a0003/) | headline prompt | For the twentieth time that evening the two men shook hands | 35.9 | 51.8 |
| vietnamese | [`TLV_arctic_a0003`](demo/vietnamese/TLV_arctic_a0003/) | headline prompt | For the twentieth time that evening the two men shook hands | 18.2 | 27.3 |
| vietnamese | [`THV_arctic_a0130`](demo/vietnamese/THV_arctic_a0130/) | best main-system MAE (3.0 ms) | She was his now forever | 3.0 | 28.0 |
| vietnamese | [`THV_arctic_a0013`](demo/vietnamese/THV_arctic_a0013/) | median main-system MAE (22.1 ms) | He was a head shorter than his companion of almost delicate physique | 22.1 | 38.3 |
| vietnamese | [`THV_arctic_a0067`](demo/vietnamese/THV_arctic_a0067/) | worst main-system MAE (148.2 ms) | Surely I will excuse you she cried | 148.2 | 148.2 |

**All 1,248 test utterances × 7 systems** (the full evidence for every number in REPORT §11) are on the Hugging Face Hub: `make outputs` downloads, sha256-checks and unpacks them into `results/<system>/test/` (`Raviteja546/meeami-tts`, `outputs/test/<system>.tar`).

## 2. Listening sets behind design decisions (dev split)

| Set | Decision | Claim | Folders |
|---|---|---|---|
| [`p2a_word_endings`](listening/p2a_word_endings/) | D-76..D-78 (REPORT §10.3) | "Words end sharply" (M1). m1_pos was chosen over m1_cont after listening; US probability 0.769 vs 0.680. Each folder: the L2-ARCTIC reference and m1, m1_cont, m1_pos outputs. | 12 |
| [`p2c_diphthongs`](listening/p2c_diphthongs/) | D-80..D-85 (REPORT §10.5) | "cry" heard as "cra" (folder cry_*: m1_pos holds /a/, m1_pos_diph glides to /ɪ/), the four largest US-probability drops (us_drop_*: no audible difference), and the 12 random refs. | 17 |

## 3. Browser-demo tests and failure cases (E-35)

[`browser_tests/`](browser_tests/): all 50 jobs the owner ran on 2026-11-12 (29 before and 21 after the MFA-cache fix), with the recordings. Open failures (letter `a`, `I` after a vowel) and fixed ones are tagged; see REPORT §15.
