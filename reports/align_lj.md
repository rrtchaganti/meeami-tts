# LJSpeech alignment (T4, `make align-lj`, 2026-11-06)

- MFA english_us_arpa v3.0.0, `--single_speaker -j 8 --beam 100 --retry_beam 400 --clean`
- **Kept 13073 / 13100 (99.79%)**; required >= 98% -> PASS
- Kept per split: train 12474, val 100, test 499
- Every kept row: sum(durations) == n_frames (training spectrogram frame count)

| Drop reason | Files |
| --- | --- |
| non-silence phone > 0.5 s | 27 |

## Details

- Internal silences >= 3 frames kept as `sp`: 31650; shorter ones merged into the preceding phone: 1710
- Speech phones that rounded to 0 frames and took 1 frame from the longer neighbour (kept rows): 382
- Raw duration sum minus n_frames (all parsed files): +1: 13100
- Dropped IDs: `data/align/lj_dropped.tsv`
