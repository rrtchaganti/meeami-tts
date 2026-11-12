---
language: en
tags: [text-to-speech, vits, duration-control, forced-alignment]
license: other
---

# meeami-tts checkpoints

Inference checkpoints for **meeami-tts**, a time-aligned US-English TTS. Give it a sentence and a
recording of that sentence by a non-US speaker. It says the sentence in the LJSpeech voice with a
US accent, on the reference's word timing, and the output is exactly as long as the reference.

Code, report and decision log: https://github.com/rrtchaganti/meeami-tts (`docs/REPORT.md`). The
checkpoints are fetched by `make models` (`scripts/fetch_models.py`), which checks every file
against `MANIFEST.sha256`; the test outputs by `make outputs` (`outputs/test/MANIFEST.sha256`).

## Files

| Path | What it is | Size |
| --- | --- | --- |
| `vits_mfa/m1_pos_diph/infer.pt` | **Main system.** VITS fine-tuned with MFA durations instead of MAS, plus a position-in-phone embedding and diphthong split | 142 MB |
| `vits_mfa/m1/infer.pt` | M1: VITS fine-tuned with MFA durations | 141 MB |
| `vits_mfa/m1_pos/infer.pt` | M1 + position-in-phone embedding | 141 MB |
| `vits_mfa/abl_a_pos/infer.pt` | Ablation A: m1_pos + 40% pace-augmented clips (LJ pace statistics) | 141 MB |
| `vits_mfa/abl_b_pos/infer.pt` | Ablation B: m1_pos + 40% pace-augmented clips (dev-accent pace statistics) | 141 MB |
| `baselines/vits/pretrained_ljs.pth` | V0 baseline: unmodified mirror of the official pretrained LJSpeech VITS (jaywalnut310/vits) | 146 MB |
| `baselines/fastspeech2/900000.pth.tar`, `.../hifigan/generator_LJSpeech.pth.tar` | M0 baseline: unmodified mirror of ming024/FastSpeech2's LJSpeech checkpoint and HiFi-GAN | 474 MB |

| `lexicon/lexicon.txt` | The frozen pronunciation lexicon (the only pronunciation source) | 0.4 MB |
| `outputs/test/<system>.tar` (7) | **Every evaluated held-out test output**: 1,248 utterances per system, each `wav/<id>/out.wav` + `timings.json`, the MFA TextGrid measured on it (`align_out/`), and `per_utt.csv` / `summary.json` / `scores.jsonl`. The exact files behind every reported number | 207 MB each |

Each `infer.pt` holds only what inference reads: the generator weights (`net_g`), the training
config and the step. The discriminators and optimizer states are dropped, which takes the file
from 956 MB to 141 MB. Loading it gives bit-identical weights and waveforms to the full training
checkpoint (`scripts/export_infer.py --verify`), and `source.sha256` names that checkpoint.

## Results on held-out accents (Arabic, Korean, Vietnamese; 12 speakers, 1,248 references)

| System | Word-boundary MAE | Pause F1 | WER | US prob | UTMOS |
| --- | --- | --- | --- | --- | --- |
| V0 (original VITS, MAS) | 40.5 ms | 0.815 | 2.1% | 0.649 | 4.31 |
| M0 (FastSpeech2) | 31.9 ms | 0.726 | 9.3% | 0.746 | 3.96 |
| M1 | 24.2 ms | 0.822 | 3.8% | 0.734 | 4.19 |
| **m1_pos_diph (main)** | **23.9 ms** | **0.835** | 2.9% | 0.762 | 4.20 |

## Licences and training data

- `m1`, `m1_pos`, `m1_pos_diph`, `abl_a_pos`: fine-tuned from the MIT-licensed pretrained VITS
  on LJSpeech 1.1 (public domain) only; `abl_a_pos` adds time-stretched LJSpeech copies.
- `abl_b_pos`: the time-stretched LJSpeech copies follow pace statistics measured on L2-ARCTIC
  dev speakers (CC BY-NC 4.0). No L2-ARCTIC audio was used in training. Use it for
  **non-commercial research only**, to be safe.
- `baselines/`: unmodified third-party weights, mirrored with their MIT licences (see
  `LICENSES.md`).
- `outputs/test/`: synthesized speech only (the LJSpeech voice), but its texts and word timings
  come from L2-ARCTIC test recordings (CC BY-NC 4.0): **non-commercial research only**.
- **No L2-ARCTIC audio is stored in this repository.** Each `timings.json` names the reference
  file it used.
