# YKWK_arctic_a0003 (korean, held-out test speaker)

**Selected as:** headline prompt

**Text:** For the twentieth time that evening the two men shook hands

**Reference (ground truth):** [`reference.wav`](reference.wav), L2-ARCTIC `YKWK/wav/arctic_a0003.wav`, 4.13 s; ground-truth word timing: [`reference_annotation.TextGrid`](reference_annotation.TextGrid) (L2-ARCTIC manual annotation, the reference for every MAE below)

![timing](timing.png)

| System | Output | Length error (samples) | Word-boundary MAE (ms, MFA on output) | WER | UTMOS | US prob |
|---|---|---|---|---|---|---|
| main (m1_pos_diph) | [`m1_pos_diph.wav`](m1_pos_diph.wav) | 0 | 16.4 | 0.000 | 4.40 | 0.98 |
| M1 | [`m1.wav`](m1.wav) | 0 | 15.0 | 0.000 | 4.50 | 1.00 |
| M1 + position embedding | [`m1_pos.wav`](m1_pos.wav) | 0 | 15.0 | 0.000 | 4.48 | 1.00 |
| Ablation A | [`abl_a_pos.wav`](abl_a_pos.wav) | 0 | 18.2 | 0.000 | 4.46 | 0.99 |
| Ablation B | [`abl_b_pos.wav`](abl_b_pos.wav) | 0 | 17.7 | 0.000 | 4.31 | 1.00 |
| V0 (pretrained VITS + MAS) | [`v0.wav`](v0.wav) | 0 | 32.3 | 0.091 | 4.48 | 0.88 |
| M0 (FastSpeech2) | [`m0.wav`](m0.wav) | 0 | 17.7 | 0.000 | 4.40 | 0.97 |

**Word timings** (seconds). *Annotation* = human ground truth; *MFA on reference* = the timing the system copies (input); *output* = MFA on the m1_pos_diph output. Δ = output − annotation; the MAE above is the mean of |Δstart| and |Δend|, recomputed from these files and checked against `per_utt.csv`.

| Word | Annotation | MFA on reference | Output | Δstart (ms) | Δend (ms) |
|---|---|---|---|---|---|
| for | 0.00–0.22 | 0.00–0.22 | 0.00–0.22 | 0.0 | 0.0 |
| the | 0.22–0.31 | 0.22–0.31 | 0.22–0.31 | 0.0 | 0.0 |
| twentieth | 0.31–1.05 | 0.31–1.06 | 0.31–1.06 | 0.0 | 10.0 |
| time | 1.05–1.56 | 1.06–1.54 | 1.06–1.55 | 10.0 | -10.0 |
| that | 1.79–2.05 | 1.80–2.01 | 1.84–2.01 | 50.0 | -40.0 |
| evening | 2.05–2.56 | 2.01–2.56 | 2.01–2.57 | -40.0 | 10.0 |
| the | 2.59–2.72 | 2.59–2.70 | 2.61–2.70 | 20.0 | -20.0 |
| two | 2.72–2.97 | 2.70–2.97 | 2.70–2.97 | -20.0 | 0.0 |
| men | 2.97–3.23 | 2.97–3.22 | 2.97–3.22 | 0.0 | -10.0 |
| shook | 3.23–3.60 | 3.22–3.58 | 3.22–3.57 | -10.0 | -30.0 |
| hands | 3.60–4.11 | 3.58–4.07 | 3.57–4.06 | -30.0 | -50.0 |

All systems' planned and measured word timings: [`timings.json`](timings.json). Each wav is a byte-exact copy of `results/<system>/test/wav/YKWK_arctic_a0003/out.wav` (see `../../../SOURCES.tsv`).
