# Demo reference clips

Eight utterances from **L2-ARCTIC v5** (Zhao et al., 2018; https://psi.engr.tamu.edu/l2-arctic-corpus/),
licensed **CC BY-NC 4.0**, unmodified (44.1 kHz WAV + transcript). They are the inputs of
`make live-demo` and `make rehearse` (configs/live_demo.yaml), so the demo runs without the dataset.

| File | Speaker | L1 accent | Prompt |
| --- | --- | --- | --- |
| HKK_arctic_a0003 | HKK | Korean | For the twentieth time that evening the two men shook hands |
| YBAA_arctic_a0003 | YBAA | Arabic | (same) |
| HQTV_arctic_a0003 | HQTV | Vietnamese | (same) |
| SKA_arctic_a0031 | SKA | Arabic | They were three hundred yards apart |
| THV_arctic_a0031 | THV | Vietnamese | (same) |
| YKWK_arctic_a0039 | YKWK | Korean | The ship should be in within a week or ten days |
| ZHAA_arctic_a0121 | ZHAA | Arabic | They ate dinner at the fifth and rested for two hours |
| HJK_arctic_a0122 | HJK | Korean | Two years ago I gave up civilization for this |

All are **test** speakers (held-out accents). They were added after the git tag `freeze-v1`
(D-100) and were never used for training or tuning.
