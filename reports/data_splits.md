# Data splits (T2, `make data`, 2026-11-06)

## LJSpeech

- 13100 clips; every file is 22050 Hz, 1 ch, PCM_16
- VITS filelists @ 2e561ba (sha256 verified): train 12500, val 100, test 500; disjoint, union = metadata

## L2-ARCTIC

| Split | Accent | Speaker | wav | annotation |
| --- | --- | --- | --- | --- |
| dev | hindi | ASI | 1131 | 150 |
| dev | hindi | RRBI | 1130 | 150 |
| dev | hindi | SVBI | 1132 | 150 |
| dev | hindi | TNI | 1131 | 150 |
| dev | mandarin | BWC | 1130 | 150 |
| dev | mandarin | LXC | 1131 | 150 |
| dev | mandarin | NCC | 1131 | 150 |
| dev | mandarin | TXHC | 1132 | 150 |
| dev | spanish | EBVS | 1007 | 150 |
| dev | spanish | ERMS | 1132 | 150 |
| dev | spanish | MBMPS | 1132 | 150 |
| dev | spanish | NJS | 1131 | 150 |
| test | arabic | ABA | 1129 | 150 |
| test | arabic | SKA | 974 | 150 |
| test | arabic | YBAA | 1130 | 149 |
| test | arabic | ZHAA | 1132 | 150 |
| test | korean | HJK | 1131 | 150 |
| test | korean | HKK | 1131 | 150 |
| test | korean | YDCK | 1131 | 150 |
| test | korean | YKWK | 1131 | 150 |
| test | vietnamese | HQTV | 1132 | 150 |
| test | vietnamese | PNV | 1132 | 150 |
| test | vietnamese | THV | 1132 | 150 |
| test | vietnamese | TLV | 1132 | 150 |

- **dev evaluation set:** 100 utterance IDs annotated for all 12 speakers -> 1200 reference utterances
- **test evaluation set:** 104 utterance IDs annotated for all 12 speakers -> 1248 reference utterances
  - test: YBAA lacks 1 ID(s) all other test speakers have: arctic_a0094
- Dev and test evaluation sets share 99 sentence IDs (same ARCTIC prompts, different speakers/accents)
- Suitcase corpus (stress-test demo only): 22 recordings
- Test split locked (no `freeze-v1` tag): test audio not opened; `sr`/`n_samples` are null in l2_test.jsonl

## Manifests

- `lj_train.jsonl`: kept (exists; use --force to overwrite)
- `lj_val.jsonl`: kept (exists; use --force to overwrite)
- `lj_test.jsonl`: kept (exists; use --force to overwrite)
- `aug_a.jsonl`: wrote 0 rows
- `aug_b.jsonl`: wrote 0 rows
- `l2_dev.jsonl`: kept (exists; use --force to overwrite)
- `l2_test.jsonl`: kept (exists; use --force to overwrite)
