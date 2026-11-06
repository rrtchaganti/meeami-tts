# MFA error on accented speech (T5, `make align-dev-eval`, 2026-11-06)

- Dev speakers: ASI, RRBI, SVBI, TNI, BWC, LXC, NCC, TXHC, EBVS, ERMS, MBMPS, NJS; all 13450 scripted utterances aligned with speaker adaptation (`mfa align`, no --single_speaker, --beam 100 --retry_beam 400)
- Alignment status: ok 13450
- Compared on every utterance with a manual TextGrid (`annotation/`, ~150 per speaker). Words matched by label sequence; phones matched within matched words by canonical label (error tags and stress ignored; additions excluded). Interval times only.
- Each matched word / phone contributes two boundaries (start, end).

| Accent | Utts | Words matched | Word MAE (ms) | Word <=20 ms | Word <=50 ms | Phone MAE (ms) | Phone <=20 ms | Phone <=50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hindi | 600 | 100.0% | 18.2 | 76.8% | 94.8% | 12.3 | 89.0% | 97.5% |
| mandarin | 600 | 99.9% | 20.2 | 74.2% | 93.0% | 14.0 | 86.7% | 96.1% |
| spanish | 600 | 100.0% | 19.9 | 74.0% | 93.5% | 13.2 | 87.6% | 96.6% |
| all | 1800 | 100.0% | 19.4 | 75.0% | 93.8% | 13.1 | 87.7% | 96.7% |
