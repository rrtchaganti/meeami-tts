# MFA error on accented speech (T5, `make align-dev-eval`, 2026-11-06)

- Dev speakers: ASI, RRBI, SVBI, TNI, BWC, LXC, NCC, TXHC, EBVS, ERMS, MBMPS, NJS; all 13450 scripted utterances aligned with speaker adaptation (`mfa align`, no --single_speaker, --beam 100 --retry_beam 400); MFA wall time 10.1 min
- Alignment status: ok 13450
- Compared on every utterance with a manual TextGrid (`annotation/`, ~150 per speaker). Words matched by label sequence; phones matched within matched words by canonical label (error tags and stress ignored; additions excluded). Interval times only.
- Each matched word / phone contributes two boundaries (start, end).

| Accent | Utts | Words matched | Word MAE (ms) | Word <=20 ms | Word <=50 ms | Phone MAE (ms) | Phone <=20 ms | Phone <=50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hindi | 600 | 100.0% | 18.2 | 62.0% | 93.4% | 12.3 | 77.7% | 97.0% |
| mandarin | 600 | 99.9% | 20.2 | 61.1% | 91.6% | 14.0 | 76.8% | 95.5% |
| spanish | 600 | 100.0% | 19.9 | 61.0% | 92.1% | 13.2 | 78.0% | 96.0% |
| all | 1800 | 100.0% | 19.4 | 61.4% | 92.4% | 13.1 | 77.5% | 96.1% |
