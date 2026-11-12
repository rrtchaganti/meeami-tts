# Browser-demo tests by the owner (2026-11-12, DECISIONS E-35)

Every job the owner ran in the browser demo, in order, unfiltered. Each folder: the recording exactly as uploaded (`recording.webm`, playable in any browser), `reference_mfa.TextGrid` (MFA on the recording, as run), the main and V0 outputs with their measured TextGrids, and `result.json`. Where present: `reference_mfa_after_fix.TextGrid` (re-aligned after the cache fix) and `reference_mfa_a_EY1.TextGrid` (scratch lexicon with `a EY1`, diagnosis only).

**Cases:**

- `letter_a`: letter `a` read and aligned as AH0 ('uh'): collapses to 30-120 ms
- `i_after_vowel`: `I` after a vowel (EY->AY): collapses to 30 ms
- `new_word_ctc`: word added at inference unknown to MFA (stale cache): CTC fallback (fixed)
- `g2p_letters`: letter string read as a word by G2P

Reference word durations come from MFA on the recording; ⚠ marks words under 60 ms (collapsed). These recordings have no human annotation, so 'Output MAE' compares MFA on the main output with MFA on the recording (the timing the system copied): it measures timing transfer, not agreement with a ground truth, and is not comparable with the test-set MAE. Job order is chronological. One earlier job ("Hi there, I'm chaganti") was deleted by the demo's job limit before archiving; its recording was re-run after the fix as the job with that text below.

| Job | Fix | Text | Normalized | Aligner | Reference words (s) | After fix (s) | Output MAE (ms) | Case |
|---|---|---|---|---|---|---|---|---|
| [`703f7b4f16b0`](703f7b4f16b0/) | before | lakshmi is here | lakshmi is here | ctc (spn) | lakshmi(0.40) is(0.22) here(0.44) | lakshmi(0.68) is(0.20) here(0.47) | 95 | new_word_ctc |
| [`994f7bd5cba1`](994f7bd5cba1/) | before | a, b, c are three letters of english language | a, b, c are three letters of english language | mfa | a(0.03⚠) b(0.79) c(0.52) are(0.39) three(0.23) letters(0.71) of(0.28) english(0.50) language(0.65) | a(0.03⚠) b(0.79) c(0.52) are(0.40) three(0.23) letters(0.71) of(0.28) english(0.50) language(0.65) | 17 | letter_a |
| [`8bee7ff4bd2a`](8bee7ff4bd2a/) | before | abc are three letters of english language | abc are three letters of english language | ctc (spn) | abc(0.36) are(0.26) three(0.24) letters(0.48) of(0.18) english(0.40) language(0.54) | abc(0.78) are(0.37) three(0.20) letters(0.49) of(0.27) english(0.45) language(0.64) | 57 | new_word_ctc |
| [`97a5de018ad4`](97a5de018ad4/) | before | bcd are three letters of english language | bcd are three letters of english language | ctc (spn) | bcd(0.48) are(0.22) three(0.20) letters(0.40) of(0.16) english(0.37) language(0.45) | b(0.17) c(0.28) d(0.26) are(0.45) three(0.17) letters(0.40) of(0.22) english(0.42) language(0.51) | 14 | new_word_ctc, g2p_letters |
| [`0fd1e15efb9a`](0fd1e15efb9a/) | before | b c d are three letters of english language | b c d are three letters of english language | mfa | b(0.39) c(0.52) d(0.32) are(0.28) three(0.13) letters(0.39) of(0.18) english(0.35) language(0.33) |  | 8 |  |
| [`0bad90d51999`](0bad90d51999/) | before | i j k c d are letters of english language | i j k c d are letters of english language | mfa | i(0.42) j(0.30) k(0.45) c(0.42) d(0.50) are(0.26) letters(0.46) of(0.29) english(0.40) language(0.35) |  | 31 |  |
| [`86d6f2c88375`](86d6f2c88375/) | before | e j k c d are letters of english language | e j k c d are letters of english language | mfa | e(0.29) j(0.48) k(0.24) c(0.38) d(0.40) are(0.17) letters(0.36) of(0.15) english(0.35) language(0.32) |  | 25 |  |
| [`b4d4a18e7d2c`](b4d4a18e7d2c/) | before | o j k c d are letters of english language | o j k c d are letters of english language | mfa | o(0.34) j(0.44) k(0.30) c(0.36) d(0.38) are(0.16) letters(0.38) of(0.13) english(0.36) language(0.33) |  | 5 |  |
| [`3e07f6c1c588`](3e07f6c1c588/) | before | m n o p q are letters of english language | m n o p q are letters of english language | mfa | m(0.39) n(0.40) o(0.36) p(0.30) q(0.43) are(0.22) letters(0.41) of(0.18) english(0.37) language(0.49) |  | 39 |  |
| [`154aeece6a18`](154aeece6a18/) | before | n o p q are letters of english language | n o p q are letters of english language | mfa | n(0.39) o(0.25) p(0.43) q(0.35) are(0.18) letters(0.37) of(0.18) english(0.36) language(0.54) |  | 14 |  |
| [`eb210e1639f9`](eb210e1639f9/) | before | p q are letters of english language | p q are letters of english language | mfa | p(0.34) q(0.36) are(0.27) letters(0.44) of(0.20) english(0.34) language(0.48) |  | 9 |  |
| [`8fe0db698ec3`](8fe0db698ec3/) | before | letters of english language have p q | letters of english language have p q | mfa | letters(0.52) of(0.19) english(0.35) language(0.39) have(0.24) p(0.17) q(0.43) |  | 25 |  |
| [`4f0239ed9458`](4f0239ed9458/) | before | letters of english language have a q | letters of english language have a q | mfa | letters(0.44) of(0.20) english(0.36) language(0.50) have(0.29) a(0.12) q(0.35) |  | 24 | letter_a |
| [`b7cbb72b5dcb`](b7cbb72b5dcb/) | before | letters of english language have a e i o u | letters of english language have a e i o u | mfa | letters(0.42) of(0.30) english(0.37) language(0.46) have(0.52) a(0.45) e(0.35) i(0.28) o(0.35) u(0.37) | letters(0.42) of(0.30) english(0.37) language(0.46) have(0.52) a(0.45) e(0.35) i(0.28) o(0.35) u(0.37) | 106 |  |
| [`100220176c8e`](100220176c8e/) | before | letters of english language have a,b,c | letters of english language have a, b, c | mfa | letters(0.41) of(0.17) english(0.37) language(0.45) have(0.30) a(0.08) b(0.45) c(0.30) | letters(0.41) of(0.17) english(0.37) language(0.45) have(0.30) a(0.08) b(0.45) c(0.30) | 40 | letter_a |
| [`cbb347b1639c`](cbb347b1639c/) | before | letters of english language have i,o,c,e | letters of english language have i, o, c, e | mfa | letters(0.45) of(0.17) english(0.37) language(0.46) have(0.33) i(0.37) o(0.34) c(0.52) e(0.33) |  | 34 |  |
| [`7cdc5446b89f`](7cdc5446b89f/) | before | a,b,c are letters of english language | a, b, c are letters of english language | mfa | a(0.03⚠) b(0.61) c(0.50) are(0.27) letters(0.39) of(0.20) english(0.36) language(0.57) | a(0.03⚠) b(0.61) c(0.50) are(0.27) letters(0.39) of(0.20) english(0.36) language(0.57) | 48 | letter_a |
| [`14771eb36882`](14771eb36882/) | before | i,b,c are letters of english language | i, b, c are letters of english language | mfa | i(0.27) b(0.33) c(0.41) are(0.22) letters(0.42) of(0.14) english(0.39) language(0.50) |  | 42 |  |
| [`445e120a7a07`](445e120a7a07/) | before | g,b,c are letters of english language | g, b, c are letters of english language | mfa | g(0.31) b(0.34) c(0.38) are(0.25) letters(0.36) of(0.15) english(0.36) language(0.52) |  | 8 |  |
| [`6541f5b1e2c7`](6541f5b1e2c7/) | before | g,a,c are letters of english language | g, a, c are letters of english language | mfa | g(0.65) a(0.06) c(0.41) are(0.25) letters(0.36) of(0.17) english(0.34) language(0.47) | g(0.65) a(0.06) c(0.41) are(0.25) letters(0.36) of(0.17) english(0.34) language(0.47) | 9 | letter_a |
| [`abe487d0f980`](abe487d0f980/) | before | g,a c are letters of english language | g, a c are letters of english language | mfa | g(0.65) a(0.06) c(0.41) are(0.25) letters(0.36) of(0.17) english(0.34) language(0.47) |  | 9 | letter_a |
| [`886aa58eeb0a`](886aa58eeb0a/) | before | g,i c are letters of english language | g, i c are letters of english language | mfa | g(0.38) i(0.19) c(0.45) are(0.20) letters(0.40) of(0.20) english(0.34) language(0.48) |  | 9 |  |
| [`175b9dae09b2`](175b9dae09b2/) | before | g,i,d are letters of english language | g, i, d are letters of english language | mfa | g(0.38) i(0.22) d(0.42) are(0.28) letters(0.30) of(0.22) english(0.34) language(0.47) |  | 9 |  |
| [`ccfb96c4821c`](ccfb96c4821c/) | before | g,a,d are letters of english language | g, a, d are letters of english language | mfa | g(0.38) a(0.23) d(0.41) are(0.28) letters(0.30) of(0.22) english(0.34) language(0.47) |  | 9 |  |
| [`2e53c8f7ae66`](2e53c8f7ae66/) | before | g,a,d are letters of english language | g, a, d are letters of english language | mfa | g(0.63) a(0.11) d(0.26) are(0.26) letters(0.36) of(0.11) english(0.37) language(0.33) |  | 13 | letter_a |
| [`4069057cdee0`](4069057cdee0/) | before | I would like to introduce myself | i would like to introduce myself | mfa | i(0.16) would(0.19) like(0.28) to(0.06) introduce(0.58) myself(0.38) | i(0.16) would(0.19) like(0.28) to(0.06) introduce(0.58) myself(0.38) | 40 |  |
| [`ac96a8d9836b`](ac96a8d9836b/) | before | may I know you? | may i know you? | mfa | may(0.40) i(0.03⚠) know(0.18) you(0.28) | may(0.40) i(0.03⚠) know(0.18) you(0.28) | 51 | i_after_vowel |
| [`64ea971eb9c0`](64ea971eb9c0/) | before | may I know you? | may i know you? | mfa | may(0.53) i(0.36) know(0.30) you(0.42) | may(0.53) i(0.36) know(0.30) you(0.42) | 50 |  |
| [`6c41ee1a96ee`](6c41ee1a96ee/) | before | may I know you? | may i know you? | mfa | may(0.31) i(0.03⚠) know(0.15) you(0.33) | may(0.31) i(0.03⚠) know(0.15) you(0.33) | 84 | i_after_vowel |
| [`9272ab5d54f7`](9272ab5d54f7/) | after | abc are three letters of english language | abc are three letters of english language | mfa | abc(0.78) are(0.37) three(0.20) letters(0.49) of(0.27) english(0.45) language(0.64) |  | – |  |
| [`ec6b2fdceb65`](ec6b2fdceb65/) | after | bcd are three letters of english language | b c d are three letters of english language | mfa | b(0.17) c(0.28) d(0.26) are(0.45) three(0.17) letters(0.40) of(0.22) english(0.42) language(0.51) |  | – |  |
| [`d26160799fa6`](d26160799fa6/) | after | Hi there, I'm chaganti | hi there, i'm chaganti | mfa | hi(0.26) there(0.33) i'm(0.25) chaganti(0.66) |  | – |  |
| [`e42915d024ff`](e42915d024ff/) | after | abc are three letters of english | abc are three letters of english | mfa | abc(0.68) are(0.32) three(0.17) letters(0.37) of(0.14) english(0.47) |  | 78 |  |
| [`18b6073bfc19`](18b6073bfc19/) | after | a b c are three letters of english | a b c are three letters of english | mfa | a(0.03⚠) b(0.36) c(0.34) are(0.31) three(0.17) letters(0.37) of(0.14) english(0.47) |  | 18 | letter_a |
| [`472bcaef6cb4`](472bcaef6cb4/) | after | I would like to introduce myself | i would like to introduce myself | mfa | i(0.15) would(0.20) like(0.30) to(0.09) introduce(0.52) myself(0.58) |  | 33 |  |
| [`b6e845d5eb57`](b6e845d5eb57/) | after | i, i, i | i, i, i | mfa | i(0.49) i(0.45) i(0.47) |  | 117 |  |
| [`3906461dbabe`](3906461dbabe/) | after | hmm, that rhythm is nice | hmm, that rhythm is nice | mfa | hmm(0.09) that(0.25) rhythm(0.53) is(0.18) nice(0.45) |  | 14 |  |
| [`1364fdf9f826`](1364fdf9f826/) | after | It costs $5.50 at 3:30pm | it costs five dollars and fifty cents at three: thirty p m | mfa | it(0.26) costs(0.51) five(0.36) dollars(0.74) and(0.22) fifty(0.29) cents(0.47) at(0.20) three(0.14) thirty(0.37) p(0.26) m(0.20) |  | 8 |  |
| [`f61c291611bc`](f61c291611bc/) | after | abc are three letters | abc are three letters | mfa | abc(0.84) are(0.36) three(0.17) letters(0.44) |  | 56 |  |
| [`82170a765ec8`](82170a765ec8/) | after | bcd are three letters | b c d are three letters | mfa | b(0.22) c(0.34) d(0.31) are(0.35) three(0.15) letters(0.61) |  | 6 |  |
| [`c3e0c3355098`](c3e0c3355098/) | after | abc are three letters | abc are three letters | mfa | abc(0.93) are(0.26) three(0.16) letters(0.55) |  | 61 |  |
| [`f955a7539628`](f955a7539628/) | after | xyz is the end | x y z is the end | mfa | x(0.27) y(0.24) z(0.20) is(0.24) the(0.10) end(0.33) |  | 9 |  |
| [`42ac9877e649`](42ac9877e649/) | after | Hi, I'm Ravi Chaganti | hi, i'm ravi chaganti | mfa | hi(0.39) i'm(0.35) ravi(0.39) chaganti(0.57) |  | 9 |  |
| [`c494473e97e6`](c494473e97e6/) | after | Meenakshi lives in Visakhapatnam | meenakshi lives in visakhapatnam | mfa | meenakshi(0.75) lives(0.41) in(0.33) visakhapatnam(0.87) |  | 34 |  |
| [`f952ff1d8c1f`](f952ff1d8c1f/) | after | ac is air conditioner | ac is air conditioner | mfa | ac(0.69) is(0.24) air(0.21) conditioner(0.62) |  | 55 |  |
| [`8a5f16c05629`](8a5f16c05629/) | after | bc means backward caste | bc means backward caste | mfa | bc(0.44) means(0.40) backward(0.49) caste(0.35) |  | 11 |  |
| [`331b15db404e`](331b15db404e/) | after | obc means other backward castes | obc means other backward castes | mfa | obc(0.25) means(1.11) other(0.29) backward(0.60) castes(0.46) |  | 51 | g2p_letters |
| [`982d8855b7fc`](982d8855b7fc/) | after | bcg means other backward gates | b c g means other backward gates | mfa | b(0.17) c(0.28) g(0.17) means(0.43) other(0.28) backward(0.58) gates(0.51) |  | 38 |  |
| [`480d4d466b41`](480d4d466b41/) | after | bcg means other backward gates | b c g means other backward gates | mfa | b(0.17) c(0.28) g(0.17) means(0.43) other(0.28) backward(0.58) gates(0.51) |  | 5 |  |
| [`d0c28266bc27`](d0c28266bc27/) | after | bcg means other backward gates | b c g means other backward gates | mfa | b(0.14) c(0.22) g(0.12) means(0.51) other(0.28) backward(0.54) gates(0.47) |  | 14 |  |
