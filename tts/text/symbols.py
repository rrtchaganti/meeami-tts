"""Model symbol set (docs/PLAN.md Section 4, change 3).

`pad`, `sil`, `sp`, then 69 ARPAbet tokens: 15 vowels x 3 stress levels + 24 consonants.
n_vocab = 72. The 69 phones equal the phone set of MFA's english_us_arpa acoustic model
(checked by `make lexicon`).
"""
PAD, SIL, SP = "pad", "sil", "sp"
SPECIAL = [PAD, SIL, SP]

VOWELS = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
CONSONANTS = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S",
              "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"]
STRESSES = ["0", "1", "2"]

PHONES = [v + s for v in VOWELS for s in STRESSES] + CONSONANTS
SYMBOLS = SPECIAL + PHONES
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
N_VOCAB = len(SYMBOLS)
PHONE_SET = frozenset(PHONES)

assert len(VOWELS) == 15 and len(CONSONANTS) == 24 and len(PHONES) == 69 and N_VOCAB == 72


def to_ids(tokens: list[str]) -> list[int]:
    """Map tokens to ids; raises KeyError on an unknown symbol (never silently mapped)."""
    return [SYMBOL_TO_ID[t] for t in tokens]
