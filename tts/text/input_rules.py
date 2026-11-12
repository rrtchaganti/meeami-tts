"""Inference-input pre-pass of the normalizer (DECISIONS D-94).

The training text (LJ metadata, L2-ARCTIC prompts) has no digits, URLs, units, codes or bare
abbreviations, so the original normalizer never had to choose between readings that typed input
needs: `12/03` (a date, not "twelve thirds"), `Dr Smith`, `10km`, `GPU`, `World War II`, a phone
number, `₹500`. Each rule here rewrites such a form into plain words, or into a form the main
normalizer already handles, before the main rules run. Tables live in `text.input_rules`
(configs/base.yaml).

Guarantee (tests/test_text_stress.py): applied to every LJ, L2-ARCTIC and suitcase transcript,
the pre-pass changes no word sequence except one LJ text whose bare "Dr" was read as "drive"
(a bug it fixes); the frozen alignments, checkpoints and results stay valid. Rules adapted from
the tutor workspace audit (meeami-learn notes/INPUT_PREPROCESSING.md).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from tts.text.normalize import MONTHS, NormalizeError, _plural, cardinal, digits, ordinal

_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"


@lru_cache(maxsize=1)
def _cmu() -> frozenset:
    from tts.text.lexicon import cmu_dict

    return frozenset(cmu_dict())


def _num_words(s: str) -> str:
    """Integer or decimal string -> words (no year reading)."""
    s = s.replace(",", "")
    if "." in s:
        w, f = s.split(".")
        return f"{cardinal(int(w or 0))} point {' '.join(cardinal(int(c)) for c in f)}"
    return cardinal(int(s))


def _indian(n: int) -> str:
    """100000 -> one lakh; 25000000 -> two crore fifty lakh."""
    crore, rest = divmod(n, 10_000_000)
    lakh, rest = divmod(rest, 100_000)
    out = ([f"{cardinal(crore)} crore"] if crore else []) + \
          ([f"{cardinal(lakh)} lakh"] if lakh else []) + ([cardinal(rest)] if rest else [])
    return " ".join(out)


_INDIAN_GROUPS = r"\d{1,2}(?:,\d\d)+,\d{3}"


def check_script(text: str) -> None:
    """The model has an English phone set: refuse non-Latin letters instead of transliterating
    them (unidecode turns Devanagari into vowel-less consonant strings)."""
    bad = sorted({c for c in text if c.isalpha() and "LATIN" not in unicodedata.name(c, "")})
    if bad:
        raise NormalizeError(f"non-Latin letters {''.join(bad)!r}: give English text")


def _slash(m: re.Match) -> str:
    """Bare a/b without a year: US M/D date when it cannot be a common fraction, else unchanged."""
    sa, sb = m.group(1), m.group(2)
    a, b = int(sa), int(sb)
    lead0 = (len(sa) == 2 and sa[0] == "0") or (len(sb) == 2 and sb[0] == "0")
    if 1 <= a <= 12 and 1 <= b <= 31 and (lead0 or a >= b or b > 12):
        return f" {MONTHS[a - 1]} {ordinal(b)} "
    return m.group(0)


class InputRules:
    def __init__(self, r: dict):
        self.fixed = [(re.compile(p), v) for p, v in r["fixed"].items()]
        self.tld = "|".join(r["tlds"])
        self.money = [self._money_rule(p, *v) for p, v in r["money"].items()]
        self.units = [(re.compile(rf"(\d(?:[\d,]*\d)?(?:\.\d+)?)\s?(?:{u})\b", re.I), one, many)
                      for u, (one, many) in r["units"].items()]
        self.weekdays = {k.lower(): v for k, v in r["weekdays"].items()}
        self.bare_titles = r["bare_titles"]
        self.bare_words = r["bare_words"]
        self.roman = r["roman"]
        self.roman_ctx = "|".join(re.escape(c) for c in r["roman_cardinal_after"])
        self.spell_max = int(r["spell_max_letters"])
        self.keep_caps = {w.lower() for w in r["keep_caps"]}
        self.spell_words = {w.lower() for w in r["spell_words"]}

    @staticmethod
    def _money_rule(sym: str, one: str, many: str, sub1: str, subn: str):
        pat = re.compile(rf"(?:{sym})\s?(\d[\d,]*)(?:\.(\d{{1,2}}))?\b"
                         r"(?:\s?(lakh|crore|thousand|million|billion)\b)?", re.I)

        def rep(m):
            whole, cents = m.group(1), m.group(2)
            if m.group(3):
                amount = _num_words(whole + ("." + cents if cents else ""))
                return f" {amount} {m.group(3).lower()} {many} "
            if re.fullmatch(_INDIAN_GROUPS, whole):
                return f" {_indian(int(whole.replace(',', '')))} {many} "
            w = int(whole.replace(",", ""))
            parts = [f"{cardinal(w)} {one if w == 1 else many}"] if (w or not cents) else []
            if cents:
                c = int(cents.ljust(2, "0"))
                parts.append(f"{cardinal(c)} {sub1 if c == 1 else subn}")
            return " " + " and ".join(parts) + " "
        return pat, rep

    def _known(self, w: str) -> bool:
        return w.lower() in _cmu() or w.lower() in self.keep_caps

    def __call__(self, text: str) -> str:
        check_script(text)
        for p, v in self.fixed:
            text = p.sub(f" {v} ", text)
        # URLs and e-mails: "dot", "slash"; the scheme is not read
        text = re.sub(r"\bhttps?://", " ", text, flags=re.I)
        text = re.sub(r"\bwww\.", " W W W dot ", text, flags=re.I)
        text = re.sub(rf"(?<=[A-Za-z0-9])\.(?=(?:{self.tld})\b)", " dot ", text, flags=re.I)
        text = re.sub(r"(?<=[A-Za-z])/(?=[A-Za-z])", " slash ", text)
        # phone numbers digit by digit (before ranges turn 123-4567 into "to")
        text = re.sub(r"(?<![\d])(?:\+(\d{1,3})[\s-]?)?\(?(\d{3})\)?[\s.-](\d{3})[\s.-](\d{4})(?!\d)",
                      lambda m: (" plus " + digits(m.group(1)) if m.group(1) else "") + " "
                      + " ".join(digits(g) for g in m.group(2, 3, 4)) + " ", text)
        # currencies the main table lacks, incl. Indian digit grouping
        for pat, rep in self.money:
            text = pat.sub(rep, text)
        text = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s?USD\b",
                      lambda m: f" {_num_words(m.group(1))} dollars ", text)
        text = re.sub(rf"\b{_INDIAN_GROUPS}\b",
                      lambda m: f" {_indian(int(m.group(0).replace(',', '')))} ", text)
        # versions 2.0.1, leading-dot decimals .5, ratings 3.5/5
        text = re.sub(r"\b(\d+(?:\.\d+){2,})\b",
                      lambda m: " " + " point ".join(digits(p) if len(p) > 1 and p[0] == "0"
                                                     else cardinal(int(p))
                                                     for p in m.group(1).split(".")) + " ", text)
        text = re.sub(r"(?<![\w.])\.(\d+)\b",
                      lambda m: " point " + " ".join(cardinal(int(c)) for c in m.group(1)) + " ",
                      text)
        text = re.sub(r"\b(\d+\.\d+)/(\d+(?:\.\d+)?)\b(?!/)|\b(\d+)/(\d+\.\d+)\b(?!/)",
                      lambda m: f" {_num_words(m.group(1) or m.group(3))} out of "
                                f"{_num_words(m.group(2) or m.group(4))} ", text)
        # bare M/D dates vs fractions (12/03 -> december third; 3/4 stays a fraction)
        text = re.sub(r"(?<![/\d])(\d{1,2})/(\d{1,2})(?![/\d])", _slash, text)
        # "the 5th of May": the main date rule would add a second "the"
        text = re.sub(rf"\b(the)\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+(?={_MON})",
                      lambda m: f"{m.group(1)} {ordinal(int(m.group(2)))} of ", text, flags=re.I)
        # durations, ratios, temperatures, units, 10k, 3x
        text = re.sub(r"(\d)\s?h\s+(\d+)\s?m\b", r"\1 hours \2 minutes", text)
        text = re.sub(r"\b(\d+):(\d)\b(?!\d)", r"\1 to \2", text)
        text = re.sub(r"°\s?C\b", " degrees celsius ", text)
        text = re.sub(r"°\s?F\b", " degrees fahrenheit ", text)
        for pat, one, many in self.units:
            text = pat.sub(lambda m, one=one, many=many:
                           f"{m.group(1)} {one if m.group(1) == '1' else many} ", text)
        text = re.sub(r"(\d)\s?[kK]\b", r"\1 thousand", text)
        text = re.sub(r"(\d)\s?[xX×]\b", r"\1 times", text)
        text = re.sub(r"(\d)\s?[xX×]\s?(?=\d)", r"\1 times ", text)
        # decades without a century ('90s is already handled)
        text = re.sub(r"(?<![\d'])\b([1-9]0)s\b",
                      lambda m: " " + _plural(cardinal(int(m.group(1)))) + " ", text)
        # undotted titles before a name, bare abbreviations, weekdays before a date, eras
        for k, v in self.bare_titles.items():
            # case-sensitive: "Dr Smith" only, so "elm dr" and "Elm Dr" (drive) stay
            text = re.sub(rf"\b{k.capitalize()}\b(?!\.)(?=\s+[A-Z])", v.capitalize(), text)
        for k, v in self.bare_words.items():
            text = re.sub(rf"\b{k}\b\.?", v, text, flags=re.I)
        wd = "|".join(sorted(self.weekdays, key=len, reverse=True))
        text = re.sub(rf"\b({wd})\.?,?(?=\s+(?:\d|{_MON}))",
                      lambda m: self.weekdays[m.group(1).lower()], text, flags=re.I)
        text = re.sub(r"(\d)\s?(AD|BC|BCE|CE)\b", lambda m: m.group(1) + " " + " ".join(m.group(2)),
                      text)
        # Roman numerals: "World War II" -> two; "Henry VIII" -> the eighth
        rom = "|".join(sorted(self.roman, key=len, reverse=True))
        text = re.sub(rf"\b({self.roman_ctx})\s+({rom})\b",
                      lambda m: f"{m.group(1)} {cardinal(self.roman[m.group(2)])}", text)
        rom1 = "|".join(r for r in sorted(self.roman, key=len, reverse=True) if len(r) > 1)
        text = re.sub(rf"\b([A-Z][a-z]+)\s+({rom1})\b(?!\.)",
                      lambda m: f"{m.group(1)} the {ordinal(self.roman[m.group(2)])}", text)

        # CamelCase names not in cmudict: ChatGPT -> Chat GPT, PyTorch -> Py Torch
        def camel(m):
            w = m.group(0)
            if self._known(w):
                return w
            return " ".join(re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+", w))
        text = re.sub(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b", camel, text)

        # all-caps tokens: spell unknown short or vowel-less ones (GPU, TTS, MP3) and spell_words
        def caps(m):
            w = m.group(1)
            if w.lower() in self.spell_words or not (
                    self._known(w) or (len(w) > self.spell_max and re.search("[AEIOU]", w))):
                return " " + " ".join(w) + " "
            return w
        text = re.sub(r"\b([A-Z]{2,5})(?=\d|\b)", caps, text)

        # lowercase tokens of 3+ letters without a vowel letter that are not words (bcd, xyz): spell
        # them instead of letting G2P invent a word (E-35). "hmm", "shh", "nth" are in cmudict;
        # 3+ letters keeps false starts like "cl" in L2 transcripts unchanged.
        def lower(m):
            w = m.group(1)
            return w if self._known(w) else " " + " ".join(w) + " "
        return re.sub(r"(?<![\w'])([b-df-hj-np-tv-z]{3,5})(?![\w'])", lower, text)
