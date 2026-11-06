"""The single text normalizer (docs/PLAN.md Section 2).

Used for training text, MFA `.lab` files and inference input. Handles currency, dates, times,
decades, ordinals, percentages, fractions, ranges, numbers, years, abbreviations, initialisms and
symbols. Output: lowercase words (letters + internal apostrophes) separated by single spaces, with
sentence punctuation (. , ? ! ; :) kept attached to the preceding word. `words()` gives the word
list used for `.lab` files and lexicon lookup; `words_with_punct()` adds a punctuation-follows
flag per word (Ablation A, Section 6 step 4).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import inflect
from omegaconf import DictConfig, OmegaConf
from unidecode import unidecode

_INF = inflect.engine()

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
_MONTH_ABBR = {m[:3]: m for m in MONTHS} | {"sept": "september"}
_MONTH_RE = (r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
             r"aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
PUNCT = ".,?!;:"
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")


class NormalizeError(ValueError):
    """Text that the normalizer cannot turn into words (e.g. a digit survives)."""


# --- number words --------------------------------------------------------------------------
def _clean(s: str) -> str:
    return " ".join(s.replace("-", " ").replace(",", " ").split())


def cardinal(n: int) -> str:
    if n < 0:
        return "minus " + cardinal(-n)
    return _clean(_INF.number_to_words(n, andword="", zero="zero"))


def ordinal(n: int) -> str:
    return _clean(_INF.ordinal(_INF.number_to_words(n, andword="", zero="zero")))


def year(y: int) -> str:
    if 2000 <= y <= 2009:
        return cardinal(y)
    hi, lo = divmod(y, 100)
    if lo == 0:
        return f"{cardinal(hi)} hundred"
    if lo < 10:
        return f"{cardinal(hi)} oh {cardinal(lo)}"
    return f"{cardinal(hi)} {cardinal(lo)}"


def digits(s: str) -> str:
    return " ".join("oh" if c == "0" else cardinal(int(c)) for c in s)


def decimal(s: str) -> str:
    whole, frac = s.split(".")
    return f"{cardinal(int(whole or 0))} point {' '.join(cardinal(int(c)) for c in frac)}"


def _plural(word_phrase: str) -> str:
    """'nineteen sixty' -> 'nineteen sixties'; 'nineteen hundred' -> 'nineteen hundreds'."""
    head, _, last = word_phrase.rpartition(" ")
    last = last[:-1] + "ies" if last.endswith("y") else last + "s"
    return f"{head} {last}".strip()


_SENT_END = r"(?=\s*$|\s+[A-Z\"'(])"  # a following capital or end of text: the dot also ends a sentence


def _end_dot(text: str, m: re.Match) -> str:
    """'.' if the abbreviation's own period also ends the sentence, else ''."""
    return "." if m.group(0).endswith(".") and re.match(_SENT_END, text[m.end():]) else ""


# --- normalizer ----------------------------------------------------------------------------
@dataclass(frozen=True)
class _Tables:
    year_lo: int
    year_hi: int
    abbreviations: tuple
    phrases: tuple
    symbols: tuple
    currency: tuple


@lru_cache(maxsize=8)
def _tables(cfg_yaml: str) -> _Tables:
    t = OmegaConf.create(cfg_yaml)
    return _Tables(int(t.year_range[0]), int(t.year_range[1]),
                   tuple((k, v) for k, v in t.abbreviations.items()),
                   tuple((k, v) for k, v in t.phrases.items()),
                   tuple((k, v) for k, v in t.symbols.items()),
                   tuple((k, tuple(v)) for k, v in t.currency.items()))


class Normalizer:
    def __init__(self, cfg: DictConfig):
        """`cfg` is a full config (uses `cfg.text`) or the `text` node itself."""
        node = cfg.text if "text" in cfg else cfg
        self.t = _tables(OmegaConf.to_yaml(node))

    # number-like spans --------------------------------------------------------------------
    def _number(self, s: str) -> str:
        s = s.replace(",", "")
        if "." in s:
            return decimal(s)
        if len(s) > 1 and s.startswith("0"):
            return digits(s)
        n = int(s)
        if self.t.year_lo <= n <= self.t.year_hi and len(s) == 4:
            return year(n)
        return cardinal(n)

    def _currency(self, text: str) -> str:
        for sym, (one, many, sub1, subn) in self.t.currency:
            pat = re.compile(re.escape(sym) + r"\s?(\d[\d,]*)(?:\.(\d{1,2}))?"
                             r"(?:\s?(thousand|million|billion|trillion)\b)?", re.I)

            def rep(m, one=one, many=many, sub1=sub1, subn=subn):
                whole = int(m.group(1).replace(",", ""))
                cents = int((m.group(2) or "0").ljust(2, "0"))
                if m.group(3):  # "$2.5 million" -> "two point five million dollars"
                    amt = m.group(1) + ("." + m.group(2) if m.group(2) else "")
                    return f" {self._number(amt)} {m.group(3).lower()} {many} "
                parts = []
                if whole or not cents:
                    parts.append(f"{cardinal(whole)} {one if whole == 1 else many}")
                if cents:
                    parts.append(f"{cardinal(cents)} {sub1 if cents == 1 else subn}")
                return " " + " and ".join(parts) + " "
            text = pat.sub(rep, text)
        text = re.sub(r"(\d+)\s?¢", lambda m: f" {cardinal(int(m.group(1)))} cents ", text)
        return text

    def _dates(self, text: str) -> str:
        def month(m):
            k = m.group("month").lower().rstrip(".")
            return _MONTH_ABBR.get(k[:4] if k.startswith("sept") else k[:3])

        # "January 5, 2020" / "Jan. 5th 2020" / "January 5"
        pat = re.compile(_MONTH_RE + r"\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(?P<y>\d{4})\b)?",
                         re.I)
        text = pat.sub(lambda m: f" {month(m)} {ordinal(int(m.group('d')))}"
                       + (f" {year(int(m.group('y')))}" if m.group("y") else "") + " ", text)
        # "5 January 2020" / "5th of January"
        pat = re.compile(r"\b(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH_RE
                         + r"(?:,?\s+(?P<y>\d{4})\b)?", re.I)
        text = pat.sub(lambda m: f" the {ordinal(int(m.group('d')))} of {month(m)}"
                       + (f" {year(int(m.group('y')))}" if m.group("y") else "") + " ", text)
        # ISO 2020-01-05
        def iso(m):
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if not (1 <= mo <= 12 and 1 <= d <= 31):
                return m.group(0)
            return f" {MONTHS[mo - 1]} {ordinal(d)} {year(y)} "
        text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", iso, text)

        # US numeric 12/25/2020 or 12/25/20
        def us(m):
            mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
            if not (1 <= mo <= 12 and 1 <= d <= 31):
                return m.group(0)
            yy = year(int(y)) if len(y) == 4 else digits(y) if y.startswith("0") else cardinal(int(y))
            return f" {MONTHS[mo - 1]} {ordinal(d)} {yy} "
        return re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b", us, text)

    def _times(self, text: str) -> str:
        ampm = r"(?:\s?(?P<ap>[ap])\.?\s?m\.?(?![a-z]))?"

        def rep(m):
            h, mi = int(m.group("h")), int(m.group("m"))
            if h > 23 or mi > 59:
                return m.group(0)
            ap = m.group("ap")
            if mi == 0:
                s = f"{cardinal(h)}" + ("" if ap else " o'clock")
            elif mi < 10:
                s = f"{cardinal(h)} oh {cardinal(mi)}"
            else:
                s = f"{cardinal(h)} {cardinal(mi)}"
            return f" {s}" + (f" {ap.lower()} m{_end_dot(text, m)} " if ap else " ")
        text = re.sub(r"\b(?P<h>\d{1,2}):(?P<m>\d{2})\b" + ampm, rep, text, flags=re.I)
        return re.sub(r"\b(\d{1,2})\s?([ap])\.?\s?m\.?(?![a-z])",
                      lambda m: f" {cardinal(int(m.group(1)))} {m.group(2).lower()} m"
                      f"{_end_dot(text, m)} ", text, flags=re.I)

    def _abbreviations(self, text: str) -> str:
        for k, v in self.t.phrases:
            # always followed by more text, so only a dot at the very end ends a sentence
            text = re.sub(r"(?<![a-z])" + re.escape(k), lambda m, v=v: f" {v}" + (
                "." if re.match(r"\s*$", text[m.end():]) else "") + " ", text, flags=re.I)
        # "St." -> saint before a capitalised word, otherwise street
        text = re.sub(r"\bSt\.(?=\s+[A-Z])", "Saint", text)
        text = re.sub(r"\bst\.", "street", text, flags=re.I)
        text = re.sub(r"\bno\.(?=\s?\d)", "number", text, flags=re.I)
        for k, v in self.t.abbreviations:
            # titles (mr., dr. ...) are followed by capitalised names, so only a dot at the very
            # end of the text is a sentence end; "etc." also ends a sentence before a capital
            end = _SENT_END if k == "etc" else r"(?=\s*$)"
            text = re.sub(rf"\b{re.escape(k)}\.", lambda m, v=v, end=end: v + (
                "." if re.match(end, text[m.end():]) else ""), text, flags=re.I)
        # dotted initialisms: "U.S." -> "u s", "F.B.I." -> "f b i"
        return re.sub(r"\b((?:[A-Za-z]\.){2,})",
                      lambda m: " " + " ".join(m.group(1).replace(".", "")) + _end_dot(text, m)
                      + " ", text)

    def _numbers(self, text: str) -> str:
        # decades: 1960s, '60s
        text = re.sub(r"\b(\d{2})(\d{2})'?s\b", lambda m: " " + _plural(year(int(m.group(0)[:4])))
                      + " ", text)
        text = re.sub(r"'(\d)0s\b", lambda m: " " + _plural(cardinal(int(m.group(1)) * 10)) + " ",
                      text)
        text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: f" {ordinal(int(m.group(1)))} ", text,
                      flags=re.I)
        text = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s?%", lambda m: f" {self._number(m.group(1))} percent ",
                      text)

        def frac(m):
            a, b = int(m.group(1)), int(m.group(2))
            if b == 0:
                return f" {cardinal(a)} over zero "
            if b == 2:
                den = "half" if a == 1 else "halves"
            elif b == 4:
                den = "quarter" if a == 1 else "quarters"
            else:
                den = ordinal(b) + ("" if a == 1 else "s")
            return f" {cardinal(a)} {den} "
        text = re.sub(r"\b(\d+)/(\d+)\b", frac, text)
        text = re.sub(r"#\s?(\d)", r" number \1", text)
        text = re.sub(r"(?<![\w.])-(?=\d)", " minus ", text)
        text = re.sub(r"\b(\d[\d,]*(?:\.\d+)?)\s?[-–]\s?(\d[\d,]*(?:\.\d+)?)\b",
                      lambda m: f" {self._number(m.group(1))} to {self._number(m.group(2))} ", text)
        return re.sub(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+",
                      lambda m: f" {self._number(m.group(0))} ", text)

    def __call__(self, text: str) -> str:
        text = self._currency(text)          # before unidecode: it would turn £ into "PS"
        for k, v in self.t.symbols:
            text = text.replace(k, f" {v} ")
        text = unidecode(text)
        text = text.replace("’", "'")
        text = self._dates(text)
        text = self._times(text)
        text = self._abbreviations(text)
        text = self._numbers(text)
        if re.search(r"\d", text):
            raise NormalizeError(f"digits left after normalization: {text!r}")
        text = text.lower()
        text = re.sub(r"[-–—/_]+", " ", text)
        text = re.sub(r"[^a-z' " + re.escape(PUNCT) + "]", " ", text)
        # apostrophes: keep only word-internal ones (don't, o'clock); drop quotes / trailing
        text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)
        # punctuation: attach to the preceding word, collapse runs ("..." -> ".")
        text = re.sub(r"\s*([" + re.escape(PUNCT) + r"])[\s" + re.escape(PUNCT) + r"]*",
                      lambda m: m.group(1) + " ", text)
        text = re.sub(r"^\s*[" + re.escape(PUNCT) + r"]+", " ", text)
        return " ".join(text.split())


def words(normalized: str) -> list[str]:
    """Words of normalized text (`.lab` content and lexicon keys)."""
    return WORD_RE.findall(normalized)


def words_with_punct(normalized: str) -> list[tuple[str, bool]]:
    """(word, punctuation follows) for each word of normalized text."""
    out = []
    for tok in normalized.split():
        w = WORD_RE.findall(tok)
        if not w:
            continue
        out.extend((x, False) for x in w[:-1])
        out.append((w[-1], tok[-1] in PUNCT))
    return out
