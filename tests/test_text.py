"""T3: normalizer (Section 2) and symbols (Section 4, change 3)."""
import re

import pytest

from tts.config import REPO_ROOT, load_config
from tts.text import symbols
from tts.text.normalize import (
    NormalizeError, Normalizer, cardinal, ordinal, words, words_with_punct, year,
)

CFG = load_config(REPO_ROOT / "configs" / "base.yaml")


@pytest.fixture(scope="module")
def norm():
    return Normalizer(CFG)


def check(norm, cases):
    for src, want in cases:
        assert norm(src) == want, f"{src!r}"


# --- numbers -------------------------------------------------------------------------------
def test_number_words():
    assert cardinal(0) == "zero"
    assert cardinal(1234) == "one thousand two hundred thirty four"
    assert cardinal(-7) == "minus seven"
    assert ordinal(21) == "twenty first" and ordinal(100) == "one hundredth"
    assert year(1999) == "nineteen ninety nine" and year(2005) == "two thousand five"
    assert year(2019) == "twenty nineteen" and year(1900) == "nineteen hundred"
    assert year(1805) == "eighteen oh five"


def test_numbers(norm):
    check(norm, [
        ("He had 3 dogs.", "he had three dogs."),
        ("2,000,000 people", "two million people"),
        ("pi is 3.14", "pi is three point one four"),
        ("It is -5 now", "it is minus five now"),
        ("agent 007", "agent oh oh seven"),
        ("pages 10-20", "pages ten to twenty"),
        ("45% of them", "forty five percent of them"),
        ("1/2 and 3/8", "one half and three eighths"),
        ("#1 hit", "number one hit"),
        ("No. 5", "number five"),
        ("in 1999 and 1500 men", "in nineteen ninety nine and fifteen hundred men"),
        ("in 1066", "in one thousand sixty six"),  # below configured year_range
    ])


def test_ordinals(norm):
    check(norm, [("the 1st, 2nd, 3rd and 21st", "the first, second, third and twenty first"),
                 ("the 100th day", "the one hundredth day")])


def test_decades(norm):
    check(norm, [("the 1960s", "the nineteen sixties"), ("the '80s", "the eighties"),
                 ("the 1900s", "the nineteen hundreds")])


# --- dates and times -----------------------------------------------------------------------
def test_dates(norm):
    check(norm, [
        ("on January 5, 2020.", "on january fifth twenty twenty."),
        ("on Jan. 5th 2020", "on january fifth twenty twenty"),
        ("on Sept. 3", "on september third"),
        ("on 5 January 2021", "on the fifth of january twenty twenty one"),
        ("on 12/25/1963", "on december twenty fifth nineteen sixty three"),
        ("on 2021-03-04", "on march fourth twenty twenty one"),
        ("may I", "may i"),  # month word without a day is left alone
    ])


def test_times(norm):
    check(norm, [
        ("at 3:30 p.m. Then", "at three thirty p m. then"),
        ("at 10:05", "at ten oh five"),
        ("at 9:00", "at nine o'clock"),
        ("at 7am. It", "at seven a m. it"),
    ])


# --- currency ------------------------------------------------------------------------------
def test_currency(norm):
    check(norm, [
        ("$1,234.56", "one thousand two hundred thirty four dollars and fifty six cents"),
        ("$1", "one dollar"),
        ("$0.50", "fifty cents"),
        ("$5.01", "five dollars and one cent"),
        ("$2.5 million", "two point five million dollars"),
        ("£3", "three pounds"),
        ("€20", "twenty euros"),
        ("50¢", "fifty cents"),
    ])


# --- abbreviations, initialisms, symbols ---------------------------------------------------
def test_abbreviations(norm):
    check(norm, [
        ("Mr. Smith and Mrs. Jones met Dr. Who.", "mister smith and missus jones met doctor who."),
        ("St. Louis on Elm St. today", "saint louis on elm street today"),
        ("apples, pears, etc. Then", "apples, pears, et cetera. then"),
        ("cats, i.e. felines, e.g. Tom", "cats, that is felines, for example tom"),
        ("Lt. Col. Smith vs. Gen. Lee", "lieutenant colonel smith versus general lee"),
        ("the U.S. and the F.B.I.", "the u s and the f b i."),
        ("We met Mr. Smith.", "we met mister smith."),
    ])


def test_symbols_and_punctuation(norm):
    check(norm, [
        ("salt & pepper @ home", "salt and pepper at home"),
        ('He said, "don\'t" -- the boys\' toys...', "he said, don't the boys toys."),
        ("Wait!! Really??", "wait! really?"),
        ("café naïve “quoted” it’s", "cafe naive quoted it's"),
        ("Philip`s lips ‘n’ it´s", "philip's lips n it's"),
        ("well-known x-ray", "well known x ray"),
        ("no punctuation here", "no punctuation here"),
    ])


def test_output_alphabet(norm):
    out = norm("Dr. O'Brien paid $3.50 on 1/2/2003 at 5:15 p.m. (45%) & left; F.B.I. #2!")
    assert re.fullmatch(r"[a-z' .,?!;:]+", out), out
    assert not re.search(r"\d", out)


def test_idempotent_on_lj_style_text(norm):
    t = "Printing, in the only sense with which we are at present concerned, differs."
    assert norm(norm(t)) == norm(t)


def test_words_helpers(norm):
    n = norm("Hello, world. How's it going")
    assert words(n) == ["hello", "world", "how's", "it", "going"]
    assert words_with_punct(n) == [("hello", True), ("world", True), ("how's", False),
                                   ("it", False), ("going", False)]


def test_unhandled_digits_raise():
    n = Normalizer(load_config(REPO_ROOT / "configs" / "base.yaml"))
    n._numbers = lambda text: text  # simulate a number pattern the normalizer misses
    with pytest.raises(NormalizeError):
        n("item 42")


# --- symbols -------------------------------------------------------------------------------
def test_symbols():
    assert symbols.N_VOCAB == 72
    assert symbols.SYMBOLS[:3] == ["pad", "sil", "sp"]
    assert len(symbols.PHONES) == 69 and len(set(symbols.SYMBOLS)) == 72
    assert symbols.to_ids(["sil", "AA0", "ZH"]) == [1, 3, 71]
    with pytest.raises(KeyError):
        symbols.to_ids(["UW"])  # unstressed-less vowel is never silently mapped


def test_punctuation_and_case_do_not_change_words(norm):
    """Test-time text: punctuation and case never reach the model (docs/DECISIONS.md D-74)."""
    t = "A cry of joy burst Philip's lips"
    for v in [t.upper(), t.lower(), t + ".", '"' + t + '!"', t.replace(" ", ", "),
              t.replace("'", "’"), t.replace("'", "`"), t.replace(" ", "  \t")]:
        assert words(norm(v)) == words(norm(t)), v
