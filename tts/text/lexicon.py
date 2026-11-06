"""Single-pronunciation lexicon (docs/PLAN.md Section 3, "Lexicon").

`data/lexicon/lexicon.txt`: one line per word, `word<TAB>P1 P2 ...`, lowercase word, ARPAbet with
stress digits, exactly one pronunciation. Source: first pronunciation in the `cmudict` package;
OOV words from g2p_en's neural model. Built by `make lexicon` from LJSpeech + L2-ARCTIC text
(all speakers' transcripts are text, not audio, plus the suitcase corpus). At inference,
`ensure_words` G2Ps and appends OOVs before alignment. Nothing else may generate pronunciations.

    python -m tts.text.lexicon --config configs/splits.yaml
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

from omegaconf import DictConfig

from tts.config import REPO_ROOT, load_config, repo_path
from tts.text.normalize import Normalizer, words
from tts.text.symbols import PHONE_SET

Lexicon = dict[str, list[str]]


class LexiconError(ValueError):
    pass


# --- file format ---------------------------------------------------------------------------
def validate_pron(word: str, pron: list[str]) -> None:
    bad = [p for p in pron if p not in PHONE_SET]
    if not pron or bad:
        raise LexiconError(f"{word!r}: invalid pronunciation {pron} (bad phones: {bad})")


def load_lexicon(path: Path) -> Lexicon:
    lex: Lexicon = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise LexiconError(f"{path}:{n}: expected 'word<TAB>phones', got {line!r}")
        word, pron = parts[0], parts[1].split()
        if word != word.lower() or not word or " " in word:
            raise LexiconError(f"{path}:{n}: word must be lowercase, no spaces: {word!r}")
        if word in lex:
            raise LexiconError(f"{path}:{n}: duplicate word {word!r} (one pronunciation only)")
        validate_pron(word, pron)
        lex[word] = pron
    return lex


def format_lexicon(lex: Lexicon) -> str:
    return "".join(f"{w}\t{' '.join(p)}\n" for w, p in sorted(lex.items()))


# --- pronunciation sources -----------------------------------------------------------------
@lru_cache(maxsize=1)
def cmu_dict() -> dict[str, list[list[str]]]:
    import cmudict

    return cmudict.dict()


class G2P:
    """g2p_en's neural model for OOV words (letters only; apostrophes are dropped)."""

    def __init__(self, cfg: DictConfig, root: Path = REPO_ROOT):
        from tts.assets import use_local_nltk

        use_local_nltk(cfg, root)
        from g2p_en import G2p

        self._g2p = G2p()

    def __call__(self, word: str) -> list[str]:
        letters = "".join(c for c in word if "a" <= c <= "z")
        if not letters:
            raise LexiconError(f"cannot G2P {word!r}: no letters")
        pron = self._g2p.predict(letters)
        validate_pron(word, pron)
        return pron


SIBILANTS = {"S", "Z", "SH", "ZH", "CH", "JH"}
VOICELESS = {"P", "T", "K", "F", "TH"}


def possessive_suffix(base: list[str]) -> list[str]:
    last = base[-1]
    if last in SIBILANTS:
        return ["IH0", "Z"]
    return ["S"] if last in VOICELESS else ["Z"]


def pronounce(word: str, g2p: Callable[[str], list[str]],
              possessive_rule: bool = True) -> tuple[list[str], str]:
    """(pronunciation, source) with source 'cmudict', 'cmudict+s' or 'g2p'."""
    entries = cmu_dict().get(word)
    if entries:
        pron = list(entries[0])
        validate_pron(word, pron)
        return pron, "cmudict"
    if possessive_rule and word.endswith("'s") and cmu_dict().get(word[:-2]):
        base = list(cmu_dict()[word[:-2]][0])
        pron = base + possessive_suffix(base)
        validate_pron(word, pron)
        return pron, "cmudict+s"
    return g2p(word), "g2p"


def build(word_list: Iterable[str], g2p: Callable[[str], list[str]],
          possessive_rule: bool = True) -> tuple[Lexicon, dict[str, str]]:
    """Returns (lexicon, {word: source}) for every word not taken verbatim from cmudict."""
    lex: Lexicon = {}
    derived = {}
    for w in sorted(set(word_list)):
        lex[w], src = pronounce(w, g2p, possessive_rule)
        if src != "cmudict":
            derived[w] = src
    return lex, derived


def ensure_words(word_list: Iterable[str], path: Path, g2p: Callable[[str], list[str]],
                 possessive_rule: bool = True) -> list[str]:
    """Inference (Section 3): G2P words missing from the lexicon and append them. Returns added."""
    lex = load_lexicon(path) if path.exists() else {}
    added = {w: pronounce(w, g2p, possessive_rule)[0] for w in sorted(set(word_list))
             if w not in lex}
    if added:
        with open(path, "a", encoding="utf-8") as f:
            f.write(format_lexicon(added))
    return sorted(added)


# --- corpus text ---------------------------------------------------------------------------
def corpus_texts(cfg: DictConfig, root: Path = REPO_ROOT) -> dict[str, list[str]]:
    """Raw texts per source: LJ metadata column 3, L2 transcripts, suitcase transcripts."""
    from tts.guard import speakers
    from tts.splits import read_lj_metadata

    lj_dir = repo_path(cfg.paths.raw_lj, root)
    l2 = repo_path(cfg.paths.raw_l2, root)
    out = {"lj": list(read_lj_metadata(lj_dir / cfg.lj.metadata).values()), "l2": [],
           "suitcase": []}
    for spk in speakers(cfg, "dev") + speakers(cfg, "test"):
        for p in sorted((l2 / spk / cfg.l2.transcript_dir).glob("*.txt")):
            out["l2"].append(p.read_text(encoding="utf-8"))
    suit = l2 / cfg.l2.suitcase_dir / cfg.l2.transcript_dir
    if suit.is_dir():
        out["suitcase"] = [p.read_text(encoding="utf-8") for p in sorted(suit.glob("*.txt"))]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build data/lexicon/lexicon.txt (make lexicon)")
    ap.add_argument("--config", default="configs/splits.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(repo_path(args.config), args.overrides)
    root = REPO_ROOT

    from tts.align.mfa import acoustic_model_phones

    norm = Normalizer(cfg)
    texts = corpus_texts(cfg, root)
    counts: dict[str, Counter] = {}
    for src, items in texts.items():
        c = Counter()
        for t in items:
            c.update(words(norm(t)))
        counts[src] = c
    all_words = set().union(*counts.values())
    lex, derived = build(all_words, G2P(cfg, root), cfg.lexicon.possessive_rule)
    oov = sorted(w for w, s in derived.items() if s == "g2p")
    poss = sorted(w for w, s in derived.items() if s == "cmudict+s")

    model_phones = acoustic_model_phones(cfg, root)
    used = {p for pron in lex.values() for p in pron}
    missing = sorted(used - model_phones)
    if missing:
        raise LexiconError(f"lexicon phones missing from {cfg.mfa.acoustic_model.name}: {missing}")

    path = repo_path(cfg.paths.lexicon, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_lexicon(lex), encoding="utf-8")
    load_lexicon(path)  # round-trip validation

    rep = [f"# Lexicon (T3, `make lexicon`, {date.today().isoformat()})", "",
           f"- `{cfg.paths.lexicon}`: {len(lex)} words, one pronunciation each",
           f"- From `cmudict` (first pronunciation): {len(lex) - len(derived)}; possessives "
           f"from cmudict base + suffix: {len(poss)}; OOV via g2p_en: {len(oov)}",
           f"- Phones used: {len(used)} of {len(PHONE_SET)}; all in the "
           f"`{cfg.mfa.acoustic_model.name}` {cfg.mfa.acoustic_model.version} phone set "
           f"({len(model_phones)} phones)"]
    unused = sorted(PHONE_SET - used)
    if unused:
        rep.append(f"- Phones never used: {', '.join(unused)}")
    rep += ["", "| Source | Texts | Word tokens | Word types | OOV types |", "| --- | --- | --- | --- | --- |"]
    oov_set = set(oov)
    for src, c in counts.items():
        rep.append(f"| {src} | {len(texts[src])} | {sum(c.values())} | {len(c)} | "
                   f"{len(set(c) & oov_set)} |")
    rep += ["", "## OOV words (g2p_en) — review list", "",
            "| Word | Pronunciation | Sources |", "| --- | --- | --- |"]
    for w in oov:
        srcs = ", ".join(s for s, c in counts.items() if w in c)
        rep.append(f"| {w} | {' '.join(lex[w])} | {srcs} |")
    rep += ["", "## Possessives from cmudict base + suffix", "",
            "| Word | Pronunciation |", "| --- | --- |"]
    rep += [f"| {w} | {' '.join(lex[w])} |" for w in poss]
    report = repo_path(cfg.paths.reports, root) / "lexicon.md"
    report.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep[:8]))
    print(f"report: {report.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
