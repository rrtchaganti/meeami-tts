"""Text-input stress test (DECISIONS D-94; `make text-stress`).

Runs every case of tests/data/text_cases.tsv (or --cases FILE, or --text "...") through the
pipeline's front end: Normalizer -> words() -> pronunciation source (lexicon / cmudict /
possessive rule / g2p). Read-only: unlike the inference pipeline it never appends OOV words to
data/lexicon/lexicon.txt. Prints one line per case:

    PASS / FAIL (expected words differ) / G2P (words that would go to g2p_en) / ERR (refused)

    python -m tts.text.stress [--cases FILE] [--text "..."] [--phones]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tts.config import REPO_ROOT, load_config, repo_path
from tts.text.lexicon import cmu_dict, load_lexicon
from tts.text.normalize import NormalizeError, Normalizer, words

CASES = REPO_ROOT / "tests" / "data" / "text_cases.tsv"


def load_cases(path: Path) -> list[tuple[str, str, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")   # the input itself may contain tabs
        out.append((parts[0], "\t".join(parts[1:-1]), parts[-1]))
    return out


def source(word: str, lex: dict, cmu: dict) -> str:
    if word in lex:
        return "lexicon"
    if word in cmu:
        return "cmudict"
    if word.endswith("'s") and word[:-2] in cmu:
        return "cmudict+s"
    return "g2p"


def run_case(norm: Normalizer, text: str, lex: dict, cmu: dict) -> dict:
    """{'normalized', 'words', 'g2p', 'error'} for one input."""
    try:
        n = norm(text)
    except NormalizeError as e:
        return {"normalized": None, "words": [], "g2p": [], "error": str(e)}
    ws = words(n)
    if not ws:
        return {"normalized": n, "words": [], "g2p": [], "error": "no words after normalization"}
    return {"normalized": n, "words": ws, "g2p": [w for w in ws if source(w, lex, cmu) == "g2p"],
            "error": None}


def verdict(r: dict, expected: str) -> str:
    if expected == "ERROR":
        return "PASS" if r["error"] else "FAIL"
    if r["error"]:
        return "ERR"
    if " ".join(r["words"]) != expected:
        return "FAIL"
    return "G2P" if r["g2p"] else "PASS"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--text", action="append", default=[], help="ad-hoc input (repeatable)")
    ap.add_argument("--phones", action="store_true", help="also print each word's phones")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "inference.yaml"))
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    norm = Normalizer(cfg)
    lex, cmu = load_lexicon(repo_path(cfg.paths.lexicon)), cmu_dict()
    cases = [("ad-hoc", t, "") for t in a.text] if a.text else load_cases(Path(a.cases))
    counts: dict[str, int] = {}
    for cat, text, expected in cases:
        r = run_case(norm, text, lex, cmu)
        v = verdict(r, expected) if expected else ("ERR" if r["error"] else
                                                   "G2P" if r["g2p"] else "OK")
        counts[v] = counts.get(v, 0) + 1
        got = r["error"] and f"refused: {r['error'][:70]}" or " ".join(r["words"])
        print(f"{v:4} [{cat}] {text!r} -> {got}"
              + (f"   (expected: {expected})" if v == "FAIL" else "")
              + (f"   g2p: {r['g2p']}" if r["g2p"] else ""))
        if a.phones and r["words"]:
            for w in r["words"]:
                pron = lex.get(w) or (cmu[w][0] if w in cmu else None)
                print(f"       {w:16} {source(w, lex, cmu):9} {' '.join(pron) if pron else '?'}")
    print(" | ".join(f"{k} {v}" for k, v in sorted(counts.items())), f"of {len(cases)}")
    return 1 if counts.get("FAIL", 0) or counts.get("ERR", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
