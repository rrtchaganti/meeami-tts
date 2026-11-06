#!/usr/bin/env python
"""T2 `make data`: verify datasets, fetch VITS LJ filelists, build splits, eval IDs, manifests.

docs/PLAN.md Sections 2 and 10. Usage:
    python scripts/prepare_data.py --config configs/splits.yaml [--force] [key=value ...]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# scripts/ must not be importable (scripts/profile.py would shadow the stdlib `profile`)
sys.path[0] = str(Path(__file__).resolve().parents[1])

from tts.config import REPO_ROOT, load_config, repo_path  # noqa: E402
from tts.guard import freeze_tag_exists  # noqa: E402
from tts.splits import (  # noqa: E402
    check_lj, common_annotated_ids, fetch_filelists, l2_inventory, l2_row, lj_row, write_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/splits.yaml")
    ap.add_argument("--force", action="store_true", help="overwrite non-empty manifests")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args(argv)
    cfg = load_config(repo_path(args.config), args.overrides)
    root = REPO_ROOT
    log: list[str] = []

    def say(msg: str = "") -> None:
        print(msg)
        log.append(msg)

    # LJSpeech
    filelists = fetch_filelists(cfg, root)
    lj = check_lj(cfg, filelists, root)
    say(f"## LJSpeech\n\n- {cfg.lj.n_clips} clips; every file is "
        f"{cfg.lj.expect.sample_rate} Hz, {cfg.lj.expect.channels} ch, {cfg.lj.expect.subtype}")
    say(f"- VITS filelists @ {cfg.lj.filelists.commit[:7]} (sha256 verified): " + ", ".join(
        f"{s} {len(ids)}" for s, ids in lj["split_ids"].items()) + "; disjoint, union = metadata")

    # L2-ARCTIC
    inv = l2_inventory(cfg, root)
    eval_ids = {s: common_annotated_ids(inv, s) for s in ("dev", "test")}
    say("\n## L2-ARCTIC\n")
    say("| Split | Accent | Speaker | wav | annotation |\n| --- | --- | --- | --- | --- |")
    for si in inv.values():
        say(f"| {si.split} | {si.accent} | {si.speaker} | {len(si.wav)} | {len(si.annotation)} |")
    say("")
    for s, ids in eval_ids.items():
        n_spk = sum(si.split == s for si in inv.values())
        say(f"- **{s} evaluation set:** {len(ids)} utterance IDs annotated for all {n_spk} "
            f"speakers -> {len(ids) * n_spk} reference utterances")
    for s, ids in eval_ids.items():
        # IDs that every other speaker of the split has but this one lacks (what shrinks the set)
        group = [si for si in inv.values() if si.split == s]
        for si in group:
            others = set.intersection(*(o.annotation for o in group if o is not si))
            lost = sorted(others - si.annotation)
            if lost:
                say(f"  - {s}: {si.speaker} lacks {len(lost)} ID(s) all other {s} speakers have: "
                    + ", ".join(lost))
    shared = set(eval_ids["dev"]) & set(eval_ids["test"])
    say(f"- Dev and test evaluation sets share {len(shared)} sentence IDs (same ARCTIC prompts, "
        "different speakers/accents)")
    suit = repo_path(cfg.paths.raw_l2, root) / cfg.l2.suitcase_dir / cfg.l2.wav_dir
    n_suit = len(list(suit.glob("*.wav"))) if suit.is_dir() else 0
    say(f"- Suitcase corpus (stress-test demo only): {n_suit} recordings")
    locked = not freeze_tag_exists(cfg.freeze_tag, root)
    if locked:
        say(f"- Test split locked (no `{cfg.freeze_tag}` tag): test audio not opened; "
            "`sr`/`n_samples` are null in l2_test.jsonl")

    splits_dir = repo_path(cfg.paths.splits, root)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for s, ids in eval_ids.items():
        (splits_dir / f"{s}_eval_ids.txt").write_text("\n".join(ids) + "\n")

    # Manifests
    man = repo_path(cfg.paths.manifests, root)
    rows: dict[str, list[dict]] = {m: [] for m in cfg.manifests}
    for s, ids in lj["split_ids"].items():
        rows[f"lj_{s}"] = [lj_row(i, lj["meta"][i], s, cfg) for i in ids]
    l2_dir = repo_path(cfg.paths.raw_l2, root)
    for si in inv.values():
        ev = set(eval_ids[si.split])
        for utt in sorted(si.wav):
            text = (l2_dir / si.speaker / cfg.l2.transcript_dir / f"{utt}.txt").read_text(
                encoding="utf-8").strip()
            rows[f"l2_{si.split}"].append(l2_row(si, utt, text, ev, cfg, root))
    say("\n## Manifests\n")
    for m in cfg.manifests:
        say(f"- `{m}.jsonl`: {write_jsonl(man / f'{m}.jsonl', rows[m], args.force)}")

    report = repo_path(cfg.paths.reports, root) / "data_splits.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"# Data splits (T2, `make data`, {date.today().isoformat()})\n\n"
                      + "\n".join(log) + "\n")
    print(f"\nreport: {report.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
