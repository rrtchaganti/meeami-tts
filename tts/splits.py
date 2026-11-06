"""T2: data checks, LJSpeech filelists, L2-ARCTIC splits, evaluation IDs, manifest skeletons.

docs/PLAN.md Section 2 (datasets, splits, evaluation set) and Section 10 (manifest schema).
Test-speaker audio is never opened here: for test speakers only file names and transcripts are
read; `sr` / `n_samples` stay null until the freeze tag exists (tts.guard).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
from omegaconf import DictConfig

from tts.config import REPO_ROOT, repo_path
from tts.guard import accent_of, assert_audio_allowed, freeze_tag_exists, speakers


class DataCheckError(RuntimeError):
    """A data invariant from PLAN.md Section 2 does not hold."""


# --- LJSpeech filelists ------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_filelists(cfg: DictConfig, root: Path = REPO_ROOT) -> dict[str, Path]:
    """Download (once) the pinned VITS filelists and verify their sha256."""
    fl = cfg.lj.filelists
    out_dir = repo_path(cfg.paths.filelists, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, spec in fl.files.items():
        dst = out_dir / spec.name
        if not dst.exists() or sha256(dst) != spec.sha256:
            url = fl.url.format(commit=fl.commit, name=spec.name)
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            got = sha256(tmp)
            if got != spec.sha256:
                tmp.unlink()
                raise DataCheckError(f"{spec.name}: sha256 {got} != pinned {spec.sha256} ({url})")
            tmp.replace(dst)
        paths[split] = dst
    return paths


def parse_filelist(path: Path) -> list[str]:
    """VITS filelist line `DUMMY1/LJ001-0001.wav|text` -> utterance ID."""
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.append(Path(line.split("|", 1)[0]).stem)
    return ids


# --- LJSpeech ----------------------------------------------------------------------------------
def read_lj_metadata(path: Path) -> dict[str, str]:
    """metadata.csv (`id|raw|normalized`) -> {id: normalized text} (column 3, Section 2)."""
    out = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("|")
        if len(parts) != 3:
            raise DataCheckError(f"{path}:{n}: expected 3 '|'-separated fields, got {len(parts)}")
        out[parts[0]] = parts[2]
    return out


def check_lj(cfg: DictConfig, filelists: dict[str, Path], root: Path = REPO_ROOT) -> dict:
    lj_dir = repo_path(cfg.paths.raw_lj, root)
    meta = read_lj_metadata(lj_dir / cfg.lj.metadata)
    wav_ids = {p.stem for p in (lj_dir / cfg.lj.wav_dir).glob("*.wav")}
    if len(wav_ids) != cfg.lj.n_clips or len(meta) != cfg.lj.n_clips:
        raise DataCheckError(f"LJ: {len(wav_ids)} wavs, {len(meta)} metadata rows, "
                             f"expected {cfg.lj.n_clips}")
    if wav_ids != set(meta):
        raise DataCheckError(f"LJ: wav IDs != metadata IDs (sym. diff {len(wav_ids ^ set(meta))})")

    split_ids = {s: parse_filelist(p) for s, p in filelists.items()}
    for s, ids in split_ids.items():
        n = cfg.lj.filelists.files[s].n
        if len(ids) != n or len(set(ids)) != n:
            raise DataCheckError(f"LJ {s}: {len(ids)} IDs ({len(set(ids))} unique), expected {n}")
    all_ids = [i for ids in split_ids.values() for i in ids]
    if len(all_ids) != len(set(all_ids)):
        raise DataCheckError("LJ filelists overlap")
    if set(all_ids) != set(meta):
        raise DataCheckError(f"LJ filelist IDs != metadata IDs (sym. diff "
                             f"{len(set(all_ids) ^ set(meta))})")

    exp = cfg.lj.expect
    bad = []
    for i in sorted(meta):
        info = sf.info(lj_dir / cfg.lj.wav_dir / f"{i}.wav")
        if (info.samplerate, info.channels, info.subtype) != (exp.sample_rate, exp.channels,
                                                              exp.subtype):
            bad.append((i, info.samplerate, info.channels, info.subtype))
    if bad:
        raise DataCheckError(f"LJ: {len(bad)} files not {dict(exp)}; first: {bad[:3]}")
    return {"meta": meta, "split_ids": split_ids}


# --- L2-ARCTIC ---------------------------------------------------------------------------------
@dataclass
class SpeakerInventory:
    speaker: str
    split: str
    accent: str
    wav: set[str] = field(default_factory=set)
    transcript: set[str] = field(default_factory=set)
    annotation: set[str] = field(default_factory=set)


def l2_inventory(cfg: DictConfig, root: Path = REPO_ROOT) -> dict[str, SpeakerInventory]:
    """File-name inventory of every L2 speaker. Lists directories only; opens no audio."""
    l2 = repo_path(cfg.paths.raw_l2, root)
    dev, test = set(speakers(cfg, "dev")), set(speakers(cfg, "test"))
    if dev & test:
        raise DataCheckError(f"speakers in both dev and test: {sorted(dev & test)}")
    expected = dev | test
    if len(expected) != cfg.l2.n_speakers:
        raise DataCheckError(f"config lists {len(expected)} L2 speakers, expected "
                             f"{cfg.l2.n_speakers}")
    found = {p.name for p in l2.iterdir()
             if p.is_dir() and p.name != cfg.l2.suitcase_dir and (p / cfg.l2.wav_dir).is_dir()}
    if found != expected:
        raise DataCheckError(f"L2 speaker dirs differ from config: missing "
                             f"{sorted(expected - found)}, unexpected {sorted(found - expected)}")

    inv = {}
    for spk, (split, acc) in accent_of(cfg).items():
        d = l2 / spk
        stems = {}
        for kind, sub, ext in (("wav", cfg.l2.wav_dir, ".wav"),
                               ("transcript", cfg.l2.transcript_dir, ".txt"),
                               ("annotation", cfg.l2.annotation_dir, ".TextGrid")):
            if not (d / sub).is_dir():
                raise DataCheckError(f"{spk}: missing {sub}/")
            stems[kind] = {p.stem for p in (d / sub).iterdir() if p.suffix == ext}
        si = SpeakerInventory(spk, split, acc, **stems)
        if si.wav != si.transcript:
            raise DataCheckError(f"{spk}: wav/transcript stems differ "
                                 f"({len(si.wav ^ si.transcript)} IDs)")
        if not si.annotation <= si.wav:
            raise DataCheckError(f"{spk}: annotations without wav: "
                                 f"{sorted(si.annotation - si.wav)[:5]}")
        inv[spk] = si
    return inv


def common_annotated_ids(inv: dict[str, SpeakerInventory], split: str) -> list[str]:
    """Utterance IDs with a manual annotation for every speaker of the split (Section 2)."""
    sets = [si.annotation for si in inv.values() if si.split == split]
    if not sets:
        raise DataCheckError(f"no speakers in split {split!r}")
    return sorted(set.intersection(*sets))


# --- Manifests (Section 10 schema; alignment fields filled by T3/T4/T5) ------------------------
def lj_row(utt: str, text: str, split: str, cfg: DictConfig) -> dict:
    return {
        "id": utt, "wav": f"{cfg.paths.raw_lj}/{cfg.lj.wav_dir}/{utt}.wav", "text": text,
        "words": None, "phones": None, "durations": None, "word_ids": None, "n_frames": None,
        "speaker": "LJ", "accent": "us", "split": split, "is_aug": False, "source_id": None,
    }


def l2_row(si: SpeakerInventory, utt: str, text: str, eval_ids: set[str], cfg: DictConfig,
           root: Path = REPO_ROOT) -> dict:
    wav_rel = f"{cfg.paths.raw_l2}/{si.speaker}/{cfg.l2.wav_dir}/{utt}.wav"
    sr = n_samples = None
    if si.split != "test" or freeze_tag_exists(cfg.freeze_tag, root):
        assert_audio_allowed(si.speaker, cfg, root)
        info = sf.info(repo_path(wav_rel, root))  # header only
        sr, n_samples = info.samplerate, info.frames
    return {
        "id": f"{si.speaker}_{utt}", "wav": wav_rel, "text": text,
        "words": None, "phones": None, "durations": None, "word_ids": None, "n_frames": None,
        "speaker": si.speaker, "accent": si.accent, "split": si.split, "is_aug": False,
        "source_id": None, "utt": utt, "in_eval_set": utt in eval_ids,
        "ref_words_manual": None, "ref_words_mfa": None, "sr": sr, "n_samples": n_samples,
    }


def write_jsonl(path: Path, rows: list[dict], force: bool) -> str:
    """Write rows unless a non-empty manifest exists (later tasks fill it); returns status."""
    if path.exists() and path.stat().st_size > 0 and not force:
        return "kept (exists; use --force to overwrite)"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return f"wrote {len(rows)} rows"
