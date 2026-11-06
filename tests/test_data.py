"""T2 unit tests: config inheritance, freeze guard, LJ/L2 checks, eval IDs, manifests.

Uses synthetic data in tmp_path only; never touches data/raw.
"""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tts.config import REPO_ROOT, load_config
from tts.guard import SplitLockedError, assert_audio_allowed, assert_split_allowed, speakers
from tts.splits import (
    DataCheckError, check_lj, common_annotated_ids, fetch_filelists, l2_inventory, l2_row,
    parse_filelist, sha256, write_jsonl,
)

SPLITS = REPO_ROOT / "configs" / "splits.yaml"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """Empty git repo with one commit, standing in for the project root."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty",
         "-m", "init")
    return tmp_path


# --- config ------------------------------------------------------------------------------------
def test_inherits_base():
    cfg = load_config(SPLITS)
    assert cfg.seed == 1234 and cfg.freeze_tag == "freeze-v1"
    assert cfg.lj.expect.sample_rate == 22050  # interpolated from base audio.sample_rate
    assert "inherit" not in cfg


def test_overrides():
    assert load_config(SPLITS, ["lj.n_clips=7"]).lj.n_clips == 7


def test_split_speakers_match_plan_and_claude_md():
    cfg = load_config(SPLITS)
    dev, test = speakers(cfg, "dev"), speakers(cfg, "test")
    assert len(dev) == len(test) == 12 and not set(dev) & set(test)
    claude = (REPO_ROOT / "CLAUDE.md").read_text()
    listed = re.search(r"Test speakers \(([A-Z ]+)\)", claude).group(1).split()
    assert sorted(listed) == sorted(test)


# --- freeze guard ------------------------------------------------------------------------------
def test_guard_locks_test_speakers_until_tag(repo):
    cfg = load_config(SPLITS)
    assert_audio_allowed("ASI", cfg, repo)  # dev: always fine
    with pytest.raises(SplitLockedError):
        assert_audio_allowed("ABA", cfg, repo)
    with pytest.raises(SplitLockedError):
        assert_audio_allowed("aba", cfg, repo)  # case-insensitive
    with pytest.raises(SplitLockedError):
        assert_split_allowed("test", cfg, repo)
    _git(repo, "tag", cfg.freeze_tag)
    assert_audio_allowed("ABA", cfg, repo)
    assert_split_allowed("test", cfg, repo)


# --- LJSpeech ----------------------------------------------------------------------------------
def _wav(path: Path, sr=22050, ch=1, subtype="PCM_16", n=256):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((n, ch), dtype="float32") if ch > 1 else np.zeros(n, dtype="float32")
    sf.write(path, data, sr, subtype=subtype)


def _fake_lj(root: Path, ids_by_split: dict[str, list[str]], bad_sr: str | None = None):
    lj = root / "data/raw/LJSpeech-1.1"
    src = root / "remote"
    src.mkdir()
    ids = [i for v in ids_by_split.values() for i in v]
    (lj / "wavs").mkdir(parents=True)
    (lj / "metadata.csv").write_text("".join(f"{i}|Raw {i}.|norm {i}\n" for i in ids))
    for i in ids:
        _wav(lj / "wavs" / f"{i}.wav", sr=16000 if i == bad_sr else 22050)
    ov = [f"lj.n_clips={len(ids)}", f"lj.filelists.url=file://{src}/{{name}}"]
    for s, v in ids_by_split.items():
        name = f"ljs_audio_text_{s}_filelist.txt"
        (src / name).write_text("".join(f"DUMMY1/{i}.wav|text {i}\n" for i in v))
        ov += [f"lj.filelists.files.{s}.n={len(v)}",
               f"lj.filelists.files.{s}.sha256={sha256(src / name)}"]
    return load_config(SPLITS, ov)


SPLIT_IDS = {"train": ["LJ001-0001", "LJ001-0002"], "val": ["LJ001-0003"], "test": ["LJ001-0004"]}


def test_lj_ok(tmp_path):
    cfg = _fake_lj(tmp_path, SPLIT_IDS)
    fl = fetch_filelists(cfg, tmp_path)
    assert parse_filelist(fl["train"]) == SPLIT_IDS["train"]
    out = check_lj(cfg, fl, tmp_path)
    assert out["meta"]["LJ001-0003"] == "norm LJ001-0003"  # column 3
    fetch_filelists(cfg, tmp_path)  # second call: cached, hash still verified


def test_filelist_sha_mismatch(tmp_path):
    cfg = _fake_lj(tmp_path, SPLIT_IDS)
    cfg.lj.filelists.files.val.sha256 = "0" * 64
    with pytest.raises(DataCheckError, match="sha256"):
        fetch_filelists(cfg, tmp_path)


def test_lj_wrong_sample_rate(tmp_path):
    cfg = _fake_lj(tmp_path, SPLIT_IDS, bad_sr="LJ001-0002")
    with pytest.raises(DataCheckError, match="not"):
        check_lj(cfg, fetch_filelists(cfg, tmp_path), tmp_path)


def test_lj_overlapping_filelists(tmp_path):
    ids = {"train": ["LJ001-0001", "LJ001-0002"], "val": ["LJ001-0002"], "test": ["LJ001-0004"]}
    cfg = _fake_lj(tmp_path, ids)
    cfg.lj.n_clips = 3  # 3 unique clips on disk
    with pytest.raises(DataCheckError):
        check_lj(cfg, fetch_filelists(cfg, tmp_path), tmp_path)


# --- L2-ARCTIC ---------------------------------------------------------------------------------
def _fake_l2(root: Path, cfg, ann: dict[str, list[str]], n_utts=4, skip_wav_for=None):
    l2 = root / cfg.paths.raw_l2
    for spk in speakers(cfg, "dev") + speakers(cfg, "test"):
        d = l2 / spk
        for sub in ("wav", "transcript", "annotation"):
            (d / sub).mkdir(parents=True)
        for k in range(1, n_utts + 1):
            u = f"arctic_a{k:04d}"
            # test-speaker wavs are deliberately invalid: any attempt to open them fails loudly
            if spk in speakers(cfg, "test"):
                (d / "wav" / f"{u}.wav").write_bytes(b"not audio")
            else:
                _wav(d / "wav" / f"{u}.wav", sr=44100)
            (d / "transcript" / f"{u}.txt").write_text(f"text {k}")
        for u in ann.get(spk, ["arctic_a0001", "arctic_a0002", "arctic_a0003"]):
            (d / "annotation" / f"{u}.TextGrid").write_text("")
    (l2 / "suitcase_corpus" / "wav").mkdir(parents=True)
    return l2


def test_l2_inventory_and_common_ids(repo):
    cfg = load_config(SPLITS)
    _fake_l2(repo, cfg, {"SVBI": ["arctic_a0001", "arctic_a0003"]})
    inv = l2_inventory(cfg, repo)
    assert len(inv) == 24 and inv["ASI"].accent == "hindi" and inv["ABA"].split == "test"
    assert common_annotated_ids(inv, "dev") == ["arctic_a0001", "arctic_a0003"]
    assert common_annotated_ids(inv, "test") == ["arctic_a0001", "arctic_a0002", "arctic_a0003"]


def test_l2_missing_speaker(repo):
    cfg = load_config(SPLITS)
    l2 = _fake_l2(repo, cfg, {})
    subprocess.run(["rm", "-r", str(l2 / "TLV")], check=True)
    with pytest.raises(DataCheckError, match="TLV"):
        l2_inventory(cfg, repo)


def test_l2_rows_never_open_test_audio_before_freeze(repo):
    cfg = load_config(SPLITS)
    _fake_l2(repo, cfg, {})
    inv = l2_inventory(cfg, repo)
    dev = l2_row(inv["ASI"], "arctic_a0001", "t", {"arctic_a0001"}, cfg, repo)
    assert dev["sr"] == 44100 and dev["n_samples"] == 256 and dev["in_eval_set"]
    test = l2_row(inv["ABA"], "arctic_a0002", "t", {"arctic_a0001"}, cfg, repo)
    assert test["sr"] is None and test["n_samples"] is None and not test["in_eval_set"]
    assert test["id"] == "ABA_arctic_a0002" and test["accent"] == "arabic"
    _git(repo, "tag", cfg.freeze_tag)  # after freeze the header is read (fake file -> error)
    with pytest.raises(Exception):
        l2_row(inv["ABA"], "arctic_a0002", "t", set(), cfg, repo)


# --- manifests ---------------------------------------------------------------------------------
def test_write_jsonl_keeps_existing(tmp_path):
    p = tmp_path / "m.jsonl"
    assert write_jsonl(p, [{"id": "a"}], force=False).startswith("wrote 1")
    assert write_jsonl(p, [{"id": "b"}], force=False).startswith("kept")
    assert json.loads(p.read_text())["id"] == "a"
    write_jsonl(p, [{"id": "b"}], force=True)
    assert json.loads(p.read_text())["id"] == "b"
    e = tmp_path / "empty.jsonl"
    assert write_jsonl(e, [], force=False) == "wrote 0 rows" and e.read_text() == ""
