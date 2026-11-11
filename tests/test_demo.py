"""Live demo runner (D-96) and hosted-checkpoint fetch (D-95): pure helpers, no GPU / network."""
import importlib.util
import sys
from pathlib import Path

import pytest

from tts.config import REPO_ROOT, load_config


def _script(name):
    """Scripts reset sys.path[0] on import, so load them by file path."""
    path = sys.path[:]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = path
    return mod


demo, fetch_models = _script("demo"), _script("fetch_models")

CFG = load_config(REPO_ROOT / "configs" / "live_demo.yaml")


def test_live_demo_config():
    ld = CFG.live_demo
    assert set(ld.systems) >= {"main", "v0"}
    assert ld.systems.main.startswith("export/m1_pos_diph/")   # the frozen main system
    assert {c.expect for c in ld.suite} <= {"ok", "ok_any_aligner", "refused"}
    assert all(("utt" in c) != ("ref" in c) for c in ld.suite)


def test_suite_texts_keep_the_spoken_words():
    """Rehearsal texts may differ in form from the ARCTIC prompt, never in the spoken words."""
    from tts.text.normalize import Normalizer, words

    norm = Normalizer(CFG)
    root = REPO_ROOT / CFG.paths.raw_l2
    for c in CFG.live_demo.suite:
        if c.expect != "ok" or "utt" not in c:
            continue
        spk, _, name = c.utt.partition("_")
        txt = root / spk / "transcript" / f"{name}.txt"
        if not txt.exists():
            pytest.skip("L2-ARCTIC transcripts not present")
        assert words(norm(c.text)) == words(norm(txt.read_text())), c.text


def test_ckpt_for_falls_back_to_training_checkpoint(tmp_path, monkeypatch):
    d = tmp_path / "export" / "x"
    d.mkdir(parents=True)
    (d / "ckpt_last.pt").write_bytes(b"")
    monkeypatch.setattr(demo, "repo_path", lambda p: tmp_path / p)
    assert demo.ckpt_for("export/x/infer.pt").endswith("ckpt_last.pt")
    (d / "infer.pt").write_bytes(b"")
    assert demo.ckpt_for("export/x/infer.pt") == "export/x/infer.pt"
    assert demo.ckpt_for("v0") == "v0"


def test_utt_ref_rejects_unknown_utterance():
    with pytest.raises(FileNotFoundError):
        demo.utt_ref(CFG, "NOPE_arctic_a9999")


def test_manifest_and_file_map():
    m = fetch_models.parse_manifest("ab12  vits_mfa/m1/infer.pt\n\ncd34  baselines/vits/x.pth\n")
    assert m == {"vits_mfa/m1/infer.pt": "ab12", "baselines/vits/x.pth": "cd34"}
    fm = fetch_models.file_map(load_config(REPO_ROOT / "configs" / "base.yaml"))
    assert fm["vits_mfa/m1_pos_diph/infer.pt"] == REPO_ROOT / "export" / "m1_pos_diph" / "infer.pt"
    assert all(isinstance(p, Path) for p in fm.values())
