"""Evidence samples and hosted test outputs (E-36): pure helpers, no GPU / network / dataset."""
import importlib.util
import sys

from tts.config import REPO_ROOT, load_config
from tts.eval.metrics import boundary_errors, boundary_summary


def _script(name):
    """Scripts reset sys.path[0] on import, so load them by file path."""
    path = sys.path[:]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = path
    return mod


samples, outputs = _script("export_samples"), _script("publish_outputs")
CFG = load_config(REPO_ROOT / "configs" / "samples.yaml")


def _row(uid, acc, mae, status="ok"):
    return {"id": uid, "accent": acc, "status": status, "wb_mae_ms": str(mae)}


def test_selection_headline_plus_best_median_worst():
    rows = [_row("AAA_arctic_a0003", "x", 30), _row("BBB_arctic_a0003", "x", 10),
            _row("AAA_arctic_a0001", "x", 5), _row("AAA_arctic_a0002", "x", 20),
            _row("AAA_arctic_a0009", "x", 90), _row("AAA_arctic_a0010", "x", 1, "failed")]
    per = {CFG.samples.main: {r["id"]: r for r in rows}}
    sel = samples.Exporter(CFG, REPO_ROOT / "samples").select(per)["x"]
    ids = [i for i, _ in sel]
    # both headline readings first, then best (5), median, worst (90); failed rows never picked
    assert ids[:2] == ["AAA_arctic_a0003", "BBB_arctic_a0003"]
    assert "AAA_arctic_a0001" in ids and "AAA_arctic_a0009" in ids
    assert "AAA_arctic_a0010" not in ids and len(ids) == len(set(ids))


def test_word_errors_and_mae_match_the_metric():
    truth = [{"word": "a", "start": 0.0, "end": 0.2}, {"word": "b", "start": 0.2, "end": 0.5}]
    out = [{"word": "a", "start": 0.01, "end": 0.18}, {"word": "b", "start": 0.18, "end": 0.55}]
    e = samples.word_errors(truth, out)
    assert [r["d_start_ms"] for r in e] == [10.0, -20.0] and e[1]["d_end_ms"] == 50.0
    mae = boundary_summary(boundary_errors(truth, out))["mae_ms"]
    samples.check_mae(truth, out, str(mae), "ok")            # equal: passes
    try:
        samples.check_mae(truth, out, str(mae + 1), "bad")
        raise AssertionError("a wrong reported MAE must stop the export")
    except SystemExit:
        pass


def test_unmatched_word_has_no_error():
    truth = [{"word": "a", "start": 0, "end": 0.1}, {"word": "b", "start": 0.1, "end": 0.2}]
    e = samples.word_errors(truth, [{"word": "a", "start": 0, "end": 0.1}])
    assert e[1]["out"] is None and e[1]["d_start_ms"] is None


def test_output_tar_is_deterministic(tmp_path):
    src = tmp_path / "res"
    (src / "wav" / "u1").mkdir(parents=True)
    (src / "wav" / "u1" / "out.wav").write_bytes(b"RIFF....")
    (src / "per_utt.csv").write_text("id\nu1\n")
    a = outputs.build_tar(src, ["wav", "per_utt.csv"], tmp_path / "a.tar")
    (src / "per_utt.csv").touch()                           # a new mtime must not change the tar
    b = outputs.build_tar(src, ["wav", "per_utt.csv"], tmp_path / "b.tar")
    assert a.read_bytes() == b.read_bytes()
    assert outputs.repo_name(load_config(REPO_ROOT / "configs" / "base.yaml").hosting.outputs,
                             "v0") == "outputs/test/v0.tar"


def test_config_systems_cover_every_test_result():
    base = load_config(REPO_ROOT / "configs" / "base.yaml").hosting.outputs
    assert list(CFG.samples.systems) == list(base.systems)
    assert set(CFG.samples.labels) == set(CFG.samples.systems)
