"""PDF report builder (D-107): the data helpers behind its tables and figures. No reportlab needed."""
import importlib.util
import sys

from tts.config import REPO_ROOT, load_config


def _script(name):
    path = sys.path[:]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = path
    return mod


rep = _script("build_report_pdf")
CFG = load_config(REPO_ROOT / "configs" / "report_pdf.yaml").report_pdf


def _r(uid, mae, ratio="", status="ok"):
    return {"id": uid, "status": status, "wb_mae_ms": str(mae), "stretch_ratio": str(ratio)}


def test_quantile_is_linear_interpolation():
    assert rep.quantile([1, 2, 3, 4], 0.5) == 2.5
    assert rep.quantile([10, 20], 0.9) == 19.0
    assert rep.quantile([], 0.5) is None


def test_mae_values_skip_failed_and_empty():
    rows = [_r("a", 10), _r("b", 99, status="failed"), {"id": "c", "status": "ok", "wb_mae_ms": ""}]
    assert rep.mae_values(rows) == [10.0]


def test_stretch_bins_use_the_main_systems_ratio():
    main = [_r("a", 10, 0.7), _r("b", 20, 1.0), _r("c", 30, "")]
    other = [_r("a", 40), _r("b", 60), _r("c", 80)]
    out = rep.stretch_bins({"main": main, "x": other}, "main", [0.0, 0.8, 1.1, 99.0])
    assert out["main"] == [(10.0, 1), (20.0, 1), (None, 0)]
    assert out["x"] == [(40.0, 1), (60.0, 1), (None, 0)]   # "c" has no ratio: excluded everywhere


def test_labels_and_formatting():
    assert [rep.bin_label(0.0, 0.8), rep.bin_label(0.9, 1.1), rep.bin_label(1.2, 99.0)] == ["< 0.8", "0.9-1.1", "> 1.2"]
    assert rep.fmt_p(0.0123) == "0.012" and rep.fmt_p(4.9e-192) == "5e-192"


def test_config_points_at_committed_results():
    for s in CFG.systems:
        assert (REPO_ROOT / "results" / s / CFG.split / "summary.json").is_file()
    for sp, a, b in CFG.paired:
        assert (REPO_ROOT / "results" / "paired" / f"{sp}_{a}_vs_{b}.csv").is_file()
        assert set(rep.paired(sp, a, b)) >= {"wb_mae_ms", "wer", "utmos", "us_prob"}
    assert set(CFG.core) <= set(CFG.systems) and set(CFG.colors) == set(CFG.systems)
    for k in ("mfa_error", "training", "pace"):
        assert (REPO_ROOT / CFG.data[k]).is_file()
    for p in CFG.data.profiles.values():
        assert (REPO_ROOT / p).is_file()


def test_training_curves_and_smoothing(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("tag,step,value\nval/dur,0,1.0\nval/dur,100,0.5\nloss/d,0,3\n")
    c = rep.read_curves(p)
    assert c["val/dur"] == ([0, 100], [1.0, 0.5]) and c["loss/d"] == ([0], [3.0])
    assert rep.moving_average([1, 2, 3, 4], 2) == [1.0, 1.5, 2.5, 3.5]


def test_glossary_covers_every_system_and_key_term():
    terms = " ".join(t for _, items in rep.GLOSSARY for t, _ in items)
    for t in ["V0", "M0", "M1", "m1_pos", "m1_pos_diph", "abl_a_pos", "MAS", "MFA", "CTC", "VITS", "UTMOS", "WER",
              "MAE", "CI", "Holm", "Wilcoxon", "RTF", "MACs", "ARPAbet", "G2P", "OOV", "F1", "LPC", "WSOLA", "LJ"]:
        assert t in terms, t
