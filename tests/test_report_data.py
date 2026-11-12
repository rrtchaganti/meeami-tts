"""Report data export (D-110): the CSV writer; the TensorBoard and pace inputs are laptop-only."""
import importlib.util
import sys

from tts.config import REPO_ROOT


def _script(name):
    path = sys.path[:]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = path
    return mod


rd = _script("report_data")


def test_write_rows_counts_and_writes_header(tmp_path):
    p = tmp_path / "sub" / "x.csv"
    n = rd.write_rows(p, ["a", "b"], ((i, i * 2) for i in range(3)))
    assert n == 3 and p.read_text().splitlines() == ["a,b", "0,0", "1,2", "2,4"]


def test_committed_report_data_exists():
    for f in ("m1_training_curves.csv", "pace_rho.csv"):
        assert (REPO_ROOT / "results" / "report_data" / f).is_file()
