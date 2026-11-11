"""Browser demo server (D-101): auth, inputs, job lifecycle and file serving, with a fake runner
(no GPU, no models)."""
import importlib.util
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from tts.config import REPO_ROOT, load_config

CFG = load_config(REPO_ROOT / "configs" / "web_demo.yaml")


def _script(name):
    path = sys.path[:]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = path
    return mod


wd = _script("web_demo")


class FakeRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, ref, text, out, measure):
        self.calls.append((ref.name, text, measure))
        if text == "boom":
            raise ValueError("bad input")
        (out / "main").mkdir(parents=True, exist_ok=True)
        (out / "main" / "out.wav").write_bytes(b"RIFF")
        return {"text": text, "systems": [{"name": "main", "audio": "main/out.wav"}]}


@pytest.fixture()
def server(tmp_path):
    runner = FakeRunner()
    jobs = wd.Jobs(runner, tmp_path / "jobs", keep=50)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), wd.make_handler(CFG, jobs, "k3y", "<html>page</html>"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", runner
    srv.shutdown()


def req(url, data=None, ctype=None):
    r = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    if ctype:
        r.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def wait(base, job):
    for _ in range(100):
        st = json.loads(req(f"{base}/api/job/{job}?key=k3y")[1])
        if st["status"] in ("done", "error"):
            return st
        time.sleep(0.05)
    raise TimeoutError


def test_every_route_needs_the_key(server):
    base, _ = server
    assert req(f"{base}/")[0] == 403
    assert req(f"{base}/?key=wrong")[0] == 403
    code, body = req(f"{base}/?key=k3y")
    assert code == 200 and b"page" in body
    assert req(f"{base}/api/synthesize?text=hi", data=b"x")[0] == 403


def test_examples_are_the_shipped_clips(server):
    base, _ = server
    ex = json.loads(req(f"{base}/api/examples?key=k3y")[1])
    assert {e["id"] for e in ex} >= {CFG.live_demo.default_utt}
    assert all(e["accent"] in ("arabic", "korean", "vietnamese") for e in ex)
    assert req(f"{base}/examples/{ex[0]['id']}.wav?key=k3y")[0] == 200


def test_upload_job_and_files(server):
    base, runner = server
    code, body = req(f"{base}/api/synthesize?key=k3y&text=Hello%20there&measure=1", b"webmdata",
                     "audio/webm;codecs=opus")
    assert code == 200
    job = json.loads(body)["job"]
    st = wait(base, job)
    assert st["status"] == "done" and runner.calls[-1] == ("upload.webm", "Hello there", True)
    assert req(f"{base}/files/{job}/main/out.wav?key=k3y") == (200, b"RIFF")
    # no escaping the job folder
    assert req(f"{base}/files/{job}/../../etc/passwd?key=k3y")[0] == 404
    assert req(f"{base}/files/..%2F..%2Fetc/passwd?key=k3y")[0] == 404


def test_example_job_and_errors(server):
    base, runner = server
    job = json.loads(req(f"{base}/api/synthesize?key=k3y&text=x&example={CFG.live_demo.default_utt}",
                         b"")[1])["job"]
    assert wait(base, job)["status"] == "done" and runner.calls[-1][0] == "upload.wav"
    job = json.loads(req(f"{base}/api/synthesize?key=k3y&text=boom", b"a", "audio/wav")[1])["job"]
    st = wait(base, job)
    assert st["status"] == "error" and "bad input" in st["error"]


def test_input_limits(server):
    base, _ = server
    assert req(f"{base}/api/synthesize?key=k3y&text=", b"a")[0] == 400
    assert req(f"{base}/api/synthesize?key=k3y&text=hi", b"")[0] == 400
    assert req(f"{base}/api/synthesize?key=k3y&text=hi&example=NOPE", b"")[0] == 400
    long = "a" * (int(CFG.web_demo.max_text_chars) + 1)
    assert req(f"{base}/api/synthesize?key=k3y&text={long}", b"a")[0] == 400
