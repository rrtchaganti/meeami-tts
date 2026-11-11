#!/usr/bin/env python
"""Browser demo: record or upload a reference, type its text, hear every system (DECISIONS D-101).

    python scripts/web_demo.py [--port 7860] [--device cuda|cpu]      # `make web-demo`

Python standard library only (no web framework enters the pinned env). The models load once, and
`scripts/demo.py`'s Demo runs each request unchanged: one MFA alignment of the reference, shared by
every system in `live_demo.systems` (configs/web_demo.yaml: all seven), then each system's
synthesis. Optional: MFA on every output, to show measured word times. Requests queue and run one
at a time on the GPU. The page polls its job, so long runs never hit a proxy timeout.

Access: every request needs the key (`?key=`), from MEEAMI_DEMO_KEY or generated at start and
printed in the URL. The server binds to 127.0.0.1. Share it through an HTTPS tunnel, because
browsers allow microphone recording only on https:// or localhost (docs/DEMO.md §6).

API: GET /  ·  GET /api/examples  ·  POST /api/synthesize?text=..&measure=0|1[&example=UTT]
(body = audio bytes, or empty with example=)  ·  GET /api/job/<id>  ·  GET /files/<id>/<path>
A result has a permanent link, /?key=..&job=<id>; results are kept on disk (out/web_demo/<id>/).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import secrets
import shutil
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path[0] = str(Path(__file__).resolve().parents[1])

from tts.config import REPO_ROOT, load_config, repo_path  # noqa: E402

EXT = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/wav": ".wav", "audio/x-wav": ".wav",
       "audio/wave": ".wav", "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a",
       "audio/flac": ".flac", "audio/aac": ".aac"}
MIME = {".wav": "audio/wav", ".png": "image/png", ".webm": "audio/webm", ".ogg": "audio/ogg",
        ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".flac": "audio/flac", ".aac": "audio/aac",
        ".json": "application/json", ".bin": "application/octet-stream"}


def load_demo_module():
    """scripts/demo.py, imported by path (scripts reset sys.path[0] on import)."""
    path = sys.path[:]
    spec = importlib.util.spec_from_file_location("demo", REPO_ROOT / "scripts" / "demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = path
    return mod


def examples(cfg) -> list[dict]:
    """The shipped demo clips with their transcript and accent."""
    from tts.guard import accent_of

    acc = accent_of(cfg)
    out = []
    for txt in sorted(repo_path(cfg.live_demo.clips_dir).glob("*.txt")):
        spk = txt.stem.split("_")[0]
        out.append({"id": txt.stem, "speaker": spk, "accent": acc.get(spk, ("", "?"))[1],
                    "text": txt.read_text(encoding="utf-8").strip()})
    return out


class Runner:
    """Loads every system once and turns (reference, text) into per-system results."""

    def __init__(self, cfg, device: str):
        self.cfg = cfg
        self.mod = load_demo_module()
        self.demo = self.mod.Demo(cfg, device)

    def __call__(self, ref: Path, text: str, out: Path, measure: bool) -> dict:
        from tts.eval.metrics import match_words
        from tts.infer.pipeline import load_reference
        from tts.text.normalize import words

        y, sr = load_reference(ref, self.cfg)
        dur = len(y) / sr
        if dur > float(self.cfg.web_demo.max_ref_seconds):
            raise ValueError(f"reference is {dur:.1f} s; the limit is "
                             f"{self.cfg.web_demo.max_ref_seconds} s")
        t0 = time.perf_counter()
        infos = self.demo.run(ref, text, out, quiet=True, measure=False)
        measured, t_meas = {}, 0.0
        if measure:
            measured, t_meas = self.demo.measure(out, words(next(iter(infos.values()))["normalized"]))
        first = next(iter(infos.values()))
        ref_words = first["ref_words"]
        systems = []
        for name, info in infos.items():
            m = dict(match_words(ref_words, measured.get(name) or []))
            per_word, errs = [], []
            for i, w in enumerate(ref_words):
                j = m.get(i)
                o = measured[name][j] if j is not None else None
                if o:
                    errs += [abs(o["start"] - w["start"]), abs(o["end"] - w["end"])]
                per_word.append(None if o is None else
                                {"start": o["start"], "end": o["end"],
                                 "d_start_ms": 1000 * (o["start"] - w["start"]),
                                 "d_end_ms": 1000 * (o["end"] - w["end"])})
            systems.append({
                "name": name, "label": str(self.cfg.web_demo.labels.get(name, name)),
                "audio": f"{name}/out.wav", "plot": f"{name}/timing.png",
                "aligner": info["aligner"], "mfa_fallback_reason": info["mfa_fallback_reason"],
                "length_error_samples": info["length_error_samples"],
                "capped_words": info["capped_words"], "rtf_model": info["rtf_model"],
                "synthesis_s": info["latency_s"]["durations"] + info["latency_s"]["synthesis"],
                "measured_mae_ms": (1000 * sum(errs) / len(errs)) if errs else None,
                "measured": per_word if measure else None})
        order = {n: i for i, n in enumerate(self.cfg.web_demo.get("order", []))}
        systems.sort(key=lambda s: order.get(s["name"], len(order)))
        return {"text": text, "normalized": first["normalized"], "oov_added": first["oov_added"],
                "reference": f"ref{ref.suffix.lower()}",   # the copy Demo.run makes
                "ref_duration_s": first["ref_duration_s"], "ref_words": ref_words,
                "align_s": first["latency_s"]["align"], "measure_s": t_meas,
                "total_s": time.perf_counter() - t0, "systems": systems}


class Jobs:
    """One worker thread: requests run one at a time (one GPU)."""

    def __init__(self, runner, root: Path, keep: int):
        self.runner, self.root, self.keep = runner, root, keep
        self.q: queue.Queue = queue.Queue()
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, ref: Path, text: str, measure: bool, job_id: str) -> str:
        with self.lock:
            self.jobs[job_id] = {"status": "queued", "submitted": time.time()}
        self.q.put((job_id, ref, text, measure))
        self._prune()
        return job_id

    def status(self, job_id: str) -> dict | None:
        with self.lock:
            j = self.jobs.get(job_id)
            if j is None:   # finished before a restart: result.json in its folder
                f = self.root / job_id / "result.json"
                if job_id.isalnum() and f.is_file():
                    return {"status": "done", "result": json.loads(f.read_text())}
                return None
            j = dict(j)
            if j["status"] == "queued":
                j["ahead"] = sum(1 for k, v in self.jobs.items() if v["status"] in
                                 ("queued", "running") and v["submitted"] < j["submitted"])
            return j

    def _work(self) -> None:
        while True:
            job_id, ref, text, measure = self.q.get()
            with self.lock:
                self.jobs[job_id].update(status="running", started=time.time())
            try:
                res = self.runner(ref, text, self.root / job_id, measure)
                (self.root / job_id / "result.json").write_text(json.dumps(res))
                upd = {"status": "done", "result": res}
            except Exception as e:  # one-line reason to the page, full trace to the console
                traceback.print_exc()
                upd = {"status": "error", "error": f"{type(e).__name__}: "
                                                   f"{str(e).splitlines()[0] if str(e) else ''}"}
            with self.lock:
                self.jobs[job_id].update(upd, finished=time.time())

    def _prune(self) -> None:
        dirs = sorted((d for d in self.root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
        for d in dirs[:-self.keep] if len(dirs) > self.keep else []:
            with self.lock:
                if self.jobs.get(d.name, {}).get("status") in ("queued", "running"):
                    continue
                self.jobs.pop(d.name, None)
            shutil.rmtree(d, ignore_errors=True)


def make_handler(cfg, jobs: Jobs, key: str, page: str):
    wd = cfg.web_demo
    ex = {e["id"]: e for e in examples(cfg)}
    clips = repo_path(cfg.live_demo.clips_dir)

    class H(BaseHTTPRequestHandler):
        server_version = "meeami-tts-demo"

        def log_message(self, fmt, *args):  # one line per request, without the key
            print(f"[web] {self.address_string()} {self.command} {urlparse(self.path).path} "
                  f"{args[1] if len(args) > 1 else ''}")

        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode())

        def _authorized(self, q) -> bool:
            if secrets.compare_digest(q.get("key", [""])[0], key):
                return True
            self._json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong key"})
            return False

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._authorized(q):
                return
            if u.path == "/":
                return self._send(200, page.encode(), "text/html; charset=utf-8")
            if u.path == "/api/examples":
                return self._json(200, list(ex.values()))
            if u.path.startswith("/api/job/"):
                st = jobs.status(u.path.rsplit("/", 1)[-1])
                return self._json(200, st) if st else self._json(404, {"error": "no such job"})
            if u.path.startswith("/examples/"):
                f = clips / (Path(u.path).name)
                if f.suffix == ".wav" and f.stem in ex and f.exists():
                    return self._send(200, f.read_bytes(), "audio/wav")
            if u.path.startswith("/files/"):
                parts = u.path.split("/")[2:]
                if len(parts) >= 2:
                    base = (jobs.root / parts[0]).resolve()
                    f = (base / "/".join(parts[1:])).resolve()
                    if base.parent == jobs.root.resolve() and base in f.parents and f.is_file():
                        return self._send(200, f.read_bytes(), MIME.get(f.suffix, "application/octet-stream"))
            self._json(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._authorized(q):
                return
            if u.path != "/api/synthesize":
                return self._json(404, {"error": "not found"})
            text = q.get("text", [""])[0].strip()
            if not text:
                return self._json(400, {"error": "type what the recording says"})
            if len(text) > int(wd.max_text_chars):
                return self._json(400, {"error": f"text is longer than {wd.max_text_chars} characters"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > int(wd.max_upload_mb) * 1024 * 1024:
                return self._json(413, {"error": f"audio is larger than {wd.max_upload_mb} MB"})
            body = self.rfile.read(n) if n else b""
            job_id = uuid.uuid4().hex[:12]
            d = jobs.root / job_id
            example = q.get("example", [""])[0]
            if example:
                if example not in ex:
                    return self._json(400, {"error": "unknown example"})
                d.mkdir(parents=True)
                ref = d / "upload.wav"
                shutil.copyfile(clips / f"{example}.wav", ref)
            elif body:
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                d.mkdir(parents=True)
                ref = d / f"upload{EXT.get(ctype, '.bin')}"
                ref.write_bytes(body)
            else:
                return self._json(400, {"error": "record or upload a reference first"})
            measure = q.get("measure", ["0"])[0] == "1"
            jobs.submit(ref, text, measure, job_id)
            self._json(200, {"job": job_id})

        do_HEAD = do_GET

    return H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "web_demo.yaml"))
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # progress lines reach a redirected log at once
    cfg = load_config(a.config)
    host, port = a.host or str(cfg.web_demo.host), a.port or int(cfg.web_demo.port)
    key = str(cfg.web_demo.key) or secrets.token_urlsafe(9)
    page = (REPO_ROOT / "scripts" / "web_demo.html").read_text(encoding="utf-8")
    print("[web] loading every system (one-time, ~30 s) ...")
    jobs = Jobs(Runner(cfg, a.device), repo_path(cfg.web_demo.out_dir), int(cfg.web_demo.keep_jobs))
    srv = ThreadingHTTPServer((host, port), make_handler(cfg, jobs, key, page))
    print(f"\n[web] ready:  http://localhost:{port}/?key={key}\n"
          f"[web] share via an HTTPS tunnel to localhost:{port}, and add  /?key={key}  to its URL "
          f"(docs/DEMO.md §6). Ctrl+C stops.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
