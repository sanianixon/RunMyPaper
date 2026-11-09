"""Local web UI: drive the pipeline from a browser and watch it work.

    .venv/bin/python -m runmypaper.web
    # then open the URL it prints

Standard library only, so there is nothing extra to install and it runs with no
network. One job at a time, executed in a worker thread; the page polls
/api/jobs/<id> for stage transitions.

The UI is a view over the same functions the CLI calls -- it does not
reimplement any pipeline logic, so what it shows is what the system does.
"""
import argparse
import json
import socket
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from . import codegen, config, extract, ingest, llm, verify
from .spec import PaperSpec

HERE = Path(__file__).resolve().parent
PAGE = HERE / "webui" / "index.html"
OUT = config.ROOT / "out"
BENCH = config.ROOT / "bench" / "papers.yaml"

# What the page draws as the pipeline strip, in order.
STAGES = ["ingest", "extract", "scaffold", "generate",
          "import", "forward", "step", "train"]
GATES = ["import", "forward", "step", "train"]


# --- jobs -------------------------------------------------------------------

class Job:
    """One pipeline run. Mutated by a worker thread, read by the HTTP thread."""

    def __init__(self, paper: str, variant: str, name: str,
                 use_cache: bool, verify_gates: bool):
        self.id = uuid.uuid4().hex[:12]
        self.paper, self.variant, self.name = paper, variant, name
        self.use_cache, self.verify_gates = use_cache, verify_gates
        self.lock = threading.Lock()
        self.stages: Dict[str, dict] = {s: {"state": "pending", "detail": ""}
                                        for s in STAGES}
        self.log: List[str] = []
        self.spec: Optional[dict] = None
        self.model_py: str = ""
        self.gate_output: Dict[str, str] = {}
        self.repairs = 0
        self.done = False
        self.error = ""

    def mark(self, stage: str, state: str, detail: str = "") -> None:
        with self.lock:
            if stage in self.stages:
                self.stages[stage] = {"state": state, "detail": detail}

    def say(self, line: str) -> None:
        with self.lock:
            self.log.append(line)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "id": self.id, "name": self.name, "paper": self.paper,
                "variant": self.variant, "done": self.done, "error": self.error,
                "stages": [{"name": s, **self.stages[s]} for s in STAGES],
                "log": list(self.log), "spec": self.spec,
                "model_py": self.model_py, "gate_output": dict(self.gate_output),
                "repairs": self.repairs,
                "ok": self.done and not self.error and all(
                    self.stages[g]["state"] == "ok" for g in GATES),
            }


JOBS: Dict[str, Job] = {}
RUNNING = threading.Lock()      # one pipeline at a time; the gates use the CPU


def _reset_gates(job: Job) -> None:
    for g in GATES:
        job.mark(g, "pending")


def _gate_detail(output: str) -> str:
    """A one-line summary of what a gate printed.

    smoke.py prints its report as one JSON line but train.py pretty-prints it,
    so taking the last line alone yields a stray brace. Parse the report
    instead and show the scalars from it.
    """
    text = output.strip()
    report = None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            report, _ = json.JSONDecoder().raw_decode(text[i:])
        except ValueError:
            continue
        break
    if not isinstance(report, dict):
        lines = [l for l in text.splitlines() if l.strip()]
        return lines[-1][:160] if lines else ""
    parts = []
    for key, value in report.items():
        if isinstance(value, (int, float, str)):
            parts.append(f"{key}={value}")
        elif isinstance(value, list) and len(value) <= 4:
            parts.append(f"{key}={value}")
    return ", ".join(parts)[:160]


def _execute(job: Job) -> None:
    out = OUT / job.name
    try:
        job.mark("ingest", "running")
        doc = ingest.load_source(job.paper)
        body = doc.relevant_text()
        job.mark("ingest", "ok",
                 f"{len(doc.sections)} sections, {len(body):,} chars retained")
        job.say(f"loaded {doc.title}")

        job.mark("extract", "running",
                 "asking the model for a PaperSpec (can take a few minutes)")
        spec = extract.extract(doc, use_cache=job.use_cache,
                               variant=job.variant or None)
        job.spec = spec.model_dump()
        n = len(spec.unresolved)
        job.mark("extract", "ok",
                 f"{len(spec.evidence)} evidence quotes, "
                 f"{n} field{'' if n == 1 else 's'} unresolved")
        job.say(f"spec: {spec.title}")

        job.mark("scaffold", "running")
        codegen.scaffold(spec, out)
        job.mark("scaffold", "ok", "data.py, train.py, smoke.py, config.yaml")

        job.mark("generate", "running", "writing model.py from the spec")
        built = codegen.write_model(spec, out, use_cache=job.use_cache)
        job.model_py = (out / "model.py").read_text()
        lines = len(job.model_py.splitlines())
        job.mark("generate", "ok", (
            f"{lines} lines compiled from the spec, no model call"
            if built.method == "compiled"
            else f"{lines} lines generated by the model"))
        if built.fallback_reason:
            job.say(f"compiler declined, so the model was asked: "
                    f"{built.fallback_reason}")

        if not job.verify_gates:
            for g in GATES:
                job.mark(g, "skipped")
            return

        # Gates run one at a time rather than through verify.run_gates, so the
        # page can light each one up as it happens. Same gate definitions.
        for attempt in range(config.MAX_REPAIRS + 1):
            _reset_gates(job)
            failure = None
            for stage in GATES:
                job.mark(stage, "running")
                result = verify.run_gate(stage, out)
                if result.ok:
                    job.mark(stage, "ok", _gate_detail(result.output))
                    continue
                job.mark(stage, "fail", result.output.strip()[-400:])
                with job.lock:
                    job.gate_output[stage] = result.output[-4000:]
                failure = result
                break
            if failure is None:
                job.say("all gates passed")
                break
            job.say(f"gate '{failure.stage}' failed")
            if attempt == config.MAX_REPAIRS:
                job.say("out of repair attempts")
                break
            job.say(f"repair {attempt + 1}/{config.MAX_REPAIRS}")
            codegen.repair_model(spec, out, failure.stage, failure.output)
            job.model_py = (out / "model.py").read_text()
            job.repairs += 1

        report = {
            "title": spec.title,
            "gates_passed": [g for g in GATES
                             if job.stages[g]["state"] == "ok"],
            "repairs": job.repairs,
            "unresolved": spec.unresolved,
        }
        (out / "report.json").write_text(json.dumps(report, indent=2))

    except llm.QuotaExhausted as exc:
        # Worth separating from a generic failure: nothing is broken, the free
        # tier is simply spent, and the page should say what actually helps.
        job.error = ("The free-tier allowance for today is used up. Open a "
                     "finished run instead, or point RMP_BASE_URL at another "
                     "free endpoint (groq, openrouter, ollama).")
        for s in STAGES:
            if job.stages[s]["state"] == "running":
                job.mark(s, "fail", "free-tier quota exhausted")
        job.say(str(exc)[:1200])
    except ValidationError as exc:
        # The schema rejected the model's spec even after the retry pass.
        job.error = "the extracted spec did not satisfy the schema"
        job.mark("extract", "fail", str(exc)[:1200])
        job.say(str(exc)[:1200])
    except Exception as exc:
        job.error = f"{type(exc).__name__}: {exc}"
        for s in STAGES:
            if job.stages[s]["state"] == "running":
                job.mark(s, "fail", str(exc)[:800])
        job.say(traceback.format_exc()[-1500:])
    finally:
        with job.lock:
            job.done = True


def _start(job: Job) -> None:
    def worker():
        with RUNNING:
            _execute(job)
    threading.Thread(target=worker, daemon=True).start()


# --- reading finished runs off disk -----------------------------------------

def _saved_runs() -> List[dict]:
    if not OUT.is_dir():
        return []
    rows = []
    for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
        spec_file = d / "spec.json"
        if not spec_file.is_file():
            continue
        try:
            title = json.loads(spec_file.read_text()).get("title", d.name)
        except json.JSONDecodeError:
            title = d.name
        report = d / "report.json"
        gates = []
        if report.is_file():
            try:
                gates = json.loads(report.read_text()).get("gates_passed", [])
            except json.JSONDecodeError:
                pass
        rows.append({"name": d.name, "title": title, "gates": gates,
                     "has_model": (d / "model.py").is_file()})
    return rows


def _load_run(name: str) -> dict:
    d = OUT / name
    spec_file = d / "spec.json"
    if not spec_file.is_file():
        raise FileNotFoundError(name)
    spec = json.loads(spec_file.read_text())
    report = {}
    if (d / "report.json").is_file():
        try:
            report = json.loads((d / "report.json").read_text())
        except json.JSONDecodeError:
            pass
    passed = report.get("gates_passed", [])
    stages = []
    for s in STAGES:
        if s in GATES:
            state = "ok" if s in passed else ("pending" if not passed else "fail")
        else:
            state = "ok"
        stages.append({"name": s, "state": state, "detail": ""})
    return {
        "id": f"saved:{name}", "name": name, "paper": name, "variant": "",
        "done": True, "error": "", "stages": stages, "log": ["loaded from disk"],
        "spec": spec,
        "model_py": (d / "model.py").read_text() if (d / "model.py").is_file() else "",
        "gate_output": {}, "repairs": report.get("repairs", 0),
        "ok": len(passed) == len(GATES),
    }


def _validate_specs() -> List[dict]:
    """Run every spec file under out/ through the schema.

    This is the structural guard made visible: a spec that contradicts itself
    is listed with the exact reason, which is what the repair loop acts on.
    """
    rows = []
    if not OUT.is_dir():
        return rows
    for path in sorted(OUT.glob("*/spec*.json")):
        row = {"file": f"{path.parent.name}/{path.name}", "errors": []}
        try:
            PaperSpec.model_validate_json(path.read_text())
        except ValidationError as exc:
            row["errors"] = [
                {"field": ".".join(str(x) for x in e["loc"]) or "(whole spec)",
                 "message": e["msg"]}
                for e in exc.errors()
            ]
        except json.JSONDecodeError as exc:
            row["errors"] = [{"field": "(file)", "message": f"not valid JSON: {exc}"}]
        rows.append(row)
    return rows


def _bench_papers() -> List[dict]:
    if not BENCH.is_file():
        return []
    try:
        import yaml
        return yaml.safe_load(BENCH.read_text()) or []
    except Exception:
        return []


# --- HTTP -------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "RunMyPaper"

    def log_message(self, fmt, *args):        # quiet; the page polls often
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/config":
                self._json({"provider": config.summary(),
                            "papers": _bench_papers(),
                            "max_repairs": config.MAX_REPAIRS})
            elif path == "/api/runs":
                self._json({"runs": _saved_runs()})
            elif path.startswith("/api/runs/"):
                self._json(_load_run(path[len("/api/runs/"):]))
            elif path == "/api/validate":
                self._json({"specs": _validate_specs()})
            elif path.startswith("/api/jobs/"):
                job = JOBS.get(path[len("/api/jobs/"):])
                if job is None:
                    self._json({"error": "no such job"}, 404)
                else:
                    self._json(job.snapshot())
            else:
                self._json({"error": "not found"}, 404)
        except FileNotFoundError as exc:
            self._json({"error": f"not found: {exc}"}, 404)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/jobs":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "body is not JSON"}, 400)
            return
        paper = str(body.get("paper", "")).strip()
        if not paper:
            self._json({"error": "paper is required"}, 400)
            return
        if RUNNING.locked():
            self._json({"error": "a run is already in progress"}, 409)
            return
        job = Job(
            paper=paper,
            variant=str(body.get("variant", "")).strip(),
            name=str(body.get("name", "")).strip() or _slug(paper),
            use_cache=bool(body.get("use_cache", True)),
            verify_gates=bool(body.get("verify", True)),
        )
        JOBS[job.id] = job
        _start(job)
        self._json({"id": job.id})


def _slug(paper: str) -> str:
    """A directory name for a run the caller did not name.

    Built from the whole reference, not its stem: an arXiv id like
    "1512.03385" has a stem of "1512", which collides with every other paper
    from that month.
    """
    base = Path(paper).name or paper
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in base).strip("-")


def _free_port(start: int, tries: int = 20) -> int:
    for port in range(start, start + tries):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"no free port in {start}..{start + tries}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="runmypaper.web")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)

    port = _free_port(args.port)
    httpd = ThreadingHTTPServer((args.host, port), Handler)
    # Flushed: the port is not always the one asked for, and this line is the
    # only place it is reported. Redirected output buffers without this.
    print(f"RunMyPaper UI  ->  http://{args.host}:{port}", flush=True)
    print(f"provider: {config.PROVIDER} / {config.MODEL_ID or '(unset)'}", flush=True)
    print("Ctrl-C to stop", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
