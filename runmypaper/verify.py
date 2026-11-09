"""Staged execution gates plus the repair loop.

Gates run in order and stop at the first failure; the failing stage and its
stderr are what the repair prompt sees. Each gate runs as a subprocess with a
timeout -- swap `_run` for a container call when generated code stops being
trusted.
"""
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .codegen import repair_model
from .config import GATE_TIMEOUT_S, MAX_REPAIRS
from .spec import PaperSpec

GATES = [
    ("import", [sys.executable, "-c", "import model"]),
    ("forward", [sys.executable, "smoke.py", "--stage", "forward"]),
    ("step", [sys.executable, "smoke.py", "--stage", "step"]),
    ("train", [sys.executable, "train.py", "--limit", "128", "--epochs", "1"]),
]


@dataclass
class GateResult:
    stage: str
    ok: bool
    output: str


@dataclass
class VerifyResult:
    passed: List[str]
    failure: Optional[GateResult]
    repairs: int

    @property
    def ok(self) -> bool:
        return self.failure is None


# A gate asks whether the generated code runs, not whether it reaches the
# paper's accuracy, so the data loader is told to synthesise batches. Otherwise
# the train gate downloads a dataset and the gates stop working offline.
GATE_ENV = {**os.environ, "RMP_DATA": "synthetic"}


def _run(cmd, cwd: Path) -> GateResult:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=GATE_TIMEOUT_S, env=GATE_ENV)
    except subprocess.TimeoutExpired:
        return GateResult(cmd[-1], False, f"timed out after {GATE_TIMEOUT_S}s")
    return GateResult(cmd[-1], p.returncode == 0, p.stdout + p.stderr)


def run_gate(stage: str, out: Path) -> GateResult:
    """One gate by name, so a caller can report progress per stage."""
    r = _run(dict(GATES)[stage], out)
    r.stage = stage
    return r


def run_gates(out: Path) -> tuple[List[str], Optional[GateResult]]:
    passed = []
    for stage, _ in GATES:
        r = run_gate(stage, out)
        if not r.ok:
            return passed, r
        passed.append(stage)
    return passed, None


def verify(spec: PaperSpec, out: Path, max_repairs: int = MAX_REPAIRS,
           on_event=print) -> VerifyResult:
    out = Path(out)
    for attempt in range(max_repairs + 1):
        passed, failure = run_gates(out)
        if failure is None:
            return VerifyResult(passed, None, attempt)
        on_event(f"  gate '{failure.stage}' failed "
                 f"(passed: {', '.join(passed) or 'none'})")
        if attempt == max_repairs:
            return VerifyResult(passed, failure, attempt)
        on_event(f"  repair {attempt + 1}/{max_repairs}")
        repair_model(spec, out, failure.stage, failure.output)
    raise AssertionError("unreachable")
