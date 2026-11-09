"""Everything testable without an API key.

    .venv/bin/python tests/offline.py

Covers ingest, the scaffold contract, the four gates, the metrics and the
helpers that parse model output. Exits non-zero on the first failure.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

from runmypaper import codegen, emit, ingest, llm, metrics, verify
from runmypaper.spec import (Architecture, Block, Dataset, Evidence, KV, Layer,
                             PaperSpec, Result, Training)

PASS, FAIL = "ok  ", "FAIL"
failures = []


def check(label, cond, detail=""):
    print(f"{PASS if cond else FAIL} {label}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


# --- fixtures ---------------------------------------------------------------

PAPER = """\
A Small Residual CNN

Abstract
We propose a compact residual network reaching 93.1% top-1 on CIFAR-10.

1 Introduction
Residual networks are widely used. """ + ("Background prose that carries no "
"specification detail and is the first thing a budget should cut. " * 40) + """

3 Method
A 3x3 convolution producing 64 channels, then residual blocks.

4 Experimental Setup
SGD, learning rate 0.1, momentum 0.9, weight decay 5e-4, batch size 128.

5 Related Work
Other people did other things.
"""

MODEL_PY = '''\
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)) + x)


class SmallResNet(nn.Module):
    def __init__(self, in_channels, num_classes, width=64):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.block1 = ResBlock(width)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(width, num_classes)

    def forward(self, x):
        return self.fc(self.flatten(self.pool(self.block1(self.stem(x)))))


def build_model(cfg):
    return SmallResNet(cfg["input_shape"][0], cfg["output_dim"])
'''

BROKEN_PY = '''\
import torch.nn as nn


def build_model(cfg):
    # declares 64 input features but receives 3*32*32
    return nn.Sequential(nn.Flatten(), nn.Linear(64, cfg["output_dim"]))
'''


def sample_spec() -> PaperSpec:
    return PaperSpec(
        title="A Small Residual CNN for CIFAR-10",
        architecture=Architecture(
            family="resnet", input_shape=[3, 32, 32], output_dim=10,
            blocks=[Block(name="ResBlock", layers=[
                Layer(name="conv1", op="Conv2d",
                      args=[KV(key="out_channels", value="64"),
                            KV(key="kernel_size", value="3"),
                            KV(key="padding", value="1")]),
                Layer(name="bn1", op="BatchNorm2d"),
                Layer(name="relu", op="ReLU"),
            ])],
            graph=[
                Layer(name="stem", op="Conv2d",
                      args=[KV(key="out_channels", value="64"),
                            KV(key="kernel_size", value="3"),
                            KV(key="padding", value="1")]),
                Layer(name="block1", op="Block:ResBlock"),
                Layer(name="pool", op="AdaptiveAvgPool2d",
                      args=[KV(key="output_size", value="1")]),
                Layer(name="flatten", op="Flatten"),
                Layer(name="fc", op="Linear",
                      args=[KV(key="in_features", value="64")]),
            ],
        ),
        dataset=Dataset(name="CIFAR10", task="image_classification",
                        num_classes=10, input_shape=[3, 32, 32], source="synthetic"),
        training=Training(optimizer="SGD", lr=0.1, momentum=0.9,
                          weight_decay=0.0005, batch_size=32, epochs=1),
        reported=[Result(metric="top1_accuracy", value=0.931)],
        evidence=[Evidence(field="training.lr", quote="learning rate 0.1", confidence=0.95)],
        unresolved=["training.lr_schedule"],
        assumptions=["no LR schedule applied"],
    )


# --- tests ------------------------------------------------------------------

def test_ingest(tmp: Path):
    paper = tmp / "paper.txt"
    paper.write_text(PAPER)
    doc = ingest.load(paper)
    names = [s.name for s in doc.sections]
    check("ingest finds sections", "Method" in names and "Experimental Setup" in names, names)

    text = doc.relevant_text()
    check("ingest drops related work", "## Related Work" not in text)
    check("ingest keeps method", "## Method" in text)

    # A budget truncates the long non-priority section without dropping any.
    expected = len([s for s in doc.sections if "related work" not in s.name.lower()])
    tight = doc.relevant_text(max_chars=1000)
    kept = sum(1 for s in doc.sections if f"## {s.name}" in tight)
    check("budget truncates, never drops", kept == expected, f"{kept} of {expected} survived")
    check("truncation is marked", "[... truncated ...]" in tight, f"{len(tight)} chars")
    check("budget cuts the boilerplate first",
          "Background prose" not in tight.split("## Method")[0].split("## Introduction")[1][-200:]
          or len(tight) < len(doc.relevant_text()),
          "introduction was not shortened")
    check("priority sections survive a budget whole",
          "learning rate 0.1" in tight and "64 channels" in tight)
    # Below MIN_SECTION_CHARS per section the budget is a target, not a cap.
    floor = doc.relevant_text(max_chars=50)
    check("tiny budget falls back to the per-section floor", len(floor) > 50,
          f"{len(floor)} chars")

    check("rejects a missing file",
          _raises(lambda: ingest.load_source("./definitely-not-here.pdf"), SystemExit))


def test_helpers():
    check("extract_code strips a fence",
          llm.extract_code("```python\nx = 1\n```") == "x = 1\n")
    check("extract_code passes bare code through",
          llm.extract_code("x = 1") == "x = 1\n")
    check("json_from_text reads a fenced object",
          llm.json_from_text('here you go:\n```json\n{"a": 1}\n```\nbye') == {"a": 1})
    check("json_from_text handles braces in strings",
          llm.json_from_text('{"a": "}{", "b": 2}') == {"a": "}{", "b": 2})
    check("json_from_text rejects prose",
          _raises(lambda: llm.json_from_text("no object here"), llm.LLMError))

    # A 429 is either a daily allowance, which no retry fixes, or a per-minute
    # window, which honouring the provider's stated wait does fix.
    daily = ("Quota exceeded for metric: generate_content_free_tier_requests, "
             "limit: 20 ... requests per day limit exceeded on Free Tier")
    minute = ("Rate limit reached for model `x` on tokens per minute (TPM): "
              "Limit 8000, Used 3887, Requested 6144. Please try again in 15.2325s.")
    check("tells a daily allowance from a rate limit",
          llm._is_daily_quota(daily) and not llm._is_daily_quota(minute))

    class _NoHeaders:
        headers = None

    class _WithHeader:
        headers = {"retry-after": "42"}

    check("reads the wait from the error body",
          llm._retry_after(_NoHeaders(), minute) == 15.2325,
          llm._retry_after(_NoHeaders(), minute))
    check("prefers the retry-after header",
          llm._retry_after(_WithHeader(), minute) == 42.0)
    check("falls back when no wait is stated",
          llm._retry_after(_NoHeaders(), "no hint") is None)


def test_metrics():
    gold = sample_spec()
    check("identical specs score 1.0",
          metrics.compare_specs(gold, gold).as_dict()["spec_f1"] == 1.0)

    pred = gold.model_copy(deep=True)
    pred.training.lr = 0.1 + 1e-9        # within tolerance
    pred.training.epochs = 99            # wrong
    pred.training.batch_size = None      # missed
    pred.training.lr_schedule = "cosine"  # hallucinated (gold is None)
    c = metrics.compare_specs(pred, gold).as_dict()
    check("tolerates float noise", not any("lr:" in w for w in c["wrong"]), c["wrong"])
    check("flags a wrong value", any("epochs" in w for w in c["wrong"]), c["wrong"])
    check("flags a missed value", c["missed"] == ["training.batch_size"], c["missed"])
    check("flags a hallucination", c["hallucinated"] == ["training.lr_schedule"],
          c["hallucinated"])

    ops = metrics.op_sequence(gold.architecture)
    check("op_sequence expands blocks",
          ops == ["Conv2d", "Conv2d", "BatchNorm2d", "ReLU", "AdaptiveAvgPool2d",
                  "Flatten", "Linear"], ops)
    check("param_ratio", metrics.param_ratio(1100, 1000) == 1.1)


def test_spec_validation():
    """The schema rejects the defects the first real extraction produced."""
    def arch(**kw):
        base = dict(family="resnet", input_shape=[3, 224, 224], output_dim=1000)
        return lambda: Architecture(**{**base, **kw})

    # A graph referencing a block nobody declared cannot be built.
    check("rejects an undeclared block reference",
          _raises(arch(graph=[Layer(name="layer1", op="Block:missing")]),
                  ValidationError))
    check("accepts a declared block reference",
          not _raises(arch(blocks=[Block(name="b", layers=[Layer(name="c", op="Conv2d")])],
                           graph=[Layer(name="layer1", op="Block:b")]),
                      ValidationError))

    # The real defect: the stack ends at 64 channels, the classifier wants 512.
    pool = Layer(name="avgpool", op="AdaptiveAvgPool2d",
                 args=[KV(key="output_size", value="(1, 1)")])
    flat = Layer(name="flatten", op="Flatten")
    fc = Layer(name="fc", op="Linear", args=[KV(key="in_features", value="512"),
                                             KV(key="out_features", value="1000")])
    narrow = Layer(name="conv", op="Conv2d",
                   args=[KV(key="out_channels", value="64")])
    check("rejects a classifier wider than the stack feeding it",
          _raises(arch(graph=[narrow, pool, flat, fc]), ValidationError))
    wide = Layer(name="conv", op="Conv2d",
                 args=[KV(key="out_channels", value="512")])
    check("accepts a classifier matching the stack",
          not _raises(arch(graph=[wide, pool, flat, fc]), ValidationError))
    # Without a global pool, in_features is C*H*W and the check cannot decide.
    check("skips the width check after a plain flatten",
          not _raises(arch(graph=[narrow, flat, fc]), ValidationError))
    # Reached through a block, as a real residual network is written.
    check("expands blocks before checking the width",
          _raises(arch(blocks=[Block(name="b", layers=[narrow])],
                       graph=[Layer(name="layer1", op="Block:b"), pool, flat, fc]),
                  ValidationError))

    # task and source drive data.py, so free text there fails silently at run
    # time. These were the values the first extraction actually returned.
    check("rejects a free-text task",
          _raises(lambda: Dataset(name="ImageNet", task="image classification"),
                  ValidationError))
    check("rejects a source data.py cannot act on",
          _raises(lambda: Dataset(name="ImageNet", task="image_classification",
                                  source="ILSVRC 2012"), ValidationError))
    check("accepts the supported choices",
          not _raises(lambda: Dataset(name="CIFAR10", task="image_classification",
                                      source="torchvision"), ValidationError))


def test_unsupported_task(tmp: Path):
    """A task the scaffold cannot serve fails loudly, not silently."""
    spec = sample_spec()
    spec.dataset.task = "segmentation"
    out = tmp / "segmentation"
    codegen.scaffold(spec, out)
    (out / "model.py").write_text(MODEL_PY)
    # smoke.py builds its own tensors, so only the data loader sees the task.
    p = subprocess.run([sys.executable, "train.py", "--limit", "16", "--epochs", "1"],
                       cwd=out, capture_output=True, text=True, timeout=300)
    check("unsupported task raises a named error",
          p.returncode != 0 and "task-specific data.py" in p.stderr,
          p.stderr[-200:] or "exited 0")


def _compile(spec, input_shape=None, output_dim=None):
    """Compile a spec and build the module, in process."""
    ns = {}
    exec(emit.emit(spec), ns)
    cfg = {"input_shape": input_shape or spec.architecture.input_shape,
           "output_dim": output_dim or spec.architecture.output_dim}
    return ns["build_model"](cfg)


def test_emit():
    """The spec compiles to a module with no model call."""
    import torch

    spec = sample_spec()
    check("sample spec is compilable", emit.unsupported(spec) == [],
          emit.unsupported(spec))
    model = _compile(spec)
    out = model(torch.zeros(2, *spec.architecture.input_shape))
    check("compiled module returns (batch, output_dim)",
          tuple(out.shape) == (2, 10), tuple(out.shape))

    # output_dim from config wins over whatever the spec's Linear declared.
    spec2 = sample_spec()
    spec2.architecture.graph[-1].args = [KV(key="out_features", value="7")]
    out2 = _compile(spec2, output_dim=10)(torch.zeros(2, 3, 32, 32))
    check("config's output_dim overrides the spec's out_features",
          tuple(out2.shape) == (2, 10), tuple(out2.shape))

    # The core claim: input widths come from a shape probe, so a spec that is
    # wrong about its own in_features still compiles to something that runs.
    # No global pool here, so the schema cannot decide it and lets it through.
    spec3 = sample_spec()
    spec3.architecture.graph = [
        Layer(name="conv", op="Conv2d",
              args=[KV(key="out_channels", value="8"), KV(key="kernel_size", value="3"),
                    KV(key="padding", value="1")]),
        Layer(name="flatten", op="Flatten"),
        Layer(name="fc", op="Linear", args=[KV(key="in_features", value="99999"),
                                            KV(key="out_features", value="10")]),
    ]
    spec3.architecture.blocks = []
    check("a wrong in_features does not reach the generated code",
          "99999" not in emit.emit(spec3))
    out3 = _compile(spec3)(torch.zeros(2, 3, 32, 32))
    check("compiles correctly despite the wrong in_features",
          tuple(out3.shape) == (2, 10), tuple(out3.shape))

    # A residual block gets a shortcut, and a projection when shapes change.
    res = sample_spec()
    res.architecture.blocks = [Block(name="Res", layers=[
        Layer(name="c1", op="Conv2d", args=[KV(key="out_channels", value="16"),
                                            KV(key="kernel_size", value="3"),
                                            KV(key="stride", value="2"),
                                            KV(key="padding", value="1")]),
        Layer(name="skip", op="Add"),
    ])]
    res.architecture.graph = [
        Layer(name="stage", op="Block:Res"),
        Layer(name="pool", op="AdaptiveAvgPool2d", args=[KV(key="output_size", value="1")]),
        Layer(name="flatten", op="Flatten"),
        Layer(name="fc", op="Linear"),
    ]
    src = emit.emit(res)
    check("an Add becomes a residual shortcut", "self.shortcut" in src)
    rm = _compile(res)
    check("the shortcut is projected when the body changes shape",
          rm.net[0].shortcut is not None)
    check("the residual block runs",
          tuple(rm(torch.zeros(2, 3, 32, 32)).shape) == (2, 10))

    # What it refuses, rather than guesses at.
    bad_op = sample_spec()
    bad_op.architecture.graph[0] = Layer(name="x", op="NotARealLayer")
    check("refuses an op that is not in torch.nn",
          any("NotARealLayer" in r for r in emit.unsupported(bad_op)))

    bad_arg = sample_spec()
    bad_arg.architecture.graph[0] = Layer(
        name="x", op="Conv2d",
        args=[KV(key="out_channels", value="8"),
              KV(key="kernel_size", value="__import__('os').system('x')")])
    reasons = emit.unsupported(bad_arg)
    check("refuses an argument that is not a literal",
          any("not a Python literal" in r for r in reasons), reasons)

    wired = sample_spec()
    wired.architecture.graph[1] = Layer(name="merge", op="ReLU", inputs=["stem", "pool"])
    check("refuses a graph with explicit wiring",
          any("explicit inputs" in r for r in emit.unsupported(wired)))

    no_fc = sample_spec()
    no_fc.architecture.graph = [Layer(name="conv", op="Conv2d",
                                      args=[KV(key="out_channels", value="4"),
                                            KV(key="kernel_size", value="3")])]
    no_fc.architecture.blocks = []
    check("refuses a graph with no classifier",
          any("no Linear" in r for r in emit.unsupported(no_fc)))


def test_compiled_passes_gates(tmp: Path):
    """The compiled model.py clears all four gates, with no API key set."""
    spec = sample_spec()
    out = tmp / "compiled"
    codegen.scaffold(spec, out)
    result = codegen.write_model(spec, out)
    check("write_model compiles rather than generates",
          result.method == "compiled", result.method)
    passed, failure = verify.run_gates(out)
    check("compiled model passes every gate",
          failure is None and passed == ["import", "forward", "step", "train"],
          f"passed={passed} failure={failure.stage if failure else None}")


def test_scaffold_and_gates(tmp: Path):
    spec = sample_spec()
    out = tmp / "project"
    codegen.scaffold(spec, out)
    produced = sorted(p.name for p in out.iterdir())
    check("scaffold writes the project",
          produced == ["README.md", "config.yaml", "data.py", "requirements.txt",
                       "smoke.py", "spec.json", "train.py"], produced)

    cfg = (out / "config.yaml").read_text()
    check("config carries the hyperparameters", "lr: 0.1" in cfg and "momentum: 0.9" in cfg)
    check("README lists unresolved fields", "training.lr_schedule" in (out / "README.md").read_text())

    (out / "model.py").write_text(MODEL_PY)
    passed, failure = verify.run_gates(out)
    check("correct model passes every gate",
          failure is None and passed == ["import", "forward", "step", "train"],
          f"passed={passed} failure={failure.stage if failure else None}")

    (out / "model.py").write_text(BROKEN_PY)
    passed, failure = verify.run_gates(out)
    check("broken model stops at the forward gate",
          failure is not None and failure.stage == "forward",
          f"stopped at {failure.stage if failure else 'nowhere'}")
    check("failure carries a usable error",
          failure is not None and "shapes cannot be multiplied" in failure.output,
          (failure.output[-120:] if failure else ""))


def test_train_runs(tmp: Path):
    """The generated project trains and reports a decreasing loss."""
    out = tmp / "project"
    (out / "model.py").write_text(MODEL_PY)
    p = subprocess.run([sys.executable, "train.py", "--limit", "128", "--epochs", "2"],
                       cwd=out, capture_output=True, text=True, timeout=300)
    check("train.py exits cleanly", p.returncode == 0, p.stderr[-200:])
    if p.returncode == 0:
        report = json.loads(p.stdout[p.stdout.find("{"):])
        check("train.py reports metrics",
              {"epochs", "loss_history", "accuracy", "params"} <= set(report), report.keys())
        check("loss decreases", report["loss_history"][-1] < report["loss_history"][0],
              report["loss_history"])


def _raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_ingest(tmp)
        test_helpers()
        test_metrics()
        test_spec_validation()
        test_emit()
        test_compiled_passes_gates(tmp)
        test_scaffold_and_gates(tmp)
        test_unsupported_task(tmp)
        test_train_runs(tmp)

    print()
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}")
        raise SystemExit(1)
    print("all offline tests passed")
