"""Comparison between an extracted PaperSpec and a hand-labelled gold spec.

Two scores, kept separate because they fail for different reasons:
  spec_f1  -- did extraction read the paper's hyperparameters correctly
  arch_f1  -- did it recover the layer sequence
"""
import difflib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .spec import Architecture, PaperSpec

# Fixed so the metric stays comparable across runs. Adding a field changes the
# denominator, so change this list only deliberately.
SCALAR_FIELDS = [
    "architecture.family", "architecture.input_shape", "architecture.output_dim",
    "dataset.name", "dataset.task", "dataset.num_classes", "dataset.input_shape",
    "training.optimizer", "training.lr", "training.lr_schedule",
    "training.weight_decay", "training.momentum", "training.batch_size",
    "training.epochs", "training.loss",
]

REL_TOL = 1e-3


def _get(spec: PaperSpec, path: str) -> Any:
    obj: Any = spec
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=REL_TOL)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().casefold() == b.strip().casefold()
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def op_sequence(arch: Architecture) -> List[str]:
    """Flatten the forward graph, expanding each Block reference in place."""
    blocks = {b.name: b for b in arch.blocks}
    ops: List[str] = []
    for layer in arch.graph:
        block = blocks.get(layer.op.split(":", 1)[1]) if layer.op.startswith("Block:") else None
        if block is not None:
            ops.extend(l.op for l in block.layers)
        else:
            ops.append(layer.op)
    return ops


@dataclass
class SpecComparison:
    scalar: Dict[str, float]
    arch: Dict[str, float]
    wrong: List[str] = field(default_factory=list)       # extracted, but not the gold value
    hallucinated: List[str] = field(default_factory=list)  # gold says nothing, we produced one
    missed: List[str] = field(default_factory=list)      # gold has it, we left it null

    def as_dict(self) -> dict:
        return {
            "spec_f1": self.scalar["f1"],
            "spec_precision": self.scalar["precision"],
            "spec_recall": self.scalar["recall"],
            "arch_f1": self.arch["f1"],
            "wrong": self.wrong,
            "hallucinated": self.hallucinated,
            "missed": self.missed,
        }


def compare_specs(pred: PaperSpec, gold: PaperSpec) -> SpecComparison:
    tp = fp = fn = 0
    wrong, hallucinated, missed = [], [], []

    for path in SCALAR_FIELDS:
        g, p = _get(gold, path), _get(pred, path)
        if g is None and p is None:
            continue                      # paper states nothing, we claim nothing
        if g is None:
            fp += 1
            hallucinated.append(path)
        elif p is None:
            fn += 1
            missed.append(path)
        elif _equal(p, g):
            tp += 1
        else:
            fp += 1
            fn += 1
            wrong.append(f"{path}: {p!r} != {g!r}")

    gold_ops, pred_ops = op_sequence(gold.architecture), op_sequence(pred.architecture)
    matched = sum(b.size for b in
                  difflib.SequenceMatcher(a=gold_ops, b=pred_ops).get_matching_blocks())
    arch = _prf(matched, len(pred_ops) - matched, len(gold_ops) - matched)

    return SpecComparison(_prf(tp, fp, fn), arch, wrong, hallucinated, missed)


def param_ratio(built: Optional[int], reported: Optional[int]) -> Optional[float]:
    """Generated parameter count over the count the paper reports."""
    if not built or not reported:
        return None
    return round(built / reported, 4)
