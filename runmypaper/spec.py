"""PaperSpec: the intermediate representation between a paper and its code.

Everything downstream reads this, never the raw paper. Keeping the IR explicit
is what makes extraction and generation separately measurable.

Schema constraints: field types stay JSON-schema-simple (no free-form dicts) so
the whole model can be enforced as a structured output.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# Datasets `scaffold/data.py` can actually download. Anything else falls back
# to synthetic tensors, so the spec must say so rather than name a source the
# generated project cannot act on.
TORCHVISION_DATASETS = ("CIFAR10", "CIFAR100", "MNIST", "FashionMNIST", "SVHN")


class KV(BaseModel):
    """One keyword argument of a layer, value kept as a literal string."""
    key: str
    value: str


class Layer(BaseModel):
    name: str                       # unique within its block/graph
    op: str                         # torch.nn class name, or "Block:<block name>"
    args: List[KV] = Field(default_factory=list)
    inputs: List[str] = Field(default_factory=list)  # [] = output of previous layer


class Block(BaseModel):
    """A repeated unit (residual block, transformer layer, ...)."""
    name: str
    layers: List[Layer]


WIDTH_KEYS = ("out_channels", "out_features", "num_features")
GLOBAL_POOL = ("AdaptiveAvgPool2d", "AdaptiveMaxPool2d")


def _arg(layer: Layer, *keys: str) -> Optional[str]:
    for kv in layer.args:
        if kv.key in keys:
            return kv.value
    return None


def _as_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _collapses_to_1x1(layer: Layer) -> bool:
    """True if this pool makes the spatial dims 1x1 (so C*H*W == C)."""
    value = _arg(layer, "output_size")
    if value is None:
        return True                 # global pooling is the only sane default
    parts = value.strip().strip("()[]").split(",")
    return all(p.strip() in ("1", "") for p in parts)


class Architecture(BaseModel):
    family: str                     # "resnet", "vit", "mlp", ...
    input_shape: List[int]          # without batch dim, e.g. [3, 32, 32]
    blocks: List[Block] = Field(default_factory=list)
    graph: List[Layer]              # top-level forward order
    output_dim: Optional[int] = None
    param_count_reported: Optional[int] = None

    def flat_layers(self) -> List[Layer]:
        """The graph with every Block reference expanded in place."""
        by_name = {b.name: b for b in self.blocks}
        out: List[Layer] = []
        for layer in self.graph:
            block = (by_name.get(layer.op.split(":", 1)[1])
                     if layer.op.startswith("Block:") else None)
            if block is None:
                out.append(layer)
            else:
                out.extend(block.layers)
        return out

    @model_validator(mode="after")
    def _structurally_sound(self) -> "Architecture":
        """Reject graphs that cannot build, so the repair loop sees why.

        Both checks are exact, never heuristic: a spec that trips one is wrong
        about itself, not merely unusual. Anything they cannot decide passes.
        """
        declared = {b.name for b in self.blocks}
        referenced = {layer.op.split(":", 1)[1] for layer in self.graph
                      if layer.op.startswith("Block:")}
        missing = sorted(referenced - declared)
        if missing:
            raise ValueError(
                f"architecture.graph references blocks that are not declared "
                f"in architecture.blocks: {', '.join(missing)}")

        mismatch = self._classifier_width_mismatch()
        if mismatch:
            raise ValueError(mismatch)
        return self

    def _classifier_width_mismatch(self) -> Optional[str]:
        """Compare the final Linear's `in_features` with the width feeding it.

        Only decidable when a global pool collapses the spatial dims to 1x1,
        which is the one case where `in_features` is exactly the channel count.
        After a plain Flatten it is C*H*W and depends on the input size, so the
        check is skipped rather than guessed.
        """
        layers = self.flat_layers()
        last = next((i for i in range(len(layers) - 1, -1, -1)
                     if layers[i].op == "Linear"), None)
        if last is None:
            return None
        in_features = _as_int(_arg(layers[last], "in_features"))
        if in_features is None:
            return None
        head = layers[:last]
        if not any(l.op in GLOBAL_POOL and _collapses_to_1x1(l) for l in head):
            return None
        width = None
        for layer in reversed(head):
            width = _as_int(_arg(layer, *WIDTH_KEYS))
            if width is not None:
                break
        if width is None or width == in_features:
            return None
        return (
            f"the final Linear declares in_features={in_features}, but the "
            f"layers before it produce {width} channels and the global pool "
            f"makes in_features exactly that channel count. Either the graph "
            f"is missing the stages that widen {width} to {in_features}, or "
            f"in_features is wrong."
        )


class Dataset(BaseModel):
    """Fixed choices for `task` and `source`.

    Both drive `scaffold/data.py`, so a free-text value there is silently
    ignored at run time: a paper naming its source "ILSVRC 2012" would get
    synthetic data without anyone being told. An enum turns that into a
    visible extraction error instead.
    """
    name: str                       # "CIFAR10", "ImageNet", ...
    task: Literal["image_classification", "text_classification",
                  "tabular_classification", "segmentation",
                  "language_modeling"]
    num_classes: Optional[int] = None
    input_shape: List[int] = Field(default_factory=list)
    source: Literal["torchvision", "synthetic"] = "synthetic"


class Training(BaseModel):
    optimizer: str = "SGD"
    lr: Optional[float] = None
    lr_schedule: Optional[str] = None
    weight_decay: Optional[float] = None
    momentum: Optional[float] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    loss: str = "CrossEntropyLoss"
    augmentations: List[str] = Field(default_factory=list)


class Result(BaseModel):
    metric: str                     # "top1_accuracy", "f1", ...
    value: float
    split: str = "test"


class Evidence(BaseModel):
    """Grounds one extracted field in the paper. No evidence -> treat as a guess."""
    field: str                      # dotted path, e.g. "training.lr"
    quote: str                      # verbatim span from the paper
    confidence: float               # 0..1


class PaperSpec(BaseModel):
    title: str
    architecture: Architecture
    dataset: Dataset
    training: Training
    reported: List[Result] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    # Fields the paper never states. Defaults were applied; both lists are
    # reported to the user and counted during evaluation.
    unresolved: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


def to_runtime_config(spec: PaperSpec) -> dict:
    """The subset the generated repo reads at run time (config.yaml)."""
    return {
        "title": spec.title,
        "dataset": spec.dataset.model_dump(),
        "training": spec.training.model_dump(),
        "input_shape": spec.architecture.input_shape,
        "output_dim": spec.architecture.output_dim or spec.dataset.num_classes,
    }
