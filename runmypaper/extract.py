"""Sectioned paper text -> PaperSpec."""
from typing import Optional

from .config import EXTRACT_MAX_CHARS
from .ingest import Document
from .llm import parse
from .spec import TORCHVISION_DATASETS, PaperSpec

SYSTEM = """\
You extract an executable specification from a machine learning paper.

Rules:
1. Report only what the paper states. Never invent a hyperparameter.
2. Every value you do extract gets an `evidence` entry: the dotted field path,
   a verbatim quote from the text, and your confidence.
3. Anything the paper omits goes in `unresolved` as a dotted field path, and the
   default you chose instead goes in `assumptions`.
4. `architecture.graph` is the forward pass in execution order. Use exact
   torch.nn class names for `op` ("Conv2d", "BatchNorm2d", "ReLU", "Linear",
   "MaxPool2d", "AdaptiveAvgPool2d", "Flatten", "Dropout").
5. A unit the paper repeats becomes one entry in `architecture.blocks`;
   reference it from the graph with op "Block:<block name>".
6. Layer `args` are literal Python values as strings: "64", "3", "(1, 1)", "True".
7. `inputs` is empty for sequential layers; name the source layers only for
   skip connections or merges.
8. Enumerate the architecture in full. A stage the paper repeats N times gets N
   entries in the graph, and each distinct width gets its own block: a network
   whose channels go 64 -> 128 -> 256 -> 512 has four stages in the graph, not
   one. The final Linear's `in_features` must equal the channel count the
   layers before it produce.
9. `dataset.task` and `dataset.source` are fixed choices, not free text. Use
   source "torchvision" only when the dataset is one of %s,
   spelled exactly that way in `dataset.name`. Every other dataset, ImageNet
   included, is "synthetic".
10. `reported` metric names are snake_case ("top1_error", "top5_error",
    "top1_accuracy", "mean_iou"). Keep the paper's own quantity: record an
    error rate as an error, never silently converted to an accuracy.
""" % ", ".join(TORCHVISION_DATASETS)

TEMPLATE = """\
Paper title: {title}

{body}

Extract the specification for the paper's own proposed model, not a baseline it
compares against.{target}"""

# Most papers report several configurations. Which one is extracted has to be
# stated, or the extractor and the gold spec can disagree about the answer
# while both stay faithful to the paper.
TARGET = """

Target this configuration and no other: {variant}. Take its depth, its dataset
and its hyperparameters, and ignore the paper's other configurations."""

ANY_TARGET = """

Where the paper reports several configurations, take the smallest one it fully
specifies, so the result can be verified cheaply."""


def extract(doc: Document, use_cache: bool = True,
            max_chars: int = EXTRACT_MAX_CHARS,
            variant: Optional[str] = None) -> PaperSpec:
    """Sections -> PaperSpec.

    `variant` names the configuration to extract, e.g. "ResNet-20, CIFAR-10".
    The benchmark passes the `note` from bench/papers.yaml, so the spec being
    scored and the gold spec it is scored against describe the same model.
    """
    # The budget bounds the request; lower RMP_MAX_CHARS when a provider's
    # token quota bites, at a cost in extraction accuracy.
    body = doc.relevant_text(max_chars=max_chars)
    target = TARGET.format(variant=variant) if variant else ANY_TARGET
    prompt = TEMPLATE.format(title=doc.title, body=body, target=target)
    return parse(PaperSpec, SYSTEM, prompt, use_cache=use_cache)
