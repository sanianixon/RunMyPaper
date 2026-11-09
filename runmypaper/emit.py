"""PaperSpec -> model.py, by compilation rather than generation.

The IR already names torch.nn classes and their arguments, so for most specs
writing model.py is a translation, not a judgement. This module does it with no
model call at all, which removes the second API request per paper and with it a
whole class of failure: a language model cannot get a dimension wrong here
because no language model is involved.

What it will not do is guess. `unsupported()` lists the reasons a spec cannot be
compiled -- an op that is not in torch.nn, an argument that is not a literal, a
graph with explicit wiring -- and `codegen` falls back to the model for those.
Every layer's *input* width comes from a shape probe rather than from the spec,
so a spec that is wrong about its own `in_features` still compiles correctly.
"""
import ast
import inspect
import keyword
import re
from typing import Dict, List, Optional, Tuple

import torch.nn as nn

from .spec import Architecture, Block, Layer, PaperSpec

# A layer that sums the block's input back in: the residual shortcut. Not a
# torch.nn class, so it is handled as a special case.
ADD = "Add"

# Arguments describing a layer's *input* width. Dropped from the emitted call
# and taken from the running shape probe instead, which is the only source that
# cannot disagree with the layers actually built.
DERIVED_ARGS = ("in_channels", "in_features", "num_features", "input_size",
                "normalized_shape", "num_embeddings")

# The argument naming a layer's output width, by op.
OUT_ARG = {"Conv2d": "out_channels", "Conv1d": "out_channels",
           "ConvTranspose2d": "out_channels", "Linear": "out_features"}


def _is_module(op: str) -> bool:
    cls = getattr(nn, op, None)
    return isinstance(cls, type) and issubclass(cls, nn.Module)


def _args(layer: Layer) -> Dict[str, str]:
    return {kv.key: kv.value for kv in layer.args}


def _missing_required(layer: Layer, supplied: Tuple[str, ...] = ()) -> List[str]:
    """Constructor arguments with no default that the spec does not provide.

    Read from the real torch signature rather than a table kept here, so a spec
    missing `kernel_size` is refused instead of compiled into a call that
    raises at build time.
    """
    cls = getattr(nn, layer.op, None)
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return []
    args = _args(layer)
    missing = []
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        if name in DERIVED_ARGS or name in args or name in supplied:
            continue
        missing.append(name)
    return missing


def _literal(value: str) -> Optional[str]:
    """The value re-rendered as a Python literal, or None if it is not one.

    Values arrive as strings written by a language model and are pasted into
    generated source, so anything that is not a plain literal is refused rather
    than emitted.
    """
    try:
        return repr(ast.literal_eval(value.strip()))
    except (ValueError, SyntaxError, MemoryError, TypeError):
        return None


def _ident(name: str, used: set) -> str:
    """A unique, valid Python class name derived from a block's spec name."""
    cleaned = re.sub(r"\W", "_", name).strip("_") or "Block"
    if not cleaned[0].isalpha() or keyword.iskeyword(cleaned) or hasattr(nn, cleaned):
        cleaned = "Block_" + cleaned
    candidate, n = cleaned, 2
    while candidate in used:
        candidate, n = f"{cleaned}_{n}", n + 1
    used.add(candidate)
    return candidate


# --- can this spec be compiled? ---------------------------------------------

def unsupported(spec: PaperSpec) -> List[str]:
    """Every reason this spec cannot be compiled. Empty means it can."""
    arch = spec.architecture
    reasons: List[str] = []
    blocks = {b.name: b for b in arch.blocks}

    if not arch.input_shape:
        reasons.append("architecture.input_shape is empty")
    if not arch.graph:
        reasons.append("architecture.graph is empty")

    flat = arch.flat_layers()
    linears = [l for l in flat if l.op == "Linear"]
    if not linears:
        reasons.append("the graph has no Linear layer, so there is no "
                       "classifier to size against output_dim")
    # The final Linear is sized from cfg["output_dim"], so it needs no
    # out_features of its own.
    last_linear = linears[-1] if linears else None

    for layer in arch.graph:
        if layer.inputs:
            reasons.append(
                f"layer {layer.name!r} names explicit inputs "
                f"({', '.join(layer.inputs)}); only a sequential top-level "
                f"graph can be compiled")
        if layer.op.startswith("Block:"):
            if layer.op.split(":", 1)[1] not in blocks:
                reasons.append(f"layer {layer.name!r} references undeclared block")
            continue
        reasons += _layer_reasons(layer, top_level=True,
                                  last_linear=last_linear)

    for block in arch.blocks:
        for layer in block.layers:
            if layer.op == ADD:
                continue
            reasons += _layer_reasons(layer, top_level=False,
                                      where=f"block {block.name!r} ",
                                      last_linear=last_linear)
    return reasons


def _layer_reasons(layer: Layer, top_level: bool, where: str = "",
                   last_linear: Optional[Layer] = None) -> List[str]:
    out: List[str] = []
    if layer.op == ADD and top_level:
        out.append(f"{where}layer {layer.name!r} is an Add outside a block, "
                   f"which needs explicit wiring")
        return out
    if not _is_module(layer.op):
        out.append(f"{where}op {layer.op!r} is not a torch.nn module")
        return out
    for key, value in _args(layer).items():
        if key in DERIVED_ARGS:
            continue
        if _literal(value) is None:
            out.append(f"{where}layer {layer.name!r} argument {key}="
                       f"{value!r} is not a Python literal")
    supplied = ("out_features",) if layer is last_linear else ()
    missing = _missing_required(layer, supplied)
    if missing:
        out.append(f"{where}layer {layer.name!r} ({layer.op}) is missing "
                   f"required argument(s): {', '.join(missing)}")
    return out


# --- emission ----------------------------------------------------------------

def _call(layer: Layer, probe: str, output_dim: Optional[str]) -> str:
    """One constructor call, with input widths read from the probe."""
    args = _args(layer)
    parts: List[str] = []

    out_key = OUT_ARG.get(layer.op)
    if layer.op in ("Conv2d", "Conv1d", "ConvTranspose2d"):
        parts.append(f"{probe}.shape[1]")
    elif layer.op == "Linear":
        parts.append(f"{probe}.shape[1]")
    elif layer.op in ("BatchNorm2d", "BatchNorm1d", "InstanceNorm2d", "GroupNorm"):
        # GroupNorm takes num_groups first; only num_channels is derived.
        if layer.op == "GroupNorm":
            parts.append(_literal(args["num_groups"]))
        parts.append(f"{probe}.shape[1]")
    elif layer.op == "LayerNorm":
        parts.append(f"{probe}.shape[1]")

    if out_key:
        # The final Linear must produce exactly cfg["output_dim"], whatever the
        # paper said, or the project's own contract breaks.
        if output_dim is not None:
            parts.append(output_dim)
        else:
            width = _literal(args.get(out_key, "")) if out_key in args else None
            if width is None:
                parts.append(f"{probe}.shape[1]")
            else:
                parts.append(width)

    skip = set(DERIVED_ARGS) | {out_key} | {"num_groups"}
    for key, value in args.items():
        if key in skip:
            continue
        lit = _literal(value)
        if lit is not None:
            parts.append(f"{key}={lit}")
    return f"nn.{layer.op}({', '.join(parts)})"


def _emit_sequence(layers: List[Layer], names: Dict[str, str],
                   last_linear: Optional[Layer], indent: str) -> str:
    """Statements that construct each layer and push the probe through it."""
    lines = []
    for layer in layers:
        if layer.op == ADD:
            continue
        if layer.op.startswith("Block:"):
            cls = names[layer.op.split(":", 1)[1]]
            call = f"{cls}(tuple(probe.shape[1:]))"
        else:
            call = _call(layer, "probe",
                         "output_dim" if layer is last_linear else None)
        lines.append(f"{indent}add({call})  # {layer.name}")
    return "\n".join(lines)


def _emit_block(block: Block, cls: str, names: Dict[str, str]) -> str:
    residual = any(l.op == ADD for l in block.layers)
    body = _emit_sequence(block.layers, names, None, " " * 12)
    doc = (f"{len(block.layers)} layers from the specification"
           + (", with an identity shortcut." if residual else "."))
    shortcut = '''
        # The shortcut only needs a projection when the body changes shape,
        # which is exactly when the paper's own diagram shows a dashed line.
        self.shortcut = None
        if tuple(out.shape[1:]) != tuple(in_shape):
            stride = max(1, in_shape[1] // out.shape[2]) if len(in_shape) > 2 else 1
            self.shortcut = nn.Conv2d(in_shape[0], out.shape[1], kernel_size=1,
                                      stride=stride, bias=False)
''' if residual else ""
    forward = ('''        y = self.body(x)
        return y + (x if self.shortcut is None else self.shortcut(x))'''
               if residual else "        return self.body(x)")
    return f'''class {cls}(nn.Module):
    """{doc}"""

    def __init__(self, in_shape):
        super().__init__()
        mods = []
        with torch.no_grad():
            probe = torch.zeros(1, *in_shape)

            def add(m):
                nonlocal probe
                mods.append(m)
                probe = m(probe)

{body}
            out = probe
        self.body = nn.Sequential(*mods)
{shortcut}
    def forward(self, x):
{forward}
'''


def emit(spec: PaperSpec) -> str:
    """model.py for this spec. Raises ValueError if it cannot be compiled."""
    reasons = unsupported(spec)
    if reasons:
        raise ValueError("; ".join(reasons))

    arch: Architecture = spec.architecture
    used: set = set()
    names = {b.name: _ident(b.name, used) for b in arch.blocks}

    flat = arch.flat_layers()
    last_linear = next((l for l in reversed(flat) if l.op == "Linear"), None)

    blocks = "\n\n".join(_emit_block(b, names[b.name], names) for b in arch.blocks)
    graph = _emit_sequence(arch.graph, names, last_linear, " " * 12)

    header = f'''"""{spec.title}

Compiled by RunMyPaper directly from the specification in spec.json. No
language model wrote this file: every layer type and argument below is a
translation of a spec entry, and every input width is measured by pushing a
probe tensor through the layers already built, so the file cannot disagree with
itself about a dimension.
"""
import torch
import torch.nn as nn
'''

    return f'''{header}

{blocks}{"""

""" if blocks else ""}class Model(nn.Module):
    """{arch.family} as specified: {len(arch.graph)} top-level layers."""

    def __init__(self, input_shape, output_dim):
        super().__init__()
        mods = []
        with torch.no_grad():
            probe = torch.zeros(1, *input_shape)

            def add(m):
                nonlocal probe
                mods.append(m)
                probe = m(probe)

{graph}
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def build_model(cfg):
    return Model(tuple(cfg["input_shape"]), int(cfg["output_dim"]))
'''
