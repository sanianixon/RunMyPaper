# RunMyPaper

Turns a machine learning paper into a project that runs.

The pipeline never generates code straight from the paper. It extracts a
typed **PaperSpec** first, then generates from the spec, then proves the result
runs before handing it over.

```
arXiv id ──▶ LaTeX source ─┐
                           ├─▶ extract ─▶ PaperSpec ─▶ codegen ─▶ verify ─▶ project/
paper.pdf ─▶ PDF text ─────┘             (reviewable)            └─ repair loop ─┘
```

Prefer the arXiv path. The e-print source keeps section structure, equations and
architecture tables; PDF text extraction loses all three.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Then pick a model provider. Anything speaking the OpenAI chat-completions
protocol works, so a free endpoint needs no code change:

```bash
# Groq (no card)
export RMP_PROVIDER=openai_compat RMP_BASE_URL=groq  RMP_MODEL=<model> RMP_API_KEY=<key>

# OpenRouter / Gemini compatibility layer -- same shape
export RMP_BASE_URL=openrouter      # or: gemini
export RMP_BASE_URL=http://host/v1  # or a full URL

# Local Ollama, no key needed
export RMP_PROVIDER=openai_compat RMP_BASE_URL=ollama RMP_MODEL=qwen2.5-coder:7b

# Anthropic (the default provider)
export ANTHROPIC_API_KEY=...
```

### Free endpoints, as measured

Published free-tier numbers disagree between sources, so these were established
by running against the endpoints:

| | Gemini `gemini-3.8-flash` | Groq `openai/gpt-oss-120b` |
|---|---|---|
| ceiling that bites | **20 requests/day** | **8,000 tokens/minute** |
| a whole paper (≈38.5k chars) | fits | does not fit |
| latency per extraction | 1–3 min, `503 high demand` is routine | seconds |
| gotcha | `gemini-2.5-flash` is retired for new accounts | Cloudflare `403 code 1010` without a `User-Agent` |

The two fail in opposite ways, which is the useful part: Gemini limits how
*often* you can ask and Groq limits how *much* you can ask at once.

**The `PaperSpec` schema costs about 4,900 tokens** on its own, sent with every
structured request. Measured on Groq: `RMP_MAX_CHARS=13000` produces a 9,196
token request, of which only ~4,300 is paper. So an 8,000 token/minute ceiling
leaves roughly 3,000 tokens for the paper itself — about 9,000 characters of a
38,500 character paper. Whole-paper extraction needs a provider that measures
requests per day rather than tokens per minute.

Both ceilings are handled rather than retried into: a daily allowance raises
`QuotaExhausted` instead of sleeping for minutes, an oversized request raises
`RequestTooLarge` naming `RMP_MAX_CHARS` instead of burning the fallback chain
on three attempts that are all the same size, and a per-minute window sleeps
for exactly as long as the provider says to.

Check it before spending quota:

```bash
.venv/bin/python -c "from runmypaper import llm; print(llm.generate('Reply with OK.', 'ping'))"
```

`parse()` asks for structured output three ways and takes the first that works:
native JSON schema, then JSON mode with the schema in the prompt, then a plain
completion validated with pydantic and retried once with the validation error.
Free endpoints differ in what they support, and small models need the retry.

## Web UI

```bash
.venv/bin/python -m runmypaper.web      # then open the URL it prints
```

Standard library only, so there is nothing extra to install and it needs no
network of its own. The page drives the same functions the CLI calls, so what it
shows is what the system does:

- the eight pipeline stages, lighting up as each one happens
- the extracted spec, every field next to the verbatim quote it came from and
  the model's confidence in it
- the forward graph, with each `Block:` reference expanded in place
- the generated `model.py`
- the four gates, and for a failing one the exact error handed to the repair loop
- a **Schema guard** tab: every spec on disk checked against the schema, with
  the reason any of them is rejected

Runs are cached, so re-opening a finished paper is instant and needs no model
call. `--port` moves it; if the port is taken it takes the next free one.

## Tests

```bash
.venv/bin/python tests/offline.py
```

Covers everything that does not need a provider: ingest, the budget, the
scaffold contract, all four gates, the metrics, and the output parsers.

## Use

```bash
# arXiv id, URL, or a local PDF -- all three go through the same commands
.venv/bin/python -m runmypaper.cli extract 1512.03385 -o out/resnet/spec.json
.venv/bin/python -m runmypaper.cli build out/resnet/spec.json -o out/resnet
.venv/bin/python -m runmypaper.cli run 1512.03385 -o out/resnet

# name the configuration to extract; papers report several
.venv/bin/python -m runmypaper.cli run 1512.03385 -o out/resnet \
    --variant "ResNet-20, CIFAR-10"
```

`--variant` matters more than it looks. ResNet reports both an ImageNet and a
CIFAR model, and extracting "the main experiment" gives the ImageNet one --
faithful to the paper, but not the configuration the benchmark scores. Without
the flag the extractor takes the smallest configuration the paper fully
specifies. The benchmark passes each paper's `note` from `bench/papers.yaml`, so
the extracted spec and the gold spec always describe the same model.

Every LLM call is cached under `.cache/llm`, keyed on the provider, model and
full request, so re-runs are free and switching providers never reads another
provider's answers. `--no-cache` forces fresh calls.

On a metered free tier, `RMP_MAX_CHARS` caps the paper text sent to the
extractor (default 300,000, i.e. no truncation for any real paper). Over budget,
sections are truncated rather than dropped, and method/experiment sections are
kept whole for as long as the budget allows -- dropping a section loses its
architecture detail completely, while truncating keeps the opening, where papers
state their structure. Below roughly 400 characters per section the budget
becomes a target rather than a cap.

## Modules

| File | Role |
|---|---|
| `spec.py` | The IR. Layers, blocks, dataset, training, plus per-field evidence and confidence. Rejects a spec that contradicts itself (see below). |
| `ingest.py` | PDF → sections (PyMuPDF + heading regex), and the source dispatcher. |
| `arxiv.py` | arXiv id → e-print LaTeX → sections, with `\input` inlining. |
| `extract.py` | Sections → `PaperSpec`, as a schema-enforced structured output. |
| `codegen.py` | Spec → project. Static scaffold + one generated `model.py`. |
| `verify.py` | Gates `import → forward → step → train`, then repair on failure. |
| `llm.py` | Provider access (Anthropic or any OpenAI-compatible endpoint), disk cache, structured-output fallbacks. |
| `metrics.py` | Extracted spec vs gold spec: field-level F1, architecture F1. |
| `baseline.py` | The no-IR ablation: paper text straight to `model.py`. |
| `bench.py` | Benchmark runner and results table. |
| `web.py` | Local web UI over the same pipeline functions. Standard library only. |

## A spec that contradicts itself is rejected

`task` and `source` are fixed choices, not free text, because both drive
`data.py`: a spec naming its source `"ILSVRC 2012"` would quietly get synthetic
data and still look correct. Two structural checks run on every spec:

- every `Block:<name>` the graph references must be declared in `blocks`
- where a global pool collapses the spatial dims to 1×1, the final `Linear`'s
  `in_features` must equal the channel width feeding it

The second one is narrow on purpose. After a plain `Flatten`, `in_features` is
`C*H*W` and depends on the input size, so the check does not fire rather than
guess. Both failures surface as a pydantic `ValidationError`, which the
structured-output retry hands back to the model, so an inconsistent spec is
usually repaired without a human seeing it.

This is not hypothetical: the first real extraction of ResNet emitted three
stages of 64 channels and a `Linear(512, 1000)`, which would have failed the
`forward` gate. The schema now catches it at extraction time.

## model.py is compiled, not generated

The spec already names torch.nn classes and their arguments, so writing
`model.py` from it is a translation. [`emit.py`](runmypaper/emit.py) does that
translation with **no model call at all**, which halves the API requests per
paper — extraction is the only one left — and removes a whole class of failure,
because a language model cannot get a dimension wrong where none is involved.

Every layer's *input* width is measured by pushing a probe tensor through the
layers already built:

```python
add(nn.Conv2d(probe.shape[1], 16, kernel_size=3, stride=1, padding=1, bias=False))
add(nn.BatchNorm2d(probe.shape[1]))
```

So `in_channels`, `num_features` and the flattened size before the classifier
are never read from the spec, and a spec that is wrong about its own
`in_features` still compiles to something that runs. The final `Linear` is sized
from `cfg["output_dim"]`, which keeps the project's contract whatever the paper
said.

The compiler refuses rather than guesses. `emit.unsupported(spec)` lists why a
spec cannot be compiled — an op that is not in `torch.nn`, an argument that is
not a Python literal (these arrive from a language model and are pasted into
source, so anything else is rejected), a graph with explicit wiring, or no
`Linear` to size against `output_dim` — and generation falls back to the model
for exactly those.

| `RMP_CODEGEN` | Behaviour |
|---|---|
| `auto` (default) | compile; ask a model only for specs the compiler refuses |
| `emit` | compile only, and fail rather than spend a request |
| `llm` | always generate — what the benchmark's generation arms measure |

A block containing an `Add` layer compiles to a residual shortcut, with a 1×1
projection added only when the body changes shape — which is exactly where the
paper's own diagram shows a dashed line.

## Generated project

`model.py` is the only generated file; it must expose
`build_model(cfg) -> nn.Module`. Everything else (`data.py`, `train.py`,
`smoke.py`) is fixed scaffold driven by `config.yaml`, which keeps the
generation surface small enough to be reliable.

`data.py` falls back to synthetic tensors of the correct shape when the real
dataset cannot be downloaded, so the gates run offline.

## Evaluation

15 papers with public reference implementations, fixed in
[`bench/papers.yaml`](bench/papers.yaml) before any measurement. Each needs a
hand-corrected gold spec in `bench/gold/<name>.json` before it is scored:

```bash
python -m runmypaper.bench gold resnet   # drafts bench/gold/resnet.json
#                                        # then correct it against the paper
python -m runmypaper.bench run
```

Three arms, so each component is credited separately:

| Arm | Spec IR | Repair | Isolates |
|---|---|---|---|
| `no_ir` | no | no | the baseline to beat |
| `no_repair` | yes | no | the value of the spec IR |
| `full` | yes | yes | the value of the repair loop |

```
arm          n  spec_f1  arch_f1   import  forward     step    train  repairs
no_ir        2        -        -     0.50     0.00     0.00     0.00        0
no_repair    2    0.885    0.815     1.00     1.00     0.50     0.50        0
full         2    0.885    0.815     1.00     1.00     1.00     1.00      0.5
```

### A confound this benchmark has to live with

Observed, not hypothesised. A Wide ResNet extraction truncated to 8,000
characters produced a spec with an **empty** `architecture.graph` and thirteen
unresolved fields. The compiler refused it; the generation fallback was then
asked for `model.py` and returned a correct WRN-28-10 with 36,479,194
parameters against the paper's reported 36.5M — from the title and little else.

So on a well-known paper a language model can reproduce the architecture from
memory without reading the paper, which means:

- **passing the gates does not show the spec was right** — the gates prove the
  code runs, never that it is the paper's architecture;
- the `no_ir` arm will look strongest exactly on the famous papers, and a
  benchmark of only famous papers measures memorisation, not reading.

This is the argument for scoring the spec against a hand-labelled gold spec
rather than trusting the gate pass rate, and for eventually holding out papers
the model is unlikely to have memorised.

Extraction errors are split three ways rather than lumped into one score:
`wrong` (read the paper and got it wrong), `missed` (paper states it, we left
it null) and `hallucinated` (paper states nothing, we invented a value). The
last is the one that matters most and the one a single accuracy number hides.

## Not done yet

- **Extraction does not record residual shortcuts.** `emit.py` compiles a
  block's `Add` layer into a shortcut, but nothing asks the extractor to emit
  one, so a ResNet spec currently compiles to a plain CNN with the right widths
  and parameter count. The fix is one rule in `extract.SYSTEM`; it is held back
  only because changing that prompt invalidates every cached extraction.
- No gold specs labelled yet, so nothing is scored. This is the bottleneck.
- Gates run as local subprocesses under this venv's interpreter (so `torch`
  must be installed here), with a timeout but no container isolation.
- Gates run on synthetic batches (`RMP_DATA=synthetic`), so they prove the code
  runs but never exercise the real `torchvision` download path.
- `\newcommand` macros in LaTeX are passed through unexpanded.
- No spec review UI; edit `spec.json` by hand and re-run `build`.
- The `no_ir` arm gets no repair loop, so it is comparable with
  `no_repair` but not with `full`.
