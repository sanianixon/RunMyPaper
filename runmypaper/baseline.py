"""Ablation: generate model.py straight from the paper, with no spec IR.

This is the comparison the project has to beat. The runtime config is taken
from the gold spec in both arms, so the only variable is whether generation
reads a structured specification or raw paper text.
"""
from pathlib import Path

import yaml

from .codegen import SYSTEM, _readme
from .config import SCAFFOLD_DIR
from .ingest import Document
from .llm import extract_code, generate
from .spec import PaperSpec, to_runtime_config

PROMPT = """\
Paper: {title}

{body}

Write model.py implementing the architecture this paper proposes, for the
configuration in config.yaml:

{config}
"""


def build(doc: Document, config_source: PaperSpec, out: Path,
          use_cache: bool = True) -> Path:
    """Scaffold + a model.py generated from paper text rather than a spec."""
    import shutil

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("data.py", "train.py", "smoke.py", "requirements.txt"):
        shutil.copy(SCAFFOLD_DIR / name, out / name)
    cfg = to_runtime_config(config_source)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "README.md").write_text(_readme(config_source))

    prompt = PROMPT.format(title=doc.title, body=doc.relevant_text(),
                           config=yaml.safe_dump(cfg, sort_keys=False))
    code = generate(SYSTEM, prompt, use_cache=use_cache)
    (out / "model.py").write_text(extract_code(code))
    return out
