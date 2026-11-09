import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path) -> None:
    """Read KEY=VALUE lines from .env without adding a dependency.

    Real environment variables win, so an export always overrides the file.

    Within the file the *last* occurrence of a key wins, because switching
    provider by pasting new lines at the end is the obvious thing to do and
    silently honouring the old ones above is the worst way to respond to it.
    Duplicates are reported, since one of the two is always dead.
    """
    if not path.is_file():
        return
    values: dict = {}
    duplicates: list = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if not key:
            continue
        if key in values and key not in duplicates:
            duplicates.append(key)
        values[key] = value          # last one wins

    if duplicates:
        print(f"warning: {path.name} sets {', '.join(duplicates)} more than "
              f"once; the last value of each is used. Delete the earlier "
              f"lines to stop guessing which is live.", file=sys.stderr)

    for key, value in values.items():
        if key not in os.environ:
            os.environ[key] = value


_load_env(ROOT / ".env")

CACHE_DIR = Path(os.environ.get("RMP_CACHE", ROOT / ".cache" / "llm"))
SCAFFOLD_DIR = Path(__file__).resolve().parent / "scaffold"

# --- provider ---------------------------------------------------------------
# "anthropic"     -- the Anthropic SDK, credentials from ANTHROPIC_API_KEY
# "openai_compat" -- any OpenAI chat-completions endpoint, set via RMP_BASE_URL.
#                    Covers Gemini's compatibility layer, Groq, OpenRouter and
#                    a local Ollama, so one backend serves every free option.
PROVIDER = os.environ.get("RMP_PROVIDER", "anthropic")

# Known endpoints, so RMP_BASE_URL can be a short name instead of a URL.
BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}

_base = os.environ.get("RMP_BASE_URL", "")
BASE_URL = BASE_URLS.get(_base, _base)

# Ollama needs no real key but the protocol requires the header.
API_KEY = os.environ.get("RMP_API_KEY") or ("ollama" if _base == "ollama" else "")

DEFAULT_MODELS = {"anthropic": "claude-opus-5-5"}
MODEL_ID = os.environ.get("RMP_MODEL") or DEFAULT_MODELS.get(PROVIDER, "")

REQUEST_TIMEOUT_S = int(os.environ.get("RMP_TIMEOUT", "600"))

# How model.py is written:
#   "auto" -- compile the spec directly, and ask a model only for the specs the
#             compiler refuses (the default; one API request per paper instead
#             of two, and no model can get a dimension wrong)
#   "emit" -- compile only; fail rather than spend a request
#   "llm"  -- always generate, which is what the benchmark's generation arms
#             measure
CODEGEN = os.environ.get("RMP_CODEGEN", "auto")

# --- limits -----------------------------------------------------------------
# Hard ceilings so a bad generation can't hang the pipeline.
GATE_TIMEOUT_S = int(os.environ.get("RMP_GATE_TIMEOUT", "300"))
MAX_REPAIRS = int(os.environ.get("RMP_MAX_REPAIRS", "3"))

# Characters of paper text sent to the extractor. The default is high enough
# that no real paper is truncated: truncation costs extraction accuracy, which
# is the thing being measured. Lower it only if a provider's token quota bites.
EXTRACT_MAX_CHARS = int(os.environ.get("RMP_MAX_CHARS", "300000"))

# Downloaded arXiv sources, one directory per paper.
ARXIV_DIR = Path(os.environ.get("RMP_ARXIV", ROOT / ".cache" / "arxiv"))


def summary() -> dict:
    """Provider settings for display. Never includes the key itself."""
    return {
        "provider": PROVIDER,
        "base_url": BASE_URL or "(unset)",
        "model": MODEL_ID or "(unset)",
        "api_key": f"set, {len(API_KEY)} chars" if API_KEY else "NOT SET",
        "max_chars": EXTRACT_MAX_CHARS,
        "cache": str(CACHE_DIR),
    }
