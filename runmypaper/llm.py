"""Model access: disk cache, structured parse, long-form generation.

Two public calls, `parse()` and `generate()`, so swapping providers touches
only this file. Everything is cached on disk keyed on the full request
(provider and model included), so re-running a paper costs nothing and
switching providers never reads another provider's answers.

Set RMP_PROVIDER=openai_compat with RMP_BASE_URL to run on a free endpoint:
Groq, OpenRouter, Gemini's compatibility layer or a local Ollama.
"""
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .config import (API_KEY, BASE_URL, CACHE_DIR, EXTRACT_MAX_CHARS,
                     MODEL_ID, PROVIDER, REQUEST_TIMEOUT_S)

T = TypeVar("T", bound=BaseModel)

# Providers behind Cloudflare reject urllib's default agent outright (403 with
# Cloudflare code 1010), so every request identifies itself. Same reason
# arxiv.py sets one.
UA = "RunMyPaper/0.1 (academic project)"


class LLMRefusal(RuntimeError):
    """The model declined to answer."""


class LLMError(RuntimeError):
    """The provider rejected the request, or returned something unusable."""


class ModelUnavailable(LLMError):
    """The endpoint does not serve this model; no fallback will help."""


class QuotaExhausted(LLMError):
    """A per-day free-tier allowance is used up. Waiting will not help today."""


class RequestTooLarge(LLMError):
    """Over the endpoint's token ceiling; the input has to shrink."""


# A 429 is either a per-minute rate limit, which a backoff fixes, or a daily
# allowance, which it does not. The provider says which in the error body.
DAILY_QUOTA = ("per day", "requests per day", "daily limit", "free_tier_requests",
               "exceeded your current quota")


def _is_daily_quota(detail: str) -> bool:
    low = detail.lower()
    return any(marker in low for marker in DAILY_QUOTA)


def _retry_after(exc, detail: str) -> Optional[float]:
    """How long the provider asked us to wait, when it says so.

    A tokens-per-minute limit clears on the provider's clock, not ours, and
    both Groq and Gemini state the wait. Honouring it turns a failure into a
    success; guessing an exponential backoff either gives up too early or
    sleeps far longer than needed.
    """
    header = exc.headers.get("retry-after") if getattr(exc, "headers", None) else None
    if header:
        try:
            return float(header)
        except (TypeError, ValueError):
            pass
    match = re.search(r"try again in ([\d.]+)\s*s", detail, re.IGNORECASE)
    return float(match.group(1)) if match else None


# --- cache ------------------------------------------------------------------

def _key(*parts: str) -> Path:
    h = hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:32]
    return CACHE_DIR / f"{h}.json"


def _store(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


# --- shared helpers ---------------------------------------------------------

def json_from_text(text: str) -> dict:
    """The first JSON object in a reply, tolerating prose or a code fence."""
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start == -1:
        raise LLMError(f"no JSON object in reply: {text[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(candidate)):
        c = candidate[i]
        if in_str:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                in_str = False
            continue
        if c == '"':
            in_str, esc = True, False
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start:i + 1])
    raise LLMError("JSON object in reply is not terminated")


def extract_code(text: str) -> str:
    """Strip a markdown fence if the model wrapped the file in one."""
    if "```" not in text:
        return text.strip() + "\n"
    body = text.split("```", 2)[1]
    if "\n" in body:
        first, rest = body.split("\n", 1)
        if first.strip() in ("python", "py", ""):
            body = rest
    return body.rstrip() + "\n"


SCHEMA_INSTRUCTION = """\
Reply with a single JSON object and nothing else -- no prose, no code fence.
It must validate against this JSON Schema:

{schema}"""


# --- Anthropic backend ------------------------------------------------------

_anthropic_client = None

NO_CREDS = SystemExit(
    "No Anthropic credentials found. Set ANTHROPIC_API_KEY, run `ant auth "
    "login`, or use a free endpoint with RMP_PROVIDER=openai_compat "
    "RMP_BASE_URL=groq|openrouter|gemini|ollama."
)


def anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _anthropic_check(resp) -> None:
    if resp.stop_reason == "refusal":
        raise LLMRefusal(getattr(resp.stop_details, "explanation", "model declined"))


def _anthropic_parse(schema: Type[T], system: str, prompt: str) -> T:
    try:
        with anthropic_client().messages.stream(
            model=MODEL_ID,
            max_tokens=32000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,          # native structured output
            output_config={"effort": "high"},
        ) as stream:
            resp = stream.get_final_message()
    except TypeError as exc:               # credentials resolve lazily
        raise NO_CREDS from exc
    _anthropic_check(resp)
    return resp.parsed_output


def _anthropic_generate(system: str, prompt: str, effort: str) -> str:
    import anthropic

    kwargs = dict(
        model=MODEL_ID,
        max_tokens=64000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": effort},
    )
    # Routes refusals to a fallback model server-side; dropped if the account
    # does not have the beta.
    fallback = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    try:
        with anthropic_client().beta.messages.stream(**kwargs, **fallback) as s:
            resp = s.get_final_message()
    except anthropic.BadRequestError:
        with anthropic_client().messages.stream(**kwargs) as s:
            resp = s.get_final_message()
    except TypeError as exc:
        raise NO_CREDS from exc
    _anthropic_check(resp)
    return "".join(b.text for b in resp.content if b.type == "text")


# --- OpenAI-compatible backend ----------------------------------------------

def _post(body: dict, retries: int = 3) -> dict:
    if not BASE_URL:
        raise SystemExit(
            "RMP_PROVIDER=openai_compat needs RMP_BASE_URL "
            "(groq, openrouter, gemini, ollama, or a full URL)."
        )
    if not MODEL_ID:
        raise SystemExit("RMP_PROVIDER=openai_compat needs RMP_MODEL.")

    req = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}",
                 "User-Agent": UA},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            if exc.code == 429 and _is_daily_quota(detail):
                # Retrying burns minutes for nothing. Say what actually helps.
                raise QuotaExhausted(
                    f"{MODEL_ID} has used up its free-tier allowance for today "
                    f"at {BASE_URL}. Either wait for the daily reset, or point "
                    f"RMP_BASE_URL/RMP_MODEL/RMP_API_KEY at another free "
                    f"endpoint -- groq, openrouter or a local ollama -- which "
                    f"needs no code change.\n\n{detail}") from exc
            # A per-minute 429 and any 5xx are transient; 503 "high demand" is
            # routine on free tiers.
            if (exc.code == 429 or exc.code >= 500) and attempt < retries - 1:
                asked = _retry_after(exc, detail)
                # The extra second keeps us clear of the boundary the provider
                # is measuring against.
                time.sleep(min(asked + 1 if asked else 2 ** attempt * 5, 90))
                continue
            if exc.code == 404:
                # Wrong or retired model name. Retrying another output mode
                # cannot fix it, so stop before burning the fallback chain.
                raise ModelUnavailable(
                    f"{MODEL_ID} is not served by {BASE_URL}: {detail}") from exc
            if exc.code == 413:
                # Over the provider's per-request or per-minute token ceiling.
                # Deterministic: no output mode is smaller, so say what to turn
                # down instead of trying the other two.
                raise RequestTooLarge(
                    f"the request is larger than {MODEL_ID} accepts at "
                    f"{BASE_URL}. Lower RMP_MAX_CHARS (currently "
                    f"{EXTRACT_MAX_CHARS:,}) so less of the paper is sent; "
                    f"sections are truncated rather than dropped, and the "
                    f"method and experiment sections are kept longest."
                    f"\n\n{detail}") from exc
            raise LLMError(f"HTTP {exc.code} from {BASE_URL}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise LLMError(f"cannot reach {BASE_URL}: {exc.reason}") from exc
    raise AssertionError("unreachable")


def _content(resp: dict) -> str:
    try:
        choice = resp["choices"][0]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected response shape: {str(resp)[:300]}") from exc
    if choice.get("finish_reason") == "content_filter":
        raise LLMRefusal("provider content filter")
    return choice["message"]["content"] or ""


def _oai_generate(system: str, prompt: str) -> str:
    return _content(_post({
        "model": MODEL_ID,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
    }))


def _oai_parse(schema: Type[T], system: str, prompt: str) -> T:
    """Structured output with three fallbacks, since free endpoints differ.

    1. native json_schema
    2. json_object mode with the schema in the prompt
    3. plain completion, validate, and retry once with the validation error
    """
    js = schema.model_json_schema()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    base = {"model": MODEL_ID, "messages": messages}

    attempts = [
        dict(base, response_format={
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": js, "strict": True},
        }),
        dict(base, response_format={"type": "json_object"},
             messages=messages[:-1] + [{
                 "role": "user",
                 "content": prompt + "\n\n" + SCHEMA_INSTRUCTION.format(
                     schema=json.dumps(js)),
             }]),
        dict(base, messages=messages[:-1] + [{
            "role": "user",
            "content": prompt + "\n\n" + SCHEMA_INSTRUCTION.format(
                schema=json.dumps(js)),
        }]),
    ]

    last: Optional[Exception] = None
    for body in attempts:
        try:
            text = _content(_post(body))
        except (ModelUnavailable, QuotaExhausted, RequestTooLarge):
            raise                           # no output mode will fix these
        except LLMError as exc:
            last = exc                      # endpoint rejected this mode
            continue
        try:
            return schema.model_validate(json_from_text(text))
        except (ValidationError, LLMError) as exc:
            last = exc
            # One repair pass: show the model what failed validation.
            retry = dict(body, messages=body["messages"] + [
                {"role": "assistant", "content": text},
                {"role": "user", "content":
                    f"That did not validate:\n\n{str(exc)[:2000]}\n\n"
                    "Reply with the corrected JSON object only."},
            ])
            try:
                return schema.model_validate(json_from_text(_content(_post(retry))))
            except (ValidationError, LLMError) as exc2:
                last = exc2
    raise LLMError(f"{MODEL_ID} could not produce a valid {schema.__name__}: {last}")


# --- public API -------------------------------------------------------------

def parse(schema: Type[T], system: str, prompt: str, use_cache: bool = True) -> T:
    """One structured-output call, validated against a pydantic model."""
    path = _key("parse", PROVIDER, MODEL_ID, schema.__name__, system, prompt)
    if use_cache and path.exists():
        return schema.model_validate_json(path.read_text())

    if PROVIDER == "anthropic":
        out = _anthropic_parse(schema, system, prompt)
    elif PROVIDER == "openai_compat":
        out = _oai_parse(schema, system, prompt)
    else:
        raise SystemExit(f"unknown RMP_PROVIDER: {PROVIDER!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out.model_dump_json(indent=2))
    return out


def generate(system: str, prompt: str, effort: str = "high",
             use_cache: bool = True) -> str:
    """Long-form text, used for a generated source file."""
    path = _key("gen", PROVIDER, MODEL_ID, effort, system, prompt)
    if use_cache and path.exists():
        return json.loads(path.read_text())["text"]

    if PROVIDER == "anthropic":
        text = _anthropic_generate(system, prompt, effort)
    elif PROVIDER == "openai_compat":
        text = _oai_generate(system, prompt)
    else:
        raise SystemExit(f"unknown RMP_PROVIDER: {PROVIDER!r}")

    _store(path, {"text": text})
    return text


def list_models() -> list:
    """Model ids the endpoint offers, for picking RMP_MODEL."""
    if PROVIDER != "openai_compat":
        return [MODEL_ID]
    req = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {API_KEY}", "User-Agent": UA},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raise LLMError(f"HTTP {exc.code} listing models: "
                       f"{exc.read().decode(errors='replace')[:300]}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"cannot reach {BASE_URL}: {exc.reason}") from exc
    return sorted(m.get("id", "?") for m in data.get("data", []))
