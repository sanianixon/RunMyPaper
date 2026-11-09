"""arXiv LaTeX source -> Document.

Preferred over the PDF path whenever a paper is on arXiv: the e-print source
gives real section structure, real equations and real tables, none of which
survive two-column PDF text extraction intact.
"""
import gzip
import io
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Optional

from .config import ARXIV_DIR
from .ingest import Document, Section

EPRINT = "https://arxiv.org/e-print/{id}"
# arXiv rejects the default urllib agent.
UA = "RunMyPaper/0.1 (academic project; contact via arxiv.org)"

# New style (2512.03385v2) and old style (cs/0701001, math.GT/0309136).
ID_RE = re.compile(
    r"(?:arxiv[:/]|abs/|pdf/|e-print/)?"
    r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)

SECTION_RE = re.compile(r"\\(?:sub)?section\*?\s*\{")
# ICML/CVPR templates use their own title macro.
TITLE_RE = re.compile(r"\\(?:icml|toc)?title\s*\{")
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")

# Environments that carry no specification detail.
DROP_ENVS = ("thebibliography", "figure", "figure*", "acks", "acknowledgments")


class NoSourceError(RuntimeError):
    """The paper has no LaTeX source on arXiv (PDF-only submission)."""


def normalise_id(ref: str) -> str:
    m = ID_RE.search(ref.strip())
    if not m:
        raise ValueError(f"not an arXiv id or URL: {ref!r}")
    return m.group(1)


def looks_like_arxiv(ref: str) -> bool:
    return not Path(ref).exists() and ID_RE.search(ref) is not None


# --- download ---------------------------------------------------------------

def fetch(arxiv_id: str, dest: Optional[Path] = None) -> Path:
    """Download and unpack the e-print into its own empty directory."""
    dest = Path(dest or ARXIV_DIR / arxiv_id.replace("/", "_"))
    if dest.exists() and any(dest.glob("*.tex")):
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(EPRINT.format(id=arxiv_id),
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        blob = r.read()

    if blob[:4] == b"%PDF":
        raise NoSourceError(f"{arxiv_id} is a PDF-only submission; use the PDF path")

    try:
        with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
            # data_filter rejects absolute paths, traversal and special files.
            tf.extractall(dest, filter="data")
    except tarfile.ReadError:
        # Single-file submission: a bare gzipped .tex.
        (dest / "main.tex").write_bytes(gzip.decompress(blob))
    return dest


# --- LaTeX -> text ----------------------------------------------------------

def _braced(text: str, open_at: int) -> tuple[str, int]:
    """Content of the brace group starting at `open_at`, and the index after it."""
    depth, i = 0, open_at
    while i < len(text):
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:i], i + 1
        i += 1
    return text[open_at + 1:], len(text)


def _strip_comments(tex: str) -> str:
    out = []
    for line in tex.splitlines():
        i, esc = 0, False
        while i < len(line):
            if line[i] == "\\":
                esc = not esc
            elif line[i] == "%" and not esc:
                line = line[:i]
                break
            else:
                esc = False
            i += 1
        out.append(line)
    return "\n".join(out)


def _drop_envs(tex: str) -> str:
    for env in DROP_ENVS:
        tex = re.sub(rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}",
                     "", tex, flags=re.DOTALL)
    # A bibliography may be unterminated at end of file.
    return re.split(r"\\bibliography\s*\{|\\begin\{thebibliography\}", tex)[0]


def _main_tex(root: Path) -> Path:
    candidates = [p for p in sorted(root.rglob("*.tex"))
                  if "\\documentclass" in p.read_text(errors="replace")]
    if not candidates:
        candidates = sorted(root.rglob("*.tex"))
    if not candidates:
        raise NoSourceError(f"no .tex file under {root}")
    return max(candidates, key=lambda p: p.stat().st_size)


def _inline_inputs(path: Path, root: Path, seen: set) -> str:
    if path in seen:
        return ""
    seen.add(path)
    tex = _strip_comments(path.read_text(errors="replace"))

    def sub(m):
        name = m.group(1).strip()
        for cand in (root / name, root / f"{name}.tex", path.parent / f"{name}.tex"):
            if cand.is_file():
                return _inline_inputs(cand, root, seen)
        return ""

    return INPUT_RE.sub(sub, tex)


def _sectionise(tex: str) -> List[Section]:
    sections: List[Section] = []

    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
    if m:
        sections.append(Section("Abstract", m.group(1).strip()))

    marks = []
    for m in SECTION_RE.finditer(tex):
        name, after = _braced(tex, m.end() - 1)
        marks.append((m.start(), after, re.sub(r"\s+", " ", name).strip()))

    for i, (_, body_from, name) in enumerate(marks):
        body_to = marks[i + 1][0] if i + 1 < len(marks) else len(tex)
        body = tex[body_from:body_to].strip()
        if body:
            sections.append(Section(name, body))
    return sections


def to_document(root: Path, arxiv_id: str) -> Document:
    main = _main_tex(root)
    tex = _inline_inputs(main, root, set())
    tex = tex.split(r"\end{document}")[0]
    tex = _drop_envs(tex)

    title = arxiv_id
    m = TITLE_RE.search(tex)
    if m:
        title = _braced(tex, m.end() - 1)[0]
        title = re.sub(r"\\\\|\\[a-zA-Z]+\s*|[{}]", " ", title)
        title = re.sub(r"\s+", " ", title).strip() or arxiv_id

    sections = _sectionise(tex)
    if not sections:
        raise NoSourceError(f"no sections found in {main}")
    return Document(title=title, sections=sections)


def load(ref: str) -> Document:
    arxiv_id = normalise_id(ref)
    return to_document(fetch(arxiv_id), arxiv_id)
