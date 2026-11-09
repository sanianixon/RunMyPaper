"""PDF -> sectioned text.

Deliberately simple: PyMuPDF text in reading order, then a heading-regex
sectioniser. Swap in GROBID or arXiv LaTeX behind `load()` when accuracy on
two-column layouts becomes the bottleneck.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pymupdf

# Numbered or all-caps headings on their own line.
HEADING = re.compile(
    r"^\s*(?:(\d+(?:\.\d+)*)\s+)?([A-Z][A-Za-z0-9 \-:&]{2,60})\s*$"
)

# Sections that never carry specification detail. Dropping these is safer than
# keeping a whitelist: papers name their method section after their method
# ("Residual Learning", "Scaled Dot-Product Attention"), so a keyword whitelist
# silently discards the most important text.
DROP = ("related work", "acknowledg", "reference", "broader impact",
        "ethics statement", "limitations")

# Sections kept whole when the paper has to be trimmed to a budget.
PRIORITY = ("method", "approach", "architecture", "model", "experiment",
            "implementation", "setup", "training", "result", "abstract")

TRUNCATION_MARK = "\n[... truncated ...]"
MIN_SECTION_CHARS = 400          # below this a section carries nothing useful


def _join(sections: List["Section"]) -> str:
    return "\n\n".join(f"## {s.name}\n{s.text}" for s in sections)


def _clip(text: str, limit: int) -> str:
    """Cut to `limit`, preferring a paragraph then a sentence boundary."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    for sep in ("\n\n", ". "):
        cut = head.rfind(sep)
        if cut > limit * 0.5:
            head = head[:cut]
            break
    return head.rstrip() + TRUNCATION_MARK


@dataclass
class Section:
    name: str
    text: str


@dataclass
class Document:
    title: str
    sections: List[Section] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return _join(self.sections)

    def relevant_text(self, max_chars: Optional[int] = None) -> str:
        """Everything but the boilerplate, trimmed to a character budget.

        Over budget, sections are truncated rather than dropped: a dropped
        section loses its architecture detail completely, whereas a truncated
        one keeps its opening, which is where papers state their structure.
        Priority sections are kept whole for as long as the budget allows.
        """
        keep = [s for s in self.sections
                if not any(d in s.name.lower() for d in DROP)]
        text = _join(keep)
        if max_chars is None or len(text) <= max_chars:
            return text

        is_priority = [any(w in s.name.lower() for w in PRIORITY) for s in keep]
        fixed = sum(len(s.text) for s, p in zip(keep, is_priority) if p)
        others = sum(1 for p in is_priority if not p)
        share = max((max_chars - fixed) // others, MIN_SECTION_CHARS) if others else 0

        trimmed = [s if p else Section(s.name, _clip(s.text, share))
                   for s, p in zip(keep, is_priority)]
        text = _join(trimmed)
        if len(text) <= max_chars:
            return text

        # The priority sections alone overflow; scale every section down.
        ratio = max_chars / len(text)
        return _join([Section(s.name, _clip(s.text, max(int(len(s.text) * ratio),
                                                        MIN_SECTION_CHARS)))
                      for s in trimmed])


def _sectionise(raw: str) -> List[Section]:
    sections, name, buf = [], "Preamble", []
    for line in raw.splitlines():
        m = HEADING.match(line)
        # A heading line is short and is not a sentence.
        if m and len(line.strip()) < 60 and not line.strip().endswith("."):
            if buf:
                sections.append(Section(name, "\n".join(buf).strip()))
            name, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append(Section(name, "\n".join(buf).strip()))
    return [s for s in sections if s.text]


def load(path: Path) -> Document:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as doc:
            raw = "\n".join(page.get_text("text") for page in doc)
            title = (doc.metadata or {}).get("title") or path.stem
    else:
        raw = path.read_text(errors="replace")
        title = path.stem
    # Drop hyphenation introduced by line wrapping.
    raw = re.sub(r"-\n(\w)", r"\1", raw)
    return Document(title=title.strip() or path.stem, sections=_sectionise(raw))


def load_source(ref: str) -> Document:
    """Dispatch on the reference: arXiv id or URL, else a local file."""
    from . import arxiv  # imported here; arxiv imports Document from this module

    if arxiv.looks_like_arxiv(ref):
        return arxiv.load(ref)
    path = Path(ref)
    if not path.is_file():
        raise SystemExit(f"not a file or an arXiv id: {ref}")
    return load(path)
