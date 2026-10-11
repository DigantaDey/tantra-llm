"""Render corrected pages as clean, self-describing Markdown."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from tantra import __version__
from tantra.bengali import SENTENCE_END

_BARE_NUMBER = re.compile(r"^\s*[0-9০-৯]{1,4}\s*$")
_MD_LEADERS = ("#", ">", "- ", "* ", "+ ", "|", "```")
_ORDERED = re.compile(r"^\d+[.)]\s")


@dataclass(frozen=True)
class DocumentMeta:
    identifier: str
    title: str
    year: int | None = None
    languages: tuple[str, ...] = ()
    licenseurl: str = ""
    source_url: str = ""
    extra: dict[str, str] = field(default_factory=dict)


def split_pages(text: str) -> list[str]:
    """Split djvu-style text on form feeds. Text without form feeds is one page."""
    pages = text.split("\f")
    while len(pages) > 1 and not pages[-1].strip():
        pages.pop()
    return pages


def strip_page_numbers(lines: list[str]) -> list[str]:
    """Drop bare page numbers at the top and bottom of a page (never inside the text)."""
    start, end = 0, len(lines)
    while start < end and (not lines[start].strip() or _BARE_NUMBER.match(lines[start])):
        start += 1
    while end > start and (not lines[end - 1].strip() or _BARE_NUMBER.match(lines[end - 1])):
        end -= 1
    return lines[start:end]


def _escape(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith(_MD_LEADERS) or _ORDERED.match(stripped):
        return line.replace(stripped, "\\" + stripped, 1)
    return line


def reflow(lines: list[str]) -> list[str]:
    """Join OCR line breaks into paragraphs; keep hard breaks after sentence ends (verse, lists).

    Returns Markdown paragraphs separated by blank lines.
    """
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line.strip():
            if current:
                paragraphs.append(_join(current))
                current = []
            continue
        current.append(_escape(line.strip()))
    if current:
        paragraphs.append(_join(current))
    return paragraphs


def _join(lines: list[str]) -> str:
    out = lines[0]
    for prev, nxt in zip(lines, lines[1:], strict=False):
        if prev.rstrip().endswith(SENTENCE_END):
            out += "  \n" + nxt  # trailing double space = Markdown hard line break
        else:
            out += " " + nxt
    return out


def front_matter(meta: DocumentMeta, corrections: dict[str, int | str]) -> str:
    """YAML front matter. JSON string literals are valid YAML, so no PyYAML dependency is needed."""
    fields: list[tuple[str, object]] = [
        ("identifier", meta.identifier),
        ("title", meta.title),
        ("year", meta.year),
        ("languages", list(meta.languages)),
        ("licence", meta.licenseurl or None),
        ("source", meta.source_url or None),
        ("generated_by", f"tantra-llm {__version__}"),
    ]
    fields += [(f"correction_{k}", v) for k, v in corrections.items()]
    lines = ["---"]
    for key, value in fields:
        if value is None:
            continue
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def render_document(meta: DocumentMeta, pages: list[list[str]], corrections: dict[str, int | str]) -> str:
    """``pages`` is a list of Markdown paragraphs per page."""
    parts = [front_matter(meta, corrections), f"\n# {meta.title}\n"]
    for number, paragraphs in enumerate(pages, start=1):
        if not paragraphs:
            continue
        parts.append(f"\n## Page {number}\n")
        parts.append("\n\n".join(paragraphs) + "\n")
    return "".join(parts)
