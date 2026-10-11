"""Corpus retrieval: ground chat answers in the cleaned archive texts.

The pipeline already turns archive.org scans into cleaned plain text under
``data/processed/``. This module turns that corpus into something the chat app
can *use* at answer time: passages are split on paragraph boundaries, indexed
with BM25 over Bengali-aware word tokens (the same tokenizer the rest of the
project uses), and the best-matching passages are injected into the system
prompt with stable ``[n]`` citation markers.

Everything is pure Python and offline; no model, no embedding service.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tantra.bengali import words

# Passage sizing: large enough to carry an argument or ritual step, small enough
# that three of them fit into a 2048-token context next to the question.
DEFAULT_MAX_PASSAGE_CHARS = 900

# Query-side function words that carry no topical signal. Ranking-only: a query
# made entirely of stopwords simply returns no hits and the chat falls back to
# the plain system prompt.
STOPWORDS = frozenset(
    {
        # English function words common in mixed queries
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "in",
        "on",
        "to",
        "is",
        "are",
        "was",
        "were",
        "what",
        "which",
        "who",
        "how",
        "why",
        "when",
        "where",
        "does",
        "do",
        "did",
        "tell",
        "about",
        "for",
        "with",
        "that",
        "this",
        "it",
        "by",
        "as",
        "at",
        "be",
        # Bengali function words
        "কি",
        "কী",
        "এবং",
        "ও",
        "একটি",
        "একটা",
        "এই",
        "সেই",
        "করে",
        "করা",
        "করলেন",
        "হয়",
        "হচ্ছে",
        "হবে",
        "আছে",
        "থেকে",
        "জন্য",
        "সঙ্গে",
        "সাথে",
        "কোনো",
        "কোথায়",
        "কেন",
        "কিভাবে",
        "কীভাবে",
        "কখন",
        "কে",
        "যা",
        "যে",
        "না",
        "তবে",
        "আর",
        "মধ্যে",
        "উপরে",
        "নিচে",
        "পরে",
        "আগে",
        "অথবা",
        "বা",
        "হয়তো",
        "কত",
    }
)

_TITLE_RE = re.compile(r"^title:\s*(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Passage:
    """One retrievable chunk of a cleaned document."""

    doc_id: str
    title: str
    ordinal: int  # 1-based position within the document
    text: str


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in words(text) if len(w) > 1 and w.lower() not in STOPWORDS]


def split_passages(text: str, *, max_chars: int = DEFAULT_MAX_PASSAGE_CHARS) -> list[str]:
    """Split a cleaned document into paragraph-based passages of bounded size.

    Short paragraphs are merged with their neighbours; a paragraph longer than
    ``max_chars`` is cut at the nearest sentence end (danda included).
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    passages: list[str] = []
    buffer = ""
    for para in paragraphs:
        for chunk in _oversized(para, max_chars):
            if buffer and len(buffer) + len(chunk) + 2 > max_chars:
                passages.append(buffer)
                buffer = chunk
            else:
                buffer = f"{buffer}\n\n{chunk}" if buffer else chunk
    if buffer:
        passages.append(buffer)
    return passages


def _oversized(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    rest = paragraph
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max(window.rfind("।"), window.rfind(" "), window.rfind("."))
        if cut < max_chars // 2:
            cut = max_chars
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    if rest.strip():
        pieces.append(rest.strip())
    return pieces


def _title_of_markdown(md_path: Path, fallback: str) -> str:
    try:
        match = _TITLE_RE.search(md_path.read_text(encoding="utf-8", errors="replace").split("\n\n", 1)[0])
    except OSError:
        return fallback
    if not match:
        return fallback
    try:
        value = json.loads(match.group(1))
        return str(value) if value else fallback
    except ValueError:
        return match.group(1).strip('"')


class CorpusIndex:
    """A BM25 index over the passages of a document corpus."""

    def __init__(self, passages: Iterable[Passage], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.passages = list(passages)
        self.k1 = k1
        self.b = b
        self._lengths: list[int] = []
        self._term_counts: list[Counter] = []
        self._postings: dict[str, list[int]] = {}
        for i, passage in enumerate(self.passages):
            counts = Counter(_tokens(passage.text))
            self._term_counts.append(counts)
            self._lengths.append(sum(counts.values()))
            for token in counts:
                self._postings.setdefault(token, []).append(i)
        total = sum(self._lengths)
        self._avg_length = total / len(self.passages) if self.passages else 0.0

    @classmethod
    def from_texts(cls, texts: Iterable[tuple[str, str, str]]) -> CorpusIndex:
        """Build from ``(doc_id, title, text)`` triples."""
        passages: list[Passage] = []
        for doc_id, title, text in texts:
            for ordinal, chunk in enumerate(split_passages(text), start=1):
                passages.append(Passage(doc_id=doc_id, title=title, ordinal=ordinal, text=chunk))
        return cls(passages)

    @classmethod
    def from_dir(cls, directory: Path) -> CorpusIndex:
        """Build from a ``data/processed`` directory (``<id>.txt`` + optional ``<id>.md``)."""
        if not directory.is_dir():
            return cls([])
        texts: list[tuple[str, str, str]] = []
        for txt in sorted(directory.glob("*.txt")):
            doc_id = txt.stem
            title = _title_of_markdown(txt.with_suffix(".md"), doc_id)
            texts.append((doc_id, title, txt.read_text(encoding="utf-8", errors="replace")))
        return cls.from_texts(texts)

    def __len__(self) -> int:
        return len(self.passages)

    @property
    def document_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for p in self.passages:
            seen.setdefault(p.doc_id, None)
        return list(seen)

    def search(self, query: str, *, k: int = 3) -> list[Hit]:
        """Top-``k`` passages for ``query`` (BM25; empty when nothing overlaps)."""
        if not self.passages or k < 1:
            return []
        terms = _tokens(query)
        if not terms:
            return []
        candidates: set[int] = set()
        for term in terms:
            candidates.update(self._postings.get(term, ()))
        n_docs = len(self.passages)
        hits: list[Hit] = []
        for i in candidates:
            counts = self._term_counts[i]
            length_norm = self.k1 * (
                1.0 - self.b + self.b * (self._lengths[i] / self._avg_length if self._avg_length else 1.0)
            )
            score = 0.0
            for term in terms:
                postings = self._postings.get(term)
                tf = counts.get(term, 0)
                if tf == 0 or not postings:
                    continue
                df = len(postings)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                score += idf * (tf * (self.k1 + 1.0)) / (tf + length_norm)
            if score > 0:
                hits.append(Hit(passage=self.passages[i], score=score))
        hits.sort(key=lambda h: (-h.score, h.passage.doc_id, h.passage.ordinal))
        return hits[:k]


def build_grounded_prompt(base_prompt: str, hits: list[Hit], *, max_chars: int = 2400) -> str:
    """Append reference passages to the system prompt, with a budget.

    The model is told to prefer the passages, cite them as ``[n]``, and admit it
    when the corpus does not answer the question.
    """
    if not hits:
        return base_prompt
    blocks: list[str] = []
    used = 0
    for number, hit in enumerate(hits, start=1):
        passage = hit.passage
        block = (
            f"[{number}] {passage.title} "
            f"(archive.org id: {passage.doc_id}, passage {passage.ordinal})\n{passage.text.strip()}"
        )
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    instructions = (
        "Reference passages from the project's cleaned archive corpus follow. "
        "Ground your answer in them when they are relevant and cite them inline as [1], [2], … . "
        "If they do not answer the question, say plainly that the corpus does not cover it "
        "and answer from your own knowledge, marking that clearly. "
        "Answer in the language of the question."
    )
    return f"{base_prompt}\n\n{instructions}\n\n" + "\n\n".join(blocks)


def format_sources(hits: list[Hit]) -> str:
    """One line for the answer footer, e.g. ``[1] সাধনতত্ত্ব · [2] শক্তি-সাধনা``."""
    return " · ".join(f"[{n}] {h.passage.title}" for n, h in enumerate(hits, start=1))


__all__ = [
    "CorpusIndex",
    "Hit",
    "Passage",
    "STOPWORDS",
    "build_grounded_prompt",
    "format_sources",
    "split_passages",
]
