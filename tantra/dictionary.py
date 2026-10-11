"""Esoteric dictionary anchor.

Ordinary spell-checkers "correct" archaic Tantric terms into common modern
words. The anchor fixes that by:

1. indexing a curated Tantric lexicon with a SymSpell-style *deletion index*,
   so fuzzy lookup of a corrupted word costs microseconds, not a full scan;
2. exposing the lexicon to the corrector, which adds a bonus to dictionary
   candidates when it chooses between the model's guess and alternatives
   (see :mod:`tantra.corrector`).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from tantra.bengali import words
from tantra.textmetrics import levenshtein

DEFAULT_DICTIONARY_PATH = Path(__file__).resolve().parent / "data" / "esoteric_terms.tsv"


@dataclass(frozen=True)
class Term:
    text: str
    translit: str
    gloss: str
    category: str


def _deletions(word: str, max_dist: int) -> set[str]:
    """All strings reachable by deleting up to ``max_dist`` characters (including ``word``)."""
    level = {word}
    result = {word}
    for _ in range(max_dist):
        nxt: set[str] = set()
        for w in level:
            for i in range(len(w)):
                nxt.add(w[:i] + w[i + 1 :])
        nxt -= result
        result |= nxt
        level = nxt
    return result


class EsotericDictionary:
    """Immutable lexicon with fuzzy lookup."""

    def __init__(self, terms: Iterable[Term], max_dist: int = 2) -> None:
        self.max_dist = max_dist
        self._terms: dict[str, Term] = {}
        for term in terms:
            self._terms.setdefault(term.text, term)
        self._index: dict[str, set[str]] = {}
        for text in self._terms:
            for d in _deletions(text, max_dist):
                self._index.setdefault(d, set()).add(text)

    # ----- construction ------------------------------------------------------
    @classmethod
    def from_tsv(cls, path: str | Path, max_dist: int = 2) -> EsotericDictionary:
        terms: list[Term] = []
        with Path(path).open(encoding="utf-8") as fh:
            reader = csv.reader((line for line in fh if line.strip() and not line.startswith("#")), delimiter="\t")
            for row in reader:
                if len(row) != 4:
                    raise ValueError(f"malformed dictionary row (need 4 tab-separated fields): {row!r}")
                bengali, translit, gloss, category = (c.strip() for c in row)
                terms.append(Term(bengali, translit, gloss, category))
        return cls(terms, max_dist=max_dist)

    @classmethod
    def default(cls, max_dist: int = 2) -> EsotericDictionary:
        """The seed lexicon shipped inside the package."""
        return cls.from_tsv(DEFAULT_DICTIONARY_PATH, max_dist=max_dist)

    # ----- queries -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._terms)

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and word in self._terms

    def __iter__(self) -> Iterator[Term]:
        return iter(self._terms.values())

    def get(self, word: str) -> Term | None:
        return self._terms.get(word)

    def lookup(self, word: str) -> list[tuple[str, int]]:
        """Dictionary terms within ``max_dist`` edits of ``word``, nearest first.

        Exact matches come back with distance 0.
        """
        if not word:
            return []
        hits: set[str] = set()
        for d in _deletions(word, self.max_dist):
            hits |= self._index.get(d, set())
        scored: list[tuple[str, int]] = []
        for term in hits:
            dist = levenshtein(word, term, max_dist=self.max_dist)
            if dist <= self.max_dist:
                scored.append((term, dist))
        scored.sort(key=lambda pair: (pair[1], pair[0]))
        return scored

    def terms_in(self, text: str) -> set[str]:
        """Dictionary terms occurring verbatim as words in ``text``."""
        return {w for w in words(text) if w in self._terms}
