"""Bengali (Bangla) text utilities.

Everything here is deliberately conservative: we normalise only what is
provably safe for Bengali, and we never strip characters that carry meaning
(hasanta, ZWJ/ZWNJ, vowel signs, nukta).
"""

from __future__ import annotations

import re
import unicodedata

BENGALI_START = 0x0980
BENGALI_END = 0x09FF
HASANTA = "\u09cd"  # virama that forms conjuncts (যুক্তাক্ষর)
ZWJ = "\u200d"
ZWNJ = "\u200c"
DANDA = "\u0964"
DOUBLE_DANDA = "\u0965"
SENTENCE_END = (DANDA, DOUBLE_DANDA, "?", "!", ".", ":")

# NFC does NOT recompose these nukta sequences (they are composition
# exclusions in Unicode), so we must map them explicitly. Verified in tests.
NUKTA_FIXES = {
    "\u09a1\u09bc": "\u09dc",  # ড় (ṛa)
    "\u09a2\u09bc": "\u09dd",  # ঢ় (ṛha)
    "\u09af\u09bc": "\u09df",  # য় (yya)
}

# Characters that are invisible garbage in OCR output. ZWJ/ZWNJ are kept.
_INVISIBLE_JUNK = {"\ufeff", "\u200b", "\u2060", "\u00ad", "\u0000"}
_HSPACE_RE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_WORD_RE = re.compile(r"[\u0980-\u09ff\w]+", re.UNICODE)


def is_bengali(ch: str) -> bool:
    """True if ``ch`` lies in the Bengali Unicode block (U+0980–U+09FF)."""
    return len(ch) == 1 and BENGALI_START <= ord(ch) <= BENGALI_END


def is_word_char(ch: str) -> bool:
    """True for characters that belong inside a word (Bengali letters, marks, digits, Latin)."""
    if ch in (DANDA, DOUBLE_DANDA):
        return False
    return is_bengali(ch) or ch.isalnum() or ch in (ZWJ, ZWNJ)


def normalize(text: str) -> str:
    """Normalise OCR/web Bengali text without altering its meaning.

    * Unicode NFC (canonical composition).
    * Explicit nukta fixes that NFC misses (ড় ঢ় য়).
    * Invisible junk removed (BOM, zero-width space, soft hyphen, NUL).
    * Horizontal whitespace collapsed; trailing spaces stripped per line.
    * Line endings unified to ``\\n``; runs of 3+ blank lines collapsed to 2.
    """
    text = unicodedata.normalize("NFC", text)
    for decomposed, composed in NUKTA_FIXES.items():
        text = text.replace(decomposed, composed)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch not in _INVISIBLE_JUNK)
    lines = [_HSPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return _MULTI_BLANK_RE.sub("\n\n", text).strip("\n") + "\n" if text.strip() else ""


def bengali_ratio(text: str) -> float:
    """Fraction of *letter-like* characters that are Bengali.

    Used to reject pages/lines that are Latin- or symbol-dominated OCR garbage.
    Returns 0.0 for text with no letters at all.
    """
    letters = [ch for ch in text if ch.isalpha() or is_bengali(ch)]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if is_bengali(ch)) / len(letters)


def words(text: str) -> list[str]:
    """Bengali/Latin/digit word tokens; danda and punctuation act as separators."""
    return _WORD_RE.findall(text)


def word_spans(text: str) -> list[tuple[int, int]]:
    """Character spans ``(start, end)`` of every word token in ``text``.

    Uses the same notion of a "word" as :func:`is_word_char`, so spans never
    include danda or surrounding punctuation.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, ch in enumerate(text):
        if is_word_char(ch):
            if start is None:
                start = i
        elif start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    return spans


def has_conjunct(word: str) -> bool:
    """True if the word contains a conjunct-forming hasanta."""
    return HASANTA in word
