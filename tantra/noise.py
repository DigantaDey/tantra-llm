"""Synthetic archive.org-style OCR degradation.

We cannot scan 19th-century Bengali prints on demand, so we *simulate* the
error modes that matter for Bengali OCR:

* conjunct breaks   — ক্ষ read as "ক ষ" (hasanta lost, space inserted)
* confusions        — visually similar glyphs (র ↔ ব ↔ ক and friends)
* space splits      — a word cut in two by a spurious space
* fading            — thin vowel signs / anusvara vanish from faded print
* drops             — random glyph dropouts
* bleed             — ghost duplicate of a neighbouring glyph (bleed-through)

The injector is deterministic given a seed, so experiments are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from tantra.bengali import HASANTA, is_bengali

# Symmetric visual-confusion pairs, hand-picked as a starting point: the r/b/k triangle
# and common Bengali OCR confusions (retroflex/dental, long/short vowel signs). They are
# assumptions to be tuned on real scans, not measured confusion rates.
CONFUSION_PAIRS: tuple[tuple[str, str], ...] = (
    ("র", "ব"),
    ("ব", "ক"),
    ("র", "ক"),
    ("ন", "ণ"),
    ("ট", "ত"),
    ("ধ", "ঘ"),
    ("ি", "ী"),
    ("ু", "ূ"),
    ("ঁ", "ং"),
)


def _build_confusable(pairs: tuple[tuple[str, str], ...]) -> dict[str, tuple[str, ...]]:
    table: dict[str, list[str]] = {}
    for a, b in pairs:
        table.setdefault(a, []).append(b)
        table.setdefault(b, []).append(a)
    return {k: tuple(v) for k, v in table.items()}


CONFUSABLE: dict[str, tuple[str, ...]] = _build_confusable(CONFUSION_PAIRS)

# Dependent vowel signs (U+09BE..U+09CC) plus anusvara, candrabindu and visarga:
# the first marks to vanish from faded print. Hasanta is deliberately excluded.
FADEABLE: frozenset[str] = frozenset([chr(c) for c in range(0x09BE, 0x09CD)] + ["\u0981", "\u0982", "\u0983"])


@dataclass(frozen=True)
class NoiseProfile:
    """Per-event probabilities. Defaults approximate a moderately degraded print."""

    conjunct_break: float = 0.20  # per hasanta
    confusion: float = 0.02  # per confusable glyph
    space_split: float = 0.04  # per word of length >= 4
    fade: float = 0.02  # per fadeable mark
    drop: float = 0.004  # per Bengali glyph
    bleed: float = 0.003  # per Bengali glyph

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability in [0, 1], got {value}")

    @classmethod
    def scaled(cls, factor: float) -> NoiseProfile:
        """A profile with every rate multiplied by ``factor`` (clipped to 1.0)."""
        base = cls()
        return cls(**{k: min(1.0, v * factor) for k, v in base.__dict__.items()})


def corrupt(text: str, rng: random.Random, profile: NoiseProfile | None = None) -> str:
    """Return a degraded copy of ``text``. Newlines are never touched."""
    profile = profile or NoiseProfile()
    out: list[str] = []
    for ch in text:
        if ch == HASANTA and rng.random() < profile.conjunct_break:
            # Either the hasanta drops out (ক্ষ -> কষ) or it becomes a space (ক ষ).
            out.append(" " if rng.random() < 0.7 else "")
            continue
        if ch in CONFUSABLE and rng.random() < profile.confusion:
            out.append(rng.choice(CONFUSABLE[ch]))
            continue
        if ch in FADEABLE and rng.random() < profile.fade:
            continue
        if is_bengali(ch) and ch != HASANTA and rng.random() < profile.drop:
            continue
        out.append(ch)
        if is_bengali(ch) and rng.random() < profile.bleed:
            out.append(ch)  # bleed-through ghost

    damaged = "".join(out)
    return "\n".join(_split_words(line, rng, profile.space_split) for line in damaged.split("\n"))


def _split_words(line: str, rng: random.Random, p: float) -> str:
    """Insert a spurious space inside some words (length >= 4) of one line."""
    if p <= 0:
        return line
    pieces: list[str] = []
    for token in line.split(" "):
        if len(token) >= 4 and rng.random() < p:
            cut = rng.randrange(2, len(token) - 1)
            # Do not cut in front of a hasanta: that would split a conjunct twice.
            while cut > 1 and token[cut] == HASANTA:
                cut -= 1
            pieces.append(token[:cut] + " " + token[cut:])
        else:
            pieces.append(token)
    return " ".join(pieces)
