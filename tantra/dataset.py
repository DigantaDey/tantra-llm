"""Synthetic training data: clean Bengali windows -> (corrupted, clean) pairs -> edit labels."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from tantra.alignment import is_trainable_label, to_labels
from tantra.bengali import bengali_ratio, normalize
from tantra.noise import NoiseProfile, corrupt


def make_windows(line: str, max_len: int) -> list[str]:
    """Split ``line`` into segments of at most ``max_len`` chars.

    Cuts happen after a space whenever possible, and each segment keeps its
    trailing space, so ``"".join(make_windows(line, n)) == line`` always holds.
    Only a single word longer than ``max_len`` is hard-cut.
    """
    if max_len < 2:
        raise ValueError("max_len must be at least 2")
    windows: list[str] = []
    start = 0
    n = len(line)
    while start < n:
        end = min(start + max_len, n)
        if end < n:
            cut = line.rfind(" ", start, end)
            if cut > start:
                end = cut + 1
        windows.append(line[start:end])
        start = end
    return windows


def clean_windows(lines: Iterable[str], max_len: int, min_len: int = 24) -> Iterator[str]:
    """Clean Bengali windows from raw corpus lines (normalised, Bengali-dominant only)."""
    for raw in lines:
        line = normalize(raw).replace("\n", " ").strip()
        if len(line) < min_len or bengali_ratio(line) < 0.6:
            continue
        for window in make_windows(line + " ", max_len):
            if len(window.strip()) >= min_len // 2:
                yield window


@dataclass(frozen=True)
class Example:
    """One supervised example: corrupted input and per-character edit labels."""

    source: str
    clean: str
    labels: tuple[str, ...]


def make_example(
    clean: str,
    rng: random.Random,
    profile: NoiseProfile,
    max_len: int,
) -> Example | None:
    """Corrupt ``clean`` and derive labels. Returns None when the pair is unusable."""
    source = corrupt(clean, rng, profile)
    if source == clean or not source.strip() or len(source) > max_len:
        return None
    labels = to_labels(source, clean)
    if labels is None:
        return None
    return Example(source=source, clean=clean, labels=tuple(labels))


def build_examples(
    windows: Iterable[str],
    *,
    copies: int,
    profile: NoiseProfile,
    max_len: int,
    seed: int,
    boost_terms: frozenset[str] = frozenset(),
    boost_factor: int = 2,
) -> list[Example]:
    """Create ``copies`` corrupted versions of each clean window.

    ``boost_terms`` implements the dictionary anchor at training time: windows
    that contain an esoteric term are sampled ``boost_factor`` times more often,
    so the model sees Tantric vocabulary, under corruption, far more than a
    plain frequency sample would show it.
    """
    rng = random.Random(seed)
    examples: list[Example] = []
    for window in windows:
        n = copies
        if boost_terms and any(term in window for term in boost_terms):
            n = copies * boost_factor
        for _ in range(n):
            example = make_example(window, rng, profile, max_len)
            if example is not None:
                examples.append(example)
    return examples


def split_is_validation(text: str, fraction: float) -> bool:
    """Deterministic train/validation split keyed on the content hash (stable across runs)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return digest[0] / 255.0 < fraction


def label_counts(examples: Iterable[Example]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for ex in examples:
        counts.update(label for label in ex.labels if is_trainable_label(label))
    return counts


def char_counts(texts: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(text)
    return counts
