"""Edit-distance helpers shared by the dictionary, the corrector and evaluation."""

from __future__ import annotations


def levenshtein(a: str, b: str, max_dist: int | None = None) -> int:
    """Levenshtein distance over code points.

    With ``max_dist`` the computation exits early and returns ``max_dist + 1``
    as soon as the distance is known to exceed it. This is what keeps
    dictionary lookups fast on whole books.
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a) if max_dist is None or len(a) <= max_dist else max_dist + 1
    if max_dist is not None and len(a) - len(b) > max_dist:
        return max_dist + 1

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate: edit distance divided by reference length."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)
