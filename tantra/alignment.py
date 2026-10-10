"""Character-level edit tags.

The corrector predicts, for every *input* character, what it should become.
That per-character "label" is a short string (usually the character itself,
sometimes a conjunct such as ``ক্``, sometimes empty for a spurious space).

Why tags instead of plain same-length classification?  Real OCR corruption
changes length (inserted spaces, dropped hasanta), so input and target are
not aligned one-to-one. Tagging, in the style of GECToR, keeps the model
fully bidirectional while still letting us train with per-position
cross-entropy.

Invariant (property-tested): ``apply_labels(src, to_labels(src, tgt)) == tgt``.
"""

from __future__ import annotations

from difflib import SequenceMatcher

KEEP = "<KEEP>"
DELETE = "<DEL>"
SPECIAL_LABELS = (KEEP, DELETE)
MAX_LABEL_LEN = 6


def align_replacements(src: str, tgt: str) -> list[str] | None:
    """Return, for each char of ``src``, the string of ``tgt`` it becomes.

    Concatenating the returned list gives exactly ``tgt``. Insertions are
    attached to the *following* source character (or the last one at the end).
    Returns ``None`` only when ``src`` is empty but ``tgt`` is not.
    """
    n = len(src)
    if n == 0:
        return [] if not tgt else None

    out: list[str] = [""] * n
    prefix: list[str] = [""] * (n + 1)  # insertions placed before source index i
    opcodes = SequenceMatcher(None, src, tgt, autojunk=False).get_opcodes()
    for op, i1, i2, j1, j2 in opcodes:
        if op == "equal":
            for k in range(i2 - i1):
                out[i1 + k] = tgt[j1 + k]
        elif op == "replace":
            out[i1] = tgt[j1:j2]
            for k in range(i1 + 1, i2):
                out[k] = ""
        elif op == "delete":
            for k in range(i1, i2):
                out[k] = ""
        elif op == "insert":
            prefix[i1] += tgt[j1:j2]

    for i in range(n):
        out[i] = prefix[i] + out[i]
    out[n - 1] += prefix[n]
    return out


def to_labels(src: str, tgt: str) -> list[str] | None:
    """Per-character labels: ``KEEP``, ``DELETE`` or a literal replacement string."""
    repl = align_replacements(src, tgt)
    if repl is None:
        return None
    labels: list[str] = []
    for ch, r in zip(src, repl, strict=True):
        if r == ch:
            labels.append(KEEP)
        elif r == "":
            labels.append(DELETE)
        else:
            labels.append(r)
    return labels


def apply_labels(src: str, labels: list[str]) -> str:
    """Apply labels produced by :func:`to_labels` (or a model prediction) to ``src``."""
    if len(labels) != len(src):
        raise ValueError(f"need one label per character: {len(labels)} labels for {len(src)} chars")
    parts: list[str] = []
    for ch, label in zip(src, labels, strict=True):
        if label == KEEP:
            parts.append(ch)
        elif label == DELETE:
            continue
        else:
            parts.append(label)
    return "".join(parts)


def is_trainable_label(label: str) -> bool:
    """Labels longer than MAX_LABEL_LEN are too rare to learn; they are masked out."""
    return len(label) <= MAX_LABEL_LEN or label in SPECIAL_LABELS
