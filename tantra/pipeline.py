"""Document-level correction: raw archive OCR text -> corrected pages -> Markdown + plain text."""

from __future__ import annotations

from dataclasses import dataclass, field

from tantra.bengali import bengali_ratio, normalize
from tantra.corrector import CorrectionStats, Corrector
from tantra.markdown import DocumentMeta, reflow, render_document, split_pages, strip_page_numbers

# Lines that are mostly Latin/digits (catalogue numbers, English notes) are left untouched:
# the denoiser was trained on Bengali only and would damage them.
MIN_BENGALI_RATIO_TO_CORRECT = 0.5


@dataclass
class DocumentResult:
    markdown: str
    plain_text: str
    pages: int
    stats: CorrectionStats
    lines_corrected: int = 0
    lines_passed_through: int = 0
    extra: dict[str, int | str] = field(default_factory=dict)


def correct_document(
    raw_text: str,
    corrector: Corrector | None,
    meta: DocumentMeta,
    *,
    use_dictionary: bool = True,
) -> DocumentResult:
    """Normalise an archive OCR document and (optionally) denoise it.

    ``corrector=None`` runs the structural pipeline only: Unicode normalisation,
    page splitting, page-number stripping and reflow into Markdown and plain
    text. That is the zero-setup path - it needs no checkpoint and no clean
    baseline corpus, so a fresh checkout can go from ``scraper.py download`` to
    calibration data in one straight line.
    """
    raw_pages = split_pages(raw_text)
    page_lines: list[list[str]] = []
    for raw in raw_pages:
        normalised = normalize(raw)
        lines = strip_page_numbers(normalised.split("\n")) if normalised else []
        page_lines.append(lines)

    to_correct: list[str] = []
    slots: list[tuple[int, int]] = []  # (page index, line index) of every corrected line
    for pi, lines in enumerate(page_lines):
        for li, line in enumerate(lines):
            if corrector is not None and bengali_ratio(line) >= MIN_BENGALI_RATIO_TO_CORRECT:
                to_correct.append(line)
                slots.append((pi, li))

    corrected, stats = (
        corrector.correct_lines(to_correct, use_dictionary=use_dictionary) if to_correct else ([], CorrectionStats())
    )
    for (pi, li), text in zip(slots, corrected, strict=True):
        page_lines[pi][li] = text

    pages_md: list[list[str]] = []
    plain_pages: list[str] = []
    for lines in page_lines:
        pages_md.append(reflow(lines))
        plain_pages.append("\n\n".join(reflow(lines)))
    plain = "\n\n".join(p for p in plain_pages if p.strip()) + "\n"

    corrections: dict[str, int | str] = {
        "pages": len(raw_pages),
        "lines_corrected_by_model": len(to_correct),
        "windows_changed": stats.changed_windows,
        "dictionary_anchors": stats.dictionary_anchors,
        "dictionary_used": "no model (structure only)" if corrector is None else ("yes" if use_dictionary else "no"),
    }
    markdown = render_document(meta, pages_md, corrections)
    return DocumentResult(
        markdown=markdown,
        plain_text=plain,
        pages=len(raw_pages),
        stats=stats,
        lines_corrected=len(to_correct),
        lines_passed_through=sum(len(lines) for lines in page_lines) - len(to_correct),
        extra=corrections,
    )
