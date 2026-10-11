"""Live integration tests against the real archive.org API.

Run explicitly (they need network access and are excluded from the default
suite)::

    python -m pytest -m live tests/test_live_archive.py -v

CI runs these on every push and pull request, so the scraper's contract with
archive.org - the query shape, the JSON field types (numeric ``year``), file
listings with MD5s, and resumable verified downloads - is continuously checked
against the live service, not only against fakes.
"""

from __future__ import annotations

import time

import pytest

from tantra.bengali import bengali_ratio
from tantra.scraper import (
    InternetArchiveClient,
    build_query,
    download_items,
    search_candidates,
)

pytestmark = pytest.mark.live

# A small, stable Digital Library of India scan in the target domain:
# Bengali, 1908, Diksha O Sadhana, with a verified _djvu.txt OCR file.
STABLE_ITEM = "dli.bengal.10689.8270"
ATTEMPTS = 3
RETRY_SLEEP = 20.0


def _with_retries(operation, description: str):
    last: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return operation()
        except AssertionError as exc:  # assertion failures are not retried: they are contract failures
            raise exc
        except Exception as exc:  # transient archive.org hiccups: 5xx, timeouts, resets
            last = exc
            if attempt < ATTEMPTS:
                time.sleep(RETRY_SLEEP)
    pytest.fail(f"{description} failed after {ATTEMPTS} attempts: {last}")


def test_live_search_finds_bengali_tantra_texts():
    """The default query must return accepted items on the live archive."""
    client = InternetArchiveClient(timeout=45.0)

    def _search() -> list:
        rows = search_candidates(client, limit=15, query=build_query())
        accepted = [(c, d) for c, d in rows if d.accepted]
        if not accepted:
            # retryable: a mid-index-update blip can serve an empty page
            raise RuntimeError(
                "no accepted items from the live archive - the domain corpus should contain "
                f"dozens (78 in October 2026); rejection reasons seen: {[d.reasons for _, d in rows][:5]}"
            )
        return accepted

    accepted = _with_retries(_search, "searching archive.org")
    first, decision = accepted[0]
    assert first.identifier
    assert first.year is not None and 1870 <= first.year <= 1935
    assert any(lang in {"bengali", "bangla", "ben", "beng"} for lang in first.languages)
    assert decision.score >= 1


def test_live_download_verifies_md5_and_bengali_ocr(tmp_path):
    """Download the stable item's OCR text end-to-end: file, MD5, metadata, Bengali content."""

    def run():
        client = InternetArchiveClient(timeout=90.0)
        rows = search_candidates(client, limit=1, query=f"identifier:{STABLE_ITEM}")
        if not rows:
            raise RuntimeError(f"the probe item {STABLE_ITEM} was not found via search (transient?)")
        candidate, decision = rows[0]
        assert decision.accepted, f"the probe item should pass the default policy: {decision.reasons}"
        results = download_items(
            client,
            [(candidate, decision)],
            out_dir=tmp_path,
            formats=("djvu_txt",),
            delay=0,
            retries=2,
            sleep=lambda s: None,
            log=lambda m: None,
        )
        assert results and results[0].status == "downloaded", results
        item_dir = tmp_path / STABLE_ITEM
        txt = item_dir / results[0].files["djvu_txt"]
        text = txt.read_text(encoding="utf-8", errors="replace")
        assert len(text) > 10_000, "the OCR text should be book-sized"
        assert bengali_ratio(text[:4000]) > 0.5, "the OCR text should be predominantly Bengali"
        assert (item_dir / "metadata.json").exists()
        return True

    assert _with_retries(run, f"downloading {STABLE_ITEM}")
