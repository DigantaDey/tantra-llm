from __future__ import annotations

import json

import pytest

from tantra.scraper import (
    Candidate,
    RemoteFile,
    build_query,
    download_items,
    evaluate,
    md5_of,
    parse_candidate,
    search_candidates,
    select_files,
)
from tests.fakes import FakeArchive

GOOD = {
    "identifier": "sadhanatattwa00unknown",
    "title": "সাধনতত্ত্ব",
    "year": "1898",
    "language": "Bengali",
    "licenseurl": "https://creativecommons.org/publicdomain/mark/1.0/",
    "subject": ["tantra", "sadhana"],
    "mediatype": "texts",
}


def _cand(**overrides) -> Candidate:
    return parse_candidate({**GOOD, **overrides})


# ----------------------------------------------------------------- query
def test_query_contains_language_year_and_terms():
    q = build_query((1870, 1935))
    assert "mediatype:texts" in q
    assert "year:[1870 TO 1935]" in q
    assert '"Bengali"' in q and '"Bangla"' in q
    assert '"tantra"' in q and "তন্ত্র" in q and "সাধন" in q


def test_query_rejects_inverted_range():
    with pytest.raises(ValueError):
        build_query((1935, 1870))


# ----------------------------------------------------------------- parsing
def test_parse_candidate_normalises_fields():
    c = parse_candidate({**GOOD, "language": ["Bengali; Bangla", "ben"], "year": ["1899-01-01"]})
    assert c.languages == ("bengali", "bangla", "ben")
    assert c.year == 1899
    assert c.title == "সাধনতত্ত্ব"


def test_parse_candidate_without_year():
    assert parse_candidate({"identifier": "x", "date": "unknown"}).year is None


# ----------------------------------------------------------------- evaluation
def test_good_item_is_accepted_with_score():
    dec = evaluate(_cand())
    assert dec.accepted
    assert dec.score >= 2  # "tantra" + "sadhana" in subjects


def test_rejects_non_bengali():
    dec = evaluate(_cand(language="Sanskrit"))
    assert not dec.accepted
    assert any("language" in r for r in dec.reasons)


def test_rejects_year_outside_range():
    dec = evaluate(_cand(year="1950"), year_range=(1870, 1935))
    assert not dec.accepted and any("outside" in r for r in dec.reasons)


def test_rejects_unmarked_licence_by_default_and_allows_override():
    cand = _cand(licenseurl="", rights="All rights reserved")
    assert not evaluate(cand).accepted
    assert evaluate(cand, require_public_domain=False).accepted


def test_rejects_when_no_keyword():
    dec = evaluate(_cand(title="Rabindra Gitabitan", subject=["poetry"]))
    assert not dec.accepted and any("keyword" in r for r in dec.reasons)


def test_undated_needs_opt_in():
    cand = _cand(year=None)
    assert not evaluate(cand).accepted
    assert evaluate(cand, allow_undated=True).accepted


def test_bengali_keyword_counts():
    dec = evaluate(_cand(title="তান্ত্রিক সাধনা", subject=[]))
    assert dec.accepted and dec.score >= 2


# ----------------------------------------------------------------- file selection
def test_select_files_prefers_exact_identifier_names():
    files = [
        RemoteFile("other.pdf", "Image Container PDF", 10, None),
        RemoteFile("sadhana.pdf", "Image Container PDF", 10, None),
        RemoteFile("sadhana_djvu.txt", "DjVuTXT", 10, None),
        RemoteFile("sadhana.djvu", "DjVu", 10, None),
        RemoteFile("sadhana_thumb.jpg", "Item Tile", 1, None),
        RemoteFile("sub/dir.djvu", "DjVu", 1, None),
    ]
    chosen, missing = select_files("sadhana", files, ("djvu_txt", "pdf", "djvu"))
    assert chosen == {"djvu_txt": "sadhana_djvu.txt", "pdf": "sadhana.pdf", "djvu": "sadhana.djvu"}
    assert missing == []


def test_select_files_reports_missing_and_rejects_unknown_format():
    chosen, missing = select_files("x", [RemoteFile("x.pdf", "PDF", 1, None)], ("djvu_txt", "pdf"))
    assert chosen == {"pdf": "x.pdf"} and missing == ["djvu_txt"]
    with pytest.raises(ValueError):
        select_files("x", [], ("epub",))


# ----------------------------------------------------------------- search orchestration
def test_search_sorts_accepted_first_and_respects_limit():
    records = [
        {**GOOD, "identifier": "a", "language": "English"},
        {**GOOD, "identifier": "b"},
        {**GOOD, "identifier": "c"},
    ]
    archive = FakeArchive(records)
    rows = search_candidates(archive, limit=2)
    assert len(rows) == 2
    assert rows[0][1].accepted
    assert archive.search_queries and "mediatype:texts" in archive.search_queries[0]


def test_search_skips_records_without_identifier():
    rows = search_candidates(FakeArchive([{"title": "no id"}]), limit=5)
    assert rows == []


def test_search_rejects_bad_limit():
    with pytest.raises(ValueError):
        search_candidates(FakeArchive([]), limit=0)


# ----------------------------------------------------------------- download
def _archive_with_item(ident="sadhana1898"):
    files = {
        ident: [
            (f"{ident}_djvu.txt", "DjVuTXT", "সাধনা তন্ত্র\fদ্বিতীয় পৃষ্ঠা".encode()),
            (f"{ident}.pdf", "Image Container PDF", b"%PDF-1.4 fake"),
            (f"{ident}.djvu", "DjVu", b"AT&TFORM fake"),
        ]
    }
    return FakeArchive([{**GOOD, "identifier": ident}], files)


def test_download_writes_files_metadata_and_manifest(tmp_path, monkeypatch):
    archive = _archive_with_item()
    rows = search_candidates(archive, limit=5)
    results = download_items(archive, rows, out_dir=tmp_path, delay=0, sleep=lambda s: None, log=lambda m: None)
    assert [r.status for r in results] == ["downloaded"]
    item = tmp_path / "sadhana1898"
    assert (item / "sadhana1898_djvu.txt").read_bytes().startswith("সাধনা".encode())
    meta = json.loads((item / "metadata.json").read_text(encoding="utf-8"))
    assert meta["source_url"] == "https://archive.org/details/sadhana1898"
    assert meta["files"] == {
        "djvu_txt": "sadhana1898_djvu.txt",
        "pdf": "sadhana1898.pdf",
        "djvu": "sadhana1898.djvu",
    }
    manifest = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(manifest[0])["status"] == "downloaded"


def test_download_is_resumable_and_skips_verified_files(tmp_path):
    ident = "sadhana1898"
    archive = _archive_with_item(ident)
    rows = search_candidates(archive, limit=5)
    download_items(archive, rows, out_dir=tmp_path, delay=0, sleep=lambda s: None, log=lambda m: None)
    first_calls = len(archive.download_calls)
    download_items(archive, rows, out_dir=tmp_path, delay=0, sleep=lambda s: None, log=lambda m: None)
    assert len(archive.download_calls) == first_calls  # nothing re-fetched


def test_checksum_mismatch_is_retried_and_fixed(tmp_path):
    archive = _archive_with_item("sadhana1898")
    archive.corrupt_once.add("sadhana1898.pdf")
    rows = search_candidates(archive, limit=5)
    results = download_items(archive, rows, out_dir=tmp_path, delay=0, sleep=lambda s: None, log=lambda m: None)
    assert results[0].status == "downloaded"
    pdf = tmp_path / "sadhana1898" / "sadhana1898.pdf"
    assert pdf.read_bytes() == b"%PDF-1.4 fake"  # the corrupted first copy was discarded
    assert md5_of(pdf) == __import__("hashlib").md5(b"%PDF-1.4 fake").hexdigest()


def test_network_errors_are_retried_then_recorded_as_failed(tmp_path):
    archive = _archive_with_item("sadhana1898")
    archive.fail_times["sadhana1898.pdf"] = 10  # more failures than retries
    rows = search_candidates(archive, limit=5)
    results = download_items(
        archive, rows, out_dir=tmp_path, delay=0, retries=2, sleep=lambda s: None, log=lambda m: None
    )
    assert results[0].status == "failed"
    assert "sadhana1898.pdf" in (results[0].error or "")
    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["status"] == "failed"


def test_dry_run_downloads_nothing(tmp_path):
    archive = _archive_with_item()
    rows = search_candidates(archive, limit=5)
    results = download_items(archive, rows, out_dir=tmp_path, dry_run=True, log=lambda m: None)
    assert results[0].status == "dry-run"
    assert archive.download_calls == []
    assert not (tmp_path / "manifest.jsonl").exists()


def test_rejected_items_are_never_downloaded(tmp_path):
    archive = _archive_with_item()
    archive.records[0]["language"] = "English"
    rows = search_candidates(archive, limit=5)
    results = download_items(archive, rows, out_dir=tmp_path, delay=0, log=lambda m: None)
    assert results == [] and archive.download_calls == []
