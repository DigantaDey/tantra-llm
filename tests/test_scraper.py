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
    assert '"Bengali"' in q and '"Bangla"' in q and '"ben"' in q
    assert '"tantra"' in q and "তন্ত্র" in q and "সাধন" in q


def test_query_drops_the_year_clause_when_undated_items_are_allowed():
    with_year = build_query((1870, 1935))
    without_year = build_query((1870, 1935), allow_undated=True)
    assert "year:" in with_year
    assert "year:" not in without_year
    assert "language:" in without_year  # the other clauses stay


def test_query_rejects_inverted_range():
    with pytest.raises(ValueError):
        build_query((1935, 1870))


# ----------------------------------------------------------------- parsing
def test_parse_candidate_normalises_fields():
    c = parse_candidate({**GOOD, "language": ["Bengali; Bangla", "ben"], "year": ["1899-01-01"]})
    assert c.languages == ("bengali", "bangla", "ben")
    assert c.year == 1899
    assert c.title == "সাধনতত্ত্ব"


def test_parse_candidate_survives_live_solr_shapes():
    """Regression: live archive.org search returns `year` as a JSON number and
    `language` with stray punctuation ("Bengali."); the old parser crashed or
    mis-parsed both."""
    c = parse_candidate(
        {
            "identifier": "in.ernet.dli.2015.324631",
            "title": "Adhyatma-sadhana",
            "year": 1917,  # int, as returned by advancedsearch.php
            "language": "Bengali.",
            "subject": "tantra",
            "licenseurl": None,
        }
    )
    assert c.year == 1917
    assert c.languages == ("bengali",)
    assert c.subjects == ("tantra",)
    assert c.licenseurl == ""


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


def test_unmarked_items_are_accepted_by_default_and_gated_by_marked_policy():
    """Default policy accepts unmarked scans (the licence travels with the data);
    --require-marked-pd (RIGHTS_MARKED) restores the conservative behaviour."""
    cand = _cand(licenseurl="", rights="")
    assert evaluate(cand).accepted  # default: RIGHTS_ANY
    dec = evaluate(cand, rights_policy="marked")
    assert not dec.accepted and any("public domain" in r for r in dec.reasons)
    # a PD-marked item passes both policies
    assert evaluate(_cand(), rights_policy="marked").accepted
    # a non-PD licence is also accepted by default (recorded, not trusted)
    assert evaluate(_cand(licenseurl="https://creativecommons.org/licenses/by-nd/3.0/")).accepted
    with pytest.raises(ValueError):
        evaluate(_cand(), rights_policy="nonsense")


def test_rights_helpers_describe_the_licence_basis():
    from tantra.scraper import describe_rights, is_marked_public_domain

    assert is_marked_public_domain(_cand())
    assert "marked public domain" in describe_rights(_cand())
    assert "verify rights" in describe_rights(_cand(licenseurl="", rights=""))
    assert "on record" in describe_rights(_cand(licenseurl="https://creativecommons.org/licenses/by-nd/3.0/"))


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


def test_search_without_undated_opt_in_keeps_the_year_filter():
    undated = {**GOOD, "identifier": "undated1", "year": None}
    with_opt_in = FakeArchive([undated])
    rows = search_candidates(with_opt_in, limit=5, allow_undated=True)
    assert "year:" not in with_opt_in.search_queries[0]
    assert rows[0][1].accepted
    without_opt_in = FakeArchive([undated])
    rows = search_candidates(without_opt_in, limit=5)
    assert "year:" in without_opt_in.search_queries[0]
    assert not rows[0][1].accepted


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
    results = download_items(
        archive,
        rows,
        out_dir=tmp_path,
        formats=("djvu_txt", "pdf", "djvu"),
        delay=0,
        sleep=lambda s: None,
        log=lambda m: None,
    )
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
    assert meta["rights_basis"] == "marked public domain on archive.org"
    manifest = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(manifest[0])["status"] == "downloaded"


def test_download_default_formats_are_djvu_txt_and_pdf(tmp_path):
    """Live scans carry no .djvu file, so the default keeps to what exists."""
    archive = _archive_with_item()
    rows = search_candidates(archive, limit=5)
    results = download_items(archive, rows, out_dir=tmp_path, delay=0, sleep=lambda s: None, log=lambda m: None)
    assert [r.status for r in results] == ["downloaded"]
    assert results[0].files == {"djvu_txt": "sadhana1898_djvu.txt", "pdf": "sadhana1898.pdf"}
    assert results[0].missing == []
    assert not (tmp_path / "sadhana1898" / "sadhana1898.djvu").exists()


def test_download_reports_no_files_when_the_item_has_nothing_usable(tmp_path):
    archive = FakeArchive([{**GOOD, "identifier": "emptyitem"}], {"emptyitem": [("thumb.jpg", "Item Tile", b"x")]})
    rows = search_candidates(archive, limit=5)
    results = download_items(archive, rows, out_dir=tmp_path, delay=0, sleep=lambda s: None, log=lambda m: None)
    assert [r.status for r in results] == ["no-files"]
    assert results[0].missing == ["djvu_txt", "pdf"]
    # provenance is still recorded so a later run knows the item was visited
    assert (tmp_path / "emptyitem" / "metadata.json").exists()
    manifest = json.loads((tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert manifest["status"] == "no-files"


def test_download_records_the_rights_basis_for_unmarked_items(tmp_path):
    archive = _archive_with_item()
    archive.records[0] = {**archive.records[0], "licenseurl": "", "rights": ""}
    rows = search_candidates(archive, limit=5)
    results = download_items(
        archive, rows, out_dir=tmp_path, formats=("djvu_txt",), delay=0, sleep=lambda s: None, log=lambda m: None
    )
    assert [r.status for r in results] == ["downloaded"]
    meta = json.loads((tmp_path / "sadhana1898" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["rights_basis"] == "no licence marked on archive.org - verify rights before redistributing"


def test_download_emits_progress_hooks(tmp_path):
    archive = _archive_with_item()
    rows = search_candidates(archive, limit=5)
    starts: list[tuple[str, str, int | None]] = []
    progress: list[tuple[int, int | None]] = []
    download_items(
        archive,
        rows,
        out_dir=tmp_path,
        formats=("djvu_txt",),
        delay=0,
        sleep=lambda s: None,
        log=lambda m: None,
        on_file_start=lambda ident, name, size: starts.append((ident, name, size)),
        on_progress=lambda done, total: progress.append((done, total)),
    )
    assert starts == [("sadhana1898", "sadhana1898_djvu.txt", len("সাধনা তন্ত্র\fদ্বিতীয় পৃষ্ঠা".encode()))]
    assert progress and progress[-1][0] == progress[-1][1]


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
