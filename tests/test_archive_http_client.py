"""Offline integration tests for the production archive.org HTTP client.

These run the *real* :class:`InternetArchiveClient` against a local HTTP server
that mirrors the live API's response shapes (verified against archive.org by the
``live-archive`` CI job): Solr search results with numeric ``year`` fields and
pagination, the ``/metadata/<id>`` endpoint, and ``/download/...`` with HTTP
Range resume. No network access required.
"""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tantra.scraper import (
    ArchiveError,
    InternetArchiveClient,
    build_query,
    search_candidates,
)

# --------------------------------------------------------------------- fake server
PAGE1_TEXT = ("সাধনা তন্ত্রের প্রথম পৃষ্ঠা। বীজমন্ত্র ও ন্যাসের বর্ণনা এখানে আছে।\n" * 8).encode("utf-8")
PAGE2_TEXT = ("দ্বিতীয় পৃষ্ঠা: কুণ্ডলিনী যোগ ও ষট্চক্র ভেদনের বিবরণ।\n" * 8).encode("utf-8")
# A file whose name contains non-ASCII, as live scanned items do.
UNICODE_NAME = "পীঠমালা মহাতন্ত্র_djvu.txt"


class FakeArchiveApp:
    """Request handlers backed by in-memory state, with request counters."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.search_requests = 0
        self.search_fields: list[list[str]] = []
        self.metadata_requests: list[str] = []
        self.download_requests: list[str] = []
        self.fail_downloads: dict[str, int] = {}  # path -> remaining 500s
        self.corrupt_once: set[str] = set()
        self.docs: list[dict] = []
        self.items: dict[str, dict] = {}

    def handle(self, handler: BaseHTTPRequestHandler) -> None:
        parsed = urllib.parse.urlparse(handler.path)
        query = urllib.parse.parse_qs(parsed.query)
        path = parsed.path

        if path == "/advancedsearch.php":
            with self.lock:
                self.search_requests += 1
                self.search_fields.append(sorted({v for k, vs in query.items() if k.startswith("fl[") for v in vs}))
            q = (query.get("q") or [""])[0]
            if "boom" in q:
                self._json(handler, 200, {"error": "org.apache.lucene.queryParser.ParseException: bad query"})
                return
            try:
                rows = int((query.get("rows") or ["100"])[0])
                page = int((query.get("page") or ["1"])[0])
            except ValueError:
                self._json(handler, 200, {"error": "invalid rows/page"})
                return
            start = (page - 1) * rows
            docs = self.docs[start : start + rows]
            self._json(handler, 200, {"response": {"numFound": len(self.docs), "start": start, "docs": docs}})
            return

        if path.startswith("/metadata/"):
            ident = urllib.parse.unquote(path[len("/metadata/") :])
            with self.lock:
                self.metadata_requests.append(ident)
            item = self.items.get(ident)
            if item is None:
                self._json(handler, 200, {"metadata": {}})  # live API: unknown items give empty metadata
            else:
                payload = {k: v for k, v in item.items() if k != "blob"}  # blobs are served by /download only
                self._json(handler, 200, payload)
            return

        if path.startswith("/download/"):
            rest = urllib.parse.unquote(path[len("/download/") :])
            ident, _, filename = rest.partition("/")
            with self.lock:
                self.download_requests.append(f"{ident}/{filename}")
                if self.fail_downloads.get(f"{ident}/{filename}", 0) > 0:
                    self.fail_downloads[f"{ident}/{filename}"] -= 1
                    handler.send_response(500)
                    handler.end_headers()
                    return
            payload = (self.items.get(ident, {}).get("blob", {})).get(filename)
            if payload is None:
                handler.send_response(404)
                handler.end_headers()
                return
            if f"{ident}/{filename}" in self.corrupt_once:
                with self.lock:
                    self.corrupt_once.discard(f"{ident}/{filename}")
                payload = payload[:-1] + b"?"  # poisoned bytes, MD5 will not match
            total = len(payload)
            rng = (handler.headers.get("Range") or "").replace("bytes=", "")
            if rng:
                start = int(rng.split("-")[0])
                data = payload[start:]
                handler.send_response(206)
                handler.send_header("Content-Range", f"bytes {start}-{total - 1}/{total}")
            else:
                start = 0
                data = payload
                handler.send_response(200)
            handler.send_header("Content-Length", str(len(data)))
            handler.send_header("Content-Type", "application/octet-stream")
            handler.end_headers()
            handler.wfile.write(data)
            return

        handler.send_response(404)
        handler.end_headers()

    def _json(self, handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)


def _make_item(text: bytes, extra_files: dict[str, bytes] | None = None) -> dict:
    blobs: dict[str, bytes] = {UNICODE_NAME: text}
    blobs.update(extra_files or {})
    files = [
        {
            "name": name,
            "format": "DjVuTXT" if name.endswith(".txt") else "Image Container PDF",
            "size": len(data),
            "md5": hashlib.md5(data).hexdigest(),
        }
        for name, data in blobs.items()
    ]
    return {"metadata": {"identifier": "x", "title": "t"}, "files": files, "blob": blobs}


@pytest.fixture()
def server():
    app = FakeArchiveApp()
    app.docs = [
        {
            "identifier": f"in.ernet.dli.2015.32{i}4631",
            "title": ["Adhyatma-sadhana", "অধ্যাত্ম সাধনা"][i % 2],
            "year": 1910 + i,  # live Solr: numeric year
            "language": "ben",
            "licenseurl": "",
            "subject": "tantra",
        }
        for i in range(5)
    ]
    app.items["dli.item"] = _make_item(PAGE1_TEXT + PAGE2_TEXT, {"dli.item.pdf": b"%PDF-1.4 fake"})
    handler = type(
        "Handler",
        (BaseHTTPRequestHandler,),
        {"app": app, "do_GET": lambda self: self.app.handle(self), "log_message": lambda *a: None},
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield app, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _client(base_url: str, **kwargs) -> InternetArchiveClient:
    return InternetArchiveClient(base_url, sleep=lambda s: None, **kwargs)


# --------------------------------------------------------------------- search
def test_search_paginates_and_parses_live_shapes(server):
    app, base = server
    client = _client(base, page_size=2, page_delay=0)
    records = list(client.search(build_query(), ["identifier", "year", "language", "subject", "title"], 5))
    assert len(records) == 5
    assert records[0]["identifier"].startswith("in.ernet.dli")
    assert isinstance(records[0]["year"], int)
    assert app.search_requests == 3  # 2 + 2 + 1 docs over three pages


def test_search_respects_the_limit_even_across_pages(server):
    app, base = server
    client = _client(base, page_size=2, page_delay=0)
    records = list(client.search(build_query(), ["identifier"], 3))
    assert len(records) == 3
    assert app.search_requests == 2


def test_search_raises_loudly_on_api_errors(server):
    _, base = server
    client = _client(base)
    with pytest.raises(ArchiveError, match="rejected the search query"):
        list(client.search("boom AND mediatype:texts", ["identifier"], 10))


def test_search_always_requests_the_identifier_field(server):
    app, base = server
    client = _client(base, page_delay=0)
    list(client.search("mediatype:texts", ["title"], 2))
    assert app.search_fields and "identifier" in app.search_fields[0]
    assert "title" in app.search_fields[0]


def test_search_end_to_end_with_evaluation(server):
    _, base = server
    client = _client(base, page_delay=0)
    rows = search_candidates(client, limit=5, query=build_query())
    assert len(rows) == 5
    assert all(dec.accepted for _, dec in rows), [d.reasons for _, d in rows]
    first = rows[0][0]
    assert first.year is not None and 1870 <= first.year <= 1935
    assert "ben" in first.languages


# --------------------------------------------------------------------- metadata / files
def test_metadata_and_files_use_one_cached_request(server):
    app, base = server
    client = _client(base)
    files = client.files("dli.item")
    meta = client.metadata("dli.item")
    again = client.files("dli.item")
    assert app.metadata_requests == ["dli.item"]
    assert meta["identifier"] == "x"
    assert [f.name for f in files] == [UNICODE_NAME, "dli.item.pdf"]
    assert files[0].md5 == hashlib.md5(PAGE1_TEXT + PAGE2_TEXT).hexdigest()
    assert files[0].size == len(PAGE1_TEXT + PAGE2_TEXT)
    assert again == files


def test_files_of_an_unknown_item_is_empty(server):
    _, base = server
    client = _client(base)
    assert client.files("nope") == []
    assert client.metadata("nope") == {}


# --------------------------------------------------------------------- download
def test_download_verifies_md5_and_handles_unicode_names(server, tmp_path):
    app, base = server
    client = _client(base)
    target = client.download("dli.item", UNICODE_NAME, tmp_path)
    assert target == tmp_path / UNICODE_NAME
    assert target.read_bytes() == PAGE1_TEXT + PAGE2_TEXT
    assert not (tmp_path / (UNICODE_NAME + ".part")).exists()
    assert f"dli.item/{UNICODE_NAME}" in app.download_requests


def test_download_resumes_a_partial_file_with_range(server, tmp_path):
    app, base = server
    client = _client(base)
    full = PAGE1_TEXT + PAGE2_TEXT
    (tmp_path / (UNICODE_NAME + ".part")).write_bytes(full[:1000])
    target = client.download("dli.item", UNICODE_NAME, tmp_path)
    assert target.read_bytes() == full  # exactly the server bytes, no duplication
    ranges = [r for r in app.download_requests if r.startswith("dli.item/")]
    assert len(ranges) == 1


def test_download_deletes_poisoned_bytes_on_checksum_mismatch(server, tmp_path):
    app, base = server
    app.corrupt_once.add(f"dli.item/{UNICODE_NAME}")
    client = _client(base)
    with pytest.raises(ArchiveError, match="checksum mismatch"):
        client.download("dli.item", UNICODE_NAME, tmp_path)
    assert not (tmp_path / (UNICODE_NAME + ".part")).exists()
    assert not (tmp_path / UNICODE_NAME).exists()


def test_download_maps_http_500_and_404_to_archive_errors(server, tmp_path):
    app, base = server
    app.fail_downloads[f"dli.item/{UNICODE_NAME}"] = 1
    client = _client(base)
    with pytest.raises(ArchiveError, match="HTTP 500"):
        client.download("dli.item", UNICODE_NAME, tmp_path)
    with pytest.raises(ArchiveError, match="HTTP 404"):
        client.download("dli.item", "missing.txt", tmp_path)


def test_download_reports_size_mismatch(server, tmp_path):
    app, base = server
    # Publish a wrong size for the file, keep the real bytes.
    item = app.items["dli.item"]
    item["files"][0]["size"] = len(PAGE1_TEXT) - 1
    client = _client(base)
    with pytest.raises(ArchiveError, match="size mismatch"):
        client.download("dli.item", UNICODE_NAME, tmp_path)


def test_download_completes_a_fully_downloaded_part_file(server, tmp_path):
    full = PAGE1_TEXT + PAGE2_TEXT
    (tmp_path / (UNICODE_NAME + ".part")).write_bytes(full)
    _, base = server
    client = _client(base)
    target = client.download("dli.item", UNICODE_NAME, tmp_path)
    assert target.read_bytes() == full


def test_full_pipeline_search_evaluate_download_over_http(server, tmp_path):
    from tantra.scraper import download_items

    app, base = server
    app.items["in.ernet.dli.2015.3204631"] = _make_item(PAGE1_TEXT)
    client = _client(base, page_delay=0)
    rows = search_candidates(client, limit=5, query=build_query())
    accepted = [(c, d) for c, d in rows if d.accepted]
    results = download_items(
        client, accepted[:1], out_dir=tmp_path, formats=("djvu_txt",), delay=0, sleep=lambda s: None, log=lambda m: None
    )
    assert [r.status for r in results] == ["downloaded"]
    item_dir = tmp_path / "in.ernet.dli.2015.3204631"
    assert (item_dir / "metadata.json").exists()
    downloaded = list(item_dir.glob("*_djvu.txt"))
    assert downloaded and downloaded[0].read_bytes() == PAGE1_TEXT
