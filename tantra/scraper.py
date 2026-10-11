"""Internet Archive scraping: find Tantra/Sadhana texts in Bengali, verify them, download them.

Design notes
------------
* The network is hidden behind :class:`ArchiveClient` so the selection logic is
  pure Python and fully unit-tested with fakes. The production implementation
  (:class:`InternetArchiveClient`) speaks to the three public archive.org
  endpoints directly over HTTPS (stdlib only, no SDK):

  ==========================  ===================================================
  ``advancedsearch.php``      item search (Solr query, paginated)
  ``/metadata/<identifier>``  item metadata + file list (name, size, MD5)
  ``/download/<id>/<file>``   file download (redirects to a mirror node)
  ==========================  ===================================================

  The endpoints were verified against the live service (see the ``live-archive``
  CI job); notable real-world shapes the client must survive: ``year`` arrives
  as a JSON *number*, filenames may contain non-ASCII Bengali characters, and
  the search API reports a query error as ``{"error": ...}`` with HTTP 200.
* Every downloaded file is checked against the MD5 that archive.org publishes
  for it, streamed to a ``.part`` file and only moved into place once verified.
  A mismatch raises, and the caller retries or records the failure - corrupt
  bytes are never silently kept, and interrupted downloads resume.
* Re-running is cheap: files that already exist with a matching checksum are skipped.
* Licence gating is explicit and configurable. The default (:data:`RIGHTS_ANY`)
  accepts items regardless of licence marking and records the exact licence
  fields with every download, because almost no real Bengali Tantra scan on
  archive.org carries a licence mark. ``--require-marked-pd`` restores the
  conservative marked-public-domain-only behaviour.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

# ---------------------------------------------------------------- query design
TANTRA_TERMS_EN = ("tantra", "tantric", "tantrik", "sadhan", "sadhana", "sadhaka", "shakta", "mantra")
TANTRA_TERMS_BN = ("তন্ত্র", "তান্ত্রিক", "সাধন", "সাধনা", "সাধক", "শাক্ত", "মন্ত্র")
BENGALI_LANGUAGE_NAMES = frozenset({"bengali", "bangla", "bengali; bangla", "ben", "beng", "bn", "bangali"})
BENGALI_QUERY_LANGUAGES = ("Bengali", "Bangla", "ben", "beng")
PUBLIC_DOMAIN_MARKERS = ("publicdomain", "public domain", "pd-", "cc0")
DEFAULT_YEAR_RANGE = (1870, 1935)  # "late 19th to early 20th century"
DEFAULT_FORMATS = ("djvu_txt", "pdf")  # .djvu is supported but rarely exists on scanned items
SEARCH_FIELDS = (
    "identifier",
    "title",
    "creator",
    "year",
    "date",
    "language",
    "licenseurl",
    "rights",
    "subject",
    "mediatype",
)

# Rights policies. The default accepts unmarked items (with the licence recorded);
# "marked" accepts only items whose licence explicitly says public domain.
RIGHTS_ANY = "any"
RIGHTS_MARKED = "marked"
RIGHTS_POLICIES = (RIGHTS_ANY, RIGHTS_MARKED)

# Callback: (bytes_done, total_bytes_or_None).
ProgressFn = Callable[[int, int | None], None]


def build_query(year_range: tuple[int, int] = DEFAULT_YEAR_RANGE, *, allow_undated: bool = False) -> str:
    """Lucene query for archive.org advanced search.

    With ``allow_undated`` the year clause is dropped so items without a
    publication year can be found at all - the local acceptance rule then keeps
    them only because the caller opted in.
    """
    lo, hi = year_range
    if lo > hi:
        raise ValueError(f"invalid year range {lo}..{hi}")
    terms = " OR ".join(f'"{t}"' for t in (*TANTRA_TERMS_EN, *TANTRA_TERMS_BN))
    languages = " OR ".join(f'"{name}"' for name in BENGALI_QUERY_LANGUAGES)
    year_clause = "" if allow_undated else f" AND year:[{lo} TO {hi}]"
    return f"mediatype:texts AND language:({languages}){year_clause} AND (title:({terms}) OR subject:({terms}))"


# ---------------------------------------------------------------- data model
@dataclass(frozen=True)
class Candidate:
    identifier: str
    title: str
    year: int | None
    languages: tuple[str, ...]
    licenseurl: str
    rights: str
    subjects: tuple[str, ...]
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class Decision:
    accepted: bool
    score: int
    reasons: tuple[str, ...]

    @property
    def status(self) -> str:
        return "accepted" if self.accepted else "rejected"


@dataclass(frozen=True)
class RemoteFile:
    name: str
    format: str
    size: int | None
    md5: str | None


@dataclass
class DownloadResult:
    identifier: str
    status: str  # downloaded | skipped | no-files | failed | dry-run
    files: dict[str, str] = field(default_factory=dict)  # kind -> filename
    missing: list[str] = field(default_factory=list)
    error: str | None = None


class ArchiveClient(Protocol):
    """The only part of the system that touches the network."""

    def search(self, query: str, fields: Iterable[str], limit: int) -> Iterator[dict[str, Any]]: ...

    def metadata(self, identifier: str) -> dict[str, Any]: ...

    def files(self, identifier: str) -> list[RemoteFile]: ...

    def download(
        self, identifier: str, filename: str, dest_dir: Path, *, on_progress: ProgressFn | None = None
    ) -> Path: ...


# ---------------------------------------------------------------- parsing helpers
def _as_list(value: Any) -> list[str]:
    """Normalise a Solr/JSON field to a list of strings.

    Live archive.org values arrive as strings, lists of strings - or plain
    numbers (``year`` is a numeric Solr field). Anything scalar is wrapped.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _parse_year(record: dict[str, Any]) -> int | None:
    for key in ("year", "date"):
        for value in _as_list(record.get(key)):
            match = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", value)
            if match:
                return int(match.group(1))
    return None


def parse_candidate(record: dict[str, Any]) -> Candidate:
    languages: list[str] = []
    for lang in _as_list(record.get("language")):
        # "Bengali; Bangla" and stray punctuation ("Bengali.") both occur in live data.
        languages.extend(p.strip(" .;").lower() for p in re.split(r"[;,/]", lang) if p.strip(" .;"))
    return Candidate(
        identifier=str(record.get("identifier", "")),
        title=" ".join(_as_list(record.get("title"))) or str(record.get("identifier", "")),
        year=_parse_year(record),
        languages=tuple(dict.fromkeys(languages)),
        licenseurl=" ".join(_as_list(record.get("licenseurl"))),
        rights=" ".join(_as_list(record.get("rights"))),
        subjects=tuple(_as_list(record.get("subject"))),
        raw=dict(record),
    )


def is_marked_public_domain(candidate: Candidate) -> bool:
    licence_text = f"{candidate.licenseurl} {candidate.rights}".lower()
    return any(marker in licence_text for marker in PUBLIC_DOMAIN_MARKERS)


def describe_rights(candidate: Candidate) -> str:
    """One-line, user-facing description of the licence basis (goes into metadata.json)."""
    if is_marked_public_domain(candidate):
        return "marked public domain on archive.org"
    if candidate.licenseurl or candidate.rights:
        return f"licence on record: {candidate.licenseurl or candidate.rights}"
    return "no licence marked on archive.org - verify rights before redistributing"


def evaluate(
    candidate: Candidate,
    *,
    year_range: tuple[int, int] = DEFAULT_YEAR_RANGE,
    rights_policy: str = RIGHTS_ANY,
    allow_undated: bool = False,
) -> Decision:
    """Decide whether an item belongs in the corpus, and explain why."""
    if rights_policy not in RIGHTS_POLICIES:
        raise ValueError(f"unknown rights policy {rights_policy!r}; choose from {', '.join(RIGHTS_POLICIES)}")

    reasons: list[str] = []
    if not any(lang in BENGALI_LANGUAGE_NAMES for lang in candidate.languages):
        reasons.append(f"language not Bengali ({', '.join(candidate.languages) or 'unknown'})")

    lo, hi = year_range
    if candidate.year is None:
        if not allow_undated:
            reasons.append("no publication year")
    elif not lo <= candidate.year <= hi:
        reasons.append(f"year {candidate.year} outside {lo}-{hi}")

    haystack = f"{candidate.title} {' '.join(candidate.subjects)}".lower()
    score = sum(haystack.count(term) for term in (*TANTRA_TERMS_EN, *TANTRA_TERMS_BN))
    if score == 0:
        reasons.append("no Tantra/Sadhana keyword in title or subject")

    if rights_policy == RIGHTS_MARKED and not is_marked_public_domain(candidate):
        reasons.append("not marked public domain (rerun without --require-marked-pd to accept unmarked scans)")

    return Decision(accepted=not reasons, score=score, reasons=tuple(reasons))


def select_files(
    identifier: str, files: Iterable[RemoteFile], formats: Iterable[str]
) -> tuple[dict[str, str], list[str]]:
    """Pick one file per requested kind. Returns ``(chosen, missing_kinds)``."""
    files = [f for f in files if "/" not in f.name and not f.name.startswith(".")]
    chosen: dict[str, str] = {}
    wanted = list(formats)
    unknown = [k for k in wanted if k not in ("djvu_txt", "pdf", "djvu")]
    if unknown:
        raise ValueError(f"unknown format(s): {', '.join(unknown)}; choose from djvu_txt, pdf, djvu")

    def pick(kind: str) -> RemoteFile | None:
        if kind == "djvu_txt":
            matches = [f for f in files if f.name.endswith("_djvu.txt")]
        elif kind == "pdf":
            matches = [f for f in files if f.name.lower().endswith(".pdf") and "thumb" not in f.format.lower()]
            matches.sort(key=lambda f: (f.name != f"{identifier}.pdf", f.name))
        else:  # djvu
            matches = [f for f in files if f.name.lower().endswith(".djvu")]
            matches.sort(key=lambda f: (f.name != f"{identifier}.djvu", f.name))
        return matches[0] if matches else None

    missing: list[str] = []
    for kind in wanted:
        found = pick(kind)
        if found is None:
            missing.append(kind)
        else:
            chosen[kind] = found.name
    return chosen, missing


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- orchestration
def search_candidates(
    client: ArchiveClient,
    *,
    limit: int,
    year_range: tuple[int, int] = DEFAULT_YEAR_RANGE,
    rights_policy: str = RIGHTS_ANY,
    allow_undated: bool = False,
    query: str | None = None,
) -> list[tuple[Candidate, Decision]]:
    """Run the archive search and return every hit with its accept/reject decision."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    results: list[tuple[Candidate, Decision]] = []
    for record in client.search(query or build_query(year_range, allow_undated=allow_undated), SEARCH_FIELDS, limit):
        candidate = parse_candidate(record)
        if not candidate.identifier:
            continue
        decision = evaluate(
            candidate,
            year_range=year_range,
            rights_policy=rights_policy,
            allow_undated=allow_undated,
        )
        results.append((candidate, decision))
        if len(results) >= limit:
            break
    results.sort(key=lambda pair: (not pair[1].accepted, -pair[1].score))
    return results


def _download_verified(
    client: ArchiveClient,
    identifier: str,
    filename: str,
    dest_dir: Path,
    remote: dict[str, RemoteFile],
    retries: int,
    sleep: Callable[[float], None],
    on_progress: ProgressFn | None = None,
) -> None:
    target = dest_dir / filename
    remote_file = remote.get(filename)
    expected = remote_file.md5 if remote_file else None
    if target.exists() and expected and md5_of(target) == expected:
        return  # already complete
    last_error: str = ""
    for attempt in range(1, retries + 2):
        try:
            client.download(identifier, filename, dest_dir, on_progress=on_progress)
        except Exception as exc:  # network / HTTP errors are retried
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if not target.exists():
                last_error = "download reported success but file is missing"
            elif expected and md5_of(target) != expected:
                last_error = f"checksum mismatch for {filename}"
                target.unlink()
            else:
                return
        if attempt <= retries:
            sleep(min(30.0, 2.0**attempt))
    raise OSError(f"{identifier}/{filename}: {last_error}")


def download_items(
    client: ArchiveClient,
    candidates: Iterable[tuple[Candidate, Decision]],
    *,
    out_dir: Path,
    formats: Iterable[str] = DEFAULT_FORMATS,
    delay: float = 1.0,
    retries: int = 3,
    dry_run: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    on_file_start: Callable[[str, str, int | None], None] | None = None,
    on_progress: ProgressFn | None = None,
) -> list[DownloadResult]:
    formats = tuple(formats)
    manifest_path = out_dir / "manifest.jsonl"
    results: list[DownloadResult] = []
    first = True
    for candidate, decision in candidates:
        if not decision.accepted:
            continue
        if not first and delay > 0 and not dry_run:
            sleep(delay)
        first = False
        ident = candidate.identifier
        item_dir = out_dir / ident

        if dry_run:
            result = DownloadResult(ident, "dry-run")
            log(f"[dry-run] would download {ident}: {candidate.title}")
            results.append(result)
            continue

        try:
            remote_files = client.files(ident)
            chosen, missing = select_files(ident, remote_files, formats)
            remote = {f.name: f for f in remote_files}
            item_dir.mkdir(parents=True, exist_ok=True)
            if not chosen:
                result = DownloadResult(ident, "no-files", files={}, missing=missing)
                log(f"○ {ident}: no usable files on the server (missing: {', '.join(missing) or 'none'})")
            else:
                for name in chosen.values():
                    if on_file_start is not None:
                        remote_file = remote.get(name)
                        on_file_start(ident, name, remote_file.size if remote_file else None)
                    _download_verified(client, ident, name, item_dir, remote, retries, sleep, on_progress)
                result = DownloadResult(ident, "downloaded", files=chosen, missing=missing)
                log(f"✓ {ident}: {', '.join(sorted(chosen)) or 'nothing matched'}")
            _atomic_write_json(
                item_dir / "metadata.json",
                {
                    "identifier": ident,
                    "title": candidate.title,
                    "year": candidate.year,
                    "languages": list(candidate.languages),
                    "licenseurl": candidate.licenseurl,
                    "rights": candidate.rights,
                    "rights_basis": describe_rights(candidate),
                    "subjects": list(candidate.subjects),
                    "source_url": f"https://archive.org/details/{ident}",
                    "files": chosen,
                    "missing_formats": missing,
                    "score": decision.score,
                    "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
            )
        except Exception as exc:  # keep going: one bad item must not stop the crawl
            result = DownloadResult(ident, "failed", error=str(exc))
            log(f"✗ {ident}: {exc}")
        results.append(result)
        _append_manifest(manifest_path, candidate, result)
    return results


def _append_manifest(path: Path, candidate: Candidate, result: DownloadResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "identifier": candidate.identifier,
        "title": candidate.title,
        "year": candidate.year,
        "status": result.status,
        "files": result.files,
        "missing": result.missing,
        "error": result.error,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- production client
class ArchiveError(RuntimeError):
    """An archive.org request failed; the message is user-facing."""


class InternetArchiveClient:
    """Direct HTTPS client for archive.org's public endpoints (stdlib only).

    * :meth:`search` pages through ``advancedsearch.php`` (100 rows per page)
      and turns an API ``{"error": ...}`` reply into a loud
      :class:`ArchiveError` instead of a silent empty result.
    * :meth:`metadata`/:meth:`files` use one cached ``/metadata/<id>`` request
      per item, so a download re-check does not re-fetch item metadata.
    * :meth:`download` streams to ``<name>.part`` with HTTP Range resume,
      verifies size and MD5 against the values archive.org publishes, and only
      then moves the file into place.
    """

    USER_AGENT = "tantra-llm/0.1 (https://github.com/DigantaDey/tantra-llm; archive.org scraper)"
    CHUNK_BYTES = 1 << 20

    def __init__(
        self,
        base_url: str = "https://archive.org",
        *,
        timeout: float = 60.0,
        page_size: int = 100,
        page_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.page_size = max(1, min(page_size, 10_000))
        self.page_delay = page_delay
        self.sleep = sleep
        self._item_cache: dict[str, dict[str, Any]] = {}

    # -------------------------------------------------- low-level HTTP
    def _open(self, url: str, *, headers: dict[str, str] | None = None) -> urllib.request.Request:
        return urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT, **(headers or {})})

    def _get_json(self, url: str) -> Any:
        try:
            with urllib.request.urlopen(self._open(url), timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ArchiveError(f"archive.org returned HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            raise ArchiveError(f"cannot reach archive.org: {exc.reason}") from exc
        except (OSError, ValueError) as exc:
            raise ArchiveError(f"bad response from archive.org: {exc}") from exc

    # -------------------------------------------------- search
    def search(self, query: str, fields: Iterable[str], limit: int) -> Iterator[dict[str, Any]]:
        """Yield up to ``limit`` result dicts from ``advancedsearch.php``."""
        if limit < 1:
            return
        wanted_fields = list(dict.fromkeys([*fields, "identifier"]))
        remaining = limit
        page = 1
        while remaining > 0:
            rows = min(self.page_size, remaining)
            params: dict[str, str] = {"q": query, "rows": str(rows), "page": str(page), "output": "json"}
            for i, name in enumerate(wanted_fields):
                params[f"fl[{i}]"] = name
            url = f"{self.base_url}/advancedsearch.php?{urllib.parse.urlencode(params)}"
            payload = self._get_json(url)
            if isinstance(payload, dict) and payload.get("error"):
                raise ArchiveError(f"archive.org rejected the search query: {payload['error']}")
            docs = ((payload or {}).get("response") or {}).get("docs") or []
            if not docs:
                return
            for doc in docs:
                if remaining <= 0:
                    return
                if isinstance(doc, dict):
                    yield dict(doc)
                    remaining -= 1
            if len(docs) < rows:
                return  # last page
            page += 1
            if self.page_delay > 0:
                self.sleep(self.page_delay)  # politeness between search pages

    # -------------------------------------------------- metadata / files
    def _item_json(self, identifier: str) -> dict[str, Any]:
        if identifier not in self._item_cache:
            url = f"{self.base_url}/metadata/{urllib.parse.quote(identifier, safe='')}"
            payload = self._get_json(url)
            if not isinstance(payload, dict):
                raise ArchiveError(f"unexpected metadata reply for {identifier}")
            self._item_cache[identifier] = payload
        return self._item_cache[identifier]

    def metadata(self, identifier: str) -> dict[str, Any]:
        return dict(self._item_json(identifier).get("metadata") or {})

    def files(self, identifier: str) -> list[RemoteFile]:
        out: list[RemoteFile] = []
        for f in self._item_json(identifier).get("files") or []:
            if not isinstance(f, dict):
                continue
            size = f.get("size")
            try:
                size_int = int(size) if size not in (None, "") else None
            except (TypeError, ValueError):
                size_int = None
            out.append(
                RemoteFile(
                    name=str(f.get("name", "")),
                    format=str(f.get("format", "")),
                    size=size_int,
                    md5=(f.get("md5") or None) if isinstance(f.get("md5"), str) else None,
                )
            )
        return out

    # -------------------------------------------------- download
    def download(
        self, identifier: str, filename: str, dest_dir: Path, *, on_progress: ProgressFn | None = None
    ) -> Path:
        """Stream ``identifier/filename`` into ``dest_dir``, resumable and verified."""
        target = dest_dir / filename
        part = target.with_name(target.name + ".part")
        remote = next((f for f in self.files(identifier) if f.name == filename), None)
        expected_md5 = remote.md5 if remote else None
        expected_size = remote.size if remote else None

        dest_dir.mkdir(parents=True, exist_ok=True)
        done = part.stat().st_size if part.is_file() else 0
        if done and expected_md5 and md5_of(part) == expected_md5:
            os.replace(part, target)  # a previous run already finished the transfer
            if on_progress:
                on_progress(target.stat().st_size, expected_size or target.stat().st_size)
            return target

        quoted = f"{urllib.parse.quote(identifier, safe='')}/{urllib.parse.quote(filename, safe='')}"
        url = f"{self.base_url}/download/{quoted}"
        headers: dict[str, str] = {}
        append = False
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            response = urllib.request.urlopen(self._open(url, headers=headers), timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise ArchiveError(f"HTTP {exc.code} while downloading {identifier}/{filename}") from exc
        except urllib.error.URLError as exc:
            raise ArchiveError(f"cannot reach archive.org for {filename}: {exc.reason}") from exc

        with response:
            status = getattr(response, "status", 200)
            length = response.headers.get("Content-Length")
            if status == 206 and done:  # server honoured the resume
                append = True
                total = done + int(length) if length else expected_size
            else:
                done = 0  # full response (or server ignored Range): start over
                total = int(length) if length else expected_size
            try:
                with part.open("ab" if append else "wb") as out:
                    while chunk := response.read(self.CHUNK_BYTES):
                        out.write(chunk)
                        done += len(chunk)
                        if on_progress:
                            on_progress(done, total)
            except (http.client.HTTPException, OSError) as exc:
                # The .part file is kept: the next attempt resumes from where this stopped.
                raise ArchiveError(f"connection lost after {done:,} bytes of {filename}: {exc}") from exc

        try:
            if expected_size is not None and part.stat().st_size != expected_size:
                got = part.stat().st_size
                raise ArchiveError(f"size mismatch for {filename}: expected {expected_size:,}, got {got:,}")
            if expected_md5 and md5_of(part) != expected_md5:
                raise ArchiveError(f"checksum mismatch for {filename}")
        except ArchiveError:
            part.unlink(missing_ok=True)  # poisoned bytes must never be resumed into place
            raise
        os.replace(part, target)
        return target


def manifest_entries(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "manifest.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = [
    "ArchiveClient",
    "ArchiveError",
    "Candidate",
    "Decision",
    "DownloadResult",
    "InternetArchiveClient",
    "ProgressFn",
    "RemoteFile",
    "RIGHTS_ANY",
    "RIGHTS_MARKED",
    "RIGHTS_POLICIES",
    "build_query",
    "describe_rights",
    "download_items",
    "evaluate",
    "is_marked_public_domain",
    "manifest_entries",
    "md5_of",
    "parse_candidate",
    "search_candidates",
    "select_files",
]
