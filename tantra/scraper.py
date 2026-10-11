"""Internet Archive scraping: find Tantra/Sadhana texts in Bengali, verify them, download them.

Design notes
------------
* The network is hidden behind :class:`ArchiveClient` so the selection logic is
  pure Python and fully unit-tested with fakes. The production implementation
  (:class:`InternetArchiveClient`) wraps the official ``internetarchive`` library.
* Every downloaded file is checked against the MD5 that archive.org publishes
  for it. A mismatch is deleted and retried, never silently kept.
* Re-running is cheap: files that already exist with a matching checksum are skipped.
* Licence gating is conservative: only items whose metadata explicitly says
  "public domain" are accepted unless the caller opts into reviewing unmarked items.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

# ---------------------------------------------------------------- query design
TANTRA_TERMS_EN = ("tantra", "tantric", "tantrik", "sadhan", "sadhana", "sadhaka", "shakta", "mantra")
TANTRA_TERMS_BN = ("তন্ত্র", "তান্ত্রিক", "সাধন", "সাধনা", "সাধক", "শাক্ত", "মন্ত্র")
BENGALI_LANGUAGE_NAMES = frozenset({"bengali", "bangla", "bengali; bangla", "ben", "beng", "bn", "bangali"})
PUBLIC_DOMAIN_MARKERS = ("publicdomain", "public domain", "pd-", "cc0")
DEFAULT_YEAR_RANGE = (1870, 1935)  # "late 19th to early 20th century"
DEFAULT_FORMATS = ("djvu_txt", "pdf", "djvu")
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


def build_query(year_range: tuple[int, int] = DEFAULT_YEAR_RANGE) -> str:
    """Lucene query for archive.org advanced search."""
    lo, hi = year_range
    if lo > hi:
        raise ValueError(f"invalid year range {lo}..{hi}")
    terms = " OR ".join(f'"{t}"' for t in (*TANTRA_TERMS_EN, *TANTRA_TERMS_BN))
    languages = " OR ".join(f'"{name}"' for name in ("Bengali", "Bangla", "ben", "beng"))
    return (
        f"mediatype:texts AND language:({languages}) AND year:[{lo} TO {hi}] AND (title:({terms}) OR subject:({terms}))"
    )


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
    status: str  # downloaded | skipped | failed | dry-run
    files: dict[str, str] = field(default_factory=dict)  # kind -> filename
    missing: list[str] = field(default_factory=list)
    error: str | None = None


class ArchiveClient(Protocol):
    """The only part of the system that touches the network."""

    def search(self, query: str, fields: Iterable[str], limit: int) -> Iterator[dict[str, Any]]: ...

    def metadata(self, identifier: str) -> dict[str, Any]: ...

    def files(self, identifier: str) -> list[RemoteFile]: ...

    def download(self, identifier: str, filename: str, dest_dir: Path) -> None: ...


# ---------------------------------------------------------------- parsing helpers
def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if v is not None]


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
        languages.extend(p.strip().lower() for p in re.split(r"[;,/]", lang) if p.strip())
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


def evaluate(
    candidate: Candidate,
    *,
    year_range: tuple[int, int] = DEFAULT_YEAR_RANGE,
    require_public_domain: bool = True,
    allow_undated: bool = False,
) -> Decision:
    """Decide whether an item belongs in the corpus, and explain why."""
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

    licence_text = f"{candidate.licenseurl} {candidate.rights}".lower()
    if require_public_domain and not any(marker in licence_text for marker in PUBLIC_DOMAIN_MARKERS):
        reasons.append("not marked public domain (review manually with --include-unmarked)")

    return Decision(accepted=not reasons, score=score, reasons=tuple(reasons))


def select_files(
    identifier: str, files: Iterable[RemoteFile], formats: Iterable[str]
) -> tuple[dict[str, str], list[str]]:
    """Pick one file per requested kind. Returns ``(chosen, missing_kinds)``."""
    files = [f for f in files if "/" not in f.name and not f.name.startswith(".")]
    chosen: dict[str, str] = {}
    wanted = list(formats)
    unknown = [k for k in wanted if k not in DEFAULT_FORMATS]
    if unknown:
        raise ValueError(f"unknown format(s): {', '.join(unknown)}; choose from {', '.join(DEFAULT_FORMATS)}")

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
    require_public_domain: bool = True,
    allow_undated: bool = False,
    query: str | None = None,
) -> list[tuple[Candidate, Decision]]:
    """Run the archive search and return every hit with its accept/reject decision."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    results: list[tuple[Candidate, Decision]] = []
    for record in client.search(query or build_query(year_range), SEARCH_FIELDS, limit):
        candidate = parse_candidate(record)
        if not candidate.identifier:
            continue
        decision = evaluate(
            candidate,
            year_range=year_range,
            require_public_domain=require_public_domain,
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
) -> None:
    target = dest_dir / filename
    remote_file = remote.get(filename)
    expected = remote_file.md5 if remote_file else None
    if target.exists() and expected and md5_of(target) == expected:
        return  # already complete
    last_error: str = ""
    for attempt in range(1, retries + 2):
        try:
            client.download(identifier, filename, dest_dir)
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
            for name in chosen.values():
                _download_verified(client, ident, name, item_dir, remote, retries, sleep)
            result = DownloadResult(ident, "downloaded", files=chosen, missing=missing)
            _atomic_write_json(
                item_dir / "metadata.json",
                {
                    "identifier": ident,
                    "title": candidate.title,
                    "year": candidate.year,
                    "languages": list(candidate.languages),
                    "licenseurl": candidate.licenseurl,
                    "rights": candidate.rights,
                    "subjects": list(candidate.subjects),
                    "source_url": f"https://archive.org/details/{ident}",
                    "files": chosen,
                    "missing_formats": missing,
                    "score": decision.score,
                    "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
            )
            log(f"✓ {ident}: {', '.join(sorted(chosen)) or 'nothing matched'}")
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
class InternetArchiveClient:
    """Adapter over the official ``internetarchive`` package."""

    def __init__(self, request_timeout: float = 60.0) -> None:
        self.timeout = request_timeout

    def search(self, query: str, fields: Iterable[str], limit: int) -> Iterator[dict[str, Any]]:
        import internetarchive as ia

        search = ia.search_items(query, fields=list(fields), params={"rows": min(max(limit, 1), 500)})
        for record in search.iter_as_results():
            yield dict(record)

    def metadata(self, identifier: str) -> dict[str, Any]:
        import internetarchive as ia

        return dict(ia.get_item(identifier).metadata)

    def files(self, identifier: str) -> list[RemoteFile]:
        import internetarchive as ia

        item = ia.get_item(identifier)
        out: list[RemoteFile] = []
        for f in item.files:
            size = f.get("size")
            out.append(
                RemoteFile(
                    name=str(f.get("name", "")),
                    format=str(f.get("format", "")),
                    size=int(size) if size not in (None, "") else None,
                    md5=f.get("md5") or None,
                )
            )
        return out

    def download(self, identifier: str, filename: str, dest_dir: Path) -> None:
        import internetarchive as ia

        item = ia.get_item(identifier)
        file_obj = item.get_file(filename)
        ok = file_obj.download(
            destdir=str(dest_dir),
            file_path=filename,
            retries=2,
            timeout=self.timeout,
            ignore_existing=True,
        )
        if not ok:
            raise OSError(f"internetarchive reported failure for {filename}")


def manifest_entries(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "manifest.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = [
    "ArchiveClient",
    "Candidate",
    "Decision",
    "DownloadResult",
    "InternetArchiveClient",
    "RemoteFile",
    "build_query",
    "download_items",
    "evaluate",
    "md5_of",
    "parse_candidate",
    "search_candidates",
    "select_files",
]
