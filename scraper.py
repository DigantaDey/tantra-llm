#!/usr/bin/env python3
"""Find and download historical Bengali Tantra OCR sources from Internet Archive.

The script is deliberately read-only with respect to Archive.org: it searches
public item metadata and downloads selected source files, but never modifies an
item. Rights metadata is treated as evidence, not as a legal determination.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

SEARCH_FIELDS = [
    "identifier",
    "title",
    "date",
    "year",
    "language",
    "subject",
    "description",
    "mediatype",
    "rights",
    "licenseurl",
    "access-restricted-item",
]

BENGALI_TERMS = ("তন্ত্র", "তান্ত্রিক", "সাধন", "সাধনা", "মন্ত্র", "শাক্ত")
LATIN_TERMS = ("tantra", "sadhana", "mantra", "shakta", "shakti")
SEARCH_QUERIES = (
    (
        "mediatype:texts AND (title:tantra OR title:sadhana OR title:mantra OR "
        "title:shakta OR subject:tantra OR subject:sadhana OR subject:mantra)"
    ),
    (
        "mediatype:texts AND (title:তন্ত্র OR title:তান্ত্রিক OR title:সাধন OR "
        "title:সাধনা OR title:মন্ত্র OR subject:তন্ত্র OR subject:সাধন OR subject:মন্ত্র)"
    ),
)
YEAR_PRIORITY_START = 1870
YEAR_PRIORITY_END = 1939


def _values(value: Any) -> list[str]:
    """Flatten metadata fields, which Archive.org may return as lists or mappings."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [str(item) for item in value.values() if item is not None]
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_values(item))
        return result
    return [str(value)]


def _field_text(metadata: Mapping[str, Any], *fields: str) -> str:
    return " ".join(
        text for field in fields for text in _values(metadata.get(field))
    ).strip()


def publication_year(metadata: Mapping[str, Any]) -> int | None:
    """Return the first plausible four-digit year, without guessing from an ID."""
    explicit_year = _field_text(metadata, "year")
    # publicdate is the Archive.org upload timestamp, not the book's publication date.
    date_text = _field_text(metadata, "date")
    for value in (explicit_year, date_text):
        match = re.search(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)", value)
        if match:
            year = int(match.group(1))
            if 1500 <= year <= 2099:
                return year
    return None


def has_bengali_language(metadata: Mapping[str, Any]) -> bool:
    """Recognize common Archive.org names and ISO language codes."""
    language = _field_text(metadata, "language", "lang").casefold()
    return bool(
        re.search(r"\b(bengali|bangla|ben|bn)\b", language)
        or "বাংলা" in language
        or "বাঙালি" in language
    )


def rights_status(metadata: Mapping[str, Any]) -> str:
    """Classify only explicit rights signals; unknown is not silently called PD."""
    rights = _field_text(metadata, "rights", "licenseurl", "license", "rightsinfo").casefold()
    restricted = _field_text(
        metadata,
        "access-restricted-item",
        "lendinglibrary",
        "is_dark",
    ).casefold()
    if any(value in restricted for value in ("true", "yes", "restricted", "dark")):
        return "restricted"
    if any(
        marker in rights
        for marker in (
            "public domain",
            "publicdomain",
            "creativecommons.org/publicdomain",
            "cc0",
        )
    ):
        return "public-domain-indicated"
    if any(marker in rights for marker in ("copyright", "all rights reserved", "in copyright")):
        return "rights-restricted-or-unclear"
    return "unverified"


def metadata_score(metadata: Mapping[str, Any]) -> int:
    """Rank likely Bengali historical works; do not treat this score as proof."""
    title = _field_text(metadata, "title").casefold()
    descriptive = _field_text(metadata, "subject", "description").casefold()
    score = 0

    if has_bengali_language(metadata):
        score += 5
    elif re.search(r"[\u0980-\u09ff]", title):
        score += 3

    if any(term in title for term in BENGALI_TERMS + LATIN_TERMS):
        score += 3
    elif any(term in descriptive for term in BENGALI_TERMS + LATIN_TERMS):
        score += 2

    year = publication_year(metadata)
    if year is not None:
        if YEAR_PRIORITY_START <= year <= YEAR_PRIORITY_END:
            score += 3
        elif 1850 <= year <= 1949:
            score += 1
        else:
            score -= 1

    rights = rights_status(metadata)
    if rights == "public-domain-indicated":
        score += 2
    elif rights in ("restricted", "rights-restricted-or-unclear"):
        score -= 5

    mediatype = _field_text(metadata, "mediatype").casefold()
    if not mediatype or "texts" in mediatype:
        score += 1
    return score


def is_eligible(
    metadata: Mapping[str, Any],
    *,
    strict_language: bool = False,
    require_public_domain: bool = False,
) -> bool:
    if strict_language and not has_bengali_language(metadata):
        return False
    if require_public_domain and rights_status(metadata) != "public-domain-indicated":
        return False
    return rights_status(metadata) != "restricted"


def _safe_identifier(value: Any) -> str | None:
    identifier = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", identifier):
        return None
    return identifier


def _entry_value(entry: Any, key: str, default: Any = None) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(key, default)
    getter = getattr(entry, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(entry, key, default)


def _safe_archive_path(name: str) -> bool:
    # IA filenames are POSIX paths even when the scraper runs on Windows.
    if not name or "\\" in name or "\x00" in name:
        return False
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and not (path.parts and path.parts[0].endswith(":"))
        and all(part not in ("", ".", "..") for part in path.parts)
    )


def select_files(files: Iterable[Any], *, all_scans: bool = False) -> dict[str, list[str]]:
    """Choose one OCR transcript and one scan (PDF preferred), if present."""
    transcripts: list[tuple[int, int, str]] = []
    scans: list[tuple[int, int, str]] = []
    seen: set[str] = set()

    for entry in files:
        name = str(_entry_value(entry, "name", "") or "").strip()
        if not _safe_archive_path(name) or name in seen:
            continue
        seen.add(name)
        lower = name.casefold()
        basename = PurePosixPath(name).name.casefold()
        if basename.startswith(".") or "/history/" in f"/{lower}":
            continue

        if lower.endswith(".txt") and not any(
            marker in basename
            for marker in ("_files", "_meta", "_metadata", "_scandata", "_marc")
        ):
            rank = 100 if "_djvu.txt" in basename else 85 if "_text.txt" in basename else 50
            size = _as_size(_entry_value(entry, "size", 0))
            transcripts.append((rank, size, name))
            continue

        suffix = PurePosixPath(lower).suffix
        if suffix not in (".pdf", ".djvu", ".djv"):
            continue
        source = str(_entry_value(entry, "source", "") or "").casefold()
        rank = 30 if suffix == ".pdf" else 20
        if source == "original":
            rank += 10
        if "_text" in basename or "textonly" in basename:
            rank -= 15
        scans.append((rank, _as_size(_entry_value(entry, "size", 0)), name))

    transcripts.sort(key=lambda row: (row[0], row[1]), reverse=True)
    scans.sort(key=lambda row: (row[0], row[1]), reverse=True)
    chosen_scans = [row[2] for row in (scans if all_scans else scans[:1])]
    return {"scan": chosen_scans, "ocr": [transcripts[0][2]] if transcripts else []}


def _as_size(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _merge_metadata(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(old)
    for key, value in new.items():
        if value not in (None, "", [], {}):
            merged[key] = value
    return merged


def discover_candidates(
    ia_module: Any,
    *,
    rows_per_query: int = 100,
    strict_language: bool = False,
    require_public_domain: bool = False,
) -> list[dict[str, Any]]:
    """Search IA using its supported Python client and return deduplicated records."""
    found: dict[str, dict[str, Any]] = {}
    query_failures = 0
    for query in SEARCH_QUERIES:
        try:
            result_stream = ia_module.search_items(
                query,
                fields=SEARCH_FIELDS,
                sorts=["downloads desc"],
                params={"rows": max(1, int(rows_per_query)), "page": 1},
                request_kwargs={"timeout": 60},
                max_retries=5,
            )
            for result in result_stream:
                if not isinstance(result, Mapping):
                    continue
                if result.get("error"):
                    raise RuntimeError(str(result["error"]))
                identifier = _safe_identifier(result.get("identifier"))
                if not identifier:
                    continue
                found[identifier] = _merge_metadata(found.get(identifier, {}), result)
        except Exception as exc:  # noqa: BLE001 - one failed remote query must not suppress the other script.
            query_failures += 1
            print(f"[warn] Archive.org search failed for query {query!r}: {exc}", file=sys.stderr)

    if query_failures == len(SEARCH_QUERIES):
        raise RuntimeError("All Archive.org searches failed; check network access and the Internet Archive API.")

    candidates = []
    for identifier, metadata in found.items():
        if not is_eligible(
            metadata,
            strict_language=strict_language,
            require_public_domain=require_public_domain,
        ):
            continue
        candidates.append(
            {
                "identifier": identifier,
                "metadata": metadata,
                "score": metadata_score(metadata),
                "rights_status": rights_status(metadata),
                "publication_year": publication_year(metadata),
            }
        )
    candidates.sort(
        key=lambda record: (
            -record["score"],
            record["publication_year"] is None,
            record["identifier"].casefold(),
        )
    )
    return candidates


def download_candidate(
    ia_module: Any,
    candidate: Mapping[str, Any],
    raw_dir: Path,
    *,
    dry_run: bool = False,
    all_scans: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    identifier = _safe_identifier(candidate.get("identifier"))
    if identifier is None:
        raise ValueError("Unsafe or invalid Archive.org identifier.")
    item = ia_module.get_item(identifier)
    item_metadata = getattr(item, "metadata", {}) or {}
    merged_metadata = _merge_metadata(candidate.get("metadata", {}), item_metadata)
    selected = select_files(getattr(item, "files", []) or [], all_scans=all_scans)
    expected = selected["scan"] + selected["ocr"]
    record = {
        "identifier": identifier,
        "title": _field_text(merged_metadata, "title") or identifier,
        "date": _field_text(merged_metadata, "date", "year"),
        "language": _field_text(merged_metadata, "language"),
        "score": int(candidate.get("score", metadata_score(merged_metadata))),
        "rights_status": rights_status(merged_metadata),
        "source_url": f"https://archive.org/details/{identifier}",
        "selected_files": selected,
        "downloaded_files": [],
        "missing_assets": [key for key in ("scan", "ocr") if not selected[key]],
        "status": "planned" if dry_run else "pending",
    }

    item_dir = raw_dir / identifier
    if dry_run:
        return record

    if expected:
        # no_directory=False creates raw/<identifier>/<archive filename>.
        download_errors = item.download(
            files=expected,
            destdir=str(raw_dir),
            no_directory=False,
            ignore_existing=True,
            retries=3,
            checksum=False,
            verbose=verbose,
        )
        if download_errors:
            record["download_errors"] = [str(error) for error in download_errors]
        for name in expected:
            local_path = item_dir.joinpath(*PurePosixPath(name).parts)
            if local_path.is_file():
                record["downloaded_files"].append(str(local_path))
            else:
                record["missing_assets"].append(name)
        downloaded_count = len(record["downloaded_files"])
        if downloaded_count == len(expected):
            record["status"] = "downloaded"
        elif downloaded_count:
            record["status"] = "partial"
        else:
            record["status"] = "download-failed"
    else:
        record["status"] = "no-assets"
    if item_metadata:
        _atomic_json(item_dir / "metadata.json", item_metadata)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search Archive.org for historical Bengali Tantra texts and download OCR + scans."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Download destination (default: data/raw).")
    parser.add_argument("--manifest", type=Path, help="Manifest JSON path (default: <output-dir>/manifest.json).")
    parser.add_argument("--max-items", type=int, default=30, help="Maximum ranked items to process; use 0 for all returned candidates.")
    parser.add_argument("--rows-per-query", type=int, default=100, help="Archive.org results fetched per search query (max 10,000).")
    parser.add_argument("--strict-language", action="store_true", help="Keep only items explicitly tagged Bengali/Bangla/ben/bn.")
    parser.add_argument("--require-public-domain", action="store_true", help="Require explicit public-domain metadata; unknown rights are otherwise reported, not inferred.")
    parser.add_argument("--all-scans", action="store_true", help="Download every available PDF/DJVU scan instead of one preferred scan.")
    parser.add_argument("--dry-run", action="store_true", help="Search and plan downloads without transferring files.")
    parser.add_argument("--quiet-downloads", action="store_true", help="Suppress per-file download progress from the IA client.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_items < 0:
        print("--max-items must be zero or a positive integer", file=sys.stderr)
        return 2
    if args.rows_per_query < 1 or args.rows_per_query > 10_000:
        print("--rows-per-query must be between 1 and 10000", file=sys.stderr)
        return 2
    try:
        import internetarchive as ia
    except ImportError:
        print("Missing dependency: install with `python -m pip install -e .` or `pip install internetarchive`.", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.output_dir / "manifest.json")
    print("Searching Archive.org (English and Bengali title/subject queries)…")
    try:
        candidates = discover_candidates(
            ia,
            rows_per_query=args.rows_per_query,
            strict_language=args.strict_language,
            require_public_domain=args.require_public_domain,
        )
    except RuntimeError as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1
    if args.max_items:
        candidates = candidates[: args.max_items]
    print(f"Selected {len(candidates)} candidate(s); downloading source scans and OCR where available.")

    records = []
    for index, candidate in enumerate(candidates, start=1):
        metadata = candidate["metadata"]
        title = _field_text(metadata, "title") or candidate["identifier"]
        print(
            f"[{index}/{len(candidates)}] {candidate['identifier']} — {title} "
            f"(score {candidate['score']}, rights: {candidate['rights_status']})"
        )
        try:
            record = download_candidate(
                ia,
                candidate,
                args.output_dir,
                dry_run=args.dry_run,
                all_scans=args.all_scans,
                verbose=not args.quiet_downloads,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch going and record per-item failures.
            record = {
                "identifier": candidate["identifier"],
                "title": title,
                "source_url": f"https://archive.org/details/{candidate['identifier']}",
                "score": candidate["score"],
                "rights_status": candidate["rights_status"],
                "status": "error",
                "error": str(exc),
            }
            print(f"[error] {candidate['identifier']}: {exc}", file=sys.stderr)
        records.append(record)
        _atomic_json(manifest_path, {"records": records, "candidate_count": len(candidates)})

    print(f"Manifest: {manifest_path}")
    if not records:
        print("No candidates matched the search and metadata filters. Try a larger --rows-per-query or omit --strict-language.")
    failed_statuses = {"error", "partial", "download-failed"}
    return 0 if all(record.get("status") not in failed_statuses for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
