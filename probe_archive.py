"""Temporary diagnostic #2: probe live archive.org download path. NOT part of the final PR."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.parse

import requests

from tantra.scraper import build_query, evaluate, parse_candidate

BASE = "https://archive.org"
UA = {"User-Agent": "tantra-llm-probe/0.1 (diagnostic)"}


def adv_search(q: str, rows: int = 100, fields: list[str] | None = None) -> dict:
    params = {"q": q, "rows": str(rows), "page": "1", "output": "json"}
    for i, f in enumerate(fields or ["identifier", "title", "year", "language", "licenseurl", "subject"]):
        params[f"fl[{i}]"] = f
    url = f"{BASE}/advancedsearch.php?{urllib.parse.urlencode(params)}"
    r = requests.get(url, headers=UA, timeout=60)
    j = r.json()
    if j.get("error"):
        print(f"SEARCH ERROR: {j['error']}")
        return {}
    return j


def main() -> None:
    q = build_query()
    j = adv_search(q, rows=100)
    docs = j.get("response", {}).get("docs", [])
    print(f"full query numFound={j['response']['numFound']} docs={len(docs)}")

    # 1) acceptance with an age-based PD policy simulation (year<=1930)
    accepted_age = 0
    accepted_marked = 0
    undated = 0
    for d in docs:
        d = dict(d)
        d["year"] = str(d["year"]) if d.get("year") is not None else None
        d["language"] = d["language"] if isinstance(d.get("language"), (list, str)) else str(d.get("language") or "")
        d["subject"] = d["subject"] if isinstance(d.get("subject"), (list, str)) else str(d.get("subject") or "")
        d["title"] = d["title"] if isinstance(d.get("title"), (list, str)) else str(d.get("title") or "")
        d["licenseurl"] = d["licenseurl"] if isinstance(d.get("licenseurl"), (list, str)) else str(d.get("licenseurl") or "")
        c = parse_candidate(d)
        dec_marked = evaluate(c, require_public_domain=True)
        if dec_marked.accepted:
            accepted_marked += 1
        if c.year is None:
            undated += 1
        dec_age = evaluate(c, require_public_domain=True)
        if (not dec_age.accepted) and dec_age.reasons == ("not marked public domain (review manually with --include-unmarked)",) and (c.year or 0) <= 1930:
            accepted_age += 1
        elif dec_age.accepted:
            accepted_age += 1
    print(f"accepted with marked-PD policy: {accepted_marked}")
    print(f"accepted with age<=1930 policy: {accepted_age}")
    print(f"undated among docs: {undated}")

    # 2) file inventory for the first 12 docs: which formats exist?
    print("\n--- file inventory ---")
    have_djvu_txt = 0
    have_pdf = 0
    checked = 0
    sample_ids = []
    for d in docs[:12]:
        ident = d["identifier"]
        try:
            m = requests.get(f"{BASE}/metadata/{ident}", headers=UA, timeout=60).json()
        except Exception as exc:
            print(f"  {ident}: metadata FAILED {exc}")
            continue
        checked += 1
        files = m.get("files", [])
        names = [f.get("name", "") for f in files]
        djvu_txt = [n for n in names if n.endswith("_djvu.txt")]
        pdfs = [n for n in names if n.lower().endswith(".pdf")]
        djvus = [n for n in names if n.lower().endswith(".djvu")]
        if djvu_txt:
            have_djvu_txt += 1
        if pdfs:
            have_pdf += 1
        print(f"  {ident}: files={len(files)} djvu_txt={djvu_txt[:1]} pdf={len(pdfs)} djvu={len(djvus)} formats={sorted({f.get('format') for f in files})[:8]}")
        if djvu_txt and len(sample_ids) < 3:
            sample_ids.append((ident, djvu_txt[0], next(f for f in files if f["name"] == djvu_txt[0])))
    print(f"  => {have_djvu_txt}/{checked} items have _djvu.txt, {have_pdf}/{checked} have pdf")

    # 3) download one djvu.txt via raw HTTP, verify md5
    print("\n--- raw download test ---")
    for ident, name, finfo in sample_ids[:2]:
        try:
            dl = f"{BASE}/download/{ident}/{urllib.parse.quote(name)}"
            r = requests.get(dl, headers=UA, timeout=120)
            md5 = hashlib.md5(r.content).hexdigest()
            size = len(r.content)
            ok = md5 == finfo.get("md5")
            print(f"  {ident}/{name}: http={r.status_code} bytes={size} md5_ok={ok} server={r.headers.get('Server', '?')!r}")
            print(f"    head: {r.content[:100]!r}")
        except Exception as exc:
            print(f"  {ident}/{name}: FAILED {type(exc).__name__}: {exc}")

    # 4) InternetArchiveClient end to end: search -> files -> download
    print("\n--- InternetArchiveClient test ---")
    try:
        from pathlib import Path

        import tempfile

        from tantra.scraper import InternetArchiveClient

        client = InternetArchiveClient()
        if not sample_ids:
            print("  no sample ids; skip")
            return
        ident, name, finfo = sample_ids[0]
        files = client.files(ident)
        print(f"  client.files({ident!r}) -> {len(files)} files")
        djvu = [f for f in files if f.name.endswith("_djvu.txt")]
        print(f"  djvu_txt files: {[(f.name, f.size, f.md5 and f.md5[:8]) for f in djvu]}")
        if djvu:
            with tempfile.TemporaryDirectory() as td:
                dest = Path(td)
                client.download(ident, djvu[0].name, dest)
                target = dest / djvu[0].name
                data = target.read_bytes()
                print(f"  client.download -> exists={target.exists()} bytes={len(data)} md5_ok={hashlib.md5(data).hexdigest() == djvu[0].md5}")
                print(f"  content head: {data[:80]!r}")
                bengali = sum(1 for ch in data.decode('utf-8', errors='replace')[:4000] if '\u0980' <= ch <= '\u09ff')
                print(f"  Bengali chars in first 4000: {bengali}")
    except Exception as exc:
        import traceback

        print(f"  InternetArchiveClient FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    # 5) advancedsearch rows cap: does rows=500 work?
    print("\n--- rows cap test ---")
    for rows in (50, 100, 200, 500):
        jj = adv_search("mediatype:texts AND language:ben AND title:tantra", rows=rows, fields=["identifier"])
        n = len(jj.get("response", {}).get("docs", [])) if jj else -1
        print(f"  rows={rows} -> got {n} docs (numFound={jj.get('response', {}).get('numFound') if jj else '?'})")


if __name__ == "__main__":
    sys.exit(main())
