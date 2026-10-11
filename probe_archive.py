"""Temporary diagnostic: probe live archive.org behaviour. NOT part of the final PR."""

from __future__ import annotations

import json
import sys
import urllib.parse

import requests  # noqa: F401 (kept: available in CI probe)

from tantra.scraper import build_query

BASE = "https://archive.org"
UA = {"User-Agent": "tantra-llm-probe/0.1 (diagnostic; contact: ci)"}


def adv_search(q: str, rows: int = 8, fields: list[str] | None = None, label: str = "") -> dict:
    params = {
        "q": q,
        "rows": str(rows),
        "page": "1",
        "output": "json",
    }
    for i, f in enumerate(fields or ["identifier", "title", "year", "language", "licenseurl", "subject"]):
        params[f"fl[{i}]"] = f
    url = f"{BASE}/advancedsearch.php?{urllib.parse.urlencode(params)}"
    try:
        r = requests.get(url, headers=UA, timeout=60)
        j = r.json()
    except Exception as exc:
        print(f"[{label}] HTTP FAILED: {type(exc).__name__}: {exc}")
        return {}
    err = j.get("error")
    docs = j.get("response", {}).get("docs", [])
    num = j.get("response", {}).get("numFound")
    print(f"[{label}] http={r.status_code} numFound={num} docs_returned={len(docs)} error={err}")
    for d in docs[:5]:
        print(f"    - {d.get('identifier')!r} year={d.get('year')!r} lang={d.get('language')!r} lic={str(d.get('licenseurl'))[:60]!r} title={str(d.get('title'))[:60]!r}")
    return j


def main() -> None:
    q_full = build_query()
    print(f"FULL QUERY:\n  {q_full}\n")
    adv_search(q_full, label="full query")
    print()

    # Which clause kills it?
    adv_search('mediatype:texts AND language:("Bengali" OR "Bangla" OR "ben" OR "beng") AND year:[1870 TO 1935] AND (title:(tantra OR tantrik OR sadhana OR mantra) OR subject:(tantra OR sadhana OR mantra))', label="english terms only")
    adv_search('mediatype:texts AND language:("Bengali") AND year:[1870 TO 1935] AND (title:tantra OR subject:tantra)', label="simple tantra")
    adv_search('mediatype:texts AND language:("Bengali" OR "Bangla") AND (title:tantra OR subject:tantra)', label="no year clause")
    adv_search('mediatype:texts AND language:"Bengali" AND year:[1870 TO 1935]', label="no terms (count only)")
    adv_search('mediatype:texts AND language:("Bengali" OR "Bangla" OR "ben" OR "beng")', label="language only (count)")
    adv_search('mediatype:texts AND year:[1870 TO 1935] AND (title:tantra OR subject:tantra)', label="no language filter")
    print()

    # Bengali unicode terms
    adv_search('mediatype:texts AND language:"Bengali" AND (title:"তন্ত্র" OR subject:"তন্ত্র")', label="unicode single term")
    adv_search('mediatype:texts AND (title:("তন্ত্র" OR "সাধনা") OR subject:("তন্ত্র" OR "সাধনা"))', label="unicode OR group")
    print()

    # What languages do Bengali-ish tantra items actually declare?
    j = adv_search('mediatype:texts AND (title:tantra OR subject:tantra) AND year:[1870 TO 1935]', rows=20, fields=["identifier", "language", "title", "licenseurl", "rights", "year"], label="lang survey")
    langs: dict[str, int] = {}
    lics: dict[str, int] = {}
    for d in j.get("response", {}).get("docs", []):
        for lang in (d.get("language") if isinstance(d.get("language"), list) else [d.get("language")] if d.get("language") else []):
            langs[str(lang)] = langs.get(str(lang), 0) + 1
        lic = str(d.get("licenseurl") or d.get("rights") or "NONE")
        lics[lic[:70]] = lics.get(lic[:70], 0) + 1
    print(f"  languages seen: {json.dumps(langs, ensure_ascii=False)}")
    print(f"  licences seen: {json.dumps(lics)}")
    print()

    # internetarchive library path
    print("--- internetarchive library ---")
    try:
        import internetarchive as ia

        print("ia version:", ia.__version__)
        search = ia.search_items(q_full, fields=["identifier", "title", "year", "language", "licenseurl", "subject"], params={"rows": 10})
        n = 0
        for rec in search.iter_as_results():
            print(f"    ia -> {rec.get('identifier')!r} {str(rec.get('title'))[:50]!r}")
            n += 1
        print(f"    ia results: {n} (numFound={search.num_found})")
    except Exception as exc:
        print(f"    ia FAILED: {type(exc).__name__}: {exc}")

    # metadata + files + download for one candidate identifier
    print("\n--- metadata/files/download path ---")
    try:
        ident = "sadhanaammrit00ramauoft"  # a guess; replaced below by any found identifier
        j2 = adv_search('mediatype:texts AND language:"Bengali" AND (title:tantra OR subject:tantra)', rows=5, fields=["identifier"], label="pick id")
        docs = j2.get("response", {}).get("docs", [])
        if docs:
            ident = docs[0]["identifier"]
        print(f"  probing item: {ident}")
        m = requests.get(f"{BASE}/metadata/{ident}", headers=UA, timeout=60).json()
        md = m.get("metadata", {})
        print(f"  metadata keys sample: {sorted(k for k in md)[:20]}")
        print(f"  language={md.get('language')!r} year={md.get('year')!r} date={md.get('date')!r} licenseurl={md.get('licenseurl')!r} rights={md.get('rights')!r}")
        txt_files = [f for f in m.get("files", []) if f.get("name", "").endswith("_djvu.txt")]
        print(f"  djvu.txt files: {[f['name'] for f in txt_files]}")
        if txt_files:
            f0 = txt_files[0]
            print(f"  md5={f0.get('md5')!r} size={f0.get('size')!r}")
            dl = f"{BASE}/download/{ident}/{urllib.parse.quote(f0['name'])}"
            r = requests.get(dl, headers=UA, timeout=120, stream=True)
            data = b""
            for chunk in r.iter_content(1 << 16):
                data += chunk
                if len(data) > 300_000:
                    break
            print(f"  download http={r.status_code} bytes={len(data)} head={data[:120]!r}")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
