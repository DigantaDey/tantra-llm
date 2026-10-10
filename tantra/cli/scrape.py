"""``scraper.py`` command line: search archive.org and download Bengali Tantra texts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich import box
from rich.table import Table

from tantra import ui
from tantra.config import RAW_DIR
from tantra.scraper import (
    DEFAULT_FORMATS,
    DEFAULT_YEAR_RANGE,
    Candidate,
    Decision,
    InternetArchiveClient,
    build_query,
    download_items,
    evaluate,
    parse_candidate,
    search_candidates,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scraper.py",
        description="Find and download public-domain Bengali Tantra / Sadhana texts from archive.org.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--limit", type=int, default=25, help="maximum search hits to inspect (default: 25)")
        p.add_argument(
            "--year-range",
            nargs=2,
            type=int,
            metavar=("FROM", "TO"),
            default=list(DEFAULT_YEAR_RANGE),
            help="publication years to accept (default: 1870 1935)",
        )
        p.add_argument("--allow-undated", action="store_true", help="keep items without a year")
        p.add_argument(
            "--include-unmarked",
            action="store_true",
            help="also consider items not explicitly marked public domain (review rights yourself!)",
        )
        p.add_argument("--query", help="override the archive.org Lucene query")

    search = sub.add_parser("search", help="list candidates and why each is accepted or rejected")
    common(search)
    search.add_argument("--show-rejected", action="store_true", help="also list rejected hits")

    download = sub.add_parser("download", help="download accepted items")
    common(download)
    download.add_argument("--identifier", action="append", default=[], help="fetch specific item(s) only")
    download.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help=f"comma-separated subset of {', '.join(DEFAULT_FORMATS)} (default: all)",
    )
    download.add_argument("--out", type=Path, default=RAW_DIR, help=f"output directory (default: {RAW_DIR})")
    download.add_argument("--delay", type=float, default=1.0, help="seconds between items (default: 1.0)")
    download.add_argument("--retries", type=int, default=3, help="retries per file (default: 3)")
    download.add_argument("--dry-run", action="store_true", help="show what would be fetched and exit")
    return parser


def _render_table(rows: list[tuple[Candidate, Decision]]) -> Table:
    table = Table(box=box.SIMPLE_HEAD, header_style=f"bold {ui.ACCENT}", expand=True)
    table.add_column("#", justify="right", style=ui.MUTED)
    table.add_column("Identifier", style="white", no_wrap=True)
    table.add_column("Year", justify="right")
    table.add_column("Title", ratio=1, overflow="fold")
    table.add_column("Score", justify="right")
    table.add_column("Verdict", overflow="fold")
    for i, (cand, dec) in enumerate(rows, start=1):
        verdict = f"[{ui.OK}]accepted[/]" if dec.accepted else f"[{ui.MUTED}]{'; '.join(dec.reasons)}[/]"
        table.add_row(
            str(i),
            cand.identifier,
            str(cand.year or "—"),
            cand.title[:90],
            str(dec.score),
            verdict,
        )
    return table


def _year_range(values: list[int]) -> tuple[int, int]:
    lo, hi = values
    if lo > hi:
        raise SystemExit(f"--year-range: FROM ({lo}) must not exceed TO ({hi})")
    return lo, hi


def _cmd_search(args: argparse.Namespace, client: InternetArchiveClient) -> int:
    year_range = _year_range(args.year_range)
    ui.banner("Search archive.org for Bengali Tantra & Sadhana texts")
    query = args.query or build_query(year_range)
    ui.step("Query")
    ui.info(query)
    with ui.console.status("[bold]Searching archive.org…[/]", spinner="dots"):
        rows = search_candidates(
            client,
            limit=args.limit,
            year_range=year_range,
            require_public_domain=not args.include_unmarked,
            allow_undated=args.allow_undated,
            query=query,
        )
    shown = rows if args.show_rejected else [r for r in rows if r[1].accepted]
    ui.step(f"{len(rows)} hits inspected · {sum(1 for _, d in rows if d.accepted)} accepted")
    if shown:
        ui.console.print(_render_table(shown))
    else:
        ui.warn("nothing accepted. Try --include-unmarked, --allow-undated or a wider --year-range.")
    return 0


def _cmd_download(args: argparse.Namespace, client: InternetArchiveClient) -> int:
    year_range = _year_range(args.year_range)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    ui.banner("Download verified Bengali Tantra texts")
    if args.identifier:
        candidates = []
        for ident in args.identifier:
            cand = parse_candidate({**client.metadata(ident), "identifier": ident})
            candidates.append(
                (
                    cand,
                    evaluate(
                        cand,
                        year_range=year_range,
                        require_public_domain=not args.include_unmarked,
                        allow_undated=args.allow_undated,
                    ),
                )
            )
    else:
        with ui.console.status("[bold]Searching archive.org…[/]", spinner="dots"):
            candidates = search_candidates(
                client,
                limit=args.limit,
                year_range=year_range,
                require_public_domain=not args.include_unmarked,
                allow_undated=args.allow_undated,
                query=args.query,
            )
    accepted = [(c, d) for c, d in candidates if d.accepted]
    ui.step(f"{len(accepted)} accepted item(s) → {args.out}")
    if not accepted:
        ui.warn("nothing to download")
        return 1
    results = download_items(
        client,
        accepted,
        out_dir=args.out,
        formats=formats,
        delay=args.delay,
        retries=args.retries,
        dry_run=args.dry_run,
        log=ui.console.print,
    )
    done = sum(r.status == "downloaded" for r in results)
    failed = sum(r.status == "failed" for r in results)
    ui.step("Summary")
    ui.console.print(ui.kv_table([("downloaded", str(done)), ("failed", str(failed)), ("output", str(args.out))]))
    return 0 if failed == 0 else 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    client = InternetArchiveClient()
    try:
        if args.command == "search":
            return _cmd_search(args, client)
        return _cmd_download(args, client)
    except KeyboardInterrupt:
        ui.warn("interrupted — completed items are kept; re-run to resume")
        return 130
    except ValueError as exc:
        ui.fail(str(exc))
        return 2
    except Exception as exc:  # network failure etc.: friendly, not a traceback
        ui.fail(f"{type(exc).__name__}: {exc}")
        ui.info("check your connection to archive.org and try again")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
