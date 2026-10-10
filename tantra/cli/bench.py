"""``benchmark.py`` command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich import box
from rich.table import Table

from tantra import ui
from tantra.bench import run_benchmark, write_report
from tantra.cli.chat import find_default_model, pick_threads
from tantra.config import MEMORY_BUDGET_MB, REPORTS_DIR
from tantra.memory import physical_cores


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="benchmark.py", description="Measure speed and RAM of a GGUF model on this CPU.")
    p.add_argument("--model", type=Path, help="GGUF file (default: newest Q4_K_M in models/gguf/)")
    p.add_argument(
        "--threads", type=int, choices=[1, 2, 3, 4, 6, 8], help="CPU threads (default: 4 or 2 by host cores)"
    )
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--budget-mb", type=float, default=MEMORY_BUDGET_MB)
    p.add_argument("--reports", type=Path, default=REPORTS_DIR, help="where to write the JSON report")
    p.add_argument("--enforce", action="store_true", help="exit non-zero when the RAM budget is exceeded")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ui.banner("Benchmark · speed and RAM on this CPU")
    model = args.model or find_default_model()
    if model is None or not model.is_file():
        ui.fail("no GGUF model found; pass --model or run quantize.sh first")
        return 1
    threads = args.threads or pick_threads(physical_cores())
    ui.step(f"Running {model.name} with {threads} threads")
    try:
        with ui.console.status("[bold]benchmarking (load + two prompts)…[/]", spinner="dots"):
            result = run_benchmark(
                model, threads=threads, context=args.ctx, max_tokens=args.max_tokens, budget_mb=args.budget_mb
            )
    except (FileNotFoundError, ImportError) as exc:
        ui.fail(str(exc))
        return 1

    table = Table(box=box.SIMPLE_HEAD, header_style=f"bold {ui.ACCENT}", show_header=False, expand=False)
    table.add_column(style=ui.MUTED)
    table.add_column(justify="right")
    table.add_row("model size", f"{result.model_size_mb:,.1f} MB")
    table.add_row("load time", f"{result.load_seconds:.2f} s")
    table.add_row("RAM before load", f"{result.rss_idle_mb / 1024:.2f} GB")
    table.add_row("RAM after load", f"{result.rss_after_load_mb / 1024:.2f} GB")
    table.add_row(
        "RAM peak",
        f"[{ui.memory_style(result.rss_peak_mb, args.budget_mb)}]{result.rss_peak_mb / 1024:.2f} GB[/]",
    )
    table.add_row("generation speed", f"{result.tokens_per_second:.2f} tok/s")
    table.add_row("tokens generated", f"{result.generated_tokens:,}")
    ui.console.print(table)
    report = write_report(result, args.reports)
    if result.passed:
        ui.ok(f"within the {args.budget_mb / 1024:.1f} GB RAM budget")
    else:
        ui.fail(f"peak RAM {result.rss_peak_mb / 1024:.2f} GB exceeds the {args.budget_mb / 1024:.1f} GB budget")
    ui.info(f"report: {report}")
    return 0 if (result.passed or not args.enforce) else 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
