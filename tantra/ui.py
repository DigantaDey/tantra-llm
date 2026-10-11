"""Terminal presentation: one place for colours, banners and tables so every CLI looks the same."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tantra import __version__

ACCENT = "#e0a45c"  # saffron-gold
MUTED = "grey62"
OK = "#7bd88f"
WARN = "#f2c94c"
BAD = "#ff6b6b"

console = Console(highlight=False)


def banner(subtitle: str) -> None:
    title = Text.assemble(("तन्त्र", f"bold {ACCENT}"), ("  ·  ", MUTED), ("Tantra LLM", "bold white"))
    body = Text.assemble((subtitle, MUTED), ("\n", ""), (f"v{__version__}", f"dim {MUTED}"))
    console.print(Panel(body, title=title, title_align="left", border_style=ACCENT, box=box.ROUNDED, padding=(0, 2)))


def step(title: str) -> None:
    console.print(f"\n[bold {ACCENT}]▌[/] [bold]{title}[/]")


def ok(message: str) -> None:
    console.print(f"  [{OK}]✓[/] {message}")


def warn(message: str) -> None:
    console.print(f"  [{WARN}]![/] {message}")


def fail(message: str) -> None:
    console.print(f"  [{BAD}]✗[/] {message}")


def info(message: str) -> None:
    console.print(f"  [{MUTED}]{message}[/]")


def kv_table(rows: list[tuple[str, str]]) -> Table:
    table = Table(box=box.SIMPLE_HEAD, show_header=False, pad_edge=False, expand=False)
    table.add_column(style=MUTED, no_wrap=True)
    table.add_column(style="white")
    for key, value in rows:
        table.add_row(key, value)
    return table


def memory_style(mb: float, budget_mb: float) -> str:
    if mb <= 0.6 * budget_mb:
        return OK
    if mb <= budget_mb:
        return WARN
    return BAD
