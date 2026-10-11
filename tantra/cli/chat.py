"""``app.py`` command line: chat with the local GGUF model, fully offline, under a memory budget."""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

from tantra import ui
from tantra.config import GGUF_DIR, MEMORY_BUDGET_MB
from tantra.llm import (
    DEFAULT_SYSTEM_PROMPT,
    ChatSession,
    GenerationParams,
    LlamaCppBackend,
    ModelNotFoundError,
    RuntimeMissingError,
)
from tantra.memory import PeakMonitor, physical_cores, rss_mb

HELP = """\
[bold]Commands[/]
  [cyan]/help[/]      show this help
  [cyan]/reset[/]     forget the conversation
  [cyan]/stats[/]     memory and generation statistics
  [cyan]/exit[/]      leave (Ctrl-D also works)
"""


def pick_threads(cores: int) -> int:
    """Use 4 threads on hosts with 4+ physical cores, otherwise 2 (the two settings benchmarked)."""
    return 4 if cores >= 4 else 2


def find_default_model(directory: Path = GGUF_DIR) -> Path | None:
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("*.gguf"))
    preferred = [p for p in candidates if "Q4_K_M" in p.name.upper()]
    return (preferred or candidates or [None])[0]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="app.py", description="Chat with the local Tantra LLM (CPU, GGUF).")
    p.add_argument("--model", type=Path, help="path to a .gguf file (default: newest Q4_K_M in models/gguf/)")
    p.add_argument("--threads", type=int, help="CPU threads (default: 4 on 4+ core hosts, else 2)")
    p.add_argument("--ctx", type=int, default=2048, help="context window in tokens (default: 2048)")
    p.add_argument("--max-tokens", type=int, default=384, help="maximum answer length (default: 384)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repeat-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--system", help="override the system prompt")
    p.add_argument("--prompt", help="answer one question and exit (non-interactive)")
    p.add_argument(
        "--budget-mb",
        type=float,
        default=MEMORY_BUDGET_MB,
        help=f"warn above this RSS in MiB (default: {MEMORY_BUDGET_MB})",
    )
    return p


def _footer(tokens: int, seconds: float, peak_mb: float, budget: float) -> str:
    rate = tokens / seconds if seconds > 0 else 0.0
    style = ui.memory_style(peak_mb, budget)
    return (
        f"[{ui.MUTED}]{tokens} tokens · {rate:.1f} tok/s · "
        f"RAM [{style}]{peak_mb / 1024:.2f} GB[/] of {budget / 1024:.1f} GB budget[/]"
    )


def _answer(session: ChatSession, text: str, params: GenerationParams, budget: float) -> None:
    ui.console.print()
    started = time.perf_counter()
    produced = 0
    with PeakMonitor() as mon:
        for piece in session.ask(text, params):
            produced += 1
            sys.stdout.write(piece)
            sys.stdout.flush()
    seconds = time.perf_counter() - started
    ui.console.print()
    ui.console.print(_footer(produced, seconds, mon.peak_mb, budget))


def _enable_line_editing() -> None:
    # Importing readline switches input() to line editing and history. Not available on Windows.
    with contextlib.suppress(ImportError):
        import readline  # noqa: F401


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ui.banner("Local Tantra LLM · runs offline on CPU")
    model_path = args.model or find_default_model()
    if model_path is None:
        ui.fail(f"no .gguf model found in {GGUF_DIR}")
        ui.info(
            "build one with:  ./quantize.sh --hf-dir <converted-model> --calibration data/calibration/calibration.txt"
        )
        return 1

    threads = args.threads or pick_threads(physical_cores())
    params = GenerationParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        seed=args.seed,
    )
    ui.step("Loading model")
    started = time.perf_counter()
    try:
        with ui.console.status("[bold]loading weights…[/]", spinner="dots"):
            backend = LlamaCppBackend(model_path, n_ctx=args.ctx, threads=threads, seed=args.seed)
    except (ModelNotFoundError, RuntimeMissingError) as exc:
        ui.fail(str(exc))
        return 1
    load_seconds = time.perf_counter() - started
    loaded_mb = rss_mb()
    ui.ok(f"{model_path.name} · {threads} threads · ctx {args.ctx} · loaded in {load_seconds:.1f}s")
    style = ui.memory_style(loaded_mb, args.budget_mb)
    ui.info(
        f"resident memory after load: [{style}]{loaded_mb / 1024:.2f} GB[/] (budget {args.budget_mb / 1024:.1f} GB)"
    )
    if loaded_mb > args.budget_mb:
        ui.warn("above the memory budget: expect swapping on a 4 GB machine; try a smaller quantization")

    session = ChatSession(backend, system_prompt=args.system or DEFAULT_SYSTEM_PROMPT)

    if args.prompt:
        _answer(session, args.prompt, params, args.budget_mb)
        return 0

    _enable_line_editing()
    ui.console.print(f"\n[{ui.MUTED}]Type a question. /help for commands.[/]")
    while True:
        try:
            text = ui.console.input(f"\n[bold {ui.ACCENT}]❯[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            ui.console.print()
            break
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            break
        if text == "/help":
            ui.console.print(HELP)
            continue
        if text == "/reset":
            session.reset()
            ui.ok("conversation cleared")
            continue
        if text == "/stats":
            ui.console.print(
                ui.kv_table(
                    [
                        ("resident memory", f"{rss_mb() / 1024:.2f} GB"),
                        ("budget", f"{args.budget_mb / 1024:.1f} GB"),
                        ("turns kept", str(len(session.history) // 2)),
                        ("tokens generated", f"{session.tokens_generated:,}"),
                    ]
                )
            )
            continue
        try:
            _answer(session, text, params, args.budget_mb)
        except KeyboardInterrupt:
            ui.console.print(f"\n[{ui.WARN}]generation stopped[/]")
        except ValueError as exc:
            ui.fail(str(exc))
    ui.console.print(f"[{ui.MUTED}]শুভ সাধনা · goodbye[/]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
