"""``fetch_model.py`` command line: download a verified GGUF chat model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from tantra import ui
from tantra.config import GGUF_DIR
from tantra.fetch import DEFAULT_MODEL_KEY, MODELS, FetchError, ModelSpec, ensure_model


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fetch_model.py",
        description="Download a ready-made Qwen2.5 Instruct GGUF (verified by size and SHA-256).",
    )
    p.add_argument(
        "--size",
        choices=sorted(MODELS),
        default=DEFAULT_MODEL_KEY,
        help="which model to fetch (default: %(default)s)",
    )
    p.add_argument("--out-dir", type=Path, default=GGUF_DIR, help=f"destination directory (default: {GGUF_DIR})")
    p.add_argument("--force", action="store_true", help="download again even if a verified copy exists")
    p.add_argument("--list", action="store_true", help="list the available models and exit")
    return p


def _describe(spec: ModelSpec) -> str:
    return f"{spec.key:<9} {spec.label}"


def _fetch(spec: ModelSpec, out_dir: Path, force: bool) -> int:
    ui.step(f"Fetching {spec.label}")
    ui.info(f"source  {spec.url}")
    ui.info(f"target  {out_dir / spec.filename}")
    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        transient=True,
        console=ui.console,
    ) as progress:
        task = progress.add_task("download", total=spec.size_bytes)

        def on_progress(done: int, total: int | None) -> None:
            progress.update(task, total=total or spec.size_bytes, completed=done)

        try:
            path = ensure_model(spec, out_dir, force=force, on_progress=on_progress)
        except FetchError as exc:
            ui.fail(str(exc))
            return 1
    size_mb = path.stat().st_size / (1024 * 1024)
    ui.ok(f"{path.name} · {size_mb:,.1f} MiB · sha256 verified · GGUF")
    ui.info(f"saved to {path}")
    ui.console.print(f"\n  [bold]next[/]: python app.py        [{ui.MUTED}]# chat with the model[/]")
    ui.console.print(f"           python benchmark.py  [{ui.MUTED}]# measure speed and RAM[/]\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ui.banner("Model download · one verified GGUF into models/gguf")
    if args.list:
        ui.step("Available models")
        for spec in MODELS.values():
            ui.info(_describe(spec))
        return 0
    return _fetch(MODELS[args.size], args.out_dir, args.force)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
