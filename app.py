#!/usr/bin/env python3
"""A small, offline CLI for a local Tantra/Bengali GGUF model."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "You are an evidence-conscious research assistant specializing in historical "
    "Bengali Tantra sources. Distinguish source text from inference; never invent "
    "citations, line numbers, or manuscript readings. Flag OCR uncertainty and "
    "preserve ritual terms unless evidence supports normalization. Answer in the "
    "language of the user's question."
)
DEFAULT_RAM_LIMIT_GB = 2.5


def recommended_threads(cpu_count: int | None = None) -> int:
    logical_cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 2)
    return 4 if logical_cpus >= 4 else 2


def validate_gguf(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"GGUF model not found: {path}")
    with path.open("rb") as model_file:
        magic = model_file.read(4)
    if magic != b"GGUF":
        raise ValueError(f"{path} does not have a GGUF header (expected bytes GGUF).")


def memory_rss_mb() -> float | None:
    """Return process RSS in MiB via psutil, with a Linux /proc fallback."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except (OSError, psutil.Error):
            return None

    status_path = Path("/proc/self/status")
    try:
        if status_path.is_file():
            for line in status_path.read_text(encoding="ascii", errors="ignore").splitlines():
                if line.startswith("VmRSS:"):
                    try:
                        return float(line.split()[1]) / 1024
                    except (IndexError, ValueError):
                        return None
    except OSError:
        return None
    return None


def load_llama_model(
    model_path: Path,
    *,
    threads: int,
    context_size: int,
    batch_size: int,
    chat_format: str | None = None,
):
    validate_gguf(model_path)
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "Runtime dependency missing. Install `pip install -e '.[runtime]'` "
            "or `pip install llama-cpp-python psutil`."
        ) from exc

    options: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": context_size,
        "n_batch": batch_size,
        "n_threads": threads,
        "n_threads_batch": threads,
        "n_gpu_layers": 0,
        "use_mmap": True,
        "use_mlock": False,
        "verbose": False,
    }
    if chat_format:
        options["chat_format"] = chat_format
    return Llama(**options)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a local, CPU-only Bengali Tantra GGUF model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a validated .gguf model.")
    parser.add_argument("--prompt", help="Run one prompt and exit; omit for interactive chat.")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="System instruction.")
    parser.add_argument("--threads", type=int, choices=(2, 4), help="CPU threads (default: 4 when host has >=4 logical CPUs, otherwise 2).")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context window; 2048 is a memory-conscious default.")
    parser.add_argument("--batch-size", type=int, default=128, help="Prompt-processing batch size.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated tokens per turn.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--history-turns", type=int, default=4, help="Number of previous user/assistant turns retained.")
    parser.add_argument("--chat-format", help="Optional llama-cpp-python chat format override.")
    parser.add_argument("--ram-limit-gb", type=float, default=DEFAULT_RAM_LIMIT_GB, help="Warn if observed process RSS exceeds this value; use benchmark.py for peak sampling.")
    return parser


def _stream_response(model: Any, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
    stream = model.create_chat_completion(
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=True,
    )
    collected: list[str] = []
    sampled = 0
    for chunk in stream:
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {}) or {}
        content = delta.get("content")
        if content:
            print(content, end="", flush=True)
            collected.append(content)
            sampled += 1
            if sampled % 24 == 0:
                _report_memory(args.ram_limit_gb, prefix="[memory]")
    print()
    return "".join(collected)


def _report_memory(limit_gb: float, *, prefix: str = "[memory]") -> None:
    rss = memory_rss_mb()
    if rss is None:
        return
    print(f"{prefix} process RSS {rss:.1f} MiB", file=sys.stderr)
    if limit_gb > 0 and rss > limit_gb * 1024:
        print(
            f"[warning] RSS is above the configured {limit_gb:.2f} GiB budget; "
            "try a smaller model, context, or batch size.",
            file=sys.stderr,
        )


def _messages_for_turn(
    system: str,
    history: list[dict[str, str]],
    prompt: str,
    history_turns: int,
) -> list[dict[str, str]]:
    retained = history[-max(0, history_turns) * 2 :] if history_turns else []
    return [{"role": "system", "content": system}, *retained, {"role": "user", "content": prompt}]


def run(args: argparse.Namespace) -> int:
    if args.ctx_size < 256 or args.batch_size < 1 or args.max_tokens < 1:
        print("--ctx-size must be >=256; --batch-size and --max-tokens must be positive.", file=sys.stderr)
        return 2
    if args.history_turns < 0 or args.ram_limit_gb < 0:
        print("--history-turns and --ram-limit-gb cannot be negative.", file=sys.stderr)
        return 2
    threads = args.threads or recommended_threads()
    before = memory_rss_mb()
    print(f"Loading {args.model} (CPU threads={threads}, ctx={args.ctx_size}, batch={args.batch_size})…")
    started = time.perf_counter()
    try:
        model = load_llama_model(
            args.model,
            threads=threads,
            context_size=args.ctx_size,
            batch_size=args.batch_size,
            chat_format=args.chat_format,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Unable to load model: {exc}", file=sys.stderr)
        return 2
    elapsed = time.perf_counter() - started
    after = memory_rss_mb()
    print(f"Model ready in {elapsed:.1f}s; process RSS: {after:.1f} MiB" if after is not None else f"Model ready in {elapsed:.1f}s.")
    if before is not None and after is not None and args.ram_limit_gb and after > args.ram_limit_gb * 1024:
        print(f"[warning] model-load RSS exceeds {args.ram_limit_gb:.2f} GiB.", file=sys.stderr)

    history: list[dict[str, str]] = []
    prompts = [args.prompt] if args.prompt is not None else None
    try:
        if prompts is not None:
            prompt = prompts[0]
            print("Assistant> ", end="", flush=True)
            messages = _messages_for_turn(args.system, history, prompt, args.history_turns)
            _stream_response(model, messages, args)
            _report_memory(args.ram_limit_gb)
            return 0

        print("Offline chat ready. Type /exit or press Ctrl-D to quit.\n")
        while True:
            try:
                prompt = input("You> ").strip()
            except EOFError:
                print()
                break
            if not prompt:
                continue
            if prompt.casefold() in ("/exit", "/quit", "exit", "quit"):
                break
            messages = _messages_for_turn(args.system, history, prompt, args.history_turns)
            print("Assistant> ", end="", flush=True)
            answer = _stream_response(model, messages, args)
            history.extend(
                (
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                )
            )
            history = history[-args.history_turns * 2 :] if args.history_turns else []
            _report_memory(args.ram_limit_gb)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:  # noqa: BLE001 - backend exceptions vary across llama-cpp-python builds.
        print(
            f"\nGeneration failed: {exc}\n"
            "Check that the GGUF contains a compatible chat template or pass --chat-format.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
