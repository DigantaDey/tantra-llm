#!/usr/bin/env python3
"""Measure local GGUF load/inference latency and sampled process RSS with psutil."""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from app import (
    DEFAULT_SYSTEM_PROMPT,
    load_llama_model,
    recommended_threads,
    validate_gguf,
)


class MemoryMonitor:
    def __init__(self, *, interval: float = 0.25, quiet: bool = False):
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("benchmark.py requires psutil; install `pip install psutil`.") from exc
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.quiet = quiet
        self.samples_mb: list[float] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def sample(self) -> float:
        value = self.process.memory_info().rss / (1024 * 1024)
        self.samples_mb.append(value)
        return value

    def _run(self) -> None:
        while not self.stop_event.is_set():
            value = self.sample()
            if not self.quiet:
                print(f"[rss] {value:,.1f} MiB", file=sys.stderr, flush=True)
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self._run, name="rss-monitor", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=max(1.0, self.interval * 4))
        self.sample()

    @property
    def peak_mb(self) -> float:
        return max(self.samples_mb, default=0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark CPU-only inference for a GGUF model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True, help="A representative Bengali or English test prompt.")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--threads", type=int, choices=(2, 4), help="Default: 4 with >=4 logical CPUs, otherwise 2.")
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=128, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--chat-format", help="Optional llama-cpp-python chat format override.")
    parser.add_argument("--sample-interval", type=float, default=0.25, help="RSS sampling interval in seconds.")
    parser.add_argument("--ram-limit-gb", type=float, default=2.5, help="Fail if sampled process RSS exceeds this budget; 0 disables the limit.")
    parser.add_argument("--quiet-memory", action="store_true", help="Suppress per-sample RSS log lines (final peak is still reported).")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.tokens < 1 or args.ctx_size < 256 or args.batch_size < 1:
        print("--tokens and --batch-size must be positive; --ctx-size must be >=256.", file=sys.stderr)
        return 2
    if args.sample_interval <= 0 or args.ram_limit_gb < 0:
        print("--sample-interval must be positive and --ram-limit-gb cannot be negative.", file=sys.stderr)
        return 2
    threads = args.threads or recommended_threads()
    try:
        validate_gguf(args.model)
        monitor = MemoryMonitor(interval=args.sample_interval, quiet=args.quiet_memory)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Benchmark setup failed: {exc}", file=sys.stderr)
        return 2

    model: Any = None
    load_seconds = 0.0
    generation_seconds = 0.0
    response: dict[str, Any] = {}
    print(
        f"Benchmarking {args.model} | CPU threads={threads} | ctx={args.ctx_size} "
        f"| batch={args.batch_size} | token cap={args.tokens}"
    )
    try:
        with monitor:
            load_started = time.perf_counter()
            model = load_llama_model(
                args.model,
                threads=threads,
                context_size=args.ctx_size,
                batch_size=args.batch_size,
                chat_format=args.chat_format,
            )
            load_seconds = time.perf_counter() - load_started
            print(f"Model load: {load_seconds:.2f}s; RSS after load: {monitor.sample():,.1f} MiB")
            generation_started = time.perf_counter()
            response = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": args.system},
                    {"role": "user", "content": args.prompt},
                ],
                max_tokens=args.tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=False,
            )
            generation_seconds = time.perf_counter() - generation_started
    except Exception as exc:  # noqa: BLE001 - report backend errors from several llama.cpp versions.
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        print("Confirm llama-cpp-python is installed and the GGUF provides a compatible chat template (or pass --chat-format).", file=sys.stderr)
        return 1

    choices = response.get("choices", []) if isinstance(response, dict) else []
    answer = ""
    if choices:
        answer = str((choices[0].get("message") or {}).get("content") or "")
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    if not completion_tokens:
        # The library normally reports usage; this fallback remains explicit and
        # approximate because Unicode character count is not a tokenizer count.
        completion_tokens = max(1, len(answer) // 4)
    throughput = completion_tokens / max(generation_seconds, 1e-9)
    peak = monitor.peak_mb

    print("\n--- model response ---")
    print(answer)
    print("\n--- benchmark summary ---")
    print(f"Prompt tokens: {prompt_tokens if prompt_tokens else 'not reported'}")
    print(f"Completion tokens: {completion_tokens}{' (estimated)' if not usage.get('completion_tokens') else ''}")
    print(f"Generation time: {generation_seconds:.2f}s ({throughput:.2f} tokens/s)")
    print(f"Peak sampled RSS: {peak:,.1f} MiB")
    print(f"Configured CPU threads: {threads}")
    if args.ram_limit_gb and peak > args.ram_limit_gb * 1024:
        print(f"[fail] sampled RSS exceeded the {args.ram_limit_gb:.2f} GiB budget.", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
