"""Benchmark a GGUF model under the 4 GB machine constraints: speed, RAM, and a pass/fail verdict."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tantra.llm import GenerationParams, LlamaCppBackend
from tantra.memory import PeakMonitor, available_mb, physical_cores, rss_mb, total_mb

DEFAULT_PROMPTS = (
    "তন্ত্র সাধনায় বীজমন্ত্রের ভূমিকা কী? সংক্ষেপে বুঝিয়ে বলো।",
    "Explain what nyāsa means in Tantric ritual, in two sentences.",
)


@dataclass(frozen=True)
class BenchResult:
    model: str
    model_size_mb: float
    threads: int
    context: int
    max_tokens: int
    load_seconds: float
    rss_idle_mb: float
    rss_after_load_mb: float
    rss_peak_mb: float
    budget_mb: float
    prompt_tokens: int
    generated_tokens: int
    prompt_eval_seconds: float
    generation_seconds: float
    tokens_per_second: float
    passed: bool
    host: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def verdict(peak_mb: float, budget_mb: float) -> bool:
    return peak_mb <= budget_mb


def host_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "physical_cores": physical_cores(),
        "ram_total_mb": round(total_mb()),
        "ram_available_mb": round(available_mb()),
    }


def run_benchmark(
    model_path: Path,
    *,
    threads: int,
    context: int = 2048,
    max_tokens: int = 128,
    budget_mb: float,
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    backend_factory: Any = None,
) -> BenchResult:
    if backend_factory is None:
        backend_factory = LlamaCppBackend

    idle = rss_mb()
    with PeakMonitor() as mon:
        t0 = time.perf_counter()
        backend = backend_factory(model_path, n_ctx=context, threads=threads)
        load_seconds = time.perf_counter() - t0
        after_load = rss_mb()

        prompt_tokens = 0
        generated = 0
        prompt_eval = 0.0
        gen_seconds = 0.0
        params = GenerationParams(max_tokens=max_tokens, temperature=0.0, repeat_penalty=1.1, seed=0)
        for prompt in prompts:
            prompt_tokens += backend.count_tokens(prompt)
            # Time-to-first-token approximates prompt evaluation; the rest is decoding.
            start = time.perf_counter()
            first: float | None = None
            count = 0
            for _piece in backend.stream_chat([{"role": "user", "content": prompt}], params):
                if first is None:
                    first = time.perf_counter()
                count += 1
            end = time.perf_counter()
            if first is None:
                first = end
            prompt_eval += first - start
            gen_seconds += end - first
            generated += count
        del backend

    peak = mon.peak_mb
    tps = generated / gen_seconds if gen_seconds > 0 else 0.0
    return BenchResult(
        model=model_path.name,
        model_size_mb=round(model_path.stat().st_size / (1024 * 1024), 1),
        threads=threads,
        context=context,
        max_tokens=max_tokens,
        load_seconds=round(load_seconds, 2),
        rss_idle_mb=round(idle, 1),
        rss_after_load_mb=round(after_load, 1),
        rss_peak_mb=round(peak, 1),
        budget_mb=budget_mb,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated,
        prompt_eval_seconds=round(prompt_eval, 3),
        generation_seconds=round(gen_seconds, 3),
        tokens_per_second=round(tps, 2),
        passed=verdict(peak, budget_mb),
        host=host_info(),
    )


def write_report(result: BenchResult, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = reports_dir / f"bench-{Path(result.model).stem}-t{result.threads}-{stamp}.json"
    path.write_text(json.dumps(result.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
