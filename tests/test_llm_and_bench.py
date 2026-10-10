from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence

import pytest

from tantra.bench import run_benchmark, verdict, write_report
from tantra.config import MEMORY_BUDGET_MB
from tantra.llm import ChatSession, GenerationParams, LlamaCppBackend, ModelNotFoundError
from tantra.memory import PeakMonitor, rss_mb


class FakeBackend:
    context_size = 512

    def __init__(self, reply: str = "উত্তর একটি দুই তিন", pieces: int = 3) -> None:
        self.reply = reply
        self.pieces = pieces
        self.calls: list[list[dict[str, str]]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def stream_chat(self, messages: Sequence[dict[str, str]], params: GenerationParams) -> Iterator[str]:
        self.calls.append(list(messages))
        words = self.reply.split()
        for w in words[: self.pieces]:
            yield w + " "


PARAMS = GenerationParams(max_tokens=64)


def test_session_streams_and_remembers_turns():
    session = ChatSession(FakeBackend())
    assert "".join(session.ask("প্রশ্ন", PARAMS)).strip() == "উত্তর একটি দুই"
    assert [t.role for t in session.history] == ["user", "assistant"]
    assert session.history[0].content == "প্রশ্ন"


def test_interrupted_answer_is_not_recorded():
    session = ChatSession(FakeBackend())
    gen = session.ask("প্রশ্ন", PARAMS)
    next(gen)
    gen.close()  # user pressed Ctrl-C mid-answer
    assert session.history == []


def test_history_is_sent_to_the_model_and_trimmed_to_fit():
    backend = FakeBackend()
    session = ChatSession(backend, max_turns=3)
    for i in range(6):
        list(session.ask(f"প্রশ্ন {i}", PARAMS))
    assert len(session.history) == 6  # 3 turns = 6 entries
    list(session.ask("শেষ", PARAMS))
    sent = backend.calls[-1]
    assert sent[0]["role"] == "system"
    assert sent[-1] == {"role": "user", "content": "শেষ"}


def test_old_turns_are_dropped_when_the_context_is_full():
    class Tiny(FakeBackend):
        context_size = 80

    backend = Tiny(reply=" ".join(["শব্দ"] * 20), pieces=20)
    session = ChatSession(backend, system_prompt="সিস্টেম", max_turns=8)
    list(session.ask("প্রথম প্রশ্ন", GenerationParams(max_tokens=20)))
    list(session.ask("দ্বিতীয় প্রশ্ন", GenerationParams(max_tokens=20)))
    list(session.ask("তৃতীয় প্রশ্ন", GenerationParams(max_tokens=20)))
    sent = backend.calls[-1]
    contents = [m["content"] for m in sent]
    assert "তৃতীয় প্রশ্ন" in contents
    assert "প্রথম প্রশ্ন" not in contents  # oldest turn evicted to fit the budget


def test_reset_clears_history():
    session = ChatSession(FakeBackend())
    list(session.ask("প্রশ্ন", PARAMS))
    session.reset()
    assert session.history == [] and session.tokens_generated == 0


def test_too_large_max_tokens_is_a_clear_error():
    with pytest.raises(ValueError):
        list(ChatSession(FakeBackend()).ask("প্রশ্ন", GenerationParams(max_tokens=600)))


def test_missing_model_file_is_reported(tmp_path):
    with pytest.raises(ModelNotFoundError):
        LlamaCppBackend(tmp_path / "absent.gguf")


def test_peak_monitor_records_samples():
    baseline = rss_mb()
    with PeakMonitor(interval=0.01) as mon:
        # Touch 64 MiB (bytes * n writes every page) and hold it long enough for several samples.
        block = bytearray(b"x") * (64 * 1024 * 1024)
        time.sleep(0.1)
        held_peak = mon.peak_mb
        del block
    assert len(mon.samples) >= 3
    assert all(s > 0 for s in mon.samples)
    assert mon.peak_mb == max(mon.samples)
    # The peak seen while the 64 MiB block was alive must exceed the baseline by most of that size.
    assert held_peak - baseline > 40
    assert mon.peak_mb >= held_peak


def test_verdict_respects_budget():
    assert verdict(1000, MEMORY_BUDGET_MB) and not verdict(3000, MEMORY_BUDGET_MB)


def test_benchmark_reports_speed_and_budget_verdict(tmp_path):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 1024)
    result = run_benchmark(
        model,
        threads=2,
        budget_mb=10_000,
        max_tokens=8,
        prompts=("এক দুই", "three four"),
        backend_factory=lambda path, n_ctx, threads: FakeBackend(reply="এক দুই তিন চার পাঁচ"),
    )
    assert result.generated_tokens == 2 * 3  # FakeBackend yields 3 pieces per prompt
    assert result.passed is True
    assert result.threads == 2 and result.model == "tiny.gguf"
    path = write_report(result, tmp_path / "reports")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["passed"] is True and "host" in data


def test_benchmark_fails_when_over_budget(tmp_path):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"GGUF")
    result = run_benchmark(
        model,
        threads=2,
        budget_mb=1,
        max_tokens=4,
        prompts=("x",),
        backend_factory=lambda p, n_ctx, threads: FakeBackend(),
    )
    assert result.passed is False
