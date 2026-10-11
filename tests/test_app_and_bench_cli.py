"""``app.py`` and ``benchmark.py`` command lines, driven with a fake model so no GGUF file is needed."""

from __future__ import annotations

import pytest

from tantra import ui
from tantra.cli import bench as bench_cli
from tantra.cli import chat as chat_cli
from tests.fakes import EchoBackend


@pytest.fixture
def fake_model(tmp_path):
    path = tmp_path / "tantra-test-Q4_K_M.gguf"
    path.write_bytes(b"GGUF" + b"\0" * 64)
    return path


@pytest.fixture
def backends(monkeypatch):
    """Route LlamaCppBackend in both CLIs to EchoBackend and remember every instance created."""
    created: list[EchoBackend] = []

    def factory(model_path, **kwargs):
        backend = EchoBackend(model_path, **kwargs)
        created.append(backend)
        return backend

    monkeypatch.setattr(chat_cli, "LlamaCppBackend", factory)
    monkeypatch.setattr("tantra.bench.LlamaCppBackend", factory)
    return created


def _scripted_input(monkeypatch, lines: list[str]):
    """Feed the REPL a fixed sequence of lines, then end the input (Ctrl-D)."""
    queue = iter(lines)

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(queue)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(ui.console, "input", fake_input)


# ----------------------------------------------------------------- app.py
def test_prompt_mode_answers_once_and_prints_the_memory_footer(fake_model, backends, capsys):
    code = chat_cli.main(["--model", str(fake_model), "--threads", "2", "--prompt", "তন্ত্র কী?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "উত্তর" in out and "একটি" in out and "দুই" in out
    assert "tok/s" in out and "budget" in out
    assert len(backends) == 1 and backends[0].threads == 2
    assert backends[0].calls[-1][-1] == {"role": "user", "content": "তন্ত্র কী?"}


def test_interactive_session_keeps_context_until_reset(fake_model, backends, monkeypatch, capsys):
    _scripted_input(monkeypatch, ["নমস্কার", "/stats", "/reset", "/stats", "/help", "/exit"])
    code = chat_cli.main(["--model", str(fake_model), "--threads", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "turns kept" in out and "conversation cleared" in out and "/reset" in out
    # The first /stats runs after one answer (1 turn); the second runs after /reset (0 turns).
    stats_rows = [line for line in out.splitlines() if "turns kept" in line]
    assert len(stats_rows) == 2
    turns = [int(row.split("turns kept")[1].split()[0]) for row in stats_rows]
    assert turns == [1, 0]


def test_end_of_input_exits_cleanly_without_answering(fake_model, backends, monkeypatch):
    _scripted_input(monkeypatch, [])
    assert chat_cli.main(["--model", str(fake_model), "--threads", "2"]) == 0
    assert backends[0].calls == []


def test_system_prompt_override_is_sent_to_the_model(fake_model, backends, capsys):
    chat_cli.main(["--model", str(fake_model), "--system", "সংক্ষেপে উত্তর দাও।", "--prompt", "হ্যালো"])
    first_message = backends[0].calls[-1][0]
    assert first_message == {"role": "system", "content": "সংক্ষেপে উত্তর দাও।"}


def test_memory_over_budget_is_warned_about(fake_model, backends, capsys):
    # The test process itself is far above 1 MiB, so a 1 MiB budget must trigger the warning.
    chat_cli.main(["--model", str(fake_model), "--threads", "2", "--budget-mb", "1", "--prompt", "হ্যালো"])
    assert "above the memory budget" in capsys.readouterr().out


# ----------------------------------------------------------------- benchmark.py --enforce
def test_bench_without_enforce_reports_but_exits_zero(fake_model, backends, tmp_path, capsys):
    reports = tmp_path / "reports"
    code = bench_cli.main(["--model", str(fake_model), "--threads", "2", "--budget-mb", "1", "--reports", str(reports)])
    out = capsys.readouterr().out
    assert code == 0
    assert "exceeds" in out
    assert list(reports.glob("bench-*.json"))


def test_bench_enforce_exits_3_when_over_budget(fake_model, backends, tmp_path, capsys):
    code = bench_cli.main(
        ["--model", str(fake_model), "--threads", "2", "--budget-mb", "1", "--reports", str(tmp_path), "--enforce"]
    )
    assert code == 3
    assert "exceeds" in capsys.readouterr().out


def test_bench_enforce_exits_0_within_budget(fake_model, backends, tmp_path, capsys):
    code = bench_cli.main(
        ["--model", str(fake_model), "--threads", "2", "--budget-mb", "100000", "--reports", str(tmp_path), "--enforce"]
    )
    assert code == 0
    assert "within the" in capsys.readouterr().out


def test_bench_missing_model_is_a_clean_failure(tmp_path, capsys):
    code = bench_cli.main(["--model", str(tmp_path / "absent.gguf"), "--reports", str(tmp_path)])
    assert code == 1
    assert "no GGUF model found" in capsys.readouterr().out


# ----------------------------------------------------------------- first-run guidance
def test_chat_without_any_model_points_to_the_fetch_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # a fresh checkout has no models/gguf
    code = chat_cli.main([])
    out = capsys.readouterr().out
    assert code == 1
    assert "fetch_model.py" in out and "setup.sh" in out
    assert "no .gguf model found" in out


def test_bench_without_any_model_points_to_the_fetch_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = bench_cli.main(["--reports", str(tmp_path)])
    assert code == 1
    assert "fetch_model.py" in capsys.readouterr().out


# ----------------------------------------------------------------- gpu layers
def test_gpu_layers_flag_reaches_the_backend(fake_model, backends, capsys):
    chat_cli.main(["--model", str(fake_model), "--gpu-layers", "7", "--prompt", "হ্যালো"])
    assert backends[-1].gpu_layers == 7
    assert "GPU (7 layers)" in capsys.readouterr().out


def test_gpu_layers_auto_default_and_cpu_label(fake_model, backends, capsys):
    chat_cli.main(["--model", str(fake_model), "--gpu-layers", "0", "--prompt", "হ্যালো"])
    assert backends[-1].gpu_layers == 0
    assert "CPU" in capsys.readouterr().out


def test_bench_gpu_layers_flag_reaches_the_backend(fake_model, backends, tmp_path):
    code = bench_cli.main(
        ["--model", str(fake_model), "--gpu-layers", "3", "--budget-mb", "100000", "--reports", str(tmp_path)]
    )
    assert code == 0
    assert backends[-1].gpu_layers == 3
