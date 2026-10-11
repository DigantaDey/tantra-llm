"""Runtime choices: GPU-layer auto-detection (Metal on Apple Silicon) and install guidance."""

from __future__ import annotations

import sys
import types

import pytest

from tantra.llm import (
    LlamaCppBackend,
    RuntimeMissingError,
    device_label,
    gpu_offload_supported,
    resolve_gpu_layers,
)


def _fake_llama_cpp(monkeypatch, *, gpu: bool):
    """Install a stand-in ``llama_cpp`` module (None in sys.modules fakes a missing install)."""
    module = types.ModuleType("llama_cpp")
    captured: dict = {}

    class Llama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def tokenize(self, text, add_bos=False, special=False):
            return []

    module.Llama = Llama  # type: ignore[attr-defined]
    module.llama_supports_gpu_offload = lambda: gpu  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    return captured


def test_auto_gpu_layers_offload_all_when_supported(monkeypatch):
    _fake_llama_cpp(monkeypatch, gpu=True)
    assert gpu_offload_supported() is True
    assert resolve_gpu_layers(None) == -1
    assert resolve_gpu_layers(0) == 0  # explicit CPU still wins
    assert resolve_gpu_layers(12) == 12


def test_auto_gpu_layers_stay_on_cpu_without_offload(monkeypatch):
    _fake_llama_cpp(monkeypatch, gpu=False)
    assert gpu_offload_supported() is False
    assert resolve_gpu_layers(None) == 0
    assert resolve_gpu_layers(-1) == -1


def test_missing_runtime_counts_as_no_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # import raises ImportError
    assert gpu_offload_supported() is False


def test_backend_passes_resolved_layers_to_llama(tmp_path, monkeypatch):
    captured = _fake_llama_cpp(monkeypatch, gpu=True)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    backend = LlamaCppBackend(model, n_ctx=128, threads=2)
    assert backend.gpu_layers == -1
    assert captured["n_gpu_layers"] == -1 and captured["n_ctx"] == 128
    backend = LlamaCppBackend(model, n_gpu_layers=0)
    assert backend.gpu_layers == 0
    assert captured["n_gpu_layers"] == 0


def test_missing_runtime_error_explains_the_install(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    with pytest.raises(RuntimeMissingError) as info:
        LlamaCppBackend(model)
    message = str(info.value)
    assert "pip install -e '.[llm]'" in message
    assert "xcode-select --install" in message  # the macOS prerequisite, called out up front


def test_device_labels_are_human_readable():
    assert device_label(0) == "CPU"
    assert "all layers" in device_label(-1)
    assert device_label(12) == "GPU (12 layers)"
