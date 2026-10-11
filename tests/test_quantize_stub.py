"""Run ``quantize.sh`` end to end against stubbed llama.cpp tools.

This is the portability net for the shell script: it executes the real pipeline
(fetch, convert, i-matrix, quantize, validate, checksums) with stub binaries, so
the BSD/macOS userland and bash 3.2 differences (``stat -c`` vs ``stat -f``,
``sha256sum`` vs ``shasum``, empty arrays under ``set -u``) are exercised on every
platform the test suite runs on. No compilers, network or real model needed.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

QUANTIZE_TOOL = """#!/usr/bin/env bash
set -e
if [[ "$1" == "--imatrix" ]]; then shift 2; fi
in="$1"; out="$2"; typ="$3"
[[ -f "$in" ]] || { echo "missing input $in" >&2; exit 1; }
printf 'GGUF-stub-%s' "$typ" > "$out"
"""

IMATRIX_TOOL = """#!/usr/bin/env bash
set -e
out=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[[ -n "$out" ]] || { echo "no -o argument" >&2; exit 1; }
printf 'imatrix-stub' > "$out"
"""

FAKE_PYTHON = """#!/usr/bin/env bash
# Impersonates the python that runs convert_hf_to_gguf.py for the stub pipeline.
set -e
if [[ "$1" == "-c" ]]; then exit 0; fi   # the dependency check: import gguf, numpy, transformers
out=""
prev=""
for arg in "$@"; do
  if [[ "$prev" == "--outfile" ]]; then out="$arg"; fi
  prev="$arg"
done
[[ -n "$out" ]] || { echo "no --outfile argument" >&2; exit 1; }
printf 'GGUF-fake-f16' > "$out"
"""

FAKE_CMAKE = """#!/usr/bin/env bash
# Impersonates cmake: the build step drops the two stub tools into build/bin.
set -e
build=""
prev=""
for arg in "$@"; do
  if [[ "$prev" == "-B" ]]; then build="$arg"; fi
  if [[ "$prev" == "--build" ]]; then build="$arg"; fi
  prev="$arg"
done
[[ -n "$build" ]] || { echo "no build directory found" >&2; exit 1; }
mkdir -p "$build/bin"
cp "$STUB_QUANTIZE" "$build/bin/llama-quantize"
cp "$STUB_IMATRIX" "$build/bin/llama-imatrix"
"""


def _write_exec(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _make_stub_world(tmp_path: Path, *, prebuilt_tools: bool) -> tuple[Path, Path, Path]:
    """Return (llama_cpp_dir, fake_python, stub_bin)."""
    llama_dir = tmp_path / "llama.cpp"
    (llama_dir / "build" / "bin").mkdir(parents=True)
    (llama_dir / "CMakeLists.txt").write_text("# stub\n", encoding="utf-8")
    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir()
    _write_exec(stub_bin / "stub-quantize", QUANTIZE_TOOL)
    _write_exec(stub_bin / "stub-imatrix", IMATRIX_TOOL)
    _write_exec(stub_bin / "cmake", FAKE_CMAKE)
    if prebuilt_tools:
        _write_exec(llama_dir / "build" / "bin" / "llama-quantize", QUANTIZE_TOOL)
        _write_exec(llama_dir / "build" / "bin" / "llama-imatrix", IMATRIX_TOOL)
    fake_python = tmp_path / "fake-python"
    _write_exec(fake_python, FAKE_PYTHON)
    return llama_dir, fake_python, stub_bin


def _run(
    tmp_path: Path, llama_dir: Path, fake_python: Path, stub_bin: Path, *extra: str
) -> subprocess.CompletedProcess:
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir(exist_ok=True)
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    env = dict(os.environ, STUB_QUANTIZE=str(stub_bin / "stub-quantize"), STUB_IMATRIX=str(stub_bin / "stub-imatrix"))
    env["PATH"] = f"{stub_bin}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run(
        [
            "bash",
            str(ROOT / "quantize.sh"),
            "--hf-dir",
            str(hf_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--name",
            "stubtest",
            "--llama-cpp-dir",
            str(llama_dir),
            "--python",
            str(fake_python),
            "--threads",
            "2",
            *extra,
        ],
        capture_output=True,
        text=True,
        env=env,
    )


def test_full_pipeline_with_prebuilt_stubs(tmp_path):
    llama_dir, fake_python, stub_bin = _make_stub_world(tmp_path, prebuilt_tools=True)
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("তন্ত্র সাধনা\n", encoding="utf-8")
    result = _run(tmp_path, llama_dir, fake_python, stub_bin, "--calibration", str(calibration))
    assert result.returncode == 0, result.stderr + result.stdout

    out = tmp_path / "out"
    produced = [out / "stubtest-Q4_K_M.gguf", out / "stubtest-Q3_K_L.gguf"]
    for path in produced:
        assert path.read_bytes().startswith(b"GGUF")
    imatrix = out / "stubtest.imatrix.gguf"
    assert imatrix.read_bytes() == b"imatrix-stub"

    checksums = (out / "stubtest.SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert len(checksums) == 2
    listed = {line.split()[1]: line.split()[0] for line in checksums}
    for path in produced:
        assert listed[path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "valid GGUF" in result.stdout


def test_pipeline_builds_tools_with_stub_cmake_when_missing(tmp_path):
    # Exercises the cmake configure+build path (where the bash 3.2 empty-array bug used to hide).
    llama_dir, fake_python, stub_bin = _make_stub_world(tmp_path, prebuilt_tools=False)
    result = _run(tmp_path, llama_dir, fake_python, stub_bin, "--skip-imatrix")
    assert result.returncode == 0, result.stderr + result.stdout
    assert (tmp_path / "out" / "stubtest-Q4_K_M.gguf").exists()
    assert "tools ready" in result.stdout


def test_pipeline_without_calibration_skips_the_imatrix(tmp_path):
    llama_dir, fake_python, stub_bin = _make_stub_world(tmp_path, prebuilt_tools=True)
    result = _run(tmp_path, llama_dir, fake_python, stub_bin, "--calibration", str(tmp_path / "absent.txt"))
    assert result.returncode == 0, result.stderr + result.stdout
    assert "continuing without an i-matrix" in result.stderr
    assert not (tmp_path / "out" / "stubtest.imatrix.gguf").exists()
    assert (tmp_path / "out" / "stubtest-Q4_K_M.gguf").exists()


def test_existing_f16_is_reused(tmp_path):
    llama_dir, fake_python, stub_bin = _make_stub_world(tmp_path, prebuilt_tools=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "stubtest-f16.gguf").write_bytes(b"GGUF-earlier-run")
    result = _run(tmp_path, llama_dir, fake_python, stub_bin, "--skip-imatrix")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "reusing" in result.stdout


def test_invalid_quant_type_is_rejected_before_any_work(tmp_path):
    llama_dir, fake_python, stub_bin = _make_stub_world(tmp_path, prebuilt_tools=True)
    result = _run(tmp_path, llama_dir, fake_python, stub_bin, "--types", "Q9", "--skip-imatrix")
    assert result.returncode == 1
    assert "unsupported quantization type" in result.stderr


@pytest.mark.parametrize("types", [",", "Q4_K_M,", "Q4 K M"])
def test_malformed_type_lists_are_rejected(tmp_path, types):
    llama_dir, fake_python, stub_bin = _make_stub_world(tmp_path, prebuilt_tools=True)
    result = _run(tmp_path, llama_dir, fake_python, stub_bin, "--types", types, "--skip-imatrix")
    assert result.returncode == 1
    assert "quantization type" in result.stderr


@pytest.mark.parametrize("name", ["with space", "bad/name"])
def test_suspicious_output_names_are_rejected(tmp_path, name):
    llama_dir, fake_python, stub_bin = _make_stub_world(tmp_path, prebuilt_tools=True)
    result = _run(tmp_path, llama_dir, fake_python, stub_bin, "--name", name, "--skip-imatrix")
    assert result.returncode == 1
    assert "--name may only contain" in result.stderr
