#!/usr/bin/env bash
# setup.sh — one-command bootstrap for tantra-llm on macOS (Apple Silicon or Intel) and Linux.
#
#   ./setup.sh                  # venv + PyTorch + training tools + chat runtime + model
#   ./setup.sh --small-model    # lighter 0.5B model instead of the default 1.5B
#   ./setup.sh --no-model       # everything except the model download
#   ./setup.sh --no-llm         # skip llama-cpp-python (no chat/benchmark runtime)
#
# The chat runtime (llama-cpp-python) is compiled from source on first install; on macOS
# that takes a few minutes and builds the Metal backend so Apple Silicon runs on the GPU.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PYTHON_BIN="${PYTHON:-}"
DO_LLM=1
DO_MODEL=1
MODEL_SIZE="standard"

# ----------------------------------------------------------------- output helpers
if [[ -t 1 ]]; then
  C_BOLD=$'\033[1m'; C_GOLD=$'\033[38;5;178m'; C_OK=$'\033[38;5;114m'; C_WARN=$'\033[38;5;221m'; C_ERR=$'\033[38;5;203m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
  C_BOLD=""; C_GOLD=""; C_OK=""; C_WARN=""; C_ERR=""; C_DIM=""; C_OFF=""
fi
step() { printf '\n%s▌%s %s%s%s\n' "$C_GOLD" "$C_OFF" "$C_BOLD" "$*" "$C_OFF"; }
ok()   { printf '  %s✓%s %s\n' "$C_OK" "$C_OFF" "$*"; }
info() { printf '  %s%s%s\n' "$C_DIM" "$*" "$C_OFF"; }
warn() { printf '  %s!%s %s\n' "$C_WARN" "$C_OFF" "$*" >&2; }
die()  { printf '  %s✗%s %s\n' "$C_ERR" "$C_OFF" "$*" >&2; exit 1; }

usage() {
  cat <<USAGE
${C_BOLD}setup.sh${C_OFF} — set up tantra-llm on this machine

Usage:
  ./setup.sh [options]

Options:
  --small-model     fetch Qwen2.5-0.5B-Instruct instead of the 1.5B default
  --no-model        skip the model download
  --no-llm          skip llama-cpp-python (chat and benchmark will not run)
  --python PATH     python interpreter to use (default: python3)
  -h, --help        show this help

Environment:
  CMAKE_ARGS        passed through to the llama-cpp-python build
                    (e.g. CMAKE_ARGS=-DGGML_METAL=OFF for a CPU-only build on macOS)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --small-model) MODEL_SIZE="small"; shift ;;
    --no-model)    DO_MODEL=0; shift ;;
    --no-llm)      DO_LLM=0; shift ;;
    --python)      [[ $# -ge 2 ]] || die "--python needs a value"; PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *)             usage >&2; die "unknown argument: $1" ;;
  esac
done

# ----------------------------------------------------------------- host checks
step "Host"
if [[ -z "$PYTHON_BIN" ]]; then PYTHON_BIN=python3; fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python3 not found (macOS: brew install python@3.12 · Debian/Ubuntu: apt install python3)"
"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
  || die "$PYTHON_BIN is older than 3.11. Install Python 3.11+ (macOS: brew install python@3.12) and rerun with --python"
ok "python: $("$PYTHON_BIN" --version 2>&1)"

OS="$(uname -s)"
case "$OS" in
  Darwin)
    if ! xcode-select -p >/dev/null 2>&1; then
      die "Xcode Command Line Tools are missing. Run: xcode-select --install   then rerun ./setup.sh"
    fi
    ok "macOS $(uname -m) · Xcode Command Line Tools present"
    ;;
  Linux)
    ok "linux $(uname -m)"
    ;;
  *)
    warn "untested operating system: $OS (continuing anyway)"
    ;;
esac

# ----------------------------------------------------------------- virtualenv
step "Virtual environment"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV" || die "could not create the virtual environment at $VENV"
fi
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
ok "$VENV"

"$PY" -m pip install --quiet --upgrade pip
ok "pip up to date"

# ----------------------------------------------------------------- dependencies
step "Python packages"
if [[ "$OS" == "Darwin" ]]; then
  # macOS wheels on PyPI are CPU builds (unified Metal is used only through llama.cpp).
  "$PIP" install --quiet torch || die "PyTorch install failed"
else
  "$PIP" install --quiet torch --index-url https://download.pytorch.org/whl/cpu || die "PyTorch install failed"
fi
ok "PyTorch"

"$PIP" install --quiet -e "${SCRIPT_DIR}[train]" || die "project install failed"
ok "tantra-llm (scraper, cleaner, tests)"

if [[ $DO_LLM -eq 1 ]]; then
  info "llama-cpp-python: compiling llama.cpp from source, this takes a few minutes…"
  if [[ "$OS" == "Darwin" ]] && ! xcrun -f metal >/dev/null 2>&1; then
    warn "Metal toolchain not found: building the CPU-only runtime (install full Xcode for GPU acceleration)"
    CMAKE_ARGS="${CMAKE_ARGS:-} -DGGML_METAL=OFF"
    export CMAKE_ARGS
  fi
  "$PIP" install --quiet -e "${SCRIPT_DIR}[llm]" || die "llama-cpp-python build failed (see output above; on macOS check xcode-select -p)"
  ok "chat runtime (llama-cpp-python)"
else
  info "skipped llama-cpp-python (--no-llm)"
fi

# ----------------------------------------------------------------- model
if [[ $DO_MODEL -eq 1 ]]; then
  step "Chat model"
  "$PY" "$SCRIPT_DIR/fetch_model.py" --size "$MODEL_SIZE" || die "model download failed"
else
  info "skipped the model download (--no-model)"
fi

# ----------------------------------------------------------------- done
step "Done"
info "activate the environment:  . $VENV/bin/activate"
info "chat:                     python app.py"
info "benchmark:                python benchmark.py --enforce"
info "run the test suite:       pytest -q"
