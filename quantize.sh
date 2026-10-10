#!/usr/bin/env bash
# quantize.sh — Hugging Face model -> f16 GGUF -> importance-matrix -> Q4_K_M / Q3_K_L GGUF files.
#
# Pipeline
#   1. fetch + build a pinned llama.cpp (llama-quantize, llama-imatrix)
#   2. convert the HF checkpoint to an f16 GGUF        (convert_hf_to_gguf.py)
#   3. compute an importance matrix on calibration text (llama-imatrix)  [optional]
#   4. quantize to each requested type                  (llama-quantize --imatrix)
#   5. validate every output (GGUF magic, non-empty) and write SHA256SUMS
#
# Memory note: steps 2-3 load the full-precision model (~3 GB for a 1.5B model).
# Run them on a machine with >= 8 GB RAM; the *output* runs fine on 4 GB.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------- defaults
LLAMA_CPP_REPO="${TANTRA_LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_CPP_REF="${TANTRA_LLAMA_CPP_REF:-v0.6.0}"
LLAMA_CPP_DIR="${TANTRA_LLAMA_CPP_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/tantra-llm/llama.cpp}"
PYTHON_BIN="${PYTHON:-}"
HF_DIR=""
NAME="tantra-qwen2.5-1.5b"
OUT_DIR="${SCRIPT_DIR}/models/gguf"
TYPES="Q4_K_M,Q3_K_L"
CALIBRATION="${SCRIPT_DIR}/data/calibration/calibration.txt"
IMATRIX_CHUNKS=100
THREADS=""
SKIP_IMATRIX=0
DROP_F16=0
DRY_RUN=0

readonly SUPPORTED_TYPES="Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K Q8_0"

# ----------------------------------------------------------------- output helpers
if [[ -t 1 ]]; then
  C_BOLD=$'\033[1m'; C_GOLD=$'\033[38;5;178m'; C_OK=$'\033[38;5;114m'; C_WARN=$'\033[38;5;221m'; C_ERR=$'\033[38;5;203m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
  C_BOLD=""; C_GOLD=""; C_OK=""; C_WARN=""; C_ERR=""; C_DIM=""; C_OFF=""
fi
step()  { printf '\n%s▌%s %s%s%s\n' "$C_GOLD" "$C_OFF" "$C_BOLD" "$*" "$C_OFF"; }
ok()    { printf '  %s✓%s %s\n' "$C_OK" "$C_OFF" "$*"; }
info()  { printf '  %s%s%s\n' "$C_DIM" "$*" "$C_OFF"; }
warn()  { printf '  %s!%s %s\n' "$C_WARN" "$C_OFF" "$*" >&2; }
die()   { printf '  %s✗%s %s\n' "$C_ERR" "$C_OFF" "$*" >&2; exit 1; }

usage() {
  cat <<USAGE
${C_BOLD}quantize.sh${C_OFF} — convert and quantize a Hugging Face model to GGUF (llama.cpp, i-matrix aware)

Usage:
  ./quantize.sh --hf-dir DIR [options]

Required:
  --hf-dir DIR          local Hugging Face model directory (config.json + weights + tokenizer)

Options:
  --name NAME           output base name (default: ${NAME})
  --out-dir DIR         output directory (default: models/gguf)
  --types LIST          comma-separated quantization types (default: ${TYPES})
                        supported: ${SUPPORTED_TYPES}
  --calibration FILE    plain-text calibration corpus for the i-matrix
                        (default: data/calibration/calibration.txt)
  --imatrix-chunks N    chunks of calibration text to process (default: ${IMATRIX_CHUNKS})
  --skip-imatrix        quantize without an importance matrix (lower quality at Q3/Q4)
  --threads N           CPU threads for build and quantization (default: all cores)
  --llama-cpp-dir DIR   llama.cpp checkout to use or create (default: ${LLAMA_CPP_DIR})
  --python PATH         python interpreter with transformers/gguf/numpy (default: .venv or python3)
  --drop-f16            delete the f16 intermediate after quantizing
  --dry-run             print the plan and exit
  -h, --help            show this help

Environment:
  TANTRA_LLAMA_CPP_REF  llama.cpp tag to build (default: ${LLAMA_CPP_REF})
  TANTRA_LLAMA_CPP_DIR  same as --llama-cpp-dir
USAGE
}

# ----------------------------------------------------------------- arguments
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 needs a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-dir)          need_value "$@"; HF_DIR="$2"; shift 2 ;;
    --name)            need_value "$@"; NAME="$2"; shift 2 ;;
    --out-dir)         need_value "$@"; OUT_DIR="$2"; shift 2 ;;
    --types)           need_value "$@"; TYPES="$2"; shift 2 ;;
    --calibration)     need_value "$@"; CALIBRATION="$2"; shift 2 ;;
    --imatrix-chunks)  need_value "$@"; IMATRIX_CHUNKS="$2"; shift 2 ;;
    --threads)         need_value "$@"; THREADS="$2"; shift 2 ;;
    --llama-cpp-dir)   need_value "$@"; LLAMA_CPP_DIR="$2"; shift 2 ;;
    --python)          need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --skip-imatrix)    SKIP_IMATRIX=1; shift ;;
    --drop-f16)        DROP_F16=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    *)                 usage >&2; die "unknown argument: $1" ;;
  esac
done

[[ -n "$HF_DIR" ]] || { usage >&2; die "--hf-dir is required"; }
[[ -d "$HF_DIR" && -f "$HF_DIR/config.json" ]] || die "$HF_DIR is not a Hugging Face model directory (no config.json)"
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "--name may only contain letters, digits, '.', '_' and '-'"
[[ "$IMATRIX_CHUNKS" =~ ^[0-9]+$ ]] || die "--imatrix-chunks must be a positive integer"

IFS=',' read -r -a REQUESTED <<<"$TYPES"
[[ ${#REQUESTED[@]} -gt 0 ]] || die "--types is empty"
for t in "${REQUESTED[@]}"; do
  [[ " $SUPPORTED_TYPES " == *" $t "* ]] || die "unsupported quantization type '$t' (supported: $SUPPORTED_TYPES)"
done

if [[ -z "$THREADS" ]]; then
  THREADS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
fi
[[ "$THREADS" =~ ^[0-9]+$ && "$THREADS" -ge 1 ]] || die "--threads must be a positive integer"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"; else PYTHON_BIN="python3"; fi
fi

mkdir -p "$OUT_DIR"
F16_PATH="$OUT_DIR/$NAME-f16.gguf"
IMATRIX_PATH="$OUT_DIR/$NAME.imatrix.gguf"
USE_IMATRIX=0
if [[ $SKIP_IMATRIX -eq 0 ]]; then
  if [[ -s "$CALIBRATION" ]]; then USE_IMATRIX=1; else warn "calibration file missing or empty ($CALIBRATION): continuing without an i-matrix"; fi
fi

step "Plan"
info "source        $HF_DIR"
info "llama.cpp     $LLAMA_CPP_DIR @ $LLAMA_CPP_REF"
info "output        $OUT_DIR/$NAME-<type>.gguf"
info "types         ${TYPES}"
info "i-matrix      $([[ $USE_IMATRIX -eq 1 ]] && echo "$CALIBRATION ($IMATRIX_CHUNKS chunks)" || echo "disabled")"
info "threads       $THREADS"
if [[ $DRY_RUN -eq 1 ]]; then ok "dry run: nothing executed"; exit 0; fi

# ----------------------------------------------------------------- 1. llama.cpp
step "llama.cpp $LLAMA_CPP_REF"
if [[ ! -d "$LLAMA_CPP_DIR/.git" && ! -f "$LLAMA_CPP_DIR/CMakeLists.txt" ]]; then
  command -v git >/dev/null 2>&1 || die "git is required to fetch llama.cpp"
  info "cloning $LLAMA_CPP_REPO"
  mkdir -p "$(dirname -- "$LLAMA_CPP_DIR")"
  git clone --quiet --depth 1 --branch "$LLAMA_CPP_REF" "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR" \
    || die "could not clone $LLAMA_CPP_REPO at $LLAMA_CPP_REF"
fi
if [[ -d "$LLAMA_CPP_DIR/.git" ]]; then
  current="$(git -C "$LLAMA_CPP_DIR" describe --tags --exact-match 2>/dev/null || git -C "$LLAMA_CPP_DIR" rev-parse --short HEAD)"
  if [[ "$current" != "$LLAMA_CPP_REF" ]]; then
    warn "llama.cpp checkout is at '$current', expected '$LLAMA_CPP_REF' (conversion may differ)"
  else
    ok "source at $LLAMA_CPP_REF"
  fi
fi

LLAMA_BIN="$LLAMA_CPP_DIR/build/bin"
if [[ ! -x "$LLAMA_BIN/llama-quantize" || ! -x "$LLAMA_BIN/llama-imatrix" ]]; then
  if command -v cmake >/dev/null 2>&1; then CMAKE_BIN=cmake; elif [[ -x "$SCRIPT_DIR/.venv/bin/cmake" ]]; then CMAKE_BIN="$SCRIPT_DIR/.venv/bin/cmake"; else die "cmake not found (pip install cmake, or install it with your package manager)"; fi
  GENERATOR=()
  if command -v ninja >/dev/null 2>&1; then GENERATOR=(-G Ninja); fi
  info "building llama-quantize and llama-imatrix (first time only)"
  "$CMAKE_BIN" -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" "${GENERATOR[@]}" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null
  "$CMAKE_BIN" --build "$LLAMA_CPP_DIR/build" --config Release -j "$THREADS" --target llama-quantize llama-imatrix >/dev/null
fi
[[ -x "$LLAMA_BIN/llama-quantize" && -x "$LLAMA_BIN/llama-imatrix" ]] || die "build did not produce llama-quantize / llama-imatrix"
ok "tools ready: $LLAMA_BIN"

# ----------------------------------------------------------------- 2. convert
step "Convert Hugging Face checkpoint -> f16 GGUF"
"$PYTHON_BIN" -c "import gguf, numpy, transformers" 2>/dev/null \
  || die "$PYTHON_BIN lacks gguf/numpy/transformers: pip install -e '.[convert]'"
if [[ -s "$F16_PATH" ]]; then
  ok "reusing $F16_PATH"
else
  "$PYTHON_BIN" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$HF_DIR" --outtype f16 --outfile "$F16_PATH" \
    || die "conversion failed"
  ok "$F16_PATH"
fi

# ----------------------------------------------------------------- 3. i-matrix
if [[ $USE_IMATRIX -eq 1 ]]; then
  step "Importance matrix from calibration text"
  total_mem_kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  if [[ "$total_mem_kb" -gt 0 && "$total_mem_kb" -lt 6000000 ]]; then
    warn "only $((total_mem_kb / 1024)) MB RAM: the f16 model may not fit during imatrix; consider a larger machine"
  fi
  "$LLAMA_BIN/llama-imatrix" -m "$F16_PATH" -f "$CALIBRATION" -o "$IMATRIX_PATH" \
    --chunks "$IMATRIX_CHUNKS" -t "$THREADS" -ngl 0 --no-ppl >/dev/null \
    || die "llama-imatrix failed"
  [[ -s "$IMATRIX_PATH" ]] || die "i-matrix file is empty"
  ok "$IMATRIX_PATH"
fi

# ----------------------------------------------------------------- 4. quantize
step "Quantize"
produced=()
for t in "${REQUESTED[@]}"; do
  out="$OUT_DIR/$NAME-$t.gguf"
  args=("$LLAMA_BIN/llama-quantize")
  [[ $USE_IMATRIX -eq 1 ]] && args+=(--imatrix "$IMATRIX_PATH")
  args+=("$F16_PATH" "$out" "$t" "$THREADS")
  "${args[@]}" >/dev/null || die "quantization to $t failed"
  produced+=("$out")
  ok "$out"
done

# ----------------------------------------------------------------- 5. validate
step "Validate"
for f in "${produced[@]}"; do
  [[ -s "$f" ]] || die "$f is empty"
  magic="$(head -c 4 "$f")"
  [[ "$magic" == "GGUF" ]] || die "$f does not start with the GGUF magic bytes"
  size_mb=$(( $(stat -c %s "$f") / 1048576 ))
  ok "$(basename -- "$f") · valid GGUF · ${size_mb} MB"
done
names=()
for f in "${produced[@]}"; do names+=("$(basename -- "$f")"); done
( cd "$OUT_DIR" && sha256sum -- "${names[@]}" > "$NAME.SHA256SUMS" )
ok "checksums: $OUT_DIR/$NAME.SHA256SUMS"

if [[ $DROP_F16 -eq 1 ]]; then rm -f -- "$F16_PATH"; info "removed f16 intermediate"; fi

step "Done"
info "run the chat:       python app.py --model ${produced[0]}"
info "measure RAM/speed:  python benchmark.py --model ${produced[0]} --enforce"
