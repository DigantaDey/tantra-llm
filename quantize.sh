#!/usr/bin/env bash
# Convert a local Hugging Face model to F16 GGUF, compute a corpus i-matrix,
# quantize with llama.cpp, then verify the resulting GGUF before publishing it.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./quantize.sh MODEL_DIR CALIBRATION_TEXT_OR_DIR OUTPUT.gguf [Q4_K_M|Q3_K_L]

Required environment:
  LLAMA_CPP_DIR       llama.cpp checkout (default: ./llama.cpp)
  LLAMA_IMATRIX       optional explicit path to llama-imatrix
  LLAMA_QUANTIZE      optional explicit path to llama-quantize
  PYTHON_BIN          Python used by the HF-to-GGUF converter (default: python3)
  CALIBRATION_MAX_CHARS  maximum UTF-8 characters assembled from calibration input (default: 250000)
  KEEP_INTERMEDIATES=1  retain the temporary F16 model and i-matrix for inspection

The calibration input should be corrected Tantra text (a UTF-8 .txt file or a
folder of .txt/.md files). This script does not download weights or build
llama.cpp for you.
USAGE
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

MODEL_DIR=$1
CALIBRATION_INPUT=$2
OUTPUT_ARGUMENT=$3
QUANT_TYPE=${4:-Q4_K_M}
PYTHON_BIN=${PYTHON_BIN:-python3}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-"$(pwd)/llama.cpp"}
CALIBRATION_MAX_CHARS=${CALIBRATION_MAX_CHARS:-250000}
KEEP_INTERMEDIATES=${KEEP_INTERMEDIATES:-0}

case "$QUANT_TYPE" in
  Q4_K_M|Q3_K_L) ;;
  *) echo "Unsupported quantization: $QUANT_TYPE (choose Q4_K_M or Q3_K_L)." >&2; exit 2 ;;
esac

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Hugging Face model directory not found: $MODEL_DIR" >&2
  exit 2
fi
if [[ ! -e "$CALIBRATION_INPUT" ]]; then
  echo "Calibration text/directory not found: $CALIBRATION_INPUT" >&2
  exit 2
fi
if [[ "$OUTPUT_ARGUMENT" != *.gguf ]]; then
  echo "Output filename must end in .gguf: $OUTPUT_ARGUMENT" >&2
  exit 2
fi
if ! [[ "$CALIBRATION_MAX_CHARS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CALIBRATION_MAX_CHARS must be a positive integer." >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

if [[ -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]]; then
  CONVERTER="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
elif [[ -f "$LLAMA_CPP_DIR/convert-hf-to-gguf.py" ]]; then
  CONVERTER="$LLAMA_CPP_DIR/convert-hf-to-gguf.py"
else
  echo "Could not find convert_hf_to_gguf.py under $LLAMA_CPP_DIR. Set LLAMA_CPP_DIR to a llama.cpp checkout." >&2
  exit 2
fi

if [[ -n "${LLAMA_IMATRIX:-}" ]]; then
  IMATRIX_BIN=$LLAMA_IMATRIX
elif [[ -x "$LLAMA_CPP_DIR/build/bin/llama-imatrix" ]]; then
  IMATRIX_BIN="$LLAMA_CPP_DIR/build/bin/llama-imatrix"
elif command -v llama-imatrix >/dev/null 2>&1; then
  IMATRIX_BIN=$(command -v llama-imatrix)
else
  echo "llama-imatrix not found. Build llama.cpp (including its tools) or set LLAMA_IMATRIX." >&2
  exit 2
fi

if [[ -n "${LLAMA_QUANTIZE:-}" ]]; then
  QUANTIZE_BIN=$LLAMA_QUANTIZE
elif [[ -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]]; then
  QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
elif [[ -x "$LLAMA_CPP_DIR/build/bin/quantize" ]]; then
  QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/quantize"
elif command -v llama-quantize >/dev/null 2>&1; then
  QUANTIZE_BIN=$(command -v llama-quantize)
else
  echo "llama-quantize not found. Build llama.cpp (including its tools) or set LLAMA_QUANTIZE." >&2
  exit 2
fi

OUTPUT_DIR=$(dirname "$OUTPUT_ARGUMENT")
OUTPUT_NAME=$(basename "$OUTPUT_ARGUMENT")
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
OUTPUT="$OUTPUT_DIR/$OUTPUT_NAME"
PARTIAL_OUTPUT="$OUTPUT_DIR/.${OUTPUT_NAME%.gguf}.$$.partial.gguf"
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/tantra-quant.XXXXXX")
cleanup() {
  rm -f "$PARTIAL_OUTPUT"
  if [[ "$KEEP_INTERMEDIATES" != "1" ]]; then
    rm -rf "$WORK_DIR"
  else
    echo "Intermediates retained in: $WORK_DIR" >&2
  fi
}
trap cleanup EXIT

F16_MODEL="$WORK_DIR/model-f16.gguf"
IMATRIX_FILE="$WORK_DIR/tantra-imatrix.gguf"
CALIBRATION_FILE="$WORK_DIR/calibration.txt"

"$PYTHON_BIN" - "$CALIBRATION_INPUT" "$CALIBRATION_FILE" "$CALIBRATION_MAX_CHARS" <<'PY'
from pathlib import Path
import re
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
limit = int(sys.argv[3])
paths = [source] if source.is_file() else sorted(
    path for path in source.rglob("*")
    if path.is_file() and path.suffix.casefold() in {".txt", ".md"}
)
if not paths:
    raise SystemExit(f"No .txt or .md calibration files found in {source}")
remaining = limit
parts = []
for path in paths:
    if remaining <= 0:
        break
    with path.open("r", encoding="utf-8-sig", errors="replace") as source_file:
        text = source_file.read(remaining + 4096)
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                lines = lines[index + 1:]
                break
    text = "\n".join(line for line in lines if not line.lstrip().startswith("# "))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        continue
    snippet = text[:remaining]
    parts.append(snippet)
    remaining -= len(snippet)
calibration = "\n\n".join(parts).strip()
if not calibration:
    raise SystemExit("Calibration input contains no usable text")
destination.write_text(calibration + "\n", encoding="utf-8")
print(f"Prepared {len(calibration):,} calibration characters from {len(paths)} source file(s).")
PY

printf '\n[1/3] Converting Hugging Face weights to F16 GGUF…\n'
"$PYTHON_BIN" "$CONVERTER" "$MODEL_DIR" --outfile "$F16_MODEL" --outtype f16
[[ -s "$F16_MODEL" ]] || { echo "F16 conversion did not create a model." >&2; exit 1; }

printf '\n[2/3] Computing an importance matrix from corrected Tantra calibration text…\n'
"$IMATRIX_BIN" -m "$F16_MODEL" -f "$CALIBRATION_FILE" -o "$IMATRIX_FILE"
[[ -s "$IMATRIX_FILE" ]] || { echo "llama-imatrix did not create an importance matrix." >&2; exit 1; }

printf '\n[3/3] Quantizing with %s and the i-matrix…\n' "$QUANT_TYPE"
rm -f "$PARTIAL_OUTPUT"
"$QUANTIZE_BIN" --imatrix "$IMATRIX_FILE" "$F16_MODEL" "$PARTIAL_OUTPUT" "$QUANT_TYPE"
[[ -s "$PARTIAL_OUTPUT" ]] || { echo "Quantizer did not create an output file." >&2; exit 1; }
"$PYTHON_BIN" - "$PARTIAL_OUTPUT" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
with path.open("rb") as model:
    magic = model.read(4)
if magic != b"GGUF":
    raise SystemExit(f"Output header is {magic!r}, not GGUF")
if path.stat().st_size < 1024:
    raise SystemExit("Output is unexpectedly small to be a valid model")
PY
mv -f "$PARTIAL_OUTPUT" "$OUTPUT"
printf '\nVerified GGUF: %s (%s bytes)\n' "$OUTPUT" "$(wc -c < "$OUTPUT" | tr -d ' ')"
printf 'Load it with: python app.py --model "%s"\n' "$OUTPUT"
