# Tantra LLM

**A reproducible, offline-first workflow for finding historical Bengali Tantra texts, auditing Archive.org OCR, and preparing a small CPU-only GGUF assistant.**

The repository contains the workflow—not a pre-downloaded corpus, trained correction checkpoint, or model weights. That keeps source rights reviewable and avoids pretending that an OCR model has been trained or that unreviewed ritual text is ground truth.

## What is included

| Component | Purpose |
| --- | --- |
| `scraper.py` | Search Internet Archive text records in Bengali and transliteration; rank by Bengali-language metadata, date, and rights signals; download one preferred scan and one OCR transcript. |
| `cleaner.py` | Normalize Unicode and scan layout, use a small editable Tantra lexicon, optionally score candidates against a clean Bengali corpus in both directions, and emit source-linked Markdown. |
| `train_corrector.py` | Optional CPU trainer for a compact character-level encoder/decoder Transformer using synthetic OCR corruption and domain-term loss weighting. |
| `quantize.sh` | Convert a local Hugging Face model to F16 GGUF, compute an i-matrix from corrected text, quantize to `Q4_K_M` or `Q3_K_L`, and validate the GGUF header. |
| `app.py` | Offline interactive or one-shot local chat with CPU-thread limits, a conservative research system prompt, bounded history, and RSS warnings. |
| `benchmark.py` | Run one representative prompt while `psutil` samples process RSS, then report generation throughput and the sampled peak. |
| `data/tantra_terms.txt` | Editable starter anchors. This is a seed list, not an authoritative or exhaustive dictionary. |

## 1. Install

Python 3.10+ is supported. The scraper and deterministic cleaner only need the small core dependencies:

```bash
python -m pip install -r requirements.txt
```

The optional neural corrector trainer needs PyTorch:

```bash
python -m pip install -e '.[train]'
```

`llama-cpp-python` is only needed to run `app.py` or `benchmark.py`; it can take extra setup on CPU-only systems:

```bash
python -m pip install -e '.[runtime]'
```

If a source build is required, install the platform's C/C++ build tools first. To keep the final runtime lean, install training/conversion dependencies on a build machine and install only `llama-cpp-python` plus `psutil` on the 4 GB target.

## 2. Discover and download Archive.org material

Preview the ranked results first:

```bash
python scraper.py --dry-run --max-items 20
```

Download the selected source files:

```bash
python scraper.py --output-dir data/raw --max-items 50
```

The search uses `internetarchive.search_items()` for Bengali-script and Latin-script title/subject queries, deduplicates identifiers, and ranks explicit Bengali/Bangla tags and dates from 1870–1939 above less certain matches. `--strict-language` requires an explicit Bengali language tag; `--rows-per-query` controls the result window; `--all-scans` downloads every matching PDF/DJVU rather than just the preferred scan.

For each item, the scraper looks for an automatically generated `*_djvu.txt` (then other text transcripts) and a PDF/DJVU scan. It uses the Internet Archive Python library's `Item.download()` with an item-specific directory and resumable `ignore_existing` behavior. A `manifest.json` records selected files, source URLs, and failures. The default output stays under `data/raw/<identifier>/`.

### Rights and provenance

The scraper never edits Archive.org items. It skips records explicitly marked access-restricted and reports rights as `public-domain-indicated`, `rights-restricted-or-unclear`, or `unverified`. A publication date is **not** treated as proof of public-domain status in every jurisdiction. Use `--require-public-domain` to keep only records with explicit public-domain/CC0 metadata, then verify item-level terms and applicable law before redistribution. Metadata can be missing or wrong; review every item before using it in a release or model dataset.

## 3. Clean OCR conservatively

Use the scraped tree directly; an optional clean Bengali baseline improves word-vocabulary and bidirectional character-context ranking:

```bash
python cleaner.py \
  --input data/raw \
  --output data/clean \
  --baseline data/baseline \
  --terms data/tantra_terms.txt
```

Without `--baseline`, the cleaner still normalizes NFC Unicode, removes common invisible scan characters, repairs spaces after an explicit Bengali virama, collapses physical line wrapping into paragraphs, protects known terms, and writes Markdown plus `clean_manifest.json`. Use `--preserve-lines` for verse or sources where a single line break carries meaning. Existing outputs are left untouched unless `--overwrite` is passed.

**Safety policy:** visually similar letters (`র/ব/ক`, and the other pairs in `cleaner.py`) are not substituted by a blind mapping. A one-character suggestion must be in the supplied corpus or domain lexicon; a broken conjunct without an explicit virama is merged only when the resulting word is known. The small forward/reverse character trigram model helps rank ambiguous corpus-backed candidates. Exact domain terms are protected from neural rewriting. Add locally attested archaic spellings and mantras to `data/tantra_terms.txt` before running a large batch.

This is still OCR correction, not critical-edition scholarship. Review corrections against page images, especially mantras, personal names, variant readings, and any character sequence that changes ritual meaning. The markdown manifest stores input/output hashes to make review and regeneration easier.

## 4. Optional synthetic-degradation training

Prepare a clean, UTF-8 Bengali text baseline in a directory you have rights to use, then train the compact local denoiser:

```bash
python train_corrector.py \
  --corpus data/baseline \
  --terms data/tantra_terms.txt \
  --output models/tantra-char-denoiser.pt \
  --epochs 8 --threads 2
```

`synthetic_corrupt()` generates paired examples with broken conjuncts, low-rate look-alike substitutions, and random intra-word spaces. Training uses a small character-level encoder/decoder Transformer (default 128 dimensions, 2 layers) with the complete corrupted span as input, so the encoder has access to left **and** right context. Domain terms receive extra target-character loss weight and are added as examples. This sequence-to-sequence denoising objective is used instead of claiming an MLM checkpoint or downloading a large multilingual model; it learns only from the corpus you provide.

Apply the local checkpoint with a conservative edit-distance gate:

```bash
python cleaner.py \
  --input data/raw \
  --output data/clean-neural \
  --baseline data/baseline \
  --model models/tantra-char-denoiser.pt \
  --max-edit-ratio 0.34 \
  --overwrite
```

The cleaner skips a neural rewrite when it is empty, non-Bengali, or differs too much from the OCR input. Modern Bengali baselines can teach useful ligature and spacing regularities but can also erase archaic forms. Build a held-out noisy validation set from representative scans and manually assess character error rate before trusting a trained checkpoint at scale. Do not load checkpoints from untrusted sources.

## 5. Build an i-matrix-tuned GGUF

Choose a Bengali-capable instruct checkpoint such as `Qwen/Qwen2.5-0.5B-Instruct` for the smallest runtime, or try `Qwen/Qwen2.5-1.5B-Instruct` only after measuring the target hardware. Keep the Hugging Face model in a local directory; this project does not download weights automatically. For example:

```bash
python -m pip install huggingface_hub
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/base-qwen
```

Build the relevant llama.cpp tools on the conversion machine (CPU build shown):

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build \
  -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF
cmake --build llama.cpp/build --config Release -j2 --target llama-quantize llama-imatrix
python -m pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

The converter's Python dependencies (including a CPU PyTorch build) belong on the conversion machine, not the 4 GB inference target. Then run:

```bash
LLAMA_CPP_DIR="$PWD/llama.cpp" \
  ./quantize.sh \
  models/base-qwen \
  data/clean \
  models/tantra-qwen-0.5b-Q4_K_M.gguf \
  Q4_K_M
```

`quantize.sh` accepts a UTF-8 calibration text file or a directory of `.txt`/`.md` files. It assembles at most 250,000 calibration characters by default (tune with `CALIBRATION_MAX_CHARS`), strips the cleaner's Markdown front matter, converts HF weights to F16 GGUF, computes an i-matrix using the corrected text, and passes that matrix to `llama-quantize`. It writes the final `.gguf` only after checking its magic header. Temporary F16 and matrix files are removed unless `KEEP_INTERMEDIATES=1` is set. Conversion needs substantially more disk/RAM than inference; do not run that step on the 4 GB target unless it has enough swap and free storage.

The final model's memory is not determined by its weight file alone. KV cache, context size, backend, and the operating system change observed RSS. `Q4_K_M` with the 0.5B base and `--ctx-size 2048` is the conservative starting point; treat 2.5 GiB as a measured budget rather than a guarantee. A 1.5B model may or may not fit the same runtime budget on a given build.

## 6. Run and benchmark locally

Interactive chat:

```bash
python app.py --model models/tantra-qwen-0.5b-Q4_K_M.gguf
```

One-shot prompt:

```bash
python app.py \
  --model models/tantra-qwen-0.5b-Q4_K_M.gguf \
  --threads 2 \
  --ctx-size 2048 \
  --prompt 'এই পাঠে ন্যাস ও মুদ্রার উল্লেখ কীভাবে আলাদা?'
```

Run a measured sample (default thread choice is 4 on hosts with at least four logical CPUs, otherwise 2):

```bash
python benchmark.py \
  --model models/tantra-qwen-0.5b-Q4_K_M.gguf \
  --threads 2 \
  --tokens 96 \
  --prompt 'তন্ত্রসাধনায় মন্ত্র ও মুদ্রার ভূমিকা সংক্ষেপে ব্যাখ্যা করো।'
```

The runtime uses `n_gpu_layers=0`, memory-mapped weights, a bounded 2048-token context and 128-token batch by default. `app.py` reports RSS and warns above 2.5 GiB. `benchmark.py` samples RSS via `psutil` during model load and generation, prints throughput, and exits non-zero if the sampled peak exceeds its budget. For a strict deployment gate, benchmark several long prompts and consider leaving additional headroom for the OS and other processes.

## Development checks

The core test suite requires no network, PyTorch, or llama.cpp. To match CI exactly, install Ruff and run:

```bash
python -m pip install ruff
ruff check scraper.py cleaner.py train_corrector.py neural_corrector.py app.py benchmark.py tests
python -m compileall -q scraper.py cleaner.py train_corrector.py neural_corrector.py app.py benchmark.py tests
python -m unittest discover -s tests -v
bash -n quantize.sh
```

GitHub Actions runs those checks on pushes and pull requests. Live Archive.org downloads, neural training, GGUF conversion, and runtime benchmarks are deliberately opt-in because they need network access, user-supplied corpora/model weights, and/or significant compute.
