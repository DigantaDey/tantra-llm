# tantra-llm

An offline pipeline for Bengali Tantra and Sadhana texts from the Internet Archive: find and download the scans, clean the OCR, and run a small local chat model on a 4 GB CPU-only machine — with answers that can be grounded in the fetched corpus.

```
archive.org ──► scraper.py ──► data/raw ──► cleaner.py ──► data/processed/*.md ──► calibration.txt
                (search,        (djvu.txt,   (denoiser +       (unified Markdown    │
                 download,      pdf, djvu,    lexicon anchor    and plain text)     ▼
                 verify)        metadata)    or --no-model)                       quantize.sh
                                                                         (f16 → i-matrix → Q4_K_M, Q3_K_L)
                                                                                     │
                                              app.py ◄──────────────┴──► benchmark.py
                                    (chat, grounded in data/processed,    (speed, RAM, JSON report)
                                     2.5 GiB budget)
```

## Status at a glance

| Part | What is verified | What is not verified here |
|---|---|---|
| Scraper | Query building, acceptance rules, file selection, MD5 checks, retries, resume, dry-run — against an in-memory fake **and** against a local HTTP server that mirrors the live API's exact JSON shapes (numeric `year` fields, `{"error": ...}` replies with HTTP 200, Range-resumable downloads, non-ASCII filenames). CI additionally runs a **live archive.org integration job on every push and PR**: real search, real metadata, a real MD5-verified download. | Occasional archive.org outages (the live job retries transient failures). |
| Cleaner | Synthetic corruption, alignment round-trips, training, checkpoint reload, dictionary anchor, end-to-end correct/calibration/evaluate on small fixtures — including the `--no-model` structural path that needs no checkpoint. | Quality on real scans. The corruption rates are assumptions (see below). |
| Quantization | `quantize.sh` runs the full chain (HF → f16 GGUF → i-matrix → Q4_K_M and Q3_K_L → GGUF magic check → SHA256SUMS) on a tiny random LLaMA, and the same chain is exercised on every platform by an integration test with stub tools. `llama-cpp-python` 0.3.36 loads both outputs. Works with GNU and BSD userland (macOS) and bash 3.2. | The Qwen2.5 conversion. The sandbox cannot reach Hugging Face. |
| Model download | `fetch_model.py` resumable download, size/SHA-256/GGUF verification, against a local fake Hub; the pinned sizes and hashes of both Qwen GGUFs were read from the Hugging Face tree API. | The live download itself (the sandbox cannot reach huggingface.co). |
| Grounded chat | BM25 retrieval over the processed corpus (Bengali-aware tokenisation), citation markers, prompt budgeting, per-turn system-prompt override — all unit-tested with a fake backend. | Retrieval quality on the real corpus, and answer quality of the underlying LLM. |
| App and benchmark | `app.py` and `benchmark.py` run end-to-end on the tiny model with the real `llama-cpp-python`, including `--prompt`, the interactive REPL, `--gpu-layers`, `--ground`, JSON reports and `--enforce`. CI builds the chat runtime from source on macOS (Apple Silicon) and runs the tests there. | Memory and speed of the 1.5B model. Not measured yet. |

Everything is covered by 600 offline unit and integration tests that run in a few seconds, plus 2 live archive.org integration tests (`pytest -m live`) that CI runs on every push and pull request.

## Quick start

Requires Python 3.11–3.13 (3.14 works with current PyTorch but is not in CI).

The one-command path (macOS Apple Silicon / Intel, or Linux):

```bash
./setup.sh            # venv + PyTorch + chat runtime + the default chat model
python app.py         # chat
```

`setup.sh` does the right thing per platform: on macOS it checks for the Xcode Command Line Tools and builds the Metal backend so Apple Silicon runs on the GPU; on Linux it installs the small CPU-only PyTorch build. Flags: `--small-model` (0.5B instead of 1.5B), `--no-model`, `--no-llm`, `--python PATH`.

With setup done, the archive-to-chat journey is three commands:

```bash
python scraper.py download --limit 10     # fetch Bengali Tantra texts from archive.org (MD5-verified)
python cleaner.py correct --no-model      # → data/processed (Markdown + plain text, zero setup)
python app.py --ground                    # chat with answers grounded in the fetched texts
```

Manual setup, step by step:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                                              # scraper, cleaner, tests (pulls in PyTorch)
pytest -q                                                            # 600 tests, a few seconds
```

On Linux you can install the CPU-only PyTorch wheel first (`pip install --index-url https://download.pytorch.org/whl/cpu torch`) to keep the download small. On macOS the PyPI wheel already is the CPU build; there is no separate CPU index.

For the chat app and benchmark, also install the local runtime and (optionally) the conversion tools:

```bash
pip install -e ".[llm]"       # llama-cpp-python, compiled from source: a few minutes
pip install -e ".[convert]"   # only for quantize.sh: transformers, gguf, sentencepiece
./quantize.sh --help          # builds llama.cpp v0.6.0 into ~/.cache/tantra-llm/llama.cpp
```

On macOS, `pip install -e ".[llm]"` needs the Xcode Command Line Tools (`xcode-select --install`). The build enables the Metal backend automatically and compiles the shaders into the library; `setup.sh` falls back to a CPU-only build when the Metal compiler is missing (`xcrun -f metal` fails). The quantize tools built by `quantize.sh` are CPU-only either way.

## Get the chat model

No weights are shipped in this repository. `fetch_model.py` downloads an official Qwen GGUF build, resumes interrupted downloads, and verifies the exact size, the SHA-256 and the GGUF magic before the file is put in place:

```bash
python fetch_model.py                # Qwen2.5-1.5B-Instruct · Q4_K_M · 1.04 GiB (default)
python fetch_model.py --size small   # Qwen2.5-0.5B-Instruct · Q4_K_M · 468 MiB
python fetch_model.py --list
```

The pins (`tantra/fetch.py`) were read from the Hugging Face tree API in October 2026:

| Option | File | Bytes | SHA-256 |
|---|---|---|---|
| `standard` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1,117,320,736 | `6a1a2eb6…34e9407e` |
| `small` | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 491,400,032 | `74a4da8c…d7a9db` |

Both come from `Qwen/Qwen2.5-1.5B-Instruct-GGUF` and `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (Apache-2.0). Point `app.py` at any other GGUF with `--model`.

## Usage

### 1. Find and download texts

```bash
python scraper.py search --limit 25                    # what would be accepted, and why others are not
python scraper.py search --show-rejected               # include rejected hits and their reasons
python scraper.py download --dry-run                   # plan only, no files written
python scraper.py download --identifier <archive-id>   # one item
python scraper.py download --limit 25 --delay 1.0      # everything accepted, polite rate
```

- **Language:** the item must declare Bengali (`Bengali`, `Bangla`, `ben`, `beng`, `bn`, …; stray punctuation like `Bengali.` is tolerated).
- **Date:** publication year between 1870 and 1935 (`--year-range` to change). `--allow-undated` both keeps undated items *and* drops the year filter from the search query, so such items are actually found.
- **Topic:** a Tantra or Sadhana keyword (English or Bengali) in the title or subject. The query searches the same fields.
- **Rights:** the default accepts items regardless of licence marking, because almost no real Bengali Tantra scan on archive.org carries a licence mark (in October 2026, one of the 78 items matching the default query was marked). The licence fields and a plain-language `rights_basis` are recorded in every item's `metadata.json`, so provenance travels with the text. `--require-marked-pd` restores the conservative accept-only-marked-public-domain behaviour. **Whatever the setting, checking the actual rights of what you redistribute remains your responsibility.**
- **Downloads:** `djvu_txt` (the OCR text) and `pdf` by default (`--formats djvu_txt,pdf,djvu` to add `.djvu`, which scanned items rarely carry). Files are streamed to a `.part` file, resumed over HTTP Range after an interrupted run, checked against the archive's published size and MD5, and only then moved into place. A mismatch raises and is retried; a pre-existing verified file is skipped, so re-running is cheap. Each downloaded item gets `data/raw/<id>/metadata.json` (title, year, languages, licence, rights basis, source URL, files, score). Every download attempt — downloaded, no-files, failed — is appended to `data/raw/manifest.jsonl`. Rejected hits are not written to disk; `search --show-rejected` lists them with reasons.
- **Search API errors are loud:** archive.org reports a bad query as `{"error": ...}` with HTTP 200; the client turns that into a visible failure instead of a silent empty result list.
- **Implementation:** the scraper talks to the three public archive.org endpoints directly over HTTPS (stdlib only — no SDK dependency): `advancedsearch.php` (paginated), `/metadata/<id>` (one cached request per item) and `/download/<id>/<file>`.

### 2. Clean the OCR

```bash
python cleaner.py correct --no-model                                        # zero-setup: structure only, no checkpoint
python cleaner.py train --corpus data/baseline --out models/denoiser.pt     # a clean Bengali corpus you supply (several paths are allowed)
python cleaner.py correct --checkpoint models/denoiser.pt                   # data/raw → data/processed/*.md
python cleaner.py calibration                                               # concatenate for llama-imatrix
python cleaner.py evaluate --checkpoint models/denoiser.pt --json reports/eval.json
```

Two paths through this stage:

- **Zero-setup (`--no-model`).** No checkpoint, no baseline corpus: Unicode normalisation, page splitting, page-number stripping and reflow into Markdown + plain text. A fresh checkout can go from `scraper.py download` to calibration data in one straight line.
- **With the denoiser.** A character-level, bidirectional tagger (see below) additionally repairs the Bengali lines. Train it on a clean corpus, or bootstrap it on the archive itself: correct once (with a baseline-trained or `--no-model` pass), then retrain with `--corpus data/baseline data/processed` — the corrected archive text acts as pseudo-clean domain data for a second generation. This self-training loop is wired and tested, but its *quality* is unproven (see Results); the honest default for a fresh user is `--no-model` plus calibration.

The denoiser is a character-level, bidirectional tagger in the style of GECToR. It does not generate text. For each character it predicts an edit: keep it, delete it, or replace it with a short string such as a missing hasanta. This matters because OCR errors often change the length of a word (a conjunct split into two letters, for example). A per-character tagger can express that directly, which a fixed-length masked-token model does not.

The **esoteric lexicon** (`tantra/data/esoteric_terms.tsv`) acts as an anchor. For every word within edit distance 2 of a lexicon term, the candidate that is in the lexicon gets a bonus of `--dict-bonus` nats (default 3.0). The anchor can be switched off with `--no-dictionary`. The lexicon is a 64-term seed list that needs curation by a specialist before it is relied on.

Training takes `--edit-weight` (default 1.0). Values above 1 up-weight the rare non-KEEP labels. The results below show that values above 1 raise recall but lower precision and CER, so leave it at the default unless you have measured otherwise.

`evaluate` reports, for several noise levels:

- **CER noisy / model / +dictionary**: character error rate against the clean text.
- **Edit recall (non-KEEP)**: the share of gold edit positions the model tags exactly. Plain label accuracy is inflated by the KEEP majority, so this is the number to watch.
- **Term recall**: the share of lexicon words restored exactly.

### 3. Quantize for the edge

```bash
./quantize.sh --hf-dir /path/to/Qwen2.5-1.5B-Instruct \
              --name tantra-qwen2.5-1.5b \
              --calibration data/calibration/calibration.txt
```

Output goes to `models/gguf/`: `<name>-Q4_K_M.gguf` and `<name>-Q3_K_L.gguf`, the i-matrix, and `SHA256SUMS`. The script checks each file's GGUF magic bytes. Other flags: `--types`, `--imatrix-chunks`, `--skip-imatrix`, `--threads`, `--drop-f16`, `--dry-run`. The llama.cpp version is pinned (`TANTRA_LLAMA_CPP_REF`, default `v0.6.0`).

### 4. Chat and benchmark

```bash
python app.py                                  # newest Q4_K_M model in models/gguf/
python app.py --prompt "তন্ত্র কী?"            # one answer and exit
python app.py --ground                         # ground answers in data/processed (retrieval + citations)
python app.py --ground /path/to/corpus --ground-chars 1600
python app.py --gpu-layers 0                   # force CPU (default: offload everything the GPU supports)
python benchmark.py --threads 2 --enforce      # exit 3 if peak RAM exceeds the budget
```

**Grounded mode** (`--ground`) is where the fetched archive texts enter the product: the processed corpus is indexed with BM25 over Bengali-aware tokens (pure Python, offline), the best-matching passages for each question are injected into the system prompt under a character budget (`--ground-chars`, default 2400), and the answer footer shows which documents were used (`[1] সাধনতত্ত্ব · [2] …`). The model is instructed to cite passages as `[1]`, `[2]` and to say plainly when the corpus does not cover the question. With no processed corpus the app warns and continues ungrounded.

Inside the chat, `/help`, `/reset`, `/stats` and `/exit` are available; `/stats` includes the corpus size in grounded mode. Threads default to 4 on hosts with 4 or more physical cores, otherwise 2. Each answer ends with a footer showing tokens per second and resident memory against the budget. Benchmark reports are written as JSON to `reports/`.

On Apple Silicon the runtime built by `setup.sh` includes the Metal backend, and `--gpu-layers` defaults to offloading every layer (`-1`) whenever the runtime supports GPU offload; on a CPU-only build the default is `0`. The load line and `/stats` show which device is in use.

## Memory budget

The target machine has 4 GB of RAM and no GPU. The operational budget there is **2.5 GiB** resident (`MEMORY_BUDGET_MB = 2560`), leaving the rest for the operating system. On larger hosts the advisory ceiling scales up (3 GiB at 8 GB RAM, 4 GiB at 16 GB) so the warning stays meaningful — Apple Silicon keeps GPU-resident weights in the same unified memory that RSS accounting sees. `--budget-mb` overrides the default.

The planned model is Qwen2.5-1.5B-Instruct quantized to Q4_K_M. A rough estimate is about 1.1 GB of weights, about 57 MB of KV cache at a 2048-token context, plus runtime overhead. That is an estimate. The measured figure comes from `benchmark.py` on the target hardware, and it is not in this repository yet.

## Data, licensing and rights

Read this before you publish or redistribute anything.

- **No text is shipped in this repository.** `data/` and `models/` are git-ignored. The scraper stores the metadata, licence, rights basis and source URL with every item, so provenance travels with the text.
- **The scraper accepts unmarked items by default.** Almost no Bengali Tantra scan on archive.org carries a licence mark, so a marked-only default would fetch nothing. The default policy is a practical choice, not a legal clearance: the licence fields and a plain-language `rights_basis` are recorded per item, and `--require-marked-pd` restores the strict behaviour. Checking the actual rights of anything you redistribute is your responsibility — colonial-era Indian imprints usually are public domain by age, but *usually* is not a licence.
- **Licence.** The code, the lexicon and the documentation are licensed under GPL-3.0-only (see `LICENSE`). The repository uses the same licence as the baseline corpus below, so it does not add a conflicting one.
- **Baseline corpus.** The denoiser is trained on eight news files from SUMono (`maamumin/SUMono`, GPL-3.0). The copy in `data/baseline/` is git-ignored and is not redistributed here. Whether a checkpoint counts as derived from that text is a legal question this project has not settled, so the conservative policy is to publish checkpoints only under GPL-3.0 and say so next to them.
- **Extra clean text (optional, not used by default).** `data/extra/` is git-ignored. The NER-Bangla dataset (`MISabic/NER-Bangla-Dataset`, MIT) provides 70,277 Bengali sentences, which were tried as extra clean text. Its README does not state where the sentences come from, so their rights are unverified. The MIT licence covers the dataset as published, not necessarily the underlying text. Check before using it.
- **Lexicon.** The lexicon is a hand-written seed list. It needs review by a specialist before it is relied on.

## Network and verification limits

Archive.org is now exercised live: CI runs a dedicated `live-archive` job on every push and pull request that searches the real archive, checks the acceptance contract, and downloads a stable item's OCR text with MD5 verification. Transient archive.org failures are retried.

Still not verifiable from the build environment:

- Hugging Face and the live Qwen download were unreachable when `tantra/fetch.py` was written. The pinned sizes and hashes were read from the Hub tree API, but the download itself was not run from the build sandbox.
- The Qwen2.5 conversion path in `quantize.sh` has not been run. The same script was verified end-to-end on a tiny random LLaMA with a SentencePiece tokenizer.
- Model quality and memory on the 1.5B model are unmeasured. They should be measured on the target hardware before any claim is made.
- macOS behaviour (bash 3.2, BSD `stat`/`shasum`, Apple Silicon Metal build) is exercised by CI on `macos-14` runners: the test suite runs `quantize.sh` end-to-end against stub tools there, and a separate job compiles `llama-cpp-python` from source exactly as `setup.sh` does.

## How the corruption is modelled

The denoiser is trained on synthetic OCR errors applied to clean text. The error types are:

- **Conjunct breaks:** a hasanta is dropped and a space inserted (`ক্ষ` → `ক ষ`). Default rate 0.20 per hasanta.
- **Confusions:** nine symmetric glyph pairs, including `র`/`ব`/`ক` and the vowel-sign pairs `ি`/`ী`, `ু`/`ূ`. Default rate 0.02 per confusable glyph.
- **Space splits:** a word of four or more letters is split by a space. Rate 0.04 per word.
- **Fading:** vowel signs and anusvara disappear. Rate 0.02 per mark.
- **Drops and bleed:** a glyph is removed (0.004) or a neighbouring glyph is duplicated (0.003).

**These rates and pairs are hand-picked assumptions, not measured OCR statistics.** They were chosen to cover the error types that are common in Bengali OCR. They need to be calibrated against real scans, and `--noise` scales the whole profile for sensitivity tests.

## Design decisions and deviations from the brief

- **Denoiser as a tagger, not a masked LM.** The brief suggested a bidirectional character-level model. Corruption changes length, so the model predicts per-character edit tags instead, and the tags come from alignment of corrupt and clean pairs.
- **Loss weighting.** Edits are about 3% of the labels, so an unweighted loss learns to keep every character. `--edit-weight` is the knob for this, but it is a trade-off between recall and precision. The best measured setting (2.0 with the extra corpus) helps only under heavy noise, and it costs precision. See Results.
- **Model size.** The denoiser is small: about 0.77 M parameters with the settings used in our training run (`--d-model 128 --layers 3 --heads 4 --max-len 160 --max-tags 1024`). That lets it train on a 4 GB CPU-only machine.
- **Fine-tuning and distillation are out of scope.** A 7B teacher needs a GPU and Hugging Face access, neither of which was available.
- **Base model.** Qwen2.5-1.5B-Instruct is used for chat because the instruct variant is needed. Qwen2.5-0.5B-Instruct is the lighter alternative.

## Results

### The denoiser does not improve OCR in any meaningful way yet

All models use the same model and schedule: 0.77 M parameters, 2,500 steps, batch 48, 2 CPU threads, seed 13, noise ×1 for training. Each is evaluated with `cleaner.py evaluate` on the same 368 held-out SUMono windows (44,544 clean characters) at noise ×1, the default corruption profile. These numbers come from the code at `cd9636c`. The edit-weight 1.0 baseline was re-trained after the environment was rebuilt and reproduced exactly. The 2.0, 3.0 and 5.0 SUMono runs were not repeated after the rebuild, because their checkpoints were lost in the restore. Their numbers come from logs written before it.

Earlier runs were affected by two train/eval mode bugs. Scoring left dropout on, so corrections drifted between runs. An interim checkpoint turned dropout off for the rest of training. Those checkpoints and numbers were discarded.

**Edit weight, SUMono only**

| `--edit-weight` | CER noisy | CER model | CER + dictionary | Edit precision | Edit recall | Predicted edits (gold ≈1,504) |
|---|---|---|---|---|---|---|
| 1.0 (default) | 0.0342 | 0.0341 | 0.0341 | 0.64 | 0.006 | 14 |
| 2.0 | 0.0342 | 0.0341 | 0.0341 | 0.59 | 0.009 | 22 |
| 3.0 | 0.0342 | 0.0355 | 0.0355 | 0.30 | 0.036 | 180 |
| 5.0 | 0.0342 | 0.0405 | 0.0406 | 0.26 | 0.121 | 694 |

**Extra corpus (NER-Bangla sentences), edit weight 1.0 and 2.0**

| Training data | `--edit-weight` | CER model | Edit precision | Edit recall | Predicted edits |
|---|---|---|---|---|---|
| SUMono (default) | 1.0 | 0.0341 | 0.64 | 0.006 | 14 |
| SUMono + NER-Bangla | 1.0 | 0.0341 | 1.00 | 0.002 | 3 |
| SUMono + NER-Bangla | 2.0 | 0.0340 | 0.54 | 0.019 | 54 |

**Paired comparison against the noisy input.** Per-window character errors were compared on the same windows. The 95% intervals come from a bootstrap over windows, and negative numbers mean fewer errors.

| Noise | Model | Character change (total) | 95% interval | Windows better / worse |
|---|---|---|---|---|
| ×1 | SUMono, 1.0 (default) | −5 | −12 to +2 | 9 / 4 |
| ×1 | SUMono + NER, 2.0 | −7 | −20 to +6 | 26 / 19 |
| ×2 | SUMono, 1.0 (default) | −8 | −15 to −2 | 10 / 2 |
| ×2 | SUMono + NER, 2.0 | −23 | −41 to −6 | 40 / 21 |

What these numbers show:

- **The default gains almost nothing.** At noise ×1 it changes total errors by −5 characters out of 44,544. At ×2 the change of −8 is significant, but it is still a very small improvement.
- **The best candidate helps only under heavy noise.** SUMono + NER at weight 2.0 is significantly better at ×2, but at ×1 the interval includes zero. Its precision is 0.54, so about half of its edits are wrong. It does not meet the bar of "CER improves and precision holds" that the default was held to.
- **Raising the weight trades precision for recall, and CER gets worse.** With SUMono alone, recall rises from 0.006 to 0.121 at weight 5.0, while precision falls from 0.64 to 0.26 and CER rises from 0.0341 to 0.0405.
- **The default model has learned the label prior.** About 96.6% of the evaluation labels are KEEP, meaning "leave this character alone". Predicting KEEP everywhere scores 96.61% label accuracy on those labels. The default model scores 96.62%.
- **The NER sentences did not help at the default weight.** At weight 1.0 they make the model almost never edit: 3 edits, against 14 with SUMono alone.
- **The dictionary anchor has no measurable effect on this synthetic set.** CER is the same with and without it, and term recall is unchanged. The synthetic noise rarely damages lexicon words, so this test cannot show the anchor's value.

The default `--edit-weight` stays at 1.0 with the SUMono baseline only. It is the most precise setting tried, and it does not make the text worse. But **the correction stage is experimental** and should not be trusted to fix OCR yet. The pipeline, scraping, quantization and chat app are the parts that work end to end. A useful denoiser probably needs more clean Bengali text than is available here, longer training, and noise levels calibrated against real scans rather than synthetic corruption.

## Development

```bash
ruff check . && ruff format --check .   # lint and formatting (ruff pinned to 0.17.0)
pytest -q                               # 600 offline tests (2 live tests: pytest -m live)
shellcheck quantize.sh setup.sh         # shell checks (the CI runner uses stock macOS bash 3.2 too)
```

CI runs the test suite on Linux (Python 3.11–3.13) and on macOS 14 (Apple Silicon), runs the live archive.org integration job, builds the sdist/wheel, and compiles the chat runtime from source on a macOS runner.

Layout:

```
tantra/            library: text, alignment, noise, dictionary, model, corrector, training,
                   pipeline, markdown, scraper (direct-HTTP archive.org client), retrieval
                   (BM25 corpus grounding), llm, bench, memory, ui
tantra/cli/        command-line front ends: scrape, clean, chat (app.py), bench (benchmark.py)
scripts/           make_tiny_model.py builds the offline test fixture (never ship its output)
tests/             pytest suite; tests/fakes.py holds the offline doubles,
                   test_archive_http_client.py runs the real client against a local
                   HTTP server that mirrors archive.org's response shapes,
                   test_live_archive.py (marker: live) hits the real archive.org
.github/workflows/ CI: lint, tests on Python 3.11–3.13, live archive.org integration,
                   package build and wheel smoke test
```

CI runs on every push to `main` and every pull request.
