#!/usr/bin/env python3
"""Benchmark speed and peak RAM of a GGUF model against the 2.5 GB budget.

Example:
    python benchmark.py --model models/gguf/tantra-qwen2.5-1.5b-Q4_K_M.gguf --threads 4 --enforce
"""

from tantra.cli.bench import main

if __name__ == "__main__":
    raise SystemExit(main())
