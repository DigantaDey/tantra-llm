#!/usr/bin/env python3
"""Download a ready-made GGUF chat model into models/gguf (verified by size and SHA-256).

Examples:
    python fetch_model.py                      # Qwen2.5-1.5B-Instruct · Q4_K_M · 1.04 GiB
    python fetch_model.py --size small         # Qwen2.5-0.5B-Instruct · Q4_K_M · 468 MiB
    python fetch_model.py --list
"""

from tantra.cli.fetch import main

if __name__ == "__main__":
    raise SystemExit(main())
