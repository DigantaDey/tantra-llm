#!/usr/bin/env python3
"""Chat with the local Tantra LLM (CPU-only GGUF via llama-cpp-python).

Examples:
    python app.py
    python app.py --threads 2 --prompt "নিত্যপূজা কী?"
"""

from tantra.cli.chat import main

if __name__ == "__main__":
    raise SystemExit(main())
