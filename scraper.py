#!/usr/bin/env python3
"""Search archive.org for Bengali Tantra/Sadhana texts and download verified copies.

Examples:
    python scraper.py search --limit 50
    python scraper.py download --limit 10 --dry-run
    python scraper.py download --identifier <archive-id>
"""

from tantra.cli.scrape import main

if __name__ == "__main__":
    raise SystemExit(main())
