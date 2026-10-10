#!/usr/bin/env python3
"""Bengali OCR correction: train the denoiser, correct archive texts, build calibration data.

Examples:
    python cleaner.py train --corpus data/baseline --steps 4000
    python cleaner.py correct --input data/raw --out-dir data/processed
    python cleaner.py calibration
    python cleaner.py evaluate --noise 0.5 1 2
"""

from tantra.cli.clean import main

if __name__ == "__main__":
    raise SystemExit(main())
