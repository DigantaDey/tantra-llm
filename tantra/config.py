"""Project-wide defaults. Everything here can be overridden from the command line."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
BASELINE_DIR = DATA_DIR / "baseline"
PROCESSED_DIR = DATA_DIR / "processed"
CALIBRATION_DIR = DATA_DIR / "calibration"
MODELS_DIR = Path("models")
DENOISER_CHECKPOINT = MODELS_DIR / "denoiser.pt"
GGUF_DIR = MODELS_DIR / "gguf"
REPORTS_DIR = Path("reports")

# Memory ceiling for the deployed model: 2.5 GiB resident (the 4 GB host keeps the rest for OS and runtime).
MEMORY_BUDGET_MB = 2560

# Default base model for the GGUF conversion (any HF causal LM with a supported architecture works).
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
