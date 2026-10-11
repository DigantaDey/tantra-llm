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


def default_budget_mb(total_mb: float | None = None) -> float:
    """Advisory RSS ceiling: 2.5 GiB on the 4 GB target machine, more headroom on larger hosts.

    GPU offload (Metal on Apple Silicon) keeps weights in the same unified-memory pool that
    RSS accounting sees, so a 16 GB Mac needs a higher ceiling before the warning means anything.
    """
    if total_mb is None:
        from tantra.memory import total_mb as system_total_mb

        total_mb = system_total_mb()
    if total_mb <= 5.5 * 1024:
        return float(MEMORY_BUDGET_MB)
    if total_mb <= 9.5 * 1024:
        return 3072.0
    return 4096.0


# Default base model for the GGUF conversion (any HF causal LM with a supported architecture works).
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
