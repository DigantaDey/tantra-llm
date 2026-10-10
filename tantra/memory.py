"""Resident-memory accounting for the 4 GB target machine."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import psutil


def rss_mb(pid: int | None = None) -> float:
    """Resident set size of a process in MiB (includes memory-mapped model pages once touched)."""
    return psutil.Process(pid).memory_info().rss / (1024 * 1024)


def available_mb() -> float:
    return psutil.virtual_memory().available / (1024 * 1024)


def total_mb() -> float:
    return psutil.virtual_memory().total / (1024 * 1024)


def physical_cores() -> int:
    return psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1


@dataclass
class PeakMonitor:
    """Samples RSS in a background thread and remembers the peak. Use as a context manager."""

    interval: float = 0.05
    samples: list[float] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(rss_mb())
            time.sleep(self.interval)

    def __enter__(self) -> PeakMonitor:
        self._stop.clear()
        self.samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.samples.append(rss_mb())  # final reading after the block

    @property
    def peak_mb(self) -> float:
        return max(self.samples) if self.samples else rss_mb()
