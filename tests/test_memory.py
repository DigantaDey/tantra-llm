"""Memory helpers used by the benchmark and the chat footer."""

from __future__ import annotations

import os

from tantra.memory import available_mb, physical_cores, rss_mb, total_mb


def test_own_process_rss_is_positive_and_plausible():
    rss = rss_mb()
    assert 1 < rss < total_mb()  # a Python process takes more than 1 MiB and less than the whole machine


def test_rss_of_this_process_matches_explicit_pid():
    assert abs(rss_mb(os.getpid()) - rss_mb()) < 64  # same process; allow for allocation between reads


def test_system_memory_figures_are_consistent():
    total = total_mb()
    assert total > 0
    assert 0 < available_mb() <= total


def test_physical_cores_is_at_least_one():
    assert physical_cores() >= 1
