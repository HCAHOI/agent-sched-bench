"""Tests for the benchmark plugin registry."""

from __future__ import annotations

import pytest

from agents.benchmarks import REGISTRY, get_benchmark_class
from agents.benchmarks.swe_bench_verified import SWEBenchVerified


def test_swe_bench_verified_registered() -> None:
    assert "swe-bench-verified" in REGISTRY
    assert REGISTRY["swe-bench-verified"] is SWEBenchVerified


def test_get_benchmark_class_known_slug() -> None:
    assert get_benchmark_class("swe-bench-verified") is SWEBenchVerified


def test_get_benchmark_class_unknown_slug_raises() -> None:
    with pytest.raises(KeyError, match="swe-rebench|not registered|unknown"):
        get_benchmark_class("bogus-benchmark-xyz")
