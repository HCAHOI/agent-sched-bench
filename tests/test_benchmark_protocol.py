"""Benchmark protocol contract tests — PM-2 mitigation.

The default build_runner must raise NotImplementedError. A BFCL-v4-like
function-call benchmark must satisfy the protocol without touching
SWE-specific fields.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from agents.benchmarks.base import Benchmark, BenchmarkConfig


def _make_config(slug: str) -> BenchmarkConfig:
    return BenchmarkConfig(
        slug=slug,
        display_name=f"Test {slug}",
        harness_dataset="test/ds",
        harness_split="test",
        data_root=Path(f"data/{slug}"),
        repos_root=None,
        trace_root=Path(f"traces/{slug}"),
        default_max_iterations=10,
        selection_n=2,
        selection_seed=0,
    )


class MockFunctionCallBenchmark(Benchmark):
    slug = "mock-fc"
    task_shape = "function_call"

    def load_tasks(self) -> list[dict]:
        return [{"instance_id": "t1"}, {"instance_id": "t2"}]

    def normalize_task(self, raw: dict) -> dict:
        return raw


def test_default_build_runner_raises_not_implemented() -> None:
    plugin = MockFunctionCallBenchmark(_make_config("mock-fc"))
    with pytest.raises(NotImplementedError, match="mock-fc"):
        plugin.build_runner(scaffold="miniswe")


def test_mock_function_call_benchmark_satisfies_protocol_without_repos_root() -> None:
    """A non-SWE benchmark can be instantiated with repos_root=None and
    satisfies the load_tasks + normalize_task contract."""
    plugin = MockFunctionCallBenchmark(_make_config("mock-fc"))
    assert plugin.config.repos_root is None
    assert plugin.task_shape == "function_call"
    tasks = plugin.load_tasks()
    assert len(tasks) == 2
    normalized = plugin.normalize_task({"instance_id": "x"})
    assert normalized == {"instance_id": "x"}
