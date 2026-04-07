"""Tests for the benchmark plugin registry and task_shape dispatch gates."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agents.benchmarks import REGISTRY, get_benchmark_class
from agents.benchmarks.base import Benchmark, BenchmarkConfig
from agents.benchmarks.swe_bench_verified import SWEBenchVerified


def test_swe_bench_verified_registered() -> None:
    assert "swe-bench-verified" in REGISTRY
    assert REGISTRY["swe-bench-verified"] is SWEBenchVerified


def test_get_benchmark_class_known_slug() -> None:
    assert get_benchmark_class("swe-bench-verified") is SWEBenchVerified


def test_get_benchmark_class_unknown_slug_raises() -> None:
    with pytest.raises(KeyError, match="swe-rebench|not registered|unknown"):
        get_benchmark_class("bogus-benchmark-xyz")


# ── task_shape dispatch gates (collector.py guards) ────────────────────


def _make_config(tmp_path: Path, *, repos_root: Path | None) -> BenchmarkConfig:
    """Build a minimal BenchmarkConfig for collector gate tests."""
    return BenchmarkConfig(
        slug="dummy-test",
        display_name="Dummy Test",
        harness_dataset="dummy/dummy",
        harness_split="test",
        data_root=tmp_path / "data",
        repos_root=repos_root,
        trace_root=tmp_path / "traces",
        default_max_steps=10,
        selection_n=1,
        selection_seed=0,
        docker_namespace=None,
    )


class _DummySwePatchBenchmark(Benchmark):
    slug = "dummy-swe"
    task_shape = "swe_patch"

    def load_tasks(self) -> list[dict[str, Any]]:
        return []

    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw


class _DummyFunctionCallBenchmark(Benchmark):
    slug = "dummy-fc"
    task_shape = "function_call"

    def load_tasks(self) -> list[dict[str, Any]]:
        return []

    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw


def test_swe_patch_with_repos_root_none_raises(tmp_path: Path) -> None:
    """A swe_patch benchmark with repos_root=None must still raise the
    legacy ValueError — this is the guardrail for SWE-bench misconfig."""
    from trace_collect.collector import collect_traces

    config = _make_config(tmp_path, repos_root=None)
    benchmark = _DummySwePatchBenchmark(config)

    with pytest.raises(ValueError, match="task_shape='swe_patch'.*repos_root"):
        asyncio.run(
            collect_traces(
                api_base="http://localhost",
                api_key="fake",
                model="test",
                benchmark=benchmark,
                scaffold="openclaw",
            )
        )


def test_function_call_task_shape_skips_repos_root_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A function_call benchmark with repos_root=None must NOT raise the
    repos_root ValueError — it should proceed past that gate and fail
    later (or succeed) for other reasons. We mock the downstream pipeline
    to isolate the guard under test.
    """
    from trace_collect import collector

    # Empty tasks.json (valid JSON) so load_tasks doesn't explode.
    tasks_path = tmp_path / "data" / "tasks.json"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text("[]\n", encoding="utf-8")

    config = _make_config(tmp_path, repos_root=None)
    benchmark = _DummyFunctionCallBenchmark(config)

    # Stub load_tasks to return [] so dispatch falls through with no work.
    monkeypatch.setattr(collector, "load_tasks", lambda _: [])

    # With zero tasks, openclaw path reaches _collect_openclaw. Stub that
    # so the test doesn't need a real LLM provider.
    async def _fake_collect_openclaw(**kwargs: Any) -> Path:
        return tmp_path / "traces" / "dummy_run"

    monkeypatch.setattr(collector, "_collect_openclaw", _fake_collect_openclaw)

    # Must not raise the repos_root ValueError.
    run_dir = asyncio.run(
        collector.collect_traces(
            api_base="http://localhost",
            api_key="fake",
            model="test",
            benchmark=benchmark,
            scaffold="openclaw",
        )
    )
    assert run_dir.name == "dummy_run"


def test_mini_swe_scaffold_rejects_function_call_benchmark(tmp_path: Path) -> None:
    """mini-swe-agent scaffold must refuse function_call benchmarks loudly."""
    from trace_collect.collector import collect_traces

    config = _make_config(tmp_path, repos_root=None)
    benchmark = _DummyFunctionCallBenchmark(config)

    with pytest.raises(ValueError, match="mini-swe-agent.*task_shape='swe_patch'"):
        asyncio.run(
            collect_traces(
                api_base="http://localhost",
                api_key="fake",
                model="test",
                benchmark=benchmark,
                scaffold="mini-swe-agent",
            )
        )
