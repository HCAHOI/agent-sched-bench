"""Runtime-selection tests for OpenClaw SWE collection."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.benchmarks import get_benchmark_class
from agents.benchmarks.base import BenchmarkConfig
from trace_collect.collector import collect_openclaw_traces


def _make_verified_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        slug="swe-bench-verified",
        display_name="SWE-Bench Verified",
        harness_dataset="princeton-nlp/SWE-bench_Verified",
        harness_split="test",
        data_root=Path("data/swebench_verified"),
        repos_root=Path("data/swebench_repos"),
        trace_root=Path("traces/swebench_verified"),
        default_max_iterations=50,
        selection_n=32,
        selection_seed=42,
        default_prompt_template="default",
    )


def test_swe_bench_verified_openclaw_uses_task_container_agent() -> None:
    plugin = get_benchmark_class("swe-bench-verified")(_make_verified_config())

    assert plugin.runtime_mode_for("openclaw") == "task_container_agent"


def test_swe_bench_verified_miniswe_stays_on_host_controller() -> None:
    plugin = get_benchmark_class("swe-bench-verified")(_make_verified_config())

    assert plugin.runtime_mode_for("miniswe") == "host_controller"


def test_swe_bench_verified_normalize_task_derives_image_name() -> None:
    plugin = get_benchmark_class("swe-bench-verified")(_make_verified_config())

    normalized = plugin.normalize_task(
        {
            "instance_id": "Kinto__kinto-http.py-384",
            "FAIL_TO_PASS": "[]",
        }
    )

    assert normalized["image_name"] == (
        "docker.io/swebench/sweb.eval.x86_64.kinto_1776_kinto-http.py-384:latest"
    )


def test_collect_openclaw_traces_rejects_non_task_container_runtime() -> None:
    benchmark = SimpleNamespace(
        runtime_mode_for=lambda scaffold: "host_controller",
        load_tasks=lambda: (_ for _ in ()).throw(AssertionError("should not load")),
        config=SimpleNamespace(default_prompt_template="default"),
    )

    with pytest.raises(NotImplementedError, match="task_container_agent"):
        asyncio.run(
            collect_openclaw_traces(
                api_base="https://example.com",
                api_key="test-key",
                model="qwen-plus-latest",
                benchmark=benchmark,
            )
        )
