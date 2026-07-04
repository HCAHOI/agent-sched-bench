"""Runtime-selection tests for OpenClaw SWE collection."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.benchmarks import get_benchmark_class
from agents.benchmarks.base import BenchmarkConfig
from trace_collect.collector import collect_traces


def _make_rebench_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        slug="swe-rebench",
        display_name="SWE-rebench (filtered)",
        harness_dataset="nebius/SWE-rebench",
        harness_split="filtered",
        data_root=Path("data/swe-rebench"),
        repos_root=Path("data/swe-rebench/repos"),
        trace_root=Path("traces/swe-rebench"),
        default_max_iterations=100,
        selection_n=32,
        selection_seed=42,
        default_prompt_template="cc_aligned",
    )


def test_swe_rebench_openclaw_uses_host_agent_docker_tools() -> None:
    plugin = get_benchmark_class("swe-rebench")(_make_rebench_config())

    assert plugin.runtime_mode_for("openclaw") == "host_agent_docker_tools"


def test_swe_rebench_rejects_unsupported_scaffold() -> None:
    plugin = get_benchmark_class("swe-rebench")(_make_rebench_config())

    with pytest.raises(NotImplementedError):
        plugin.runtime_mode_for("unsupported")


def test_swe_rebench_normalize_task_pins_docker_image() -> None:
    plugin = get_benchmark_class("swe-rebench")(_make_rebench_config())

    normalized = plugin.normalize_task(
        {
            "instance_id": "encode__httpx-2701",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_fix"],
            "docker_image": "swerebench/sweb.eval.x86_64.encode_1776_httpx-2701:latest",
        }
    )

    assert normalized["image_name"] == (
        "swerebench/sweb.eval.x86_64.encode_1776_httpx-2701:latest"
    )


def test_collect_traces_rejects_non_task_container_runtime() -> None:
    benchmark = SimpleNamespace(
        validate_scaffold_support=lambda scaffold: None,
        runtime_mode_for=lambda scaffold: "unsupported",
        load_tasks=lambda: (_ for _ in ()).throw(AssertionError("should not load")),
        execution_environment="container",
        config=SimpleNamespace(default_prompt_template="default"),
    )

    with pytest.raises(NotImplementedError, match="Unsupported benchmark.runtime_mode_for"):
        asyncio.run(
            collect_traces(
                scaffold="openclaw",
                provider_name="openrouter",
                api_base="https://example.com",
                api_key="test-key",
                model="qwen-plus-latest",
                benchmark=benchmark,
                container_executable="docker",
            )
        )


def test_collect_traces_supports_host_controller_runner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "tb" / "trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text('{"type":"trace_metadata"}\n', encoding="utf-8")

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, *, container_executable: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, *, container_executable: False,
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda *, container_executable: None,
    )

    class FakeRunner:
        async def run_task(self, task, *, attempt_ctx, prompt_template):
            from trace_collect.attempt_pipeline import AttemptResult

            return AttemptResult(
                success=True,
                exit_status="completed",
                trace_path=trace_path,
                model_patch="",
            )

    benchmark = SimpleNamespace(
        validate_scaffold_support=lambda scaffold: None,
        runtime_mode_for=lambda scaffold: "host_controller",
        load_tasks=lambda: [{"instance_id": "tb-1"}],
        build_runner=lambda **kwargs: FakeRunner(),
        execution_environment="host",
        config=SimpleNamespace(
            slug="terminal-bench",
            default_prompt_template="default",
            trace_root=tmp_path / "traces",
            harness_split=None,
            selection_seed=42,
        ),
        image_name_for=lambda task: None,
    )

    run_dir = asyncio.run(
        collect_traces(
            scaffold="openclaw",
            provider_name="openrouter",
            api_base="https://example.com/v1",
            api_key="test-key",
            model="z-ai/glm-5.1",
            benchmark=benchmark,
            sample=1,
            min_free_disk_gb=0.001,
            container_executable=None,
        )
    )

    results_path = run_dir / "results.jsonl"
    payload = results_path.read_text(encoding="utf-8")
    assert '"success": true' in payload


def test_collect_traces_requires_explicit_container_runtime(
    tmp_path: Path,
) -> None:
    benchmark = SimpleNamespace(
        validate_scaffold_support=lambda scaffold: None,
        runtime_mode_for=lambda scaffold: "host_controller",
        load_tasks=lambda: [],
        build_runner=lambda **kwargs: None,
        execution_environment="container",
        config=SimpleNamespace(
            slug="terminal-bench",
            default_prompt_template="default",
            trace_root=tmp_path / "traces",
            harness_split=None,
        ),
        image_name_for=lambda task: None,
    )

    with pytest.raises(ValueError, match="--container required"):
        asyncio.run(
            collect_traces(
                scaffold="openclaw",
                provider_name="openrouter",
                api_base="https://example.com/v1",
                api_key="test-key",
                model="z-ai/glm-5.1",
                benchmark=benchmark,
                sample=1,
                min_free_disk_gb=0.001,
            )
        )


def test_collectors_max_iterations_defaults_to_100() -> None:
    assert inspect.signature(collect_traces).parameters["max_iterations"].default == 100
