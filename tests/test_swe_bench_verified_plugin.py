"""Tests for the SWEBenchVerified benchmark plugin.

Fast tests run unconditionally.  The ``@pytest.mark.skipif`` test that hits
HuggingFace is gated behind the ``HF_HUB_ONLINE`` environment variable so
the CI suite stays offline by default.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agents.benchmarks.base import BenchmarkConfig
from agents.benchmarks.swe_bench_verified import SWEBenchVerified


def _make_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        slug="swe-bench-verified",
        display_name="SWE-bench Verified",
        harness_dataset="princeton-nlp/SWE-bench_Verified",
        harness_split="test",
        data_root=Path("data/swebench_verified"),
        repos_root=None,
        trace_root=Path("traces/swebench_verified"),
        default_max_iterations=80,
        selection_n=32,
        selection_seed=42,
        docker_namespace="swebench",
    )


@pytest.fixture()
def plugin() -> SWEBenchVerified:
    return SWEBenchVerified(_make_config())


# ---------------------------------------------------------------------------
# Fast tests — no network required
# ---------------------------------------------------------------------------


def test_plugin_slug_and_task_shape(plugin: SWEBenchVerified) -> None:
    assert plugin.slug == "swe-bench-verified"
    assert plugin.task_shape == "swe_patch"


def test_derive_test_cmd_accepts_native_list(plugin: SWEBenchVerified) -> None:
    task = {
        "FAIL_TO_PASS": ["tests/a.py::test_x", "tests/b.py::test_y"],
    }
    result = plugin.derive_test_cmd(task)
    assert "tests/a.py::test_x" in result
    assert "tests/b.py::test_y" in result
    assert "pytest" in result


def test_derive_test_cmd_accepts_json_string(plugin: SWEBenchVerified) -> None:
    task = {
        "FAIL_TO_PASS": '["tests/a.py::test_x"]',
    }
    result = plugin.derive_test_cmd(task)
    assert "tests/a.py::test_x" in result
    assert "pytest" in result


def test_build_runner_returns_swebench_runner(plugin: SWEBenchVerified) -> None:
    from agents.openclaw.eval.runner import SWEBenchRunner
    from agents.openclaw.providers.base import LLMProvider

    # Use a minimal stub provider — we're only checking the return type.
    class _StubProvider(LLMProvider):
        def get_default_model(self) -> str:
            return "stub-model"

        async def chat(  # type: ignore[override]
            self,
            messages,
            tools=None,
            model=None,
            max_tokens=4096,
            temperature=0.7,
            reasoning_effort=None,
            tool_choice=None,
        ):
            raise NotImplementedError

    runner = plugin.build_runner(
        scaffold="openclaw",
        provider=_StubProvider(),
        workspace_base=Path("/tmp/ws"),
        max_iterations=5,
        context_window_tokens=8192,
        model="stub-model",
    )
    assert isinstance(runner, SWEBenchRunner)
    # Plugin self-injects repos_root and benchmark from its own config.
    assert runner.repos_root == plugin.config.repos_root
    assert runner.benchmark is plugin


# ---------------------------------------------------------------------------
# Slow / online test — skipped unless HF_HUB_ONLINE is set
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("HF_HUB_ONLINE"),
    reason="HF fetch skipped by default; set HF_HUB_ONLINE=1 to enable",
)
def test_load_tasks_matches_legacy(plugin: SWEBenchVerified) -> None:
    from agents.swebench_data import load_swebench_verified

    plugin_tasks = plugin.load_tasks()
    legacy_tasks = load_swebench_verified()

    plugin_ids = {t["instance_id"] for t in plugin_tasks}
    legacy_ids = {t["instance_id"] for t in legacy_tasks}
    assert plugin_ids == legacy_ids
