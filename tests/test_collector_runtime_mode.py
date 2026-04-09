"""Tests for benchmark-owned runtime selection in collector orchestration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from trace_collect.attempt_pipeline import AttemptResult
from trace_collect.collector import _run_scaffold_tasks


def _write_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"type":"trace_metadata","scaffold":"miniswe","trace_format_version":5}\n',
        encoding="utf-8",
    )


def test_run_scaffold_tasks_uses_benchmark_prompt_default_and_runtime_mode(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    seen: dict[str, str] = {}

    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-rebench",
            harness_split="filtered",
            trace_root=tmp_path / "traces",
            default_prompt_template="cc_aligned",
        ),
        runtime_mode_for=lambda scaffold: "task_container_agent",
    )

    def make_inner(task: dict):
        async def inner(ctx) -> AttemptResult:
            seen["prompt_template"] = ctx.prompt_template
            seen["agent_runtime_mode"] = ctx.agent_runtime_mode
            return AttemptResult(
                success=True,
                exit_status="ok",
                trace_path=trace_path,
            )

        return inner

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[{"instance_id": "encode__httpx-2701", "image_name": "img"}],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="miniswe",
            prompt_template=None,
            min_free_disk_gb=0.001,
            inner_factory=make_inner,
        )
    )

    assert seen == {
        "prompt_template": "cc_aligned",
        "agent_runtime_mode": "task_container_agent",
    }


def test_run_scaffold_tasks_prompt_override_stays_independent_of_runtime_mode(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    seen: dict[str, str] = {}

    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-rebench",
            harness_split="filtered",
            trace_root=tmp_path / "traces",
            default_prompt_template="cc_aligned",
        ),
        runtime_mode_for=lambda scaffold: "task_container_agent",
    )

    def make_inner(task: dict):
        async def inner(ctx) -> AttemptResult:
            seen["prompt_template"] = ctx.prompt_template
            seen["agent_runtime_mode"] = ctx.agent_runtime_mode
            return AttemptResult(
                success=True,
                exit_status="ok",
                trace_path=trace_path,
            )

        return inner

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[{"instance_id": "encode__httpx-2701", "image_name": "img"}],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="miniswe",
            prompt_template="default",
            min_free_disk_gb=0.001,
            inner_factory=make_inner,
        )
    )

    assert seen == {
        "prompt_template": "default",
        "agent_runtime_mode": "task_container_agent",
    }
