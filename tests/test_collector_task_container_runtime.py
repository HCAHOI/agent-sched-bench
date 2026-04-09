"""Tests for collector task-container runtime helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from trace_collect.attempt_pipeline import AttemptContext
from trace_collect.collector import (
    _run_miniswe_in_task_container,
    _run_openclaw_in_task_container,
)
from trace_collect.runtime.task_container import (
    TaskContainerPreflightProof,
    TaskContainerRunResult,
)


def _make_ctx(tmp_path: Path, *, scaffold: str) -> AttemptContext:
    return AttemptContext(
        run_dir=tmp_path / "run",
        instance_id="encode__httpx-2701",
        attempt=1,
        task={
            "instance_id": "encode__httpx-2701",
            "repo": "encode/httpx",
            "base_commit": "deadbeef",
            "problem_statement": "Fix bug",
            "image_name": "swerebench/example",
        },
        model="qwen-plus-latest",
        scaffold=scaffold,
        source_image="swerebench/example",
        prompt_template="cc_aligned",
        agent_runtime_mode="task_container_agent",
    )


def _make_relative_ctx(monkeypatch, tmp_path: Path, *, scaffold: str) -> AttemptContext:
    monkeypatch.chdir(tmp_path)
    return AttemptContext(
        run_dir=Path("run"),
        instance_id="encode__httpx-2701",
        attempt=1,
        task={
            "instance_id": "encode__httpx-2701",
            "repo": "encode/httpx",
            "base_commit": "deadbeef",
            "problem_statement": "Fix bug",
            "image_name": "swerebench/example",
        },
        model="qwen-plus-latest",
        scaffold=scaffold,
        source_image="swerebench/example",
        prompt_template="cc_aligned",
        agent_runtime_mode="task_container_agent",
    )


def test_run_miniswe_in_task_container_collects_runtime_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}
    ctx = _make_relative_ctx(monkeypatch, tmp_path, scaffold="miniswe")
    runtime_dir = ctx.attempt_dir / "_task_container_runtime" / "miniswe"
    stdout_path = runtime_dir / "stdout.txt"
    stderr_path = runtime_dir / "stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("mini stdout", encoding="utf-8")
    stderr_path.write_text("mini stderr", encoding="utf-8")
    trace_path = runtime_dir / "trace.jsonl"
    trace_path.write_text(
        '{"type":"trace_metadata","scaffold":"miniswe","trace_format_version":5}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "trace_collect.collector.start_task_container",
        lambda *args, **kwargs: "cid-mini",
    )
    monkeypatch.setattr(
        "trace_collect.collector.stop_task_container",
        lambda *args, **kwargs: "container logs",
    )
    monkeypatch.setattr(
        "trace_collect.collector.preflight_task_container_runtime",
        lambda **kwargs: TaskContainerPreflightProof(
            container_id="cid-mini",
            hostname="host-a",
            cwd="/testbed",
            python_executable="/opt/conda/envs/ML/bin/python",
            project_root="/work/project",
            python_prefix="/opt/conda/envs/ML",
            sys_path=["/work/project/src"],
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.run_task_container_agent",
        lambda **kwargs: (
            seen.update(kwargs["request"]),
            TaskContainerRunResult(
                success=True,
                trace_path=trace_path,
                model_patch="diff --git a/x b/x",
                exit_status="Submitted",
                error=None,
                n_iterations=7,
                total_llm_ms=1.0,
                total_tool_ms=2.0,
                total_tokens=3,
                runtime_proof={"hostname": "container-a"},
                raw_stdout_path=stdout_path,
                raw_stderr_path=stderr_path,
            ),
        )[1],
    )

    result = asyncio.run(
        _run_miniswe_in_task_container(
            ctx=ctx,
            task=dict(ctx.task),
            benchmark=SimpleNamespace(
                config=SimpleNamespace(slug="swe-rebench", harness_split="filtered")
            ),
            api_base="https://example.com",
            api_key="test-key",
            model="qwen-plus-latest",
            max_iterations=10,
            command_timeout_s=60.0,
            task_timeout_s=120.0,
            max_context_tokens=1024,
        )
    )

    assert result.success is True
    assert result.exit_status == "Submitted"
    assert result.runtime_proof["container_id"] == "cid-mini"
    assert result.runtime_proof["python_executable"] == "/opt/conda/envs/ML/bin/python"
    assert seen["kind"] == "run_miniswe"
    assert Path(str(seen["result_path"])).is_absolute()
    assert Path(str(seen["trace_file"])).is_absolute()
    assert Path(str(seen["raw_stdout_path"])).is_absolute()
    assert Path(str(seen["raw_stderr_path"])).is_absolute()
    assert Path(str(seen["result_path"])) == runtime_dir.resolve() / "run.result.json"
    assert seen["exec_working_dir"] == "/testbed"
    assert "mini stdout" in ctx.container_stdout
    assert "container logs" in ctx.container_stdout


def test_run_miniswe_in_task_container_keeps_raw_logs_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ctx = _make_ctx(tmp_path, scaffold="miniswe")
    runtime_dir = ctx.attempt_dir / "_task_container_runtime" / "miniswe"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = runtime_dir / "stdout.txt"
    stderr_path = runtime_dir / "stderr.txt"
    stdout_path.write_text("mini stdout", encoding="utf-8")
    stderr_path.write_text("mini stderr", encoding="utf-8")

    monkeypatch.setattr(
        "trace_collect.collector.start_task_container",
        lambda *args, **kwargs: "cid-mini",
    )
    monkeypatch.setattr(
        "trace_collect.collector.stop_task_container",
        lambda *args, **kwargs: "container logs",
    )
    monkeypatch.setattr(
        "trace_collect.collector.preflight_task_container_runtime",
        lambda **kwargs: TaskContainerPreflightProof(
            container_id="cid-mini",
            hostname="host-a",
            cwd="/testbed",
            python_executable="/opt/conda/envs/ML/bin/python",
            project_root="/work/project",
            python_prefix="/opt/conda/envs/ML",
            sys_path=["/work/project/src"],
        ),
    )

    def fail_run(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "trace_collect.collector.run_task_container_agent",
        fail_run,
    )

    try:
        asyncio.run(
            _run_miniswe_in_task_container(
                ctx=ctx,
                task=dict(ctx.task),
                benchmark=SimpleNamespace(
                    config=SimpleNamespace(slug="swe-rebench", harness_split="filtered")
                ),
                api_base="https://example.com",
                api_key="test-key",
                model="qwen-plus-latest",
                max_iterations=10,
                command_timeout_s=60.0,
                task_timeout_s=120.0,
                max_context_tokens=1024,
            )
        )
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("expected runtime failure")

    assert "mini stdout" in ctx.container_stdout
    assert "mini stderr" in ctx.container_stdout
    assert "container logs" in ctx.container_stdout


def test_run_openclaw_in_task_container_normalizes_trace_on_host(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}
    ctx = _make_relative_ctx(monkeypatch, tmp_path, scaffold="openclaw")
    runtime_dir = ctx.attempt_dir / "_task_container_runtime" / "openclaw"
    stdout_path = runtime_dir / "stdout.txt"
    stderr_path = runtime_dir / "stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("openclaw stdout", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    raw_trace_path = runtime_dir / "trace.raw.jsonl"
    raw_trace_path.write_text(
        json.dumps(
            {
                "type": "trace_metadata",
                "scaffold": "openclaw",
                "trace_format_version": 5,
                "model": "qwen-plus-latest",
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "action",
                "action_type": "llm_call",
                "action_id": "llm_0",
                "agent_id": "encode__httpx-2701",
                "iteration": 0,
                "ts_start": 1.0,
                "ts_end": 2.0,
                "data": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "trace_collect.collector.start_task_container",
        lambda *args, **kwargs: "cid-openclaw",
    )
    monkeypatch.setattr(
        "trace_collect.collector.stop_task_container",
        lambda *args, **kwargs: "container logs",
    )
    monkeypatch.setattr(
        "trace_collect.collector.preflight_task_container_runtime",
        lambda **kwargs: TaskContainerPreflightProof(
            container_id="cid-openclaw",
            hostname="host-b",
            cwd="/testbed",
            python_executable="/opt/conda/envs/ML/bin/python",
            project_root="/work/project",
            python_prefix="/opt/conda/envs/ML",
            sys_path=["/work/project/src"],
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.run_task_container_agent",
        lambda **kwargs: (
            seen.update(kwargs["request"]),
            TaskContainerRunResult(
                success=True,
                trace_path=raw_trace_path,
                model_patch="diff --git a/httpx.py b/httpx.py",
                exit_status="completed",
                error=None,
                n_iterations=4,
                total_llm_ms=12.0,
                total_tool_ms=6.0,
                total_tokens=99,
                runtime_proof={"hostname": "container-b"},
                raw_stdout_path=stdout_path,
                raw_stderr_path=stderr_path,
            ),
        )[1],
    )

    result = asyncio.run(
        _run_openclaw_in_task_container(
            ctx=ctx,
            task=dict(ctx.task),
            benchmark=SimpleNamespace(
                config=SimpleNamespace(slug="swe-rebench", harness_split="filtered")
            ),
            api_base="https://example.com",
            api_key="test-key",
            model="qwen-plus-latest",
            max_iterations=10,
            max_context_tokens=1024,
            mcp_config=None,
        )
    )

    metadata = json.loads((ctx.attempt_dir / "trace.jsonl").read_text().splitlines()[0])
    assert result.trace_path == ctx.attempt_dir / "trace.jsonl"
    assert metadata["prompt_template"] == "cc_aligned"
    assert metadata["agent_runtime_mode"] == "task_container_agent"
    assert metadata["runtime_proof"]["container_id"] == "cid-openclaw"
    assert seen["kind"] == "run_openclaw"
    assert Path(str(seen["result_path"])).is_absolute()
    assert Path(str(seen["workspace_base"])).is_absolute()
    assert Path(str(seen["workspace_dir"])).is_absolute()
    assert Path(str(seen["trace_file"])).is_absolute()
    assert Path(str(seen["raw_stdout_path"])).is_absolute()
    assert Path(str(seen["raw_stderr_path"])).is_absolute()
    assert Path(str(seen["result_path"])) == runtime_dir.resolve() / "run.result.json"
    assert seen["tool_workspace"] == "/testbed"
    assert seen["exec_working_dir"] == "/testbed"
    assert result.total_llm_ms == 12.0
    assert result.total_tool_ms == 6.0
    assert result.total_tokens == 99
    assert "openclaw stdout" in ctx.container_stdout
