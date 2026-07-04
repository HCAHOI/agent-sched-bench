"""Tests for collector task-container runtime helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from trace_collect.attempt_pipeline import AttemptContext
from trace_collect.collector import (
    _run_openclaw_in_task_container,
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
        agent_runtime_mode="host_agent_docker_tools",
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
        agent_runtime_mode="host_agent_docker_tools",
    )


def test_run_openclaw_in_task_container_normalizes_trace_on_host(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ctx = _make_relative_ctx(monkeypatch, tmp_path, scaffold="openclaw")
    runtime_dir = ctx.attempt_dir / "_task_container_runtime" / "openclaw"
    stdout_path = runtime_dir / "stdout.txt"
    stderr_path = runtime_dir / "stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("openclaw stdout", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    trace_path = ctx.attempt_dir / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"type": "trace_metadata", "scaffold": "openclaw",
                     "trace_format_version": 5, "model": "qwen-plus-latest"})
        + "\n"
        + json.dumps({"type": "action", "action_type": "llm_call",
                       "action_id": "llm_0", "agent_id": "encode__httpx-2701",
                       "iteration": 0, "ts_start": 1.0, "ts_end": 2.0, "data": {}})
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

    from agents.openclaw._session_runner import SessionRunResult

    async def fake_session_run(self, prompt, workspace, *,
        tool_workspace=None, session_key="", trace_file=None,
        runtime_dir=None, instance_id=None, channel="cli", prepare_ms=None,
    ):
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text(
            '{"type":"trace_metadata","scaffold":"openclaw","trace_format_version":5}\n',
            encoding="utf-8",
        )
        return SessionRunResult(
            content="Task completed.",
            elapsed_s=1.0,
            trace_file=trace_file,
            session_key=session_key,
            stop_reason="completed",
            error=None,
        )

    monkeypatch.setattr(
        "trace_collect.collector.SessionRunner.run",
        fake_session_run,
    )

    monkeypatch.setattr(
        "agents.openclaw.eval.runner.SWEBenchRunner._extract_container_patch",
        lambda diff_cwd, *, base_commit, run: "diff --git a/main.py b/main.py\n+fix",
    )

    result = asyncio.run(
        _run_openclaw_in_task_container(
            ctx=ctx,
            task=dict(ctx.task),
            benchmark=SimpleNamespace(
                execution_environment="container",
                config=SimpleNamespace(slug="swe-rebench", harness_split="filtered"),
            ),
            container_executable="docker",
            provider_name="openrouter",
            api_base="https://example.com",
            api_key="test-key",
            model="qwen-plus-latest",
            max_iterations=10,
            generation_config=None,
            max_context_tokens=1024,
            mcp_config=None,
        )
    )

    metadata = json.loads((ctx.attempt_dir / "trace.jsonl").read_text().splitlines()[0])
    assert result.trace_path == ctx.attempt_dir / "trace.jsonl"
    assert metadata["prompt_template"] == "cc_aligned"
    assert metadata["agent_runtime_mode"] == "host_agent_docker_tools"
    assert metadata["runtime_proof"]["container_id"] == "cid-openclaw"
    assert result.success is True
    assert result.exit_status == "completed"
    assert "container logs" in ctx.container_stdout


def test_run_openclaw_in_task_container_adds_mcp_bootstrap_requirements(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ctx = _make_relative_ctx(monkeypatch, tmp_path, scaffold="openclaw")
    runtime_dir = ctx.attempt_dir / "_task_container_runtime" / "openclaw"
    stdout_path = runtime_dir / "stdout.txt"
    stderr_path = runtime_dir / "stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("openclaw stdout", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    trace_path = ctx.attempt_dir / "trace.jsonl"
    trace_path.write_text(
        '{"type":"trace_metadata","scaffold":"openclaw","trace_format_version":5}\n',
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

    from agents.openclaw._session_runner import SessionRunResult

    async def fake_session_run(self, prompt, workspace, *,
        tool_workspace=None, session_key="", trace_file=None,
        runtime_dir=None, instance_id=None, channel="cli", prepare_ms=None,
    ):
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text(
            '{"type":"trace_metadata","scaffold":"openclaw","trace_format_version":5}\n',
            encoding="utf-8",
        )
        return SessionRunResult(
            content="Task completed.",
            elapsed_s=1.0,
            trace_file=trace_file,
            session_key=session_key,
            stop_reason="completed",
            error=None,
        )

    monkeypatch.setattr(
        "trace_collect.collector.SessionRunner.run",
        fake_session_run,
    )
    monkeypatch.setattr(
        "trace_collect.collector.load_mcp_servers",
        lambda config: {},
    )
    monkeypatch.setattr(
        "agents.openclaw.eval.runner.SWEBenchRunner._extract_container_patch",
        lambda diff_cwd, *, base_commit, run: None,
    )

    asyncio.run(
        _run_openclaw_in_task_container(
            ctx=ctx,
            task=dict(ctx.task),
            benchmark=SimpleNamespace(
                execution_environment="container",
                config=SimpleNamespace(slug="swe-rebench", harness_split="filtered"),
            ),
            container_executable="docker",
            provider_name="openrouter",
            api_base="https://example.com",
            api_key="test-key",
            model="qwen-plus-latest",
            max_iterations=10,
            generation_config=None,
            max_context_tokens=1024,
            mcp_config="configs/mcp/context7.yaml",
        )
    )


def test_success_false_when_completed_but_no_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ctx = _make_relative_ctx(monkeypatch, tmp_path, scaffold="openclaw")
    runtime_dir = ctx.attempt_dir / "_task_container_runtime" / "openclaw"
    stdout_path = runtime_dir / "stdout.txt"
    stderr_path = runtime_dir / "stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("openclaw stdout", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    trace_path = ctx.attempt_dir / "trace.jsonl"
    trace_path.write_text(
        '{"type":"trace_metadata","scaffold":"openclaw","trace_format_version":5}\n',
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

    from agents.openclaw._session_runner import SessionRunResult

    async def fake_session_run(self, prompt, workspace, *,
        tool_workspace=None, session_key="", trace_file=None,
        runtime_dir=None, instance_id=None, channel="cli", prepare_ms=None,
    ):
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text(
            '{"type":"trace_metadata","scaffold":"openclaw","trace_format_version":5}\n',
            encoding="utf-8",
        )
        return SessionRunResult(
            content="Task completed.",
            elapsed_s=1.0,
            trace_file=trace_file,
            session_key=session_key,
            stop_reason="completed",
            error=None,
        )

    monkeypatch.setattr(
        "trace_collect.collector.SessionRunner.run",
        fake_session_run,
    )

    # Return None → model_patch stays "" → success=False
    monkeypatch.setattr(
        "agents.openclaw.eval.runner.SWEBenchRunner._extract_container_patch",
        lambda diff_cwd, *, base_commit, run: None,
    )

    result = asyncio.run(
        _run_openclaw_in_task_container(
            ctx=ctx,
            task=dict(ctx.task),
            benchmark=SimpleNamespace(
                execution_environment="container",
                config=SimpleNamespace(slug="swe-rebench", harness_split="filtered"),
            ),
            container_executable="docker",
            provider_name="openrouter",
            api_base="https://example.com",
            api_key="test-key",
            model="qwen-plus-latest",
            max_iterations=10,
            generation_config=None,
            max_context_tokens=1024,
            mcp_config=None,
        )
    )

    assert result.success is False
    assert result.exit_status == "completed"
    assert result.model_patch == ""


def test_success_true_when_completed_with_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ctx = _make_relative_ctx(monkeypatch, tmp_path, scaffold="openclaw")
    runtime_dir = ctx.attempt_dir / "_task_container_runtime" / "openclaw"
    stdout_path = runtime_dir / "stdout.txt"
    stderr_path = runtime_dir / "stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("openclaw stdout", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    trace_path = ctx.attempt_dir / "trace.jsonl"
    trace_path.write_text(
        '{"type":"trace_metadata","scaffold":"openclaw","trace_format_version":5}\n',
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

    from agents.openclaw._session_runner import SessionRunResult

    async def fake_session_run(self, prompt, workspace, *,
        tool_workspace=None, session_key="", trace_file=None,
        runtime_dir=None, instance_id=None, channel="cli", prepare_ms=None,
    ):
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text(
            '{"type":"trace_metadata","scaffold":"openclaw","trace_format_version":5}\n',
            encoding="utf-8",
        )
        return SessionRunResult(
            content="Task completed.",
            elapsed_s=1.0,
            trace_file=trace_file,
            session_key=session_key,
            stop_reason="completed",
            error=None,
        )

    monkeypatch.setattr(
        "trace_collect.collector.SessionRunner.run",
        fake_session_run,
    )

    # Return patch → model_patch is non-empty → success=True
    monkeypatch.setattr(
        "agents.openclaw.eval.runner.SWEBenchRunner._extract_container_patch",
        lambda diff_cwd, *, base_commit, run: "diff --git a/main.py b/main.py\n+fix",
    )

    result = asyncio.run(
        _run_openclaw_in_task_container(
            ctx=ctx,
            task=dict(ctx.task),
            benchmark=SimpleNamespace(
                execution_environment="container",
                config=SimpleNamespace(slug="swe-rebench", harness_split="filtered"),
            ),
            container_executable="docker",
            provider_name="openrouter",
            api_base="https://example.com",
            api_key="test-key",
            model="qwen-plus-latest",
            max_iterations=10,
            generation_config=None,
            max_context_tokens=1024,
            mcp_config=None,
        )
    )

    assert result.success is True
    assert result.exit_status == "completed"
    assert result.model_patch != ""
