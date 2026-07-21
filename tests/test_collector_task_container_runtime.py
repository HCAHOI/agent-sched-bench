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
from trace_collect.runtime.task_container import TaskContainerExecConfig


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


def test_run_openclaw_in_task_container_runs_openclaw_on_host_with_container_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    start_seen: dict[str, object] = {}
    agent_seen: dict[str, object] = {}
    build_seen: dict[str, object] = {}
    run_seen: dict[str, object] = {}
    provider_seen: dict[str, object] = {}
    ctx = _make_relative_ctx(monkeypatch, tmp_path, scaffold="openclaw")
    runtime_dir = ctx.attempt_dir.resolve() / "_task_container_runtime" / "openclaw"

    def fake_start_task_container(*args, **kwargs):
        start_seen["args"] = args
        start_seen.update(kwargs)
        return "cid-openclaw"

    class FakeContainerAgent:
        def __init__(self, container_id: str, container_executable: str, **kwargs):
            agent_seen["container_id"] = container_id
            agent_seen["container_executable"] = container_executable
            agent_seen["init_kwargs"] = kwargs

        async def start(self) -> None:
            agent_seen["started"] = True

        async def stop(self) -> None:
            agent_seen["stopped"] = True

        async def execute(self, request: dict, *, timeout_s: float = 600.0) -> dict:
            agent_seen.setdefault("requests", []).append((request, timeout_s))
            return {
                "ok": True,
                "result": "0\n/testbed\nLinux\nx86_64\nPython 3.11.0\n",
                "returncode": 0,
            }

    class FakeRunner:
        async def run_task(self, eval_task, **kwargs):
            run_seen["eval_task"] = eval_task
            run_seen.update(kwargs)
            Path(kwargs["trace_file"]).write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "trace_metadata",
                                "scaffold": "openclaw",
                                "trace_format_version": 5,
                                "model": "qwen-plus-latest",
                            }
                        ),
                        json.dumps(
                            {
                                "type": "summary",
                                "agent_id": ctx.instance_id,
                                "total_llm_ms": 12.0,
                                "total_tool_ms": 6.0,
                                "total_tokens": 99,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            return SimpleNamespace(
                model_patch="diff --git a/httpx.py b/httpx.py",
                stop_reason="completed",
                error=None,
                n_iterations=4,
                usage={},
            )

    class FakeBenchmark:
        execution_environment = "container"
        config = SimpleNamespace(slug="swe-rebench", harness_split="filtered")

        def build_runner(self, **kwargs):
            build_seen.update(kwargs)
            return FakeRunner()

    def fail_run_task_container_agent(**kwargs):
        raise AssertionError(
            "OpenClaw must run on the host; do not docker-exec the OpenClaw runner"
        )

    monkeypatch.setattr(
        "trace_collect.collector.start_task_container",
        fake_start_task_container,
    )
    monkeypatch.setattr(
        "trace_collect.collector.stop_task_container",
        lambda *args, **kwargs: "container logs",
    )
    monkeypatch.setattr(
        "trace_collect.collector.resolve_task_container_exec_config",
        lambda **kwargs: TaskContainerExecConfig(
            runtime="/usr/bin/python3",
            pythonpath="/deps:/repo/src:/repo",
            start_extra_args=("--platform", "linux/amd64"),
            bootstrap=True,
            bootstrap_site_dir=ctx.attempt_dir
            / "_task_container_runtime"
            / "bootstrap"
            / "pydeps",
            image_platform="linux/amd64",
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.run_task_container_agent",
        fail_run_task_container_agent,
    )
    monkeypatch.setattr(
        "trace_collect.openclaw_tools.ContainerAgent",
        FakeContainerAgent,
    )
    monkeypatch.setattr(
        "trace_collect.collector.create_provider",
        lambda **kwargs: (provider_seen.update(kwargs), SimpleNamespace())[1],
    )

    result = asyncio.run(
        _run_openclaw_in_task_container(
            ctx=ctx,
            task=dict(ctx.task),
            benchmark=FakeBenchmark(),
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
    assert metadata["agent_runtime_mode"] == "task_container_agent"
    assert metadata["runtime_proof"]["agent_execution_environment"] == "host"
    assert metadata["runtime_proof"]["tool_execution_environment"] == "task_container"
    assert metadata["runtime_proof"]["tool_container_id"] == "cid-openclaw"
    assert metadata["runtime_proof"]["tool_container_user"] == "root"
    assert metadata["runtime_proof"]["tool_container_user_id"] == 0
    assert metadata["runtime_proof"]["tool_container_workdir"] == "/testbed"
    assert start_seen["run_as_host_user"] is False
    assert start_seen["mount_host_home"] is False
    assert start_seen["container_home"] == "/root"
    assert agent_seen["container_id"] == "cid-openclaw"
    assert agent_seen["container_executable"] == "docker"
    assert agent_seen["started"] is True
    assert agent_seen["stopped"] is True
    assert provider_seen["api_base"] == "https://example.com"
    assert provider_seen["api_key"] == "test-key"
    assert provider_seen["default_model"] == "qwen-plus-latest"
    assert build_seen["scaffold"] == "openclaw"
    assert Path(str(build_seen["workspace_base"])).is_absolute()
    assert Path(build_seen["workspace_base"]) == runtime_dir / "workspace_base"
    assert build_seen["max_iterations"] == 10
    assert build_seen["context_window_tokens"] == 1024
    assert build_seen["tool_overrides"]
    assert callable(build_seen["container_patch_extractor"])
    assert run_seen["tool_workspace"] == Path("/testbed")
    assert run_seen["exec_working_dir"] == "/testbed"
    assert Path(run_seen["trace_file"]) == (ctx.attempt_dir / "trace.jsonl").resolve()
    assert "Shell/file tools runtime: Linux x86_64" in run_seen["runtime_label"]
    assert "Shell/file tools `python3`: Python 3.11.0" in run_seen["runtime_label"]
    assert Path(run_seen["eval_task"].workspace_dir) == runtime_dir / "workspace_base" / ctx.instance_id
    assert result.total_llm_ms == 12.0
    assert result.total_tool_ms == 6.0
    assert result.total_tokens == 99


def test_run_openclaw_in_task_container_completed_without_patch_is_not_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ctx = _make_ctx(tmp_path, scaffold="openclaw")

    def fake_start_task_container(*args, **kwargs):
        return "cid-openclaw"

    class FakeContainerAgent:
        def __init__(self, container_id: str, container_executable: str, **kwargs):
            self.container_id = container_id
            self.container_executable = container_executable

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def execute(self, request: dict, *, timeout_s: float = 600.0) -> dict:
            return {
                "ok": True,
                "result": "0\n/testbed\nLinux\nx86_64\nPython 3.11.0\n",
                "returncode": 0,
            }

    class FakeRunner:
        async def run_task(self, eval_task, **kwargs):
            trace_file = Path(kwargs["trace_file"])
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "trace_metadata",
                                "scaffold": "openclaw",
                                "trace_format_version": 5,
                                "model": "qwen-plus-latest",
                            }
                        ),
                        json.dumps(
                            {
                                "type": "summary",
                                "agent_id": ctx.instance_id,
                                "total_llm_ms": 0.0,
                                "total_tool_ms": 0.0,
                                "total_tokens": 0,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            return SimpleNamespace(
                model_patch="",
                stop_reason="completed",
                error=None,
                n_iterations=1,
                usage={},
            )

    class FakeBenchmark:
        execution_environment = "container"
        config = SimpleNamespace(slug="swe-rebench", harness_split="filtered")

        def build_runner(self, **kwargs):
            return FakeRunner()

    monkeypatch.setattr(
        "trace_collect.collector.start_task_container",
        fake_start_task_container,
    )
    monkeypatch.setattr(
        "trace_collect.collector.stop_task_container",
        lambda *args, **kwargs: "container logs",
    )
    monkeypatch.setattr(
        "trace_collect.collector.configure_task_container_apt_mirror",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.resolve_task_container_exec_config",
        lambda **kwargs: TaskContainerExecConfig(
            runtime="/usr/bin/python3",
            pythonpath="/deps:/repo/src:/repo",
            start_extra_args=(),
            bootstrap=False,
            bootstrap_site_dir=None,
            image_platform=None,
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.run_task_container_agent",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("OpenClaw must run on the host")
        ),
    )
    monkeypatch.setattr(
        "trace_collect.openclaw_tools.ContainerAgent",
        FakeContainerAgent,
    )
    monkeypatch.setattr(
        "trace_collect.collector.create_provider",
        lambda **kwargs: SimpleNamespace(),
    )

    result = asyncio.run(
        _run_openclaw_in_task_container(
            ctx=ctx,
            task=dict(ctx.task),
            benchmark=FakeBenchmark(),
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
    assert result.error is None


