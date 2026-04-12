"""Tests for src/trace_collect/attempt_pipeline.run_attempt orchestration."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness.container_image_prep import clear_image_cache  # noqa: E402
from harness.disk_preflight import DiskSpaceError  # noqa: E402
from trace_collect.attempt_pipeline import (  # noqa: E402
    AttemptContext,
    AttemptResult,
    start_task_container,
    run_attempt,
)


def _make_ctx(tmp_path: Path) -> AttemptContext:
    return AttemptContext(
        run_dir=tmp_path / "run",
        instance_id="mozilla__bleach-259",
        attempt=1,
        task={"instance_id": "mozilla__bleach-259", "repo": "mozilla/bleach"},
        model="qwen-plus-latest",
        scaffold="miniswe",
        source_image="swerebench/img:latest",
        prompt_template="default",
    )


@pytest.fixture(autouse=True)
def _reset_image_cache() -> None:
    clear_image_cache()
    yield
    clear_image_cache()


@pytest.fixture(autouse=True)
def _mock_image_exists() -> None:
    """Pretend the writable derivative image already exists so no build runs."""

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["podman", "image", "exists"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "harness.container_image_prep.subprocess.run",
        side_effect=fake_run,
    ):
        yield


def _write_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"type":"trace_metadata","scaffold":"miniswe","trace_format_version":5}\n'
        '{"type":"summary","agent_id":"mozilla__bleach-259","success":true}\n',
        encoding="utf-8",
    )


def test_run_attempt_success_writes_all_six_files(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    ctx.agent_runtime_mode = "task_container_agent"
    trace_source = tmp_path / "scratch" / "trace.jsonl"
    _write_trace(trace_source)

    async def inner(ctx: AttemptContext) -> AttemptResult:
        ctx.mark_container_ready("fake_container_id_xyz")
        ctx.container_stdout = "hello from container stdout"
        return AttemptResult(
            success=True,
            exit_status="Submitted",
            trace_path=trace_source,
            tool_calls=[
                {"tool": "Bash", "input": {"command": "ls"}, "duration_ms": 12.0}
            ],
            n_iterations=18,
            total_llm_ms=94000.0,
            total_tool_ms=12000.0,
            total_tokens=98088,
            runtime_proof={
                "container_id": "fake_container_id_xyz",
                "python_executable": "/opt/conda/envs/ML/bin/python",
            },
        )

    result = asyncio.run(
        run_attempt(
            ctx,
            inner=inner,
            min_free_disk_gb=0.001,
            container_executable="docker",
        )
    )

    assert result.success is True
    assert ctx.attempt_dir.exists()
    for name in (
        "trace.jsonl",
        "run_manifest.json",
        "results.json",
        "resources.json",
        "tool_calls.json",
        "container_stdout.txt",
    ):
        assert (ctx.attempt_dir / name).exists(), f"{name} missing"

    manifest = json.loads((ctx.attempt_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["task"]["instance_id"] == "mozilla__bleach-259"
    assert manifest["task"]["repo"] == "mozilla/bleach"
    assert manifest["attempt"] == "attempt_1"
    assert manifest["model"]["name"] == "qwen-plus-latest"
    assert manifest["result_summary"]["exit_code"] == 0
    assert manifest["result_summary"]["total_time"] >= 0.0
    assert manifest["scaffold"] == "miniswe"
    assert manifest["prompt_template"] == "default"
    assert manifest["agent_runtime_mode"] == "task_container_agent"
    assert manifest["runtime"]["agent_runtime_mode"] == "task_container_agent"
    assert (
        manifest["runtime"]["runtime_proof"]["container_id"] == "fake_container_id_xyz"
    )
    assert "tool_call_count" not in manifest["replay"]

    results = json.loads((ctx.attempt_dir / "results.json").read_text())
    assert results["instance_id"] == "mozilla__bleach-259"
    assert results["success"] is True
    assert results["model"] == "qwen-plus-latest"
    assert results["agent_runtime_mode"] == "task_container_agent"
    assert (
        results["runtime_proof"]["python_executable"] == "/opt/conda/envs/ML/bin/python"
    )
    assert "container_stdout" not in results
    assert "resource_samples" not in results

    tool_calls = json.loads((ctx.attempt_dir / "tool_calls.json").read_text())
    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "Bash"

    resources = json.loads((ctx.attempt_dir / "resources.json").read_text())
    assert "samples" in resources

    container_stdout = (ctx.attempt_dir / "container_stdout.txt").read_text()
    assert container_stdout == "hello from container stdout"

    trace = (ctx.attempt_dir / "trace.jsonl").read_text()
    assert "trace_metadata" in trace


def test_run_attempt_inner_exception_writes_error_manifest(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)

    async def inner(ctx: AttemptContext) -> AttemptResult:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(
            run_attempt(
                ctx,
                inner=inner,
                min_free_disk_gb=0.001,
                container_executable="docker",
            )
        )

    # Even on failure, manifest + claude output files exist
    assert (ctx.attempt_dir / "run_manifest.json").exists()
    manifest = json.loads((ctx.attempt_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "error"
    assert manifest["result_summary"]["exit_code"] == 1
    assert "boom" in (manifest["result_summary"]["error"] or "")


def test_run_attempt_noncompleted_exit_status_writes_error_manifest(
    tmp_path: Path,
) -> None:
    ctx = _make_ctx(tmp_path)
    trace_source = tmp_path / "scratch" / "trace.jsonl"
    _write_trace(trace_source)

    async def inner(ctx: AttemptContext) -> AttemptResult:
        return AttemptResult(
            success=False,
            exit_status="max_iterations",
            trace_path=trace_source,
            error="I reached the maximum number of tool call iterations.",
        )

    result = asyncio.run(
        run_attempt(
            ctx,
            inner=inner,
            min_free_disk_gb=0.001,
            container_executable="docker",
        )
    )

    assert result.success is False
    manifest = json.loads((ctx.attempt_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "error"
    assert manifest["result_summary"]["exit_code"] == 1
    assert manifest["result_summary"]["error"] == (
        "I reached the maximum number of tool call iterations."
    )


def test_run_attempt_supports_non_image_success_without_patch(
    tmp_path: Path,
) -> None:
    ctx = AttemptContext(
        run_dir=tmp_path / "run",
        instance_id="hello-world",
        attempt=1,
        task={
            "instance_id": "hello-world",
            "repo": None,
            "task_source_kind": "terminal_bench_registry",
            "task_source_id": "hello-world",
            "task_source_path": "/tmp/tasks/hello-world",
            "tb_version": "0.2.18",
            "tb_dataset": "terminal-bench-core",
            "tb_registry_source": "registry.json",
            "adapter_kind": "terminal_bench_openclaw",
            "agent_import_path": "agents.terminal_bench.openclaw_agent:TerminalBenchOpenClawAgent",
        },
        model="z-ai/glm-5.1",
        scaffold="openclaw",
        source_image=None,
        prompt_template="default",
    )
    trace_source = tmp_path / "scratch" / "trace.jsonl"
    _write_trace(trace_source)

    async def inner(ctx: AttemptContext) -> AttemptResult:
        return AttemptResult(
            success=True,
            exit_status="completed",
            trace_path=trace_source,
            model_patch="",
            summary={
                "tb_version": "0.2.18",
                "tb_dataset": "terminal-bench-core",
                "tb_registry_source": "registry.json",
                "adapter_kind": "terminal_bench_openclaw",
                "agent_import_path": "agents.terminal_bench.openclaw_agent:TerminalBenchOpenClawAgent",
            },
        )

    result = asyncio.run(run_attempt(ctx, inner=inner, min_free_disk_gb=0.001))

    assert result.success is True
    manifest = json.loads((ctx.attempt_dir / "run_manifest.json").read_text())
    assert manifest["task"]["docker_image"] is None
    assert manifest["task"]["repo"] is None
    assert manifest["replay"]["source_image"] is None
    assert manifest["replay"]["fixed_image_name"] is None
    results = json.loads((ctx.attempt_dir / "results.json").read_text())
    assert results["success"] is True
    assert results["docker_image"] is None
    assert results["image"] is None
    assert results["repo"] is None
    assert results["tb_version"] == "0.2.18"


def test_run_attempt_disk_shortfall_aborts_early(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)

    called = {"inner": False}

    async def inner(ctx: AttemptContext) -> AttemptResult:
        called["inner"] = True
        return AttemptResult(
            success=True, exit_status="ok", trace_path=tmp_path / "nope.jsonl"
        )

    with pytest.raises(DiskSpaceError):
        asyncio.run(
            run_attempt(
                ctx,
                inner=inner,
                min_free_disk_gb=10**12,
                container_executable="docker",
            )
        )

    assert called["inner"] is False
    # attempt_dir should NOT have been created
    assert not ctx.attempt_dir.exists()


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_run_attempt_passes_container_executable_to_fixed_image(
    tmp_path: Path,
    container_executable: str,
) -> None:
    ctx = _make_ctx(tmp_path)
    trace_source = tmp_path / "scratch" / "trace.jsonl"
    _write_trace(trace_source)
    seen: dict[str, object] = {}

    async def inner(ctx: AttemptContext) -> AttemptResult:
        return AttemptResult(
            success=True,
            exit_status="Submitted",
            trace_path=trace_source,
        )

    with patch(
        "trace_collect.attempt_pipeline.ensure_fixed_image",
        side_effect=lambda source_image, *, container_executable: (
            seen.update(
                {
                    "source_image": source_image,
                    "container_executable": container_executable,
                }
            )
            or ("fixed-image", 0.0)
        ),
    ):
        asyncio.run(
            run_attempt(
                ctx,
                inner=inner,
                min_free_disk_gb=0.001,
                container_executable=container_executable,
            )
        )

    assert seen == {
        "source_image": "swerebench/img:latest",
        "container_executable": container_executable,
    }


@pytest.mark.parametrize(
    ("container_executable", "expected_user_args"),
    [
        ("docker", ["--user", f"{os.getuid()}:{os.getgid()}"]),
        ("podman", ["--userns=keep-id"]),
    ],
)
def test_start_task_container_uses_runtime_specific_user_args(
    tmp_path: Path,
    container_executable: str,
    expected_user_args: list[str],
) -> None:
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="cid-1\n", stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        container_id = start_task_container(
            "docker.io/swerebench/example:latest",
            executable=container_executable,
        )

    assert container_id == "cid-1"
    assert seen["cmd"][:3] == [container_executable, "run", "-d"]
    assert "-e" in seen["cmd"]
    assert f"HOME={os.environ.get('HOME', '/root')}" in seen["cmd"]
    for arg in expected_user_args:
        assert arg in seen["cmd"]
    if container_executable == "docker":
        assert "--userns=keep-id" not in seen["cmd"]


def test_start_task_container_passes_through_network_env_when_present() -> None:
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="cid-1\n", stderr="")

    env = {
        "HTTP_PROXY": "http://127.0.0.1:7890",
        "HTTPS_PROXY": "http://127.0.0.1:7890",
        "ALL_PROXY": "socks5://127.0.0.1:7890",
        "NO_PROXY": "localhost,127.0.0.1",
        "PIP_INDEX_URL": "https://pypi.tuna.tsinghua.edu.cn/simple",
        "TASK_CONTAINER_PIP_INDEX_URL": "https://mirror.example/simple",
    }
    with (
        patch("subprocess.run", side_effect=fake_run),
        patch.dict(os.environ, env, clear=False),
    ):
        start_task_container("docker.io/swerebench/example:latest", executable="docker")

    cmd = seen["cmd"]
    for name, value in env.items():
        assert "-e" in cmd
        assert f"{name}={value}" in cmd


def test_start_task_container_skips_empty_network_env() -> None:
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="cid-1\n", stderr="")

    env = {
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "NO_PROXY": "",
        "PIP_INDEX_URL": "",
        "TASK_CONTAINER_PIP_INDEX_URL": "",
    }
    with (
        patch("subprocess.run", side_effect=fake_run),
        patch.dict(os.environ, env, clear=False),
    ):
        start_task_container("docker.io/swerebench/example:latest", executable="docker")

    cmd = seen["cmd"]
    for name in env:
        assert not any(
            isinstance(part, str) and part.startswith(f"{name}=") for part in cmd
        )
