"""Tests for src/trace_collect/attempt_pipeline.run_attempt orchestration."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness.container_image_prep import clear_image_cache  # noqa: E402
from harness.disk_preflight import DiskSpaceError  # noqa: E402
from trace_collect.attempt_pipeline import (  # noqa: E402
    AttemptContext,
    AttemptResult,
    configure_task_container_apt_mirror,
    start_task_container,
    stop_task_container,
    run_attempt,
)


def _make_ctx(tmp_path: Path) -> AttemptContext:
    return AttemptContext(
        run_dir=tmp_path / "run",
        instance_id="mozilla__bleach-259",
        attempt=1,
        task={"instance_id": "mozilla__bleach-259", "repo": "mozilla/bleach"},
        model="qwen-plus-latest",
        scaffold="openclaw",
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
        '{"type":"trace_metadata","scaffold":"openclaw","trace_format_version":5}\n'
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
        tool_result_artifact = (
            ctx.attempt_dir
            / "openclaw-runtime"
            / "tool-results"
            / "tool-results"
            / "cli_mozilla__bleach-259"
            / "large-output.txt"
        )
        tool_result_artifact.parent.mkdir(parents=True, exist_ok=True)
        tool_result_artifact.write_text("large output", encoding="utf-8")
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
                "python_executable": "/usr/bin/python3",
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
    # Wall-clock breakdown checkpoints must be recorded and add up.
    timing = manifest["timing"]
    assert set(timing) == {
        "wall_total_s",
        "setup_s",
        "agent_exec_s",
        "teardown_s",
        "permission_fix_s",
    }
    assert timing["wall_total_s"] >= 0.0
    assert timing["setup_s"] >= 0.0
    assert timing["agent_exec_s"] >= 0.0
    assert timing["teardown_s"] >= 0.0
    assert timing["permission_fix_s"] >= 0.0
    # The breakdown must reconcile with the wall total within float noise.
    assert (
        abs(
            timing["wall_total_s"]
            - (timing["setup_s"] + timing["agent_exec_s"] + timing["teardown_s"])
        )
        < 1e-3
    )
    assert manifest["scaffold"] == "openclaw"
    assert manifest["prompt_template"] == "default"
    assert manifest["agent_runtime_mode"] == "task_container_agent"
    assert manifest["runtime"]["agent_runtime_mode"] == "task_container_agent"
    assert (
        manifest["runtime"]["runtime_proof"]["container_id"] == "fake_container_id_xyz"
    )
    assert "tool_call_count" not in manifest["replay"]
    assert manifest["artifacts"]["trace_jsonl"] == "trace.jsonl"
    assert manifest["artifacts"]["results_json"] == "results.json"
    assert (
        manifest["artifacts"]["openclaw_tool_results_dir"]
        == "openclaw-runtime/tool-results"
    )

    results = json.loads((ctx.attempt_dir / "results.json").read_text())
    assert results["instance_id"] == "mozilla__bleach-259"
    assert results["success"] is True
    assert results["model"] == "qwen-plus-latest"
    assert results["agent_runtime_mode"] == "task_container_agent"
    assert results["runtime_proof"]["python_executable"] == "/usr/bin/python3"
    assert results["timing"]["wall_total_s"] >= 0.0
    assert results["timing"]["setup_s"] >= 0.0
    assert results["timing"]["agent_exec_s"] >= 0.0
    assert results["timing"]["teardown_s"] >= 0.0
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
    timing = manifest["timing"]
    assert timing["wall_total_s"] >= 0.0
    assert timing["setup_s"] >= 0.0
    assert timing["agent_exec_s"] >= 0.0
    assert timing["teardown_s"] >= 0.0
    assert abs(
        timing["wall_total_s"]
        - (
            timing["setup_s"]
            + timing["agent_exec_s"]
            + timing["teardown_s"]
        )
    ) < 1e-3


def test_run_attempt_max_iterations_writes_exhausted_manifest(
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
    assert manifest["status"] == "exhausted"
    assert manifest["result_summary"]["exit_code"] == 1
    assert manifest["result_summary"]["exit_status"] == "max_iterations"
    assert manifest["result_summary"]["error"] == (
        "I reached the maximum number of tool call iterations."
    )


def test_run_attempt_error_exit_status_writes_error_manifest(
    tmp_path: Path,
) -> None:
    ctx = _make_ctx(tmp_path)
    trace_source = tmp_path / "scratch" / "trace.jsonl"
    _write_trace(trace_source)

    async def inner(ctx: AttemptContext) -> AttemptResult:
        return AttemptResult(
            success=False,
            exit_status="tool_error",
            trace_path=trace_source,
            error="tool failed",
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
    assert manifest["result_summary"]["exit_status"] == "tool_error"
    assert manifest["result_summary"]["error"] == "tool failed"


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
        execution_environment="host",
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

    result = asyncio.run(
        run_attempt(
            ctx,
            inner=inner,
            min_free_disk_gb=0.001,
            container_executable="docker",
        )
    )

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
    resources = json.loads((ctx.attempt_dir / "resources.json").read_text())
    assert resources["samples"]
    assert resources["summary"]["sample_count"] >= 1


def test_run_attempt_waits_for_published_container_name_before_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = AttemptContext(
        run_dir=tmp_path / "run",
        instance_id="hello-world",
        attempt=1,
        task={"instance_id": "hello-world"},
        model="z-ai/glm-5.1",
        scaffold="openclaw",
        source_image=None,
        prompt_template="default",
        execution_environment="container",
    )
    trace_source = tmp_path / "scratch" / "trace.jsonl"
    _write_trace(trace_source)
    inspect_calls = {"count": 0}
    sampled: dict[str, str] = {}

    def fake_container_is_inspectable(
        container_id: str, *, container_executable: str
    ) -> bool:
        inspect_calls["count"] += 1
        return inspect_calls["count"] >= 2

    class FakeContainerStatsSampler:
        def __init__(self, container_id: str, **kwargs) -> None:
            sampled["container_id"] = container_id
            sampled["executable"] = kwargs["executable"]

        def start(self) -> None:
            sampled["started"] = "yes"

        def stop(self) -> list[dict[str, object]]:
            return [
                {
                    "timestamp": "2026-04-28T00:00:00",
                    "epoch": time.time(),
                    "mem_usage": "128MiB / 1024MiB",
                    "mem_percent": "12.5%",
                    "cpu_percent": "42.0%",
                }
            ]

    monkeypatch.setattr(
        "trace_collect.attempt_pipeline._container_is_inspectable",
        fake_container_is_inspectable,
    )
    monkeypatch.setattr(
        "trace_collect.attempt_pipeline.ContainerStatsSampler",
        FakeContainerStatsSampler,
    )

    async def inner(ctx: AttemptContext) -> AttemptResult:
        ctx.mark_container_ready("hello-world-1-of-1-run")
        await asyncio.sleep(0.15)
        return AttemptResult(
            success=True,
            exit_status="completed",
            trace_path=trace_source,
        )

    asyncio.run(
        run_attempt(
            ctx,
            inner=inner,
            min_free_disk_gb=0.001,
            container_executable="docker",
        )
    )

    assert inspect_calls["count"] >= 2
    assert sampled == {
        "container_id": "hello-world-1-of-1-run",
        "executable": "docker",
        "started": "yes",
    }
    resources = json.loads((ctx.attempt_dir / "resources.json").read_text())
    assert len(resources["samples"]) == 1
    assert resources["summary"]["sample_count"] == 1


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


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_start_task_container_can_use_image_default_user_without_host_home_mount(
    container_executable: str,
) -> None:
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="cid-1\n", stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        container_id = start_task_container(
            "docker.io/swerebench/example:latest",
            executable=container_executable,
            run_as_host_user=False,
            mount_host_home=False,
            container_home="/root",
        )

    cmd = seen["cmd"]
    assert container_id == "cid-1"
    assert "--user" not in cmd
    assert "--userns=keep-id" not in cmd
    assert "-v" not in cmd
    assert "HOME=/root" in cmd
    # Host ~/.local/bin must NOT leak into the container PATH.
    assert "PATH=/usr/local/bin:/usr/bin:/bin" in cmd
    assert all("/.local/bin" not in str(part) for part in cmd)


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_start_task_container_prepends_bootstrap_userbase_bin(
    container_executable: str,
) -> None:
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="cid-1\n", stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        start_task_container(
            "img:latest",
            executable=container_executable,
            run_as_host_user=False,
            mount_host_home=False,
            container_home="/root",
            bootstrap_userbase_bin="/testbed/_task_container_runtime/bootstrap/.pyuserbase/bin",
        )

    cmd = seen["cmd"]
    assert (
        "PATH=/testbed/_task_container_runtime/bootstrap/.pyuserbase/bin:"
        "/usr/local/bin:/usr/bin:/bin"
    ) in cmd
    assert "PYTHONUSERBASE=/testbed/_task_container_runtime/bootstrap/.pyuserbase" in cmd
    assert "PIP_BREAK_SYSTEM_PACKAGES=1" in cmd
    # Host ~/.local/bin still must not leak.
    assert all("/.local/bin" not in str(part) for part in cmd)


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
        "TASK_CONTAINER_PIP_EXTRA_INDEX_URL": "https://extra.example/simple",
        "TASK_CONTAINER_PIP_TRUSTED_HOST": "mirror.example",
        "TASK_CONTAINER_PIP_CERT": "/certs/pip.pem",
        "TASK_CONTAINER_SSL_CERT_FILE": "/certs/ssl.pem",
        "TASK_CONTAINER_HTTP_PROXY": "http://proxy.example:8080",
        "TASK_CONTAINER_HTTPS_PROXY": "http://proxy.example:8443",
        "TASK_CONTAINER_ALL_PROXY": "socks5://proxy.example:1080",
        "TASK_CONTAINER_NO_PROXY": "localhost,127.0.0.1",
        "TASK_CONTAINER_APT_MIRROR": "https://mirror.example/debian",
        "TASK_CONTAINER_APT_SECURITY_MIRROR": "https://mirror.example/debian-security",
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
        "TASK_CONTAINER_PIP_EXTRA_INDEX_URL": "",
        "TASK_CONTAINER_PIP_TRUSTED_HOST": "",
        "TASK_CONTAINER_PIP_CERT": "",
        "TASK_CONTAINER_SSL_CERT_FILE": "",
        "TASK_CONTAINER_HTTP_PROXY": "",
        "TASK_CONTAINER_HTTPS_PROXY": "",
        "TASK_CONTAINER_ALL_PROXY": "",
        "TASK_CONTAINER_NO_PROXY": "",
        "TASK_CONTAINER_APT_MIRROR": "",
        "TASK_CONTAINER_APT_SECURITY_MIRROR": "",
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


def test_configure_task_container_apt_mirror_noops_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TASK_CONTAINER_APT_MIRROR", raising=False)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        result = configure_task_container_apt_mirror("cid-1", executable="docker")

    assert result is None
    assert calls == []


def test_configure_task_container_apt_mirror_writes_debian_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setenv("TASK_CONTAINER_APT_MIRROR", "https://mirror.example/debian/")
    monkeypatch.delenv("TASK_CONTAINER_APT_SECURITY_MIRROR", raising=False)

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["input"] = kwargs["input"]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                "apt mirror configured: main=https://mirror.example/debian "
                "security=https://mirror.example/debian-security\n"
            ),
            stderr="",
        )

    with patch("subprocess.run", side_effect=fake_run):
        result = configure_task_container_apt_mirror("cid-1", executable="docker")

    assert seen["cmd"] == [
        "docker",
        "exec",
        "-i",
        "--user",
        "0:0",
        "cid-1",
        "/bin/sh",
        "-s",
    ]
    script = str(seen["input"])
    assert "TASK_CONTAINER_APT_MIRROR" not in script
    assert "main_mirror=https://mirror.example/debian" in script
    assert 'security_mirror="${main_mirror%/debian}/debian-security"' in script
    assert "URIs: $main_mirror" in script
    assert "URIs: $security_mirror" in script
    assert "Suites: $codename $codename-updates" in script
    assert "Suites: $codename-security" in script
    assert "*.agent-sched-bench-disabled" in script
    assert result == {
        "configured": "true",
        "main_mirror": "https://mirror.example/debian",
        "security_mirror": "https://mirror.example/debian-security",
        "stdout": (
            "apt mirror configured: main=https://mirror.example/debian "
            "security=https://mirror.example/debian-security"
        ),
    }


def test_configure_task_container_apt_mirror_rejects_unsafe_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TASK_CONTAINER_APT_MIRROR",
        "https://mirror.example/debian $(touch /tmp/nope)",
    )

    with (
        patch("subprocess.run") as run,
        pytest.raises(ValueError, match="TASK_CONTAINER_APT_MIRROR"),
    ):
        configure_task_container_apt_mirror("cid-1", executable="docker")

    run.assert_not_called()


def test_configure_task_container_apt_mirror_supports_ubuntu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setenv("TASK_CONTAINER_APT_MIRROR", "https://mirror.example/ubuntu/")
    monkeypatch.delenv("TASK_CONTAINER_APT_SECURITY_MIRROR", raising=False)

    def fake_run(cmd, **kwargs):
        seen["input"] = kwargs["input"]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                "apt mirror configured: distro=ubuntu "
                "main=https://mirror.example/ubuntu "
                "security=https://mirror.example/ubuntu\n"
            ),
            stderr="",
        )

    with patch("subprocess.run", side_effect=fake_run):
        result = configure_task_container_apt_mirror("cid-1", executable="docker")

    script = str(seen["input"])
    assert "ubuntu)" in script
    assert 'components="main restricted universe multiverse"' in script
    assert 'signed_by="/usr/share/keyrings/ubuntu-archive-keyring.gpg"' in script
    assert result is not None
    assert result["configured"] == "true"
    assert result["security_mirror"] == "https://mirror.example/ubuntu"


def test_stop_task_container_raises_when_container_still_exists() -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="logs\n", stderr="")
        if cmd[:2] == ["docker", "stop"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="stop failed")
        if cmd[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="rm failed")
        if cmd[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="Failed to remove task container cid-1"):
            stop_task_container("cid-1", executable="docker")

    assert calls == [
        ["docker", "logs", "cid-1"],
        ["docker", "stop", "cid-1"],
        ["docker", "rm", "-f", "cid-1"],
        ["docker", "inspect", "cid-1"],
    ]


def test_stop_task_container_tolerates_stop_error_when_rm_removes_container() -> None:
    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="logs\n", stderr="")
        if cmd[:2] == ["docker", "stop"]:
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="already stopped"
            )
        if cmd[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="cid-1\n", stderr="")
        if cmd[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="No such object"
            )
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=fake_run):
        logs = stop_task_container("cid-1", executable="docker")

    assert logs == "logs\n"


def test_stop_task_container_waits_for_removal_already_in_progress() -> None:
    inspect_calls = 0

    def fake_run(cmd, **kwargs):
        nonlocal inspect_calls
        if cmd[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="logs\n", stderr="")
        if cmd[:2] == ["docker", "stop"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="cid-1\n", stderr="")
        if cmd[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr=(
                    "Error response from daemon: removal of container cid-1 "
                    "is already in progress"
                ),
            )
        if cmd[:2] == ["docker", "inspect"]:
            inspect_calls += 1
            if inspect_calls == 1:
                return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="Error: No such object: cid-1",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    with (
        patch("subprocess.run", side_effect=fake_run),
        patch("trace_collect.attempt_pipeline.time.sleep"),
    ):
        logs = stop_task_container("cid-1", executable="docker")

    assert logs == "logs\n"
    assert inspect_calls == 2


def test_stop_task_container_raises_on_unclassified_inspect_failure() -> None:
    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["docker", "stop"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="stop failed")
        if cmd[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="rm failed")
        if cmd[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="Cannot connect to the Docker daemon",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="Cannot connect to the Docker daemon"):
            stop_task_container("cid-1", executable="docker")
