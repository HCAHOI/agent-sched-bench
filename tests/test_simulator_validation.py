from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from trace_collect.simulator import simulate, SimulateError


def _write_manifest(path: Path, entries: list[str | dict[str, object]]) -> Path:
    lines: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            lines.append(f"- {json.dumps(entry)}")
            continue
        lines.append("-")
        for key, value in entry.items():
            lines.append(f"  {key}: {json.dumps(str(value))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _single_trace_manifest(tmp_path: Path, trace_path: Path) -> Path:
    return _write_manifest(tmp_path / "manifest.yaml", [str(trace_path)])

def test_terminal_bench_trace_identity_split_loads_action_owner_actions(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "openclaw",
                        "instance_id": "hydra-debug-slurm-mode",
                        "task_source_kind": "terminal_bench_registry",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "cli:oc-df47179e",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 2.0,
                        "data": {"messages_in": [], "raw_response": {"content": "ok"}},
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "tool_exec",
                        "action_id": "sub_tool_0",
                        "agent_id": "cli:oc-df47179e:subagent:worker",
                        "iteration": 0,
                        "ts_start": 2.0,
                        "ts_end": 3.0,
                        "data": {"tool_name": "exec"},
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "other_llm_0",
                        "agent_id": "cli:other",
                        "iteration": 0,
                        "ts_start": 4.0,
                        "ts_end": 5.0,
                        "data": {"messages_in": []},
                    }
                ),
                json.dumps(
                    {
                        "type": "summary",
                        "agent_id": "cli:oc-df47179e",
                        "success": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from trace_collect.simulator import _parse_trace_session_file

    task_instance_id, source_action_agent_id, _metadata, actions, summary = (
        _parse_trace_session_file(trace_path)
    )

    assert task_instance_id == "hydra-debug-slurm-mode"
    assert source_action_agent_id == "cli:oc-df47179e"
    assert [action["action_id"] for action in actions] == ["llm_0", "sub_tool_0"]
    assert summary == {"type": "summary", "agent_id": "cli:oc-df47179e", "success": True}


def test_trace_parser_rejects_ambiguous_action_owners_without_summary(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "openclaw",
                        "instance_id": "hydra-debug-slurm-mode",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "cli:oc-a",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 2.0,
                        "data": {"messages_in": []},
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_1",
                        "agent_id": "cli:oc-b",
                        "iteration": 0,
                        "ts_start": 3.0,
                        "ts_end": 4.0,
                        "data": {"messages_in": []},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from trace_collect.simulator import _parse_trace_session_file

    with pytest.raises(SimulateError, match="Ambiguous action owner"):
        _parse_trace_session_file(trace_path)


def test_simulator_rejects_task_without_docker_image(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "openclaw",
                        "instance_id": "fc_test_001",
                        "model": "dummy",
                        "mode": "collect",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "fc_test_001",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 2.0,
                        "data": {"messages_in": [], "completion_tokens": 1},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    task_source = tmp_path / "tasks.json"
    task_source.write_text(
        json.dumps([{"instance_id": "fc_test_001", "problem_statement": "x"}]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SimulateError, match="no resolvable docker_image"):
        asyncio.run(
            simulate(
                manifest=_single_trace_manifest(tmp_path, trace_path),
                task_source=task_source,
                output_dir=tmp_path / "out",
                api_base="http://localhost:8000/v1",
                api_key="EMPTY",
                model="dummy",
            )
        )
def test_simulator_accepts_task_with_image_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Task with image_name passes validation (prepare is mocked)."""
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "openclaw",
                        "instance_id": "fc_test_002",
                        "model": "dummy",
                        "mode": "collect",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "fc_test_002",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 2.0,
                        "data": {"messages_in": [], "completion_tokens": 1},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    task_source = tmp_path / "tasks.json"
    task_source.write_text(
        json.dumps(
            [
                {
                    "instance_id": "fc_test_002",
                    "problem_statement": "x",
                    "image_name": "swebench/test-image",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from trace_collect.simulator import PreparedContainer, PreparedTraceSession

    class _FakeAgent:
        async def stop(self): pass

    async def fake_prepare(
        loaded,
        *,
        task_output_dir=None,
        container_executable,
        network_mode="host",
    ):
        return PreparedTraceSession(
            loaded=loaded,
            container=PreparedContainer(
                container_id="fake",
                container_executable=container_executable,
                docker_image="fake",
                agent=_FakeAgent(),
            ),
        )

    async def fake_prefetch(*_args, **_kwargs) -> None:
        pass

    async def fake_prebuild(*_args, **_kwargs) -> dict[str, str]:
        return {}

    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", fake_prepare)
    monkeypatch.setattr("trace_collect.simulator._prefetch_container_images", fake_prefetch)
    monkeypatch.setattr("trace_collect.simulator._prebuild_sweep_fixed_images", fake_prebuild)
    monkeypatch.setattr(
        "trace_collect.simulator.stop_task_container",
        lambda *args, **kwargs: "",
    )
    async def _fake_exec(*a, **kw):
        return ("ok", 1.0, True)

    monkeypatch.setattr("trace_collect.simulator._exec_tool", _fake_exec)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
        )
    )
    assert trace_file.exists()



def test_openclaw_replay_provider_sleep_records_source_and_pid() -> None:
    import os

    from trace_collect.openclaw_host_runtime import OpenClawReplayProvider

    async def run_sleep() -> dict[str, float | int | str]:
        provider = OpenClawReplayProvider(
            llm_actions=[], replay_speed=1.0, timing_mode="source_scaled"
        )
        record = await provider._sleep(0.001, phase="llm_replay")
        assert record is not None
        return record.to_dict()

    sleep_record = asyncio.run(run_sleep())

    assert sleep_record["phase"] == "llm_replay"
    assert sleep_record["source"] == "openclaw_replay_provider_sleep"
    assert sleep_record["pid"] == os.getpid()


def test_source_model_prefers_summary_and_metadata_audit_fields() -> None:
    from trace_collect.simulator import LoadedTraceSession, _source_model

    loaded = LoadedTraceSession(
        source_trace=Path("trace.jsonl"),
        task_source=Path("tasks.json"),
        task_instance_id="task",
        source_action_agent_id="agent",
        run_instance_id="task",
        manifest_index=0,
        scaffold="openclaw",
        metadata={"source_model": "metadata-source-model"},
        summary={"source_model": "summary-source-model"},
        task={},
        actions=[],
        iterations={},
    )

    assert _source_model(loaded) == "summary-source-model"
    loaded.summary = None
    assert _source_model(loaded) == "metadata-source-model"


def test_replay_failure_counts_align_expected_failures_by_order() -> None:
    from trace_collect.openclaw_host_runtime import replay_action_failure_counts

    source_actions = [
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "source-blocked",
            "data": {"tool_name": "exec", "success": False},
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "source-ok",
            "data": {"tool_name": "exec", "success": True},
        },
    ]
    replay_records = [
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "synthetic-blocked-id",
            "data": {"tool_name": "exec", "success": False},
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "synthetic-unexpected-id",
            "data": {"tool_name": "exec", "success": False},
        },
    ]

    counts = replay_action_failure_counts(source_actions, replay_records)

    assert counts.emitted_actions == 2
    assert counts.source_failed_actions == 1
    assert counts.replay_failed_actions == 2
    assert counts.unexpected_replay_failed_actions == 1



def test_openclaw_container_mode_replays_llm_via_host_replay_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Container OpenClaw traces must use the host replay runner/provider for LLM calls."""
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "openclaw",
                        "instance_id": "fc_openclaw_replay",
                        "agent_runtime_mode": "task_container_agent",
                        "execution_environment": "container",
                        "model": "qwen/qwen3.7-max",
                        "mode": "collect",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "fc_openclaw_replay",
                        "iteration": 0,
                        "ts_start": 10.0,
                        "ts_end": 10.4,
                        "data": {
                            "messages_in": [{"role": "user", "content": "fix it"}],
                            "raw_response": {"content": "done"},
                            "completion_tokens": 3,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    task_source = tmp_path / "tasks.json"
    task_source.write_text(
        json.dumps(
            [
                {
                    "instance_id": "fc_openclaw_replay",
                    "problem_statement": "x",
                    "image_name": "swebench/test-image",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )


    from trace_collect.simulator import (
        PreparedContainer,
        PreparedTraceSession,
    )

    class _FakeAgent:
        async def stop(self) -> None:
            pass

    async def fake_prepare(
        loaded,
        *,
        task_output_dir=None,
        container_executable,
        network_mode="host",
        **_kwargs,
    ):
        return PreparedTraceSession(
            loaded=loaded,
            container=PreparedContainer(
                container_id="cid-openclaw-replay",
                container_executable=container_executable,
                docker_image="swebench/test-image",
                agent=_FakeAgent(),
            ),
        )

    async def fake_prefetch(*_args, **_kwargs) -> None:
        pass

    async def fake_prebuild(*_args, **_kwargs) -> dict[str, str]:
        return {}

    async def fail_on_simulator_side_llm_sleep(expected_s: float, *, phase: str):
        if phase == "llm_replay":
            raise AssertionError(
                "container-mode OpenClaw replay must not model LLM generation "
                "with simulator-side _sleep_and_measure"
            )
        return None

    launched_commands: list[list[str]] = []

    class _FakeStream:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks

        async def read(self, _n: int) -> bytes:
            return self._chunks.pop(0) if self._chunks else b""

    class _FakeWorkerProcess:
        returncode = 0

        def __init__(self, cmd: tuple[object, ...]) -> None:
            self._cmd = cmd
            self.stdout = _FakeStream([b"worker stdout"])
            self.stderr = _FakeStream([])

        async def wait(self) -> int:
            request_path = Path(self._cmd[-1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            Path(request["status_path"]).write_text(
                json.dumps(
                    {
                        "success": True,
                        "stop_reason": "completed",
                        "elapsed_s": 0.02,
                        "sleep_records": [
                            {
                                "phase": "llm_replay",
                                "expected_s": 0.2,
                                "actual_s": 0.2,
                            }
                        ],
                        "agent_execution_environment": "host",
                        "tool_execution_environment": "task_container",
                        "tool_container_id": request["container_id"],
                        "tool_container_user": "root",
                        "tool_container_user_id": "0",
                        "tool_container_workdir": "/testbed",
                        "openclaw_host_pid": 4321,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            Path(request["output_trace"]).write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "action",
                                "action_type": "llm_call",
                                "action_id": "llm_0",
                                "agent_id": request["run_instance_id"],
                                "iteration": 0,
                                "ts_start": 10.0,
                                "ts_end": 10.2,
                                "data": {"success": True},
                            }
                        ),
                        json.dumps(
                            {
                                "type": "summary",
                                "agent_id": request["run_instance_id"],
                                "success": True,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            return self.returncode

    async def fake_create_subprocess_exec(*cmd, **_kwargs):
        launched_commands.append(list(cmd))
        return _FakeWorkerProcess(cmd)

    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", fake_prepare)
    monkeypatch.setattr("trace_collect.simulator._prefetch_container_images", fake_prefetch)
    monkeypatch.setattr("trace_collect.simulator._prebuild_sweep_fixed_images", fake_prebuild)
    monkeypatch.setattr(
        "trace_collect.simulator._sleep_and_measure",
        fail_on_simulator_side_llm_sleep,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=2.0,
        )
    )

    assert trace_file.exists()
    records = [
        json.loads(line)
        for line in trace_file.read_text(encoding="utf-8").splitlines()
    ]
    summary = next(record for record in records if record["type"] == "summary")
    assert summary["agent_execution_environment"] == "host"
    assert summary["tool_execution_environment"] == "task_container"
    assert summary["tool_container_user"] == "root"
    assert summary["openclaw_host_pid"] == 4321
    assert summary["sleep_drift"]["by_phase"]["llm_replay"]["sample_count"] == 1
    assert summary["source_model"] == "qwen/qwen3.7-max"
    import sys

    assert len(launched_commands) == 1
    assert launched_commands[0][:3] == [
        sys.executable,
        "-m",
        "trace_collect.openclaw_host_replay_worker",
    ]
    assert launched_commands[0][3] == "--request"
    request = json.loads(Path(launched_commands[0][4]).read_text(encoding="utf-8"))
    assert request["container_id"] == "cid-openclaw-replay"
    assert request["task_instance_id"] == "fc_openclaw_replay"
    assert request["source_action_agent_id"] == "fc_openclaw_replay"
    assert request["source_model"] == "qwen/qwen3.7-max"


def test_openclaw_host_replay_worker_failure_marks_failed_with_audit_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failing host worker produces failed stats and source/task audit fields."""
    from harness.trace_logger import TraceLogger
    from trace_collect.simulator import (
        LLMTimingConfig,
        LoadedTraceSession,
        PreparedContainer,
        PreparedTraceSession,
        _run_openclaw_replay_session,
    )

    source_trace = tmp_path / "source.jsonl"
    source_trace.write_text("", encoding="utf-8")
    loaded = LoadedTraceSession(
        source_trace=source_trace,
        task_source=tmp_path / "tasks.json",
        run_instance_id="fc_openclaw_failed_replay",
        task_instance_id="fc_openclaw_failed_replay",
        source_action_agent_id="cli:oc-failed",
        manifest_index=0,
        scaffold="openclaw",
        metadata={"scaffold": "openclaw", "instance_id": "fc_openclaw_failed_replay"},
        summary={"success": True, "model": "source-model"},
        task={"problem_statement": "fix it"},
        actions=[
            {
                "type": "action",
                "action_type": "llm_call",
                "action_id": "llm_0",
                "agent_id": "cli:oc-failed",
                "iteration": 0,
                "ts_start": 1.0,
                "ts_end": 2.0,
                "data": {"messages_in": [], "raw_response": {"content": "run"}},
            },
            {
                "type": "action",
                "action_type": "tool_exec",
                "action_id": "tool_0",
                "agent_id": "cli:oc-failed",
                "iteration": 0,
                "ts_start": 2.0,
                "ts_end": 3.0,
                "data": {"tool_name": "exec", "tool_args": {"command": "pytest"}},
            },
        ],
        iterations={},
    )
    prepared = PreparedTraceSession(
        loaded=loaded,
        container=PreparedContainer(
            container_id="cid-openclaw-failed-replay",
            container_executable="docker",
            docker_image="swebench/test-image",
            agent=object(),
        ),
        task_output_dir=tmp_path / "task-output",
    )

    async def fake_worker_process(
        *,
        request_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        timeout_s: float,
    ) -> int:
        del timeout_s
        request = json.loads(request_path.read_text(encoding="utf-8"))
        assert request["task_instance_id"] == "fc_openclaw_failed_replay"
        assert request["source_action_agent_id"] == "cli:oc-failed"
        Path(request["status_path"]).write_text(
            json.dumps(
                {
                    "success": False,
                    "stop_reason": "error",
                    "error": "worker died",
                    "elapsed_s": 0.5,
                    "sleep_records": [],
                    "agent_execution_environment": "host",
                    "tool_execution_environment": "task_container",
                    "tool_container_id": request["container_id"],
                    "tool_container_user": "root",
                    "tool_container_user_id": "0",
                    "tool_container_workdir": "/testbed",
                    "openclaw_host_pid": 98765,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        stdout_path.write_text("stdout text", encoding="utf-8")
        stderr_path.write_text("stderr text", encoding="utf-8")
        return 7

    monkeypatch.setattr(
        "trace_collect.simulator._run_openclaw_worker_process",
        fake_worker_process,
    )
    trace_logger = TraceLogger(tmp_path / "replay-output", "replay")
    try:
        stats = asyncio.run(
            _run_openclaw_replay_session(
                prepared,
                trace_logger=trace_logger,
                replay_speed=1.0,
                llm_timing=LLMTimingConfig(),
                command_timeout_s=60.0,
            )
        )
    finally:
        trace_logger.close()

    records = [
        json.loads(line)
        for line in trace_logger.path.read_text(encoding="utf-8").splitlines()
    ]
    summary = next(record for record in records if record["type"] == "summary")
    assert stats.success is False
    assert stats.failed_action_count == 2
    assert summary["success"] is False
    assert summary["failed_actions"] == 2
    assert summary["task_instance_id"] == "fc_openclaw_failed_replay"
    assert summary["source_action_agent_id"] == "cli:oc-failed"
    assert summary["worker_returncode"] == 7
    assert summary["worker_error"] == "worker died"
    assert summary["agent_execution_environment"] == "host"
    assert summary["tool_execution_environment"] == "task_container"
    assert summary["tool_container_id"] == "cid-openclaw-failed-replay"
    assert summary["tool_container_user"] == "root"
    assert summary["tool_container_user_id"] == "0"
    assert summary["tool_container_workdir"] == "/testbed"
    assert summary["openclaw_host_pid"] == 98765


def test_terminal_bench_compose_preparation_uses_runner_env_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import subprocess

    from agents.terminal_bench.runner import TerminalBenchRunner
    from trace_collect.simulator import (
        LoadedTraceSession,
        _prepare_terminal_bench_container_session,
    )

    task_dir = tmp_path / "tb-task"
    task_dir.mkdir()
    (task_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (task_dir / "docker-compose.yaml").write_text(
        "services:\n  client:\n    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}\n",
        encoding="utf-8",
    )
    loaded = LoadedTraceSession(
        source_trace=tmp_path / "trace.jsonl",
        task_source=tmp_path / "tasks.json",
        run_instance_id="replay-run",
        task_instance_id="hello-world",
        source_action_agent_id="cli:oc-source",
        manifest_index=0,
        scaffold="openclaw",
        metadata={"instance_id": "hello-world"},
        summary={"model": "source-model"},
        task={
            "instance_id": "hello-world",
            "task_id": "hello-world",
            "task_source_kind": "terminal_bench_registry",
            "task_source_path": str(task_dir),
            "problem_statement": "fix it",
        },
        actions=[],
        iterations={},
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        if cmd[:2] == ["docker", "image"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="amd64 linux\n", stderr="")
        if cmd[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="/app/src\n", stderr="")
        if cmd[:2] == ["docker", "exec"]:
            stdout = "/usr/bin/python3\n" if "-s" in cmd else "/app/src\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        stdout = "cid-client\n" if cmd[-3:] == ["ps", "-q", "client"] else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("trace_collect.simulator.subprocess.run", fake_run)

    prepared = asyncio.run(
        _prepare_terminal_bench_container_session(
            loaded,
            task_output_dir=tmp_path / "task-output",
            container_executable="docker",
        )
    )
    assert prepared.container is not None
    prepared.container.cleanup_callback()

    expected_project = TerminalBenchRunner._expected_client_container_name(
        task_id="hello-world",
        run_id="replay-run",
    )
    expected_compose = (
        tmp_path
        / "task-output"
        / "terminal-bench-runtime"
        / "task"
        / "docker-compose.yaml"
    ).resolve()
    commands = [cmd for cmd, _kwargs in calls]
    assert commands == [
        [
            "docker",
            "compose",
            "-p",
            expected_project,
            "-f",
            str(expected_compose),
            "build",
        ],
        [
            "docker",
            "compose",
            "-p",
            expected_project,
            "-f",
            str(expected_compose),
            "up",
            "-d",
        ],
        [
            "docker",
            "compose",
            "-p",
            expected_project,
            "-f",
            str(expected_compose),
            "ps",
            "-q",
            "client",
        ],
        ["docker", "inspect", "--format", "{{.Config.WorkingDir}}", "cid-client"],
        [
            "docker",
            "exec",
            "-i",
            "--user",
            "0",
            "-w",
            "/app/src",
            "cid-client",
            "/bin/sh",
            "-c",
            "pwd",
        ],
        [
            "docker",
            "image",
            "inspect",
            "tb__hello-world__client",
            "--format",
            "{{.Architecture}} {{.Os}}",
        ],
        [
            "docker",
            "exec",
            "-i",
            "--user",
            "0",
            "-w",
            "/app/src",
            "cid-client",
            "/bin/sh",
            "-s",
            "--",
            "/usr/bin/python3",
            "/usr/bin/python",
            "/opt/miniconda3/bin/python3",
            "/opt/miniconda3/bin/python",
            "/opt/conda/bin/python3",
            "/opt/conda/bin/python",
            "python3",
            "python",
        ],
        [
            "docker",
            "compose",
            "-p",
            expected_project,
            "-f",
            str(expected_compose),
            "down",
            "--volumes",
            "--remove-orphans",
        ],
    ]
    for cmd, kwargs in calls:
        if cmd[:2] != ["docker", "compose"]:
            continue
        assert kwargs["cwd"] == expected_compose.parent
        env = kwargs["env"]
        assert isinstance(env, dict)
        assert env["T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME"] == expected_project
        assert env["T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME"] == "tb__hello-world__client"
        assert env["T_BENCH_TEST_DIR"] == "/tests"
        assert env["T_BENCH_CONTAINER_LOGS_PATH"] == "/logs"
        assert env["T_BENCH_CONTAINER_AGENT_LOGS_PATH"] == "/agent-logs"
    assert prepared.container.container_id == "cid-client"
    assert prepared.container.agent is None
    assert prepared.container.docker_image == "tb__hello-world__client"
    assert prepared.container.python_runtime == "/usr/bin/python3"
    assert prepared.container.pythonpath is None
    assert prepared.container.workdir == "/app/src"

def test_container_mode_trace_requires_container_executable(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "openclaw",
                        "instance_id": "fc_test_003",
                        "model": "dummy",
                        "mode": "collect",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "fc_test_003",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 2.0,
                        "data": {"messages_in": [], "completion_tokens": 1},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    task_source = tmp_path / "tasks.json"
    task_source.write_text(
        json.dumps(
            [
                {
                    "instance_id": "fc_test_003",
                    "problem_statement": "x",
                    "image_name": "swebench/test-image",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="container_executable is required"):
        asyncio.run(
            simulate(
                manifest=_single_trace_manifest(tmp_path, trace_path),
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="cloud_model",
            )
        )


def test_simulator_allows_duplicate_source_agent_ids_as_replay_replicas(
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "trace-a.jsonl"
    trace_b = tmp_path / "trace-b.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"

    for trace_path in (trace_a, trace_b):
        trace_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "trace_metadata",
                            "scaffold": "openclaw",
                            "instance_id": "same-id",
                            "model": "dummy",
                        }
                    ),
                    json.dumps(
                        {
                            "type": "action",
                            "action_type": "llm_call",
                            "action_id": "llm_0",
                            "agent_id": "same-id",
                            "iteration": 0,
                            "ts_start": 1.0,
                            "ts_end": 2.0,
                            "data": {"messages_in": [], "completion_tokens": 1},
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    task_source.write_text(
        json.dumps([{"instance_id": "same-id", "image_name": "img"}]) + "\n",
        encoding="utf-8",
    )
    _write_manifest(manifest, [str(trace_a), str(trace_b)])

    with pytest.raises(ValueError, match="same-id__replica-001"):
        asyncio.run(
            simulate(
                manifest=manifest,
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="cloud_model",
            )
        )


def test_simulator_rejects_relative_trace_paths_in_manifest(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    trace_path.write_text("", encoding="utf-8")
    task_source.write_text("[]\n", encoding="utf-8")
    _write_manifest(manifest, ["trace.jsonl"])

    with pytest.raises(SimulateError, match="absolute path"):
        asyncio.run(
            simulate(
                manifest=manifest,
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="cloud_model",
            )
        )


def test_simulator_requires_task_source_when_manifest_lacks_one(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    manifest = tmp_path / "manifest.yaml"
    trace_path.write_text("", encoding="utf-8")
    _write_manifest(manifest, [str(trace_path)])

    with pytest.raises(SimulateError, match="needs task_source"):
        asyncio.run(
            simulate(
                manifest=manifest,
                task_source=None,
                output_dir=tmp_path / "out",
                mode="cloud_model",
            )
        )
