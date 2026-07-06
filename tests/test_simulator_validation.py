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
                        "model": "dummy",
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
        ReplayTaskStats,
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

    provider_sleeps: list[dict[str, object]] = []

    async def fake_provider_sleep(self, expected_s: float, *, phase: str):
        from trace_collect.openclaw_host_runtime import ReplaySleepRecord

        record = ReplaySleepRecord(
            phase=phase,
            expected_s=expected_s,
            actual_s=expected_s,
        )
        self.sleep_records.append(record)
        provider_sleeps.append(record.to_dict())
        return record

    async def fake_run_openclaw_replay_session(
        prepared_session,
        *,
        trace_logger,
        replay_speed,
        llm_timing,
        command_timeout_s,
        warmup_skip_iterations,
        **_kwargs,
    ):
        from trace_collect.openclaw_host_runtime import OpenClawReplayProvider

        loaded = prepared_session.loaded
        provider = OpenClawReplayProvider(
            llm_actions=[
                action
                for action in loaded.actions
                if action["action_type"] == "llm_call"
            ],
            replay_speed=replay_speed,
            timing_mode=llm_timing.mode,
            llm_ttft_ms=llm_timing.ttft_ms,
            llm_tpot_ms=llm_timing.tpot_ms,
            model="replay-openclaw",
        )
        await provider.chat(messages=[])
        return ReplayTaskStats(
            agent_id=loaded.run_instance_id,
            run_instance_id=loaded.run_instance_id,
            source_agent_id=loaded.source_agent_id,
            manifest_index=loaded.manifest_index,
            label=loaded.label,
            source_trace=str(loaded.source_trace),
            success=True,
            elapsed_s=0.01,
            action_count=len(loaded.actions),
            llm_call_count=1,
            tool_exec_count=0,
        )

    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", fake_prepare)
    monkeypatch.setattr("trace_collect.simulator._prefetch_container_images", fake_prefetch)
    monkeypatch.setattr("trace_collect.simulator._prebuild_sweep_fixed_images", fake_prebuild)
    monkeypatch.setattr(
        "trace_collect.simulator._sleep_and_measure",
        fail_on_simulator_side_llm_sleep,
    )
    monkeypatch.setattr(
        "trace_collect.openclaw_host_runtime.OpenClawReplayProvider._sleep",
        fake_provider_sleep,
    )
    monkeypatch.setattr(
        "trace_collect.simulator._run_openclaw_replay_session",
        fake_run_openclaw_replay_session,
        raising=False,
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
    assert provider_sleeps == [
        {
            "phase": "llm_replay",
            "expected_s": pytest.approx(0.2),
            "actual_s": pytest.approx(0.2),
            "drift_s": pytest.approx(0.0),
        }
    ]


def test_openclaw_host_replay_marks_failed_emitted_and_missing_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Failed emitted actions and missing source-action coverage make replay fail."""
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
        source_agent_id="fc_openclaw_failed_replay",
        run_instance_id="fc_openclaw_failed_replay",
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
                "agent_id": "fc_openclaw_failed_replay",
                "iteration": 0,
                "ts_start": 1.0,
                "ts_end": 2.0,
                "data": {"messages_in": [], "raw_response": {"content": "run"}},
            },
            {
                "type": "action",
                "action_type": "tool_exec",
                "action_id": "tool_0",
                "agent_id": "fc_openclaw_failed_replay",
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

    class FakeSessionRunner:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def run(self, **kwargs):
            trace_file = kwargs["trace_file"]
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "action",
                                "action_type": "tool_exec",
                                "action_id": "tool_0",
                                "agent_id": "fc_openclaw_failed_replay",
                                "iteration": 0,
                                "ts_start": 2.0,
                                "ts_end": 3.0,
                                "data": {"success": False},
                            }
                        ),
                        json.dumps(
                            {
                                "type": "summary",
                                "agent_id": "fc_openclaw_failed_replay",
                                "success": True,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            class Result:
                stop_reason = "completed"
                error = None

            return Result()

    monkeypatch.setattr(
        "agents.openclaw._session_runner.SessionRunner",
        FakeSessionRunner,
    )
    monkeypatch.setattr(
        "trace_collect.openclaw_host_runtime.build_container_tools_for_agent",
        lambda *_args, **_kwargs: [],
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
