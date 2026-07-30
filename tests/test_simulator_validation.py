from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from trace_collect.simulator import SimulateError, simulate


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


def test_ear_finalization_error_still_writes_throughput_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_paths = [tmp_path / "trace-a.jsonl", tmp_path / "trace-b.jsonl"]
    agent_ids = ["ear-finalize-a", "ear-finalize-b"]
    for trace_path, agent_id in zip(trace_paths, agent_ids, strict=True):
        trace_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "trace_metadata",
                            "instance_id": agent_id,
                            "scaffold": "openclaw",
                        }
                    ),
                    json.dumps(
                        {
                            "type": "action",
                            "action_type": "llm_call",
                            "action_id": "llm_0",
                            "agent_id": agent_id,
                            "iteration": 0,
                            "ts_start": 1.0,
                            "ts_end": 1.0,
                            "data": {"completion_tokens": 1},
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
                {"instance_id": agent_id, "image_name": "image"}
                for agent_id in agent_ids
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

    class FakeRuntime:
        def __init__(self, **_kwargs) -> None:
            pass

        def metadata(self) -> dict[str, object]:
            return {"mode": "fixed"}

        def write_artifacts(self, _output_dir: Path) -> None:
            raise RuntimeError("EAR artifact failure")

        @property
        def valid(self) -> bool:
            raise AssertionError("valid is unavailable after artifact failure")

    stopped: list[str] = []

    class FakeAgent:
        def __init__(self, agent_id: str) -> None:
            self.agent_id = agent_id

        async def stop(self) -> None:
            stopped.append(self.agent_id)
            if self.agent_id == "ear-finalize-a":
                raise RuntimeError("container stop failure")

    async def fake_run(loaded_sessions, *, container_executable, **_kwargs):
        prepared = [
            PreparedTraceSession(
                loaded=loaded,
                container=PreparedContainer(
                    container_id=f"fake-{loaded.agent_id}",
                    container_executable=container_executable,
                    docker_image="image",
                    agent=FakeAgent(loaded.agent_id),
                ),
            )
            for loaded in loaded_sessions
        ]
        stats = [
            ReplayTaskStats(
                agent_id=loaded.agent_id,
                run_instance_id=loaded.run_instance_id,
                source_agent_id=loaded.source_agent_id,
                manifest_index=loaded.manifest_index,
                label=loaded.label,
                source_trace=str(loaded.source_trace),
                success=True,
                elapsed_s=0.0,
                action_count=1,
                llm_call_count=1,
                tool_exec_count=0,
            )
            for loaded in loaded_sessions
        ]
        return prepared, stats

    async def no_prefetch(*_args, **_kwargs) -> None:
        pass

    async def no_prebuild(*_args, **_kwargs) -> dict[str, str]:
        return {}

    monkeypatch.setattr(
        "trace_collect.ear_replay_runtime.EarReplayRuntime",
        FakeRuntime,
    )
    monkeypatch.setattr("trace_collect.simulator._run_cloud_model_queue", fake_run)
    monkeypatch.setattr(
        "trace_collect.simulator._prefetch_container_images",
        no_prefetch,
    )
    monkeypatch.setattr(
        "trace_collect.simulator._prebuild_sweep_fixed_images",
        no_prebuild,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.stop_task_container",
        lambda *_args, **_kwargs: None,
    )

    output_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="container stop failure"):
        asyncio.run(
            simulate(
                manifest=_write_manifest(
                    tmp_path / "manifest.yaml",
                    [str(trace_path) for trace_path in trace_paths],
                ),
                task_source=task_source,
                output_dir=output_dir,
                container_executable="docker",
                concurrency=2,
                ear_mode="fixed",
                ear_policy=tmp_path / "policy.yaml",
                resource_monitoring="off",
                pmu_monitoring="off",
                memory_bandwidth_monitoring="off",
            )
        )

    summary = json.loads((output_dir / "throughput_summary.json").read_text())
    assert summary["ear_runtime"]["status"] == "invalid"
    assert sorted(stopped) == agent_ids


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
