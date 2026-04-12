from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from trace_collect.cli import _run_simulate, parse_simulate_args
from trace_collect.simulator import simulate


class _FakeStream:
    def __init__(self) -> None:
        self._emitted = False

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self):
        if self._emitted:
            raise StopAsyncIteration
        self._emitted = True
        await asyncio.sleep(0.01)
        delta = type("Delta", (), {"content": "x"})()
        choice = type("Choice", (), {"delta": delta})()
        return type("Chunk", (), {"choices": [choice]})()


class _FakeClient:
    class _Completions:
        async def create(self, **kwargs):
            return _FakeStream()

    class _Chat:
        def __init__(self) -> None:
            self.completions = _FakeClient._Completions()

    def __init__(self) -> None:
        self.chat = self._Chat()


def _write_trace(
    path: Path,
    *,
    agent_id: str,
    scaffold: str = "openclaw",
    llm_start: float = 100.0,
    llm_end: float = 100.2,
    tool_start: float = 100.4,
    tool_end: float = 100.45,
    tool_name: str = "write_file",
) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": scaffold,
                        "instance_id": agent_id,
                        "model": "claude-haiku",
                        "mode": "collect",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": f"{agent_id}-llm-0",
                        "agent_id": agent_id,
                        "iteration": 0,
                        "ts_start": llm_start,
                        "ts_end": llm_end,
                        "data": {
                            "messages_in": [{"role": "user", "content": "fix bug"}],
                            "raw_response": {"id": f"resp-{agent_id}"},
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "llm_latency_ms": (llm_end - llm_start) * 1000,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "tool_exec",
                        "action_id": f"{agent_id}-tool-0",
                        "agent_id": agent_id,
                        "iteration": 0,
                        "ts_start": tool_start,
                        "ts_end": tool_end,
                        "data": {
                            "tool_name": tool_name,
                            "tool_args": json.dumps({"path": f"/testbed/x.txt"}),
                            "tool_result": "source-result",
                            "duration_ms": (tool_end - tool_start) * 1000,
                            "success": True,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "summary",
                        "agent_id": agent_id,
                        "model": "claude-haiku",
                        "success": True,
                        "n_iterations": 1,
                        "elapsed_s": tool_end - llm_start,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_tasks(path: Path, *agent_ids: str) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "instance_id": agent_id,
                    "problem_statement": f"problem for {agent_id}",
                    "repo": "django/django",
                    "base_commit": "deadbeef",
                    "image_name": f"swebench-test/{agent_id}",
                }
                for agent_id in agent_ids
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _patch_simulator_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    tool_delay_s: float = 0.0,
    tool_duration_ms: float = 8.0,
    tool_result_prefix: str = "executed",
    llm_client_mode: str = "forbid",
) -> None:
    class _FakeAgent:
        async def stop(self): pass

    async def fake_prepare_container(loaded, *, container_executable):
        from trace_collect.simulator import PreparedContainer, PreparedTraceSession
        container = PreparedContainer(
            container_id="fake-cid",
            container_executable=container_executable,
            docker_image="fake-image",
            agent=_FakeAgent(),
            cleanup=lambda: None,
        )
        return PreparedTraceSession(loaded=loaded, container=container)

    async def fake_exec_tool(agent, tool_name, tool_args_json, command_timeout_s):
        if tool_delay_s > 0:
            await asyncio.sleep(tool_delay_s)
        return f"{tool_result_prefix}-{tool_name}", tool_duration_ms, True

    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", fake_prepare_container)
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    if llm_client_mode == "forbid":
        monkeypatch.setattr(
            "trace_collect.simulator.create_async_openai_client",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("cloud_model must not create llm client")
            ),
        )
    elif llm_client_mode == "fake":
        monkeypatch.setattr(
            "trace_collect.simulator.create_async_openai_client",
            lambda **_kwargs: _FakeClient(),
        )
    else:
        raise AssertionError(f"unknown llm_client_mode: {llm_client_mode}")


def test_parse_simulate_args_accepts_cloud_model_manifest_without_llm_args() -> None:
    args = parse_simulate_args(
        [
            "--mode",
            "cloud_model",
            "--trace-manifest",
            "manifest.json",
        ]
    )

    assert args.mode == "cloud_model"
    assert args.trace_manifest == "manifest.json"
    assert args.source_trace is None
    assert args.replay_speed == 1.0


def test_parse_simulate_args_accepts_container_flag() -> None:
    args = parse_simulate_args(
        [
            "--mode",
            "cloud_model",
            "--source-trace",
            "trace.jsonl",
            "--container",
            "podman",
        ]
    )
    assert args.container == "podman"


def test_parse_simulate_args_defaults_container_to_docker() -> None:
    args = parse_simulate_args(
        ["--source-trace", "trace.jsonl"]
    )
    assert args.container == "docker"


def test_run_simulate_cloud_model_bypasses_llm_config(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    async def fake_simulate(**kwargs):
        seen.update(kwargs)
        return tmp_path / "out.jsonl"

    monkeypatch.setattr(
        "trace_collect.cli.resolve_llm_config",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not resolve llm config")),
    )
    monkeypatch.setattr("trace_collect.simulator.simulate", fake_simulate)

    args = parse_simulate_args(
        [
            "--mode",
            "cloud_model",
            "--source-trace",
            "trace.jsonl",
        ]
    )

    _run_simulate(args)

    assert seen["mode"] == "cloud_model"
    assert seen["source_trace"] == Path("trace.jsonl")
    assert seen["trace_manifest"] is None
    assert seen["container_executable"] == "docker"


def test_run_simulate_rejects_metrics_url_in_cloud_model(capsys: pytest.CaptureFixture[str]) -> None:
    args = parse_simulate_args(
        [
            "--mode",
            "cloud_model",
            "--source-trace",
            "trace.jsonl",
            "--metrics-url",
            "http://localhost:8000/metrics",
        ]
    )

    with pytest.raises(SystemExit, match="2"):
        _run_simulate(args)

    assert "cloud_model replay does not support --metrics-url" in capsys.readouterr().err


def test_run_simulate_local_model_resolves_llm_config(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    async def fake_simulate(**kwargs):
        seen.update(kwargs)
        return tmp_path / "out.jsonl"

    monkeypatch.setattr(
        "trace_collect.cli.resolve_llm_config",
        lambda **_kwargs: type(
            "Cfg",
            (),
            {
                "api_base": "https://example.com/v1",
                "api_key": "secret",
                "model": "z-ai/glm-5.1",
                "env_key": "OPENROUTER_API_KEY",
            },
        )(),
    )
    monkeypatch.setattr("trace_collect.simulator.simulate", fake_simulate)

    args = parse_simulate_args(
        [
            "--mode",
            "local_model",
            "--source-trace",
            "trace.jsonl",
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--api-base",
            "https://ignored.example/v1",
        ]
    )

    _run_simulate(args)

    assert seen["mode"] == "local_model"
    assert seen["api_base"] == "https://example.com/v1"
    assert seen["api_key"] == "secret"
    assert seen["model"] == "z-ai/glm-5.1"


def test_run_simulate_local_model_rejects_trace_manifest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = parse_simulate_args(
        [
            "--mode",
            "local_model",
            "--trace-manifest",
            "manifest.json",
        ]
    )

    with pytest.raises(SystemExit, match="2"):
        _run_simulate(args)

    assert "local_model mode accepts only --source-trace" in capsys.readouterr().err


def test_cloud_model_single_trace_replays_without_llm_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(
        monkeypatch,
        tmp_path,
        tool_delay_s=0.01,
        tool_duration_ms=10.0,
        llm_client_mode="forbid",
    )

    started = time.monotonic()
    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            replay_speed=10.0,
        )
    )
    elapsed = time.monotonic() - started

    records = _read_jsonl(trace_file)
    assert elapsed >= 0.03
    assert records[0]["simulate_mode"] == "cloud_model"
    assert records[0]["source_model"] == "claude-haiku"
    assert "local_model" not in records[0]
    llm_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    )
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    summary = next(record for record in records if record.get("type") == "summary")

    assert llm_record["data"]["replay_mode"] == "cloud_model"
    assert llm_record["data"]["source_llm_latency_ms"] == pytest.approx(200.0)
    assert llm_record["data"]["sim_metrics"]["warmup"] is False
    assert tool_record["data"]["replay_source"] == "executed_in_container"
    assert tool_record["data"]["source_duration_ms"] == pytest.approx(50.0)
    assert tool_record["data"]["tool_result"] == "executed-write_file"
    assert tool_record["data"]["sim_metrics"]["warmup"] is False
    assert tool_record["data"]["sim_metrics"]["sim_tool_format"] == "container_exec"
    assert summary["success"] is True
    assert summary["source_success"] is True
    assert summary["replay_mode"] == "cloud_model"
    assert summary["replay_speed"] == pytest.approx(10.0)


def test_cloud_model_replay_marks_warmup_iterations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path, llm_client_mode="forbid")

    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            replay_speed=10.0,
            warmup_skip_iterations=1,
        )
    )

    records = _read_jsonl(trace_file)
    llm_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    )
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    assert llm_record["data"]["sim_metrics"]["warmup"] is True
    assert tool_record["data"]["sim_metrics"]["warmup"] is True


def test_local_model_single_trace_still_emits_sim_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path, llm_client_mode="fake")

    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="local_model",
            api_base="https://example.com/v1",
            api_key="secret",
            model="local-qwen",
        )
    )

    records = _read_jsonl(trace_file)
    metadata = records[0]
    llm_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    )
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    assert metadata["simulate_mode"] == "local_model"
    assert metadata["local_model"] == "local-qwen"
    assert llm_record["data"]["sim_metrics"]["timing"]["total_ms"] >= 0.0
    assert llm_record["data"]["source_llm_latency_ms"] == pytest.approx(200.0)
    assert tool_record["data"]["sim_metrics"]["source"] == "executed_in_container"
    assert tool_record["data"]["tool_result"] == "executed-write_file"
    summary = next(record for record in records if record.get("type") == "summary")
    assert summary["success"] is True
    assert summary["source_success"] is True


def test_cloud_model_trace_manifest_replays_multiple_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "trace-a.jsonl"
    trace_b = tmp_path / "trace-b.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.json"
    _write_trace(trace_a, agent_id="task-a", llm_start=100.0, llm_end=100.05, tool_start=100.1, tool_end=100.12)
    _write_trace(trace_b, agent_id="task-b", llm_start=200.0, llm_end=200.05, tool_start=200.1, tool_end=200.12)
    _write_tasks(task_source, "task-a", "task-b")
    manifest.write_text(
        json.dumps(
            [
                {"source_trace": trace_a.name},
                {"source_trace": trace_b.name, "task_source": task_source.name},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _patch_simulator_runtime(
        monkeypatch,
        tmp_path,
        tool_delay_s=0.02,
        tool_duration_ms=20.0,
        tool_result_prefix="ok",
        llm_client_mode="forbid",
    )

    trace_file = asyncio.run(
        simulate(
            trace_manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    metadata = records[0]
    summaries = [record for record in records if record.get("type") == "summary"]
    llm_records = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    ]

    assert metadata["trace_manifest"] == str(manifest)
    assert metadata["source_trace_count"] == 2
    assert set(metadata["source_traces"]) == {str(trace_a), str(trace_b)}
    assert {record["agent_id"] for record in summaries} == {"task-a", "task-b"}
    assert {record["agent_id"] for record in llm_records} == {"task-a", "task-b"}
    assert abs(llm_records[0]["ts_start"] - llm_records[1]["ts_start"]) < 0.05


def test_cloud_model_manifest_with_docker_image_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Manifest-level docker_image overrides task image_name."""
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.json"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    manifest.write_text(
        json.dumps(
            [{"source_trace": trace_path.name, "docker_image": "custom/override:latest"}]
        )
        + "\n",
        encoding="utf-8",
    )

    prepared_images: list[str] = []

    class _FakeAgent2:
        async def stop(self): pass

    async def capture_prepare(loaded, *, container_executable):
        from trace_collect.simulator import PreparedContainer, PreparedTraceSession, _resolve_docker_image
        img = _resolve_docker_image(loaded)
        prepared_images.append(img)
        container = PreparedContainer(
            container_id="fake-cid",
            container_executable=container_executable,
            docker_image=img or "",
            agent=_FakeAgent2(),
            cleanup=lambda: None,
        )
        return PreparedTraceSession(loaded=loaded, container=container)

    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", capture_prepare)
    async def _fake_exec(*a, **kw):
        return ("ok", 1.0, True)

    monkeypatch.setattr("trace_collect.simulator._exec_tool", _fake_exec)
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("no llm")),
    )

    asyncio.run(
        simulate(
            trace_manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
        )
    )

    assert prepared_images == ["custom/override:latest"]


def test_cloud_model_rejects_task_without_docker_image(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a")
    # Task without image_name or docker_image
    task_source.write_text(
        json.dumps([{"instance_id": "task-a", "problem_statement": "x"}]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="no resolvable docker_image"):
        asyncio.run(
            simulate(
                source_trace=trace_path,
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="cloud_model",
            )
        )


def test_cloud_model_manifest_keeps_default_task_source_cwd_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    manifest_dir = tmp_path / "manifests"
    manifest = manifest_dir / "manifest.json"
    task_source = tmp_path / "tasks.json"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    manifest.write_text(
        json.dumps([{"source_trace": "../trace.jsonl"}]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _patch_simulator_runtime(
        monkeypatch,
        tmp_path,
        tool_duration_ms=5.0,
        tool_result_prefix="ok",
        llm_client_mode="forbid",
    )

    trace_file = asyncio.run(
        simulate(
            trace_manifest=manifest,
            task_source=Path("tasks.json"),
            output_dir=tmp_path / "out",
            mode="cloud_model",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    assert any(record.get("type") == "summary" for record in records)
