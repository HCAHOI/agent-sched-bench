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
    execution_environment: str = "container",
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
                        "execution_environment": execution_environment,
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
                            "tool_args": json.dumps({"path": "/testbed/x.txt"}),
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


def _write_host_tasks(path: Path, *agent_ids: str) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "instance_id": agent_id,
                    "problem_statement": f"problem for {agent_id}",
                    "repo": None,
                    "image_name": None,
                    "docker_image": None,
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

    async def fake_prepare_container(
        loaded,
        *,
        task_output_dir=None,
        container_executable,
        network_mode="host",
    ):
        from trace_collect.simulator import PreparedContainer, PreparedTraceSession
        container = PreparedContainer(
            container_id="fake-cid",
            container_executable=container_executable,
            docker_image="fake-image",
            agent=_FakeAgent(),
        )
        return PreparedTraceSession(loaded=loaded, container=container)

    async def fake_exec_tool(agent, tool_name, tool_args_json, command_timeout_s):
        if tool_delay_s > 0:
            await asyncio.sleep(tool_delay_s)
        return f"{tool_result_prefix}-{tool_name}", tool_duration_ms, True

    async def fake_prefetch(*_args, **_kwargs) -> None:
        pass

    class _FakeSampler:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> list[dict]:
            return []

    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", fake_prepare_container)
    monkeypatch.setattr("trace_collect.simulator._prefetch_container_images", fake_prefetch)
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
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
            "--manifest",
            "manifest.yaml",
            "--concurrency",
            "8",
        ]
    )

    assert args.mode == "cloud_model"
    assert args.manifest == "manifest.yaml"
    assert args.concurrency == "8"
    assert args.replay_speed == 1.0


def test_parse_simulate_args_accepts_container_flag() -> None:
    args = parse_simulate_args(
        [
            "--mode",
            "cloud_model",
            "--manifest",
            "manifest.yaml",
            "--container",
            "podman",
        ]
    )
    assert args.container == "podman"


def test_parse_simulate_args_defaults_container_to_none() -> None:
    args = parse_simulate_args(
        ["--manifest", "manifest.yaml"]
    )
    assert args.container is None


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
            "--manifest",
            "manifest.yaml",
        ]
    )

    _run_simulate(args)

    assert seen["mode"] == "cloud_model"
    assert seen["manifest"] == Path("manifest.yaml")
    assert seen["concurrency"] == 1
    assert seen["container_executable"] is None


def test_run_simulate_cloud_model_concurrency_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[int] = []

    async def fake_simulate(**kwargs):
        concurrency = int(kwargs["concurrency"])
        seen.append(concurrency)
        trace_file = tmp_path / f"simulate_fake_c{concurrency}.jsonl"
        trace_file.write_text("", encoding="utf-8")
        trace_file.with_name(f"{trace_file.stem}.throughput_summary.json").write_text(
            json.dumps({"concurrency": concurrency, "run_id": f"c{concurrency}"}) + "\n",
            encoding="utf-8",
        )
        return trace_file

    monkeypatch.setattr(
        "trace_collect.cli.resolve_llm_config",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not resolve llm config")),
    )
    monkeypatch.setattr("trace_collect.simulator.simulate", fake_simulate)

    args = parse_simulate_args(
        [
            "--mode",
            "cloud_model",
            "--manifest",
            "manifest.yaml",
            "--output-dir",
            str(tmp_path),
            "--concurrency",
            "2,4",
        ]
    )

    _run_simulate(args)

    assert seen == [2, 4]
    sweep_records = _read_jsonl(tmp_path / "throughput_sweep.jsonl")
    assert [record["concurrency"] for record in sweep_records] == [2, 4]


def test_run_simulate_rejects_metrics_url_in_cloud_model(capsys: pytest.CaptureFixture[str]) -> None:
    args = parse_simulate_args(
        [
            "--mode",
            "cloud_model",
            "--manifest",
            "manifest.yaml",
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
            "--manifest",
            "manifest.yaml",
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
    assert seen["manifest"] == Path("manifest.yaml")
    assert seen["api_base"] == "https://example.com/v1"
    assert seen["api_key"] == "secret"
    assert seen["model"] == "z-ai/glm-5.1"


def test_run_simulate_local_model_rejects_concurrency_sweep(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = parse_simulate_args(
        [
            "--mode",
            "local_model",
            "--manifest",
            "manifest.yaml",
            "--concurrency",
            "1,2",
        ]
    )

    with pytest.raises(SystemExit, match="2"):
        _run_simulate(args)

    assert "local_model mode requires --concurrency 1" in capsys.readouterr().err


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
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
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


def test_cloud_model_host_trace_replays_without_container_or_llm_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="host-task",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_host_tasks(task_source, "host-task")

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode replay must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        fail_prepare,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("cloud_model must not create llm client")
        ),
    )

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    metadata = records[0]
    llm_records = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    ]
    tool_records = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    ]
    summary = next(record for record in records if record.get("type") == "summary")

    assert metadata["execution_environment"] == "host"
    assert len(llm_records) == 1
    assert llm_records[0]["data"]["sim_metrics"]["warmup"] is False
    assert len(tool_records) == 1
    assert tool_records[0]["data"]["replay_source"] == "skipped_host_mode"
    assert tool_records[0]["data"]["success"] is True
    assert tool_records[0]["data"]["sim_metrics"]["sim_tool_format"] == "skipped_host_mode"
    assert summary["success"] is True

    # Regression: host-mode replay must still emit an empty resources.json so
    # downstream consumers can rely on canonical simulate layout.
    attempt_dir = (tmp_path / "out" / "host-task" / "attempt_1")
    resources_path = attempt_dir / "resources.json"
    assert resources_path.exists(), (
        "host-mode replay must write resources.json even without a sampler"
    )
    payload = json.loads(resources_path.read_text())
    assert payload["samples"] == []
    assert payload["summary"]["sample_count"] == 0

    startup_path = attempt_dir / "container_startup.json"
    assert startup_path.exists()
    startup = json.loads(startup_path.read_text())
    assert startup["status"] == "skipped"
    assert startup["reason"] == "host_execution_environment"
    assert startup["phases"] == []
    assert startup["resources"]["samples"] == []
    assert startup["resources"]["summary"]["sample_count"] == 0


def test_cloud_model_mixed_manifest_requires_container_before_replay(tmp_path: Path) -> None:
    trace_container = tmp_path / "trace-container.jsonl"
    trace_host = tmp_path / "trace-host.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    output_dir = tmp_path / "out"
    _write_trace(trace_container, agent_id="container-task")
    _write_trace(
        trace_host,
        agent_id="host-task",
        execution_environment="host",
    )
    _write_tasks(task_source, "container-task", "host-task")
    _write_manifest(manifest, [str(trace_host), str(trace_container)])

    with pytest.raises(ValueError, match="container_executable is required"):
        asyncio.run(
            simulate(
                manifest=manifest,
                task_source=task_source,
                output_dir=output_dir,
                mode="cloud_model",
                concurrency=2,
            )
        )

    assert not output_dir.exists()


def test_cloud_model_prefetches_images_before_container_prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "trace-a.jsonl"
    trace_b = tmp_path / "trace-b.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    _write_trace(trace_a, agent_id="task-a")
    _write_trace(trace_b, agent_id="task-b")
    task_source.write_text(
        json.dumps(
            [
                {
                    "instance_id": "task-a",
                    "problem_statement": "problem a",
                    "image_name": "shared/image:latest",
                },
                {
                    "instance_id": "task-b",
                    "problem_statement": "problem b",
                    "image_name": "shared/image:latest",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_manifest(manifest, [str(trace_a), str(trace_b)])

    events: list[tuple[str, str]] = []

    def fake_ensure_source_image(image, *, container_executable):
        events.append(("prefetch", image))

    async def fake_prepare_container(
        loaded,
        *,
        task_output_dir=None,
        container_executable,
        network_mode="host",
    ):
        from trace_collect.simulator import PreparedContainer, PreparedTraceSession

        events.append(("prepare", loaded.agent_id))

        class _FakeAgent:
            async def stop(self) -> None:
                pass

        return PreparedTraceSession(
            loaded=loaded,
            container=PreparedContainer(
                container_id=f"fake-{loaded.agent_id}",
                container_executable=container_executable,
                docker_image="fake-image",
                agent=_FakeAgent(),
            ),
        )

    class _FakeSampler:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> list[dict]:
            return []

    async def fake_exec_tool(*_args, **_kwargs):
        return ("ok", 1.0, True)

    monkeypatch.setattr("trace_collect.simulator.ensure_source_image", fake_ensure_source_image)
    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", fake_prepare_container)
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("no llm")),
    )

    asyncio.run(
        simulate(
            manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            concurrency=2,
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    assert events[0] == ("prefetch", "docker.io/shared/image:latest")
    assert [event for event in events if event[0] == "prefetch"] == [
        ("prefetch", "docker.io/shared/image:latest")
    ]
    assert all(event[0] == "prepare" for event in events[1:])


def test_cloud_model_prefetch_failure_happens_before_output_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")

    def fail_ensure_source_image(*_args, **_kwargs) -> None:
        raise RuntimeError("pull failed")

    monkeypatch.setattr("trace_collect.simulator.ensure_source_image", fail_ensure_source_image)

    with pytest.raises(RuntimeError, match="pull failed"):
        asyncio.run(
            simulate(
                manifest=_single_trace_manifest(tmp_path, trace_path),
                task_source=task_source,
                output_dir=output_dir,
                mode="cloud_model",
                container_executable="docker",
            )
        )

    assert not output_dir.exists()


def test_cloud_model_prefetch_uses_manifest_docker_image_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    _write_manifest(
        manifest,
        [{"trace": trace_path, "docker_image": "custom/override:latest"}],
    )

    prefetched_images: list[str] = []

    def fake_ensure_source_image(image, *, container_executable):
        prefetched_images.append(image)

    async def fake_prepare_container(
        loaded,
        *,
        task_output_dir=None,
        container_executable,
        network_mode="host",
    ):
        from trace_collect.simulator import PreparedContainer, PreparedTraceSession

        class _FakeAgent:
            async def stop(self) -> None:
                pass

        return PreparedTraceSession(
            loaded=loaded,
            container=PreparedContainer(
                container_id="fake-cid",
                container_executable=container_executable,
                docker_image="fake-image",
                agent=_FakeAgent(),
            ),
        )

    class _FakeSampler:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> list[dict]:
            return []

    async def fake_exec_tool(*_args, **_kwargs):
        return ("ok", 1.0, True)

    monkeypatch.setattr("trace_collect.simulator.ensure_source_image", fake_ensure_source_image)
    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", fake_prepare_container)
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("no llm")),
    )

    asyncio.run(
        simulate(
            manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    assert prefetched_images == ["docker.io/custom/override:latest"]


def test_cloud_model_container_startup_json_records_success_and_separates_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")

    class _FakeContainerAgent:
        def __init__(self, container_id: str, container_executable: str) -> None:
            assert container_id == "fake-cid"
            assert container_executable == "docker"

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    class _FakeSampler:
        def __init__(self, *, container_id: str, interval_s: float, executable: str) -> None:
            assert container_id == "fake-cid"
            assert executable == "docker"
            self.interval_s = interval_s

        def start(self) -> None:
            pass

        def stop(self) -> list[dict]:
            kind = "startup" if self.interval_s == 0.25 else "runtime"
            return [
                {
                    "timestamp": "2026-06-26T00:00:00Z",
                    "container_id": "fake-cid",
                    "phase": kind,
                    "cpu_percent": 1.0,
                    "memory_mib": 2.0,
                }
            ]

    async def fake_exec_tool(*_args, **_kwargs):
        return ("ok", 1.0, True)

    monkeypatch.setattr("trace_collect.simulator.ensure_source_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "trace_collect.simulator.ensure_fixed_image",
        lambda *args, **kwargs: ("fixed-image", 0.125),
    )
    monkeypatch.setattr(
        "trace_collect.simulator.start_task_container",
        lambda *args, **kwargs: "fake-cid",
    )
    monkeypatch.setattr("trace_collect.simulator.stop_task_container", lambda *args, **kwargs: "")
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
    monkeypatch.setattr("trace_collect.openclaw_tools.ContainerAgent", _FakeContainerAgent)
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("no llm")),
    )

    asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=output_dir,
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    attempt_dir = output_dir / "task-a" / "attempt_1"
    startup = json.loads((attempt_dir / "container_startup.json").read_text())
    resources = json.loads((attempt_dir / "resources.json").read_text())

    assert startup["status"] == "success"
    assert startup["agent_id"] == "task-a"
    assert startup["source_image"] == "docker.io/swebench-test/task-a"
    assert startup["fixed_image"] == "fixed-image"
    assert startup["container_id"] == "fake-cid"
    assert [phase["name"] for phase in startup["phases"]] == [
        "ensure_fixed_image",
        "start_task_container",
        "container_agent_start",
    ]
    assert startup["phases"][0]["reported_elapsed_s"] == pytest.approx(0.125)
    assert startup["resources"]["samples"][0]["phase"] == "startup"
    assert startup["resources"]["summary"]["sample_count"] == 1
    assert resources["samples"][0]["phase"] == "runtime"
    assert resources["summary"]["sample_count"] == 1


def test_cloud_model_agent_start_failure_writes_failed_container_startup_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    stopped_containers: list[str] = []

    class _FailingContainerAgent:
        def __init__(self, container_id: str, container_executable: str) -> None:
            assert container_id == "fake-cid"
            assert container_executable == "docker"

        async def start(self) -> None:
            raise RuntimeError("agent failed")

        async def stop(self) -> None:
            raise AssertionError("failed startup agent must not be finalized later")

    class _FakeSampler:
        def __init__(self, *, container_id: str, interval_s: float, executable: str) -> None:
            assert container_id == "fake-cid"
            assert interval_s == 0.25
            assert executable == "docker"

        def start(self) -> None:
            pass

        def stop(self) -> list[dict]:
            return [
                {
                    "timestamp": "2026-06-26T00:00:00Z",
                    "container_id": "fake-cid",
                    "phase": "startup",
                    "cpu_percent": 1.0,
                    "memory_mib": 2.0,
                }
            ]

    def fake_stop_task_container(container_id: str, *, executable: str) -> str:
        assert executable == "docker"
        stopped_containers.append(container_id)
        return ""

    monkeypatch.setattr("trace_collect.simulator.ensure_source_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "trace_collect.simulator.ensure_fixed_image",
        lambda *args, **kwargs: ("fixed-image", 0.125),
    )
    monkeypatch.setattr(
        "trace_collect.simulator.start_task_container",
        lambda *args, **kwargs: "fake-cid",
    )
    monkeypatch.setattr("trace_collect.simulator.stop_task_container", fake_stop_task_container)
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
    monkeypatch.setattr("trace_collect.openclaw_tools.ContainerAgent", _FailingContainerAgent)

    with pytest.raises(RuntimeError, match="agent failed"):
        asyncio.run(
            simulate(
                manifest=_single_trace_manifest(tmp_path, trace_path),
                task_source=task_source,
                output_dir=output_dir,
                mode="cloud_model",
                container_executable="docker",
                replay_speed=100.0,
            )
        )

    startup = json.loads(
        (output_dir / "task-a" / "attempt_1" / "container_startup.json").read_text()
    )
    assert startup["status"] == "failed"
    assert startup["error"]["type"] == "RuntimeError"
    assert startup["error"]["message"] == "agent failed"
    assert startup["phases"][-1]["name"] == "container_agent_start"
    assert startup["phases"][-1]["status"] == "failed"
    assert startup["phases"][-1]["error"]["message"] == "agent failed"
    assert startup["resources"]["samples"][0]["phase"] == "startup"
    assert stopped_containers == ["fake-cid"]


def test_prepare_container_session_stops_started_agent_when_startup_sampler_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trace_collect.simulator import _load_trace_session, _prepare_container_session

    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    attempt_dir = tmp_path / "attempt_1"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    loaded = _load_trace_session(trace_path, task_source)
    agent_stops = 0
    stopped_containers: list[str] = []

    class _FakeContainerAgent:
        def __init__(self, container_id: str, container_executable: str) -> None:
            assert container_id == "fake-cid"
            assert container_executable == "docker"

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            nonlocal agent_stops
            agent_stops += 1

    class _FailingStartupSampler:
        def __init__(self, *, container_id: str, interval_s: float, executable: str) -> None:
            assert container_id == "fake-cid"
            assert interval_s == 0.25
            assert executable == "docker"

        def start(self) -> None:
            pass

        def stop(self) -> list[dict]:
            raise RuntimeError("startup sampler failed")

    def fake_stop_task_container(container_id: str, *, executable: str) -> str:
        assert executable == "docker"
        stopped_containers.append(container_id)
        return ""

    monkeypatch.setattr(
        "trace_collect.simulator.ensure_fixed_image",
        lambda *args, **kwargs: ("fixed-image", 0.125),
    )
    monkeypatch.setattr(
        "trace_collect.simulator.start_task_container",
        lambda *args, **kwargs: "fake-cid",
    )
    monkeypatch.setattr("trace_collect.simulator.stop_task_container", fake_stop_task_container)
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FailingStartupSampler)
    monkeypatch.setattr("trace_collect.openclaw_tools.ContainerAgent", _FakeContainerAgent)

    with pytest.raises(RuntimeError, match="startup sampler failed"):
        asyncio.run(
            _prepare_container_session(
                loaded,
                task_output_dir=attempt_dir,
                container_executable="docker",
            )
        )

    startup = json.loads((attempt_dir / "container_startup.json").read_text())
    assert startup["status"] == "failed"
    assert startup["error"]["message"] == "startup sampler failed"
    assert agent_stops == 1
    assert stopped_containers == ["fake-cid"]


def test_cloud_model_worker_failure_waits_for_inflight_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_slow = tmp_path / "trace-slow.jsonl"
    trace_fail = tmp_path / "trace-fail.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    _write_trace(trace_slow, agent_id="task-slow")
    _write_trace(trace_fail, agent_id="task-fail")
    _write_tasks(task_source, "task-slow", "task-fail")
    _write_manifest(manifest, [str(trace_slow), str(trace_fail)])
    agent_stops: list[str] = []
    container_stops: list[str] = []

    class _FakeAgent:
        def __init__(self, agent_id: str) -> None:
            self.agent_id = agent_id

        async def stop(self) -> None:
            agent_stops.append(self.agent_id)

    class _FakeSampler:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> list[dict]:
            return []

    async def fake_prepare_container(
        loaded,
        *,
        task_output_dir=None,
        container_executable,
        network_mode="host",
    ):
        from trace_collect.simulator import PreparedContainer, PreparedTraceSession

        if loaded.agent_id == "task-fail":
            raise RuntimeError("prepare failed")
        return PreparedTraceSession(
            loaded=loaded,
            container=PreparedContainer(
                container_id="fake-slow",
                container_executable=container_executable,
                docker_image="fake-image",
                agent=_FakeAgent(loaded.agent_id),
            ),
        )

    async def fake_exec_tool(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        return ("ok", 50.0, True)

    def fake_stop_task_container(container_id: str, *, executable: str) -> str:
        assert executable == "docker"
        container_stops.append(container_id)
        return ""

    async def fake_prefetch(*_args, **_kwargs) -> None:
        pass

    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", fake_prepare_container)
    monkeypatch.setattr("trace_collect.simulator._prefetch_container_images", fake_prefetch)
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    monkeypatch.setattr("trace_collect.simulator.stop_task_container", fake_stop_task_container)
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("no llm")),
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="prepare failed"):
        asyncio.run(
            simulate(
                manifest=manifest,
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="cloud_model",
                concurrency=2,
                container_executable="docker",
                replay_speed=100.0,
            )
        )

    assert time.monotonic() - started >= 0.04
    assert agent_stops == ["task-slow"]
    assert container_stops == ["fake-slow"]


def test_cloud_model_prepare_failure_cleans_returned_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    agent_stops = 0
    container_stops = 0

    class _FakeAgent:
        async def stop(self) -> None:
            nonlocal agent_stops
            agent_stops += 1

    class _RaisingSampler:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("sampler failed")

    async def fake_prepare_container(
        loaded,
        *,
        task_output_dir=None,
        container_executable,
        network_mode="host",
    ):
        from trace_collect.simulator import PreparedContainer, PreparedTraceSession

        return PreparedTraceSession(
            loaded=loaded,
            container=PreparedContainer(
                container_id="fake-cid",
                container_executable=container_executable,
                docker_image="fake-image",
                agent=_FakeAgent(),
            ),
        )

    def fake_stop_task_container(container_id, *, executable):
        nonlocal container_stops
        assert container_id == "fake-cid"
        assert executable == "docker"
        container_stops += 1

    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", fake_prepare_container)
    monkeypatch.setattr("trace_collect.simulator.ensure_source_image", lambda *args, **kwargs: None)
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _RaisingSampler)
    monkeypatch.setattr("trace_collect.simulator.stop_task_container", fake_stop_task_container)

    with pytest.raises(RuntimeError, match="sampler failed"):
        asyncio.run(
            simulate(
                manifest=_single_trace_manifest(tmp_path, trace_path),
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="cloud_model",
                container_executable="docker",
            )
        )

    assert agent_stops == 1
    assert container_stops == 1


def test_cloud_model_host_trace_skips_mcp_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="host-task",
        scaffold="tongyi-deepresearch",
        tool_name="mcp_search",
        execution_environment="host",
    )
    _write_host_tasks(task_source, "host-task")

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode replay must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        fail_prepare,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("cloud_model must not create llm client")
        ),
    )

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    assert tool_record["data"]["replay_source"] == "skipped_host_mode"
    assert tool_record["data"]["sim_metrics"]["source"] == "skipped_host_mode"
    assert tool_record["data"]["success"] is True


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
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
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
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="local_model",
            container_executable="docker",
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


def test_local_model_failed_iteration_marks_throughput_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path, llm_client_mode="fake")

    class _FailingClient:
        class _Completions:
            async def create(self, **_kwargs):
                raise RuntimeError("local llm failed")

        class _Chat:
            def __init__(self) -> None:
                self.completions = _FailingClient._Completions()

        def __init__(self) -> None:
            self.chat = self._Chat()

    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: _FailingClient(),
    )

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="local_model",
            container_executable="docker",
            api_base="https://example.com/v1",
            api_key="secret",
            model="local-qwen",
        )
    )

    summary = next(record for record in _read_jsonl(trace_file) if record.get("type") == "summary")
    assert summary["success"] is False
    throughput = json.loads((tmp_path / "out" / "throughput_summary.json").read_text())
    assert throughput["completed_traces"] == 0
    assert throughput["failed_traces"] == 1
    assert throughput["tasks"][0]["success"] is False


def test_local_model_host_trace_completes_without_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="host-task",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_host_tasks(task_source, "host-task")

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode local simulation must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        fail_prepare,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: _FakeClient(),
    )

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
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
    summary = next(record for record in records if record.get("type") == "summary")

    assert metadata["execution_environment"] == "host"
    assert llm_record["data"]["sim_metrics"]["timing"]["total_ms"] >= 0.0
    # Host-mode tools cannot be re-executed in local_model (no container);
    # preserve source-trace timing so total_tool_ms stays faithful.
    assert tool_record["data"]["sim_metrics"]["source"] == "replayed_from_trace"
    assert (
        tool_record["data"]["sim_metrics"]["sim_tool_format"]
        == "replayed_from_trace"
    )
    assert tool_record["data"]["duration_ms"] == pytest.approx(50.0, abs=0.01)
    # ts_end - ts_start should match the replayed duration (0.05s)
    assert tool_record["ts_end"] - tool_record["ts_start"] == pytest.approx(
        0.05, abs=0.01
    )
    assert tool_record["data"]["success"] is True
    assert summary["success"] is True


def test_local_model_host_trace_replays_mcp_tool_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="host-task",
        scaffold="tongyi-deepresearch",
        tool_name="mcp_search",
        execution_environment="host",
    )
    _write_host_tasks(task_source, "host-task")

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode local simulation must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        fail_prepare,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: _FakeClient(),
    )

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="local_model",
            api_base="https://example.com/v1",
            api_key="secret",
            model="local-qwen",
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    # Both MCP and non-MCP host-mode tools go through the same replay path
    # in local_model — they cannot be re-executed without a container.
    assert tool_record["data"]["sim_metrics"]["source"] == "replayed_from_trace"
    assert (
        tool_record["data"]["sim_metrics"]["sim_tool_format"]
        == "replayed_from_trace"
    )
    assert tool_record["data"]["duration_ms"] == pytest.approx(50.0, abs=0.01)
    assert tool_record["data"]["success"] is True


def test_local_model_host_trace_replay_speed_scales_replayed_tool_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="host-task",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_host_tasks(task_source, "host-task")

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode local simulation must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        fail_prepare,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: _FakeClient(),
    )

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="local_model",
            api_base="https://example.com/v1",
            api_key="secret",
            model="local-qwen",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    assert tool_record["data"]["duration_ms"] == pytest.approx(50.0, abs=0.01)
    assert tool_record["ts_end"] - tool_record["ts_start"] == pytest.approx(
        0.005, abs=0.01
    )


def test_local_model_terminal_transport_retry_marks_failed_iteration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "tongyi-deepresearch",
                        "instance_id": "task-a",
                        "model": "source-model",
                        "execution_environment": "host",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0_transport_exhausted",
                        "agent_id": "task-a",
                        "iteration": 0,
                        "ts_start": 100.0,
                        "ts_end": 100.0,
                        "data": {
                            "transport_retry": True,
                            "transport_retry_terminal": True,
                            "messages_in": [{"role": "user", "content": "fail please"}],
                            "error": "APIConnectionError: boom",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "summary",
                        "agent_id": "task-a",
                        "model": "source-model",
                        "success": False,
                        "n_iterations": 1,
                        "elapsed_s": 0.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_host_tasks(task_source, "task-a")

    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal transport retry should not invoke local model")
        ),
    )
    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("host-mode local simulation must not prepare a container")
        ),
    )

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="local_model",
            api_base="https://example.com/v1",
            api_key="secret",
            model="local-qwen",
        )
    )

    records = _read_jsonl(trace_file)
    llm_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    )
    summary = next(record for record in records if record.get("type") == "summary")

    assert llm_record["data"]["transport_retry_terminal"] is True
    assert llm_record["data"]["sim_metrics"]["failed"] is True
    assert llm_record["data"]["messages_in"] == [{"role": "user", "content": "fail please"}]
    assert summary["success"] is False
    assert summary["failed_iterations"] == 1


def test_cloud_model_manifest_replays_multiple_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "trace-a.jsonl"
    trace_b = tmp_path / "trace-b.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    _write_trace(trace_a, agent_id="task-a", llm_start=100.0, llm_end=100.05, tool_start=100.1, tool_end=100.12)
    _write_trace(trace_b, agent_id="task-b", llm_start=200.0, llm_end=200.05, tool_start=200.1, tool_end=200.12)
    _write_tasks(task_source, "task-a", "task-b")
    _write_manifest(
        manifest,
        [
            {"trace": trace_a, "label": "a"},
            {"trace": trace_b, "task_source": task_source, "label": "b"},
        ],
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
            manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            concurrency=2,
            container_executable="docker",
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

    assert metadata["manifest"] == str(manifest)
    assert metadata["concurrency"] == 2
    assert metadata["scheduler_mode"] == "bounded_queue"
    assert metadata["source_trace_count"] == 2
    assert set(metadata["source_traces"]) == {str(trace_a), str(trace_b)}
    assert {record["agent_id"] for record in summaries} == {"task-a", "task-b"}
    assert {record["agent_id"] for record in llm_records} == {"task-a", "task-b"}
    assert abs(llm_records[0]["ts_start"] - llm_records[1]["ts_start"]) < 0.05


def test_cloud_model_concurrency_limits_active_traces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    traces = [tmp_path / f"trace-{idx}.jsonl" for idx in range(3)]
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    agent_ids = [f"task-{idx}" for idx in range(3)]
    for trace_path, agent_id in zip(traces, agent_ids, strict=True):
        _write_trace(trace_path, agent_id=agent_id)
    _write_tasks(task_source, *agent_ids)
    _write_manifest(manifest, [str(trace_path) for trace_path in traces])

    active = 0
    max_active = 0

    class _FakeAgent:
        async def stop(self) -> None:
            nonlocal active
            active -= 1

    class _FakeSampler:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> list[dict]:
            return []

    async def fake_prepare_container(
        loaded,
        *,
        task_output_dir=None,
        container_executable,
        network_mode="host",
    ):
        nonlocal active, max_active
        from trace_collect.simulator import PreparedContainer, PreparedTraceSession

        active += 1
        max_active = max(max_active, active)
        container = PreparedContainer(
            container_id=f"fake-{loaded.agent_id}",
            container_executable=container_executable,
            docker_image="fake-image",
            agent=_FakeAgent(),
        )
        return PreparedTraceSession(loaded=loaded, container=container)

    async def fake_exec_tool(*_args, **_kwargs):
        await asyncio.sleep(0.03)
        return ("ok", 30.0, True)

    async def fake_prefetch(*_args, **_kwargs) -> None:
        pass

    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", fake_prepare_container)
    monkeypatch.setattr("trace_collect.simulator._prefetch_container_images", fake_prefetch)
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("no llm")),
    )

    trace_file = asyncio.run(
        simulate(
            manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            concurrency=2,
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    assert trace_file.exists()
    assert max_active == 2
    assert active == 0
    summary = json.loads((tmp_path / "out" / "throughput_summary.json").read_text())
    assert summary["concurrency"] == 2
    assert summary["scheduler_mode"] == "bounded_queue"
    assert summary["attempted_traces"] == 3
    assert summary["completed_traces"] == 3


def test_cloud_model_structured_manifest_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "trace-a.jsonl"
    trace_b = tmp_path / "trace-b.jsonl"
    default_tasks = tmp_path / "default-tasks.json"
    override_tasks = tmp_path / "override-tasks.json"
    manifest = tmp_path / "manifest.yaml"
    _write_trace(trace_a, agent_id="task-a", execution_environment="host")
    _write_trace(trace_b, agent_id="task-b", execution_environment="host")
    _write_host_tasks(default_tasks, "task-a")
    _write_host_tasks(override_tasks, "task-b")
    manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "defaults:",
                f"  task_source: {json.dumps(str(default_tasks))}",
                "traces:",
                f"  - trace: {json.dumps(str(trace_a))}",
                "    label: default-task-source",
                f"  - trace: {json.dumps(str(trace_b))}",
                f"    task_source: {json.dumps(str(override_tasks))}",
                "    label: override-task-source",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _patch_simulator_runtime(monkeypatch, tmp_path, llm_client_mode="forbid")

    trace_file = asyncio.run(
        simulate(
            manifest=manifest,
            task_source=tmp_path / "unused-cli-tasks.json",
            output_dir=tmp_path / "out",
            mode="cloud_model",
            replay_speed=100.0,
        )
    )

    records = _read_jsonl(trace_file)
    summaries = [record for record in records if record.get("type") == "summary"]
    assert {record["agent_id"] for record in summaries} == {"task-a", "task-b"}
    summary = json.loads((tmp_path / "out" / "throughput_summary.json").read_text())
    assert {task["label"] for task in summary["tasks"]} == {
        "default-task-source",
        "override-task-source",
    }


def test_cloud_model_mixed_host_container_manifest_marks_environment_mixed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "trace-container.jsonl"
    trace_b = tmp_path / "trace-host.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    _write_trace(trace_a, agent_id="task-a")
    _write_trace(
        trace_b,
        agent_id="task-b",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_tasks(task_source, "task-a", "task-b")
    _write_manifest(
        manifest,
        [
            {"trace": trace_a},
            {"trace": trace_b, "task_source": task_source},
        ],
    )
    _patch_simulator_runtime(
        monkeypatch,
        tmp_path,
        tool_result_prefix="ok",
        llm_client_mode="forbid",
    )

    trace_file = asyncio.run(
        simulate(
            manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            concurrency=2,
            container_executable="docker",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    assert records[0]["execution_environment"] == "mixed"
    container_records = _read_jsonl(tmp_path / "out" / "task-a" / "attempt_1" / "trace.jsonl")
    host_records = _read_jsonl(tmp_path / "out" / "task-b" / "attempt_1" / "trace.jsonl")
    assert container_records[0]["execution_environment"] == "container"
    assert container_records[0]["scaffold"] == "openclaw"
    assert host_records[0]["execution_environment"] == "host"
    assert host_records[0]["scaffold"] == "tongyi-deepresearch"


def test_cloud_model_manifest_with_docker_image_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Manifest-level docker_image overrides task image_name."""
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    _write_manifest(
        manifest,
        [{"trace": trace_path, "docker_image": "custom/override:latest"}],
    )

    prepared_images: list[str] = []

    class _FakeAgent2:
        async def stop(self): pass

    async def capture_prepare(
        loaded,
        *,
        task_output_dir=None,
        container_executable,
        network_mode="host",
    ):
        from trace_collect.simulator import PreparedContainer, PreparedTraceSession, _resolve_docker_image
        img = _resolve_docker_image(loaded)
        prepared_images.append(img)
        container = PreparedContainer(
            container_id="fake-cid",
            container_executable=container_executable,
            docker_image=img or "",
            agent=_FakeAgent2(),
        )
        return PreparedTraceSession(loaded=loaded, container=container)

    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", capture_prepare)
    async def _fake_prefetch(*a, **kw):
        pass

    monkeypatch.setattr("trace_collect.simulator._prefetch_container_images", _fake_prefetch)
    async def _fake_exec(*a, **kw):
        return ("ok", 1.0, True)

    monkeypatch.setattr("trace_collect.simulator._exec_tool", _fake_exec)
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("no llm")),
    )

    asyncio.run(
        simulate(
            manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
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
                manifest=_single_trace_manifest(tmp_path, trace_path),
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="cloud_model",
            )
        )


def test_cloud_model_manifest_keeps_cli_task_source_cwd_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    manifest_dir = tmp_path / "manifests"
    manifest = manifest_dir / "manifest.yaml"
    task_source = tmp_path / "tasks.json"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    _write_manifest(manifest, [str(trace_path)])
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
            manifest=manifest,
            task_source=Path("tasks.json"),
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    assert any(record.get("type") == "summary" for record in records)


def test_cloud_model_host_tool_without_success_field_is_not_mislabeled_as_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression: host-mode scaffold tools may not emit 'success'; host replay must
    fall back to `not error` instead of defaulting to False.
    """
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"

    # Hand-crafted trace whose tool_exec action has NO "success" key, matching
    # what vendor host-mode tools emit.
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({
                    "type": "trace_metadata",
                    "trace_format_version": 5,
                    "scaffold": "tongyi-deepresearch",
                    "instance_id": "host-task",
                    "model": "qwen",
                    "mode": "collect",
                    "execution_environment": "host",
                }),
                json.dumps({
                    "type": "action",
                    "action_type": "llm_call",
                    "action_id": "host-task-llm-0",
                    "agent_id": "host-task",
                    "iteration": 0,
                    "ts_start": 100.0,
                    "ts_end": 100.2,
                    "data": {
                        "messages_in": [{"role": "user", "content": "x"}],
                        "raw_response": {"id": "r"},
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "llm_latency_ms": 200.0,
                    },
                }),
                json.dumps({
                    "type": "action",
                    "action_type": "tool_exec",
                    "action_id": "host-task-tool-0",
                    "agent_id": "host-task",
                    "iteration": 0,
                    "ts_start": 100.4,
                    "ts_end": 100.45,
                    "data": {
                        "tool_name": "web_search",
                        "args": {"query": "anything"},
                        "result": "some result",
                        "duration_ms": 50.0,
                        # NOTE: intentionally no "success" key, mirroring
                        # host-mode tool emission
                        "error": None,
                    },
                }),
                json.dumps({
                    "type": "summary",
                    "agent_id": "host-task",
                    "model": "qwen",
                    "success": True,
                    "n_iterations": 1,
                    "elapsed_s": 0.45,
                }),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_host_tasks(task_source, "host-task")

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode replay must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        fail_prepare,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("cloud_model must not create llm client")
        ),
    )

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    # The bug: without the fallback, a missing "success" would default to
    # False, mislabeling valid host-mode runs and inflating failure rates.
    assert tool_record["data"]["success"] is True


# ----------------------------------------------------------------------
# Ralplan R3 Phase H2: simulator replays tongyi-deepresearch host-mode trace
# ----------------------------------------------------------------------

_TONGYI_FIXTURE = Path(__file__).parent / "fixtures" / "tongyi_deepresearch_minimal_v5.jsonl"


def test_simulator_replays_tongyi_deepresearch_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """R3 Principle P3 / Phase H2: host-mode host_controller traces from the
    vendored Tongyi-DeepResearch scaffold are replayed by cloud_model simulator
    without any simulator code changes, and without spinning up a container or
    creating an LLM client (host mode's defining guarantees)."""
    assert _TONGYI_FIXTURE.exists(), f"missing fixture: {_TONGYI_FIXTURE}"
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_bytes(_TONGYI_FIXTURE.read_bytes())

    task_source = tmp_path / "tasks.json"
    _write_host_tasks(task_source, "tongyi-fixture-1")

    async def _fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode replay must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session", _fail_prepare,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("cloud_model must not create llm client")
        ),
    )

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    metadata = records[0]
    llm_records = [
        r for r in records
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    ]
    tool_records = [
        r for r in records
        if r.get("type") == "action" and r.get("action_type") == "tool_exec"
    ]
    summary = next(r for r in records if r.get("type") == "summary")

    # Scaffold-agnostic structural invariants: the simulator respects the
    # source trace's host-mode flag and replays each action span.
    assert metadata["execution_environment"] == "host"
    assert metadata["scaffold"] == "tongyi-deepresearch"
    assert len(llm_records) == 3, "source has 3 llm_calls, simulator must replay all"
    assert len(tool_records) == 2, "source has 2 tool_execs, simulator must replay all"
    # Host-mode tool replay gets the canonical 'skipped_host_mode' tag and
    # success=True fallback, same as any host-mode scaffold.
    for tool_record in tool_records:
        assert tool_record["data"]["replay_source"] == "skipped_host_mode"
        assert tool_record["data"]["success"] is True
    assert summary["success"] is True

    # Host-mode replay must still write an empty resources.json so downstream
    # consumers see a canonical simulate layout.
    attempt_dir = tmp_path / "out" / "tongyi-fixture-1" / "attempt_1"
    resources_path = attempt_dir / "resources.json"
    assert resources_path.exists()
    payload = json.loads(resources_path.read_text())
    assert payload["samples"] == []
    assert payload["summary"]["sample_count"] == 0


def test_execution_environment_infers_host_from_agent_runtime_mode() -> None:
    """Legacy traces that predate execution_environment still replay correctly
    when agent_runtime_mode=host_controller is present. Regression guard for
    Codex P1 feedback on cc3a18a (PR #13)."""
    from types import SimpleNamespace
    from trace_collect.simulator import _execution_environment

    # Legacy host trace: no execution_environment, but agent_runtime_mode is set
    legacy_host = SimpleNamespace(
        metadata={"agent_runtime_mode": "host_controller"},
        source_trace="/tmp/legacy_host.jsonl",
    )
    assert _execution_environment(legacy_host) == "host"

    # Legacy unknown trace: nothing → container default retained
    legacy_unknown = SimpleNamespace(metadata={}, source_trace="/tmp/legacy.jsonl")
    assert _execution_environment(legacy_unknown) == "container"

    # Explicit execution_environment wins over agent_runtime_mode
    explicit_container = SimpleNamespace(
        metadata={
            "execution_environment": "container",
            "agent_runtime_mode": "host_controller",
        },
        source_trace="/tmp/explicit.jsonl",
    )
    assert _execution_environment(explicit_container) == "container"


def test_tongyi_deepresearch_fixture_is_valid_v5() -> None:
    """Sanity: the shipped fixture file parses as valid v5 JSONL with the
    expected record shape. Prevents accidental corruption during edits."""
    records = [json.loads(ln) for ln in _TONGYI_FIXTURE.read_text().splitlines() if ln.strip()]

    # 1 metadata + 3 llm_call + 2 tool_exec + 1 summary = 7 records
    assert len(records) == 7
    metadata = records[0]
    assert metadata["type"] == "trace_metadata"
    assert metadata["trace_format_version"] == 5
    assert metadata["scaffold"] == "tongyi-deepresearch"

    llm_calls = [r for r in records if r.get("action_type") == "llm_call"]
    assert [r["action_id"] for r in llm_calls] == ["llm_1", "llm_2", "llm_3"]
    for call in llm_calls:
        assert call["data"]["ttft_ms"] is not None
        assert call["data"]["tpot_ms"] is not None
        assert "logical_turn_id" in call["data"]

    tool_execs = [r for r in records if r.get("action_type") == "tool_exec"]
    assert [r["action_id"] for r in tool_execs] == ["tool_1", "tool_2"]
    for tool in tool_execs:
        # Canonical keys (R3 Principle P2)
        assert "tool_args" in tool["data"]
        assert "tool_result" in tool["data"]
        assert "duration_ms" in tool["data"]
