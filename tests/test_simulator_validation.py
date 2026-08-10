from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tool_resource.artifact_schema import (
    CLAUSE_TELEMETRY_SCHEMA_VERSION,
    CLAUSE_TELEMETRY_STATUS_MODEL,
)
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


def _write_host_trace(path: Path, task_id: str) -> Path:
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "generic",
                        "execution_environment": "host",
                        "instance_id": task_id,
                        "model": "dummy",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "tool_exec",
                        "action_id": f"tool-{task_id}",
                        "agent_id": task_id,
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 1.0,
                        "data": {
                            "tool_name": "message",
                            "success": True,
                            "duration_ms": 0.0,
                        },
                    }
                ),
                json.dumps({"type": "summary", "agent_id": task_id, "success": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_tasks(path: Path, *tasks: dict[str, object]) -> Path:
    path.write_text(json.dumps(list(tasks)) + "\n", encoding="utf-8")
    return path


def test_simulate_manifest_parses_depends_on(tmp_path: Path) -> None:
    parent_trace = _write_host_trace(tmp_path / "parent.jsonl", "parent")
    child_trace = _write_host_trace(tmp_path / "child.jsonl", "child")
    task_source = _write_tasks(
        tmp_path / "tasks.json",
        {"instance_id": "parent", "problem_statement": "parent"},
        {"instance_id": "child", "problem_statement": "child"},
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "defaults:",
                f"  task_source: {json.dumps(str(task_source))}",
                "traces:",
                f"  - trace: {json.dumps(str(parent_trace))}",
                f"  - trace: {json.dumps(str(child_trace))}",
                "    depends_on:",
                "      - parent",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from trace_collect.simulate_manifest import _load_simulate_manifest

    entries = _load_simulate_manifest(manifest, default_task_source=None)

    assert entries[0].depends_on == ()
    assert entries[1].depends_on == ("parent",)


def test_simulate_manifest_rejects_invalid_depends_on(tmp_path: Path) -> None:
    trace_path = _write_host_trace(tmp_path / "trace.jsonl", "task-a")
    task_source = _write_tasks(
        tmp_path / "tasks.json",
        {"instance_id": "task-a", "problem_statement": "task"},
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "traces:",
                "  -",
                f"    trace: {json.dumps(str(trace_path))}",
                f"    task_source: {json.dumps(str(task_source))}",
                "    depends_on: task-b",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from trace_collect.simulate_manifest import _load_simulate_manifest

    with pytest.raises(SimulateError, match="depends_on must be a list of strings"):
        _load_simulate_manifest(manifest, default_task_source=None)


def test_load_trace_session_combines_manifest_and_task_source_depends_on(
    tmp_path: Path,
) -> None:
    trace_path = _write_host_trace(tmp_path / "child.jsonl", "child")
    task_source = _write_tasks(
        tmp_path / "tasks.json",
        {
            "instance_id": "child",
            "problem_statement": "child",
            "depends_on": ["task-parent"],
        },
    )

    from trace_collect.simulate_manifest import _load_trace_session

    loaded = _load_trace_session(
        trace_path,
        task_source,
        manifest_index=0,
        manifest_depends_on=("manifest-parent", "task-parent"),
    )

    assert loaded.depends_on == ("manifest-parent", "task-parent")


def test_simulator_rejects_missing_depends_on_task(tmp_path: Path) -> None:
    trace_path = _write_host_trace(tmp_path / "child.jsonl", "child")
    task_source = _write_tasks(
        tmp_path / "tasks.json",
        {
            "instance_id": "child",
            "problem_statement": "child",
            "depends_on": ["parent"],
        },
    )

    with pytest.raises(SimulateError, match="depends on missing task ids"):
        asyncio.run(
            simulate(
                manifest=_single_trace_manifest(tmp_path, trace_path),
                task_source=task_source,
                output_dir=tmp_path / "out",
                model="dummy",
            )
        )


def test_simulator_rejects_self_depends_on_task(tmp_path: Path) -> None:
    trace_path = _write_host_trace(tmp_path / "task-a.jsonl", "task-a")
    task_source = _write_tasks(
        tmp_path / "tasks.json",
        {
            "instance_id": "task-a",
            "problem_statement": "task",
            "depends_on": ["task-a"],
        },
    )

    with pytest.raises(SimulateError, match="depends on itself"):
        asyncio.run(
            simulate(
                manifest=_single_trace_manifest(tmp_path, trace_path),
                task_source=task_source,
                output_dir=tmp_path / "out",
                model="dummy",
            )
        )


def test_simulator_rejects_cyclic_depends_on_tasks(tmp_path: Path) -> None:
    trace_a = _write_host_trace(tmp_path / "task-a.jsonl", "task-a")
    trace_b = _write_host_trace(tmp_path / "task-b.jsonl", "task-b")
    task_source = _write_tasks(
        tmp_path / "tasks.json",
        {"instance_id": "task-a", "problem_statement": "a", "depends_on": ["task-b"]},
        {"instance_id": "task-b", "problem_statement": "b", "depends_on": ["task-a"]},
    )
    manifest = _write_manifest(tmp_path / "manifest.yaml", [str(trace_a), str(trace_b)])

    with pytest.raises(SimulateError, match="Dependency cycle detected"):
        asyncio.run(
            simulate(
                manifest=manifest,
                task_source=task_source,
                output_dir=tmp_path / "out",
                model="dummy",
            )
        )


def test_simulator_outputs_depends_on_metadata(tmp_path: Path) -> None:
    trace_parent = _write_host_trace(tmp_path / "parent.jsonl", "parent")
    trace_child = _write_host_trace(tmp_path / "child.jsonl", "child")
    task_source = _write_tasks(
        tmp_path / "tasks.json",
        {"instance_id": "parent", "problem_statement": "parent"},
        {
            "instance_id": "child",
            "problem_statement": "child",
            "depends_on": ["parent"],
        },
    )
    manifest = _write_manifest(
        tmp_path / "manifest.yaml",
        [str(trace_parent), str(trace_child)],
    )
    output_dir = tmp_path / "out"

    trace_file = asyncio.run(
        simulate(
            manifest=manifest,
            task_source=task_source,
            output_dir=output_dir,
            model="dummy",
            concurrency=2,
        )
    )

    records = [
        json.loads(line)
        for line in trace_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata = records[0]
    child_entry = next(
        entry
        for entry in metadata["source_trace_entries"]
        if entry["task_instance_id"] == "child"
    )
    throughput = json.loads((output_dir / "throughput_summary.json").read_text())
    child_task = next(
        task for task in throughput["tasks"] if task["agent_id"] == "child"
    )

    assert child_entry["depends_on"] == ["parent"]
    assert child_task["depends_on"] == ["parent"]


def test_simulator_rejects_depends_on_with_multiple_workers(tmp_path: Path) -> None:
    trace_parent = _write_host_trace(tmp_path / "parent.jsonl", "parent")
    trace_child = _write_host_trace(tmp_path / "child.jsonl", "child")
    task_source = _write_tasks(
        tmp_path / "tasks.json",
        {"instance_id": "parent", "problem_statement": "parent"},
        {
            "instance_id": "child",
            "problem_statement": "child",
            "depends_on": ["parent"],
        },
    )
    manifest = _write_manifest(
        tmp_path / "manifest.yaml",
        [str(trace_parent), str(trace_child)],
    )

    with pytest.raises(SimulateError, match="requires workers=1"):
        asyncio.run(
            simulate(
                manifest=manifest,
                task_source=task_source,
                output_dir=tmp_path / "out",
                model="dummy",
                concurrency=2,
                workers=2,
            )
        )


def test_simulator_rejects_duplicate_task_ids_when_depends_on_exists(
    tmp_path: Path,
) -> None:
    trace_a = _write_host_trace(tmp_path / "task-a.jsonl", "task-a")
    trace_b = _write_host_trace(tmp_path / "task-b.jsonl", "task-b")
    task_source = _write_tasks(
        tmp_path / "tasks.json",
        {"instance_id": "task-a", "problem_statement": "a", "depends_on": ["task-b"]},
        {"instance_id": "task-b", "problem_statement": "b"},
    )
    manifest = _write_manifest(
        tmp_path / "manifest.yaml",
        [str(trace_a), str(trace_a), str(trace_b)],
    )

    with pytest.raises(SimulateError, match="ambiguous with duplicate task ids"):
        asyncio.run(
            simulate(
                manifest=manifest,
                task_source=task_source,
                output_dir=tmp_path / "out",
                model="dummy",
            )
        )


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

    from trace_collect.simulate_manifest import _parse_trace_session_file

    task_instance_id, source_action_agent_id, _metadata, actions, summary = (
        _parse_trace_session_file(trace_path)
    )

    assert task_instance_id == "hydra-debug-slurm-mode"
    assert source_action_agent_id == "cli:oc-df47179e"
    assert [action["action_id"] for action in actions] == ["llm_0", "sub_tool_0"]
    assert summary == {
        "type": "summary",
        "agent_id": "cli:oc-df47179e",
        "success": True,
    }


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

    from trace_collect.simulate_manifest import _parse_trace_session_file

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
                        "scaffold": "tongyi-deepresearch",
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
        async def stop(self):
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
                container_id="fake",
                container_executable=container_executable,
                docker_image="fake",
                agent=_FakeAgent(),
            ),
            task_output_dir=task_output_dir,
        )

    async def fake_prefetch(*_args, **_kwargs) -> None:
        pass

    async def fake_prebuild(*_args, **_kwargs) -> dict[str, str]:
        return {}

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session", fake_prepare
    )
    monkeypatch.setattr(
        "trace_collect.simulator._prefetch_container_images", fake_prefetch
    )
    monkeypatch.setattr(
        "trace_collect.simulator._prebuild_sweep_fixed_images", fake_prebuild
    )
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


def test_openclaw_replay_provider_rejects_ttft_tpot_with_replay_speed() -> None:
    from trace_collect.openclaw_host_runtime import OpenClawReplayProvider

    with pytest.raises(ValueError, match="exclusive with llm_timing_mode='ttft_tpot'"):
        OpenClawReplayProvider(
            llm_actions=[],
            replay_speed=2.0,
            timing_mode="ttft_tpot",
            llm_ttft_ms=10.0,
            llm_tpot_ms=2.0,
        )


def test_openclaw_replay_provider_charges_streamed_shadow_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shadow generation uses the frozen source prompt but replays source actions."""
    from trace_collect.openclaw_host_runtime import (
        OpenClawReplayProvider,
        ShadowGenerationConfig,
    )

    captured: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            pass

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"shadow","token_ids":[101,102]}}]}'
            yield 'data: {"choices":[{"delta":{"token_ids":[103]},"finish_reason":"length"}],"usage":{"prompt_tokens":12,"completion_tokens":3}}'
            yield "data: [DONE]"

    class _FakeStream:
        async def __aenter__(self):
            return _FakeResponse()

        async def __aexit__(self, *_args):
            return False

    class _FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        def stream(self, method, url, *, json):
            captured.update(method=method, url=url, request=json)
            return _FakeStream()

        async def aclose(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(
        "trace_collect.openclaw_host_runtime.httpx.AsyncClient", _FakeClient
    )
    source_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-first",
                    "type": "function",
                    "function": {"name": "exec", "arguments": '{"command":"one"}'},
                },
                {
                    "id": "call-second",
                    "type": "function",
                    "function": {"name": "exec", "arguments": '{"command":"two"}'},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-first",
            "name": "exec",
            "content": {"stdout": "one"},
        },
        {
            "role": "tool",
            "tool_call_id": "call-second",
            "name": "exec",
            "content": ["two"],
        },
    ]
    source_action = {
        "action_type": "llm_call",
        "data": {
            "messages_in": source_messages,
            "completion_tokens": 3,
            "raw_response": {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "source-call",
                                    "function": {
                                        "name": "exec",
                                        "arguments": '{"command":"source"}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            },
        },
    }
    provider = OpenClawReplayProvider(
        llm_actions=[source_action],
        replay_speed=1.0,
        timing_mode="source_scaled",
        shadow_generation=ShadowGenerationConfig(
            api_base="http://127.0.0.1:8000/v1",
            model="meta-llama/Llama-3.1-8B-Instruct",
            timeout_s=12.0,
            seed=7,
        ),
    )

    response = asyncio.run(
        provider.chat([{"role": "user", "content": "live runner prompt"}])
    )
    asyncio.run(provider.aclose())

    request = captured["request"]
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert request == {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": source_messages[0]["tool_calls"],
            },
            {
                "role": "tool",
                "tool_call_id": "call-first",
                "content": '{"stdout":"one"}',
            },
            {
                "role": "tool",
                "tool_call_id": "call-second",
                "content": '["two"]',
            },
        ],
        "temperature": 0,
        "max_tokens": 3,
        "ignore_eos": True,
        "return_token_ids": True,
        "seed": 7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert response.tool_calls[0].id == "source-call"
    assert response.tool_calls[0].arguments == {"command": "source"}
    shadow_metrics = response.extra["shadow_generation"]
    assert {
        key: value
        for key, value in shadow_metrics.items()
        if key not in {"ttft_ms", "latency_ms"}
    } == {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "seed": 7,
        "requested_completion_tokens": 3,
        "returned_completion_tokens": 3,
        "completion_token_ids": [101, 102, 103],
        "finish_reason": "length",
        "prompt_tokens": 12,
    }
    assert shadow_metrics["ttft_ms"] >= 0
    assert shadow_metrics["latency_ms"] >= shadow_metrics["ttft_ms"]
    assert captured["client_kwargs"] == {"timeout": 12.0, "trust_env": False}
    assert captured["closed"] is True


def test_openclaw_replay_provider_rejects_shadow_token_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trace_collect.openclaw_host_runtime import (
        OpenClawReplayProvider,
        ShadowGenerationConfig,
    )

    class _FakeResponse:
        def raise_for_status(self) -> None:
            pass

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"token_ids":[101]}}]}'
            yield "data: [DONE]"

    class _FakeStream:
        async def __aenter__(self):
            return _FakeResponse()

        async def __aexit__(self, *_args):
            return False

    class _FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def stream(self, *_args, **_kwargs):
            return _FakeStream()

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(
        "trace_collect.openclaw_host_runtime.httpx.AsyncClient", _FakeClient
    )
    provider = OpenClawReplayProvider(
        llm_actions=[
            {
                "action_type": "llm_call",
                "data": {
                    "messages_in": [],
                    "completion_tokens": 2,
                    "raw_response": {"choices": []},
                },
            }
        ],
        replay_speed=1.0,
        timing_mode="source_scaled",
        shadow_generation=ShadowGenerationConfig(
            api_base="http://127.0.0.1:8000/v1",
            model="meta-llama/Llama-3.1-8B-Instruct",
            timeout_s=12.0,
            seed=7,
        ),
    )

    try:
        with pytest.raises(RuntimeError, match="expected 2, got 1"):
            asyncio.run(provider.chat([]))
    finally:
        asyncio.run(provider.aclose())


def test_openclaw_host_replay_request_closes_provider_after_runner_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trace_collect.openclaw_host_runtime import run_openclaw_host_replay_request

    closed: list[bool] = []
    stopped: list[bool] = []
    captured: dict[str, object] = {}

    class _FakeAgent:
        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            stopped.append(True)

    class _FailingRunner:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def run(self, **_kwargs):
            raise RuntimeError("runner failed")

    async def fake_proof(*_args, **_kwargs) -> dict[str, object]:
        return {}

    class _FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def aclose(self) -> None:
            closed.append(True)

        def get_default_model(self) -> str:
            return "replay-openclaw"

    monkeypatch.setattr(
        "trace_collect.openclaw_host_runtime.ContainerAgent",
        lambda *_args, **_kwargs: _FakeAgent(),
    )
    monkeypatch.setattr(
        "trace_collect.openclaw_host_runtime.container_runtime_proof", fake_proof
    )
    monkeypatch.setattr(
        "agents.openclaw._session_runner.SessionRunner", _FailingRunner
    )
    monkeypatch.setattr(
        "trace_collect.openclaw_host_runtime.OpenClawReplayProvider", _FakeProvider
    )
    status_path = tmp_path / "status.json"

    status = asyncio.run(
        run_openclaw_host_replay_request(
            {
                "source_actions": [],
                "container_id": "cid",
                "container_executable": "docker",
                "output_trace": str(tmp_path / "replay.jsonl"),
                "runtime_dir": str(tmp_path / "runtime"),
                "workspace": str(tmp_path),
                "status_path": str(status_path),
                "replay_speed": 1.0,
                "llm_timing": {"mode": "source_scaled"},
                "shadow_generation": {
                    "api_base": "http://127.0.0.1:8000/v1",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "timeout_s": 12.0,
                    "seed": 7,
                },
                "command_timeout_s": 60.0,
                "run_instance_id": "close-on-failure",
                "task_instance_id": "close-on-failure",
                "source_action_agent_id": "source-agent",
                "prompt": "replay",
            }
        )
    )

    assert status["success"] is False
    assert "runner failed" in status["error"]
    assert closed == [True]
    assert stopped == [True]
    shadow_generation = captured["shadow_generation"]
    assert shadow_generation.api_base == "http://127.0.0.1:8000/v1"
    assert shadow_generation.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert shadow_generation.timeout_s == 12.0
    assert shadow_generation.seed == 7
    assert json.loads(status_path.read_text(encoding="utf-8"))["success"] is False


def test_shadow_generation_config_rejects_non_loopback_api_base() -> None:
    from trace_collect.openclaw_host_runtime import ShadowGenerationConfig

    with pytest.raises(ValueError, match="loopback"):
        ShadowGenerationConfig(
            api_base="https://vllm.example.test/v1",
            model="meta-llama/Llama-3.1-8B-Instruct",
            timeout_s=12.0,
            seed=7,
        )


@pytest.mark.parametrize(
    "api_base",
    [
        "http://user@127.0.0.1:8000/v1",
        "http://:secret@127.0.0.1:8000/v1",
    ],
)
def test_shadow_generation_config_rejects_url_credentials(api_base: str) -> None:
    from trace_collect.openclaw_host_runtime import ShadowGenerationConfig

    with pytest.raises(ValueError, match="credentials"):
        ShadowGenerationConfig(
            api_base=api_base,
            model="meta-llama/Llama-3.1-8B-Instruct",
            timeout_s=12.0,
            seed=7,
        )


def test_simulate_cli_parses_shadow_generation_options() -> None:
    from trace_collect.cli import parse_simulate_args

    args = parse_simulate_args(
        [
            "--manifest",
            "manifest.yaml",
            "--shadow-llm-api-base",
            "http://127.0.0.1:8000/v1",
            "--shadow-llm-model",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--shadow-llm-timeout-s",
            "12",
            "--shadow-llm-seed",
            "7",
        ]
    )

    assert args.shadow_llm_api_base == "http://127.0.0.1:8000/v1"
    assert args.shadow_llm_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert args.shadow_llm_timeout_s == 12.0
    assert args.shadow_llm_seed == 7


def test_simulate_cli_passes_container_cpu_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trace_collect.cli import _run_simulate, parse_simulate_args

    captured: dict[str, object] = {}

    async def fake_simulate(**kwargs: object) -> Path:
        captured.update(kwargs)
        return tmp_path / "out.jsonl"

    monkeypatch.setattr("trace_collect.simulator.simulate", fake_simulate)
    args = parse_simulate_args(
        [
            "--manifest",
            "manifest.yaml",
            "--container-cpus",
            "2",
            "--shadow-llm-api-base",
            "http://127.0.0.1:8000/v1",
            "--shadow-llm-model",
            "meta-llama/Llama-3.1-8B-Instruct",
        ]
    )

    _run_simulate(args)

    assert captured["container_start_extra_args"] == ("--cpus", "2")
    assert captured["shadow_llm_api_base"] == "http://127.0.0.1:8000/v1"
    assert captured["shadow_llm_model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert captured["shadow_llm_timeout_s"] == 120.0
    assert captured["shadow_llm_seed"] == 0


@pytest.mark.parametrize(
    ("shadow_kwargs", "match"),
    [
        ({"shadow_llm_api_base": "http://127.0.0.1:8000/v1"}, "together"),
        ({"shadow_llm_model": "meta-llama/Llama-3.1-8B-Instruct"}, "together"),
        (
            {
                "shadow_llm_api_base": "http://127.0.0.1:8000/v1",
                "shadow_llm_model": "meta-llama/Llama-3.1-8B-Instruct",
                "replay_speed": 2.0,
            },
            "replay_speed=1.0",
        ),
    ],
)
def test_simulate_validates_shadow_generation_configuration(
    tmp_path: Path,
    shadow_kwargs: dict[str, object],
    match: str,
) -> None:
    trace_path = _write_host_trace(tmp_path / "trace.jsonl", "task-a")

    with pytest.raises(ValueError, match=match):
        asyncio.run(
            simulate(
                manifest=_single_trace_manifest(tmp_path, trace_path),
                output_dir=tmp_path / "out",
                **shadow_kwargs,
            )
        )


def test_simulate_rejects_shadow_generation_for_non_openclaw_before_output(
    tmp_path: Path,
) -> None:
    trace_path = _write_host_trace(tmp_path / "generic.jsonl", "generic-task")
    task_source = _write_tasks(
        tmp_path / "tasks.json",
        {"instance_id": "generic-task", "problem_statement": "generic"},
    )
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="shadow generation requires OpenClaw"):
        asyncio.run(
            simulate(
                manifest=_single_trace_manifest(tmp_path, trace_path),
                task_source=task_source,
                output_dir=output_dir,
                shadow_llm_api_base="http://127.0.0.1:8000/v1",
                shadow_llm_model="meta-llama/Llama-3.1-8B-Instruct",
            )
        )

    assert not output_dir.exists()


def test_simulate_rejects_shadow_generation_for_mixed_scaffolds_before_output(
    tmp_path: Path,
) -> None:
    generic_trace = _write_host_trace(
        tmp_path / "generic.jsonl", "generic-task"
    )
    openclaw_trace = _write_host_trace(
        tmp_path / "openclaw.jsonl", "openclaw-task"
    )
    openclaw_records = [
        json.loads(line) for line in openclaw_trace.read_text().splitlines()
    ]
    openclaw_records[0].update(scaffold="openclaw", execution_environment="container")
    openclaw_trace.write_text(
        "\n".join(json.dumps(record) for record in openclaw_records) + "\n",
        encoding="utf-8",
    )
    task_source = _write_tasks(
        tmp_path / "tasks.json",
        {"instance_id": "generic-task", "problem_statement": "generic"},
        {
            "instance_id": "openclaw-task",
            "problem_statement": "openclaw",
            "image_name": "example/openclaw:latest",
        },
    )
    manifest = _write_manifest(
        tmp_path / "manifest.yaml", [str(generic_trace), str(openclaw_trace)]
    )
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="shadow generation requires OpenClaw"):
        asyncio.run(
            simulate(
                manifest=manifest,
                task_source=task_source,
                output_dir=output_dir,
                shadow_llm_api_base="http://127.0.0.1:8000/v1",
                shadow_llm_model="meta-llama/Llama-3.1-8B-Instruct",
            )
        )

    assert not output_dir.exists()


def test_simulate_rejects_container_cpu_cap_for_non_openclaw_before_output(
    tmp_path: Path,
) -> None:
    trace_path = _write_host_trace(tmp_path / "generic.jsonl", "generic-task")
    task_source = _write_tasks(
        tmp_path / "tasks.json",
        {"instance_id": "generic-task", "problem_statement": "generic"},
    )
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="container CPU cap requires OpenClaw"):
        asyncio.run(
            simulate(
                manifest=_single_trace_manifest(tmp_path, trace_path),
                task_source=task_source,
                output_dir=output_dir,
                container_start_extra_args=("--cpus", "2"),
            )
        )

    assert not output_dir.exists()


def test_simulate_rejects_container_cpu_cap_for_terminal_bench_before_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = _write_host_trace(
        tmp_path / "terminal-bench.jsonl", "terminal-task"
    )
    trace_records = [
        json.loads(line) for line in trace_path.read_text().splitlines()
    ]
    trace_records[0].update(
        scaffold="openclaw",
        execution_environment="container",
        task_source_kind="terminal_bench_registry",
    )
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in trace_records) + "\n",
        encoding="utf-8",
    )
    task_source = _write_tasks(
        tmp_path / "tasks.json",
        {
            "instance_id": "terminal-task",
            "task_id": "terminal-task",
            "task_source_kind": "terminal_bench_registry",
            "task_source_path": str(tmp_path / "terminal-task"),
            "problem_statement": "terminal",
        },
    )
    output_dir = tmp_path / "out"

    async def fail_prefetch(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("prefetch started before CPU-cap validation")

    monkeypatch.setattr(
        "trace_collect.simulator._prefetch_container_images", fail_prefetch
    )

    with pytest.raises(ValueError, match="does not support Terminal-Bench"):
        asyncio.run(
            simulate(
                manifest=_single_trace_manifest(tmp_path, trace_path),
                task_source=task_source,
                output_dir=output_dir,
                container_executable="docker",
                container_start_extra_args=("--cpus", "2"),
            )
        )

    assert not output_dir.exists()


def test_combined_worker_trace_records_shadow_generation_metadata(tmp_path: Path) -> None:
    from trace_collect.simulate_outputs import _write_combined_worker_trace
    from trace_collect.simulate_types import (
        LLMTimingConfig,
        LoadedTraceSession,
        WorkerReplayResult,
    )

    worker_trace = tmp_path / "worker.jsonl"
    worker_trace.write_text("", encoding="utf-8")
    session = LoadedTraceSession(
        source_trace=tmp_path / "source.jsonl",
        task_source=tmp_path / "tasks.json",
        task_instance_id="task-a",
        source_action_agent_id="source-a",
        run_instance_id="run-a",
        manifest_index=0,
        scaffold="openclaw",
        metadata={"execution_environment": "container", "model": "source-model"},
        summary=None,
        task={},
        actions=[],
        iterations={},
    )
    combined_trace = tmp_path / "combined.jsonl"

    _write_combined_worker_trace(
        trace_file=combined_trace,
        worker_results=[
            WorkerReplayResult(
                wave_index=0,
                worker_index=0,
                trace_file=str(worker_trace),
                task_stats=[],
                task_output_dirs={},
            )
        ],
        sessions=[session],
        mode="cloud_model",
        replay_speed=1.0,
        llm_timing=LLMTimingConfig(),
        manifest=tmp_path / "manifest.yaml",
        concurrency=2,
        workers=2,
        prep_concurrency=0,
        network_mode="host",
        model=None,
        monitoring_policy={},
        exec_timeout_floor_s=None,
        shadow_generation={
            "api_base": "http://127.0.0.1:8000/v1",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "timeout_s": 12.0,
            "seed": 7,
        },
    )

    metadata = json.loads(combined_trace.read_text(encoding="utf-8").splitlines()[0])
    assert metadata["shadow_generation"] == {
        "api_base": "http://127.0.0.1:8000/v1",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "timeout_s": 12.0,
        "seed": 7,
    }


def test_paired_replay_contract_pins_pytest_seed_and_preserves_failed_timeout() -> None:
    from trace_collect.simulate_openclaw import _paired_replay_actions

    call_id = "call-seeded"
    failed_id = "call-failed"
    raw_arguments = json.dumps({"command": "python -m pytest", "timeout": 600})
    source_actions = [
        {
            "action_type": "llm_call",
            "data": {
                "raw_response": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": call_id,
                                        "function": {
                                            "name": "exec",
                                            "arguments": raw_arguments,
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
        },
        {
            "action_type": "tool_exec",
            "data": {
                "tool_name": "exec",
                "tool_call_id": call_id,
                "tool_args": raw_arguments,
                "tool_result": "Using --randomly-seed=12345\nExit code: 1",
                "success": True,
            },
        },
        {
            "action_type": "tool_exec",
            "data": {
                "tool_name": "exec",
                "tool_call_id": failed_id,
                "tool_args": json.dumps({"command": "slow", "timeout": 600}),
                "tool_result": "Error: [timeout]\nExit code: 124",
                "success": False,
            },
        },
    ]

    actions, contract = _paired_replay_actions(source_actions)

    amended_tool = json.loads(actions[1]["data"]["tool_args"])["command"]
    amended_call = json.loads(
        actions[0]["data"]["raw_response"]["choices"][0]["message"]["tool_calls"][0][
            "function"
        ]["arguments"]
    )["command"]
    assert amended_tool == amended_call
    assert "--randomly-seed=12345" in amended_tool
    assert json.loads(source_actions[1]["data"]["tool_args"])["command"] == (
        "python -m pytest"
    )
    assert contract == {
        "version": 1,
        "pytest_random_seeds": [{"tool_call_id": call_id, "seed": "12345"}],
        "exec_timeout_floor_exempt_call_ids": [failed_id],
        "require_source_outcome_match": False,
    }


def test_paired_replay_contract_v2_preserves_source_tool_arguments() -> None:
    from trace_collect.simulate_openclaw import _paired_replay_actions

    call_id = "call-seeded"
    raw_arguments = json.dumps({"command": "python -m pytest", "timeout": 600})
    source_actions = [
        {
            "action_type": "llm_call",
            "data": {
                "raw_response": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": call_id,
                                        "function": {
                                            "name": "exec",
                                            "arguments": raw_arguments,
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
        },
        {
            "action_type": "tool_exec",
            "data": {
                "tool_name": "exec",
                "tool_call_id": call_id,
                "tool_args": raw_arguments,
                "tool_result": (
                    "Using --randomly-seed=12345 then --randomly-seed=67890\n"
                    "Exit code: 0"
                ),
                "success": True,
            },
        },
    ]

    actions, contract = _paired_replay_actions(source_actions, contract_version=2)

    assert actions == source_actions
    assert actions is not source_actions
    assert contract == {
        "version": 2,
        "tool_args_policy": "exact_source",
        "exec_timeout_policy": "source_tool_args",
        "require_exact_tool_calls": True,
        "require_source_outcome_match": False,
    }


def test_paired_replay_contract_version_selects_exact_source_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trace_collect.simulate_openclaw import (
        replay_paired_workload_contract_version,
    )

    monkeypatch.setenv("OPENCLAW_REPLAY_PAIRED_WORKLOAD_CONTRACT", "2")

    assert replay_paired_workload_contract_version() == 2


def test_replay_exact_tool_contract_rejects_argument_changes() -> None:
    from trace_collect.openclaw_host_runtime import replay_action_failure_counts

    source = [
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "tool-1",
            "data": {
                "tool_name": "exec",
                "tool_call_id": "call-1",
                "tool_args": '{"command":"true"}',
                "success": True,
            },
        }
    ]
    replay = [
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "tool-1",
            "data": {
                "tool_name": "exec",
                "tool_call_id": "call-1",
                "tool_args": '{"command":"false"}',
                "success": True,
            },
        }
    ]

    counts = replay_action_failure_counts(
        source, replay, require_exact_tool_calls=True
    )

    assert not counts.action_sequence_matches


def test_worker_trace_applies_exact_tool_contract(tmp_path: Path) -> None:
    from trace_collect.openclaw_host_runtime import _worker_trace_action_counts

    source = [
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "tool-1",
            "data": {
                "tool_name": "exec",
                "tool_call_id": "call-1",
                "tool_args": '{"command":"true"}',
                "success": True,
            },
        }
    ]
    trace = tmp_path / "replay.jsonl"
    trace.write_text(
        json.dumps(
            {
                "type": "action",
                "action_type": "tool_exec",
                "action_id": "tool-1",
                "data": {
                    "tool_name": "exec",
                    "tool_call_id": "call-1",
                    "tool_args": '{"command":"false"}',
                    "success": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    counts = _worker_trace_action_counts(
        trace, source, require_exact_tool_calls=True
    )

    assert not counts.action_sequence_matches


def test_openclaw_replay_stops_before_unrecorded_final_tool_call() -> None:
    from trace_collect.openclaw_host_runtime import (
        OpenClawReplayProvider,
        _replay_execution_completed,
        replay_action_failure_counts,
    )

    source_action = {
        "type": "action",
        "action_type": "llm_call",
        "data": {
            "raw_response": {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "pending",
                                    "function": {
                                        "name": "exec",
                                        "arguments": '{"command":"true"}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        },
    }
    provider = OpenClawReplayProvider(
        llm_actions=[source_action],
        replay_speed=1.0,
        timing_mode="source_scaled",
        stop_before_final_tool_calls=True,
    )
    response = asyncio.run(provider.chat([]))
    replay_action = {"type": "action", "action_type": "llm_call", "data": {}}
    counts = replay_action_failure_counts([source_action], [replay_action])

    assert response.finish_reason == "error"
    assert response.tool_calls == []
    assert provider.source_terminal_boundary_reached
    assert _replay_execution_completed(
        action_counts=counts,
        expected_actions=1,
        stop_reason="error",
        error=response.content,
        source_terminal_reason="trace_ended_before_tools",
        source_terminal_boundary_reached=True,
        provider_request_sequence_matches=True,
    )


def test_openclaw_replay_does_not_retry_recorded_transient_error() -> None:
    from trace_collect.openclaw_host_runtime import OpenClawReplayProvider

    provider = OpenClawReplayProvider(
        llm_actions=[
            {
                "action_type": "llm_call",
                "data": {
                    "raw_response": {
                        "choices": [
                            {
                                "finish_reason": "error",
                                "message": {"content": "504 Gateway Timeout"},
                            }
                        ]
                    }
                },
            }
        ],
        replay_speed=1.0,
        timing_mode="source_scaled",
    )

    response = asyncio.run(
        provider.chat_with_retry(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AA=="},
                        }
                    ],
                }
            ]
        )
    )

    assert response.content == "504 Gateway Timeout"
    assert response.finish_reason == "error"
    assert provider._index == 1
    assert provider.request_sequence_matches


def test_openclaw_replay_rejects_provider_error_before_source_response() -> None:
    from trace_collect.openclaw_host_runtime import OpenClawReplayProvider

    provider = OpenClawReplayProvider(
        llm_actions=[
            {
                "action_type": "llm_call",
                "data": {
                    "completion_tokens": "invalid",
                    "raw_response": {
                        "choices": [
                            {
                                "finish_reason": "error",
                                "message": {"content": "recorded error"},
                            }
                        ]
                    },
                },
            }
        ],
        replay_speed=1.0,
        timing_mode="ttft_tpot",
        llm_ttft_ms=1.0,
        llm_tpot_ms=1.0,
    )

    response = asyncio.run(provider.chat_with_retry([]))

    assert response.finish_reason == "error"
    assert not provider.request_sequence_matches


def test_failed_source_terminal_reason_comes_from_trace_shape() -> None:
    from trace_collect.simulate_openclaw import _source_terminal_reason

    def loaded(
        last: dict[str, object],
        *,
        n_iterations: int,
        max_iterations: int = 100,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            summary={"success": False, "n_iterations": n_iterations},
            metadata={"max_iterations": max_iterations},
            actions=[last],
        )

    def llm(finish_reason: str, *, tool_calls: bool = False) -> dict[str, object]:
        return {
            "action_type": "llm_call",
            "data": {
                "raw_response": {
                    "choices": [
                        {
                            "finish_reason": finish_reason,
                            "message": {
                                "tool_calls": [{"id": "pending"}] if tool_calls else []
                            },
                        }
                    ]
                }
            },
        }

    assert _source_terminal_reason(loaded(llm("error"), n_iterations=11)) == "llm_error"
    assert (
        _source_terminal_reason(loaded({"action_type": "tool_exec"}, n_iterations=100))
        == "max_iterations"
    )
    assert (
        _source_terminal_reason(
            loaded(llm("tool_calls", tool_calls=True), n_iterations=13)
        )
        == "trace_ended_before_tools"
    )
    assert (
        _source_terminal_reason(
            loaded(llm("tool_calls", tool_calls=True), n_iterations=100)
        )
        == "trace_ended_before_tools"
    )
    assert (
        _source_terminal_reason(loaded({"action_type": "tool_exec"}, n_iterations=13))
        == "trace_ended_after_tools"
    )


def test_llm_replay_duration_rejects_invalid_completion_tokens() -> None:
    from trace_collect.openclaw_host_runtime import llm_replay_duration_s

    with pytest.raises(ValueError):
        llm_replay_duration_s(
            data={"completion_tokens": "not-an-int"},
            source_duration_s=1.0,
            replay_speed=1.0,
            timing_mode="ttft_tpot",
            llm_ttft_ms=10.0,
            llm_tpot_ms=2.0,
        )


def test_llm_replay_duration_rejects_negative_completion_tokens() -> None:
    from trace_collect.openclaw_host_runtime import llm_replay_duration_s

    with pytest.raises(ValueError, match="completion_tokens must be non-negative"):
        llm_replay_duration_s(
            data={"completion_tokens": -1},
            source_duration_s=1.0,
            replay_speed=1.0,
            timing_mode="ttft_tpot",
            llm_ttft_ms=10.0,
            llm_tpot_ms=2.0,
        )


def test_source_model_prefers_summary_and_metadata_audit_fields() -> None:
    from trace_collect.simulate_utils import _source_model
    from trace_collect.simulator import LoadedTraceSession

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
    from trace_collect.openclaw_host_runtime import (
        _replay_execution_completed,
        replay_action_failure_counts,
    )

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
    assert counts.action_sequence_matches
    completion_args = {
        "action_counts": counts,
        "expected_actions": 2,
        "stop_reason": "completed",
        "error": None,
        "source_terminal_reason": "completed",
        "source_terminal_boundary_reached": False,
        "provider_request_sequence_matches": True,
    }
    assert not _replay_execution_completed(**completion_args)
    assert _replay_execution_completed(
        **completion_args, require_source_outcome_match=False
    )


def test_replay_failure_counts_rejects_extra_actions() -> None:
    from trace_collect.openclaw_host_runtime import replay_action_failure_counts

    source = [{"type": "action", "action_type": "llm_call", "data": {}}]
    replay = [
        {"type": "action", "action_type": "llm_call", "data": {}},
        {"type": "action", "action_type": "tool_exec", "data": {"tool_name": "exec"}},
    ]

    counts = replay_action_failure_counts(source, replay)
    wrong_tool = replay_action_failure_counts(
        [{"type": "action", "action_type": "tool_exec", "data": {"tool_name": "exec"}}],
        [
            {
                "type": "action",
                "action_type": "tool_exec",
                "data": {"tool_name": "read_file"},
            }
        ],
    )

    assert counts.emitted_actions == 2
    assert not counts.action_sequence_matches
    assert not wrong_tool.action_sequence_matches


def test_allowed_outcome_drift_is_not_a_framework_failure() -> None:
    from trace_collect.openclaw_host_runtime import (
        ReplayActionFailureCounts,
        replay_framework_failure_count,
    )

    counts = ReplayActionFailureCounts(
        emitted_actions=1,
        source_failed_actions=0,
        replay_failed_actions=1,
        unexpected_replay_failed_actions=1,
        action_sequence_matches=True,
    )

    assert replay_framework_failure_count(
        counts, missing_actions=0, require_source_outcome_match=False
    ) == 0
    assert replay_framework_failure_count(
        counts, missing_actions=0, require_source_outcome_match=True
    ) == 1


def test_openclaw_container_mode_replays_llm_via_host_replay_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Container OpenClaw traces must use the host replay runner/provider for LLM calls."""
    monkeypatch.setenv("OPENCLAW_REPLAY_EXEC_TIMEOUT_FLOOR_S", "3600")
    monkeypatch.setenv("OPENCLAW_REPLAY_PAIRED_WORKLOAD_CONTRACT", "1")
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
        assert _kwargs.get("start_agent") is False
        return PreparedTraceSession(
            loaded=loaded,
            container=PreparedContainer(
                container_id="cid-openclaw-replay",
                container_executable=container_executable,
                docker_image="swebench/test-image",
                agent=_FakeAgent(),
            ),
            task_output_dir=task_output_dir,
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

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session", fake_prepare
    )
    monkeypatch.setattr(
        "trace_collect.simulator._prefetch_container_images", fake_prefetch
    )
    monkeypatch.setattr(
        "trace_collect.simulator._prebuild_sweep_fixed_images", fake_prebuild
    )
    monkeypatch.setattr(
        "trace_collect.simulator._sleep_and_measure",
        fail_on_simulator_side_llm_sleep,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.stop_task_container",
        lambda *_args, **_kwargs: "",
    )

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=1.0,
            shadow_llm_api_base="http://127.0.0.1:8000/v1",
            shadow_llm_model="meta-llama/Llama-3.1-8B-Instruct",
            shadow_llm_timeout_s=12.0,
            shadow_llm_seed=7,
        )
    )

    assert trace_file.exists()
    records = [
        json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()
    ]
    summary = next(record for record in records if record["type"] == "summary")
    metadata = next(record for record in records if record["type"] == "trace_metadata")
    assert summary["agent_execution_environment"] == "host"
    assert summary["tool_execution_environment"] == "task_container"
    assert summary["tool_container_user"] == "root"
    assert summary["openclaw_host_pid"] == 4321
    assert summary["sleep_drift"]["by_phase"]["llm_replay"]["sample_count"] == 1
    assert summary["source_model"] == "qwen/qwen3.7-max"
    assert metadata["exec_timeout_floor_s"] == 3_600.0
    assert metadata["shadow_generation"] == {
        "api_base": "http://127.0.0.1:8000/v1",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "timeout_s": 12.0,
        "seed": 7,
    }
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
    assert request["exec_timeout_floor_s"] == 3_600.0
    assert request["shadow_generation"] == metadata["shadow_generation"]
    assert request["paired_workload_contract"] is True
    assert request["replay_action_contract"]["require_source_outcome_match"] is False


def test_openclaw_host_replay_worker_failure_marks_failed_with_audit_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failing host worker produces failed stats and source/task audit fields."""
    from harness.trace_logger import TraceLogger
    from trace_collect.simulate_openclaw import _run_openclaw_replay_session
    from trace_collect.simulator import (
        LLMTimingConfig,
        LoadedTraceSession,
        PreparedContainer,
        PreparedTraceSession,
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
    worker_timeouts: list[float] = []

    async def fake_worker_process(
        *,
        request_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        timeout_s: float,
    ) -> int:
        worker_timeouts.append(timeout_s)
        request = json.loads(request_path.read_text(encoding="utf-8"))
        assert request["task_instance_id"] == "fc_openclaw_failed_replay"
        assert request["source_action_agent_id"] == "cli:oc-failed"
        assert request["tool_resource_run_token"] == "shared-run-token"
        Path(request["status_path"]).write_text(
            json.dumps(
                {
                    "success": True,
                    "stop_reason": "completed",
                    "error": None,
                    "replay_execution": "completed",
                    "elapsed_s": 0.5,
                    "sleep_records": [],
                    "agent_execution_environment": "host",
                    "tool_execution_environment": "task_container",
                    "tool_container_id": request["container_id"],
                    "tool_container_user": "root",
                    "tool_container_user_id": "0",
                    "tool_container_workdir": "/testbed",
                    "openclaw_host_pid": 98765,
                    "telemetry_integrity_failed": False,
                    "telemetry_quality": "ok",
                    "formal_completeness": "partial",
                    "call_coverage": {
                        "total_call_count": 31,
                        "eligible_call_count": 30,
                        "withheld_call_count": 1,
                        "eligible_fraction": 30 / 31,
                    },
                    "collection_validity": "valid",
                    "telemetry_errors": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        Path(request["output_trace"]).write_text(
            "".join(json.dumps(action) + "\n" for action in request["source_actions"]),
            encoding="utf-8",
        )
        Path(request["resource_artifact_path"]).write_text(
            json.dumps(
                {
                    "version": CLAUSE_TELEMETRY_SCHEMA_VERSION,
                    "status_model": CLAUSE_TELEMETRY_STATUS_MODEL,
                    "replay_execution": "completed",
                    "telemetry_quality": "ok",
                    "formal_completeness": "partial",
                    "call_coverage": {
                        "total_call_count": 31,
                        "eligible_call_count": 30,
                        "withheld_call_count": 1,
                        "eligible_fraction": 30 / 31,
                    },
                    "collection_validity": "valid",
                    "integrity": {"status": "ok", "errors": []},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        stdout_path.write_text("stdout text", encoding="utf-8")
        stderr_path.write_text("stderr text", encoding="utf-8")
        return 7

    monkeypatch.setattr(
        "trace_collect.simulate_openclaw._run_openclaw_worker_process",
        fake_worker_process,
    )
    monkeypatch.setenv("TOOL_RESOURCE_PROFILE", str(tmp_path / "resource.yaml"))
    monkeypatch.setenv(
        "TOOL_RESOURCE_RUN_TOKENS",
        json.dumps({"task:fc_openclaw_failed_replay": "shared-run-token"}),
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
    assert stats.failed_action_count == 1
    assert summary["success"] is False
    assert summary["failed_actions"] == 1
    assert summary["task_instance_id"] == "fc_openclaw_failed_replay"
    assert summary["source_action_agent_id"] == "cli:oc-failed"
    assert summary["worker_returncode"] == 7
    assert summary["worker_error"] is None
    assert worker_timeouts == [602.0]
    assert summary["replay_execution"] == "failed"
    assert summary["telemetry_quality"] == "ok"
    assert summary["formal_completeness"] == "partial"
    assert summary["call_coverage"]["eligible_call_count"] == 30
    assert summary["collection_validity"] == "valid"
    assert summary["telemetry_integrity_failed"] is False
    resource_artifact = json.loads(
        (prepared.task_output_dir / "resource_observations.json").read_text(
            encoding="utf-8"
        )
    )
    assert resource_artifact["replay_execution"] == "completed"
    assert resource_artifact["formal_completeness"] == "partial"
    assert resource_artifact["collection_validity"] == "valid"
    assert resource_artifact["integrity"]["status"] == "ok"
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
            return subprocess.CompletedProcess(
                cmd, 0, stdout="amd64 linux\n", stderr=""
            )
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
            "/installed-agent/python/bin/python3",
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
