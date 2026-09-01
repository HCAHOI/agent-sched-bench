from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path

import pytest

import trace_collect.simulator as simulator_module
from harness.container_image_prep import normalize_image_reference
from harness.trace_logger import TraceLogger
from trace_collect.cli import _run_simulate, parse_simulate_args
from trace_collect.resource_timeline import RESOURCE_TIMELINE_SCHEMA_VERSION
from trace_collect.simulate_manifest import _parse_trace_session_file
from trace_collect.simulate_outputs import (
    _make_trace_action,
    _replay_agent_id_for_action,
)
from trace_collect.simulate_types import (
    LLMTimingConfig,
    LoadedTraceSession,
    PreparedContainer,
    PreparedTraceSession,
    ReplayTaskStats,
    SimulateError,
    WorkerTraceInput,
)
from trace_collect.simulate_utils import _resolve_prep_concurrency
from trace_collect.simulator import (
    _build_image_cleanup_state,
    _chunk_worker_inputs_by_concurrency,
    _partition_worker_inputs,
    _release_source_image,
    _run_cloud_model_queue,
    _run_worker_wave_async,
    _source_exec_timeout_s,
    simulate,
)


def _write_trace(
    path: Path,
    *,
    agent_id: str,
    scaffold: str = "tongyi-deepresearch",
    llm_start: float = 100.0,
    llm_end: float = 100.2,
    tool_start: float = 100.4,
    tool_end: float = 100.45,
    tool_name: str = "write_file",
    execution_environment: str = "container",
    resource_timeline: dict | None = None,
    tool_args: dict | None = None,
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
                            "tool_args": json.dumps(tool_args or {"path": "/testbed/x.txt"}),
                            "tool_result": "source-result",
                            "duration_ms": (tool_end - tool_start) * 1000,
                            "success": True,
                            **(
                                {"resource_timeline": resource_timeline}
                                if resource_timeline is not None
                                else {}
                            ),
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


@pytest.fixture(autouse=True)
def _fake_container_resource_recorders(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    recorders: list[object] = []

    class _FakeContainerResourceRecorder:
        def __init__(
            self,
            *,
            output_dir: Path,
            run_id: str,
            interval_s: float,
            executable: str,
            sample_all_containers: bool,
            collect_cgroup_memory_access: bool = True,
            monitoring_policy: dict[str, object] | None = None,
        ) -> None:
            self.output_dir = Path(output_dir)
            self.run_id = run_id
            self.interval_s = interval_s
            self.executable = executable
            self.sample_all_containers = sample_all_containers
            self.collect_cgroup_memory_access = collect_cgroup_memory_access
            self.monitoring_policy = dict(monitoring_policy or {})
            self.started = False
            self.stopped = False
            self.registered: list[str] = []
            self.unregistered: list[str] = []
            self.jsonl_path = self.output_dir / f"{run_id}.container_resources.jsonl"
            self.summary_path = (
                self.output_dir / f"{run_id}.container_resources_summary.json"
            )
            recorders.append(self)

        def start(self) -> None:
            self.started = True

        def register_container(self, container_id: str) -> None:
            self.registered.append(container_id)

        def unregister_container(self, container_id: str) -> None:
            self.unregistered.append(container_id)

        def stop(self) -> dict:
            self.stopped = True
            self.output_dir.mkdir(parents=True, exist_ok=True)
            sample = {
                "timestamp": "2026-06-26T00:00:00Z",
                "epoch": 1782470400.0,
                "resource_scope": "global_container",
                "sampler_run_id": self.run_id,
                "container_id": "fake-cid",
                "container_short_id": "fake-cid",
                "container_name": "fake-task",
                "container_image": "fake-image",
                "mem_usage": "1MiB / 1GiB",
                "mem_percent": "0.1%",
                "cpu_percent": "0.5%",
                "net_io": "0B / 0B",
                "net_rx_bytes": 0,
                "net_tx_bytes": 0,
            }
            self.jsonl_path.write_text(
                json.dumps(sample, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            summary = {
                "run_id": self.run_id,
                "jsonl_path": str(self.jsonl_path),
                "summary_path": str(self.summary_path),
                "sample_count": 1,
                "monitoring": dict(self.monitoring_policy),
                "sampling": {
                    "interval_s": self.interval_s,
                    "scope": "registered_containers",
                    "sample_all_containers": self.sample_all_containers,
                    "collect_cgroup_memory_access": self.collect_cgroup_memory_access,
                    "tick_count": 1,
                    "empty_tick_count": 0,
                    "stop_complete": True,
                },
                "containers": [],
                "errors": [],
            }
            self.summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return summary

    monkeypatch.setattr(
        "trace_collect.simulator.ContainerResourceRecorder",
        _FakeContainerResourceRecorder,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.stop_task_container",
        lambda *args, **kwargs: "",
    )
    return recorders


def _patch_simulator_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    tool_delay_s: float = 0.0,
    tool_duration_ms: float = 8.0,
    tool_result_prefix: str = "executed",
) -> None:
    class _FakeAgent:
        async def stop(self): pass

    async def fake_prepare_container(
        loaded,
        *,
        task_output_dir=None,
        container_executable,
        network_mode="host",
        **_kwargs,
    ):
        from trace_collect.simulator import PreparedContainer, PreparedTraceSession
        container = PreparedContainer(
            container_id="fake-cid",
            container_executable=container_executable,
            docker_image="fake-image",
            agent=_FakeAgent(),
        )
        return PreparedTraceSession(
            loaded=loaded,
            container=container,
            task_output_dir=task_output_dir,
        )

    async def fake_exec_tool(
        agent,
        tool_name,
        tool_args_json,
        command_timeout_s,
        source_exec_timeout_s=None,
        allow_source_runtime_artifacts=False,
        source_resource_timeline=None,
    ):
        if tool_delay_s > 0:
            await asyncio.sleep(tool_delay_s)
        return f"{tool_result_prefix}-{tool_name}", tool_duration_ms, True

    async def fake_prefetch(*_args, **_kwargs) -> None:
        pass

    async def fake_prebuild(*_args, **_kwargs) -> dict[str, str]:
        return {}

    class _FakeSampler:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> list[dict]:
            return []

    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", fake_prepare_container)
    monkeypatch.setattr("trace_collect.simulator._prefetch_container_images", fake_prefetch)
    monkeypatch.setattr("trace_collect.simulator._prebuild_sweep_fixed_images", fake_prebuild)
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)


def _patch_noop_sweep_fixed_prebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_prebuild(*_args, **_kwargs) -> dict[str, str]:
        return {}

    monkeypatch.setattr(
        "trace_collect.simulator._prebuild_sweep_fixed_images",
        fake_prebuild,
    )


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
    assert args.workers == 1
    assert args.prep_concurrency == 0
    assert args.resource_monitoring == "auto"
    assert args.pmu_monitoring == "auto"
    assert args.memory_bandwidth_monitoring == "auto"
    assert args.replay_speed == 1.0
    assert args.llm_timing == "source-scaled"
    assert args.llm_ttft_ms is None
    assert args.llm_tpot_ms is None
    assert args.shadow_llm_max_concurrency is None
    assert args.stage_all_before_replay is False


def test_parse_simulate_args_accepts_workers_and_monitoring_policy() -> None:
    args = parse_simulate_args(
        [
            "--manifest",
            "manifest.yaml",
            "--workers",
            "16",
            "--prep-concurrency",
            "64",
            "--resource-monitoring",
            "off",
            "--pmu-monitoring",
            "off",
            "--memory-bandwidth-monitoring",
            "off",
        ]
    )

    assert args.workers == 16
    assert args.prep_concurrency == 64
    assert args.resource_monitoring == "off"
    assert args.pmu_monitoring == "off"
    assert args.memory_bandwidth_monitoring == "off"


def test_parse_simulate_args_accepts_stage_all_before_replay() -> None:
    args = parse_simulate_args(
        ["--manifest", "manifest.yaml", "--stage-all-before-replay"]
    )

    assert args.stage_all_before_replay is True


def test_parse_simulate_args_accepts_ttft_tpot_llm_timing() -> None:
    args = parse_simulate_args(
        [
            "--mode",
            "cloud_model",
            "--manifest",
            "manifest.yaml",
            "--llm-timing",
            "ttft-tpot",
            "--llm-ttft-ms",
            "800",
            "--llm-tpot-ms",
            "20",
        ]
    )

    assert args.llm_timing == "ttft-tpot"
    assert args.llm_ttft_ms == 800.0
    assert args.llm_tpot_ms == 20.0


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


def test_worker_partition_helpers_preserve_order_and_limits() -> None:
    inputs = [
        WorkerTraceInput(
            source_trace=f"/tmp/trace-{index}.jsonl",
            task_source="/tmp/tasks.json",
            manifest_index=index,
            docker_image_override=None,
            label=None,
            run_instance_id=f"task-{index}",
            task_instance_id=f"task-{index}",
            source_action_agent_id=f"task-{index}",
        )
        for index in range(7)
    ]

    waves = _chunk_worker_inputs_by_concurrency(inputs, 3)
    assert [[entry.run_instance_id for entry in wave] for wave in waves] == [
        ["task-0", "task-1", "task-2"],
        ["task-3", "task-4", "task-5"],
        ["task-6"],
    ]
    chunks = _partition_worker_inputs(waves[0], 2)
    assert [[entry.run_instance_id for entry in chunk] for chunk in chunks] == [
        ["task-0", "task-1"],
        ["task-2"],
    ]


def test_staged_queue_prepares_every_task_before_fifo_admission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sessions = [
        LoadedTraceSession(
            source_trace=tmp_path / f"trace-{index}.jsonl",
            task_source=tmp_path / "tasks.json",
            task_instance_id=f"task-{index}",
            source_action_agent_id=f"task-{index}",
            run_instance_id=f"task-{index}",
            manifest_index=index,
            scaffold="openclaw",
            metadata={"source_model": "model"},
            summary=None,
            task={"instance_id": f"task-{index}"},
            actions=[],
            iterations={},
        )
        for index in range(3)
    ]
    prepared_ids: list[str] = []
    admitted_ids: list[str] = []
    finalized_ids: list[str] = []
    active = 0
    max_active = 0

    async def fake_prepare(
        loaded: LoadedTraceSession,
        **_kwargs,
    ) -> PreparedTraceSession:
        prepared_ids.append(loaded.run_instance_id)
        return PreparedTraceSession(loaded=loaded)

    async def fake_replay(
        prepared: PreparedTraceSession,
        **_kwargs,
    ) -> ReplayTaskStats:
        nonlocal active, max_active
        assert len(prepared_ids) == len(sessions)
        admitted_ids.append(prepared.loaded.run_instance_id)
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        loaded = prepared.loaded
        return ReplayTaskStats(
            agent_id=loaded.agent_id,
            run_instance_id=loaded.run_instance_id,
            source_agent_id=loaded.source_action_agent_id,
            manifest_index=loaded.manifest_index,
            label=loaded.label,
            source_trace=str(loaded.source_trace),
            success=True,
            elapsed_s=0.01,
            action_count=0,
            llm_call_count=0,
            tool_exec_count=0,
        )

    async def fake_finalize(prepared: PreparedTraceSession) -> None:
        finalized_ids.append(prepared.loaded.run_instance_id)

    monkeypatch.setattr(simulator_module, "_prepare_replay_session", fake_prepare)
    monkeypatch.setattr(simulator_module, "_replay_cloud_model_session", fake_replay)
    monkeypatch.setattr(simulator_module, "_finalize_prepared_session", fake_finalize)
    monkeypatch.setattr(simulator_module, "_REPLAY_START_DELAY_S", 0.001)

    prepared, stats, common_ready_wall_time_s = asyncio.run(
        simulator_module._run_staged_cloud_model_queue(
            sessions,
            output_path=tmp_path / "out",
            trace_logger=object(),
            concurrency=2,
            prep_concurrency=2,
            container_executable=None,
            network_mode="host",
            container_resource_recorder=None,
            replay_speed=1.0,
            llm_timing=LLMTimingConfig(),
            command_timeout_s=1.0,
            warmup_skip_iterations=0,
        )
    )

    assert [item.loaded.run_instance_id for item in prepared] == [
        "task-0",
        "task-1",
        "task-2",
    ]
    assert admitted_ids == ["task-0", "task-1", "task-2"]
    assert sorted(finalized_ids) == ["task-0", "task-1", "task-2"]
    assert max_active == 2
    assert common_ready_wall_time_s <= time.time()
    stats_by_id = {stat.run_instance_id: stat for stat in stats}
    assert stats_by_id["task-0"].admission_wait_s is not None
    assert stats_by_id["task-1"].admission_wait_s is not None
    assert stats_by_id["task-2"].admission_wait_s > 0.005
    assert all(
        stat.ready_to_terminal_s >= stat.admission_wait_s for stat in stats
    )


def test_tool_gap_loan_staged_queue_admits_one_waiter_per_loan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sessions = [
        LoadedTraceSession(
            source_trace=tmp_path / f"trace-{index}.jsonl",
            task_source=tmp_path / "tasks.json",
            task_instance_id=f"task-{index}",
            source_action_agent_id=f"task-{index}",
            run_instance_id=f"task-{index}",
            manifest_index=index,
            scaffold="openclaw",
            metadata={"source_model": "model"},
            summary=None,
            task={"instance_id": f"task-{index}"},
            actions=[],
            iterations={},
        )
        for index in range(8)
    ]
    admitted: list[str] = []
    configs: dict[str, object] = {}
    active = 0
    max_active = 0
    four_started = asyncio.Event()
    borrower_finished = asyncio.Event()
    release_foreground = asyncio.Event()
    reused_loan = False
    finalized: list[str] = []

    async def fake_prepare(
        loaded: LoadedTraceSession,
        **_kwargs,
    ) -> PreparedTraceSession:
        return PreparedTraceSession(loaded=loaded)

    async def fake_replay(
        prepared: PreparedTraceSession,
        *,
        tool_gap_loan: object,
        **_kwargs,
    ) -> ReplayTaskStats:
        nonlocal active, max_active, reused_loan
        task_id = prepared.loaded.task_instance_id
        admitted.append(task_id)
        configs[task_id] = tool_gap_loan
        active += 1
        max_active = max(max_active, active)
        if len(admitted) == 4:
            four_started.set()
        if task_id == "task-0":
            await four_started.wait()
            loan_dir = tmp_path / "out" / ".tool-gap-loan" / "loans"
            loan_dir.mkdir(parents=True)
            (loan_dir / "task-0.json").write_text(
                json.dumps(
                    {
                        "task_id": "task-0",
                        "trigger": "feedback",
                        "decision_wall_time_s": 2.0,
                        "call_id": "call-0",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            await release_foreground.wait()
        elif task_id == "task-1":
            await borrower_finished.wait()
            await asyncio.sleep(0.03)
            reused_loan = "task-5" in admitted
            release_foreground.set()
        elif task_id in {"task-2", "task-3"}:
            await release_foreground.wait()
        elif task_id == "task-4":
            borrower_finished.set()
        await asyncio.sleep(0)
        active -= 1
        loaded = prepared.loaded
        return ReplayTaskStats(
            agent_id=loaded.agent_id,
            run_instance_id=loaded.run_instance_id,
            source_agent_id=loaded.source_action_agent_id,
            manifest_index=loaded.manifest_index,
            label=loaded.label,
            source_trace=str(loaded.source_trace),
            success=True,
            elapsed_s=0.01,
            action_count=0,
            llm_call_count=0,
            tool_exec_count=0,
        )

    async def fake_finalize(prepared: PreparedTraceSession) -> None:
        task_id = prepared.loaded.run_instance_id
        if task_id not in finalized:
            finalized.append(task_id)

    monkeypatch.setattr(simulator_module, "_prepare_replay_session", fake_prepare)
    monkeypatch.setattr(simulator_module, "_replay_cloud_model_session", fake_replay)
    monkeypatch.setattr(simulator_module, "_finalize_prepared_session", fake_finalize)
    monkeypatch.setattr(simulator_module, "_REPLAY_START_DELAY_S", 0.001)

    asyncio.run(
        simulator_module._run_staged_cloud_model_queue(
            sessions,
            output_path=tmp_path / "out",
            trace_logger=object(),
            concurrency=4,
            prep_concurrency=8,
            container_executable=None,
            network_mode="host",
            container_resource_recorder=None,
            replay_speed=1.0,
            llm_timing=LLMTimingConfig(),
            command_timeout_s=1.0,
            warmup_skip_iterations=0,
            tool_gap_arm="feedback",
            tool_gap_predictions={},
            tool_gap_borrower_priority=1,
        )
    )

    assert admitted[:5] == ["task-0", "task-1", "task-2", "task-3", "task-4"]
    assert max_active == 5
    assert reused_loan is False
    assert configs["task-0"].can_lend is True
    assert configs["task-4"].can_lend is False
    assert configs["task-4"].borrower_priority == 1
    assert sorted(finalized) == [f"task-{index}" for index in range(8)]
    summary = json.loads(
        (tmp_path / "out" / ".tool-gap-loan" / "summary.json").read_text()
    )
    assert summary["effective_max_concurrency"] == 5
    assert summary["observed_loan_count"] == 1
    assert [
        (release["lender_task_id"], release["borrower_task_id"])
        for release in summary["loan_releases"]
    ] == [("task-0", "task-4")]


def test_tool_gap_staged_queue_cancellation_joins_active_replays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sessions = [
        LoadedTraceSession(
            source_trace=tmp_path / f"trace-{index}.jsonl",
            task_source=tmp_path / "tasks.json",
            task_instance_id=f"task-{index}",
            source_action_agent_id=f"task-{index}",
            run_instance_id=f"task-{index}",
            manifest_index=index,
            scaffold="openclaw",
            metadata={"source_model": "model"},
            summary=None,
            task={"instance_id": f"task-{index}"},
            actions=[],
            iterations={},
        )
        for index in range(8)
    ]
    all_started = asyncio.Event()
    started: list[str] = []
    cancelled: list[str] = []
    finalized: list[str] = []

    async def fake_prepare(
        loaded: LoadedTraceSession,
        **_kwargs,
    ) -> PreparedTraceSession:
        return PreparedTraceSession(loaded=loaded)

    async def fake_replay(
        prepared: PreparedTraceSession,
        **_kwargs,
    ) -> ReplayTaskStats:
        task_id = prepared.loaded.run_instance_id
        started.append(task_id)
        if len(started) == 4:
            all_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(task_id)
            raise

    async def fake_finalize(prepared: PreparedTraceSession) -> None:
        finalized.append(prepared.loaded.run_instance_id)

    monkeypatch.setattr(simulator_module, "_prepare_replay_session", fake_prepare)
    monkeypatch.setattr(simulator_module, "_replay_cloud_model_session", fake_replay)
    monkeypatch.setattr(simulator_module, "_finalize_prepared_session", fake_finalize)
    monkeypatch.setattr(simulator_module, "_REPLAY_START_DELAY_S", 0.001)

    async def cancel_queue() -> None:
        queue = asyncio.create_task(
            simulator_module._run_staged_cloud_model_queue(
                sessions,
                output_path=tmp_path / "out",
                trace_logger=object(),
                concurrency=4,
                prep_concurrency=8,
                container_executable=None,
                network_mode="host",
                container_resource_recorder=None,
                replay_speed=1.0,
                llm_timing=LLMTimingConfig(),
                command_timeout_s=1.0,
                warmup_skip_iterations=0,
                tool_gap_arm="fixed",
            )
        )
        await all_started.wait()
        queue.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queue

    asyncio.run(cancel_queue())

    assert sorted(started) == [f"task-{index}" for index in range(4)]
    assert sorted(cancelled) == sorted(started)
    assert sorted(finalized) == sorted(started)


def test_staged_queue_cleans_successful_preparations_when_one_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sessions = [
        LoadedTraceSession(
            source_trace=tmp_path / f"{task_id}.jsonl",
            task_source=tmp_path / "tasks.json",
            task_instance_id=task_id,
            source_action_agent_id=task_id,
            run_instance_id=task_id,
            manifest_index=index,
            scaffold="openclaw",
            metadata={"source_model": "model"},
            summary=None,
            task={"instance_id": task_id},
            actions=[],
            iterations={},
        )
        for index, task_id in enumerate(("good", "bad"))
    ]
    finalized: list[str] = []
    released: list[str] = []

    async def fake_prepare(
        loaded: LoadedTraceSession,
        **_kwargs,
    ) -> PreparedTraceSession:
        if loaded.run_instance_id == "bad":
            raise RuntimeError("prep failed")
        return PreparedTraceSession(loaded=loaded)

    async def fake_finalize(prepared: PreparedTraceSession) -> None:
        finalized.append(prepared.loaded.run_instance_id)

    async def fake_release(_cleanup_state: object, run_instance_id: str) -> None:
        released.append(run_instance_id)

    monkeypatch.setattr(simulator_module, "_prepare_replay_session", fake_prepare)
    monkeypatch.setattr(simulator_module, "_finalize_prepared_session", fake_finalize)
    monkeypatch.setattr(simulator_module, "_release_source_image", fake_release)

    with pytest.raises(SimulateError, match="1/2 staged preparations failed"):
        asyncio.run(
            simulator_module._run_staged_cloud_model_queue(
                sessions,
                output_path=tmp_path / "out",
                trace_logger=object(),
                concurrency=1,
                prep_concurrency=2,
                container_executable=None,
                network_mode="host",
                container_resource_recorder=None,
                replay_speed=1.0,
                llm_timing=LLMTimingConfig(),
                command_timeout_s=1.0,
                warmup_skip_iterations=0,
                cleanup_state=object(),
            )
        )

    assert finalized == ["good"]
    assert sorted(released) == ["bad", "good"]


def test_staged_queue_cleans_every_preparation_when_replay_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sessions = [
        LoadedTraceSession(
            source_trace=tmp_path / f"{task_id}.jsonl",
            task_source=tmp_path / "tasks.json",
            task_instance_id=task_id,
            source_action_agent_id=task_id,
            run_instance_id=task_id,
            manifest_index=index,
            scaffold="openclaw",
            metadata={"source_model": "model"},
            summary=None,
            task={"instance_id": task_id},
            actions=[],
            iterations={},
        )
        for index, task_id in enumerate(("bad", "peer"))
    ]
    finalized: list[str] = []
    released: list[str] = []

    async def fake_prepare(
        loaded: LoadedTraceSession,
        **_kwargs,
    ) -> PreparedTraceSession:
        return PreparedTraceSession(loaded=loaded)

    async def fake_replay(
        prepared: PreparedTraceSession,
        **_kwargs,
    ) -> ReplayTaskStats:
        if prepared.loaded.run_instance_id == "bad":
            raise RuntimeError("replay failed")
        loaded = prepared.loaded
        return ReplayTaskStats(
            agent_id=loaded.agent_id,
            run_instance_id=loaded.run_instance_id,
            source_agent_id=loaded.source_action_agent_id,
            manifest_index=loaded.manifest_index,
            label=loaded.label,
            source_trace=str(loaded.source_trace),
            success=True,
            elapsed_s=0.0,
            action_count=0,
            llm_call_count=0,
            tool_exec_count=0,
        )

    async def fake_finalize(prepared: PreparedTraceSession) -> None:
        task_id = prepared.loaded.run_instance_id
        if task_id not in finalized:
            finalized.append(task_id)

    async def fake_release(_cleanup_state: object, run_instance_id: str) -> None:
        released.append(run_instance_id)

    monkeypatch.setattr(simulator_module, "_prepare_replay_session", fake_prepare)
    monkeypatch.setattr(simulator_module, "_replay_cloud_model_session", fake_replay)
    monkeypatch.setattr(simulator_module, "_finalize_prepared_session", fake_finalize)
    monkeypatch.setattr(simulator_module, "_release_source_image", fake_release)
    monkeypatch.setattr(simulator_module, "_REPLAY_START_DELAY_S", 0.001)

    with pytest.raises(SimulateError, match="staged replay workers failed"):
        asyncio.run(
            simulator_module._run_staged_cloud_model_queue(
                sessions,
                output_path=tmp_path / "out",
                trace_logger=object(),
                concurrency=2,
                prep_concurrency=2,
                container_executable=None,
                network_mode="host",
                container_resource_recorder=None,
                replay_speed=1.0,
                llm_timing=LLMTimingConfig(),
                command_timeout_s=1.0,
                warmup_skip_iterations=0,
                cleanup_state=object(),
            )
        )

    assert sorted(finalized) == ["bad", "peer"]
    assert sorted(released) == ["bad", "peer"]



def test_cloud_model_dependency_queue_releases_child_after_parent_finalizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def loaded(task_id: str, *, manifest_index: int, depends_on: tuple[str, ...] = ()) -> LoadedTraceSession:
        return LoadedTraceSession(
            source_trace=tmp_path / f"{task_id}.jsonl",
            task_source=tmp_path / "tasks.json",
            task_instance_id=task_id,
            source_action_agent_id=task_id,
            run_instance_id=task_id,
            manifest_index=manifest_index,
            scaffold="generic",
            metadata={"execution_environment": "host"},
            summary=None,
            task={"instance_id": task_id, "problem_statement": task_id},
            actions=[],
            iterations={},
            depends_on=depends_on,
        )

    events: list[str] = []
    parent = loaded("parent", manifest_index=1)
    child = loaded("child", manifest_index=0, depends_on=("parent",))

    async def fake_prepare(loaded_session: LoadedTraceSession, **_kwargs) -> PreparedTraceSession:
        return PreparedTraceSession(loaded=loaded_session, container=None)

    async def fake_replay(prepared: PreparedTraceSession, **_kwargs) -> ReplayTaskStats:
        loaded_session = prepared.loaded
        events.append(f"start:{loaded_session.task_instance_id}")
        if loaded_session.task_instance_id == "parent":
            await asyncio.sleep(0.01)
        events.append(f"done:{loaded_session.task_instance_id}")
        return ReplayTaskStats(
            agent_id=loaded_session.agent_id,
            run_instance_id=loaded_session.run_instance_id,
            source_agent_id=loaded_session.source_action_agent_id,
            manifest_index=loaded_session.manifest_index,
            label=loaded_session.label,
            source_trace=str(loaded_session.source_trace),
            success=True,
            elapsed_s=0.0,
            action_count=0,
            llm_call_count=0,
            tool_exec_count=0,
            depends_on=loaded_session.depends_on,
        )

    async def fake_finalize(prepared: PreparedTraceSession) -> None:
        events.append(f"finalize:{prepared.loaded.task_instance_id}")

    monkeypatch.setattr("trace_collect.simulator._prepare_replay_session", fake_prepare)
    monkeypatch.setattr("trace_collect.simulator._replay_cloud_model_session", fake_replay)
    monkeypatch.setattr("trace_collect.simulator._finalize_prepared_session", fake_finalize)

    _prepared, task_stats, arrival_zero_wall_time_s = asyncio.run(
        _run_cloud_model_queue(
            [child, parent],
            output_path=tmp_path / "out",
            trace_logger=object(),
            concurrency=2,
            container_executable=None,
            network_mode="host",
            container_resource_recorder=None,
            replay_speed=1.0,
            llm_timing=LLMTimingConfig(),
            command_timeout_s=1.0,
            warmup_skip_iterations=0,
        )
    )

    assert arrival_zero_wall_time_s is None

    stats_by_task = {stat.agent_id: stat for stat in task_stats}
    assert stats_by_task["child"].depends_on == ("parent",)

    assert events == [
        "start:parent",
        "done:parent",
        "finalize:parent",
        "start:child",
        "done:child",
        "finalize:child",
    ]


def test_cloud_model_queue_releases_sessions_at_manifest_arrivals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def loaded(task_id: str, manifest_index: int, arrival_s: float) -> LoadedTraceSession:
        return LoadedTraceSession(
            source_trace=tmp_path / f"{task_id}.jsonl",
            task_source=tmp_path / "tasks.json",
            task_instance_id=task_id,
            source_action_agent_id=task_id,
            run_instance_id=task_id,
            manifest_index=manifest_index,
            scaffold="generic",
            metadata={"execution_environment": "host"},
            summary=None,
            task={"instance_id": task_id, "problem_statement": task_id},
            actions=[],
            iterations={},
            arrival_s=arrival_s,
        )

    starts: dict[str, float] = {}
    sessions = [loaded("now", 0, 0.0), loaded("later", 1, 0.03)]

    async def fake_prepare(loaded_session: LoadedTraceSession, **_kwargs) -> PreparedTraceSession:
        starts[loaded_session.task_instance_id] = time.monotonic()
        return PreparedTraceSession(loaded=loaded_session, container=None)

    async def fake_replay(prepared: PreparedTraceSession, **_kwargs) -> ReplayTaskStats:
        loaded_session = prepared.loaded
        return ReplayTaskStats(
            agent_id=loaded_session.agent_id,
            run_instance_id=loaded_session.run_instance_id,
            source_agent_id=loaded_session.source_action_agent_id,
            manifest_index=loaded_session.manifest_index,
            label=loaded_session.label,
            source_trace=str(loaded_session.source_trace),
            success=True,
            elapsed_s=0.0,
            action_count=0,
            llm_call_count=0,
            tool_exec_count=0,
            arrival_s=loaded_session.arrival_s,
        )

    async def fake_finalize(_prepared: PreparedTraceSession) -> None:
        return None

    monkeypatch.setattr("trace_collect.simulator._prepare_replay_session", fake_prepare)
    monkeypatch.setattr("trace_collect.simulator._replay_cloud_model_session", fake_replay)
    monkeypatch.setattr("trace_collect.simulator._finalize_prepared_session", fake_finalize)

    _prepared, stats, arrival_zero_wall_time_s = asyncio.run(
        _run_cloud_model_queue(
            sessions,
            output_path=tmp_path / "out",
            trace_logger=object(),
            concurrency=2,
            container_executable=None,
            network_mode="host",
            container_resource_recorder=None,
            replay_speed=1.0,
            llm_timing=LLMTimingConfig(),
            command_timeout_s=1.0,
            warmup_skip_iterations=0,
        )
    )

    assert starts["later"] - starts["now"] >= 0.02
    later = next(stat for stat in stats if stat.agent_id == "later")
    assert later.arrival_s == 0.03
    assert later.ready_to_terminal_s is not None
    assert arrival_zero_wall_time_s is not None


def test_cloud_model_queue_replaces_completed_tasks_until_measured_cycle_finishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def loaded(
        task_id: str, manifest_index: int, arrival_s: float
    ) -> LoadedTraceSession:
        return LoadedTraceSession(
            source_trace=tmp_path / f"{task_id}.jsonl",
            task_source=tmp_path / "tasks.json",
            task_instance_id=task_id,
            source_action_agent_id=task_id,
            run_instance_id=task_id,
            manifest_index=manifest_index,
            scaffold="openclaw",
            metadata={"execution_environment": "host"},
            summary=None,
            task={"instance_id": task_id, "problem_statement": f"issue {task_id}"},
            actions=[
                {
                    "type": "action",
                    "action_type": "llm_call",
                    "data": {
                        "messages_in": [{"role": "system", "content": "instructions"}]
                    },
                }
            ],
            iterations={},
            arrival_s=arrival_s,
        )

    sessions = [loaded("short", 0, 0.0), loaded("tail", 1, 0.02)]
    prepared_ids: list[str] = []
    finalized_ids: list[str] = []
    replacement_started_at: list[float] = []
    replacement_cancelled_at: list[float] = []
    tail_finalized_at: list[float] = []

    async def fake_prepare(
        loaded_session: LoadedTraceSession,
        **_kwargs,
    ) -> PreparedTraceSession:
        prepared_ids.append(loaded_session.run_instance_id)
        return PreparedTraceSession(loaded=loaded_session, container=None)

    async def fake_replay(
        prepared: PreparedTraceSession,
        **_kwargs,
    ) -> ReplayTaskStats:
        loaded_session = prepared.loaded
        if "__replacement-" in loaded_session.run_instance_id:
            replacement_started_at.append(time.monotonic())
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                replacement_cancelled_at.append(time.monotonic())
                raise
        elif loaded_session.task_instance_id == "tail":
            await asyncio.sleep(0.04)
        return ReplayTaskStats(
            agent_id=loaded_session.agent_id,
            run_instance_id=loaded_session.run_instance_id,
            source_agent_id=loaded_session.source_action_agent_id,
            manifest_index=loaded_session.manifest_index,
            label=loaded_session.label,
            source_trace=str(loaded_session.source_trace),
            success=True,
            elapsed_s=0.0,
            action_count=0,
            llm_call_count=0,
            tool_exec_count=0,
            arrival_s=loaded_session.arrival_s,
        )

    async def fake_finalize(prepared: PreparedTraceSession) -> None:
        if prepared.loaded.run_instance_id == "tail":
            await asyncio.sleep(0.02)
            tail_finalized_at.append(time.monotonic())
        finalized_ids.append(prepared.loaded.run_instance_id)

    monkeypatch.setattr("trace_collect.simulator._prepare_replay_session", fake_prepare)
    monkeypatch.setattr(
        "trace_collect.simulator._replay_cloud_model_session", fake_replay
    )
    monkeypatch.setattr(
        "trace_collect.simulator._finalize_prepared_session", fake_finalize
    )

    replacement_summary: dict[str, object] = {}
    started_at = time.monotonic()
    prepared, stats, arrival_zero_wall_time_s = asyncio.run(
        _run_cloud_model_queue(
            sessions,
            output_path=tmp_path / "out",
            trace_logger=object(),
            concurrency=2,
            container_executable=None,
            network_mode="host",
            container_resource_recorder=None,
            replay_speed=1.0,
            llm_timing=LLMTimingConfig(),
            command_timeout_s=1.0,
            warmup_skip_iterations=0,
            replacement_delay_mean_s=0.0001,
            replacement_seed=42,
            replacement_summary=replacement_summary,
        )
    )

    assert arrival_zero_wall_time_s is not None
    assert {stat.run_instance_id for stat in stats} == {"short", "tail"}
    assert replacement_started_at[0] - started_at >= 0.015
    replacement = next(
        item.loaded
        for item in prepared
        if "__replacement-" in item.loaded.run_instance_id
    )
    assert replacement.actions[0]["data"]["messages_in"][0]["content"] == (
        f"Replay session: {replacement.run_instance_id}\n\ninstructions"
    )
    assert replacement_cancelled_at[0] < tail_finalized_at[0]
    assert replacement.run_instance_id in prepared_ids
    assert replacement.run_instance_id in finalized_ids
    assert replacement_summary["measured_completed"] == 2
    assert replacement_summary["measured_terminal"] == 2
    assert replacement_summary["background_started"] == 1
    assert replacement_summary["background_completed"] == 0
    assert replacement_summary["background_cancelled"] == 1


def test_cloud_model_queue_continues_after_container_prep_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def loaded(task_id: str, manifest_index: int) -> LoadedTraceSession:
        return LoadedTraceSession(
            source_trace=tmp_path / f"{task_id}.jsonl",
            task_source=tmp_path / "tasks.json",
            task_instance_id=task_id,
            source_action_agent_id=task_id,
            run_instance_id=task_id,
            manifest_index=manifest_index,
            scaffold="openclaw",
            metadata={"source_model": "model"},
            summary=None,
            task={"instance_id": task_id},
            actions=[],
            iterations={},
        )

    bad = loaded("broken-networking", 0)
    good = loaded("next-task", 1)
    events: list[str] = []

    async def fake_prepare(
        loaded_session: LoadedTraceSession,
        **_kwargs,
    ) -> PreparedTraceSession:
        events.append(f"prepare:{loaded_session.task_instance_id}")
        if loaded_session is bad:
            raise RuntimeError(
                "task-container python probe failed: "
                "no Python >=3.11 interpreter found in container"
            )
        return PreparedTraceSession(loaded=loaded_session)

    async def fake_replay(
        prepared: PreparedTraceSession,
        **_kwargs,
    ) -> ReplayTaskStats:
        events.append(f"replay:{prepared.loaded.task_instance_id}")
        loaded_session = prepared.loaded
        return ReplayTaskStats(
            agent_id=loaded_session.agent_id,
            run_instance_id=loaded_session.run_instance_id,
            source_agent_id=loaded_session.source_action_agent_id,
            manifest_index=loaded_session.manifest_index,
            label=loaded_session.label,
            source_trace=str(loaded_session.source_trace),
            success=True,
            elapsed_s=0.0,
            action_count=0,
            llm_call_count=0,
            tool_exec_count=0,
        )

    async def fake_finalize(_prepared: PreparedTraceSession) -> None:
        return None

    monkeypatch.setattr("trace_collect.simulator._prepare_replay_session", fake_prepare)
    monkeypatch.setattr("trace_collect.simulator._replay_cloud_model_session", fake_replay)
    monkeypatch.setattr("trace_collect.simulator._finalize_prepared_session", fake_finalize)
    trace_logger = TraceLogger(tmp_path, "prep-fail-soft")
    try:
        prepared, stats, arrival_zero_wall_time_s = asyncio.run(
            _run_cloud_model_queue(
                [bad, good],
                output_path=tmp_path / "out",
                trace_logger=trace_logger,
                concurrency=1,
                container_executable="docker",
                network_mode="host",
                container_resource_recorder=None,
                replay_speed=1.0,
                llm_timing=LLMTimingConfig(),
                command_timeout_s=1.0,
                warmup_skip_iterations=0,
            )
        )
    finally:
        trace_logger.close()

    assert arrival_zero_wall_time_s is not None
    assert events == [
        "prepare:broken-networking",
        "prepare:next-task",
        "replay:next-task",
    ]
    stats_by_task = {stat.agent_id: stat for stat in stats}
    assert stats_by_task["broken-networking"].success is False
    assert stats_by_task["next-task"].success is True
    failed_prepared = next(item for item in prepared if item.loaded is bad)
    assert failed_prepared.task_output_dir is not None
    startup = json.loads(
        (failed_prepared.task_output_dir / "container_startup.json").read_text(
            encoding="utf-8"
        )
    )
    assert startup["status"] == "failed"
    assert startup["reason"] == "container_python_unavailable"
    failed_summary = next(
        record
        for record in _read_jsonl(trace_logger.path)
        if record.get("type") == "summary"
        and record.get("agent_id") == "broken-networking"
    )
    assert failed_summary["status"] == "failed"
    assert failed_summary["reason"] == "container_python_unavailable"


def test_prepare_failure_preserves_attempt_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = LoadedTraceSession(
        source_trace=tmp_path / "broken.jsonl",
        task_source=tmp_path / "tasks.json",
        task_instance_id="broken",
        source_action_agent_id="broken",
        run_instance_id="broken",
        manifest_index=0,
        scaffold="tongyi-deepresearch",
        metadata={"source_model": "model"},
        summary=None,
        task={"instance_id": "broken"},
        actions=[],
        iterations={},
    )
    finalized: list[PreparedTraceSession] = []

    def fail_startup_write(*_args, **_kwargs) -> None:
        raise RuntimeError(
            "task-container python probe failed: "
            "no Python >=3.11 interpreter found in container"
        )

    async def fail_finalize(prepared: PreparedTraceSession) -> None:
        finalized.append(prepared)
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(simulator_module, "_requires_task_container", lambda _loaded: False)
    monkeypatch.setattr(
        simulator_module.ContainerStartupRecorder,
        "write",
        fail_startup_write,
    )
    monkeypatch.setattr(simulator_module, "_finalize_prepared_session", fail_finalize)

    with pytest.raises(simulator_module._ReplayPreparationError) as exc_info:
        asyncio.run(
            simulator_module._prepare_replay_session(
                loaded,
                output_path=tmp_path / "out",
                container_executable=None,
                network_mode="host",
                container_resource_recorder=None,
                resource_monitoring_enabled=False,
                memory_bandwidth_enabled=False,
                monitoring_policy={},
            )
        )

    assert finalized == [exc_info.value.prepared]
    assert exc_info.value.prepared.task_output_dir is not None
    assert exc_info.value.reason == "container_python_unavailable"
    assert str(exc_info.value.cleanup_error) == "cleanup failed"


def test_resolve_prep_concurrency_preserves_default_limit() -> None:
    assert _resolve_prep_concurrency(0, 640) == 20
    assert _resolve_prep_concurrency(64, 640) == 64
    assert _resolve_prep_concurrency(64, 2) == 2
    with pytest.raises(ValueError, match="prep_concurrency must be >= 0"):
        _resolve_prep_concurrency(-1, 4)


def test_container_control_inspection_records_cpu_and_oom_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(
        [
            '{"NanoCpus":2000000000,"CpuShares":1024,"CpusetCpus":"0-7"}\n',
            '{"Status":"running","Running":true,"OOMKilled":false,"ExitCode":0}\n',
            "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n",
        ]
    )

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(simulator_module.subprocess, "run", fake_run)
    assert simulator_module._inspect_container_cpu_controls(
        container_executable="docker", container_id="container"
    ) == {
        "nano_cpus": 2_000_000_000,
        "cpu_shares": 1024,
        "cpuset_cpus": "0-7",
    }
    assert simulator_module._inspect_container_final_state(
        container_executable="docker", container_id="container"
    ) == {
        "status": "running",
        "running": True,
        "oom_killed": False,
        "exit_code": 0,
        "memory_events": {"low": 0, "high": 0, "max": 0, "oom": 0, "oom_kill": 0},
    }


def test_final_container_inspection_retains_stopped_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **_kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            [],
            0,
            stdout='{"Status":"exited","Running":false,"OOMKilled":true,"ExitCode":137}\n',
            stderr="",
        )

    monkeypatch.setattr(simulator_module.subprocess, "run", fake_run)
    assert simulator_module._inspect_container_final_state(
        container_executable="docker", container_id="container"
    ) == {
        "status": "exited",
        "running": False,
        "oom_killed": True,
        "exit_code": 137,
        "memory_events": None,
    }
    assert len(calls) == 1


def test_final_state_inspection_failure_still_stops_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = LoadedTraceSession(
        source_trace=tmp_path / "trace.jsonl",
        task_source=tmp_path / "tasks.json",
        task_instance_id="task",
        source_action_agent_id="source",
        run_instance_id="task",
        manifest_index=0,
        scaffold="openclaw",
        metadata=None,
        summary=None,
        task={},
        actions=[],
        iterations={},
    )
    prepared = PreparedTraceSession(
        loaded=loaded,
        container=PreparedContainer(
            container_id="container",
            container_executable="docker",
            docker_image="image",
            agent=None,
            inspect_final_state=True,
        ),
    )
    stopped: list[str] = []

    def fail_inspection(**_kwargs):
        raise RuntimeError("inspection failed")

    def record_stop(container_id: str, *, executable: str) -> None:
        assert executable == "docker"
        stopped.append(container_id)

    monkeypatch.setattr(
        simulator_module, "_inspect_container_final_state", fail_inspection
    )
    monkeypatch.setattr(simulator_module, "stop_task_container", record_stop)
    with pytest.raises(RuntimeError, match="inspection failed"):
        asyncio.run(simulator_module._finalize_prepared_session(prepared))
    assert stopped == ["container"]


class _AbortOnlyBarrier:
    def __init__(self) -> None:
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True


class _SetOnlyEvent:
    def __init__(self) -> None:
        self.set_called = False

    def set(self) -> None:
        self.set_called = True


class _ReadyBarrier:
    def wait(self) -> None:
        return None


class _SharedWallTime:
    value = 0.0


def test_worker_wave_continues_after_container_prep_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_trace = tmp_path / "bad.jsonl"
    good_trace = tmp_path / "good.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(bad_trace, agent_id="bad", execution_environment="host")
    _write_trace(good_trace, agent_id="good", execution_environment="host")
    _write_host_tasks(task_source, "bad", "good")
    inputs = [
        WorkerTraceInput(
            source_trace=str(trace),
            task_source=str(task_source),
            manifest_index=index,
            docker_image_override=None,
            label=None,
            run_instance_id=agent_id,
            task_instance_id=agent_id,
            source_action_agent_id=agent_id,
        )
        for index, (agent_id, trace) in enumerate(
            (("bad", bad_trace), ("good", good_trace))
        )
    ]
    output_path = tmp_path / "out"
    finalized: list[str] = []
    prepared_control_args: list[tuple[str, ...]] = []

    async def fake_prepare(
        loaded: LoadedTraceSession,
        **_kwargs,
    ) -> PreparedTraceSession:
        prepared_control_args.append(_kwargs["container_start_extra_args"])
        task_output_dir = output_path / loaded.agent_id / "attempt_1"
        task_output_dir.mkdir(parents=True)
        prepared = PreparedTraceSession(
            loaded=loaded,
            task_output_dir=task_output_dir,
        )
        if loaded.agent_id == "bad":
            (task_output_dir / "container_startup.json").write_text(
                '{"status":"failed","marker":"original-attempt"}\n',
                encoding="utf-8",
            )
            raise simulator_module._ReplayPreparationError(
                prepared,
                "container_python_unavailable",
                RuntimeError("python probe failed"),
            )
        return prepared

    async def fake_run(
        prepared_sessions: list[PreparedTraceSession],
        **_kwargs,
    ) -> list[ReplayTaskStats]:
        assert [prepared.loaded.agent_id for prepared in prepared_sessions] == ["good"]
        loaded = prepared_sessions[0].loaded
        return [
            ReplayTaskStats(
                agent_id=loaded.agent_id,
                run_instance_id=loaded.run_instance_id,
                source_agent_id=loaded.source_action_agent_id,
                manifest_index=loaded.manifest_index,
                label=loaded.label,
                source_trace=str(loaded.source_trace),
                success=True,
                elapsed_s=0.0,
                action_count=0,
                llm_call_count=0,
                tool_exec_count=0,
            )
        ]

    async def fake_finalize(prepared: PreparedTraceSession) -> None:
        finalized.append(prepared.loaded.agent_id)

    monkeypatch.setattr(
        simulator_module,
        "_prepare_replay_session_with_shared_limit",
        fake_prepare,
    )
    monkeypatch.setattr(
        simulator_module,
        "_run_prepared_cloud_model_sessions",
        fake_run,
    )
    monkeypatch.setattr(simulator_module, "_finalize_prepared_session", fake_finalize)

    result = asyncio.run(
        _run_worker_wave_async(
            worker_inputs=inputs,
            output_path=output_path,
            worker_run_id="worker",
            global_run_id="global",
            global_concurrency=2,
            wave_index=0,
            worker_index=0,
            worker_count=1,
            container_executable=None,
            network_mode="host",
            replay_speed=1.0,
            llm_timing=LLMTimingConfig(),
            command_timeout_s=1.0,
            warmup_skip_iterations=0,
            fixed_images_by_source=None,
            resource_monitoring_enabled=False,
            memory_bandwidth_enabled=False,
            monitoring_policy={},
            prep_semaphore=object(),
            replay_start_barrier=_ReadyBarrier(),
            replay_start_event=_SetOnlyEvent(),
            replay_start_wall_time=_SharedWallTime(),
            container_start_extra_args=("--cpus", "2"),
        )
    )

    assert [stat.agent_id for stat in result.task_stats] == ["bad", "good"]
    assert [stat.success for stat in result.task_stats] == [False, True]
    assert result.task_output_dirs["bad"].endswith("bad/attempt_1")
    startup = json.loads(
        (Path(result.task_output_dirs["bad"]) / "container_startup.json").read_text(
            encoding="utf-8"
        )
    )
    assert startup["marker"] == "original-attempt"
    failed_summary = next(
        record
        for record in _read_jsonl(Path(result.trace_file))
        if record.get("type") == "summary" and record.get("agent_id") == "bad"
    )
    assert failed_summary["status"] == "failed"
    assert failed_summary["reason"] == "container_python_unavailable"
    assert finalized == ["bad", "good"]
    assert prepared_control_args == [("--cpus", "2"), ("--cpus", "2")]


def test_worker_wave_finalizes_successful_preparations_after_prepare_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    good_trace = tmp_path / "good.jsonl"
    bad_trace = tmp_path / "bad.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        good_trace,
        agent_id="good",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_trace(
        bad_trace,
        agent_id="bad",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_host_tasks(task_source, "good", "bad")
    inputs = [
        WorkerTraceInput(
            source_trace=str(good_trace),
            task_source=str(task_source),
            manifest_index=0,
            docker_image_override=None,
            label=None,
            run_instance_id="good",
            task_instance_id="good",
            source_action_agent_id="good",
        ),
        WorkerTraceInput(
            source_trace=str(bad_trace),
            task_source=str(task_source),
            manifest_index=1,
            docker_image_override=None,
            label=None,
            run_instance_id="bad",
            task_instance_id="bad",
            source_action_agent_id="bad",
        ),
    ]
    finalized: list[str] = []

    async def fake_prepare(loaded, **_kwargs):
        if loaded.agent_id == "bad":
            raise ValueError("prepare failed")
        return PreparedTraceSession(
            loaded=loaded,
            task_output_dir=tmp_path / loaded.agent_id / "attempt_1",
        )

    async def fake_finalize(prepared: PreparedTraceSession) -> None:
        finalized.append(prepared.loaded.agent_id)

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_replay_session_with_shared_limit",
        fake_prepare,
    )
    monkeypatch.setattr("trace_collect.simulator._finalize_prepared_session", fake_finalize)
    barrier = _AbortOnlyBarrier()
    event = _SetOnlyEvent()

    with pytest.raises(SimulateError, match="worker preparations failed"):
        asyncio.run(
            _run_worker_wave_async(
                worker_inputs=inputs,
                output_path=tmp_path / "out",
                worker_run_id="worker",
                global_run_id="global",
                global_concurrency=2,
                wave_index=0,
                worker_index=0,
                worker_count=1,
                container_executable=None,
                network_mode="host",
                replay_speed=1.0,
                llm_timing=LLMTimingConfig(),
                command_timeout_s=1.0,
                warmup_skip_iterations=0,
                fixed_images_by_source=None,
                resource_monitoring_enabled=False,
                memory_bandwidth_enabled=False,
                monitoring_policy={},
                prep_semaphore=object(),
                replay_start_barrier=barrier,
                replay_start_event=event,
                replay_start_wall_time=object(),
            )
        )

    assert finalized == ["good"]
    assert barrier.aborted is True
    assert event.set_called is True


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
            "--stage-all-before-replay",
        ]
    )

    _run_simulate(args)

    assert seen["mode"] == "cloud_model"
    assert seen["manifest"] == Path("manifest.yaml")
    assert seen["concurrency"] == 1
    assert seen["workers"] == 1
    assert seen["prep_concurrency"] == 0
    assert seen["resource_monitoring"] == "auto"
    assert seen["pmu_monitoring"] == "auto"
    assert seen["memory_bandwidth_monitoring"] == "auto"
    assert seen["container_executable"] is None
    assert seen["llm_timing_mode"] == "source_scaled"
    assert seen["llm_ttft_ms"] is None
    assert seen["llm_tpot_ms"] is None
    assert seen["shadow_llm_max_concurrency"] is None
    assert seen["stage_all_before_replay"] is True


def test_stage_all_before_replay_rejects_multiple_process_workers(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="stage_all_before_replay requires workers=1",
    ):
        asyncio.run(
            simulate(
                manifest=tmp_path / "missing.yaml",
                output_dir=tmp_path / "out",
                workers=2,
                stage_all_before_replay=True,
            )
        )


def test_stage_all_before_replay_rejects_dependency_dag(tmp_path: Path) -> None:
    parent_trace = tmp_path / "parent.jsonl"
    child_trace = tmp_path / "child.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(parent_trace, agent_id="parent", execution_environment="host")
    _write_trace(child_trace, agent_id="child", execution_environment="host")
    task_source.write_text(
        json.dumps(
            [
                {"instance_id": "parent", "problem_statement": "parent"},
                {
                    "instance_id": "child",
                    "problem_statement": "child",
                    "depends_on": ["parent"],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(
        tmp_path / "manifest.yaml",
        [str(parent_trace), str(child_trace)],
    )

    with pytest.raises(
        ValueError,
        match="stage_all_before_replay does not support task dependencies",
    ):
        asyncio.run(
            simulate(
                manifest=manifest,
                task_source=task_source,
                output_dir=tmp_path / "out",
                stage_all_before_replay=True,
            )
        )


@pytest.mark.parametrize("error_type", [ValueError, SimulateError])
def test_run_simulate_cloud_model_reports_policy_errors_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_type: type[Exception],
) -> None:
    async def fake_simulate(**_kwargs):
        raise error_type("--pmu-monitoring on is forbidden for concurrent simulate replay")

    monkeypatch.setattr("trace_collect.simulator.simulate", fake_simulate)
    args = parse_simulate_args(["--manifest", "manifest.yaml"])

    with pytest.raises(SystemExit) as exc_info:
        _run_simulate(args)

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert (
        "ERROR: --pmu-monitoring on is forbidden for concurrent simulate replay"
        in captured.err
    )
    assert "Traceback" not in captured.err


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


def test_cloud_model_ttft_tpot_llm_timing_records_simulated_latency(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_trace(
        trace_path,
        agent_id="host-task",
        scaffold="tongyi-deepresearch",
        llm_start=100.0,
        llm_end=100.2,
        tool_start=100.4,
        tool_end=100.45,
        execution_environment="host",
    )
    _write_host_tasks(task_source, "host-task")

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=output_dir,
            mode="cloud_model",
            replay_speed=1.0,
            llm_timing_mode="ttft_tpot",
            llm_ttft_ms=10.0,
            llm_tpot_ms=2.0,
        )
    )

    records = _read_jsonl(trace_file)
    metadata = records[0]
    llm_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    )
    summary = next(record for record in records if record.get("type") == "summary")

    assert metadata["llm_timing_mode"] == "ttft_tpot"
    assert metadata["llm_ttft_ms"] == 10.0
    assert metadata["llm_tpot_ms"] == 2.0
    assert llm_record["data"]["llm_timing_mode"] == "ttft_tpot"
    assert llm_record["data"]["simulated_ttft_ms"] == 10.0
    assert llm_record["data"]["simulated_tpot_ms"] == 2.0
    assert llm_record["data"]["simulated_llm_latency_ms"] == 18.0
    assert llm_record["data"]["source_llm_latency_ms"] == pytest.approx(200.0)
    assert llm_record["data"]["llm_latency_ms"] == pytest.approx(18.0, abs=25.0)
    assert summary["llm_timing_mode"] == "ttft_tpot"


def test_simulate_preserves_source_resource_timeline_as_metadata(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    resource_timeline = {
        "version": RESOURCE_TIMELINE_SCHEMA_VERSION,
        "source": "cgroup_cpu_proc_net",
        "scope": "openclaw_exec_tool_interval",
        "samples": [
            {
                "offset_s": 0.5,
                "dt_s": 0.5,
                "cpu_core_s": 1.0,
                "net_rx_bytes": 128,
                "net_tx_bytes": 64,
            }
        ],
        "summary": {
            "sample_count": 1,
            "wall_s": 0.5,
            "cpu_core_s": 1.0,
            "net_rx_bytes": 128,
            "net_tx_bytes": 64,
        },
    }
    _write_trace(
        trace_path,
        agent_id="host-task",
        scaffold="tongyi-deepresearch",
        tool_name="exec",
        execution_environment="host",
        resource_timeline=resource_timeline,
        tool_args={"command": "pytest"},
    )
    _write_host_tasks(task_source, "host-task")

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            replay_speed=100.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    assert tool_record["data"]["source_resource_timeline"] == resource_timeline
    assert tool_record["data"]["resource_timeout_policy"] == "wall_clock"


def test_simulate_uses_resource_integrated_policy_for_container_exec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    resource_timeline = {
        "version": RESOURCE_TIMELINE_SCHEMA_VERSION,
        "source": "cgroup_cpu_proc_net",
        "scope": "openclaw_exec_tool_interval",
        "samples": [{"offset_s": 0.5, "dt_s": 0.5, "cpu_core_s": 1.0}],
    }
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="exec",
        resource_timeline=resource_timeline,
        tool_args={"command": "pytest"},
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)
    captured_timelines: list[dict | None] = []

    async def fake_exec_tool(
        _agent,
        _tool_name,
        _tool_args_json,
        _command_timeout_s,
        _source_exec_timeout_s=None,
        _allow_source_runtime_artifacts=False,
        source_resource_timeline=None,
    ):
        captured_timelines.append(source_resource_timeline)
        return (
            "ok\n\nExit code: 0",
            1.0,
            True,
            {"resource_virtual_time_s": 0.5},
        )

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    assert captured_timelines == [resource_timeline]
    assert tool_record["data"]["resource_timeout_policy"] == "resource_integrated"
    assert tool_record["data"]["resource_virtual_time_s"] == 0.5


def test_simulate_keeps_wall_policy_for_commands_resource_timeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    resource_timeline = {
        "version": RESOURCE_TIMELINE_SCHEMA_VERSION,
        "samples": [{"offset_s": 0.5, "dt_s": 0.5, "cpu_core_s": 1.0}],
    }
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="exec",
        resource_timeline=resource_timeline,
        tool_args={"commands": ["pytest"]},
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)
    captured_timelines: list[dict | None] = []

    async def fake_exec_tool(
        _agent,
        _tool_name,
        _tool_args_json,
        _command_timeout_s,
        _source_exec_timeout_s=None,
        _allow_source_runtime_artifacts=False,
        source_resource_timeline=None,
    ):
        captured_timelines.append(source_resource_timeline)
        return "ok\n\nExit code: 0", 1.0, True

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    tool_record = next(
        record
        for record in _read_jsonl(trace_file)
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    assert captured_timelines == [None]
    assert tool_record["data"]["resource_timeout_policy"] == "wall_clock"


def test_simulate_ignores_invalid_resource_timeline_for_timeout_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    invalid_timeline = {
        "version": RESOURCE_TIMELINE_SCHEMA_VERSION,
        "samples": [{"dt_s": 0.0, "cpu_core_s": 1.0}],
    }
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="exec",
        resource_timeline=invalid_timeline,
        tool_args={"command": "pytest"},
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    async def fake_exec_tool(*_args, **_kwargs):
        return "ok\n\nExit code: 0", 1.0, True

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    tool_record = next(
        record
        for record in _read_jsonl(trace_file)
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    assert "source_resource_timeline" not in tool_record["data"]
    assert "resource_timeout_policy" not in tool_record["data"]


def test_parse_trace_session_file_includes_subagent_actions(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "type": "trace_metadata",
                    "scaffold": "tongyi-deepresearch",
                    "instance_id": "task-a",
                    "execution_environment": "container",
                },
                {
                    "type": "action",
                    "action_type": "llm_call",
                    "action_id": "primary-llm",
                    "agent_id": "task-a",
                    "iteration": 0,
                    "ts_start": 100.2,
                    "ts_end": 100.3,
                    "data": {"messages_in": [], "raw_response": {}},
                },
                {
                    "type": "action",
                    "action_type": "tool_exec",
                    "action_id": "child-tool",
                    "agent_id": "task-a:subagent:child",
                    "iteration": 0,
                    "ts_start": 100.0,
                    "ts_end": 100.1,
                    "data": {
                        "tool_name": "list_dir",
                        "tool_args": '{"path":"/testbed"}',
                        "tool_result": "child-result",
                        "duration_ms": 100.0,
                        "success": True,
                    },
                },
                {
                    "type": "summary",
                    "agent_id": "task-a",
                    "success": True,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    task_instance_id, source_agent_id, _metadata, actions, _summary = (
        _parse_trace_session_file(trace_path)
    )

    assert task_instance_id == "task-a"
    assert source_agent_id == "task-a"
    assert [action["action_id"] for action in actions] == [
        "child-tool",
        "primary-llm",
    ]


def test_parse_trace_session_file_excludes_other_top_level_agents(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "type": "trace_metadata",
                    "scaffold": "tongyi-deepresearch",
                    "instance_id": "task-a",
                    "execution_environment": "container",
                },
                {
                    "type": "action",
                    "action_type": "tool_exec",
                    "action_id": "task-a-tool",
                    "agent_id": "task-a",
                    "iteration": 0,
                    "ts_start": 100.0,
                    "ts_end": 100.1,
                    "data": {"tool_name": "list_dir", "success": True},
                },
                {
                    "type": "action",
                    "action_type": "tool_exec",
                    "action_id": "task-a-child-tool",
                    "agent_id": "task-a:subagent:child",
                    "iteration": 0,
                    "ts_start": 100.2,
                    "ts_end": 100.3,
                    "data": {"tool_name": "read_file", "success": True},
                },
                {
                    "type": "action",
                    "action_type": "tool_exec",
                    "action_id": "task-b-tool",
                    "agent_id": "task-b",
                    "iteration": 0,
                    "ts_start": 100.4,
                    "ts_end": 100.5,
                    "data": {"tool_name": "exec", "success": True},
                },
                {
                    "type": "summary",
                    "agent_id": "task-a",
                    "success": True,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    task_instance_id, source_agent_id, _metadata, actions, _summary = (
        _parse_trace_session_file(trace_path)
    )

    assert task_instance_id == "task-a"
    assert source_agent_id == "task-a"
    assert [action["action_id"] for action in actions] == [
        "task-a-tool",
        "task-a-child-tool",
    ]


def test_simulate_replays_spawn_and_sessions_yield_as_control_noops(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    records = [
        {
            "type": "trace_metadata",
            "trace_format_version": 5,
            "scaffold": "tongyi-deepresearch",
            "instance_id": "task-a",
            "model": "claude-haiku",
            "mode": "collect",
            "execution_environment": "container",
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "spawn-tool",
            "agent_id": "task-a",
            "iteration": 0,
            "ts_start": 100.0,
            "ts_end": 100.1,
            "data": {
                "tool_name": "spawn",
                "tool_args": '{"task":"inspect"}',
                "tool_result": "",
                "duration_ms": 100.0,
                "success": True,
            },
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "yield-tool",
            "agent_id": "task-a",
            "iteration": 0,
            "ts_start": 100.2,
            "ts_end": 100.3,
            "data": {
                "tool_name": "sessions_yield",
                "tool_args": "{}",
                "tool_result": "",
                "duration_ms": 100.0,
                "success": True,
            },
        },
        {
            "type": "summary",
            "agent_id": "task-a",
            "model": "claude-haiku",
            "success": True,
            "n_iterations": 1,
            "elapsed_s": 0.3,
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    tool_records = [
        record
        for record in _read_jsonl(trace_file)
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    ]
    by_name = {record["data"]["tool_name"]: record for record in tool_records}

    assert by_name["spawn"]["data"]["replay_source"] == "control_noop"
    assert by_name["spawn"]["data"]["tool_result"] == "Subagent spawn replayed as no-op"
    assert by_name["sessions_yield"]["data"]["replay_source"] == "control_noop"
    assert (
        by_name["sessions_yield"]["data"]["tool_result"]
        == "Session yield replayed as no-op"
    )


def test_simulate_replays_overlapping_tools_concurrently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    trace_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "type": "trace_metadata",
                    "trace_format_version": 5,
                    "scaffold": "tongyi-deepresearch",
                    "instance_id": "task-a",
                    "model": "claude-haiku",
                    "mode": "collect",
                    "execution_environment": "container",
                },
                {
                    "type": "action",
                    "action_type": "tool_exec",
                    "action_id": "tool-a",
                    "agent_id": "task-a",
                    "iteration": 0,
                    "ts_start": 100.0,
                    "ts_end": 100.5,
                    "data": {
                        "tool_name": "list_dir",
                        "tool_args": '{"path":"/testbed"}',
                        "tool_result": "source-a",
                        "duration_ms": 500.0,
                        "success": True,
                    },
                },
                {
                    "type": "action",
                    "action_type": "tool_exec",
                    "action_id": "tool-b",
                    "agent_id": "task-a:subagent:child",
                    "iteration": 0,
                    "ts_start": 100.0,
                    "ts_end": 100.5,
                    "data": {
                        "tool_name": "read_file",
                        "tool_args": '{"path":"/testbed/file.py"}',
                        "tool_result": "source-b",
                        "duration_ms": 500.0,
                        "success": True,
                    },
                },
                {
                    "type": "summary",
                    "agent_id": "task-a",
                    "model": "claude-haiku",
                    "success": True,
                    "n_iterations": 1,
                    "elapsed_s": 0.5,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    extra_agent_starts = 0
    extra_agent_stops = 0

    class _ExtraAgent:
        def __init__(
            self,
            container_id: str,
            container_executable: str,
            **_kwargs,
        ) -> None:
            self.container_id = container_id
            self.container_executable = container_executable

        async def start(self) -> None:
            nonlocal extra_agent_starts
            extra_agent_starts += 1
            await asyncio.sleep(0.05)

        async def stop(self) -> None:
            nonlocal extra_agent_stops
            extra_agent_stops += 1

    exec_starts: list[tuple[str, float, int]] = []

    async def fake_exec_tool(
        agent,
        tool_name,
        tool_args_json,
        command_timeout_s,
        source_exec_timeout_s=None,
        allow_source_runtime_artifacts=False,
        source_resource_timeline=None,
    ):
        del (
            tool_args_json,
            command_timeout_s,
            source_exec_timeout_s,
            allow_source_runtime_artifacts,
            source_resource_timeline,
        )
        exec_starts.append((tool_name, time.perf_counter(), id(agent)))
        await asyncio.sleep(0.05)
        return f"executed-{tool_name}", 50.0, True

    monkeypatch.setattr("trace_collect.openclaw_tools.ContainerAgent", _ExtraAgent)
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    assert len(exec_starts) == 2
    assert len({agent_id for _tool, _started, agent_id in exec_starts}) == 2
    assert max(started for _tool, started, _agent in exec_starts) - min(
        started for _tool, started, _agent in exec_starts
    ) < 0.04
    assert extra_agent_starts == 1
    assert extra_agent_stops == 1

    tool_records = [
        record
        for record in _read_jsonl(trace_file)
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    ]
    assert len(tool_records) == 2
    assert {record["data"]["replay_source"] for record in tool_records} == {
        "executed_in_container"
    }
    provenance = {
        record["data"]["tool_name"]: record["data"]["source_action_agent_id"]
        for record in tool_records
    }
    assert provenance == {
        "list_dir": "task-a",
        "read_file": "task-a:subagent:child",
    }
    replay_agent_ids = {
        record["data"]["tool_name"]: record["agent_id"] for record in tool_records
    }
    assert replay_agent_ids == {
        "list_dir": "task-a",
        "read_file": "task-a:subagent:child",
    }


def test_cloud_model_ttft_tpot_requires_parameters(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="host-task",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_host_tasks(task_source, "host-task")

    with pytest.raises(ValueError, match="llm_ttft_ms is required"):
        asyncio.run(
            simulate(
                manifest=_single_trace_manifest(tmp_path, trace_path),
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="cloud_model",
                llm_timing_mode="ttft_tpot",
                llm_tpot_ms=2.0,
            )
        )


def test_cloud_model_ttft_tpot_rejects_replay_speed_acceleration(
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

    with pytest.raises(ValueError, match="exclusive with llm_timing_mode='ttft_tpot'"):
        asyncio.run(
            simulate(
                manifest=_single_trace_manifest(tmp_path, trace_path),
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="cloud_model",
                replay_speed=2.0,
                llm_timing_mode="ttft_tpot",
                llm_ttft_ms=10.0,
                llm_tpot_ms=2.0,
            )
        )


def test_cloud_model_source_scaled_rejects_ttft_tpot_parameters(
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

    with pytest.raises(ValueError, match="require llm_timing_mode='ttft_tpot'"):
        asyncio.run(
            simulate(
                manifest=_single_trace_manifest(tmp_path, trace_path),
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="cloud_model",
                llm_timing_mode="source_scaled",
                llm_ttft_ms=10.0,
            )
        )


def test_cloud_model_tool_success_false_fails_trace_when_source_succeeded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    async def fake_exec_tool(*_args, **_kwargs):
        return "Error: Unsupported replay tool 'bad_tool'", 1.0, False

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=output_dir,
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    summary = next(record for record in records if record.get("type") == "summary")
    throughput = json.loads((output_dir / "throughput_summary.json").read_text())

    assert tool_record["data"]["success"] is False
    assert summary["success"] is False
    assert summary["failed_actions"] == 1
    assert summary["replay_failed_actions"] == 1
    assert summary["unexpected_replay_failed_actions"] == 1
    assert summary["replay_execution_errors"] == 0
    assert throughput["completed_traces"] == 0
    assert throughput["failed_traces"] == 1
    assert throughput["tasks"][0]["success"] is False
    assert throughput["tasks"][0]["failed_action_count"] == 1


def test_cloud_model_source_failed_tool_match_does_not_fail_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_trace(trace_path, agent_id="task-a", tool_name="read_file")
    records = _read_jsonl(trace_path)
    for record in records:
        if record.get("action_type") == "tool_exec":
            record["data"]["success"] = False
            record["data"]["tool_result"] = "Error: Not a file: /testbed/pkg"
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    async def fake_exec_tool(*_args, **_kwargs):
        return "Error: [Errno 21] Is a directory: '/testbed/pkg'", 1.0, False

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=output_dir,
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    summary = next(record for record in records if record.get("type") == "summary")
    throughput = json.loads((output_dir / "throughput_summary.json").read_text())

    assert tool_record["data"]["success"] is False
    assert tool_record["data"]["source_success"] is False
    assert summary["success"] is True
    assert summary["failed_actions"] == 0
    assert summary["source_failed_actions"] == 1
    assert summary["replay_failed_actions"] == 1
    assert summary["unexpected_replay_failed_actions"] == 0
    assert summary["replay_execution_errors"] == 0
    assert throughput["completed_traces"] == 1
    assert throughput["failed_traces"] == 0


def test_source_exec_timeout_detects_source_timeout_failure() -> None:
    timeout_s = _source_exec_timeout_s(
        tool_name="exec",
        tool_args_json=json.dumps({"exec": {"command": "slow command"}}),
        source_duration_ms=300056.8,
        source_success=False,
        source_tool_result="Error: Command timed out after 300 seconds",
    )

    assert timeout_s == pytest.approx(300.0568)
    assert _source_exec_timeout_s(
        tool_name="exec",
        tool_args_json=json.dumps({"exec": {"command": "timeout 1 sleep 2"}}),
        source_duration_ms=1000.0,
        source_success=False,
        source_tool_result="shell returned timeout status\n\nExit code: 124",
    ) is None
    assert _source_exec_timeout_s(
        tool_name="exec",
        tool_args_json=json.dumps({"exec": {"command": "printf '[timeout]' && false"}}),
        source_duration_ms=50.0,
        source_success=False,
        source_tool_result="app printed [timeout]\n\nExit code: 1",
    ) is None
    assert _source_exec_timeout_s(
        tool_name="exec",
        tool_args_json=json.dumps({"exec": {"command": "slow command"}}),
        source_duration_ms=300056.8,
        source_success=False,
        source_tool_result="[timeout]\n\nExit code: 124",
    ) == pytest.approx(300.0568)
    assert _source_exec_timeout_s(
        tool_name="exec",
        tool_args_json=json.dumps({"exec": {"command": "false"}}),
        source_duration_ms=50.0,
        source_success=False,
        source_tool_result="failed\n\nExit code: 1",
    ) is None
    assert _source_exec_timeout_s(
        tool_name="read_file",
        tool_args_json=json.dumps({"path": "/testbed/file.txt"}),
        source_duration_ms=300056.8,
        source_success=False,
        source_tool_result="[timeout]\n\nExit code: 124",
    ) is None


def test_cloud_model_preserves_source_exec_timeout_for_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_trace(trace_path, agent_id="task-a", tool_name="exec")
    records = _read_jsonl(trace_path)
    for record in records:
        if record.get("action_type") == "tool_exec":
            record["data"]["tool_args"] = json.dumps(
                {"exec": {"command": "cd /testbed && slow command"}}
            )
            record["data"]["duration_ms"] = 300056.8
            record["data"]["success"] = False
            record["data"]["tool_result"] = "Error: Command timed out after 300 seconds"
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    captured_source_timeouts: list[float | None] = []

    async def fake_exec_tool(
        _agent,
        _tool_name,
        _tool_args_json,
        _command_timeout_s,
        source_exec_timeout_s=None,
        allow_source_runtime_artifacts=False,
    ):
        assert allow_source_runtime_artifacts is False
        captured_source_timeouts.append(source_exec_timeout_s)
        return "[timeout]\n\nExit code: 124", 300056.8, False

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=output_dir,
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
            command_timeout_s=600.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    summary = next(record for record in records if record.get("type") == "summary")

    assert captured_source_timeouts == [pytest.approx(300.0568)]
    assert tool_record["data"]["source_exec_timeout_s"] == pytest.approx(300.0568)
    assert tool_record["data"]["source_success"] is False
    assert tool_record["data"]["success"] is False
    assert summary["success"] is True
    assert summary["failed_actions"] == 0
    assert summary["unexpected_replay_failed_actions"] == 0
    assert summary["replay_execution_errors"] == 0


def test_cloud_model_source_runtime_artifact_path_fails_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_trace(trace_path, agent_id="task-a", tool_name="read_file")
    records = _read_jsonl(trace_path)
    artifact_path = (
        "/root/agent-sched-bench/traces/x/attempt_1/"
        "openclaw-runtime/tool-results/tool-results/cli_task/out.txt"
    )
    for record in records:
        if record.get("action_type") == "tool_exec":
            record["data"]["tool_args"] = json.dumps({"path": artifact_path})
            record["data"]["success"] = True
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    async def fail_exec_tool(*_args, **_kwargs):
        raise AssertionError("runtime artifact path must not execute in container")

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fail_exec_tool)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=output_dir,
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    summary = next(record for record in records if record.get("type") == "summary")
    throughput = json.loads((output_dir / "throughput_summary.json").read_text())

    assert tool_record["data"]["success"] is False
    assert tool_record["data"]["source_success"] is True
    assert tool_record["data"]["replay_source"] == "source_artifact_unavailable"
    assert tool_record["data"]["sim_metrics"]["sim_tool_format"] == "source_artifact_unavailable"
    assert artifact_path in tool_record["data"]["tool_result"]
    assert summary["success"] is False
    assert summary["failed_actions"] == 1
    assert summary["source_failed_actions"] == 0
    assert summary["replay_failed_actions"] == 1
    assert summary["fatal_replay_errors"] == 1
    assert throughput["completed_traces"] == 0
    assert throughput["failed_traces"] == 1


def test_cloud_model_missing_source_runtime_artifact_is_fatal_for_source_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_trace(trace_path, agent_id="task-a", tool_name="read_file")
    records = _read_jsonl(trace_path)
    artifact_path = (
        "/root/agent-sched-bench/traces/x/attempt_1/"
        "openclaw-runtime/tool-results/tool-results/cli_task/out.txt"
    )
    for record in records:
        if record.get("action_type") == "tool_exec":
            record["data"]["tool_args"] = json.dumps({"path": artifact_path})
            record["data"]["success"] = False
            record["data"]["tool_result"] = "Error: missing artifact"
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=output_dir,
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    summary = next(record for record in records if record.get("type") == "summary")

    assert tool_record["data"]["success"] is False
    assert tool_record["data"]["source_success"] is False
    assert tool_record["data"]["replay_source"] == "source_artifact_unavailable"
    assert summary["success"] is False
    assert summary["failed_actions"] == 1
    assert summary["fatal_replay_errors"] == 1


def test_cloud_model_missing_specific_runtime_artifact_file_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_attempt = tmp_path / "source" / "task-a" / "attempt_1"
    source_attempt.mkdir(parents=True)
    trace_path = source_attempt / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_trace(trace_path, agent_id="task-a", tool_name="read_file")
    artifact_dir = source_attempt / "openclaw-runtime" / "tool-results"
    artifact_dir.mkdir(parents=True)
    (source_attempt / "run_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "openclaw_tool_results_dir": "openclaw-runtime/tool-results",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_artifact_path = (
        "/root/agent-sched-bench/traces/source/task-a/attempt_1/"
        "openclaw-runtime/tool-results/tool-results/cli_task/missing.txt"
    )
    records = _read_jsonl(trace_path)
    for record in records:
        if record.get("action_type") == "tool_exec":
            record["data"]["tool_args"] = json.dumps({"path": source_artifact_path})
            record["data"]["success"] = False
            record["data"]["tool_result"] = "Error: missing source spill file"
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    monkeypatch.setattr(
        "trace_collect.simulator._copy_source_runtime_artifacts_to_container",
        lambda **_kwargs: None,
    )

    async def fail_exec_tool(*_args, **_kwargs):
        raise AssertionError("missing specific artifact file must not execute")

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fail_exec_tool)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=output_dir,
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    summary = next(record for record in records if record.get("type") == "summary")

    assert tool_record["data"]["success"] is False
    assert tool_record["data"]["source_success"] is False
    assert tool_record["data"]["replay_source"] == "source_artifact_unavailable"
    assert summary["success"] is False
    assert summary["failed_actions"] == 1
    assert summary["fatal_replay_errors"] == 1


def test_cloud_model_restores_source_runtime_artifact_into_simulator_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_attempt = tmp_path / "source" / "task-a" / "attempt_1"
    source_attempt.mkdir(parents=True)
    trace_path = source_attempt / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_trace(trace_path, agent_id="task-a", tool_name="read_file")
    artifact_dir = source_attempt / "openclaw-runtime" / "tool-results"
    artifact_file = artifact_dir / "tool-results" / "cli_task" / "out.txt"
    artifact_file.parent.mkdir(parents=True)
    artifact_file.write_text("full saved output", encoding="utf-8")
    (source_attempt / "run_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "openclaw_tool_results_dir": "openclaw-runtime/tool-results",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_artifact_path = (
        "/root/agent-sched-bench/traces/source/task-a/attempt_1/"
        "openclaw-runtime/tool-results/tool-results/cli_task/out.txt"
    )
    records = _read_jsonl(trace_path)
    for record in records:
        if record.get("action_type") == "tool_exec":
            record["data"]["tool_args"] = json.dumps({"path": source_artifact_path})
            record["data"]["success"] = True
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    copied: list[tuple[Path, str]] = []

    def fake_copy_source_runtime_artifacts_to_container(
        *,
        source_dir: Path,
        container_id: str,
        container_executable: str,
        destination_dir: str,
    ) -> None:
        assert container_id == "fake-cid"
        assert container_executable == "docker"
        copied.append((source_dir, destination_dir))

    async def fake_exec_tool(
        _agent,
        tool_name,
        tool_args_json,
        _command_timeout_s,
        _source_exec_timeout_s=None,
        allow_source_runtime_artifacts=False,
    ):
        args = json.loads(tool_args_json)
        assert tool_name == "read_file"
        assert allow_source_runtime_artifacts is True
        assert args["path"].endswith(
            "openclaw-runtime/tool-results/tool-results/cli_task/out.txt"
        )
        assert str(output_dir.resolve()) in args["path"]
        return "full saved output", 1.0, True

    monkeypatch.setattr(
        "trace_collect.simulator._copy_source_runtime_artifacts_to_container",
        fake_copy_source_runtime_artifacts_to_container,
    )
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=output_dir,
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    simulator_artifact = (
        output_dir
        / "task-a"
        / "attempt_1"
        / "openclaw-runtime"
        / "tool-results"
        / "tool-results"
        / "cli_task"
        / "out.txt"
    )
    assert simulator_artifact.read_text(encoding="utf-8") == "full saved output"
    assert copied == [
        (
            output_dir / "task-a" / "attempt_1" / "openclaw-runtime" / "tool-results",
            str((output_dir / "task-a" / "attempt_1" / "openclaw-runtime" / "tool-results").resolve()),
        )
    ]
    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    summary = next(record for record in records if record.get("type") == "summary")

    assert tool_record["data"]["replay_source"] == "restored_runtime_artifact"
    assert tool_record["data"]["source_artifact_path"] == source_artifact_path
    assert tool_record["data"]["simulator_artifact_path"] == str(simulator_artifact.resolve())
    assert summary["success"] is True
    assert summary["fatal_replay_errors"] == 0


def test_cloud_model_message_tool_replays_as_noop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_trace(trace_path, agent_id="task-a", tool_name="message")
    records = _read_jsonl(trace_path)
    for record in records:
        if record.get("action_type") == "tool_exec":
            record["data"]["tool_args"] = json.dumps({"content": "finished"})
            record["data"]["tool_result"] = "Message sent to cli:task-a"
            record["data"]["success"] = True
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    async def fail_exec_tool(*_args, **_kwargs):
        raise AssertionError("message replay must not execute in container")

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fail_exec_tool)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=output_dir,
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    summary = next(record for record in records if record.get("type") == "summary")

    assert tool_record["data"]["success"] is True
    assert tool_record["data"]["source_success"] is True
    assert tool_record["data"]["replay_source"] == "message_noop"
    assert tool_record["data"]["sim_metrics"]["sim_tool_format"] == "message_noop"
    assert summary["success"] is True


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
    assert payload["summary"]["monitoring_disabled"] is True
    assert payload["summary"]["monitoring"]["status"] == "disabled"

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
    _write_trace(trace_container, agent_id="container-task", scaffold="openclaw")
    _write_trace(
        trace_host,
        agent_id="host-task",
        scaffold="tongyi-deepresearch",
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
    fixed_images: list[str] = []
    removed_images: list[str] = []

    def fake_ensure_source_image(image, *, container_executable):
        events.append(("prefetch", image))

    def fake_ensure_fixed_image(
        source_image: str,
        *,
        container_executable: str,
        fixed_image_name: str,
        rebuild: bool,
        **_kwargs,
    ) -> tuple[str, float]:
        assert source_image == "docker.io/shared/image:latest"
        assert container_executable == "docker"
        assert rebuild is True
        fixed_images.append(fixed_image_name)
        return (fixed_image_name, 0.1)

    def fake_start_task_container(
        image: str,
        *,
        executable: str,
        network_mode: str,
        run_as_host_user: bool,
        mount_host_home: bool,
        container_home: str,
        extra_args: list[str] | None = None,
    ) -> str:
        assert executable == "docker"
        assert network_mode == "host"
        assert run_as_host_user is False
        assert mount_host_home is False
        assert container_home == "/root"
        assert extra_args is not None
        assert "agent-sched-bench.component=simulate-replay" in extra_args
        events.append(("prepare", image))
        return f"fake-{len(fixed_images)}"

    def fake_remove_image(image: str, *, container_executable: str) -> bool:
        assert container_executable == "docker"
        removed_images.append(image)
        return True

    class _FakeAgent:
        def __init__(
            self,
            container_id: str,
            container_executable: str,
            **_kwargs,
        ) -> None:
            self.container_id = container_id
            self.container_executable = container_executable

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

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
    monkeypatch.setattr("trace_collect.simulator.ensure_fixed_image", fake_ensure_fixed_image)
    monkeypatch.setattr("trace_collect.simulator.start_task_container", fake_start_task_container)
    monkeypatch.setattr("trace_collect.simulator.stop_task_container", lambda *args, **kwargs: "")
    monkeypatch.setattr("trace_collect.simulator.remove_image", fake_remove_image)
    monkeypatch.setattr("trace_collect.openclaw_tools.ContainerAgent", _FakeAgent)
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)

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
    assert len(fixed_images) == 1
    assert fixed_images[0].startswith(
        "swebench-fixed-docker.io_shared_image_latest:simulate-sweep-"
    )
    assert [event for event in events if event[0] == "prepare"] == [
        ("prepare", fixed_images[0]),
        ("prepare", fixed_images[0]),
    ]
    assert removed_images == fixed_images


def test_throughput_wall_time_excludes_sweep_fixed_image_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")

    clock = {"value": 0.0}

    def fake_monotonic() -> float:
        return clock["value"]

    async def fake_prebuild(*_args, **_kwargs) -> dict[str, str]:
        clock["value"] += 100.0
        return {"docker.io/swebench-test/task-a": "fixed-image"}

    async def fake_queue(*_args, **_kwargs):
        from trace_collect.simulator import ReplayTaskStats

        clock["value"] += 10.0
        return [], [
            ReplayTaskStats(
                agent_id="task-a",
                run_instance_id="task-a",
                source_agent_id="task-a",
                manifest_index=0,
                label=None,
                source_trace=str(trace_path),
                success=True,
                elapsed_s=10.0,
                action_count=2,
                llm_call_count=1,
                tool_exec_count=1,
            )
        ], None

    async def fake_cleanup(*_args, **_kwargs) -> None:
        clock["value"] += 100.0

    async def fake_prefetch(*_args, **_kwargs) -> None:
        pass

    monkeypatch.setattr("trace_collect.simulator.time.monotonic", fake_monotonic)
    monkeypatch.setattr("trace_collect.simulator._prefetch_container_images", fake_prefetch)
    monkeypatch.setattr("trace_collect.simulator._prebuild_sweep_fixed_images", fake_prebuild)
    monkeypatch.setattr("trace_collect.simulator._run_cloud_model_queue", fake_queue)
    monkeypatch.setattr("trace_collect.simulator._cleanup_sweep_fixed_images", fake_cleanup)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
        )
    )

    assert trace_file.exists()
    summary = json.loads((tmp_path / "out" / "throughput_summary.json").read_text())
    assert summary["wall_time_s"] == pytest.approx(10.0)
    assert summary["traces_per_s"] == pytest.approx(0.1)
    assert clock["value"] == pytest.approx(210.0)


def test_cloud_model_prebuild_failure_keeps_prebuilt_sweep_fixed_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "trace-a.jsonl"
    trace_b = tmp_path / "trace-b.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    _write_trace(trace_a, agent_id="task-a")
    _write_trace(trace_b, agent_id="task-b")
    _write_tasks(task_source, "task-a", "task-b")
    _write_manifest(manifest, [str(trace_a), str(trace_b)])

    fixed_images: list[str] = []
    removed_images: list[str] = []

    async def fake_prefetch(*_args, **_kwargs) -> None:
        pass

    def fake_ensure_fixed_image(
        source_image: str,
        *,
        container_executable: str,
        fixed_image_name: str,
        rebuild: bool,
        **_kwargs,
    ) -> tuple[str, float]:
        assert container_executable == "docker"
        assert rebuild is True
        fixed_images.append(fixed_image_name)
        if source_image == "docker.io/swebench-test/task-b":
            raise RuntimeError("prebuild failed")
        return (fixed_image_name, 0.1)

    monkeypatch.setattr("trace_collect.simulator._prefetch_container_images", fake_prefetch)
    monkeypatch.setattr("trace_collect.simulator.ensure_fixed_image", fake_ensure_fixed_image)
    monkeypatch.setattr(
        "trace_collect.simulator.remove_image",
        lambda image, *, container_executable: removed_images.append(image) or True,
    )

    with pytest.raises(RuntimeError, match="prebuild failed"):
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

    assert len(fixed_images) == 2
    assert all(":simulate-sweep-" in image for image in fixed_images)
    assert removed_images == []


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
    _patch_noop_sweep_fixed_prebuild(monkeypatch)
    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", fake_prepare_container)
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)

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
        def __init__(
            self,
            container_id: str,
            container_executable: str,
            **_kwargs,
        ) -> None:
            assert container_id == "fake-cid"
            assert container_executable == "docker"

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    class _FakeSampler:
        def __init__(
            self,
            *,
            container_id: str,
            interval_s: float,
            executable: str,
            enable_memory_bandwidth: bool = True,
        ) -> None:
            assert container_id == "fake-cid"
            assert executable == "docker"
            assert enable_memory_bandwidth is True
            self.interval_s = interval_s
            kind = "startup" if self.interval_s == 0.25 else "runtime"
            self._samples = [
                {
                    "timestamp": "2026-06-26T00:00:00Z",
                    "epoch": 1782470400.0,
                    "container_id": "fake-cid",
                    "phase": kind,
                    "cpu_percent": "1.00%",
                    "mem_usage": "2MiB / 1GiB",
                },
                {
                    "timestamp": "2026-06-26T00:00:01Z",
                    "epoch": 1782470401.0,
                    "container_id": "fake-cid",
                    "phase": kind,
                    "cpu_percent": "2.00%",
                    "mem_usage": "3MiB / 1GiB",
                },
            ]

        def start(self) -> None:
            pass

        def stop(self) -> list[dict]:
            return self._samples[:1]

    async def fake_exec_tool(*_args, **_kwargs):
        return ("ok", 1.0, True)

    ensure_calls: list[dict[str, object]] = []
    removed_images: list[str] = []

    def fake_ensure_fixed_image(
        source_image: str,
        *,
        container_executable: str,
        fixed_image_name: str,
        rebuild: bool,
        **_kwargs,
    ) -> tuple[str, float]:
        ensure_calls.append(
            {
                "source_image": source_image,
                "container_executable": container_executable,
                "fixed_image_name": fixed_image_name,
                "rebuild": rebuild,
            }
        )
        return (fixed_image_name, 0.125)

    def fake_remove_image(image: str, *, container_executable: str) -> bool:
        assert container_executable == "docker"
        removed_images.append(image)
        return True

    monkeypatch.setattr("trace_collect.simulator.ensure_source_image", lambda *args, **kwargs: None)
    monkeypatch.setattr("trace_collect.simulator.ensure_fixed_image", fake_ensure_fixed_image)
    monkeypatch.setattr("trace_collect.simulator.remove_image", fake_remove_image)
    monkeypatch.setattr(
        "trace_collect.simulator.start_task_container",
        lambda *args, **kwargs: "fake-cid",
    )
    monkeypatch.setattr("trace_collect.simulator.stop_task_container", lambda *args, **kwargs: "")
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
    monkeypatch.setattr("trace_collect.openclaw_tools.ContainerAgent", _FakeContainerAgent)
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)

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
    assert startup["fixed_image"].startswith(
        "swebench-fixed-docker.io_swebench-test_task-a:simulate-sweep-"
    )
    assert startup["container_id"] == "fake-cid"
    assert [phase["name"] for phase in startup["phases"]] == [
        "ensure_fixed_image",
        "start_task_container",
        "configure_apt_mirror",
        "container_agent_start",
    ]
    assert startup["phases"][0]["prebuilt"] is True
    assert startup["phases"][0]["reported_elapsed_s"] == pytest.approx(0.0)
    assert startup["phases"][2]["status"] == "skipped"
    assert startup["phases"][2]["reason"] == "TASK_CONTAINER_APT_MIRROR unset"
    assert startup["resources"]["samples"] == []
    assert startup["resources"]["summary"]["sample_count"] == 0
    assert resources["samples"][0]["phase"] == "runtime"
    assert resources["summary"]["sample_count"] == 1
    assert resources["summary"]["monitoring_disabled"] is False
    assert resources["summary"]["monitoring"]["status"] == "collected"
    assert ensure_calls == [
        {
            "source_image": "docker.io/swebench-test/task-a",
            "container_executable": "docker",
            "fixed_image_name": startup["fixed_image"],
            "rebuild": True,
        }
    ]
    assert removed_images == [startup["fixed_image"]]


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
        def __init__(
            self,
            container_id: str,
            container_executable: str,
            **_kwargs,
        ) -> None:
            assert container_id == "fake-cid"
            assert container_executable == "docker"

        async def start(self) -> None:
            raise RuntimeError("agent failed")

        async def stop(self) -> None:
            raise AssertionError("failed startup agent must not be finalized later")

    def fake_stop_task_container(container_id: str, *, executable: str) -> str:
        assert executable == "docker"
        stopped_containers.append(container_id)
        return ""

    removed_images: list[str] = []

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
    monkeypatch.setattr(
        "trace_collect.simulator.remove_image",
        lambda image, *, container_executable: removed_images.append(image) or True,
    )
    monkeypatch.setattr("trace_collect.openclaw_tools.ContainerAgent", _FailingContainerAgent)

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
    assert startup["resources"]["samples"] == []
    assert startup["resources"]["summary"]["sample_count"] == 0
    assert stopped_containers == ["fake-cid"]
    assert removed_images == ["fixed-image"]


def test_cloud_model_start_container_failure_cleans_sweep_fixed_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    removed_images: list[str] = []

    def fail_start_task_container(*_args, **_kwargs) -> str:
        raise RuntimeError("container start failed")

    monkeypatch.setattr("trace_collect.simulator.ensure_source_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "trace_collect.simulator.ensure_fixed_image",
        lambda *args, **kwargs: ("fixed-image", 0.125),
    )
    monkeypatch.setattr(
        "trace_collect.simulator.start_task_container",
        fail_start_task_container,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.stop_task_container",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("container was never started")
        ),
    )
    monkeypatch.setattr(
        "trace_collect.simulator.remove_image",
        lambda image, *, container_executable: removed_images.append(image) or True,
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

    startup = json.loads(
        (output_dir / "task-a" / "attempt_1" / "container_startup.json").read_text()
    )
    assert startup["status"] == "failed"
    assert startup["phases"][-1]["name"] == "start_task_container"
    assert startup["phases"][-1]["status"] == "failed"
    assert removed_images == ["fixed-image"]


def test_cloud_model_agent_start_failure_is_fail_soft_when_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")

    class _FailingContainerAgent:
        def __init__(
            self,
            container_id: str,
            container_executable: str,
            **_kwargs,
        ) -> None:
            assert container_id == "fake-cid"
            assert container_executable == "docker"

        async def start(self) -> None:
            raise RuntimeError("agent failed")

        async def stop(self) -> None:
            raise AssertionError("failed startup agent must not be finalized later")

    def fake_stop_task_container(container_id: str, *, executable: str) -> str:
        assert container_id == "fake-cid"
        assert executable == "docker"
        raise RuntimeError("container cleanup failed")

    removed_images: list[str] = []

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
    monkeypatch.setattr(
        "trace_collect.simulator.remove_image",
        lambda image, *, container_executable: removed_images.append(image) or True,
    )
    monkeypatch.setattr("trace_collect.openclaw_tools.ContainerAgent", _FailingContainerAgent)

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
    assert removed_images == ["fixed-image"]


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
    _patch_noop_sweep_fixed_prebuild(monkeypatch)
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    monkeypatch.setattr("trace_collect.simulator.stop_task_container", fake_stop_task_container)

    started = time.monotonic()
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
    _patch_noop_sweep_fixed_prebuild(monkeypatch)
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _RaisingSampler)
    monkeypatch.setattr("trace_collect.simulator.stop_task_container", fake_stop_task_container)

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


def _loaded_for_finalize(tmp_path: Path) -> object:
    from trace_collect.simulator import LoadedTraceSession

    return LoadedTraceSession(
        source_trace=tmp_path / "trace.jsonl",
        task_source=tmp_path / "tasks.json",
        task_instance_id="task-a",
        source_action_agent_id="task-a",
        run_instance_id="task-a",
        manifest_index=0,
        scaffold="openclaw",
        metadata=None,
        summary=None,
        task={},
        actions=[],
        iterations={},
    )


def test_replay_subagent_actions_keep_distinct_agent_id(tmp_path: Path) -> None:
    loaded = _loaded_for_finalize(tmp_path)
    parent = _make_trace_action(
        loaded=loaded,
        action_type="llm_call",
        action_id="llm_0",
        iteration=0,
        ts_start=1.0,
        ts_end=2.0,
        agent_id=_replay_agent_id_for_action(loaded, "task-a"),
        data={},
    )
    child = _make_trace_action(
        loaded=loaded,
        action_type="llm_call",
        action_id="llm_0",
        iteration=0,
        ts_start=1.1,
        ts_end=2.1,
        agent_id=_replay_agent_id_for_action(loaded, "task-a:subagent:child"),
        data={},
    )

    assert parent.action_id == child.action_id
    assert parent.agent_id == "task-a"
    assert child.agent_id == "task-a:subagent:child"
    assert {
        (parent.agent_id, parent.action_id),
        (child.agent_id, child.action_id),
    } == {
        ("task-a", "llm_0"),
        ("task-a:subagent:child", "llm_0"),
    }


def test_finalize_prepared_session_stops_container_when_agent_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trace_collect.simulator import (
        PreparedContainer,
        PreparedTraceSession,
        _finalize_prepared_session,
    )

    class _FailingAgent:
        async def stop(self) -> None:
            raise RuntimeError("agent stop failed")

    class _Recorder:
        def __init__(self) -> None:
            self.unregistered: list[str] = []

        def unregister_container(self, container_id: str) -> None:
            self.unregistered.append(container_id)

    container_stops: list[str] = []

    def fake_stop_task_container(container_id: str, *, executable: str) -> str:
        assert executable == "docker"
        container_stops.append(container_id)
        return ""

    recorder = _Recorder()
    prepared = PreparedTraceSession(
        loaded=_loaded_for_finalize(tmp_path),
        container=PreparedContainer(
            container_id="fake-cid",
            container_executable="docker",
            docker_image="fake-image",
            agent=_FailingAgent(),
            fixed_image="fixed-image",
        ),
        container_resource_recorder=recorder,
    )
    monkeypatch.setattr("trace_collect.simulator.stop_task_container", fake_stop_task_container)
    removed_images: list[str] = []
    monkeypatch.setattr(
        "trace_collect.simulator.remove_image",
        lambda image, *, container_executable: removed_images.append(image) or True,
    )

    with pytest.raises(RuntimeError, match="agent stop failed"):
        asyncio.run(_finalize_prepared_session(prepared))

    assert container_stops == ["fake-cid"]
    assert recorder.unregistered == ["fake-cid"]
    assert removed_images == ["fixed-image"]


def test_finalize_prepared_session_keeps_target_when_container_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trace_collect.simulator import (
        PreparedContainer,
        PreparedTraceSession,
        _finalize_prepared_session,
    )

    class _Agent:
        async def stop(self) -> None:
            pass

    class _Recorder:
        def __init__(self) -> None:
            self.unregistered: list[str] = []

        def unregister_container(self, container_id: str) -> None:
            self.unregistered.append(container_id)

    def fake_stop_task_container(container_id: str, *, executable: str) -> str:
        assert container_id == "fake-cid"
        assert executable == "docker"
        raise RuntimeError("container stop failed")

    recorder = _Recorder()
    prepared = PreparedTraceSession(
        loaded=_loaded_for_finalize(tmp_path),
        container=PreparedContainer(
            container_id="fake-cid",
            container_executable="docker",
            docker_image="fake-image",
            agent=_Agent(),
            fixed_image="fixed-image",
        ),
        container_resource_recorder=recorder,
    )
    monkeypatch.setattr("trace_collect.simulator.stop_task_container", fake_stop_task_container)
    removed_images: list[str] = []
    monkeypatch.setattr(
        "trace_collect.simulator.remove_image",
        lambda image, *, container_executable: removed_images.append(image) or True,
    )

    with pytest.raises(RuntimeError, match="container stop failed"):
        asyncio.run(_finalize_prepared_session(prepared))

    assert recorder.unregistered == []
    assert removed_images == []


def test_finalize_prepared_session_cleans_fixed_image_after_resource_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trace_collect.simulator import (
        PreparedContainer,
        PreparedTraceSession,
        _finalize_prepared_session,
    )

    class _Agent:
        async def stop(self) -> None:
            pass

    class _Sampler:
        def stop(self) -> list[dict]:
            raise RuntimeError("resource failed")

    container_stops: list[str] = []
    removed_images: list[str] = []

    monkeypatch.setattr(
        "trace_collect.simulator.stop_task_container",
        lambda container_id, *, executable: container_stops.append(container_id) or "",
    )
    monkeypatch.setattr(
        "trace_collect.simulator.remove_image",
        lambda image, *, container_executable: removed_images.append(image) or True,
    )

    prepared = PreparedTraceSession(
        loaded=_loaded_for_finalize(tmp_path),
        container=PreparedContainer(
            container_id="fake-cid",
            container_executable="docker",
            docker_image="fake-image",
            agent=_Agent(),
            fixed_image="fixed-image",
        ),
        sampler=_Sampler(),
        task_output_dir=tmp_path / "task-a" / "attempt_1",
    )

    with pytest.raises(RuntimeError, match="resource failed"):
        asyncio.run(_finalize_prepared_session(prepared))

    assert container_stops == ["fake-cid"]
    assert removed_images == ["fixed-image"]


def test_finalize_prepared_session_unregisters_before_resource_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trace_collect.simulator import (
        PreparedContainer,
        PreparedTraceSession,
        _finalize_prepared_session,
    )

    events: list[str] = []

    class _Agent:
        async def stop(self) -> None:
            events.append("agent_stop")

    class _Sampler:
        def stop(self) -> list[dict]:
            events.append("sampler_stop")
            return [{"container_id": "fake-cid"}]

    class _Recorder:
        def unregister_container(self, container_id: str) -> None:
            assert container_id == "fake-cid"
            events.append("unregister")

    def fake_stop_task_container(container_id: str, *, executable: str) -> str:
        assert container_id == "fake-cid"
        assert executable == "docker"
        events.append("container_stop")
        return ""

    def fake_write_resources(*_args, **_kwargs) -> None:
        events.append("resource_write")

    monkeypatch.setattr(
        "trace_collect.simulator.stop_task_container",
        fake_stop_task_container,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.attempt_layout.write_resources_json",
        fake_write_resources,
    )

    prepared = PreparedTraceSession(
        loaded=_loaded_for_finalize(tmp_path),
        container=PreparedContainer(
            container_id="fake-cid",
            container_executable="docker",
            docker_image="fake-image",
            agent=_Agent(),
        ),
        sampler=_Sampler(),
        task_output_dir=tmp_path / "task-a" / "attempt_1",
        container_resource_recorder=_Recorder(),
    )

    asyncio.run(_finalize_prepared_session(prepared))

    assert events == [
        "sampler_stop",
        "agent_stop",
        "container_stop",
        "unregister",
        "resource_write",
    ]


def test_finalize_prepared_session_raises_host_resource_write_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trace_collect.simulator import PreparedTraceSession, _finalize_prepared_session

    def fail_write_resources(*_args, **_kwargs) -> None:
        raise RuntimeError("resource write failed")

    monkeypatch.setattr(
        "trace_collect.simulator.attempt_layout.write_resources_json",
        fail_write_resources,
    )
    prepared = PreparedTraceSession(
        loaded=_loaded_for_finalize(tmp_path),
        container=None,
        task_output_dir=tmp_path / "task-a" / "attempt_1",
    )

    with pytest.raises(RuntimeError, match="resource write failed"):
        asyncio.run(_finalize_prepared_session(prepared))


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


def test_cloud_model_multi_worker_host_smoke(tmp_path: Path) -> None:
    trace_paths: list[Path] = []
    agent_ids = [f"host-task-{index}" for index in range(4)]
    for agent_id in agent_ids:
        trace_path = tmp_path / f"{agent_id}.jsonl"
        _write_trace(
            trace_path,
            agent_id=agent_id,
            scaffold="tongyi-deepresearch",
            execution_environment="host",
        )
        trace_paths.append(trace_path)
    task_source = tmp_path / "tasks.json"
    _write_host_tasks(task_source, *agent_ids)
    manifest = _write_manifest(tmp_path / "manifest.yaml", [str(path) for path in trace_paths])

    trace_file = asyncio.run(
        simulate(
            manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            concurrency=4,
            workers=2,
            prep_concurrency=2,
            replay_speed=100.0,
        )
    )

    summary = json.loads(
        trace_file.with_name(f"{trace_file.stem}.throughput_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["scheduler_mode"] == "multi_process_workers"
    assert summary["workers"] == 2
    assert summary["prep_concurrency"] == 2
    assert summary["effective_prep_concurrency"] == 2
    assert summary["attempted_traces"] == 4
    assert summary["monitoring"]["memory_bandwidth_enabled"] is False
    assert summary["container_resources"]["status"] == "disabled"
    assert summary["container_resources"]["monitoring"]["pmu_enabled"] is False
    records = _read_jsonl(trace_file)
    assert sum(1 for record in records if record.get("type") == "trace_metadata") == 1
    summary_records = [record for record in records if record.get("type") == "summary"]
    assert len(summary_records) == 4
    assert all(record["sleep_drift"]["sample_count"] >= 3 for record in summary_records)
    llm_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    )
    assert "action_sleep" in llm_record["data"]["sim_metrics"]
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    assert "source_gap_sleep" in tool_record["data"]["sim_metrics"]
    assert "action_sleep" in tool_record["data"]["sim_metrics"]
    action_starts = [
        record["ts_start"]
        for record in records
        if record.get("type") == "action"
    ]
    assert action_starts == sorted(action_starts)
    llm_starts = [
        record["ts_start"]
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    ]
    assert max(llm_starts) - min(llm_starts) < 0.25
    for agent_id in agent_ids:
        per_task_trace = tmp_path / "out" / agent_id / "attempt_1" / "trace.jsonl"
        assert per_task_trace.exists()
        per_task_metadata = _read_jsonl(per_task_trace)[0]
        assert per_task_metadata["manifest"] == str(manifest)
        assert per_task_metadata["concurrency"] == 4
        assert per_task_metadata["scheduler_mode"] == "multi_process_workers"
        assert per_task_metadata["monitoring"]["memory_bandwidth_enabled"] is False
        assert per_task_metadata["source_trace_count"] == 1
        assert per_task_metadata["instance_id"] == agent_id


def test_cloud_model_replay_marks_warmup_iterations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

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


def test_cloud_model_bounded_queue_records_arrival_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=10.0,
        )
    )

    summary = json.loads((tmp_path / "out" / "throughput_summary.json").read_text())
    task = summary["tasks"][0]
    assert summary["scheduler_mode"] == "bounded_queue"
    assert "replacement_load" not in summary
    assert summary["arrival_zero_wall_time_s"] > 0
    assert task["admission_wait_s"] is not None
    assert task["ready_to_terminal_s"] >= task["admission_wait_s"]


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
        tool_result_prefix="ok"
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
    assert "replacement_load" not in metadata
    assert metadata["source_trace_count"] == 2
    assert set(metadata["source_traces"]) == {str(trace_a), str(trace_b)}
    assert {record["agent_id"] for record in summaries} == {"task-a", "task-b"}
    assert {record["agent_id"] for record in llm_records} == {"task-a", "task-b"}
    assert abs(llm_records[0]["ts_start"] - llm_records[1]["ts_start"]) < 0.05


def test_cloud_model_staged_queue_records_common_ready_jct(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "trace-a.jsonl"
    trace_b = tmp_path / "trace-b.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    _write_trace(trace_a, agent_id="task-a")
    _write_trace(trace_b, agent_id="task-b")
    _write_tasks(task_source, "task-a", "task-b")
    _write_manifest(manifest, [str(trace_a), str(trace_b)])
    _patch_simulator_runtime(monkeypatch, tmp_path, tool_delay_s=0.01)

    trace_file = asyncio.run(
        simulate(
            manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            concurrency=1,
            prep_concurrency=2,
            container_executable="docker",
            replay_speed=100.0,
            stage_all_before_replay=True,
        )
    )

    metadata = _read_jsonl(trace_file)[0]
    summary = json.loads((tmp_path / "out" / "throughput_summary.json").read_text())
    assert metadata["scheduler_mode"] == "staged_bounded_queue"
    assert metadata["stage_all_before_replay"] is True
    assert summary["scheduler_mode"] == "staged_bounded_queue"
    assert summary["common_ready_wall_time_s"] > 0
    assert summary["ready_to_all_terminal_s"] == max(
        task["ready_to_terminal_s"] for task in summary["tasks"]
    )
    assert [task["run_instance_id"] for task in summary["tasks"]] == [
        "task-a",
        "task-b",
    ]
    assert summary["tasks"][0]["admission_wait_s"] < 0.05
    assert summary["tasks"][1]["admission_wait_s"] > 0


def test_staged_queue_finalizes_every_session_when_trace_split_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "trace-a.jsonl"
    trace_b = tmp_path / "trace-b.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    _write_trace(trace_a, agent_id="task-a")
    _write_trace(trace_b, agent_id="task-b")
    _write_tasks(task_source, "task-a", "task-b")
    _write_manifest(manifest, [str(trace_a), str(trace_b)])
    _patch_simulator_runtime(monkeypatch, tmp_path, tool_delay_s=0.001)
    finalized: list[str] = []

    def fake_split(*_args, **_kwargs) -> None:
        raise RuntimeError("split failed")

    async def fake_finalize(prepared: PreparedTraceSession) -> None:
        task_id = prepared.loaded.run_instance_id
        if task_id in finalized:
            return
        finalized.append(task_id)
        if task_id == "task-a":
            raise RuntimeError("first finalizer failed")

    monkeypatch.setattr(simulator_module, "_split_trace_by_agent", fake_split)
    monkeypatch.setattr(simulator_module, "_finalize_prepared_session", fake_finalize)

    with pytest.raises(RuntimeError, match="split failed"):
        asyncio.run(
            simulate(
                manifest=manifest,
                task_source=task_source,
                output_dir=tmp_path / "out",
                concurrency=1,
                prep_concurrency=2,
                container_executable="docker",
                replay_speed=100.0,
                stage_all_before_replay=True,
            )
        )

    assert sorted(finalized) == ["task-a", "task-b"]


def test_cloud_model_manifest_allows_duplicate_trace_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace-a.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    _write_manifest(manifest, [str(trace_path), str(trace_path)])
    _patch_simulator_runtime(monkeypatch, tmp_path)

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

    records = _read_jsonl(trace_file)
    metadata = records[0]
    summaries = [record for record in records if record.get("type") == "summary"]
    llm_records = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    ]
    expected_ids = {"task-a__replica-001", "task-a__replica-002"}

    assert metadata["source_traces"] == [str(trace_path), str(trace_path)]
    assert metadata["source_agent_ids"] == ["task-a", "task-a"]
    assert set(metadata["run_instance_ids"]) == expected_ids
    assert {record["agent_id"] for record in summaries} == expected_ids
    assert {record["source_agent_id"] for record in summaries} == {"task-a"}
    assert {record["task_id"] for record in summaries} == {"task-a"}
    assert {record["agent_id"] for record in llm_records} == expected_ids
    assert {record["data"]["source_agent_id"] for record in llm_records} == {"task-a"}
    assert {record["data"]["run_instance_id"] for record in llm_records} == expected_ids

    summary = json.loads((tmp_path / "out" / "throughput_summary.json").read_text())
    assert summary["attempted_traces"] == 2
    assert {task["source_agent_id"] for task in summary["tasks"]} == {"task-a"}
    assert {task["run_instance_id"] for task in summary["tasks"]} == expected_ids
    for run_instance_id in expected_ids:
        per_task_trace = tmp_path / "out" / run_instance_id / "attempt_1" / "trace.jsonl"
        per_task_records = _read_jsonl(per_task_trace)
        assert per_task_records[0]["instance_id"] == run_instance_id
        assert per_task_records[0]["source_trace_count"] == 1
        assert per_task_records[0]["source_traces"] == [str(trace_path)]
        assert per_task_records[0]["source_agent_ids"] == ["task-a"]
        assert per_task_records[0]["run_instance_ids"] == [run_instance_id]
        assert per_task_records[0]["source_models"] == ["claude-haiku"]
        assert per_task_records[0]["source_model"] == "claude-haiku"
        assert per_task_records[0]["source_trace_entries"] == [
            {
                "manifest_index": per_task_records[0]["manifest_index"],
                "source_trace": str(trace_path),
                "task_instance_id": "task-a",
                "source_action_agent_id": "task-a",
                "source_agent_id": "task-a",
                "run_instance_id": run_instance_id,
                "label": None,
                "source_model": "claude-haiku",
            }
        ]
        assert per_task_records[0]["source_agent_id"] == "task-a"


def test_cloud_model_duplicate_trace_run_ids_avoid_real_task_id_collision(
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "trace-a.jsonl"
    trace_replica = tmp_path / "trace-replica.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    _write_trace(
        trace_a,
        agent_id="task-a",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_trace(
        trace_replica,
        agent_id="task-a__replica-001",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_host_tasks(task_source, "task-a", "task-a__replica-001")
    _write_manifest(manifest, [str(trace_a), str(trace_a), str(trace_replica)])

    trace_file = asyncio.run(
        simulate(
            manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            concurrency=3,
            replay_speed=100.0,
        )
    )

    records = _read_jsonl(trace_file)
    summaries = [record for record in records if record.get("type") == "summary"]
    expected_ids = {
        "task-a__replica-001__entry-0000",
        "task-a__replica-002",
        "task-a__replica-001",
    }

    assert {record["agent_id"] for record in summaries} == expected_ids
    by_manifest_index = {record["manifest_index"]: record for record in summaries}
    assert by_manifest_index[0]["agent_id"] == "task-a__replica-001__entry-0000"
    assert by_manifest_index[0]["source_agent_id"] == "task-a"
    assert by_manifest_index[1]["agent_id"] == "task-a__replica-002"
    assert by_manifest_index[1]["source_agent_id"] == "task-a"
    assert by_manifest_index[2]["agent_id"] == "task-a__replica-001"
    assert by_manifest_index[2]["source_agent_id"] == "task-a__replica-001"


def test_cloud_model_duplicate_trace_run_ids_avoid_repeated_real_replica_id_collision(
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "trace-a.jsonl"
    trace_replica = tmp_path / "trace-replica.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.yaml"
    _write_trace(
        trace_a,
        agent_id="task-a",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_trace(
        trace_replica,
        agent_id="task-a__replica-001",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_host_tasks(task_source, "task-a", "task-a__replica-001")
    _write_manifest(
        manifest,
        [str(trace_a), str(trace_a), str(trace_replica), str(trace_replica)],
    )

    trace_file = asyncio.run(
        simulate(
            manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            concurrency=4,
            replay_speed=100.0,
        )
    )

    summaries = [
        record for record in _read_jsonl(trace_file) if record.get("type") == "summary"
    ]
    agent_ids = {record["agent_id"] for record in summaries}

    assert len(agent_ids) == 4
    assert "task-a__replica-001" not in agent_ids
    assert "task-a__replica-001__entry-0000" in agent_ids
    assert "task-a__replica-001__replica-001" in agent_ids
    assert "task-a__replica-001__replica-002" in agent_ids
    assert {record["source_agent_id"] for record in summaries} == {
        "task-a",
        "task-a__replica-001",
    }


def test_cloud_model_concurrency_limits_active_traces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _fake_container_resource_recorders: list[object],
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
    _patch_noop_sweep_fixed_prebuild(monkeypatch)
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)

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
    assert summary["effective_concurrency"] == 2
    assert summary["scheduler_mode"] == "bounded_queue"
    assert summary["attempted_traces"] == 3
    assert summary["completed_traces"] == 3
    assert len(_fake_container_resource_recorders) == 1
    recorder = _fake_container_resource_recorders[0]
    assert getattr(recorder, "started") is True
    assert getattr(recorder, "stopped") is True
    assert getattr(recorder, "sample_all_containers") is False
    assert sorted(getattr(recorder, "registered")) == [
        "fake-task-0",
        "fake-task-1",
        "fake-task-2",
    ]
    assert sorted(getattr(recorder, "unregistered")) == [
        "fake-task-0",
        "fake-task-1",
        "fake-task-2",
    ]
    assert summary["container_resources"]["sample_count"] == 1
    assert summary["container_resources"]["monitoring"] == summary["monitoring"]
    recorder_summary = json.loads(
        Path(summary["container_resources"]["summary_path"]).read_text()
    )
    assert recorder_summary["monitoring"] == summary["monitoring"]
    assert Path(summary["container_resources"]["jsonl_path"]).exists()
    assert Path(summary["container_resources"]["summary_path"]).exists()


def test_cloud_model_structured_manifest_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "trace-a.jsonl"
    trace_b = tmp_path / "trace-b.jsonl"
    default_tasks = tmp_path / "default-tasks.json"
    override_tasks = tmp_path / "override-tasks.json"
    manifest = tmp_path / "manifest.yaml"
    _write_trace(
        trace_a,
        agent_id="task-a",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_trace(
        trace_b,
        agent_id="task-b",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
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
    _patch_simulator_runtime(monkeypatch, tmp_path)

    trace_file = asyncio.run(
        simulate(
            manifest=manifest,
            task_source=None,
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
        tool_result_prefix="ok"
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
    assert container_records[0]["scaffold"] == "tongyi-deepresearch"
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
    _patch_noop_sweep_fixed_prebuild(monkeypatch)
    async def _fake_exec(*a, **kw):
        return ("ok", 1.0, True)

    monkeypatch.setattr("trace_collect.simulator._exec_tool", _fake_exec)

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
        tool_result_prefix="ok"
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
    assert payload["summary"]["monitoring_disabled"] is True
    assert payload["summary"]["monitoring"]["status"] == "disabled"


def test_execution_environment_infers_host_from_agent_runtime_mode() -> None:
    """Legacy traces that predate execution_environment still replay correctly
    when agent_runtime_mode=host_controller is present. Regression guard for
    Codex P1 feedback on cc3a18a (PR #13)."""
    from types import SimpleNamespace
    from trace_collect.simulate_utils import _execution_environment

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


def _cleanup_loaded(
    tmp_path: Path,
    task_id: str,
    *,
    manifest_index: int,
    image_name: str | None,
    execution_environment: str = "container",
    task_source_kind: str | None = None,
) -> LoadedTraceSession:
    task: dict[str, object] = {"instance_id": task_id, "problem_statement": task_id}
    if image_name is not None:
        task["image_name"] = image_name
    if task_source_kind is not None:
        task["task_source_kind"] = task_source_kind
    return LoadedTraceSession(
        source_trace=tmp_path / f"{task_id}.jsonl",
        task_source=tmp_path / "tasks.json",
        task_instance_id=task_id,
        source_action_agent_id=task_id,
        run_instance_id=task_id,
        manifest_index=manifest_index,
        scaffold="generic",
        metadata={"execution_environment": execution_environment},
        summary=None,
        task=task,
        actions=[],
        iterations={},
    )


def test_parse_simulate_args_cleanup_images_flag_defaults_off() -> None:
    default_args = parse_simulate_args(["--manifest", "manifest.yaml"])
    assert default_args.cleanup_images is False
    enabled = parse_simulate_args(["--manifest", "manifest.yaml", "--cleanup-images"])
    assert enabled.cleanup_images is True


def test_build_image_cleanup_state_refcounts_shared_and_unique(tmp_path: Path) -> None:
    sessions = [
        _cleanup_loaded(tmp_path, "shared-a", manifest_index=0, image_name="repo/shared"),
        _cleanup_loaded(tmp_path, "shared-b", manifest_index=1, image_name="repo/shared"),
        _cleanup_loaded(tmp_path, "unique", manifest_index=2, image_name="repo/unique"),
        # Host-mode session: no source image, excluded from refcounts.
        _cleanup_loaded(
            tmp_path,
            "host",
            manifest_index=3,
            image_name=None,
            execution_environment="host",
        ),
        # Terminal-bench registry: image resolves inside the harness, excluded.
        _cleanup_loaded(
            tmp_path,
            "tb",
            manifest_index=4,
            image_name="repo/tb",
            task_source_kind="terminal_bench_registry",
        ),
    ]

    state = _build_image_cleanup_state(sessions, container_executable="docker")

    shared = normalize_image_reference("repo/shared")
    unique = normalize_image_reference("repo/unique")
    assert state.refcounts == {shared: 2, unique: 1}
    assert state.image_by_instance == {
        "shared-a": shared,
        "shared-b": shared,
        "unique": unique,
    }


def test_release_source_image_removes_only_when_refcount_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sessions = [
        _cleanup_loaded(tmp_path, "a", manifest_index=0, image_name="repo/shared"),
        _cleanup_loaded(tmp_path, "b", manifest_index=1, image_name="repo/shared"),
    ]
    state = _build_image_cleanup_state(sessions, container_executable="docker")

    removed: list[str] = []

    def fake_remove_image(image: str, *, container_executable: str) -> bool:
        removed.append(image)
        return True

    monkeypatch.setattr("trace_collect.simulator.remove_image", fake_remove_image)

    async def run() -> None:
        await _release_source_image(state, "a")
        assert removed == []  # one holder remains
        await _release_source_image(state, "b")

    asyncio.run(run())

    shared = normalize_image_reference("repo/shared")
    assert removed == [shared]
    assert state.refcounts[shared] == 0
    assert state.skipped == 0


def test_release_source_image_is_best_effort_on_remove_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sessions = [
        _cleanup_loaded(tmp_path, "a", manifest_index=0, image_name="repo/unique"),
    ]
    state = _build_image_cleanup_state(sessions, container_executable="docker")

    def boom(image: str, *, container_executable: str) -> bool:
        raise RuntimeError("docker rm failed")

    monkeypatch.setattr("trace_collect.simulator.remove_image", boom)

    # Must not raise: cleanup failures are logged and counted, never fatal.
    asyncio.run(_release_source_image(state, "a"))
    assert state.skipped == 1


def test_simulate_cleanup_images_skips_prefetch_and_removes_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a", tool_name="write_file")
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    events: list[tuple[str, str]] = []

    async def recording_prefetch(*_args, **_kwargs) -> None:
        events.append(("prefetch", ""))

    async def recording_prebuild(*_args, **_kwargs) -> dict[str, str]:
        events.append(("prebuild", ""))
        return {}

    async def recording_finalize(prepared) -> None:
        events.append(("finalize", prepared.loaded.task_instance_id))

    def recording_remove(image: str, *, container_executable: str) -> bool:
        events.append(("remove", image))
        return True

    monkeypatch.setattr(
        "trace_collect.simulator._prefetch_container_images", recording_prefetch
    )
    monkeypatch.setattr(
        "trace_collect.simulator._prebuild_sweep_fixed_images", recording_prebuild
    )
    monkeypatch.setattr(
        "trace_collect.simulator._finalize_prepared_session", recording_finalize
    )
    monkeypatch.setattr("trace_collect.simulator.remove_image", recording_remove)

    asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
            cleanup_images=True,
        )
    )

    kinds = [kind for kind, _ in events]
    # Prefetch / prebuild are skipped entirely under --cleanup-images.
    assert "prefetch" not in kinds
    assert "prebuild" not in kinds
    # The task's source image is removed, and only after its session finalizes.
    expected_image = normalize_image_reference("swebench-test/task-a")
    assert ("finalize", "task-a") in events
    assert ("remove", expected_image) in events
    assert events.index(("finalize", "task-a")) < events.index(
        ("remove", expected_image)
    )
