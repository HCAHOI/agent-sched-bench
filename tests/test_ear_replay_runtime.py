from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from ear.controller import ResourceController
from ear.executor.docker import DockerCgroupView, DockerResourceManager

from scripts.plot_ear_replay import _load_run
from trace_collect.ear_replay_runtime import EarReplayRuntime, _ear_git_commit
from trace_collect.simulator import _parse_trace_session_file


def _policy(tmp_path: Path) -> Path:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "name": "asb-ear-test",
                "executor": "docker_fresh",
                "docker": {
                    "run_timeout_s": 1.0,
                    "oom_score_adj": -1000,
                    "resource_manager": {
                        "enabled": True,
                        "mode": "adaptive",
                        "policy_id": "asb-ear-test",
                        "guest_vcpus": 8,
                        "base_memory_mib": 2048,
                        "hotplug_total_mib": 6144,
                        "cpu_pages": [1, 2, 4, 8],
                        "initial_cpu_pages": 1,
                        "sample_interval_s": 0.01,
                        "elastic_cpu_lease": True,
                        "elastic_memory_lease": True,
                        "memory_reclaim_guest_hints": False,
                        "memory_reclaim_deadline_s": 0.0,
                    },
                },
                "controller": {
                    "total_cpus": 8,
                    "total_memory_gb": 8.0,
                    "default_agent_quota": {
                        "cpu_cores": 8.0,
                        "memory_gb": 8.0,
                        "max_concurrent_leases": 1,
                    },
                    "admission_policy": "block_with_timeout",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return policy_path


def _cgroup(tmp_path: Path, *, oom_kill: int = 0) -> Path:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "cpu.stat").write_text(
        "usage_usec 0\nnr_periods 0\nnr_throttled 0\nthrottled_usec 0\n",
        encoding="utf-8",
    )
    (cgroup / "cpu.max").write_text("100000 100000\n", encoding="utf-8")
    (cgroup / "memory.current").write_text("0\n", encoding="utf-8")
    (cgroup / "memory.max").write_text(str(2 * 1024**3), encoding="utf-8")
    (cgroup / "memory.events").write_text(
        f"low 0\nhigh 0\nmax 0\noom 0\noom_kill {oom_kill}\n",
        encoding="utf-8",
    )
    return cgroup


@pytest.mark.parametrize(
    ("mode", "expected_cpus", "expected_memory"),
    [("fixed", "4", str(4 * 1024**3)), ("elastic", "1", str(2 * 1024**3))],
)
def test_ear_replay_lifecycle(
    tmp_path: Path,
    monkeypatch: Any,
    mode: str,
    expected_cpus: str,
    expected_memory: str,
) -> None:
    cgroup = _cgroup(tmp_path)
    monkeypatch.setattr(
        "trace_collect.ear_replay_runtime.resolve_docker_container_cgroup",
        lambda *args, **kwargs: cgroup,
    )
    monkeypatch.setattr(DockerCgroupView, "_update", lambda *args, **kwargs: None)
    runtime = EarReplayRuntime(
        policy_path=_policy(tmp_path),
        mode=mode,  # type: ignore[arg-type]
        concurrency=2,
        container_executable="docker",
    )
    starts: list[list[str]] = []
    stops: list[str] = []

    def start(_image: str, **kwargs: Any) -> str:
        starts.append(kwargs["extra_args"])
        return "container-1"

    def stop(container_id: str, **_kwargs: Any) -> None:
        stops.append(container_id)

    container_id = runtime.start_task_container(
        "image",
        agent_id="task-1",
        start_fn=start,
        stop_fn=stop,
        executable="docker",
    )
    runtime.stop_task_container(
        container_id,
        stop_fn=stop,
        executable="docker",
    )
    runtime.write_artifacts(tmp_path / "out")

    assert runtime.valid is True
    assert stops == ["container-1"]
    assert starts[0][starts[0].index("--cpus") + 1] == expected_cpus
    assert starts[0][starts[0].index("--memory") + 1] == expected_memory
    events = [
        json.loads(line)
        for line in (tmp_path / "out" / "lease_events.jsonl").read_text().splitlines()
    ]
    assert [event["event_type"] for event in events] == ["acquired", "released"]
    summary = json.loads(
        (tmp_path / "out" / "controller_summary.json").read_text()
    )
    assert summary["ear_runtime"]["status"] == "valid"


def test_ear_replay_oom_is_invalid(tmp_path: Path, monkeypatch: Any) -> None:
    cgroup = _cgroup(tmp_path, oom_kill=1)
    monkeypatch.setattr(
        "trace_collect.ear_replay_runtime.resolve_docker_container_cgroup",
        lambda *args, **kwargs: cgroup,
    )
    runtime = EarReplayRuntime(
        policy_path=_policy(tmp_path),
        mode="fixed",
        concurrency=2,
        container_executable="docker",
    )

    def start(_image: str, **_kwargs: Any) -> str:
        return "container-oom"

    def stop(_container_id: str, **_kwargs: Any) -> None:
        return None

    runtime.start_task_container(
        "image",
        agent_id="task-oom",
        start_fn=start,
        stop_fn=stop,
        executable="docker",
    )
    runtime.stop_task_container(
        "container-oom",
        stop_fn=stop,
        executable="docker",
    )
    runtime.write_artifacts(tmp_path / "out")

    assert runtime.valid is False
    summary = json.loads(
        (tmp_path / "out" / "controller_summary.json").read_text()
    )
    assert summary["ear_runtime"]["oom_kill_count"] == 1
    assert summary["ear_runtime"]["status"] == "invalid"


def test_ear_replay_rejections_are_invalid(tmp_path: Path, monkeypatch: Any) -> None:
    runtime = EarReplayRuntime(
        policy_path=_policy(tmp_path),
        mode="fixed",
        concurrency=2,
        container_executable="docker",
    )

    def reject(*_args: Any, **_kwargs: Any) -> Any:
        raise TimeoutError("pool exhausted")

    monkeypatch.setattr(ResourceController, "acquire", reject)
    with pytest.raises(TimeoutError, match="pool exhausted"):
        runtime.start_task_container(
            "image",
            agent_id="task-rejected",
            start_fn=lambda *_args, **_kwargs: "never",
            stop_fn=lambda *_args, **_kwargs: None,
            executable="docker",
        )
    runtime.write_artifacts(tmp_path / "rejected")

    assert runtime.valid is False
    summary = json.loads(
        (tmp_path / "rejected" / "controller_summary.json").read_text()
    )
    assert summary["ear_runtime"]["status"] == "invalid"
    assert "lease acquisition failed" in summary["ear_runtime"]["errors"][0]


def test_ear_replay_resize_rejection_is_invalid(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    cgroup = _cgroup(tmp_path)
    monkeypatch.setattr(
        "trace_collect.ear_replay_runtime.resolve_docker_container_cgroup",
        lambda *args, **kwargs: cgroup,
    )

    class RejectedMonitor:
        def start(self) -> None:
            pass

        def finish(self) -> Any:
            return SimpleNamespace(
                errors=[],
                cpu_resize_rejected_events=1,
                memory_resize_rejected_events=0,
            )

    monkeypatch.setattr(
        DockerResourceManager,
        "begin_command",
        lambda _self: RejectedMonitor(),
    )
    runtime = EarReplayRuntime(
        policy_path=_policy(tmp_path),
        mode="elastic",
        concurrency=2,
        container_executable="docker",
    )
    runtime.start_task_container(
        "image",
        agent_id="task-resize-rejected",
        start_fn=lambda *_args, **_kwargs: "container-rejected",
        stop_fn=lambda *_args, **_kwargs: None,
        executable="docker",
    )
    runtime.stop_task_container(
        "container-rejected",
        stop_fn=lambda *_args, **_kwargs: None,
        executable="docker",
    )
    runtime.write_artifacts(tmp_path / "resize-rejected")

    assert runtime.valid is False
    summary = json.loads(
        (tmp_path / "resize-rejected" / "controller_summary.json").read_text()
    )
    assert summary["ear_runtime"]["status"] == "invalid"
    assert "resize events rejected" in summary["ear_runtime"]["errors"][0]


def test_collect_trace_maps_tool_container_to_task_identity(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "type": "trace_metadata",
                    "instance_id": "task-slug",
                    "tool_container_id": "container-id",
                },
                {
                    "type": "action",
                    "agent_id": "container-id",
                    "action_type": "tool_exec",
                    "action_id": "tool-1",
                    "iteration": 0,
                    "ts_start": 1.0,
                    "ts_end": 2.0,
                    "data": {},
                },
                {
                    "type": "action",
                    "agent_id": "container-id/subagent-1",
                    "action_type": "tool_exec",
                    "action_id": "tool-2",
                    "iteration": 0,
                    "ts_start": 2.0,
                    "ts_end": 3.0,
                    "data": {},
                },
                {
                    "type": "summary",
                    "agent_id": "container-id",
                    "success": True,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    source_agent_id, _metadata, actions, summary = _parse_trace_session_file(trace)

    assert source_agent_id == "task-slug"
    assert [action["agent_id"] for action in actions] == [
        "task-slug",
        "task-slug/subagent-1",
    ]
    assert summary == {
        "type": "summary",
        "agent_id": "container-id",
        "success": True,
    }


def test_plot_loader_keeps_container_series_separate(tmp_path: Path) -> None:
    (tmp_path / "throughput_summary.json").write_text(
        json.dumps(
            {
                "wall_time_s": 2.0,
                "completed_traces": 2,
                "failed_traces": 0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "controller_summary.json").write_text(
        json.dumps(
            {
                "total_reserved_cpu_core_seconds": 4.0,
                "total_reserved_memory_gb_seconds": 8.0,
                "ear_runtime": {
                    "status": "valid",
                    "oom_kill_count": 0,
                    "clock_anchor": {"monotonic_s": 10.0, "epoch_s": 100.0},
                },
            }
        ),
        encoding="utf-8",
    )
    events = [
        {
            "event_type": event_type,
            "lease_id": lease_id,
            "agent_id": agent_id,
            "timestamp": timestamp,
            "cpu_cores": cpu,
            "memory_gb": memory,
        }
        for lease_id, agent_id, timestamp, cpu, memory, event_type in [
            ("lease-a", "task-a", 10.0, 1, 2.0, "acquired"),
            ("lease-b", "task-b", 10.5, 2, 3.0, "acquired"),
            ("lease-a", "task-a", 11.0, 1, 2.0, "released"),
            ("lease-b", "task-b", 12.0, 2, 3.0, "released"),
        ]
    ]
    (tmp_path / "lease_events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    for agent_id, cpu_percent in (("task-a", "100%"), ("task-b", "200%")):
        resources = tmp_path / agent_id / "attempt_1" / "resources.json"
        resources.parent.mkdir(parents=True)
        resources.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "epoch": 100.0,
                            "cpu_percent": cpu_percent,
                            "mem_usage": "1GiB / 4GiB",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    run = _load_run(tmp_path)

    assert [series["agent_id"] for series in run["series"]] == ["task-a", "task-b"]
    assert [series["cpu_steps"][0][1] for series in run["series"]] == [1.0, 2.0]
    assert [series["observed"][0]["cpu_cores"] for series in run["series"]] == [
        1.0,
        2.0,
    ]

    (tmp_path / "lease_events.jsonl").write_text(
        json.dumps(
            {
                "event_type": "admission_rejected",
                "lease_id": None,
                "agent_id": "task-rejected",
                "timestamp": 10.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert _load_run(tmp_path)["series"] == []


def test_ear_git_commit_is_optional_without_git(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "trace_collect.ear_replay_runtime.distribution",
        lambda _name: SimpleNamespace(read_text=lambda _path: "{}"),
    )

    def no_git(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError("git")

    monkeypatch.setattr("trace_collect.ear_replay_runtime.subprocess.run", no_git)

    assert _ear_git_commit() is None
