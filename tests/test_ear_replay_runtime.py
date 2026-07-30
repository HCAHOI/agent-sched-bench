from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from ear.controller import ResourceController
from ear.executor.docker import DockerResourceManager

from trace_collect.ear_replay_runtime import EarReplayRuntime
from trace_collect.simulator import (
    _run_terminal_bench_compose,
    _write_ear_compose_override,
)


def _policy(tmp_path: Path) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(
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
    return path


def _cgroup(
    tmp_path: Path,
    *,
    cpu_cores: int,
    memory_bytes: int,
    oom_kill: int = 0,
) -> Path:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "cpu.stat").write_text(
        "usage_usec 0\nnr_periods 0\nnr_throttled 0\nthrottled_usec 0\n",
        encoding="utf-8",
    )
    (cgroup / "cpu.max").write_text(
        f"{cpu_cores * 100000} 100000\n",
        encoding="utf-8",
    )
    (cgroup / "memory.current").write_text("0\n", encoding="utf-8")
    (cgroup / "memory.max").write_text(f"{memory_bytes}\n", encoding="utf-8")
    (cgroup / "memory.events").write_text(
        f"low 0\nhigh 0\nmax 0\noom 0\noom_kill {oom_kill}\n",
        encoding="utf-8",
    )
    return cgroup


@pytest.mark.parametrize(
    ("mode", "cpu_cores", "memory_bytes"),
    [("fixed", 8, 8 * 1024**3), ("elastic", 1, 2 * 1024**3)],
)
def test_ear_replay_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    cpu_cores: int,
    memory_bytes: int,
) -> None:
    cgroup = _cgroup(
        tmp_path,
        cpu_cores=cpu_cores,
        memory_bytes=memory_bytes,
    )
    monkeypatch.setattr(
        "trace_collect.ear_replay_runtime.resolve_docker_container_cgroup",
        lambda *args, **kwargs: cgroup,
    )

    class Monitor:
        def start(self) -> None:
            pass

        def finish(self) -> Any:
            return SimpleNamespace(
                errors=[],
                cpu_resize_rejected_events=0,
                memory_resize_rejected_events=0,
            )

    monkeypatch.setattr(
        DockerResourceManager,
        "begin_command",
        lambda _self: Monitor(),
    )
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

    def stop(container_id: str) -> None:
        stops.append(container_id)

    container_id = runtime.start_task_container(
        "image",
        agent_id="task-1",
        start_fn=start,
        stop_fn=stop,
    )
    runtime.stop_task_container(container_id, stop_fn=stop)
    runtime.write_artifacts(tmp_path / "out")

    assert runtime.valid is True
    assert stops == ["container-1"]
    assert starts[0][starts[0].index("--cpus") + 1] == str(cpu_cores)
    assert starts[0][starts[0].index("--memory") + 1] == str(memory_bytes)
    events = [
        json.loads(line)
        for line in (tmp_path / "out" / "lease_events.jsonl").read_text().splitlines()
    ]
    assert [event["event_type"] for event in events] == ["acquired", "released"]


def test_compose_override_uses_admitted_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = EarReplayRuntime(
        policy_path=_policy(tmp_path),
        mode="elastic",
        concurrency=2,
        container_executable="docker",
    )
    reservation = runtime.reserve_task("task-compose")
    override = tmp_path / "ear.override.yaml"
    _write_ear_compose_override(override, reservation)

    payload = yaml.safe_load(override.read_text())
    assert payload == {
        "services": {
            "client": {
                "cpuset": reservation.cpuset,
                "cpus": 1.0,
                "mem_limit": str(2 * 1024**3),
                "memswap_limit": str(2 * 1024**3),
                "oom_score_adj": -1000,
            }
        }
    }

    calls: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: Any) -> Any:
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("trace_collect.simulator.subprocess.run", run)
    _run_terminal_bench_compose(
        container_executable="docker",
        project="task",
        compose_file=tmp_path / "compose.yaml",
        compose_override_file=override,
        env={},
        args=["up", "-d"],
    )
    assert calls == [
        [
            "docker",
            "compose",
            "-p",
            "task",
            "-f",
            str(tmp_path / "compose.yaml"),
            "-f",
            str(override),
            "up",
            "-d",
        ]
    ]
    runtime.cancel_reservation(reservation)


def test_ear_replay_oom_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cgroup = _cgroup(
        tmp_path,
        cpu_cores=8,
        memory_bytes=8 * 1024**3,
        oom_kill=1,
    )
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
    runtime.start_task_container(
        "image",
        agent_id="task-oom",
        start_fn=lambda *_args, **_kwargs: "container-oom",
        stop_fn=lambda _container_id: None,
    )
    runtime.stop_task_container(
        "container-oom",
        stop_fn=lambda _container_id: None,
    )
    runtime.write_artifacts(tmp_path / "out")

    assert runtime.valid is False
    summary = json.loads((tmp_path / "out" / "controller_summary.json").read_text())
    assert summary["ear_runtime"]["oom_kill_count"] == 1


def test_ear_admission_failure_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = EarReplayRuntime(
        policy_path=_policy(tmp_path),
        mode="fixed",
        concurrency=16,
        container_executable="docker",
    )

    def reject(*_args: Any, **_kwargs: Any) -> Any:
        raise TimeoutError("pool exhausted")

    monkeypatch.setattr(ResourceController, "acquire", reject)
    with pytest.raises(TimeoutError, match="pool exhausted"):
        runtime.reserve_task("task-rejected")
    runtime.write_artifacts(tmp_path / "out")

    assert runtime.valid is False
    summary = json.loads((tmp_path / "out" / "controller_summary.json").read_text())
    assert "lease acquisition failed" in summary["ear_runtime"]["errors"][0]
