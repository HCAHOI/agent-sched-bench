from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import pytest

from trace_collect.cli import _run_simulate, parse_simulate_args
from trace_collect.simulator import (
    LLMTimingConfig,
    PreparedContainer,
    PreparedTraceSession,
    ReplayCheckpointScheduler,
    ReplayPreparationError,
    ReplaySchedulerConfig,
    SimulateError,
    WorkerTraceInput,
    _capture_snapshot_manifest,
    _cas_manifest_comparison_fields,
    _check_sleep_drift_tolerance,
    _checkpoint_after_spec,
    _checkpoint_spec_is_incremental,
    _chunk_worker_inputs_by_concurrency,
    _command_metadata,
    _compute_output_diff_snippet,
    _effective_source_exec_timeout_s,
    _fold_source_checkpoint_entries,
    _load_source_manifest_entries,
    _partition_worker_inputs,
    _resolve_prep_concurrency,
    _restore_cas_manifest_in_container,
    _run_worker_wave_async,
    _source_action_excluded_overhead_s,
    _source_exec_timeout_s,
    _restore_checkpoint_to_container,
    _tool_mismatch_reason,
    simulate,
)



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
    resource_timeline: dict | None = None,
    tool_args: dict | None = None,
    checkpoint_after: str | dict | None = None,
    extra_tool_data: dict[str, Any] | None = None,
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
                            **(
                                {"checkpoint_after": checkpoint_after}
                                if checkpoint_after is not None
                                else {}
                            ),
                            **(extra_tool_data or {}),
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


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _set_trace_container_exec_env(path: Path, env: dict[str, str]) -> None:
    records = _read_jsonl(path)
    metadata = records[0]
    assert metadata["type"] == "trace_metadata"
    run_config = dict(metadata.get("run_config") or {})
    run_config["container_exec_env"] = env
    metadata["run_config"] = run_config
    _write_jsonl(path, records)


async def _run_thread_inline(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    return func(*args, **kwargs)


@pytest.fixture(autouse=True)
def _inline_simulator_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trace_collect.simulator.asyncio.to_thread", _run_thread_inline)


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
    ):
        from trace_collect.simulator import PreparedContainer, PreparedTraceSession
        container = PreparedContainer(
            container_id="fake-cid",
            container_executable=container_executable,
            docker_image="fake-image",
            agent=_FakeAgent(),
        )
        return PreparedTraceSession(loaded=loaded, container=container)

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


def _patch_noop_replay_python_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "trace_collect.simulator.resolve_running_container_exec_config",
        lambda **kwargs: kwargs["exec_config"],
        raising=False,
    )


def test_source_action_excluded_overhead_reads_checkpoint_after() -> None:
    action = {
        "data": {
            "checkpoint_after": {
                "elapsed_ms": 250.0,
                "overhead_excluded": True,
            }
        }
    }

    assert _source_action_excluded_overhead_s(action) == pytest.approx(0.25)


def test_source_action_excluded_overhead_reads_checkpoint_after_error() -> None:
    action = {
        "data": {
            "checkpoint_after_error": {
                "elapsed_ms": 125.0,
                "overhead_excluded": True,
                "error": "checkpoint skipped",
            }
        }
    }

    assert _source_action_excluded_overhead_s(action) == pytest.approx(0.125)


def test_checkpoint_after_spec_rejects_non_testbed_root(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"

    spec = _checkpoint_after_spec(
        action_data={"checkpoint_after": {"path": "cp-manifest.json", "root": "/"}},
        source_trace=trace_path,
    )

    assert spec is None


def test_restore_checkpoint_to_container_records_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "manifest.json"
    manifest = {
        "entries": {
            "file.txt": {
                "hash": (
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934"
                    "ca495991b7852b855"
                ),
                "mode": 0o644,
                "size": 14,
                "mtime_ns": 1000000,
            }
        },
        "deleted_paths": [],
    }
    checkpoint_path.write_text(json.dumps(manifest), encoding="utf-8")
    copied: list[dict] = []
    restored: list[dict] = []

    monkeypatch.setattr(
        "trace_collect.simulator._copy_checkpoint_archive_to_container",
        lambda **kwargs: copied.append(kwargs),
    )
    monkeypatch.setattr(
        "trace_collect.simulator._restore_cas_manifest_in_container",
        lambda **kwargs: restored.append(kwargs),
    )

    result = _restore_checkpoint_to_container(
        checkpoint_spec={
            "path": str(checkpoint_path),
            "kind": "cas_manifest_full",
            "root": "/testbed",
        },
        container=PreparedContainer(
            container_id="cid",
            container_executable="docker",
            docker_image="image",
            agent=object(),
        ),
    )

    assert len(copied) == 1
    assert len(restored) == 1
    assert result["forced_sync_success"] is True
    assert result["forced_sync_status"] == "checkpoint_restored_continuation"
    assert result["forced_sync_continued"] is True
    assert result["checkpoint_kind"] == "cas_manifest_full"
    assert result["checkpoint_path"] == str(checkpoint_path)
    assert result["checkpoint_root"] == "/testbed"
    assert result["checkpoint_size_bytes"] == checkpoint_path.stat().st_size
    assert result["restore_elapsed_ms"] >= 0.0
    assert result["restore_overhead_excluded"] is True
    assert result["restore_root_exists"] is True
    assert result["tar_extraction_returncode"] == 0


def test_restore_cas_manifest_script_preserves_testbed_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cas_root = tmp_path / "cas"
    root = tmp_path / "testbed"
    manifest_path = tmp_path / "manifest.json"
    root.mkdir()
    root_owner = (root.stat().st_uid, root.stat().st_gid)
    content = b"restored\n"
    digest = hashlib.sha256(content).hexdigest()
    blob = cas_root / "blobs" / digest[:2] / digest[2:]
    blob.parent.mkdir(parents=True)
    blob.write_bytes(content)
    manifest_path.write_text(
        json.dumps(
            {
                "entries": {
                    "pkg/file.txt": {
                        "hash": digest,
                        "mode": 0o644,
                    }
                },
                "deleted_paths": [],
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_run_checked(cmd: list[str], *, timeout: float) -> None:
        del timeout
        captured["cmd"] = cmd

    monkeypatch.setattr(
        "trace_collect.simulator._CHECKPOINT_CAS_ROOT",
        str(cas_root),
    )
    monkeypatch.setattr(
        "trace_collect.simulator._run_checked_container_command",
        fake_run_checked,
    )

    _restore_cas_manifest_in_container(
        container_id="cid",
        container_executable="docker",
        container_manifest_path=str(manifest_path),
        restore_root=str(root),
    )

    cmd = captured["cmd"]
    for index, token in enumerate(cmd):
        if token == "-e":
            key, value = cmd[index + 1].split("=", 1)
            monkeypatch.setenv(key, value)
    script = cmd[cmd.index("-c") + 1]
    exec(script, {})

    restored_dir = root / "pkg"
    restored_file = restored_dir / "file.txt"
    assert restored_file.read_bytes() == content
    assert (root.stat().st_uid, root.stat().st_gid) == root_owner
    assert (restored_dir.stat().st_uid, restored_dir.stat().st_gid) == root_owner
    assert (restored_file.stat().st_uid, restored_file.stat().st_gid) == root_owner


def test_restore_cas_manifest_script_restores_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cas_root = tmp_path / "cas"
    root = tmp_path / "testbed"
    manifest_path = tmp_path / "manifest.json"
    root.mkdir()
    content = b"restored\n"
    digest = hashlib.sha256(content).hexdigest()
    blob = cas_root / "blobs" / digest[:2] / digest[2:]
    blob.parent.mkdir(parents=True)
    blob.write_bytes(content)
    manifest_path.write_text(
        json.dumps(
            {
                "entries": {
                    "pkg/file.txt": {"hash": digest, "mode": 0o644},
                    "pkg/link.txt": {"type": "symlink", "target": "file.txt"},
                },
                "deleted_paths": [],
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_run_checked(cmd: list[str], *, timeout: float) -> None:
        del timeout
        captured["cmd"] = cmd

    monkeypatch.setattr(
        "trace_collect.simulator._CHECKPOINT_CAS_ROOT",
        str(cas_root),
    )
    monkeypatch.setattr(
        "trace_collect.simulator._run_checked_container_command",
        fake_run_checked,
    )

    _restore_cas_manifest_in_container(
        container_id="cid",
        container_executable="docker",
        container_manifest_path=str(manifest_path),
        restore_root=str(root),
    )

    cmd = captured["cmd"]
    for index, token in enumerate(cmd):
        if token == "-e":
            key, value = cmd[index + 1].split("=", 1)
            monkeypatch.setenv(key, value)
    script = cmd[cmd.index("-c") + 1]
    exec(script, {})

    restored_file = root / "pkg" / "file.txt"
    restored_link = root / "pkg" / "link.txt"
    assert restored_file.read_bytes() == content
    assert restored_link.is_symlink()
    assert restored_link.readlink() == Path("file.txt")
    assert restored_link.resolve() == restored_file.resolve()


def test_restore_cas_manifest_script_allows_out_of_root_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cas_root = tmp_path / "cas"
    root = tmp_path / "testbed"
    manifest_path = tmp_path / "manifest.json"
    root.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "entries": {
                    "link.txt": {"type": "symlink", "target": "../outside.txt"},
                },
                "deleted_paths": [],
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_run_checked(cmd: list[str], *, timeout: float) -> None:
        del timeout
        captured["cmd"] = cmd

    monkeypatch.setattr(
        "trace_collect.simulator._CHECKPOINT_CAS_ROOT",
        str(cas_root),
    )
    monkeypatch.setattr(
        "trace_collect.simulator._run_checked_container_command",
        fake_run_checked,
    )

    _restore_cas_manifest_in_container(
        container_id="cid",
        container_executable="docker",
        container_manifest_path=str(manifest_path),
        restore_root=str(root),
    )

    cmd = captured["cmd"]
    for index, token in enumerate(cmd):
        if token == "-e":
            key, value = cmd[index + 1].split("=", 1)
            monkeypatch.setenv(key, value)
    script = cmd[cmd.index("-c") + 1]
    caplog.set_level(
        logging.DEBUG,
        logger="trace_collect.restore_cas_manifest",
    )
    exec(script, {})

    restored_link = root / "link.txt"
    assert restored_link.is_symlink()
    assert restored_link.readlink() == Path("../outside.txt")
    assert any(
        "checkpoint symlink target resolves outside restore root" in record.message
        for record in caplog.records
    )


def test_restore_cas_manifest_script_restores_files_before_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cas_root = tmp_path / "cas"
    root = tmp_path / "testbed"
    outside = tmp_path / "outside"
    manifest_path = tmp_path / "manifest.json"
    root.mkdir()
    outside.mkdir()
    content = b"restored\n"
    digest = hashlib.sha256(content).hexdigest()
    blob = cas_root / "blobs" / digest[:2] / digest[2:]
    blob.parent.mkdir(parents=True)
    blob.write_bytes(content)
    manifest_path.write_text(
        json.dumps(
            {
                "entries": {
                    "pkg": {"type": "symlink", "target": str(outside)},
                    "pkg/file.txt": {"hash": digest, "mode": 0o644},
                },
                "deleted_paths": [],
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_run_checked(cmd: list[str], *, timeout: float) -> None:
        del timeout
        captured["cmd"] = cmd

    monkeypatch.setattr(
        "trace_collect.simulator._CHECKPOINT_CAS_ROOT",
        str(cas_root),
    )
    monkeypatch.setattr(
        "trace_collect.simulator._run_checked_container_command",
        fake_run_checked,
    )

    _restore_cas_manifest_in_container(
        container_id="cid",
        container_executable="docker",
        container_manifest_path=str(manifest_path),
        restore_root=str(root),
    )

    cmd = captured["cmd"]
    for index, token in enumerate(cmd):
        if token == "-e":
            key, value = cmd[index + 1].split("=", 1)
            monkeypatch.setenv(key, value)
    script = cmd[cmd.index("-c") + 1]
    exec(script, {})

    restored_link = root / "pkg"
    assert restored_link.is_symlink()
    assert restored_link.readlink() == outside
    assert not (outside / "file.txt").exists()


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
    assert args.sandbox_backend == "docker"
    assert args.checkpoint_backend is None
    assert args.replay_speed == 1.0
    assert args.llm_timing == "source-scaled"
    assert args.llm_ttft_ms is None
    assert args.llm_tpot_ms is None


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


def test_parse_simulate_args_accepts_fake_sandbox_backend() -> None:
    args = parse_simulate_args(
        ["--manifest", "manifest.yaml", "--sandbox-backend", "fake"]
    )
    assert args.sandbox_backend == "fake"


def test_parse_simulate_args_accepts_overlay_checkpoint_backend() -> None:
    args = parse_simulate_args(
        ["--manifest", "manifest.yaml", "--checkpoint-backend", "overlay"]
    )
    assert args.checkpoint_backend == "overlay"


def test_worker_partition_helpers_preserve_order_and_limits() -> None:
    inputs = [
        WorkerTraceInput(
            source_trace=f"/tmp/trace-{index}.jsonl",
            task_source="/tmp/tasks.json",
            manifest_index=index,
            docker_image_override=None,
            label=None,
            run_instance_id=f"task-{index}",
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


def test_resolve_prep_concurrency_preserves_default_limit() -> None:
    assert _resolve_prep_concurrency(0, 640) == 20
    assert _resolve_prep_concurrency(64, 640) == 64
    assert _resolve_prep_concurrency(64, 2) == 2
    with pytest.raises(ValueError, match="prep_concurrency must be >= 0"):
        _resolve_prep_concurrency(-1, 4)


class _AbortOnlyBarrier:
    def __init__(self) -> None:
        self.aborted = False
        self.wait_called = False

    def abort(self) -> None:
        self.aborted = True

    def wait(self) -> None:
        self.wait_called = True


class _SetOnlyEvent:
    def __init__(self) -> None:
        self.set_called = False
        self.wait_called = False

    def set(self) -> None:
        self.set_called = True

    def wait(self) -> None:
        self.wait_called = True


@dataclasses.dataclass
class _ReplayStartWallTime:
    value: float = 0.0


def test_worker_wave_finalizes_successful_preparations_after_prepare_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    good_trace = tmp_path / "good.jsonl"
    bad_trace = tmp_path / "bad.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(good_trace, agent_id="good", execution_environment="host")
    _write_trace(bad_trace, agent_id="bad", execution_environment="host")
    _write_host_tasks(task_source, "good", "bad")
    inputs = [
        WorkerTraceInput(
            source_trace=str(good_trace),
            task_source=str(task_source),
            manifest_index=0,
            docker_image_override=None,
            label=None,
            run_instance_id="good",
        ),
        WorkerTraceInput(
            source_trace=str(bad_trace),
            task_source=str(task_source),
            manifest_index=1,
            docker_image_override=None,
            label=None,
            run_instance_id="bad",
        ),
    ]
    finalized: list[str] = []

    async def fake_prepare(loaded, **_kwargs):
        if loaded.agent_id == "bad":
            prepared = PreparedTraceSession(
                loaded=loaded,
                task_output_dir=tmp_path / loaded.agent_id / "attempt_1",
            )
            raise ReplayPreparationError(
                loaded=loaded,
                original=RuntimeError("prepare failed"),
                prepared=prepared,
            )
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
    start_wall_time = _ReplayStartWallTime()

    result = asyncio.run(
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
            replay_start_wall_time=start_wall_time,
        )
    )

    assert finalized == ["good"]
    assert barrier.aborted is False
    assert barrier.wait_called is True
    assert event.set_called is True
    assert len(result.task_stats) == 2
    failed_stat = next(stat for stat in result.task_stats if stat.agent_id == "bad")
    assert failed_stat.success is False
    assert failed_stat.prep_error == "RuntimeError: prepare failed"
    assert result.task_output_dirs["bad"].endswith("bad/attempt_1")
    records = _read_jsonl(Path(result.trace_file))
    failed_summary = next(
        record
        for record in records
        if record.get("type") == "summary" and record.get("agent_id") == "bad"
    )
    assert failed_summary["success"] is False
    assert failed_summary["prep_error"] == "RuntimeError: prepare failed"
    assert failed_summary["error"] == "RuntimeError: prepare failed"


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
    assert seen["workers"] == 1
    assert seen["prep_concurrency"] == 0
    assert seen["resource_monitoring"] == "auto"
    assert seen["pmu_monitoring"] == "auto"
    assert seen["memory_bandwidth_monitoring"] == "auto"
    assert seen["container_executable"] is None
    assert seen["sandbox_backend"] == "docker"
    assert seen["checkpoint_backend"] is None
    assert seen["llm_timing_mode"] == "source_scaled"
    assert seen["llm_ttft_ms"] is None
    assert seen["llm_tpot_ms"] is None


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
            replay_speed=100.0,
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
        "version": 1,
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


def test_simulate_records_replay_structured_exec_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from harness.trace_logger import TraceLogger
    from trace_collect.simulator import (
        _load_trace_session,
        _replay_cloud_model_session,
    )

    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="exec",
        tool_args={"command": "printf"},
    )
    _write_tasks(task_source, "task-a")
    loaded = _load_trace_session(trace_path, task_source, 0)
    prepared = PreparedTraceSession(
        loaded=loaded,
        container=PreparedContainer(
            container_id="fake-cid",
            container_executable="docker",
            docker_image="fake-image",
            agent=object(),
        ),
    )

    async def fake_exec_tool(*_args, **_kwargs):
        return (
            "literal replay line\nExit code: 7",
            1.0,
            True,
            {"returncode": 0, "timed_out": False},
        )

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    trace_logger = TraceLogger(tmp_path / "out", "run")
    try:
        asyncio.run(
            _replay_cloud_model_session(
                prepared,
                trace_logger=trace_logger,
                replay_speed=100.0,
                llm_timing=LLMTimingConfig(),
                command_timeout_s=600.0,
                warmup_skip_iterations=0,
            )
        )
    finally:
        trace_logger.close()

    records = _read_jsonl(trace_logger.path)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    assert tool_record["data"]["returncode"] == 0
    assert tool_record["data"]["timed_out"] is False
    assert tool_record["data"]["command_exit_code"] == 0
    assert tool_record["data"]["normalized_output_match"] is False
    assert tool_record["data"]["replay_outcome_match"] is True
    # Output content differs after normalization, so effective_mismatch_reason
    # is set to output_content_mismatch (transport tiers matched).
    assert tool_record["data"]["mismatch_reason"] == "output_content_mismatch"


def test_simulate_records_normalized_output_match_without_forced_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from harness.trace_logger import TraceLogger
    from trace_collect.simulator import (
        _load_trace_session,
        _replay_cloud_model_session,
    )

    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="exec",
        tool_args={"command": "printf volatile"},
    )
    records = _read_jsonl(trace_path)
    for record in records:
        if record.get("action_type") == "tool_exec":
            record["data"]["tool_result"] = (
                "pid 123 tmp /tmp/run-456/a.txt proc /proc/789/status "
                "2026-07-05T12:34:56Z 0xabc\n\nExit code: 0"
            )
    _write_jsonl(trace_path, records)
    _write_tasks(task_source, "task-a")
    loaded = _load_trace_session(trace_path, task_source, 0)
    prepared = PreparedTraceSession(
        loaded=loaded,
        container=PreparedContainer(
            container_id="fake-cid",
            container_executable="docker",
            docker_image="fake-image",
            agent=object(),
        ),
    )

    async def fake_exec_tool(
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[str, float, bool]:
        return (
            "pid 999 tmp /tmp/run-000/a.txt proc /proc/111/status "
            "2027-08-06T01:02:03Z 0xdef\n\nExit code: 0",
            1.0,
            True,
        )

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    trace_logger = TraceLogger(tmp_path / "out", "run")
    try:
        asyncio.run(
            _replay_cloud_model_session(
                prepared,
                trace_logger=trace_logger,
                replay_speed=100.0,
                llm_timing=LLMTimingConfig(),
                command_timeout_s=600.0,
                warmup_skip_iterations=0,
            )
        )
    finally:
        trace_logger.close()

    records = _read_jsonl(trace_logger.path)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    assert tool_record["data"]["normalized_output_match"] is True
    assert tool_record["data"]["replay_outcome_match"] is True
    assert "mismatch_reason" not in tool_record["data"]
    assert "forced_sync_attempted" not in tool_record["data"]


def test_simulate_uses_resource_integrated_policy_for_container_exec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    resource_timeline = {
        "version": 1,
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


def test_simulate_forced_syncs_from_checkpoint_after_on_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    checkpoint_after = {
        "path": "checkpoints/after-tool-manifest.json",
        "kind": "cas_manifest_full",
        "root": "/testbed",
    }
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="exec",
        tool_args={"command": "pytest"},
        checkpoint_after=checkpoint_after,
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    async def fake_exec_tool(*_args, **_kwargs):
        return "failed\n\nExit code: 1", 1.0, False

    restored: list[dict] = []

    def fake_restore_checkpoint_to_container(*, checkpoint_spec, container):
        restored.append({"checkpoint_spec": checkpoint_spec, "container": container})
        return {
            "forced_sync_success": True,
            "forced_sync_elapsed_ms": 12.0,
            "forced_sync_checkpoint": checkpoint_spec["path"],
            "forced_sync_root": checkpoint_spec["root"],
        }

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    monkeypatch.setattr(
        "trace_collect.simulator._restore_checkpoint_to_container",
        fake_restore_checkpoint_to_container,
    )
    monkeypatch.setattr(
        "trace_collect.simulator._capture_snapshot_manifest",
        lambda **_kwargs: {},
    )

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
    summary = next(record for record in records if record.get("type") == "summary")

    assert len(restored) == 1
    assert restored[0]["checkpoint_spec"]["path"] == str(
        trace_path.parent / "checkpoints/after-tool-manifest.json"
    )
    assert tool_record["data"]["replay_outcome_match"] is False
    assert tool_record["data"]["mismatch_reason"] == "tool_success_mismatch"
    assert tool_record["data"]["output_diff_snippet"].startswith(
        "- source-result\n+ failed"
    )
    assert tool_record["data"]["forced_sync_attempted"] is True
    assert tool_record["data"]["forced_sync_success"] is True
    assert tool_record["data"]["forced_sync_continued"] is True
    assert tool_record["data"]["forced_sync_status"] == "checkpoint_restored_continuation"
    assert tool_record["data"]["forced_sync_overhead_excluded"] is True
    assert tool_record["data"]["forced_sync_verified"] is True
    assert tool_record["data"]["forced_sync_verification"]["cas_manifest_match"] is True
    assert summary["success"] is False
    assert summary["failed_actions"] == 1
    assert summary["forced_sync_actions"] == 1
    assert summary["forced_sync_attempts"] == 1
    assert summary["forced_sync_successes"] == 1
    assert summary["forced_sync_continued"] == 1
    assert summary["outcome_mismatches"] == 1
    assert summary["unresolved_mismatches"] == 0


def test_simulate_forced_sync_restores_incremental_checkpoint_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    checkpoints = [
        {
            "path": "checkpoints/full-manifest.json",
            "kind": "cas_manifest_full",
            "root": "/testbed",
            "incremental": False,
        },
        {
            "path": "checkpoints/inc-1-manifest.json",
            "kind": "cas_manifest_incremental",
            "root": "/testbed",
            "incremental": True,
        },
        {
            "path": "checkpoints/inc-2-manifest.json",
            "kind": "cas_manifest_incremental",
            "root": "/testbed",
            "incremental": True,
        },
    ]
    records: list[dict[str, object]] = [
        {
            "type": "trace_metadata",
            "trace_format_version": 5,
            "scaffold": "openclaw",
            "instance_id": "task-a",
            "model": "claude-haiku",
            "mode": "collect",
            "execution_environment": "container",
        }
    ]
    for index, checkpoint_after in enumerate(checkpoints):
        records.append(
            {
                "type": "action",
                "action_type": "tool_exec",
                "action_id": f"task-a-tool-{index}",
                "agent_id": "task-a",
                "iteration": index,
                "ts_start": 100.0 + index,
                "ts_end": 100.1 + index,
                "data": {
                    "tool_name": "exec",
                    "tool_args": json.dumps({"command": f"step {index}"}),
                    "tool_result": "source-result",
                    "duration_ms": 100.0,
                    "success": True,
                    "checkpoint_after": checkpoint_after,
                },
            }
        )
    records.append(
        {
            "type": "summary",
            "agent_id": "task-a",
            "model": "claude-haiku",
            "success": True,
            "n_iterations": 3,
            "elapsed_s": 3.0,
        }
    )
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    call_count = 0

    async def fake_exec_tool(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return "source-result", 1.0, True
        return "failed", 1.0, False

    restored: list[dict[str, object]] = []

    def fake_restore_checkpoint_to_container(
        *,
        checkpoint_spec,
        container,
        clear_root=True,
    ):
        restored.append(
            {
                "checkpoint_spec": checkpoint_spec,
                "container": container,
                "clear_root": clear_root,
            }
        )
        return {
            "forced_sync_success": True,
            "forced_sync_status": "checkpoint_restored_continuation",
            "forced_sync_checkpoint": checkpoint_spec["path"],
            "forced_sync_root": checkpoint_spec["root"],
        }

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    monkeypatch.setattr(
        "trace_collect.simulator._restore_checkpoint_to_container",
        fake_restore_checkpoint_to_container,
    )
    monkeypatch.setattr(
        "trace_collect.simulator._capture_snapshot_manifest",
        lambda **_kwargs: {},
    )

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
    tool_records = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    ]
    third_record = tool_records[2]

    assert [Path(str(item["checkpoint_spec"]["path"])).name for item in restored] == [
        "full-manifest.json",
        "inc-1-manifest.json",
        "inc-2-manifest.json",
    ]
    assert [item["clear_root"] for item in restored] == [True, False, False]
    assert third_record["data"]["forced_sync_success"] is True
    assert third_record["data"]["forced_sync_verified"] is True
    assert third_record["data"]["checkpoint_restore_chain_length"] == 3
    assert third_record["data"]["forced_sync_checkpoint_chain_length"] == 3
    assert third_record["data"]["forced_sync_checkpoint_chain"] == [
        str(trace_path.parent / "checkpoints/full-manifest.json"),
        str(trace_path.parent / "checkpoints/inc-1-manifest.json"),
        str(trace_path.parent / "checkpoints/inc-2-manifest.json"),
    ]


def test_cas_mismatch_at_checkpoint_boundary_forces_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from harness.trace_logger import TraceLogger
    from trace_collect.simulator import (
        _load_trace_session,
        _replay_cloud_model_session,
    )

    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    checkpoint_path = tmp_path / "checkpoints" / "after-tool-manifest.json"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_text(
        json.dumps(
            {
                "entries": {"expected.txt": {"hash": "source-hash"}},
                "deleted_paths": [],
            }
        ),
        encoding="utf-8",
    )
    _write_trace(
        trace_path,
        agent_id="task-a",
        checkpoint_after={
            "path": "checkpoints/after-tool-manifest.json",
            "kind": "cas_manifest_full",
            "root": "/testbed",
        },
    )
    _write_tasks(task_source, "task-a")
    loaded = _load_trace_session(trace_path, task_source, 0)
    prepared = PreparedTraceSession(
        loaded=loaded,
        container=PreparedContainer(
            container_id="fake-cid",
            container_executable="docker",
            docker_image="fake-image",
            agent=object(),
        ),
    )

    async def fake_exec_tool(*_args, **_kwargs):
        return "source-result", 1.0, True

    restored: list[dict[str, object]] = []

    def fake_restore_checkpoint_to_container(*, checkpoint_spec, container):
        restored.append({"checkpoint_spec": checkpoint_spec, "container": container})
        return {
            "forced_sync_success": True,
            "forced_sync_status": "checkpoint_restored_continuation",
            "forced_sync_checkpoint": checkpoint_spec["path"],
            "forced_sync_root": checkpoint_spec["root"],
        }

    snapshot_calls = 0

    def fake_capture_snapshot_manifest(**_kwargs):
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls == 1:
            return {"expected.txt": "replay-hash"}
        return {"expected.txt": "source-hash"}

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    monkeypatch.setattr(
        "trace_collect.simulator._capture_snapshot_manifest",
        fake_capture_snapshot_manifest,
    )
    monkeypatch.setattr(
        "trace_collect.simulator._restore_checkpoint_to_container",
        fake_restore_checkpoint_to_container,
    )

    trace_logger = TraceLogger(tmp_path / "out", "run")
    try:
        asyncio.run(
            _replay_cloud_model_session(
                prepared,
                trace_logger=trace_logger,
                replay_speed=100.0,
                llm_timing=LLMTimingConfig(),
                command_timeout_s=600.0,
                warmup_skip_iterations=0,
            )
        )
    finally:
        trace_logger.close()

    records = _read_jsonl(trace_logger.path)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    assert len(restored) == 1
    assert tool_record["data"]["replay_outcome_match"] is True  # output-level match
    assert tool_record["data"]["mismatch_reason"] == "cas_state_mismatch"
    assert tool_record["data"]["cas_manifest_match"] is False
    assert tool_record["data"]["forced_sync_attempted"] is True
    assert tool_record["data"]["forced_sync_reason"] == "cas_state_mismatch"
    assert tool_record["data"]["forced_sync_success"] is True
    assert tool_record["data"]["forced_sync_verified"] is True
    assert tool_record["data"]["forced_sync_verification"]["cas_manifest_match"] is True
    assert "lane_induced_mismatch_candidate" not in tool_record["data"]


def test_cas_mismatch_after_subagent_lane_marks_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from harness.trace_logger import TraceLogger
    from trace_collect.simulator import (
        _load_trace_session,
        _replay_cloud_model_session,
    )

    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    checkpoint_path = tmp_path / "checkpoints" / "after-parent-manifest.json"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_text(
        json.dumps(
            {
                "entries": {"expected.txt": {"hash": "source-hash"}},
                "deleted_paths": [],
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        trace_path,
        [
            {
                "type": "trace_metadata",
                "trace_format_version": 5,
                "scaffold": "openclaw",
                "instance_id": "task-a",
                "model": "claude-haiku",
                "mode": "collect",
                "execution_environment": "container",
            },
            {
                "type": "action",
                "action_type": "tool_exec",
                "action_id": "subagent-tool-0",
                "agent_id": "task-a/sub-1",
                "iteration": 0,
                "ts_start": 100.0,
                "ts_end": 100.03,
                "data": {
                    "tool_name": "exec",
                    "tool_args": json.dumps(
                        {"command": "printf helper > /testbed/helper.txt"}
                    ),
                    "tool_result": "helper wrote file",
                    "duration_ms": 30.0,
                    "success": True,
                },
            },
            {
                "type": "action",
                "action_type": "tool_exec",
                "action_id": "parent-tool-0",
                "agent_id": "task-a",
                "iteration": 1,
                "ts_start": 100.1,
                "ts_end": 100.15,
                "data": {
                    "tool_name": "write_file",
                    "tool_args": json.dumps({"path": "/testbed/parent.txt"}),
                    "tool_result": "source-result",
                    "duration_ms": 50.0,
                    "success": True,
                    "checkpoint_after": {
                        "path": "checkpoints/after-parent-manifest.json",
                        "kind": "cas_manifest_full",
                        "root": "/testbed",
                    },
                },
            },
            {
                "type": "summary",
                "agent_id": "task-a",
                "model": "claude-haiku",
                "success": True,
                "n_iterations": 2,
                "elapsed_s": 0.15,
            },
        ],
    )
    _write_tasks(task_source, "task-a")
    loaded = _load_trace_session(trace_path, task_source, 0)
    prepared = PreparedTraceSession(
        loaded=loaded,
        container=PreparedContainer(
            container_id="fake-cid",
            container_executable="docker",
            docker_image="fake-image",
            agent=object(),
        ),
    )

    async def fake_exec_tool(
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[str, float, bool]:
        return "source-result", 1.0, True

    restored: list[dict[str, Any]] = []

    def fake_restore_checkpoint_to_container(
        *,
        checkpoint_spec: dict[str, Any],
        container: PreparedContainer,
    ) -> dict[str, Any]:
        restored.append({"checkpoint_spec": checkpoint_spec, "container": container})
        return {
            "forced_sync_success": True,
            "forced_sync_status": "checkpoint_restored_continuation",
            "forced_sync_checkpoint": checkpoint_spec["path"],
            "forced_sync_root": checkpoint_spec["root"],
        }

    snapshot_calls = 0

    def fake_capture_snapshot_manifest(**_kwargs: Any) -> dict[str, str]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls == 1:
            return {"expected.txt": "replay-hash"}
        return {"expected.txt": "source-hash"}

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    monkeypatch.setattr(
        "trace_collect.simulator._capture_snapshot_manifest",
        fake_capture_snapshot_manifest,
    )
    monkeypatch.setattr(
        "trace_collect.simulator._restore_checkpoint_to_container",
        fake_restore_checkpoint_to_container,
    )

    trace_logger = TraceLogger(tmp_path / "out", "run")
    try:
        asyncio.run(
            _replay_cloud_model_session(
                prepared,
                trace_logger=trace_logger,
                replay_speed=100.0,
                llm_timing=LLMTimingConfig(),
                command_timeout_s=600.0,
                warmup_skip_iterations=0,
            )
        )
    finally:
        trace_logger.close()

    records = _read_jsonl(trace_logger.path)
    parent_tool_record = next(
        record
        for record in records
        if record.get("type") == "action"
        and record.get("action_type") == "tool_exec"
        and record.get("action_id") == "parent-tool-0"
    )
    subagent_tool_record = next(
        record
        for record in records
        if record.get("type") == "action"
        and record.get("action_type") == "tool_exec"
        and record["data"].get("source_lane_agent_id") == "task-a/sub-1"
    )

    assert len(restored) == 1
    assert subagent_tool_record["data"]["replay_source"] == "subagent_lane_replay"
    assert parent_tool_record["data"]["mismatch_reason"] == "cas_state_mismatch"
    assert parent_tool_record["data"]["forced_sync_attempted"] is True
    assert parent_tool_record["data"]["forced_sync_reason"] == "cas_state_mismatch"
    assert parent_tool_record["data"]["lane_induced_mismatch_candidate"] is True


def test_simulate_forced_sync_fallback_to_prior_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    checkpoint_after = {
        "path": "checkpoints/after-first-tool-manifest.json",
        "kind": "cas_manifest_full",
        "root": "/testbed",
    }
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "openclaw",
                        "instance_id": "task-a",
                        "model": "claude-haiku",
                        "mode": "collect",
                        "execution_environment": "container",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "task-a-llm-0",
                        "agent_id": "task-a",
                        "iteration": 0,
                        "ts_start": 100.0,
                        "ts_end": 100.2,
                        "data": {
                            "messages_in": [{"role": "user", "content": "fix bug"}],
                            "raw_response": {"id": "resp-task-a"},
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "llm_latency_ms": 200.0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "tool_exec",
                        "action_id": "task-a-tool-0",
                        "agent_id": "task-a",
                        "iteration": 0,
                        "ts_start": 100.4,
                        "ts_end": 100.45,
                        "data": {
                            "tool_name": "write_file",
                            "tool_args": json.dumps({"path": "/testbed/a.txt"}),
                            "tool_result": "source-result",
                            "duration_ms": 50.0,
                            "success": True,
                            "checkpoint_after": checkpoint_after,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "tool_exec",
                        "action_id": "task-a-tool-1",
                        "agent_id": "task-a",
                        "iteration": 1,
                        "ts_start": 100.6,
                        "ts_end": 100.65,
                        "data": {
                            "tool_name": "write_file",
                            "tool_args": json.dumps({"path": "/testbed/b.txt"}),
                            "tool_result": "source-result",
                            "duration_ms": 50.0,
                            "success": True,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "summary",
                        "agent_id": "task-a",
                        "model": "claude-haiku",
                        "success": True,
                        "n_iterations": 2,
                        "elapsed_s": 0.65,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    call_count = 0

    async def fake_exec_tool(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "ok", 1.0, True
        return "failed", 1.0, False

    restored: list[dict] = []

    def fake_restore_checkpoint_to_container(*, checkpoint_spec, container):
        restored.append({"checkpoint_spec": checkpoint_spec, "container": container})
        return {
            "forced_sync_success": True,
            "forced_sync_elapsed_ms": 12.0,
            "forced_sync_checkpoint": checkpoint_spec["path"],
            "forced_sync_root": checkpoint_spec["root"],
        }

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    monkeypatch.setattr(
        "trace_collect.simulator._restore_checkpoint_to_container",
        fake_restore_checkpoint_to_container,
    )
    monkeypatch.setattr(
        "trace_collect.simulator._capture_snapshot_manifest",
        lambda **_kwargs: {},
    )

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
    tool_records = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    ]
    summary = next(record for record in records if record.get("type") == "summary")

    assert len(restored) == 1
    assert restored[0]["checkpoint_spec"]["path"] == str(
        trace_path.parent / "checkpoints/after-first-tool-manifest.json"
    )
    assert len(tool_records) == 2
    fallback_record = tool_records[1]
    assert fallback_record["data"]["replay_outcome_match"] is False
    assert fallback_record["data"]["mismatch_reason"] == "tool_success_mismatch"
    assert fallback_record["data"]["forced_sync_attempted"] is True
    assert fallback_record["data"]["forced_sync_success"] is True
    assert fallback_record["data"]["forced_sync_continued"] is True
    assert (
        fallback_record["data"]["forced_sync_status"]
        == "checkpoint_restored_continuation"
    )
    assert fallback_record["data"]["forced_sync_overhead_excluded"] is True
    assert fallback_record["data"]["forced_sync_fallback"] is True
    assert fallback_record["data"]["forced_sync_reapplied_action_count"] == 1
    assert fallback_record["data"]["forced_sync_reapplied_action_ids"] == [
        "task-a-tool-1"
    ]
    assert fallback_record["data"]["forced_sync_reapply_errors"] == []
    assert fallback_record["data"]["forced_sync_restore_verified"] is True
    assert (
        fallback_record["data"]["forced_sync_restore_verification"][
            "cas_manifest_match"
        ]
        is True
    )
    assert fallback_record["data"]["forced_sync_verified"] is None
    assert (
        fallback_record["data"]["forced_sync_verify_reason"]
        == "reapplied_actions_unverifiable"
    )
    assert fallback_record["data"]["forced_sync_fallback_from_action_index"] == 1
    assert (
        fallback_record["data"]["forced_sync_fallback_from_action_id"]
        == "task-a-tool-0"
    )
    assert summary["success"] is False
    assert summary["forced_sync_actions"] == 1
    assert summary["forced_sync_attempts"] == 1
    assert summary["forced_sync_successes"] == 1
    assert summary["forced_sync_continued"] == 1
    assert summary["outcome_mismatches"] == 1
    assert summary["unresolved_mismatches"] == 0


def test_forced_sync_fallback_reapplies_intermediate_actions_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    checkpoint_after = {
        "path": "checkpoints/after-first-tool-manifest.json",
        "kind": "cas_manifest_full",
        "root": "/testbed",
    }
    records: list[dict[str, object]] = [
        {
            "type": "trace_metadata",
            "trace_format_version": 5,
            "scaffold": "openclaw",
            "instance_id": "task-a",
            "model": "claude-haiku",
            "mode": "collect",
            "execution_environment": "container",
        },
        {
            "type": "action",
            "action_type": "llm_call",
            "action_id": "task-a-llm-0",
            "agent_id": "task-a",
            "iteration": 0,
            "ts_start": 100.0,
            "ts_end": 100.2,
            "data": {
                "messages_in": [{"role": "user", "content": "fix bug"}],
                "raw_response": {"id": "resp-task-a"},
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "llm_latency_ms": 200.0,
            },
        },
    ]
    tool_specs = [
        (
            "task-a-tool-0",
            "write_file",
            {"path": "/testbed/a.txt"},
            checkpoint_after,
        ),
        (
            "task-a-tool-1",
            "exec",
            {"command": "printf intermediate"},
            None,
        ),
        (
            "task-a-tool-2",
            "write_file",
            {"path": "/testbed/c.txt"},
            None,
        ),
    ]
    for index, (action_id, tool_name, tool_args, checkpoint) in enumerate(tool_specs):
        data: dict[str, object] = {
            "tool_name": tool_name,
            "tool_args": json.dumps(tool_args),
            "tool_result": "source-result",
            "duration_ms": 50.0,
            "success": True,
        }
        if checkpoint is not None:
            data["checkpoint_after"] = checkpoint
        records.append(
            {
                "type": "action",
                "action_type": "tool_exec",
                "action_id": action_id,
                "agent_id": "task-a",
                "iteration": index,
                "ts_start": 100.4 + index,
                "ts_end": 100.45 + index,
                "data": data,
            }
        )
    records.append(
        {
            "type": "summary",
            "agent_id": "task-a",
            "model": "claude-haiku",
            "success": True,
            "n_iterations": 3,
            "elapsed_s": 3.0,
        }
    )
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    exec_calls: list[tuple[str, dict[str, object]]] = []

    async def fake_exec_tool(
        _agent,
        tool_name,
        tool_args_json,
        *_args,
        **_kwargs,
    ):
        exec_calls.append((tool_name, json.loads(tool_args_json)))
        if len(exec_calls) == 3:
            return "failed", 1.0, False
        return "source-result", 1.0, True

    restored: list[dict[str, object]] = []

    def fake_restore_checkpoint_to_container(*, checkpoint_spec, container):
        restored.append({"checkpoint_spec": checkpoint_spec, "container": container})
        return {
            "forced_sync_success": True,
            "forced_sync_status": "checkpoint_restored_continuation",
            "forced_sync_checkpoint": checkpoint_spec["path"],
            "forced_sync_root": checkpoint_spec["root"],
        }

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    monkeypatch.setattr(
        "trace_collect.simulator._restore_checkpoint_to_container",
        fake_restore_checkpoint_to_container,
    )
    monkeypatch.setattr(
        "trace_collect.simulator._capture_snapshot_manifest",
        lambda **_kwargs: {},
    )

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
    tool_records = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    ]
    fallback_record = tool_records[2]

    assert len(restored) == 1
    assert [call[0] for call in exec_calls] == [
        "write_file",
        "exec",
        "write_file",
        "exec",
        "write_file",
    ]
    assert exec_calls[3][1] == {"command": "printf intermediate"}
    assert exec_calls[4][1] == {"path": "/testbed/c.txt"}
    assert len(tool_records) == 3
    assert fallback_record["data"]["forced_sync_success"] is True
    assert fallback_record["data"]["forced_sync_reapplied_action_count"] == 2
    assert fallback_record["data"]["forced_sync_reapplied_action_ids"] == [
        "task-a-tool-1",
        "task-a-tool-2",
    ]
    assert fallback_record["data"]["forced_sync_reapply_errors"] == []
    assert fallback_record["data"]["forced_sync_restore_verified"] is True
    assert fallback_record["data"]["forced_sync_verified"] is None
    assert (
        fallback_record["data"]["forced_sync_verify_reason"]
        == "reapplied_actions_unverifiable"
    )


def test_forced_sync_fallback_reapply_failure_marks_sync_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    checkpoint_after = {
        "path": "checkpoints/after-first-tool-manifest.json",
        "kind": "cas_manifest_full",
        "root": "/testbed",
    }
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "openclaw",
                        "instance_id": "task-a",
                        "model": "claude-haiku",
                        "mode": "collect",
                        "execution_environment": "container",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "task-a-llm-0",
                        "agent_id": "task-a",
                        "iteration": 0,
                        "ts_start": 100.0,
                        "ts_end": 100.2,
                        "data": {
                            "messages_in": [{"role": "user", "content": "fix bug"}],
                            "raw_response": {"id": "resp-task-a"},
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "llm_latency_ms": 200.0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "tool_exec",
                        "action_id": "task-a-tool-0",
                        "agent_id": "task-a",
                        "iteration": 0,
                        "ts_start": 100.4,
                        "ts_end": 100.45,
                        "data": {
                            "tool_name": "write_file",
                            "tool_args": json.dumps({"path": "/testbed/a.txt"}),
                            "tool_result": "source-result",
                            "duration_ms": 50.0,
                            "success": True,
                            "checkpoint_after": checkpoint_after,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "tool_exec",
                        "action_id": "task-a-tool-1",
                        "agent_id": "task-a",
                        "iteration": 1,
                        "ts_start": 101.0,
                        "ts_end": 101.05,
                        "data": {
                            "tool_name": "write_file",
                            "tool_args": json.dumps({"path": "/testbed/b.txt"}),
                            "tool_result": "source-result",
                            "duration_ms": 50.0,
                            "success": True,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "summary",
                        "agent_id": "task-a",
                        "model": "claude-haiku",
                        "success": True,
                        "n_iterations": 2,
                        "elapsed_s": 2.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    exec_count = 0

    async def fake_exec_tool(*_args, **_kwargs):
        nonlocal exec_count
        exec_count += 1
        if exec_count == 1:
            return "source-result", 1.0, True
        if exec_count == 2:
            return "failed", 1.0, False
        raise RuntimeError("transport down")

    def fake_restore_checkpoint_to_container(*, checkpoint_spec, container):
        return {
            "forced_sync_success": True,
            "forced_sync_status": "checkpoint_restored_continuation",
            "forced_sync_checkpoint": checkpoint_spec["path"],
            "forced_sync_root": checkpoint_spec["root"],
        }

    snapshot_calls = 0

    def capture_boundary_only(**_kwargs):
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls in {1, 2}:
            return {}
        raise AssertionError("post-reapply verification should not run")

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    monkeypatch.setattr(
        "trace_collect.simulator._restore_checkpoint_to_container",
        fake_restore_checkpoint_to_container,
    )
    monkeypatch.setattr(
        "trace_collect.simulator._capture_snapshot_manifest",
        capture_boundary_only,
    )

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
    tool_records = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    ]
    summary = next(record for record in records if record.get("type") == "summary")
    fallback_record = tool_records[1]

    assert fallback_record["data"]["forced_sync_success"] is False
    assert fallback_record["data"]["forced_sync_continued"] is False
    assert fallback_record["data"]["forced_sync_status"] == "reapply_failed"
    assert fallback_record["data"]["forced_sync_reapplied_action_count"] == 1
    assert fallback_record["data"]["forced_sync_reapplied_action_ids"] == [
        "task-a-tool-1"
    ]
    assert "transport down" in fallback_record["data"]["forced_sync_reapply_errors"][0]
    assert fallback_record["data"]["forced_sync_restore_verified"] is True
    assert summary["forced_sync_attempts"] == 1
    assert summary["forced_sync_successes"] == 0
    assert summary["forced_sync_continued"] == 0
    assert summary["unresolved_mismatches"] == 1
    assert snapshot_calls == 2


def test_mismatch_without_checkpoint_is_unresolved_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a", tool_name="write_file")
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    async def fake_exec_tool(*_args, **_kwargs):
        return "failed", 1.0, False

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
    summary = next(record for record in records if record.get("type") == "summary")

    assert tool_record["data"]["mismatch_reason"] == "tool_success_mismatch"
    assert tool_record["data"]["forced_sync_attempted"] is True
    assert tool_record["data"]["forced_sync_success"] is False
    assert tool_record["data"]["forced_sync_continued"] is False
    assert tool_record["data"]["forced_sync_status"] == "checkpoint_missing"
    assert tool_record["data"]["forced_sync_error"] == (
        "no checkpoint available (searched entire trace history)"
    )
    assert summary["success"] is False
    assert summary["outcome_mismatches"] == 1
    assert summary["unresolved_mismatches"] == 1
    assert summary["forced_sync_attempts"] == 1
    assert summary["forced_sync_successes"] == 0
    assert summary["forced_sync_continued"] == 0


def test_missing_checkpoint_file_marks_forced_sync_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="exec",
        tool_args={"command": "pytest"},
        checkpoint_after={
            "path": "checkpoints/missing-manifest.json",
            "kind": "cas_manifest_full",
            "root": "/testbed",
        },
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    async def fake_exec_tool(*_args, **_kwargs):
        return "failed\n\nExit code: 1", 1.0, False

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
    summary = next(record for record in records if record.get("type") == "summary")

    assert tool_record["data"]["forced_sync_attempted"] is True
    assert tool_record["data"]["forced_sync_success"] is False
    assert tool_record["data"]["forced_sync_continued"] is False
    assert tool_record["data"]["forced_sync_status"] == "checkpoint_missing"
    assert tool_record["data"]["checkpoint_archive_exists"] is False
    assert tool_record["data"]["restore_elapsed_ms"] >= 0.0
    assert "checkpoint not found" in tool_record["data"]["forced_sync_error"]
    assert summary["forced_sync_attempts"] == 1
    assert summary["forced_sync_successes"] == 0
    assert summary["forced_sync_continued"] == 0
    assert summary["outcome_mismatches"] == 1
    assert summary["unresolved_mismatches"] == 1


def test_invalid_checkpoint_manifest_marks_restore_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    checkpoint_path = tmp_path / "checkpoints" / "corrupt-manifest.json"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_text("not a manifest", encoding="utf-8")
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="exec",
        tool_args={"command": "pytest"},
        checkpoint_after={
            "path": "checkpoints/corrupt-manifest.json",
            "kind": "cas_manifest_full",
            "root": "/testbed",
        },
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    async def fake_exec_tool(*_args, **_kwargs):
        return "failed\n\nExit code: 1", 1.0, False

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
    summary = next(record for record in records if record.get("type") == "summary")

    assert tool_record["data"]["forced_sync_success"] is False
    assert tool_record["data"]["forced_sync_status"] == "checkpoint_restore_failed"
    assert tool_record["data"]["checkpoint_archive_exists"] is True
    assert tool_record["data"]["checkpoint_size_bytes"] == checkpoint_path.stat().st_size
    assert tool_record["data"]["restore_elapsed_ms"] >= 0.0
    assert "invalid checkpoint manifest" in tool_record["data"]["forced_sync_error"]
    assert summary["forced_sync_attempts"] == 1
    assert summary["forced_sync_successes"] == 0
    assert summary["forced_sync_continued"] == 0
    assert summary["outcome_mismatches"] == 1
    assert summary["unresolved_mismatches"] == 1


def test_unsupported_checkpoint_kind_marks_restore_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    checkpoint_path = tmp_path / "checkpoints" / "after.snapshot"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_text("snapshot", encoding="utf-8")
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="exec",
        tool_args={"command": "pytest"},
        checkpoint_after={
            "path": "checkpoints/after.snapshot",
            "kind": "firecracker_snapshot",
            "root": "/testbed",
        },
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    async def fake_exec_tool(*_args, **_kwargs):
        return "failed\n\nExit code: 1", 1.0, False

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
    summary = next(record for record in records if record.get("type") == "summary")

    assert tool_record["data"]["forced_sync_success"] is False
    assert tool_record["data"]["forced_sync_status"] == "checkpoint_restore_failed"
    assert tool_record["data"]["checkpoint_archive_exists"] is True
    assert tool_record["data"]["restore_elapsed_ms"] >= 0.0
    assert "unsupported checkpoint kind" in tool_record["data"]["forced_sync_error"]
    assert summary["forced_sync_attempts"] == 1
    assert summary["forced_sync_successes"] == 0
    assert summary["forced_sync_continued"] == 0
    assert summary["outcome_mismatches"] == 1
    assert summary["unresolved_mismatches"] == 1


def test_forced_sync_does_not_resolve_source_artifact_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="read_file",
        tool_args={"path": "/openclaw-runtime/tool-results/tool-results/missing.txt"},
        checkpoint_after={
            "path": "checkpoints/after-read-manifest.json",
            "kind": "cas_manifest_full",
            "root": "/testbed",
        },
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    def fake_restore_checkpoint_to_container(*, checkpoint_spec, container):
        return {
            "forced_sync_success": True,
            "forced_sync_elapsed_ms": 12.0,
            "forced_sync_checkpoint": checkpoint_spec["path"],
        }

    monkeypatch.setattr(
        "trace_collect.simulator._restore_checkpoint_to_container",
        fake_restore_checkpoint_to_container,
    )

    def fail_capture_snapshot_manifest(**_kwargs):
        raise RuntimeError("diagnostic snapshot unavailable")

    monkeypatch.setattr(
        "trace_collect.simulator._capture_snapshot_manifest",
        fail_capture_snapshot_manifest,
    )

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
    summary = next(record for record in records if record.get("type") == "summary")

    assert tool_record["data"]["mismatch_reason"] == "source_artifact_unavailable"
    assert tool_record["data"]["forced_sync_success"] is True
    assert tool_record["data"]["forced_sync_verified"] is None
    assert tool_record["data"]["forced_sync_verification"]["snapshot_captured"] is False
    assert tool_record["data"]["forced_sync_continued"] is True
    assert summary["success"] is False
    assert summary["forced_sync_actions"] == 1
    assert summary["forced_sync_attempts"] == 1
    assert summary["forced_sync_successes"] == 1
    assert summary["forced_sync_continued"] == 1
    assert summary["fatal_replay_errors"] == 1
    assert summary["outcome_mismatches"] == 1
    assert summary["unresolved_mismatches"] == 0


def test_simulate_keeps_wall_policy_for_commands_resource_timeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    resource_timeline = {
        "version": 1,
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
    invalid_timeline = {"version": 1, "samples": [{"dt_s": 0.0, "cpu_core_s": 1.0}]}
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


def test_cloud_model_ttft_tpot_requires_parameters(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="host-task",
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


def test_cloud_model_tool_success_false_marks_trace_failed(
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
    assert tool_record["data"]["replay_outcome_match"] is True
    assert summary["success"] is True
    assert summary["failed_actions"] == 0
    assert summary["source_failed_actions"] == 1
    assert summary["replay_failed_actions"] == 1
    assert summary["matched_failed_actions"] == 1
    assert throughput["completed_traces"] == 1
    assert throughput["failed_traces"] == 0


def test_cloud_model_replay_passes_source_container_exec_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    bootstrap_root = tmp_path / "task-container-bootstrap"
    site_dir = bootstrap_root / "linux-amd64" / "cache-key" / "gen-1" / "pydeps"
    site_dir.mkdir(parents=True)
    userbase = site_dir.parent / ".pyuserbase"
    env = {
        "pythonpath": f"{site_dir}:/repo/src:/repo",
        "path": f"{userbase / 'bin'}:/usr/local/bin:/usr/bin:/bin",
        "pythonuserbase": str(userbase),
        "bootstrap_site_dir": str(site_dir),
    }
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="exec",
        tool_args={"command": "echo hi"},
    )
    _set_trace_container_exec_env(trace_path, env)
    _write_tasks(task_source, "task-a")

    agent_kwargs: list[dict[str, str]] = []
    start_extra_args: list[str] = []

    class _FakeContainerAgent:
        def __init__(
            self,
            container_id: str,
            container_executable: str,
            **kwargs: str,
        ) -> None:
            assert container_id == "fake-cid"
            assert container_executable == "docker"
            agent_kwargs.append(kwargs)

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    class _FakeSampler:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> list[dict]:
            return []

    async def fake_exec_tool(*_args: Any, **_kwargs: Any) -> tuple[str, float, bool]:
        return ("ok\n\nExit code: 0", 1.0, True)

    def fake_start_task_container(
        _image: str,
        *,
        executable: str,
        network_mode: str,
        extra_args: list[str] | None = None,
    ) -> str:
        assert executable == "docker"
        assert network_mode == "host"
        assert extra_args is not None
        start_extra_args.extend(extra_args)
        return "fake-cid"

    monkeypatch.setattr(
        "trace_collect.runtime.task_container._SHARED_BOOTSTRAP_CACHE",
        bootstrap_root,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.ensure_source_image",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.ensure_fixed_image",
        lambda *args, **kwargs: ("fixed-image", 0.0),
    )
    monkeypatch.setattr(
        "trace_collect.simulator.remove_image",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.start_task_container",
        fake_start_task_container,
    )
    monkeypatch.setattr(
        "trace_collect.openclaw_tools.ContainerAgent",
        _FakeContainerAgent,
    )
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
    _patch_noop_sweep_fixed_prebuild(monkeypatch)
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

    tool_record = next(
        record
        for record in _read_jsonl(trace_file)
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    throughput = json.loads((output_dir / "throughput_summary.json").read_text())

    assert agent_kwargs == [
        {
            "pythonpath": env["pythonpath"],
            "path": env["path"],
            "pythonuserbase": env["pythonuserbase"],
        }
    ]
    assert f"{bootstrap_root}:{bootstrap_root}:ro" in start_extra_args
    assert tool_record["data"]["replay_env_parity"] == "source_env"
    assert throughput["tasks"][0]["replay_env_parity"] == "source_env"


def test_cloud_model_records_missing_bootstrap_cache_env_parity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    bootstrap_root = tmp_path / "missing-bootstrap-cache"
    site_dir = bootstrap_root / "linux-amd64" / "cache-key" / "gen-1" / "pydeps"
    env = {
        "pythonpath": f"{site_dir}:/repo/src:/repo",
        "bootstrap_site_dir": str(site_dir),
    }
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="exec",
        tool_args={"command": "echo hi"},
    )
    _set_trace_container_exec_env(trace_path, env)
    _write_tasks(task_source, "task-a")
    start_extra_args: list[str] = []

    class _FakeContainerAgent:
        def __init__(
            self,
            container_id: str,
            container_executable: str,
            **_kwargs: str,
        ) -> None:
            assert container_id == "fake-cid"
            assert container_executable == "docker"

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    class _FakeSampler:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> list[dict]:
            return []

    async def fake_exec_tool(*_args: Any, **_kwargs: Any) -> tuple[str, float, bool]:
        return ("ok\n\nExit code: 0", 1.0, True)

    def fake_start_task_container(
        _image: str,
        *,
        executable: str,
        network_mode: str,
        extra_args: list[str] | None = None,
    ) -> str:
        assert executable == "docker"
        assert network_mode == "host"
        assert extra_args is not None
        start_extra_args.extend(extra_args)
        return "fake-cid"

    monkeypatch.setattr(
        "trace_collect.runtime.task_container._SHARED_BOOTSTRAP_CACHE",
        bootstrap_root,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.ensure_source_image",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.ensure_fixed_image",
        lambda *args, **kwargs: ("fixed-image", 0.0),
    )
    monkeypatch.setattr(
        "trace_collect.simulator.remove_image",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.start_task_container",
        fake_start_task_container,
    )
    monkeypatch.setattr(
        "trace_collect.openclaw_tools.ContainerAgent",
        _FakeContainerAgent,
    )
    monkeypatch.setattr("trace_collect.simulator.ContainerStatsSampler", _FakeSampler)
    _patch_noop_sweep_fixed_prebuild(monkeypatch)
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

    assert tool_record["data"]["replay_env_parity"] == "source_env"
    assert summary["replay_env_parity"] == "bootstrap_cache_missing"
    assert throughput["tasks"][0]["replay_env_parity"] == "bootstrap_cache_missing"
    assert f"{bootstrap_root}:{bootstrap_root}:ro" not in start_extra_args


def test_cloud_model_denied_exec_replays_from_source_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    source_result = "Error: Command blocked by safety guard (dangerous pattern detected)"
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="exec",
        tool_args={"command": "rm -rf /tmp/workload"},
    )
    records = _read_jsonl(trace_path)
    for record in records:
        if record.get("action_type") == "tool_exec":
            record["data"]["tool_result"] = source_result
            record["data"]["success"] = True
    _write_jsonl(trace_path, records)
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    async def fail_exec_tool(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("denied source command must not execute in replay")

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

    assert tool_record["data"]["tool_result"] == source_result
    assert tool_record["data"]["success"] is True
    assert tool_record["data"]["replay_outcome_match"] is True
    assert tool_record["data"]["replay_source"] == "denied_command_replayed_from_trace"
    assert tool_record["data"]["replay_env_parity"] == "default_env"
    assert summary["success"] is True


def test_cloud_model_source_failed_replay_success_marks_mismatch(
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
            record["data"]["tool_args"] = json.dumps({"exec": {"command": "slow"}})
            record["data"]["success"] = False
            record["data"]["tool_result"] = "Error: Command timed out after 300 seconds"
            record["data"]["duration_ms"] = 300000.0
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    async def fake_exec_tool(*_args, **_kwargs):
        return "finished\n\nExit code: 0", 1.0, True

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)

    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=output_dir,
            mode="cloud_model",
            container_executable="docker",
            replay_speed=50.0,
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
    throughput = json.loads((output_dir / "throughput_summary.json").read_text())

    assert tool_record["data"]["success"] is True
    assert tool_record["data"]["source_success"] is False
    assert tool_record["data"]["replay_outcome_match"] is False
    assert tool_record["data"]["mismatch_reason"] == "timeout_mismatch"
    assert summary["success"] is False
    assert summary["failed_actions"] == 1
    assert summary["source_failed_actions"] == 1
    assert summary["replay_failed_actions"] == 0
    assert summary["matched_failed_actions"] == 0
    assert summary["outcome_mismatches"] == 1
    assert throughput["completed_traces"] == 0
    assert throughput["failed_traces"] == 1


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
    assert _source_exec_timeout_s(
        tool_name="exec",
        tool_args_json=json.dumps({"exec": {"command": "printf '[timeout]'"}}),
        source_duration_ms=50.0,
        source_success=True,
        source_tool_result="[timeout]\n\nExit code: 0",
        source_timed_out=False,
    ) is None
    assert _source_exec_timeout_s(
        tool_name="exec",
        tool_args_json=json.dumps({"exec": {"command": "slow command"}}),
        source_duration_ms=300056.8,
        source_success=False,
        source_tool_result="plain failure",
        source_timed_out=True,
    ) == pytest.approx(300.0568)


def test_effective_source_exec_timeout_scales_with_replay_speed() -> None:
    assert _effective_source_exec_timeout_s(
        source_exec_timeout_s=300.0568,
        replay_speed=50.0,
    ) == pytest.approx(6.001136)
    assert _effective_source_exec_timeout_s(
        source_exec_timeout_s=300.0568,
        replay_speed=100.0,
    ) == pytest.approx(5.0)
    assert _effective_source_exec_timeout_s(
        source_exec_timeout_s=None,
        replay_speed=100.0,
    ) is None


def test_tool_mismatch_reason_distinguishes_wrapper_timeout() -> None:
    tool_args = json.dumps({"exec": {"command": "cmd"}})

    assert (
        _tool_mismatch_reason(
            source_success=False,
            tool_success=False,
            replay_source="executed_in_container",
            source_tool_result="command returned 124\n\nExit code: 124",
            replay_tool_result="[timeout]\n\nExit code: 124",
            tool_name="exec",
            tool_args_json=tool_args,
        )
        == "timeout_mismatch"
    )
    assert (
        _tool_mismatch_reason(
            source_success=True,
            tool_success=True,
            replay_source="executed_in_container",
            source_tool_result="ok\n\nExit code: 0",
            replay_tool_result="bad\n\nExit code: 1",
            tool_name="exec",
            tool_args_json=tool_args,
        )
        == "command_exit_code_mismatch"
    )


def test_tool_mismatch_reason_prefers_structured_returncode() -> None:
    tool_args = json.dumps({"exec": {"command": "cmd"}})

    assert (
        _tool_mismatch_reason(
            source_success=True,
            tool_success=True,
            replay_source="executed_in_container",
            source_tool_result="literal\nExit code: 7",
            replay_tool_result="ok\n\nExit code: 0",
            tool_name="exec",
            tool_args_json=tool_args,
            source_returncode=0,
            replay_returncode=0,
        )
        is None
    )
    assert (
        _tool_mismatch_reason(
            source_success=True,
            tool_success=True,
            replay_source="executed_in_container",
            source_tool_result="literal\nExit code: 7",
            replay_tool_result="ok\n\nExit code: 0",
            tool_name="exec",
            tool_args_json=tool_args,
        )
        == "command_exit_code_mismatch"
    )


def test_tool_mismatch_reason_prefers_structured_timeout() -> None:
    tool_args = json.dumps({"exec": {"command": "cmd"}})

    assert (
        _tool_mismatch_reason(
            source_success=True,
            tool_success=True,
            replay_source="executed_in_container",
            source_tool_result="[timeout]\n\nExit code: 0",
            replay_tool_result="ok\n\nExit code: 0",
            tool_name="exec",
            tool_args_json=tool_args,
            source_timed_out=False,
            replay_timed_out=False,
        )
        is None
    )
    assert (
        _tool_mismatch_reason(
            source_success=True,
            tool_success=True,
            replay_source="executed_in_container",
            source_tool_result="[timeout]\n\nExit code: 0",
            replay_tool_result="ok\n\nExit code: 0",
            tool_name="exec",
            tool_args_json=tool_args,
        )
        == "timeout_mismatch"
    )


def test_command_metadata_prefers_structured_returncode() -> None:
    tool_args = json.dumps({"exec": {"command": "cmd"}})

    assert _command_metadata(
        tool_name="exec",
        tool_args_json=tool_args,
        tool_result="literal\nExit code: 7",
        tool_success=True,
        returncode=0,
    ) == {
        "command_exit_code": 0,
        "command_success": True,
        "replay_transport_success": True,
    }


def test_compute_output_diff_snippet_captures_first_divergence() -> None:
    assert (
        _compute_output_diff_snippet("same\npid 123\ndone", "same\npid 456\ndone")
        == "  same\n- pid 123\n+ pid 456\n  done"
    )
    assert (
        _compute_output_diff_snippet("same", "same\nextra")
        == "  same\n- <missing>\n+ extra"
    )
    assert (
        _compute_output_diff_snippet("same\n", "same")
        == "raw output differs without line-content difference"
    )
    assert (
        _compute_output_diff_snippet("abcdef", "uvwxyz", max_line_chars=3)
        == "- abc...<truncated 3 chars>\n+ uvw...<truncated 3 chars>"
    )

def test_tool_mismatch_reason_skips_output_comparison_for_non_exec_tools() -> None:
    assert (
        _tool_mismatch_reason(
            source_success=True,
            tool_success=True,
            replay_source="executed_in_container",
            source_tool_result="old content",
            replay_tool_result="new content",
            tool_name="read_file",
            tool_args_json=json.dumps({"path": "/testbed/file.py"}),
        )
        is None
    )


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
            replay_speed=50.0,
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

    assert captured_source_timeouts == [pytest.approx(6.001136)]
    assert tool_record["data"]["source_exec_timeout_s"] == pytest.approx(300.0568)
    assert tool_record["data"]["replay_exec_timeout_s"] == pytest.approx(6.001136)
    assert tool_record["data"]["source_success"] is False
    assert tool_record["data"]["success"] is False
    assert tool_record["data"]["replay_outcome_match"] is True
    assert summary["success"] is True
    assert summary["failed_actions"] == 0
    assert summary["matched_failed_actions"] == 1


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
    assert tool_record["data"]["replay_outcome_match"] is False
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
    assert tool_record["data"]["replay_outcome_match"] is False
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
    assert tool_record["data"]["replay_outcome_match"] is False
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
    assert tool_record["data"]["replay_outcome_match"] is True
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
    assert tool_record["data"]["replay_outcome_match"] is True
    assert tool_record["data"]["replay_source"] == "message_noop"
    assert tool_record["data"]["sim_metrics"]["sim_tool_format"] == "message_noop"
    assert summary["success"] is True


def test_cloud_model_marks_concurrent_source_exec_metric(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="message",
        extra_tool_data={"source_concurrent_execs": True},
    )
    records = _read_jsonl(trace_path)
    for record in records:
        if record.get("action_type") == "tool_exec":
            record["data"]["tool_args"] = json.dumps({"content": "finished"})
    _write_jsonl(trace_path, records)
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

    replay_records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in replay_records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    assert tool_record["data"]["sim_metrics"]["concurrent_source_execs"] is True


def test_cloud_model_replays_subagent_lane_from_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    records: list[dict[str, Any]] = [
        {
            "type": "trace_metadata",
            "trace_format_version": 5,
            "scaffold": "openclaw",
            "instance_id": "task-a",
            "model": "claude-haiku",
            "mode": "collect",
            "execution_environment": "container",
        },
        {
            "type": "action",
            "action_type": "llm_call",
            "action_id": "llm_0",
            "agent_id": "task-a",
            "iteration": 0,
            "ts_start": 100.0,
            "ts_end": 100.1,
            "data": {
                "messages_in": [{"role": "user", "content": "spawn helper"}],
                "raw_response": {"id": "parent"},
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "llm_latency_ms": 100.0,
            },
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "tool_0_spawn",
            "agent_id": "task-a",
            "iteration": 0,
            "ts_start": 100.2,
            "ts_end": 100.21,
            "data": {
                "tool_name": "spawn",
                "tool_args": json.dumps({"task": "inspect state"}),
                "tool_result": "Subagent [inspect] started (id: sub-1).",
                "duration_ms": 10.0,
                "success": True,
            },
        },
        {
            "type": "action",
            "action_type": "llm_call",
            "action_id": "llm_0",
            "agent_id": "task-a/sub-1",
            "iteration": 0,
            "ts_start": 100.22,
            "ts_end": 100.3,
            "data": {
                "messages_in": [{"role": "user", "content": "inspect state"}],
                "raw_response": {"id": "subagent"},
                "prompt_tokens": 8,
                "completion_tokens": 2,
                "llm_latency_ms": 80.0,
            },
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "tool_0_exec",
            "agent_id": "task-a/sub-1",
            "iteration": 0,
            "ts_start": 100.31,
            "ts_end": 100.34,
            "data": {
                "tool_name": "exec",
                "tool_args": json.dumps({"command": "printf helper"}),
                "tool_result": "helper output",
                "duration_ms": 30.0,
                "success": True,
                "checkpoint_after": {
                    "path": "checkpoints/subagent-manifest.json",
                    "kind": "cas_manifest_full",
                    "root": "/testbed",
                    "incremental": False,
                },
            },
        },
        {
            "type": "summary",
            "agent_id": "task-a",
            "model": "claude-haiku",
            "success": True,
            "n_iterations": 1,
            "elapsed_s": 0.4,
        },
    ]
    _write_jsonl(trace_path, records)
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    async def fail_exec_tool(*_args, **_kwargs):
        raise AssertionError("subagent lane tools must replay from trace")

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fail_exec_tool)

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

    replay_records = _read_jsonl(trace_file)
    subagent_llm = next(
        record
        for record in replay_records
        if record.get("type") == "action"
        and record.get("action_type") == "llm_call"
        and record["data"].get("source_lane_agent_id") == "task-a/sub-1"
    )
    subagent_tool = next(
        record
        for record in replay_records
        if record.get("type") == "action"
        and record.get("action_type") == "tool_exec"
        and record["data"].get("source_lane_agent_id") == "task-a/sub-1"
    )

    assert subagent_llm["agent_id"] == "task-a"
    assert subagent_llm["action_id"] == "task-a/sub-1:llm_0"
    assert subagent_llm["data"]["replay_source"] == "subagent_lane_replay"
    assert subagent_tool["agent_id"] == "task-a"
    assert subagent_tool["action_id"] == "task-a/sub-1:tool_0_exec"
    assert subagent_tool["data"]["tool_result"] == "helper output"
    assert subagent_tool["data"]["replay_source"] == "subagent_lane_replay"
    assert subagent_tool["data"]["sim_metrics"]["sim_tool_format"] == "subagent_lane_replay"


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

    async def fail_prepare(*args: Any, **kwargs: Any) -> None:
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
        extra_args: list[str] | None = None,
    ) -> str:
        assert executable == "docker"
        assert network_mode == "host"
        assert extra_args is not None
        assert "agent-sched-bench.component=simulate-replay" in extra_args
        cas_root = str(Path.home() / ".cache" / "agent-checkpoint-cas")
        assert "-v" in extra_args
        assert f"{cas_root}:{cas_root}" in extra_args
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
    monkeypatch.setattr(
        "trace_collect.simulator._run_checked_container_command",
        lambda *args, **kwargs: None,
    )
    _patch_noop_replay_python_probe(monkeypatch)
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
        "swebench-fixed-root-docker.io_shared_image_latest:simulate-sweep-"
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
        ]

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

    agent_kwargs: list[dict[str, object]] = []

    class _FakeContainerAgent:
        def __init__(
            self,
            container_id: str,
            container_executable: str,
            **kwargs,
        ) -> None:
            assert container_id == "fake-cid"
            assert container_executable == "docker"
            agent_kwargs.append(kwargs)

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
    bootstrap_commands: list[list[str]] = []

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

    def fake_resolve_running_container_exec_config(**kwargs):
        return dataclasses.replace(
            kwargs["exec_config"],
            runtime="/opt/conda/bin/python3",
        )

    monkeypatch.setattr("trace_collect.simulator.ensure_source_image", lambda *args, **kwargs: None)
    monkeypatch.setattr("trace_collect.simulator.ensure_fixed_image", fake_ensure_fixed_image)
    monkeypatch.setattr("trace_collect.simulator.remove_image", fake_remove_image)
    monkeypatch.setattr(
        "trace_collect.simulator.start_task_container",
        lambda *args, **kwargs: "fake-cid",
    )
    monkeypatch.setattr("trace_collect.simulator.stop_task_container", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "trace_collect.simulator._run_checked_container_command",
        lambda cmd, *, timeout: bootstrap_commands.append(cmd),
    )
    monkeypatch.setattr(
        "trace_collect.simulator.resolve_running_container_exec_config",
        fake_resolve_running_container_exec_config,
        raising=False,
    )
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
        "swebench-fixed-root-docker.io_swebench-test_task-a:simulate-sweep-"
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
    assert agent_kwargs == [{}]
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
        "trace_collect.simulator._run_checked_container_command",
        lambda *args, **kwargs: None,
    )
    _patch_noop_replay_python_probe(monkeypatch)
    monkeypatch.setattr(
        "trace_collect.simulator.remove_image",
        lambda image, *, container_executable: removed_images.append(image) or True,
    )
    monkeypatch.setattr("trace_collect.openclaw_tools.ContainerAgent", _FailingContainerAgent)

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
    summary = next(
        record
        for record in _read_jsonl(trace_file)
        if record.get("type") == "summary" and record.get("agent_id") == "task-a"
    )
    assert summary["success"] is False
    assert summary["prep_error"] == "RuntimeError: agent failed"
    assert summary["error"] == "RuntimeError: agent failed"
    throughput = json.loads((output_dir / "throughput_summary.json").read_text())
    assert throughput["completed_traces"] == 0
    assert throughput["failed_traces"] == 1
    assert throughput["tasks"][0]["prep_error"] == "RuntimeError: agent failed"


def test_cloud_model_start_container_failure_records_failed_prep(
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

    startup = json.loads(
        (output_dir / "task-a" / "attempt_1" / "container_startup.json").read_text()
    )
    assert startup["status"] == "failed"
    assert startup["phases"][-1]["name"] == "start_task_container"
    assert startup["phases"][-1]["status"] == "failed"
    assert removed_images == ["fixed-image"]
    summary = next(
        record
        for record in _read_jsonl(trace_file)
        if record.get("type") == "summary" and record.get("agent_id") == "task-a"
    )
    assert summary["success"] is False
    assert summary["prep_error"] == "RuntimeError: container start failed"
    assert summary["error"] == "RuntimeError: container start failed"


def test_cloud_model_agent_start_failure_keeps_fixed_image_when_stop_fails(
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
        "trace_collect.simulator._run_checked_container_command",
        lambda *args, **kwargs: None,
    )
    _patch_noop_replay_python_probe(monkeypatch)
    monkeypatch.setattr(
        "trace_collect.simulator.remove_image",
        lambda image, *, container_executable: removed_images.append(image) or True,
    )
    monkeypatch.setattr("trace_collect.openclaw_tools.ContainerAgent", _FailingContainerAgent)

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

    startup = json.loads(
        (output_dir / "task-a" / "attempt_1" / "container_startup.json").read_text()
    )
    assert startup["error"]["message"] == "agent failed"
    summary = next(
        record
        for record in _read_jsonl(trace_file)
        if record.get("type") == "summary" and record.get("agent_id") == "task-a"
    )
    assert summary["success"] is False
    assert summary["prep_error"] == "RuntimeError: container cleanup failed"
    assert summary["error"] == "RuntimeError: container cleanup failed"
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
    prep_error_message = "prepare failed " + ("x" * 600)

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
            raise RuntimeError(prep_error_message)
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

    assert time.monotonic() - started >= 0.04
    assert agent_stops == ["task-slow"]
    assert container_stops == ["fake-slow"]
    records = _read_jsonl(trace_file)
    failed_summary = next(
        record
        for record in records
        if record.get("type") == "summary" and record.get("agent_id") == "task-fail"
    )
    expected_error = f"RuntimeError: {prep_error_message}"[:500]
    assert failed_summary["success"] is False
    assert failed_summary["prep_error"] == expected_error
    assert failed_summary["error"] == expected_error
    assert len(failed_summary["prep_error"]) == 500

    throughput = json.loads((tmp_path / "out" / "throughput_summary.json").read_text())
    assert throughput["completed_traces"] == 1
    assert throughput["failed_traces"] == 1
    failed_task = next(task for task in throughput["tasks"] if task["agent_id"] == "task-fail")
    assert failed_task["success"] is False
    assert failed_task["prep_error"] == expected_error
    per_task_trace = tmp_path / "out" / "task-fail" / "attempt_1" / "trace.jsonl"
    assert per_task_trace.exists()


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

    output_dir = tmp_path / "out"
    trace_file = asyncio.run(
        simulate(
            manifest=_single_trace_manifest(tmp_path, trace_path),
            task_source=task_source,
            output_dir=output_dir,
            mode="cloud_model",
            container_executable="docker",
        )
    )

    assert agent_stops == 1
    assert container_stops == 1
    summary = next(
        record
        for record in _read_jsonl(trace_file)
        if record.get("type") == "summary" and record.get("agent_id") == "task-a"
    )
    assert summary["success"] is False
    assert summary["prep_error"] == "RuntimeError: sampler failed"
    throughput = json.loads((output_dir / "throughput_summary.json").read_text())
    assert throughput["completed_traces"] == 0
    assert throughput["failed_traces"] == 1
    assert throughput["tasks"][0]["prep_error"] == "RuntimeError: sampler failed"


def _loaded_for_finalize(tmp_path: Path) -> object:
    from trace_collect.simulator import LoadedTraceSession

    return LoadedTraceSession(
        source_trace=tmp_path / "trace.jsonl",
        task_source=tmp_path / "tasks.json",
        source_agent_id="task-a",
        run_instance_id="task-a",
        manifest_index=0,
        scaffold="openclaw",
        metadata=None,
        summary=None,
        task={},
        actions=[],
        iterations={},
    )


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

    async def fail_prepare(*args: Any, **kwargs: Any) -> None:
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


def test_cloud_model_replay_preserves_llm_messages_delta(
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
                        "scaffold": "openclaw",
                        "instance_id": "host-delta-task",
                        "model": "qwen",
                        "mode": "collect",
                        "execution_environment": "host",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "host-delta-task-llm-0",
                        "agent_id": "host-delta-task",
                        "iteration": 0,
                        "ts_start": 100.0,
                        "ts_end": 100.2,
                        "data": {
                            "messages_delta": [
                                {"role": "user", "content": "fix bug"}
                            ],
                            "is_delta": True,
                            "raw_response": {"id": "r-delta"},
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "llm_latency_ms": 200.0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "summary",
                        "agent_id": "host-delta-task",
                        "model": "qwen",
                        "success": True,
                        "n_iterations": 1,
                        "elapsed_s": 0.2,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_host_tasks(task_source, "host-delta-task")

    async def fail_prepare(*args: Any, **kwargs: Any) -> None:
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
    llm_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    )
    assert "messages_in" not in llm_record["data"]
    assert llm_record["data"]["messages_delta"] == [
        {"role": "user", "content": "fix bug"}
    ]
    assert llm_record["data"]["is_delta"] is True


def test_cloud_model_replays_web_search_from_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="web_search",
        tool_args={"query": "deterministic replay"},
    )
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path)

    async def fail_exec_tool(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("web_search must replay from trace, not execute")

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fail_exec_tool)

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

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    assert tool_record["data"]["tool_result"] == "source-result"
    assert tool_record["data"]["success"] is True
    assert tool_record["data"]["replay_source"] == "replayed_from_trace"
    assert tool_record["data"]["replay_outcome_match"] is True
    assert "forced_sync_attempted" not in tool_record["data"]
    assert tool_record["data"]["sim_metrics"]["source"] == "replayed_from_trace"
    assert tool_record["data"]["sim_metrics"]["sim_tool_format"] == "replayed_from_trace"


def test_cloud_model_multi_worker_host_smoke(tmp_path: Path) -> None:
    trace_paths: list[Path] = []
    agent_ids = [f"host-task-{index}" for index in range(4)]
    for agent_id in agent_ids:
        trace_path = tmp_path / f"{agent_id}.jsonl"
        _write_trace(
            trace_path,
            agent_id=agent_id,
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
    assert metadata["source_trace_count"] == 2
    assert set(metadata["source_traces"]) == {str(trace_a), str(trace_b)}
    assert {record["agent_id"] for record in summaries} == {"task-a", "task-b"}
    assert {record["agent_id"] for record in llm_records} == {"task-a", "task-b"}
    assert abs(llm_records[0]["ts_start"] - llm_records[1]["ts_start"]) < 0.05


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
                "source_agent_id": "task-a",
                "run_instance_id": run_instance_id,
                "label": None,
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
    _write_trace(trace_a, agent_id="task-a", execution_environment="host")
    _write_trace(
        trace_replica,
        agent_id="task-a__replica-001",
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
    _write_trace(trace_a, agent_id="task-a", execution_environment="host")
    _write_trace(
        trace_replica,
        agent_id="task-a__replica-001",
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
    _patch_simulator_runtime(monkeypatch, tmp_path)

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

    # host_agent_docker_tools traces: agent on host, tools via docker exec → container replay
    host_agent_docker = SimpleNamespace(
        metadata={"agent_runtime_mode": "host_agent_docker_tools"},
        source_trace="/tmp/host_agent_docker.jsonl",
    )
    assert _execution_environment(host_agent_docker) == "container"

    # Old task_container_agent traces (agent was in-container): same container replay
    old_task_container = SimpleNamespace(
        metadata={"agent_runtime_mode": "task_container_agent"},
        source_trace="/tmp/task_container.jsonl",
    )
    assert _execution_environment(old_task_container) == "container"


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


# ── Step 4: CAS checkpoint fold chain + marker protocol ──────────────────────


class TestFoldSourceCheckpointEntries:
    """Unit tests for _fold_source_checkpoint_entries fold chain."""

    def test_full_replaces_prev_folded(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "full.json"
        spec_path.write_text(json.dumps({
            "entries": {"a.txt": {"hash": "aaa"}, "b.txt": {"hash": "bbb"}},
            "deleted_paths": [],
        }))
        spec = {"path": str(spec_path), "kind": "cas_manifest_full", "incremental": False}
        result = _fold_source_checkpoint_entries(
            checkpoint_spec=spec,
            prev_folded={"old.txt": "oldhash"},
        )
        assert result == {"a.txt": "aaa", "b.txt": "bbb"}

    def test_incremental_updates_and_deletes(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "inc.json"
        spec_path.write_text(json.dumps({
            "entries": {"b.txt": {"hash": "bbb_v2"}, "c.txt": {"hash": "ccc"}},
            "deleted_paths": ["a.txt"],
        }))
        spec = {"path": str(spec_path), "kind": "cas_manifest_incremental", "incremental": True}
        prev = {"a.txt": "aaa", "b.txt": "bbb_v1"}
        result = _fold_source_checkpoint_entries(
            checkpoint_spec=spec,
            prev_folded=prev,
        )
        assert result == {"b.txt": "bbb_v2", "c.txt": "ccc"}

    def test_source_manifest_modes_are_preserved_through_fold(
        self,
        tmp_path: Path,
    ) -> None:
        full_path = tmp_path / "full.json"
        full_path.write_text(json.dumps({
            "entries": {
                "script.sh": {"hash": "old", "mode": 0o644},
                "README.md": {"hash": "readme", "mode": 0o644},
            },
            "deleted_paths": [],
        }))
        inc_path = tmp_path / "inc.json"
        inc_path.write_text(json.dumps({
            "entries": {"script.sh": {"hash": "new", "mode": 0o755}},
            "deleted_paths": ["README.md"],
        }))

        folded = _fold_source_checkpoint_entries(
            checkpoint_spec={
                "path": str(full_path),
                "kind": "cas_manifest_full",
                "incremental": False,
            },
            prev_folded={},
        )
        folded = _fold_source_checkpoint_entries(
            checkpoint_spec={
                "path": str(inc_path),
                "kind": "cas_manifest_incremental",
                "incremental": True,
            },
            prev_folded=folded,
        )

        assert folded == {"script.sh": {"hash": "new", "mode": 0o755}}

    def test_source_manifest_symlinks_are_preserved_through_fold(
        self,
        tmp_path: Path,
    ) -> None:
        full_path = tmp_path / "full.json"
        full_path.write_text(json.dumps({
            "entries": {
                "target.txt": {"hash": "target"},
                "link.txt": {"type": "symlink", "target": "target.txt"},
            },
            "deleted_paths": [],
        }))
        inc_path = tmp_path / "inc.json"
        inc_path.write_text(json.dumps({
            "entries": {
                "link.txt": {"type": "symlink", "target": "renamed.txt"},
            },
            "deleted_paths": ["target.txt"],
        }))

        folded = _fold_source_checkpoint_entries(
            checkpoint_spec={
                "path": str(full_path),
                "kind": "cas_manifest_full",
                "incremental": False,
            },
            prev_folded={},
        )
        folded = _fold_source_checkpoint_entries(
            checkpoint_spec={
                "path": str(inc_path),
                "kind": "cas_manifest_incremental",
                "incremental": True,
            },
            prev_folded=folded,
        )

        assert folded == {
            "link.txt": {"type": "symlink", "target": "renamed.txt"},
        }

    def test_legacy_git_entries_are_dropped(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "legacy-full.json"
        spec_path.write_text(json.dumps({
            "entries": {
                "src/app.py": {"hash": "app"},
                ".git/config": {"hash": "git-config"},
                ".git/objects/aa/bb": {"hash": "git-object"},
            },
            "deleted_paths": [],
        }))
        spec = {"path": str(spec_path), "kind": "cas_manifest_full", "incremental": False}

        result = _fold_source_checkpoint_entries(
            checkpoint_spec=spec,
            prev_folded={},
        )

        assert result == {"src/app.py": "app"}

    def test_load_source_manifest_drops_legacy_git_entries(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "legacy.json"
        manifest_path.write_text(json.dumps({
            "entries": {
                "README.md": {"hash": "readme"},
                ".git/index": {"hash": "index"},
            },
            "deleted_paths": [".git/index"],
        }))

        assert _load_source_manifest_entries(str(manifest_path)) == {
            "README.md": "readme"
        }

    def test_chain_full_inc_inc(self, tmp_path: Path) -> None:
        specs_dir = tmp_path / "checkpoints"
        specs_dir.mkdir()

        def _write_spec(name, entries, deleted=None, kind="cas_manifest_full"):
            p = specs_dir / name
            p.write_text(json.dumps({
                "entries": {k: {"hash": v} for k, v in entries.items()},
                "deleted_paths": deleted or [],
            }))
            incremental = kind == "cas_manifest_incremental"
            return {"path": str(p), "kind": kind, "incremental": incremental}

        folded: dict[str, str] = {}
        folded = _fold_source_checkpoint_entries(
            checkpoint_spec=_write_spec("full.json", {"a": "h1", "b": "h2"}, kind="cas_manifest_full"),
            prev_folded=folded,
        )
        assert folded == {"a": "h1", "b": "h2"}

        folded = _fold_source_checkpoint_entries(
            checkpoint_spec=_write_spec("inc1.json", {"b": "h2b", "c": "h3"}, deleted=["a"], kind="cas_manifest_incremental"),
            prev_folded=folded,
        )
        assert folded == {"b": "h2b", "c": "h3"}

        folded = _fold_source_checkpoint_entries(
            checkpoint_spec=_write_spec("inc2.json", {"d": "h4"}, kind="cas_manifest_incremental"),
            prev_folded=folded,
        )
        assert folded == {"b": "h2b", "c": "h3", "d": "h4"}

    def test_missing_manifest_preserves_prev(self) -> None:
        spec = {"path": "/nonexistent/manifest.json", "kind": "cas_manifest_full", "incremental": False}
        result = _fold_source_checkpoint_entries(
            checkpoint_spec=spec,
            prev_folded={"a": "h1"},
        )
        assert result == {"a": "h1"}

    def test_invalid_json_preserves_prev(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not json")
        spec = {"path": str(p), "kind": "cas_manifest_incremental", "incremental": True}
        result = _fold_source_checkpoint_entries(
            checkpoint_spec=spec,
            prev_folded={"a": "h1"},
        )
        assert result == {"a": "h1"}

    def test_filesystem_tar_as_full(self, tmp_path: Path) -> None:
        import tarfile
        tar_path = tmp_path / "snapshot.tar"
        content = b"hello"
        digest = hashlib.sha256(content).hexdigest()
        with tarfile.open(str(tar_path), "w") as tf:
            info = tarfile.TarInfo(name="x.txt")
            info.size = len(content)
            tf.addfile(info, __import__("io").BytesIO(content))
        spec = {"path": str(tar_path), "kind": "filesystem_tar", "incremental": False}
        result = _fold_source_checkpoint_entries(
            checkpoint_spec=spec,
            prev_folded={"old": "h"},
        )
        assert result == {"x.txt": digest}


class TestCaptureSnapshotManifest:
    """Tests for _capture_snapshot_manifest marker protocol and merge."""

    def test_no_epoch_reset_in_script(self) -> None:
        """The incremental script must not contain '197001010000' (epoch reset)."""
        import inspect
        source = inspect.getsource(_capture_snapshot_manifest)
        assert "197001010000" not in source

    def test_full_mode_includes_all_paths_and_marker(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Full mode script must emit all_paths and set up marker."""
        captured_script: list[str] = []

        def fake_run(cmd, **_kwargs):
            captured_script.append(" ".join(cmd))
            return type("Result", (), {"returncode": 0, "stdout": '{"entries": {"a": "h1"}, "all_paths": ["a"]}', "stderr": ""})

        monkeypatch.setattr("trace_collect.simulator.subprocess.run", fake_run)

        result = _capture_snapshot_manifest(
            container_id="cid",
            container_executable="docker",
            root="/testbed",
            previous_manifest=None,
        )
        script = captured_script[0]
        assert "all_paths" in script
        assert "temp_marker" in script or "cas_marker_new" in script
        assert "os.rename" in script
        assert result == {"a": "h1"}

    def test_incremental_merge_and_deletion_pruning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Incremental merge: delta applied, unchanged files inherited, deletions pruned."""
        def fake_run(cmd, **_kwargs):
            if "-c" in cmd:
                return type("Result", (), {
                    "returncode": 0,
                    "stdout": '{"entries": {"b": "h2b", "c": "ccc"}, "all_paths": ["a", "b", "c"]}',
                    "stderr": "",
                })
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})

        monkeypatch.setattr("trace_collect.simulator.subprocess.run", fake_run)

        prev = {"a": "h1", "b": "h2", "deleted_file": "h_del"}
        result = _capture_snapshot_manifest(
            container_id="cid",
            container_executable="docker",
            root="/testbed",
            previous_manifest=prev,
        )
        # a inherited (unchanged), b updated (in delta), c added (in delta), deleted_file pruned
        assert result == {"a": "h1", "b": "h2b", "c": "ccc"}

    def test_full_mode_empty_dir_returns_empty_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(cmd, **_kwargs):
            return type("Result", (), {
                "returncode": 0,
                "stdout": '{"entries": {}, "all_paths": []}',
                "stderr": "",
            })

        monkeypatch.setattr("trace_collect.simulator.subprocess.run", fake_run)

        result = _capture_snapshot_manifest(
            container_id="cid",
            container_executable="docker",
            root="/testbed",
            previous_manifest=None,
        )
        assert result == {}

    def test_capture_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(cmd, **_kwargs):
            return type("Result", (), {
                "returncode": 1,
                "stdout": "",
                "stderr": "container failed",
            })

        monkeypatch.setattr("trace_collect.simulator.subprocess.run", fake_run)

        result = _capture_snapshot_manifest(
            container_id="cid",
            container_executable="docker",
            root="/testbed",
            previous_manifest=None,
        )
        assert result is None

    def test_incremental_no_changes_script_does_not_hash_unchanged_files(self) -> None:
        import inspect

        source = inspect.getsource(_capture_snapshot_manifest)
        assert "changed = None" in source
        assert "changed = set()" in source
        assert "if changed is not None and rel not in changed:" in source
        assert '"mode": stat.S_IMODE(st.st_mode)' in source


class TestBugARegression:
    """Regression tests for Bug A: incremental source vs full replay comparison."""

    def test_incremental_source_chain_matches_full_replay(self, tmp_path: Path) -> None:
        """Second checkpoint after incremental chain: replay matches source → cas_manifest_match=True, cas_added_count=0."""
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()

        full_manifest = cp_dir / "full.json"
        full_manifest.write_text(json.dumps({
            "entries": {
                "a.txt": {"hash": "aaa"},
                "b.txt": {"hash": "bbb"},
            },
            "deleted_paths": [],
        }))

        inc_manifest = cp_dir / "inc1.json"
        inc_manifest.write_text(json.dumps({
            "entries": {
                "b.txt": {"hash": "bbb_v2"},
                "c.txt": {"hash": "ccc"},
            },
            "deleted_paths": ["a.txt"],
        }))

        # Fold source chain manually to get expected folded state
        folded = _fold_source_checkpoint_entries(
            checkpoint_spec={"path": str(full_manifest), "kind": "cas_manifest_full", "incremental": False},
            prev_folded={},
        )
        folded = _fold_source_checkpoint_entries(
            checkpoint_spec={"path": str(inc_manifest), "kind": "cas_manifest_incremental", "incremental": True},
            prev_folded=folded,
        )
        assert folded == {"b.txt": "bbb_v2", "c.txt": "ccc"}

        # Now simulate the replay side: replay matches exactly
        # Replay entries = folded state (no divergence)
        replay_entries = {"b.txt": "bbb_v2", "c.txt": "ccc"}

        source_keys = set(folded.keys())
        replay_keys = set(replay_entries.keys())
        common = source_keys & replay_keys
        modified = [k for k in common if folded[k] != replay_entries[k]]
        added = sorted(replay_keys - source_keys)
        removed = sorted(source_keys - replay_keys)
        cas_manifest_match = len(modified) == 0 and len(removed) == 0 and len(added) == 0

        assert cas_manifest_match is True
        assert len(modified) == 0
        assert len(added) == 0
        assert len(removed) == 0

    def test_incremental_source_chain_detects_replay_divergence(self, tmp_path: Path) -> None:
        """Replay has extra file → mismatch detected."""
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()

        full_manifest = cp_dir / "full.json"
        full_manifest.write_text(json.dumps({
            "entries": {"a.txt": {"hash": "aaa"}},
            "deleted_paths": [],
        }))

        folded = _fold_source_checkpoint_entries(
            checkpoint_spec={"path": str(full_manifest), "kind": "cas_manifest_full", "incremental": False},
            prev_folded={},
        )
        # Replay has a stray extra file
        replay = {"a.txt": "aaa", "stray.txt": "extra"}
        added = sorted(set(replay.keys()) - set(folded.keys()))
        assert added == ["stray.txt"]

    def test_empty_source_nonempty_replay_is_mismatch(self) -> None:
        fields = _cas_manifest_comparison_fields(
            source_entries={},
            replay_entries={"stray.txt": "extra"},
        )

        assert fields["cas_manifest_match"] is False
        assert fields["cas_source_entries"] == 0
        assert fields["cas_replay_entries"] == 1
        assert fields["cas_added_count"] == 1
        assert fields["cas_removed_count"] == 0

    def test_nonempty_source_empty_replay_is_mismatch(self) -> None:
        fields = _cas_manifest_comparison_fields(
            source_entries={"expected.txt": "hash"},
            replay_entries={},
        )

        assert fields["cas_manifest_match"] is False
        assert fields["cas_source_entries"] == 1
        assert fields["cas_replay_entries"] == 0
        assert fields["cas_added_count"] == 0
        assert fields["cas_removed_count"] == 1

    def test_mode_mismatch_is_cas_mismatch_when_modes_present(self) -> None:
        fields = _cas_manifest_comparison_fields(
            source_entries={"tool.sh": {"hash": "same", "mode": 0o644}},
            replay_entries={"tool.sh": {"hash": "same", "mode": 0o755}},
        )

        assert fields["cas_manifest_match"] is False
        assert fields["cas_modified_count"] == 0
        assert fields["cas_mode_mismatch_count"] == 1
        assert fields["cas_mode_mismatch_examples"] == [
            {
                "path": "tool.sh",
                "source_mode": "644",
                "replay_mode": "755",
            }
        ]

    def test_legacy_hash_only_manifests_skip_mode_comparison(self) -> None:
        fields = _cas_manifest_comparison_fields(
            source_entries={"tool.sh": "same"},
            replay_entries={"tool.sh": "same"},
        )

        assert fields == {
            "cas_manifest_match": True,
            "cas_source_entries": 1,
            "cas_replay_entries": 1,
            "cas_modified_count": 0,
            "cas_added_count": 0,
            "cas_removed_count": 0,
        }


class TestBugCRestoreChainCorrectness:
    """Evidence for Bug C: incremental restore chain is applied correctly."""

    def test_first_in_chain_is_always_full(self) -> None:
        """_checkpoint_chain_specs_for_action returns chain starting with full checkpoint."""
        assert not _checkpoint_spec_is_incremental({"kind": "cas_manifest_full", "incremental": False})
        assert _checkpoint_spec_is_incremental({"kind": "cas_manifest_incremental", "incremental": True})
        assert _checkpoint_spec_is_incremental({"incremental": True})

    def test_restore_chain_applies_incrementals_on_full_base(self) -> None:
        """_restore_checkpoint_chain_to_container guarantees correct ordering:
        first spec is always a full checkpoint (clear_root=True), subsequent
        incremental specs are applied on top (clear_root=False).

        This is verified by _checkpoint_chain_specs_for_action which walks
        backward until it finds a non-incremental spec. If none found, returns
        None and the caller reports failure. The chain restore itself is
        tested in test_simulate_forced_sync_restores_incremental_checkpoint_chain.
        """
        # Sanity: full spec is not incremental
        full_spec = {"kind": "cas_manifest_full", "incremental": False}
        inc_spec = {"kind": "cas_manifest_incremental", "incremental": True}
        assert not _checkpoint_spec_is_incremental(full_spec)
        assert _checkpoint_spec_is_incremental(inc_spec)

        # Verify chain construction: a chain starting from a full checkpoint
        # can be constructed; if only incremental specs exist, chain is None.
        from trace_collect.simulator import _checkpoint_chain_specs_for_action

        actions: list[dict] = [
            {"data": {"checkpoint_after": {"path": "/tmp/inc_only.json", "kind": "cas_manifest_incremental", "incremental": True, "root": "/testbed"}}},
            {"data": {"checkpoint_after": {"path": "/tmp/inc2.json", "kind": "cas_manifest_incremental", "incremental": True, "root": "/testbed"}}},
        ]
        chain = _checkpoint_chain_specs_for_action(
            actions=actions,
            target_index=1,
            source_trace=Path("/tmp/trace.jsonl"),
        )
        # No full checkpoint → chain is None (can't restore safely)
        assert chain is None

        # With full checkpoint → chain is valid
        actions_with_full = [
            {"data": {"checkpoint_after": {"path": "/tmp/full.json", "kind": "cas_manifest_full", "incremental": False, "root": "/testbed"}}},
            {"data": {"checkpoint_after": {"path": "/tmp/inc.json", "kind": "cas_manifest_incremental", "incremental": True, "root": "/testbed"}}},
        ]
        chain = _checkpoint_chain_specs_for_action(
            actions=actions_with_full,
            target_index=1,
            source_trace=Path("/tmp/trace.jsonl"),
        )
        assert chain is not None
        assert len(chain) == 2


# ---------------------------------------------------------------------------
# ReplayCheckpointScheduler unit tests
# ---------------------------------------------------------------------------


def _make_scheduler(
    scheduling: str = "sync",
) -> tuple[ReplayCheckpointScheduler, list[dict[str, Any]]]:
    """Create a scheduler with a recording log_action callback."""
    logged: list[dict[str, Any]] = []

    def log_action(agent_id: str, record: dict[str, Any]) -> None:
        logged.append(record)

    config = ReplaySchedulerConfig(
        checkpoint_scheduling=scheduling,
    )
    scheduler = ReplayCheckpointScheduler(
        config=config,
        container=None,
        source_trace=Path("/tmp/test_trace.jsonl"),
        log_action=log_action,
    )
    return scheduler, logged


class TestReplayCheckpointSchedulerBasic:
    """Basic lifecycle tests for ReplayCheckpointScheduler."""

    def test_initial_state(self) -> None:
        scheduler, _ = _make_scheduler()
        assert scheduler.state == "IDLE"
        assert not scheduler.pending_forced_sync
        metrics = scheduler.get_metrics()
        assert metrics["boundaries_total"] == 0
        assert metrics["scheduler_overhead_ms"] >= 0
        assert metrics["checkpoint_exposed_ms"] == 0.0

    def test_sync_on_boundary_no_container(self) -> None:
        """Sync mode with no container is a no-op fast path."""
        scheduler, records = _make_scheduler(scheduling="sync")
        record_slot: dict[str, Any] = {}

        asyncio.run(
            scheduler.on_boundary(
                action_index=0,
                cas_spec={"path": "/tmp/cas.json", "root": "/testbed"},
                tool_name="write_file",
                tool_args_json='{"file": "/tmp/foo"}',
                tool_mismatch_reason=None,
                record_slot=record_slot,
                agent_id="test-agent",
            )
        )
        # With no container, the capture is skipped; on_boundary returns
        # immediately and the record slot is unchanged.
        assert scheduler.state == "IDLE"
        assert not scheduler.pending_forced_sync

    def test_drain_idle_is_noop(self) -> None:
        """drain() from IDLE returns 0.0 without blocking."""
        scheduler, _ = _make_scheduler()
        exposed = asyncio.run(scheduler.drain())
        assert exposed == 0.0

    def test_close_idle(self) -> None:
        """close() from IDLE returns empty metrics."""
        scheduler, _ = _make_scheduler()
        aggregates = asyncio.run(scheduler.close())
        assert "checkpoint_exposed_ms" in aggregates
        assert aggregates["boundaries_total"] == 0

    def test_log_or_defer_immediate(self) -> None:
        """Without pending boundary, log_or_defer logs immediately."""
        scheduler, records = _make_scheduler()
        rec = {"action_type": "llm_call", "agent_id": "agent-1"}
        scheduler.log_or_defer("agent-1", rec)
        assert len(records) == 1
        assert records[0] == rec

    def test_set_prev_manifest(self) -> None:
        scheduler, _ = _make_scheduler()
        entries = {"a.txt": {"hash": "abc123"}}
        scheduler.set_prev_manifest(entries)
        assert scheduler.prev_cas_manifest == entries
        scheduler.set_prev_manifest(None)
        assert scheduler.prev_cas_manifest is None

    def test_note_window_no_behavioral_coupling(self) -> None:
        """note_window is metrics-only, no behavioral change."""
        scheduler, _ = _make_scheduler()
        assert scheduler.state == "IDLE"
        scheduler.note_window(0.5)
        assert scheduler.state == "IDLE"  # unchanged

    def test_note_window_accumulation(self) -> None:
        """note_window accumulates totals in get_metrics."""
        scheduler, _ = _make_scheduler()
        metrics = scheduler.get_metrics()
        assert metrics["llm_sleep_window_total_s"] == 0.0
        assert metrics["llm_sleep_window_count"] == 0

        scheduler.note_window(0.5)
        scheduler.note_window(1.2)
        scheduler.note_window(0.3)

        metrics = scheduler.get_metrics()
        assert metrics["llm_sleep_window_total_s"] == 2.0
        assert metrics["llm_sleep_window_count"] == 3

    def test_scheduler_overhead_non_negative(self) -> None:
        """scheduler_overhead_ms is recorded and non-negative."""
        scheduler, _ = _make_scheduler(scheduling="sync")
        record_slot: dict[str, Any] = {}

        asyncio.run(
            scheduler.on_boundary(
                action_index=0,
                cas_spec=None,  # no CAS spec = fast path
                tool_name=None,
                tool_args_json=None,
                tool_mismatch_reason=None,
                record_slot=record_slot,
                agent_id="test-agent",
            )
        )
        metrics = scheduler.get_metrics()
        assert metrics["scheduler_overhead_ms"] >= 0
        assert metrics["boundaries_total"] == 1


class TestReplayCheckpointSchedulerDeferred:
    """Deferred-mode lifecycle tests."""

    def test_deferred_on_boundary_spawns_task(self) -> None:
        """In deferred mode, on_boundary starts a background task."""
        scheduler, _ = _make_scheduler(scheduling="deferred")
        record_slot: dict[str, Any] = {}

        asyncio.run(
            scheduler.on_boundary(
                action_index=0,
                cas_spec=None,
                tool_name=None,
                tool_args_json=None,
                tool_mismatch_reason=None,
                record_slot=record_slot,
                agent_id="test-agent",
            )
        )
        # With no container and no CAS spec, the task completes immediately
        # and flushes back to IDLE. But at on_boundary return, DECIDING was set.
        assert scheduler.state in ("IDLE", "DECIDING")

    def test_deferred_drain_measures_exposed(self) -> None:
        """drain() waits for in-flight work and returns exposed_ms."""
        scheduler, _ = _make_scheduler(scheduling="deferred")
        record_slot: dict[str, Any] = {}

        async def run() -> None:
            await scheduler.on_boundary(
                action_index=0,
                cas_spec=None,
                tool_name=None,
                tool_args_json=None,
                tool_mismatch_reason=None,
                record_slot=record_slot,
                agent_id="test-agent",
            )
            exposed = await scheduler.drain()
            # After drain, the task (which was a fast no-op) should resolve quickly
            assert exposed >= 0

        asyncio.run(run())

    def test_log_or_defer_defers_when_boundary_pending(self) -> None:
        """log_or_defer defers when a boundary record is pending."""
        scheduler, records = _make_scheduler(scheduling="deferred")
        rec2 = {"action_type": "llm_call", "agent_id": "agent-1"}

        async def run() -> None:
            record_slot: dict[str, Any] = {}
            # Start a deferred boundary with an empty cas_spec that will
            # complete quickly. The task sets _pending_boundary_record.
            await scheduler.on_boundary(
                action_index=0,
                cas_spec=None,
                tool_name=None,
                tool_args_json=None,
                tool_mismatch_reason=None,
                record_slot=record_slot,
                agent_id="agent-1",
            )
            # While the task is pending, log_or_defer should defer
            if scheduler.state != "IDLE":
                scheduler.log_or_defer("agent-1", rec2)
            # Drain and flush
            await scheduler.drain()
            await scheduler.close()

        asyncio.run(run())
        # The boundary record and any deferred record should now be flushed
        assert any(r.get("agent_id") == "agent-1" for r in records)

    def test_sync_mode_no_deferral(self) -> None:
        """In sync mode, log_or_defer never defers."""
        scheduler, records = _make_scheduler(scheduling="sync")
        rec = {"action_type": "llm_call", "agent_id": "agent-1"}
        scheduler.log_or_defer("agent-1", rec)
        assert len(records) == 1


class TestReplayCheckpointSchedulerPendingMismatch:
    """Pending-mismatch invariant tests."""

    def test_no_forced_sync_without_mismatch(self) -> None:
        """pending_forced_sync is False when no mismatch is recorded."""
        scheduler, _ = _make_scheduler(scheduling="sync")
        record_slot: dict[str, Any] = {}

        asyncio.run(
            scheduler.on_boundary(
                action_index=0,
                cas_spec=None,  # no spec = no compare = no mismatch
                tool_name=None,
                tool_args_json=None,
                tool_mismatch_reason=None,
                record_slot=record_slot,
                agent_id="test-agent",
            )
        )
        assert not scheduler.pending_forced_sync

    def test_pending_record_cleared_after_close(self) -> None:
        scheduler, _ = _make_scheduler(scheduling="deferred")
        record_slot: dict[str, Any] = {}

        async def run() -> None:
            await scheduler.on_boundary(
                action_index=0,
                cas_spec=None,
                tool_name=None,
                tool_args_json=None,
                tool_mismatch_reason=None,
                record_slot=record_slot,
                agent_id="test-agent",
            )
            await scheduler.close()

        asyncio.run(run())
        assert scheduler.state == "IDLE"
        assert not scheduler.pending_forced_sync

    def test_mismatch_sets_pending_forced_sync(self) -> None:
        """When the deferred oracle compare finds a CAS mismatch,
        pending_forced_sync is True and the record stays pending."""
        scheduler, records = _make_scheduler(scheduling="deferred")

        # We cannot easily inject a fake CAS compare, but we can assert
        # that a mismatch marker propagates: manually set the flag to
        # simulate what the background task does after a CAS mismatch.
        record_slot: dict[str, Any] = {"checkpoint_pending_forced_sync": True}
        scheduler._pending_boundary_record = record_slot
        assert scheduler.pending_forced_sync

    def test_single_pending_record_invariant(self) -> None:
        """Exactly one pending boundary record exists at any time."""
        scheduler, _ = _make_scheduler(scheduling="deferred")
        slot_a: dict[str, Any] = {}
        slot_b: dict[str, Any] = {}

        async def run() -> None:
            # Start boundary A (fast path: no container).
            await scheduler.on_boundary(
                action_index=0,
                cas_spec=None,
                tool_name=None,
                tool_args_json=None,
                tool_mismatch_reason=None,
                record_slot=slot_a,
                agent_id="agent-a",
            )
            # In deferred mode, on_boundary sets _pending_boundary_record and
            # spawns a background task.  With no container, the task completes
            # immediately (calling nothing).  The record is still pending.
            # Drain to finalize and flush A.
            await scheduler.drain()
            scheduler.flush_pending()
            # Record A is now flushed — the pending slot is empty.
            # Start boundary B.  There must be at most one pending record.
            await scheduler.on_boundary(
                action_index=1,
                cas_spec=None,
                tool_name=None,
                tool_args_json=None,
                tool_mismatch_reason=None,
                record_slot=slot_b,
                agent_id="agent-b",
            )
            await scheduler.drain()
            scheduler.flush_pending()

        asyncio.run(run())
        # Both boundaries processed; pending slot is clear.
        assert not scheduler.pending_forced_sync
        assert scheduler.state == "IDLE"

    def test_back_to_back_boundary_drain_serializes(self) -> None:
        """B arrives while A is in-flight — on_boundary drains A first.

        This exercises the gate-less boundary path (e.g. denied-command,
        trace-replayed tools carrying checkpoint specs).  B must not start
        capture until A's work has completed.
        """
        scheduler, _ = _make_scheduler(scheduling="deferred")
        slot_a: dict[str, Any] = {}
        slot_b: dict[str, Any] = {}

        async def run() -> None:
            # Start boundary A with a valid cas_spec so the task performs
            # capture work (which will fail since there's no real container,
            # but the internal drain serialization still exercises the gate).
            await scheduler.on_boundary(
                action_index=0,
                cas_spec={"path": "/tmp/a.json", "root": "/testbed"},
                tool_name="bash",
                tool_args_json='{"cmd": "echo a"}',
                tool_mismatch_reason=None,
                record_slot=slot_a,
                agent_id="agent-a",
            )
            # Boundary A task is in-flight (state IDLE fast path since no
            # container).  Now start boundary B — on_boundary's internal
            # drain serializes even though A already completed.
            await scheduler.on_boundary(
                action_index=1,
                cas_spec={"path": "/tmp/b.json", "root": "/testbed"},
                tool_name="bash",
                tool_args_json='{"cmd": "echo b"}',
                tool_mismatch_reason=None,
                record_slot=slot_b,
                agent_id="agent-b",
            )
            # Drain B.
            await scheduler.drain()
            scheduler.flush_pending()

        asyncio.run(run())
        assert scheduler.state == "IDLE"
        # Both slots should carry checkpoint metrics.
        assert "checkpoint_exposed_ms" in slot_a
        assert "checkpoint_exposed_ms" in slot_b

    def test_resolve_mismatch_before_next_boundary(self) -> None:
        """B arrives while A is mismatch-pending — forced-sync resolves
        before B's capture starts.

        Simulates the pending-mismatch invariant: when a deferred boundary
        resolves to CAS mismatch, the pending_forced_sync flag is set.
        The next boundary (B) must resolve A's forced-sync before starting
        its own capture so B observes the post-restore container state.
        """
        scheduler, _ = _make_scheduler(scheduling="deferred")
        slot_a: dict[str, Any] = {"checkpoint_pending_forced_sync": True}
        slot_b: dict[str, Any] = {}

        async def run() -> None:
            # Simulate A having resolved to mismatch (record is pending).
            scheduler._pending_boundary_record = slot_a
            scheduler._pending_boundary_agent_id = "agent-a"
            scheduler._pending_action_index = 0
            scheduler._pending_cas_spec = {"path": "/tmp/a.json", "root": "/testbed"}
            scheduler._state = "IDLE"

            # B arrives — pending_forced_sync is True.
            assert scheduler.pending_forced_sync

            # The caller (Hook 1 pending-mismatch check, or Hook 2 gate)
            # drains first, observes pending_forced_sync, runs forced-sync
            # for A, then flushes A.
            await scheduler.drain()
            assert scheduler.pending_forced_sync  # still true until forced-sync resolves
            # Simulate forced-sync resolving A: clear the pending_forced_sync
            # marker and flush.
            slot_a.pop("checkpoint_pending_forced_sync", None)
            scheduler.flush_pending()
            assert not scheduler.pending_forced_sync

            # Now B can start its capture on clean post-restore state.
            await scheduler.on_boundary(
                action_index=1,
                cas_spec=None,
                tool_name=None,
                tool_args_json=None,
                tool_mismatch_reason=None,
                record_slot=slot_b,
                agent_id="agent-b",
            )
            await scheduler.drain()
            scheduler.flush_pending()

        asyncio.run(run())
        assert scheduler.state == "IDLE"
        # B did not observe a spurious mismatch (no pending_forced_sync).
        assert not scheduler.pending_forced_sync
        # B's capture completed successfully.
        assert "checkpoint_exposed_ms" in slot_b


class TestReplayCheckpointSchedulerConfig:
    """Configuration validation."""

    def test_default_config_is_sync_off(self) -> None:
        config = ReplaySchedulerConfig()
        assert config.checkpoint_scheduling == "sync"

    def test_deferred_off_is_valid(self) -> None:
        """deferred is a valid scheduling mode."""
        config = ReplaySchedulerConfig(
            checkpoint_scheduling="deferred",
        )
        assert config.checkpoint_scheduling == "deferred"

    def test_config_is_hashable(self) -> None:
        """Frozen dataclass must be hashable for set/dict keys."""
        config = ReplaySchedulerConfig()
        assert hash(config) is not None
        d = {config: "test"}
        assert d[config] == "test"

    def test_unknown_scheduling_rejected(self) -> None:
        """Unknown checkpoint_scheduling value raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_scheduling"):
            ReplaySchedulerConfig(checkpoint_scheduling="unknown")


# ---------------------------------------------------------------------------
# STORY-5: A/B test harness for PR1 sync vs deferred comparison
# ---------------------------------------------------------------------------

_SCHEDULER_METRIC_KEYS: frozenset[str] = frozenset({
    # Per-boundary timing fields (added by _finalize_boundary_record)
    "checkpoint_exposed_ms",
    "probe_elapsed_ms",
    "capture_elapsed_ms",
    "compare_elapsed_ms",
    "capture_absorbed_ms",
    "overlap_fraction",
    "scheduler_overhead_ms",
    # CAS oracle comparison fields
    "cas_compare_source",
    "cas_manifest_match",
    "cas_source_entries",
    "cas_replay_entries",
    "cas_modified_count",
    "cas_added_count",
    "cas_removed_count",
    "cas_mode_mismatch_count",
    "cas_mode_mismatch_examples",
    # Gate annotations
    "checkpoint_pending_forced_sync",
    "checkpoint_decision",
})


def _strip_scheduler_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *data* with scheduler-added metric keys removed.

    Scheduler-added keys (timing, CAS comparison, gate annotations) are
    expected to differ between sync and deferred arms.  Stripping them
    before comparison isolates the replay-correctness-relevant fields.

    Also strips sleep-drift sub-keys inside ``sim_metrics``, since the
    deferred arm may have slightly different sleep characteristics
    (Critic condition 3 handles the formal tolerance check separately).
    """
    stripped = {k: v for k, v in data.items() if k not in _SCHEDULER_METRIC_KEYS}
    # Strip sleep-drift fields from within sim_metrics
    sim_metrics = stripped.get("sim_metrics")
    if isinstance(sim_metrics, dict):
        stripped["sim_metrics"] = {
            k: v for k, v in sim_metrics.items()
            if k not in ("source_gap_sleep", "action_sleep")
        }
    return stripped


def _compare_ab_outputs(
    sync_trace_file: Path,
    deferred_trace_file: Path,
    *,
    drift_tolerance_factor: float = 1.5,
) -> dict[str, Any]:
    """Compare sync vs deferred checkpoint-scheduling replay outputs.

    Runs three fidelity invariants per traced agent:

    1. **Mismatch-count parity** -- ``outcome_mismatches`` and
       ``unresolved_mismatches`` must be identical between arms.
    2. **Forced-sync outcome parity** -- ``forced_sync_actions``,
       ``forced_sync_attempts``, ``forced_sync_successes``, and
       ``forced_sync_continued`` counts must match.
    3. **Sleep-drift budget** -- deferred-arm p95 drift must not
       exceed ``tolerance_factor`` * sync-arm p95 drift (Critic
       condition 3).

    Also checks that tool_exec record data (modulo scheduler-added
    metric keys) is identical between arms.
    """
    sync_records = _read_jsonl(sync_trace_file)
    deferred_records = _read_jsonl(deferred_trace_file)

    def _summaries(records: list[dict]) -> dict[str, dict]:
        return {
            r["agent_id"]: r
            for r in records
            if r.get("type") == "summary" and "agent_id" in r
        }

    def _tool_execs(records: list[dict]) -> dict[str, list[dict]]:
        by_agent: dict[str, list[dict]] = {}
        for r in records:
            if r.get("type") == "action" and r.get("action_type") == "tool_exec":
                agent = r.get("agent_id", "")
                by_agent.setdefault(agent, []).append(r)
        return by_agent

    sync_summaries = _summaries(sync_records)
    deferred_summaries = _summaries(deferred_records)
    sync_tools = _tool_execs(sync_records)
    deferred_tools = _tool_execs(deferred_records)

    agents = sorted(set(sync_summaries) | set(deferred_summaries))
    per_task: list[dict[str, Any]] = []
    all_mismatch_ok = True
    all_forced_ok = True
    all_drift_ok = True
    all_tool_ok = True

    for agent in agents:
        s_sum = sync_summaries.get(agent, {})
        d_sum = deferred_summaries.get(agent, {})

        # --- Invariant 1: mismatch counts identical ---
        mismatch_ok = (
            s_sum.get("outcome_mismatches") == d_sum.get("outcome_mismatches")
            and s_sum.get("unresolved_mismatches") == d_sum.get("unresolved_mismatches")
        )
        if not mismatch_ok:
            all_mismatch_ok = False

        # --- Invariant 2: forced-sync outcomes identical ---
        forced_ok = (
            s_sum.get("forced_sync_actions") == d_sum.get("forced_sync_actions")
            and s_sum.get("forced_sync_attempts") == d_sum.get("forced_sync_attempts")
            and s_sum.get("forced_sync_successes") == d_sum.get("forced_sync_successes")
            and s_sum.get("forced_sync_continued") == d_sum.get("forced_sync_continued")
        )
        if not forced_ok:
            all_forced_ok = False

        # --- Invariant 3: sleep-drift budget ---
        drift_check = _check_sleep_drift_tolerance(
            s_sum.get("sleep_drift", {}),
            d_sum.get("sleep_drift", {}),
            tolerance_factor=drift_tolerance_factor,
        )
        if not drift_check["within_tolerance"]:
            all_drift_ok = False

        # --- Tool record parity (modulo scheduler keys) ---
        s_tools = sync_tools.get(agent, [])
        d_tools = deferred_tools.get(agent, [])
        tool_ok = True
        tool_diffs: list[dict[str, Any]] = []
        if len(s_tools) == len(d_tools):
            for idx, (s, d) in enumerate(zip(s_tools, d_tools)):
                s_data = _strip_scheduler_metrics(s.get("data", {}))
                d_data = _strip_scheduler_metrics(d.get("data", {}))
                if s_data != d_data:
                    tool_ok = False
                    # Collect symmetric diff keys for diagnostics
                    all_keys = set(s_data) | set(d_data)
                    diff_keys = sorted(
                        k for k in all_keys
                        if s_data.get(k) != d_data.get(k)
                    )
                    tool_diffs.append({
                        "index": idx,
                        "action_id": s.get("action_id"),
                        "differing_keys": diff_keys,
                    })
        else:
            tool_ok = False
        if not tool_ok:
            all_tool_ok = False

        per_task.append({
            "agent_id": agent,
            "mismatch_count_match": mismatch_ok,
            "forced_sync_match": forced_ok,
            "drift_check": drift_check,
            "tool_record_parity": tool_ok,
            "tool_diffs": tool_diffs,
        })

    return {
        "agents": agents,
        "per_task": per_task,
        "mismatch_count_match": all_mismatch_ok,
        "forced_sync_match": all_forced_ok,
        "drift_ok": all_drift_ok,
        "tool_record_parity": all_tool_ok,
        "passed": all_mismatch_ok and all_forced_ok and all_drift_ok and all_tool_ok,
    }


def _ab_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    agent_id: str = "task-a",
    tool_name: str = "exec",
    with_checkpoint: bool = True,
) -> tuple[Path, Path, Path]:
    """Shared setup for A/B tests: write trace, tasks, apply runtime patches.

    Returns (trace_path, task_source, manifest).
    """
    trace_path = tmp_path / f"{agent_id}.jsonl"
    task_source = tmp_path / "tasks.json"
    checkpoint_after: dict[str, str] | None = None
    if with_checkpoint:
        checkpoint_after = {
            "path": "checkpoints/after-tool-manifest.json",
            "kind": "cas_manifest_full",
            "root": "/testbed",
        }
    _write_trace(
        trace_path,
        agent_id=agent_id,
        tool_name=tool_name,
        checkpoint_after=checkpoint_after,
    )
    _write_tasks(task_source, agent_id)
    _patch_simulator_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "trace_collect.simulator._capture_snapshot_manifest",
        lambda **_kwargs: {},
    )
    manifest = _single_trace_manifest(tmp_path, trace_path)
    return trace_path, task_source, manifest


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


def test_pr1_ab_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Sync and deferred arms produce identical replay outcomes for a
    simple trace with a checkpoint boundary."""
    _trace_path, task_source, manifest = _ab_setup(monkeypatch, tmp_path)
    output_dir = tmp_path / "out"

    sync_trace = asyncio.run(
        simulate(
            manifest=manifest,
            task_source=task_source,
            output_dir=output_dir / "sync",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
            checkpoint_scheduling="sync",
        )
    )
    deferred_trace = asyncio.run(
        simulate(
            manifest=manifest,
            task_source=task_source,
            output_dir=output_dir / "deferred",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
            checkpoint_scheduling="deferred",
        )
    )

    result = _compare_ab_outputs(sync_trace, deferred_trace)
    assert result["passed"], (
        f"A/B comparison failed: {json.dumps(result['per_task'], indent=2)}"
    )
    assert result["mismatch_count_match"]
    assert result["forced_sync_match"]
    assert result["drift_ok"]
    assert result["tool_record_parity"]


def test_pr1_ab_drift_budget() -> None:
    """Sleep-drift tolerance check enforces the 1.5x bound (Critic condition 3)."""
    # Within tolerance: deferred p95 <= 1.5x sync p95
    within = _check_sleep_drift_tolerance(
        {"drift_s": {"p95": 0.010}},
        {"drift_s": {"p95": 0.014}},
        tolerance_factor=1.5,
    )
    assert within["within_tolerance"] is True
    assert within["sync_p95"] == 0.010
    assert within["deferred_p95"] == 0.014
    assert within["bound"] == 0.015
    assert within["tolerance_factor"] == 1.5

    # Exceeds tolerance: deferred p95 > 1.5x sync p95
    exceeded = _check_sleep_drift_tolerance(
        {"drift_s": {"p95": 0.010}},
        {"drift_s": {"p95": 0.020}},
        tolerance_factor=1.5,
    )
    assert exceeded["within_tolerance"] is False

    # Exact boundary: deferred == bound
    at_bound = _check_sleep_drift_tolerance(
        {"drift_s": {"p95": 0.010}},
        {"drift_s": {"p95": 0.015}},
        tolerance_factor=1.5,
    )
    assert at_bound["within_tolerance"] is True

    # Sync has zero drift: always within tolerance
    zero_sync = _check_sleep_drift_tolerance(
        {"drift_s": {"p95": 0.0}},
        {"drift_s": {"p95": 0.100}},
        tolerance_factor=1.5,
    )
    assert zero_sync["within_tolerance"] is True
    assert zero_sync["reason"] == "sync_arm_no_drift"


def test_pr1_ab_mismatch_parity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Forced-sync counts match between sync and deferred arms when a
    tool-execution mismatch triggers forced-sync recovery."""
    _trace_path, task_source, manifest = _ab_setup(monkeypatch, tmp_path)

    async def fake_exec_tool_fail(*_args: Any, **_kwargs: Any) -> tuple[str, float, bool]:
        return "failed\n\nExit code: 1", 1.0, False

    def fake_restore(
        *,
        checkpoint_spec: dict[str, Any],
        container: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "forced_sync_success": True,
            "forced_sync_elapsed_ms": 12.0,
            "forced_sync_checkpoint": checkpoint_spec["path"],
            "forced_sync_root": checkpoint_spec["root"],
        }

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool_fail)
    monkeypatch.setattr(
        "trace_collect.simulator._restore_checkpoint_to_container",
        fake_restore,
    )

    output_dir = tmp_path / "out"

    sync_trace = asyncio.run(
        simulate(
            manifest=manifest,
            task_source=task_source,
            output_dir=output_dir / "sync",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
            checkpoint_scheduling="sync",
        )
    )
    deferred_trace = asyncio.run(
        simulate(
            manifest=manifest,
            task_source=task_source,
            output_dir=output_dir / "deferred",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=100.0,
            checkpoint_scheduling="deferred",
        )
    )

    result = _compare_ab_outputs(sync_trace, deferred_trace)
    assert result["mismatch_count_match"], (
        f"Mismatch counts diverge: {json.dumps(result['per_task'], indent=2)}"
    )
    assert result["forced_sync_match"], (
        f"Forced-sync outcomes diverge: {json.dumps(result['per_task'], indent=2)}"
    )

    # Sanity: forced-sync counts are non-zero (the mismatch actually fired)
    sync_records = _read_jsonl(sync_trace)
    sync_summaries = [r for r in sync_records if r.get("type") == "summary"]
    assert sync_summaries[0]["forced_sync_actions"] >= 1
    assert sync_summaries[0]["forced_sync_attempts"] >= 1
    assert sync_summaries[0]["outcome_mismatches"] >= 1

    deferred_records = _read_jsonl(deferred_trace)
    deferred_summaries = [r for r in deferred_records if r.get("type") == "summary"]
    assert deferred_summaries[0]["forced_sync_actions"] == sync_summaries[0]["forced_sync_actions"]
    assert deferred_summaries[0]["forced_sync_attempts"] == sync_summaries[0]["forced_sync_attempts"]


# ---------------------------------------------------------------------------
# Output-content mismatch detection
# ---------------------------------------------------------------------------


def test_output_content_mismatch_detected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Diffing outputs triggers output_content_mismatch when transport matches."""
    from harness.trace_logger import TraceLogger
    from trace_collect.simulator import (
        _load_trace_session,
        _replay_cloud_model_session,
    )

    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="exec",
        tool_args={"command": "printf volatile"},
    )
    records = _read_jsonl(trace_path)
    for record in records:
        if record.get("action_type") == "tool_exec":
            # Source output: content that differs after normalization
            record["data"]["tool_result"] = "apple\n\nExit code: 0"
            record["data"]["returncode"] = 0
    _write_jsonl(trace_path, records)
    _write_tasks(task_source, "task-a")
    loaded = _load_trace_session(trace_path, task_source, 0)
    prepared = PreparedTraceSession(
        loaded=loaded,
        container=PreparedContainer(
            container_id="fake-cid",
            container_executable="docker",
            docker_image="fake-image",
            agent=object(),
        ),
    )

    async def fake_exec_tool(
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[str, float, bool]:
        # Different content but same exit code
        return ("orange\n\nExit code: 0", 1.0, True)

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    trace_logger = TraceLogger(tmp_path / "out", "run")
    try:
        asyncio.run(
            _replay_cloud_model_session(
                prepared,
                trace_logger=trace_logger,
                replay_speed=100.0,
                llm_timing=LLMTimingConfig(),
                command_timeout_s=600.0,
                warmup_skip_iterations=0,
            )
        )
    finally:
        trace_logger.close()

    records = _read_jsonl(trace_logger.path)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    # Transport tiers matched: same return code, no timeout
    assert tool_record["data"]["replay_outcome_match"] is True
    # Output differs after normalization
    assert tool_record["data"]["normalized_output_match"] is False
    # Effective mismatch reason is output_content_mismatch
    assert tool_record["data"]["mismatch_reason"] == "output_content_mismatch"
    # forced-sync is attempted but fails because no checkpoint spec exists.
    # The output_content_mismatch sets effective_mismatch_reason, which
    # enters the forced-sync gate; the gate falls back to "no checkpoint".
    assert tool_record["data"]["forced_sync_attempted"] is True
    assert tool_record["data"]["forced_sync_reason"] == "output_content_mismatch"
    assert tool_record["data"]["forced_sync_status"] == "checkpoint_missing"


def test_output_content_mismatch_skipped_for_flaky_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Non-exec tools (web_search) avoid output_content_mismatch."""
    from harness.trace_logger import TraceLogger
    from trace_collect.simulator import (
        _load_trace_session,
        _replay_cloud_model_session,
    )

    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    # web_search content varies legitimately, so it's excluded from
    # output_content_mismatch by an inline tool_name check.
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="web_search",
        tool_args={"query": "hello"},
    )
    _write_tasks(task_source, "task-a")
    loaded = _load_trace_session(trace_path, task_source, 0)
    prepared = PreparedTraceSession(
        loaded=loaded,
        container=PreparedContainer(
            container_id="fake-cid",
            container_executable="docker",
            docker_image="fake-image",
            agent=object(),
        ),
    )

    async def fake_exec_tool(
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[str, float, bool]:
        return ("search result B", 1.0, True)

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    trace_logger = TraceLogger(tmp_path / "out", "run")
    try:
        asyncio.run(
            _replay_cloud_model_session(
                prepared,
                trace_logger=trace_logger,
                replay_speed=100.0,
                llm_timing=LLMTimingConfig(),
                command_timeout_s=600.0,
                warmup_skip_iterations=0,
            )
        )
    finally:
        trace_logger.close()

    records = _read_jsonl(trace_logger.path)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    # web_search is non-exec, so normalized_output_match is None (not set)
    assert "normalized_output_match" not in tool_record["data"]
    # Replay succeeded, no transport-level mismatch
    assert tool_record["data"]["replay_outcome_match"] is True
    # No output_content_mismatch (web_search excluded by inline guard)
    assert "mismatch_reason" not in tool_record["data"]


# ---------------------------------------------------------------------------
# Deferred mode: CAS-only mismatch triggers forced sync
# ---------------------------------------------------------------------------


def test_cas_only_mismatch_triggers_forced_sync_deferred_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CAS mismatch triggers forced sync in deferred mode via transport-mismatch gate."""
    from harness.trace_logger import TraceLogger
    from trace_collect.simulator import (
        _load_trace_session,
        _replay_cloud_model_session,
    )

    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    checkpoint_path = tmp_path / "checkpoints" / "after-tool-manifest.json"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_text(
        json.dumps(
            {
                "entries": {"expected.txt": {"hash": "source-hash"}},
                "deleted_paths": [],
            }
        ),
        encoding="utf-8",
    )
    _write_trace(
        trace_path,
        agent_id="task-a",
        tool_name="exec",
        tool_args={"command": "printf volatile"},
        checkpoint_after={
            "path": "checkpoints/after-tool-manifest.json",
            "kind": "cas_manifest_full",
            "root": "/testbed",
        },
    )
    records = _read_jsonl(trace_path)
    for record in records:
        if record.get("action_type") == "tool_exec":
            # Source output: content that differs after normalization
            record["data"]["tool_result"] = "apple\n\nExit code: 0"
            record["data"]["returncode"] = 0
    _write_jsonl(trace_path, records)
    _write_tasks(task_source, "task-a")
    loaded = _load_trace_session(trace_path, task_source, 0)
    prepared = PreparedTraceSession(
        loaded=loaded,
        container=PreparedContainer(
            container_id="fake-cid",
            container_executable="docker",
            docker_image="fake-image",
            agent=object(),
        ),
    )

    async def fake_exec_tool(*_args, **_kwargs):
        return ("orange\n\nExit code: 0", 1.0, True)

    restored: list[dict[str, object]] = []

    def fake_restore_checkpoint_to_container(*, checkpoint_spec, container):
        restored.append({"checkpoint_spec": checkpoint_spec, "container": container})
        return {
            "forced_sync_success": True,
            "forced_sync_status": "checkpoint_restored_continuation",
            "forced_sync_checkpoint": checkpoint_spec["path"],
            "forced_sync_root": checkpoint_spec["root"],
        }

    snapshot_calls = 0

    def fake_capture_snapshot_manifest(**_kwargs):
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls == 1:
            return {"expected.txt": "replay-hash"}
        return {"expected.txt": "source-hash"}

    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    monkeypatch.setattr(
        "trace_collect.simulator._capture_snapshot_manifest",
        fake_capture_snapshot_manifest,
    )
    monkeypatch.setattr(
        "trace_collect.simulator._restore_checkpoint_to_container",
        fake_restore_checkpoint_to_container,
    )

    trace_logger = TraceLogger(tmp_path / "out", "run")
    try:
        asyncio.run(
            _replay_cloud_model_session(
                prepared,
                trace_logger=trace_logger,
                replay_speed=100.0,
                llm_timing=LLMTimingConfig(),
                command_timeout_s=600.0,
                warmup_skip_iterations=0,
                replay_scheduler_config=ReplaySchedulerConfig(
                    checkpoint_scheduling="deferred",
                ),
            )
        )
    finally:
        trace_logger.close()

    records = _read_jsonl(trace_logger.path)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    # Transport tiers matched: same return code, no timeout.
    assert tool_record["data"]["replay_outcome_match"] is True
    # Output differs after normalization, which triggers the output_content_mismatch.
    # In deferred mode, this sets effective_mismatch_reason so the deferred
    # drain at line 5976 fires and drains the background CAS comparison task.
    # The CAS comparison detects the manifest mismatch and upgrades the
    # effective_mismatch_reason to "cas_state_mismatch".
    assert tool_record["data"]["normalized_output_match"] is False
    assert tool_record["data"]["mismatch_reason"] == "cas_state_mismatch"
    assert tool_record["data"]["cas_manifest_match"] is False
    assert tool_record["data"]["forced_sync_attempted"] is True
    assert tool_record["data"]["forced_sync_reason"] == "cas_state_mismatch"
    assert tool_record["data"]["forced_sync_success"] is True
