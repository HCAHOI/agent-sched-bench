import json
from pathlib import Path

import pytest

from scripts.evaluation.evaluate_pennylane_xdist_cgroup_memory import (
    _action_signature,
    _expected_collection_contract,
    _load_cgroup_peaks,
    _simulate_measured_arm,
)


def test_action_identity_includes_call_id() -> None:
    first = {
        "action_type": "tool_exec",
        "data": {"tool_call_id": "first", "tool_name": "exec", "tool_args": "{}"},
    }
    second = {
        "action_type": "tool_exec",
        "data": {"tool_call_id": "second", "tool_name": "exec", "tool_args": "{}"},
    }
    assert _action_signature(first) != _action_signature(second)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_cgroup_peak_loader_requires_complete_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id = "repo__project-1"
    monkeypatch.setattr(
        "scripts.evaluation.evaluate_pennylane_xdist_cgroup_memory._EXPECTED_TASK_IDS",
        (task_id,),
    )
    source = tmp_path / "source.jsonl"
    run = tmp_path / "run"
    attempt = run / task_id / "attempt_1"
    action = {
        "action_type": "tool_exec",
        "ts_start": 1.0,
        "ts_end": 1.1,
        "data": {
            "tool_name": "exec",
            "tool_args": '{"command":"true"}',
            "tool_call_id": "call",
        },
    }
    _write_jsonl(source, [action])
    metadata = {
        "type": "trace_metadata",
        "mode": "simulate",
        "simulate_mode": "cloud_model",
        "replay_speed": 20,
        "llm_timing_mode": "source_scaled",
        "concurrency": 1,
        "effective_concurrency": 1,
        "workers": 1,
        "network_mode": "host",
        "container_start_extra_args": ["--cpus", "8"],
        "instance_id": task_id,
        "manifest_index": 0,
        "source_trace": str(source),
        "source_model": "gpt-5.6-sol",
        "monitoring": {
            "resource_requested": "off",
            "pmu_requested": "off",
            "memory_bandwidth_requested": "off",
            "resource_enabled": False,
            "pmu_enabled": False,
            "memory_bandwidth_enabled": False,
            "concurrency": 1,
            "workers": 1,
        },
        "tool_resource": {"service_enabled": True},
    }
    _write_jsonl(attempt / "trace.jsonl", [metadata, action])
    artifact = {
        "collection_validity": "valid",
        "telemetry_quality": "ok",
        "cleanup": "ok",
        "calls": [
            {
                "tool_call_id": "call",
                "command_window_rss": {"status": "ok"},
                "command_window_memory_current": {
                    "status": "ok",
                    "cadence_ms": 2,
                    "error": None,
                    "read_failures": 0,
                    "sample_count": 2,
                    "sampled_peak_mb": 12.5,
                },
            }
        ],
    }
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "resource_observations.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )
    (run / "collection_contract.json").write_text(
        json.dumps(_expected_collection_contract()), encoding="utf-8"
    )
    (run / "throughput_summary.json").write_text(
        json.dumps(
            {
                "attempted_traces": 1,
                "completed_traces": 1,
                "failed_traces": 0,
                "concurrency": 1,
                "effective_concurrency": 1,
                "effective_workers": 1,
                "mode": "cloud_model",
                "llm_timing_mode": "source_scaled",
                "monitoring": {
                    "resource_requested": "off",
                    "pmu_requested": "off",
                    "memory_bandwidth_requested": "off",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.evaluation.evaluate_pennylane_xdist_cgroup_memory._ROOT", tmp_path
    )
    peaks, evidence = _load_cgroup_peaks(
        run, [{"task_id": task_id, "trace": "source.jsonl"}]
    )
    assert peaks == {f"{task_id}:call": 12.5}
    assert evidence[0]["minimum_samples"] == 2

    artifact["calls"][0]["command_window_memory_current"]["sample_count"] = 1
    (attempt / "resource_observations.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incomplete cgroup-memory sidecar"):
        _load_cgroup_peaks(run, [{"task_id": task_id, "trace": "source.jsonl"}])


def test_over_capacity_peak_is_a_negative_result_without_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.evaluation.evaluate_pennylane_xdist_cgroup_memory.simulate_idle_backfill",
        lambda *args, **kwargs: pytest.fail("over-capacity input reached simulator"),
    )
    candidate, command_ids = _simulate_measured_arm(
        None,
        {"task:call"},
        {"task:call": 16_000.001},
        {"task:call": 500.0},
    )
    assert candidate is None
    assert command_ids == ["task:call"]
