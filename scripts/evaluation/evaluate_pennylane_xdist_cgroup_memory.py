#!/usr/bin/env python3
"""Recheck the frozen PennyLane fit-envelope action with cgroup memory peaks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_pennylane_temporal_rss_ceiling import (  # noqa: E402
    _EXPECTED_TASK_IDS,
)
from scripts.evaluation.evaluate_pennylane_xdist_rss_admission import (  # noqa: E402
    _RSS_CAPACITY_MB,
    _compact,
)
from scripts.evaluation.evaluate_pennylane_xdist_rss_fit_envelope import (  # noqa: E402
    _CPU_CAPACITY,
    continuous_reservations,
)
from scripts.evaluation.evaluate_zarr_rss_backfill import _load_dataset  # noqa: E402
from tool_resource_eval.resource_admission import simulate_idle_backfill  # noqa: E402


_PROTOCOL_GIT_SHA = "6bdf98c0449be49b88221a048a596d147efb7a3a"
_SPLIT = _ROOT / "analysis/development/pennylane-survival-action-split.json"
_PREDICTIONS = _ROOT / "analysis/results/pennylane-xdist-rss-positive-control-v2"
_PREDECESSOR = (
    _ROOT / "analysis/results/pennylane-xdist-rss-fit-envelope-admission-v1/result.json"
)
_RUN = _ROOT / (
    "traces/swe-rebench/gpt-5.6-sol/"
    "pennylane-memory-replay15-sync-c1-20x-ebpf-20260812"
)
_OUTPUT = _ROOT / "analysis/results/pennylane-xdist-cgroup-memory-calibration-v1"
_COLLECTION_SCHEMA = "pennylane-cgroup-memory-calibration-collection-v1"


def _require_clean_checkout() -> str:
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise ValueError("formal evaluation requires a clean committed checkout")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", _PROTOCOL_GIT_SHA, "HEAD"],
        cwd=_ROOT,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _frozen_bytes(path: Path) -> bytes:
    relative = path.relative_to(_ROOT).as_posix()
    frozen = subprocess.run(
        ["git", "show", f"{_PROTOCOL_GIT_SHA}:{relative}"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if path.read_bytes() != frozen:
        raise ValueError(f"frozen input changed: {relative}")
    return frozen


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _action_signature(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    action_type = row.get("action_type")
    if action_type is None:
        return None
    if action_type != "tool_exec":
        return (action_type,)
    data = row.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("tool action has no data")
    return (
        action_type,
        data.get("tool_call_id"),
        data.get("tool_name"),
        data.get("tool_args"),
    )


def _expected_collection_contract() -> dict[str, Any]:
    return {
        "schema": _COLLECTION_SCHEMA,
        "protocol_git_sha": _PROTOCOL_GIT_SHA,
        "task_ids": list(_EXPECTED_TASK_IDS),
        "mode": "cloud_model",
        "concurrency": 1,
        "workers": 1,
        "container": "docker",
        "container_cpus": 8,
        "network_mode": "host",
        "replay_speed": 20,
        "llm_generation": False,
        "monitoring": {
            "resource": "off",
            "pmu": "off",
            "memory_bandwidth": "off",
        },
        "tool_resource": {
            "behavior": "observe_predict_learn",
            "update_policy": "causal",
            "snapshot": "latest_at_run_start",
            "telemetry_requirement": "required_for_valid_evidence",
            "latency_bucket_edges_ms": [500, 2000, 8000, 30000],
        },
        "telemetry": {
            "ebpf": True,
            "command_rss_oracle": True,
            "command_memory_current_oracle": True,
            "memory_current_cadence_ms": 2,
        },
        "synchronous_telemetry_registration": True,
        "created_before_first_task": True,
    }


def _validate_replay_metadata(
    metadata: Mapping[str, Any],
    *,
    task_id: str,
    manifest_index: int,
    source_path: Path,
) -> None:
    monitoring = metadata.get("monitoring")
    required_monitoring = {
        "resource_requested": "off",
        "pmu_requested": "off",
        "memory_bandwidth_requested": "off",
        "resource_enabled": False,
        "pmu_enabled": False,
        "memory_bandwidth_enabled": False,
        "concurrency": 1,
        "workers": 1,
    }
    if (
        metadata.get("type") != "trace_metadata"
        or metadata.get("mode") != "simulate"
        or metadata.get("simulate_mode") != "cloud_model"
        or metadata.get("replay_speed") != 20
        or metadata.get("llm_timing_mode") != "source_scaled"
        or metadata.get("concurrency") != 1
        or metadata.get("effective_concurrency") != 1
        or metadata.get("workers") != 1
        or metadata.get("network_mode") != "host"
        or metadata.get("container_start_extra_args") != ["--cpus", "8"]
        or metadata.get("instance_id") != task_id
        or metadata.get("manifest_index") != manifest_index
        or metadata.get("source_trace") != str(source_path)
        or metadata.get("source_model") != "gpt-5.6-sol"
        or not isinstance(monitoring, Mapping)
        or any(monitoring.get(key) != value for key, value in required_monitoring.items())
        or metadata.get("tool_resource", {}).get("service_enabled") is not True
    ):
        raise ValueError(f"replay metadata differs from the collection contract: {task_id}")


def _load_cgroup_peaks(
    run_dir: Path,
    task_items: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    peaks: dict[str, float] = {}
    evidence: list[dict[str, Any]] = []
    contract = json.loads((run_dir / "collection_contract.json").read_text())
    if contract != _expected_collection_contract():
        raise ValueError("collection contract differs from the freeze")
    for manifest_index, item in enumerate(task_items):
        task_id = str(item["task_id"])
        source_path = _ROOT / str(item["trace"])
        attempt = run_dir / task_id / "attempt_1"
        replay_path = attempt / "trace.jsonl"
        artifact_path = attempt / "resource_observations.json"
        source = _jsonl(source_path)
        replay = _jsonl(replay_path)
        if not replay:
            raise ValueError(f"empty replay trace: {task_id}")
        _validate_replay_metadata(
            replay[0],
            task_id=task_id,
            manifest_index=manifest_index,
            source_path=source_path,
        )
        source_actions = [value for row in source if (value := _action_signature(row))]
        replay_actions = [value for row in replay if (value := _action_signature(row))]
        if replay_actions != source_actions:
            raise ValueError(f"source/replay action mismatch: {task_id}")

        durations_ms = {
            str(row["data"]["tool_call_id"]):
            (float(row["ts_end"]) - float(row["ts_start"])) * 1_000.0
            for row in replay
            if row.get("action_type") == "tool_exec"
            and row.get("data", {}).get("tool_name") == "exec"
        }
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if (
            artifact.get("collection_validity") != "valid"
            or artifact.get("telemetry_quality") != "ok"
            or artifact.get("cleanup") != "ok"
        ):
            raise ValueError(f"invalid resource artifact: {task_id}")
        calls = artifact.get("calls")
        if not isinstance(calls, list) or len(calls) != len(durations_ms):
            raise ValueError(f"resource call coverage differs: {task_id}")
        task_peaks: list[float] = []
        task_samples: list[int] = []
        for call in calls:
            if not isinstance(call, Mapping):
                raise ValueError(f"invalid resource call: {task_id}")
            call_id = call.get("tool_call_id")
            sidecar = call.get("command_window_memory_current")
            if not isinstance(call_id, str) or call_id not in durations_ms:
                raise ValueError(f"resource call identity differs: {task_id}")
            if not isinstance(sidecar, Mapping):
                raise ValueError(f"missing cgroup-memory sidecar: {task_id}:{call_id}")
            if not isinstance(call.get("command_window_rss"), Mapping):
                raise ValueError(f"missing RSS-oracle sidecar: {task_id}:{call_id}")
            peak = sidecar.get("sampled_peak_mb")
            samples = sidecar.get("sample_count")
            if (
                sidecar.get("status") != "ok"
                or sidecar.get("cadence_ms") != 2
                or sidecar.get("error") is not None
                or sidecar.get("read_failures") != 0
                or isinstance(peak, bool)
                or not isinstance(peak, (int, float))
                or not math.isfinite(peak)
                or peak < 0.0
                or isinstance(samples, bool)
                or not isinstance(samples, int)
                or samples < 1
                or (durations_ms[call_id] >= 50.0 and samples < 2)
            ):
                raise ValueError(f"incomplete cgroup-memory sidecar: {task_id}:{call_id}")
            command_id = f"{task_id}:{call_id}"
            if command_id in peaks:
                raise ValueError(f"duplicate command identity: {command_id}")
            peaks[command_id] = float(peak)
            task_peaks.append(float(peak))
            task_samples.append(samples)
        evidence.append(
            {
                "task_id": task_id,
                "actions": len(replay_actions),
                "commands": len(calls),
                "minimum_samples": min(task_samples),
                "maximum_peak_mb": max(task_peaks),
            }
        )
    summary = json.loads((run_dir / "throughput_summary.json").read_text())
    summary_monitoring = summary.get("monitoring")
    if (
        summary.get("attempted_traces") != len(task_items)
        or summary.get("completed_traces") != len(task_items)
        or summary.get("failed_traces") != 0
        or summary.get("concurrency") != 1
        or summary.get("effective_concurrency") != 1
        or summary.get("effective_workers") != 1
        or summary.get("mode") != "cloud_model"
        or summary.get("llm_timing_mode") != "source_scaled"
        or not isinstance(summary_monitoring, Mapping)
        or summary_monitoring.get("resource_requested") != "off"
        or summary_monitoring.get("pmu_requested") != "off"
        or summary_monitoring.get("memory_bandwidth_requested") != "off"
    ):
        raise ValueError("throughput summary differs from the collection contract")
    return peaks, evidence


def _simulate_measured_arm(
    target: Any,
    command_ids: set[str],
    peaks: Mapping[str, float],
    reservations: Mapping[str, float],
) -> tuple[dict[str, Any] | None, list[str]]:
    over_capacity_ids = sorted(
        command_id
        for command_id, peak in peaks.items()
        if peak > _RSS_CAPACITY_MB
    )
    if over_capacity_ids:
        return None, over_capacity_ids
    programs = [
        replace(
            target.programs[task_id],
            commands=tuple(
                replace(command, rss_mb=peaks[command.command_id])
                for command in target.programs[task_id].commands
            ),
        )
        for task_id in _EXPECTED_TASK_IDS
    ]
    return (
        _compact(
            simulate_idle_backfill(
                programs,
                cpu_capacity=_CPU_CAPACITY,
                rss_capacity_mb=_RSS_CAPACITY_MB,
                cpu_work_profiles=target.profiles,
                speculative_eligible_command_ids=command_ids,
                rss_reservations=reservations,
                selection="fcfs",
            )
        ),
        [],
    )


def evaluate(evaluation_git_sha: str) -> dict[str, Any]:
    started = time.monotonic()
    split = json.loads(_frozen_bytes(_SPLIT))
    replay_by_id = {str(item["task_id"]): item for item in split["replay"]}
    task_items = [replay_by_id[task_id] for task_id in _EXPECTED_TASK_IDS]
    predecessor = json.loads(_frozen_bytes(_PREDECESSOR))
    rows = [
        json.loads(line)
        for line in _frozen_bytes(_PREDICTIONS / "rows.jsonl").splitlines()
    ]
    if (
        predecessor.get("schema")
        != "pennylane-xdist-rss-fit-envelope-admission-v1"
        or predecessor.get("evaluation_git_sha")
        != "0b16bc6b5bdfb8350faa8622c8ad6aa6c6d3ce4b"
        or predecessor.get("evidence", {}).get("replay_tasks") != 15
    ):
        raise ValueError("committed fit-envelope predecessor differs")

    target = _load_dataset(
        _EXPECTED_TASK_IDS,
        {task_id: _ROOT / str(replay_by_id[task_id]["trace"]) for task_id in _EXPECTED_TASK_IDS},
    )
    command_ids = set(target.profiles)
    peaks, measurement_evidence = _load_cgroup_peaks(_RUN, task_items)
    if len(command_ids) != 570 or set(peaks) != command_ids:
        raise ValueError("cgroup-memory labels differ from the frozen command population")

    finite_envelopes = {
        (int(key.split("/", 1)[0]), key.split("/", 1)[1]): float(
            cell["scaled_max_rss_mb"]
        )
        for key, cell in predecessor["evidence"]["cells"].items()
        if cell["finite"]
    }
    reservations, activated = continuous_reservations(
        command_ids, rows, finite_envelopes
    )
    if activated != predecessor["evidence"]["activated_reservations"]:
        raise ValueError("fit-envelope activation differs from the predecessor")

    previous = predecessor["arm"]
    unchanged_fields = (
        "command_count",
        "recorded_command_service_s",
        "total_command_service_s",
        "makespan_s",
        "mean_task_completion_s",
        "total_command_queue_s",
        "max_concurrent_commands",
        "speculative_starts",
        "speculative_task_count",
        "speculative_start_ids",
        "capacity_violation",
        "physical_capacity_violation",
        "total_cpu_work_core_s",
        "served_cpu_work_core_s",
    )
    candidate, over_capacity_ids = _simulate_measured_arm(
        target, command_ids, peaks, reservations
    )
    action_unchanged = candidate is not None and all(
        candidate[key] == previous[key] for key in unchanged_fields
    )
    checks = {
        "complete_cgroup_memory_for_all_570_commands": len(peaks) == 570,
        "source_replay_actions_match_for_all_15_tasks": True,
        "no_individual_peak_above_capacity": not over_capacity_ids,
        "fit_envelope_action_unchanged": action_unchanged,
        "zero_static_peak_sum_exposures": candidate is not None
        and candidate["modeled_capacity_exposure_events"] == 0,
        "zero_reservation_capacity_violation": candidate is not None
        and not candidate["capacity_violation"],
        "zero_physical_cpu_capacity_violation": candidate is not None
        and not candidate["physical_capacity_violation"],
        "cpu_work_conserved": candidate is not None
        and math.isclose(
            candidate["total_cpu_work_core_s"],
            candidate["served_cpu_work_core_s"],
            rel_tol=1e-12,
            abs_tol=1e-7,
        ),
    }
    sufficient = all(checks.values())
    return {
        "schema": "pennylane-xdist-cgroup-memory-calibration-v1",
        "status": "development_sufficient" if sufficient else "development_insufficient",
        "claim_bearing": False,
        "protocol_git_sha": _PROTOCOL_GIT_SHA,
        "evaluation_git_sha": evaluation_git_sha,
        "evidence": {
            "tasks": measurement_evidence,
            "task_count": len(measurement_evidence),
            "command_count": len(peaks),
            "minimum_peak_mb": min(peaks.values()),
            "maximum_peak_mb": max(peaks.values()),
            "over_capacity_command_ids": over_capacity_ids,
            "activated_reservations": activated,
            "untouched_pennylane_tasks_read": 0,
        },
        "arm": candidate,
        "prior_fit_envelope_arm": previous,
        "reported_comparisons": predecessor["comparisons"],
        "reported_service_inflation": predecessor["comparisons"]["vs_serial8"][
            "service_inflation"
        ],
        "gate": checks | {"sufficient": sufficient},
        "cost": {
            "prediction_time_agent_calls": 0,
            "gpu_runtime_s": 0.0,
            "evaluation_wall_s": time.monotonic() - started,
        },
        "limitations": [
            "Development-exposed replay tasks; this is calibration, not fresh confirmation.",
            "Per-command cgroup peaks are conservatively summed and are not simultaneous samples.",
            "Cgroup memory includes workload state such as page cache retained between commands.",
            "Action identity is checked, but replay outputs and wall times need not match the source.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_OUTPUT / "result.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate(_require_clean_checkout())
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
