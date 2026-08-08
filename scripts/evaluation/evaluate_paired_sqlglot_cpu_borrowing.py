#!/usr/bin/env python3
"""Run the frozen paired SQLGlot hard-quota versus CPU-borrowing replay."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import statistics
import traceback
from typing import Any

import numpy as np

from trace_collect.simulator import simulate
from trace_collect.simulate_openclaw import (
    OPENCLAW_EXEC_TIMEOUT_FLOOR_ENV,
    OPENCLAW_PAIRED_WORKLOAD_CONTRACT_ENV,
    _paired_replay_actions,
)


ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "analysis/development/sqlglot-relational-task-split.json"
SOURCE = (
    ROOT
    / "traces/swe-rebench/gpt-5.6-sol"
    / "sqlglot-prev100-c2-fast-requested-ebpf-20260804"
)
TASKS = ROOT / "data/swe-rebench/tasks.json"
LEGACY_REMAINING_RESULT_DIR = (
    ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot26-paired-cpu-borrowing-remaining-v1"
)
LEGACY_REMAINING_REPLAY_DIR = (
    ROOT
    / "traces/swe-rebench/gpt-5.6-sol"
    / "sqlglot26-paired-cpu-borrowing-remaining-v1"
)
PAIR09_PREFLIGHT_RESULT_DIR = (
    ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot-pair09-cpu-borrowing-contract-v2-preflight"
)
PAIR09_PREFLIGHT_REPLAY_DIR = (
    ROOT
    / "traces/swe-rebench/gpt-5.6-sol"
    / "sqlglot-pair09-cpu-borrowing-contract-v2-preflight"
)
SAFETY_GUARD_REJECTION = (
    "Error: Command blocked by safety guard (dangerous pattern detected)\n\n"
    "[Analyze the error above and try a different approach.]"
)
EXEC_TIMEOUT_FLOOR_S = 3_600
SELECTION_SEED = 20_260_807
ARM_ARGS = {
    "hard_two": ("--cpuset-cpus", "0-7", "--cpu-shares", "1024", "--cpus", "2"),
    "burstable_two": ("--cpuset-cpus", "0-7", "--cpu-shares", "1024"),
}
COHORTS: dict[str, dict[str, Any]] = {
    "initial24": {
        "selection_start": 0,
        "selection_count": 24,
        "arm_order_seed": 20_260_808,
        "bootstrap_seed": 20_260_809,
        "minimum_improving_pairs": 9,
        "schema": "sqlglot-paired-cpu-borrowing-timeout-floor-v2",
        "result_dir": ROOT
        / "analysis/results/tool-resource-5-3-3-3-20260804"
        / "sqlglot24-paired-cpu-borrowing-timeout-floor-v2",
        "replay_dir": ROOT
        / "traces/swe-rebench/gpt-5.6-sol"
        / "sqlglot24-paired-cpu-borrowing-timeout-floor-v2",
    },
    "remaining26": {
        "selection_start": 24,
        "selection_count": 26,
        "arm_order_seed": 20_260_810,
        "bootstrap_seed": 20_260_811,
        "minimum_improving_pairs": 10,
        "schema": "sqlglot-paired-cpu-borrowing-remaining-v1",
        "result_dir": ROOT
        / "analysis/results/tool-resource-5-3-3-3-20260804"
        / "sqlglot26-paired-cpu-borrowing-remaining-v1",
        "replay_dir": ROOT
        / "traces/swe-rebench/gpt-5.6-sol"
        / "sqlglot26-paired-cpu-borrowing-remaining-v1",
    },
    "pair09_contract_v2": {
        "selection_start": 40,
        "selection_count": 2,
        "arm_orders": [["burstable_two", "hard_two"]],
        "bootstrap_seed": 20_260_811,
        "minimum_improving_pairs": 1,
        "pair_number_offset": 8,
        "mechanism_only": True,
        "paired_workload_contract": True,
        "schema": "sqlglot-paired-cpu-borrowing-contract-v2-preflight",
        "result_dir": ROOT
        / "analysis/results/tool-resource-5-3-3-3-20260804"
        / "sqlglot-pair09-cpu-borrowing-contract-v2-preflight",
        "replay_dir": ROOT
        / "traces/swe-rebench/gpt-5.6-sol"
        / "sqlglot-pair09-cpu-borrowing-contract-v2-preflight",
    },
    "remaining26_contract_v2": {
        "selection_start": 24,
        "selection_count": 26,
        "arm_order_seed": 20_260_810,
        "bootstrap_seed": 20_260_811,
        "minimum_improving_pairs": 10,
        "paired_workload_contract": True,
        "schema": "sqlglot-paired-cpu-borrowing-contract-v2",
        "result_dir": ROOT
        / "analysis/results/tool-resource-5-3-3-3-20260804"
        / "sqlglot26-paired-cpu-borrowing-contract-v2",
        "replay_dir": ROOT
        / "traces/swe-rebench/gpt-5.6-sol"
        / "sqlglot26-paired-cpu-borrowing-contract-v2",
    },
    "remaining26_compatible_v2": {
        "selection_start": 24,
        "selection_count": 26,
        "arm_order_seed": 20_260_810,
        "bootstrap_seed": 20_260_811,
        "minimum_improving_pairs": 10,
        "paired_workload_contract": True,
        "compatible_prefix_pairs": 9,
        "schema": "sqlglot-paired-cpu-borrowing-compatible-v2",
        "result_dir": ROOT
        / "analysis/results/tool-resource-5-3-3-3-20260804"
        / "sqlglot26-paired-cpu-borrowing-compatible-v2",
        "replay_dir": ROOT
        / "traces/swe-rebench/gpt-5.6-sol"
        / "sqlglot26-paired-cpu-borrowing-compatible-v2",
    },
    "quartet48_contract_v2": {
        "selection_start": 0,
        "selection_count": 48,
        "group_size": 4,
        "arm_order_seed": 20_260_812,
        "bootstrap_seed": 20_260_813,
        "minimum_improving_pairs": 9,
        "paired_workload_contract": True,
        "schema": "sqlglot-quartet-cpu-borrowing-contract-v2",
        "result_dir": ROOT
        / "analysis/results/tool-resource-5-3-3-3-20260804"
        / "sqlglot48-quartet-cpu-borrowing-contract-v2",
        "replay_dir": ROOT
        / "traces/swe-rebench/gpt-5.6-sol"
        / "sqlglot48-quartet-cpu-borrowing-contract-v2",
    },
    "rolling48_contract_v1": {
        "selection_start": 0,
        "selection_count": 48,
        "group_size": 48,
        "queue_concurrency": 4,
        "queue_workers": 1,
        "decision_unit": "queue",
        "immediate_refill": True,
        "cleanup_images": True,
        "arm_level_resume": True,
        "makespan_source": "throughput_summary.wall_time_s",
        "arm_orders": [["burstable_two", "hard_two"]],
        "bootstrap_seed": 20_260_814,
        "minimum_improving_pairs": 1,
        "paired_workload_contract": True,
        "schema": "sqlglot-rolling-cpu-borrowing-contract-v1",
        "result_dir": ROOT
        / "analysis/results/tool-resource-5-3-3-3-20260804"
        / "sqlglot48-rolling-cpu-borrowing-contract-v1",
        "replay_dir": ROOT
        / "traces/swe-rebench/gpt-5.6-sol"
        / "sqlglot48-rolling-cpu-borrowing-contract-v1",
    },
}


def _protocol(cohort: str = "initial24") -> dict[str, Any]:
    config = COHORTS[cohort]
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    task_ids = np.array(sorted(split["validation"]), dtype=object)
    np.random.Generator(np.random.PCG64(SELECTION_SEED)).shuffle(task_ids)
    start = int(config["selection_start"])
    stop = start + int(config["selection_count"])
    selected = [str(value) for value in task_ids[start:stop]]
    group_size = int(config.get("group_size", 2))
    if len(selected) % group_size:
        raise ValueError("selected task count is not divisible by group size")
    pairs = [
        selected[index : index + group_size]
        for index in range(0, len(selected), group_size)
    ]
    if "arm_orders" in config:
        arm_orders = [list(order) for order in config["arm_orders"]]
    else:
        arm_rng = np.random.Generator(np.random.PCG64(config["arm_order_seed"]))
        arm_orders = []
        for _pair in pairs:
            order = np.array(["hard_two", "burstable_two"], dtype=object)
            arm_rng.shuffle(order)
            arm_orders.append([str(value) for value in order])
    if len(arm_orders) != len(pairs):
        raise ValueError("arm order count differs from pair count")
    traces = {
        task_id: SOURCE / task_id / "attempt_1" / "trace.jsonl" for task_id in selected
    }
    missing = [str(path) for path in traces.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"selected source traces are missing: {missing}")
    protocol = {
        "cohort": cohort,
        "selection_seed": SELECTION_SEED,
        "selection_range": [start, stop],
        "arm_order_seed": config.get("arm_order_seed"),
        "bootstrap_seed": config["bootstrap_seed"],
        "minimum_improving_pairs": config["minimum_improving_pairs"],
        "schema": config["schema"],
        "mechanism_only": bool(config.get("mechanism_only", False)),
        "paired_workload_contract": bool(config.get("paired_workload_contract", False)),
        "selected_task_ids": selected,
        "pairs": [
            {
                "pair": index + 1 + int(config.get("pair_number_offset", 0)),
                "task_ids": pair,
                "arm_order": arm_orders[index],
            }
            for index, pair in enumerate(pairs)
        ],
        "source_dir": str(SOURCE),
        "task_source": str(TASKS),
        "exec_timeout_floor_s": EXEC_TIMEOUT_FLOOR_S,
    }
    if config.get("compatible_prefix_pairs"):
        protocol["compatible_prefix_pairs"] = int(config["compatible_prefix_pairs"])
    if group_size != 2:
        protocol["group_size"] = group_size
        protocol["decision_unit"] = str(config.get("decision_unit", "group"))
    if "queue_concurrency" in config:
        protocol["queue_concurrency"] = int(config["queue_concurrency"])
        protocol["queue_workers"] = int(config["queue_workers"])
        protocol["immediate_refill"] = bool(config["immediate_refill"])
        protocol["cleanup_images"] = bool(config["cleanup_images"])
        protocol["arm_level_resume"] = bool(config["arm_level_resume"])
        protocol["makespan_source"] = str(config["makespan_source"])
    return protocol


def _write_manifest(path: Path, task_ids: list[str]) -> None:
    payload = {
        "version": 1,
        "defaults": {"task_source": str(TASKS)},
        "traces": [
            {"trace": str(SOURCE / task_id / "attempt_1" / "trace.jsonl")}
            for task_id in task_ids
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _action_sequences(trace_file: Path) -> dict[str, list[list[str]]]:
    sequences: dict[str, list[list[str]]] = {}
    with trace_file.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") != "action":
                continue
            data = record.get("data") or {}
            task_id = str(data.get("task_instance_id") or record.get("instance_id"))
            sequences.setdefault(task_id, []).append(
                [
                    str(record.get("agent_id")),
                    str(record.get("action_type")),
                    str(record.get("action_id")),
                ]
            )
    return sequences


def _source_success_replay_timeout_count(
    request: dict[str, Any], trace_path: Path
) -> int:
    source = [
        action
        for action in request["source_actions"]
        if action.get("action_type") in {"llm_call", "tool_exec"}
    ]
    replay = [
        record
        for record in (
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if record.get("type") == "action"
    ]
    if len(source) != len(replay):
        raise AssertionError("source and replay action counts differ")
    count = 0
    for source_action, replay_action in zip(source, replay, strict=True):
        source_data = source_action.get("data") or {}
        replay_data = replay_action.get("data") or {}
        source_identity = (
            source_action.get("action_type"),
            source_action.get("action_id"),
            source_data.get("tool_name"),
            source_data.get("tool_call_id"),
        )
        replay_identity = (
            replay_action.get("action_type"),
            replay_action.get("action_id"),
            replay_data.get("tool_name"),
            replay_data.get("tool_call_id"),
        )
        if source_identity != replay_identity:
            raise AssertionError(
                f"source and replay action identities differ: "
                f"{source_identity!r} != {replay_identity!r}"
            )
        if (
            source_action.get("action_type") == "tool_exec"
            and source_data.get("tool_name") == "exec"
            and source_data.get("success") is not False
            and replay_data.get("success") is False
        ):
            lines = {
                line.strip()
                for line in str(replay_data.get("tool_result") or "").splitlines()
            }
            if lines & {
                "[timeout]",
                "Error: [timeout]",
                "[resource_timeout]",
                "Error: [resource_timeout]",
                "[resource_stall_timeout]",
                "Error: [resource_stall_timeout]",
            }:
                count += 1
    return count


def _task_artifacts(
    arm_dir: Path,
    task_id: str,
    arm: str,
    *,
    paired_workload_contract: bool | None,
) -> dict[str, Any]:
    attempt = arm_dir / task_id / "attempt_1"
    startup_path = attempt / "container_startup.json"
    status_path = attempt / "openclaw_host_replay_status.json"
    resources_path = attempt / "resources.json"
    request_path = attempt / "openclaw_host_replay_request.json"
    for path in (startup_path, status_path, resources_path, request_path):
        if not path.is_file():
            raise AssertionError(f"missing task artifact: {path}")

    startup = json.loads(startup_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    resources = json.loads(resources_path.read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("exec_timeout_floor_s") != EXEC_TIMEOUT_FLOOR_S:
        raise AssertionError(f"wrong timeout floor for {task_id} {arm}")
    if paired_workload_contract is None:
        if "paired_workload_contract" in request:
            raise AssertionError(
                f"unexpected replay contract field for {task_id} {arm}"
            )
    elif request.get("paired_workload_contract") is not paired_workload_contract:
        raise AssertionError(f"wrong replay contract for {task_id} {arm}")
    replay_action_contract = dict(request.get("replay_action_contract") or {})
    if (
        paired_workload_contract
        and replay_action_contract.get("require_source_outcome_match") is not False
    ):
        raise AssertionError(f"source outcome matching enabled for {task_id} {arm}")
    start_phase = next(
        phase for phase in startup["phases"] if phase["name"] == "start_task_container"
    )
    expected_controls = {
        "nano_cpus": 2_000_000_000 if arm == "hard_two" else 0,
        "cpu_shares": 1024,
        "cpuset_cpus": "0-7",
    }
    if startup.get("status") != "success" or start_phase.get("status") != "success":
        raise AssertionError(f"container startup failed for {task_id} {arm}")
    if start_phase.get("start_extra_args") != list(ARM_ARGS[arm]):
        raise AssertionError(f"wrong requested CPU controls for {task_id} {arm}")
    if start_phase.get("cpu_controls") != expected_controls:
        raise AssertionError(f"wrong observed CPU controls for {task_id} {arm}")

    required_status = {
        "success": True,
        "missing_source_action_count": 0,
        "action_sequence_matches": True,
        "error": None,
    }
    if not paired_workload_contract:
        required_status["unexpected_replay_failed_actions"] = 0
    for key, expected in required_status.items():
        if status.get(key) != expected:
            raise AssertionError(
                f"invalid replay status for {task_id} {arm}: {key}={status.get(key)!r}"
            )
    if status.get("emitted_actions") != status.get("expected_actions"):
        raise AssertionError(f"incomplete replay for {task_id} {arm}")
    timeout_count = _source_success_replay_timeout_count(
        request, attempt / "openclaw_host_replay.jsonl"
    )
    if timeout_count:
        raise AssertionError(
            f"source-success replay timeout for {task_id} {arm}: {timeout_count}"
        )

    summary = resources.get("summary") or {}
    if summary.get("monitoring_disabled") is not False:
        raise AssertionError(f"resource monitoring disabled for {task_id} {arm}")
    final_state = summary.get("container_final_state") or {}
    memory_events = final_state.get("memory_events") or {}
    if (
        final_state.get("status") != "running"
        or final_state.get("running") is not True
        or final_state.get("oom_killed") is not False
        or final_state.get("exit_code") != 0
        or memory_events.get("oom", 0) != 0
        or memory_events.get("oom_kill", 0) != 0
    ):
        raise AssertionError(f"invalid final container state for {task_id} {arm}")
    return {
        "task_id": task_id,
        "attempt_dir": str(attempt),
        "replay_status": {
            key: status.get(key)
            for key in (
                "success",
                "expected_actions",
                "emitted_actions",
                "source_failed_actions",
                "replay_failed_actions",
                "unexpected_replay_failed_actions",
                "missing_source_action_count",
                "action_sequence_matches",
            )
        },
        "resource_sample_count": len(resources.get("samples") or []),
        "cpu_controls": expected_controls,
        "exec_timeout_floor_s": request["exec_timeout_floor_s"],
        "replay_action_contract": replay_action_contract,
        "source_success_replay_timeout_count": timeout_count,
        "final_container_state": final_state,
    }


async def _run_arm(
    *,
    pair: int,
    task_ids: list[str],
    arm: str,
    manifest: Path,
    replay_dir: Path,
    paired_workload_contract: bool,
    concurrency: int | None = None,
    workers: int | None = None,
    makespan_source: str = "max_task_elapsed_s",
    cleanup_images: bool = False,
) -> dict[str, Any]:
    arm_dir = replay_dir / f"pair_{pair:02d}" / arm
    concurrency = len(task_ids) if concurrency is None else concurrency
    workers = concurrency if workers is None else workers
    trace_file = await simulate(
        manifest=manifest,
        output_dir=arm_dir,
        concurrency=concurrency,
        workers=workers,
        prep_concurrency=concurrency,
        container_executable="docker",
        network_mode="host",
        command_timeout_s=3_600.0,
        replay_speed=20.0,
        resource_monitoring="on",
        pmu_monitoring="off",
        memory_bandwidth_monitoring="off",
        container_start_extra_args=ARM_ARGS[arm],
        cleanup_images=cleanup_images,
    )
    summary_path = arm_dir / "throughput_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("attempted_traces") != len(task_ids):
        raise AssertionError(f"invalid throughput summary for pair {pair} {arm}")
    stats = {str(item["agent_id"]): item for item in summary["tasks"]}
    if set(stats) != set(task_ids):
        raise AssertionError(f"wrong tasks for pair {pair} {arm}")
    if any(item["failed_action_count"] for item in stats.values()):
        raise AssertionError(f"replay action failure for pair {pair} {arm}")
    if makespan_source == "throughput_summary.wall_time_s":
        expected_summary = {
            "concurrency": concurrency,
            "effective_concurrency": min(concurrency, len(task_ids)),
            "workers": workers,
            "scheduler_mode": "bounded_queue",
        }
        for key, expected in expected_summary.items():
            if summary.get(key) != expected:
                raise AssertionError(
                    f"wrong rolling queue summary for {pair} {arm}: "
                    f"{key}={summary.get(key)!r}"
                )
        makespan_s = float(summary["wall_time_s"])
    elif makespan_source == "max_task_elapsed_s":
        makespan_s = max(float(item["elapsed_s"]) for item in stats.values())
    else:
        raise ValueError(f"unsupported makespan source: {makespan_source}")
    return {
        "arm": arm,
        "output_dir": str(arm_dir),
        "trace_file": str(trace_file),
        "summary_path": str(summary_path),
        "pair_makespan_s": makespan_s,
        "tasks": [
            _task_artifacts(
                arm_dir,
                task_id,
                arm,
                paired_workload_contract=paired_workload_contract,
            )
            for task_id in task_ids
        ],
        "task_stats": [stats[task_id] for task_id in task_ids],
        "action_sequences": _action_sequences(trace_file),
    }


def _legacy_contract_compatibility(
    source_actions: list[dict[str, Any]],
    replay_actions_by_arm: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    _, contract = _paired_replay_actions(source_actions)
    seeds = contract["pytest_random_seeds"]
    if seeds:
        raise AssertionError("legacy reuse contains a pytest-randomly seed")
    failed_call_ids = set(contract["exec_timeout_floor_exempt_call_ids"])
    source_by_call = {
        str((action.get("data") or {}).get("tool_call_id")): action
        for action in source_actions
        if action.get("action_type") == "tool_exec"
    }
    for call_id in failed_call_ids:
        source = source_by_call[call_id].get("data") or {}
        source_result = str(source.get("tool_result") or "")
        if source_result != SAFETY_GUARD_REJECTION:
            raise AssertionError(
                f"legacy source failure is not a pre-execution safety rejection: {call_id}"
            )
        for arm, replay_actions in replay_actions_by_arm.items():
            matches = [
                action
                for action in replay_actions
                if str((action.get("data") or {}).get("tool_call_id")) == call_id
            ]
            if len(matches) != 1:
                raise AssertionError(
                    f"legacy replay call identity differs for {call_id} {arm}"
                )
            replay = matches[0].get("data") or {}
            if (
                replay.get("success") is not False
                or str(replay.get("tool_result") or "") != source_result
            ):
                raise AssertionError(
                    f"legacy source failure was not preserved for {call_id} {arm}"
                )
    return {
        "pytest_seed_count": 0,
        "preserved_preexecution_failure_count": len(failed_call_ids),
    }


def _reconstruct_saved_run(
    declared: dict[str, Any],
    stored_run: dict[str, Any],
    *,
    paired_workload_contract: bool | None,
    expected_replay_root: Path,
) -> dict[str, Any]:
    pair = int(declared["pair"])
    stored_arms = stored_run.get("arms")
    if (
        stored_run.get("pair") != pair
        or not isinstance(stored_arms, list)
        or [item.get("arm") for item in stored_arms] != declared["arm_order"]
    ):
        raise AssertionError(f"saved pair {pair} identity or arm order differs")
    arms = []
    for stored_arm in stored_arms:
        arm = str(stored_arm["arm"])
        arm_dir = Path(stored_arm["output_dir"])
        trace_file = Path(stored_arm["trace_file"])
        summary_path = Path(stored_arm["summary_path"])
        expected_arm_dir = expected_replay_root / f"pair_{pair:02d}" / arm
        if (
            arm_dir.resolve() != expected_arm_dir.resolve()
            or trace_file.parent.resolve() != arm_dir.resolve()
            or not trace_file.is_file()
            or summary_path.resolve() != (arm_dir / "throughput_summary.json").resolve()
            or not summary_path.is_file()
        ):
            raise AssertionError(f"saved pair {pair} {arm} paths are invalid")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stats = {str(item["agent_id"]): item for item in summary.get("tasks", [])}
        if (
            summary.get("attempted_traces") != 2
            or set(stats) != set(declared["task_ids"])
            or any(item["failed_action_count"] for item in stats.values())
        ):
            raise AssertionError(f"saved pair {pair} {arm} summary is invalid")
        task_artifacts = [
            _task_artifacts(
                arm_dir,
                task_id,
                arm,
                paired_workload_contract=paired_workload_contract,
            )
            for task_id in declared["task_ids"]
        ]
        for task in task_artifacts:
            request = json.loads(
                (
                    Path(task["attempt_dir"]) / "openclaw_host_replay_request.json"
                ).read_text(encoding="utf-8")
            )
            task_id = task["task_id"]
            expected_source = SOURCE / task_id / "attempt_1" / "trace.jsonl"
            if (
                Path(request.get("source_trace", "")).resolve()
                != expected_source.resolve()
                or request.get("replay_speed") != 20.0
                or request.get("task_instance_id") != task_id
            ):
                raise AssertionError(
                    f"saved pair {pair} {arm} source provenance differs for {task_id}"
                )
        arms.append(
            {
                "arm": arm,
                "output_dir": str(arm_dir),
                "trace_file": str(trace_file),
                "summary_path": str(summary_path),
                "pair_makespan_s": max(
                    float(item["elapsed_s"]) for item in stats.values()
                ),
                "tasks": task_artifacts,
                "task_stats": [stats[task_id] for task_id in declared["task_ids"]],
                "action_sequences": _action_sequences(trace_file),
            }
        )
    if arms[0]["action_sequences"] != arms[1]["action_sequences"]:
        raise AssertionError(f"saved pair {pair} action identity differs")
    return {"pair": pair, "arms": arms}


def _load_compatible_prefix(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    legacy = json.loads(
        (LEGACY_REMAINING_RESULT_DIR / "partial.json").read_text(encoding="utf-8")
    )
    expected_legacy_protocol = _protocol("remaining26")
    expected_legacy_protocol.pop("mechanism_only")
    expected_legacy_protocol.pop("paired_workload_contract")
    if legacy.get("protocol") != expected_legacy_protocol:
        raise AssertionError("legacy source protocol differs from the frozen protocol")
    stored_legacy = legacy.get("runs")
    if not isinstance(stored_legacy, list) or len(stored_legacy) != 8:
        raise AssertionError("legacy compatible prefix must contain eight pairs")
    runs = []
    for declared, stored in zip(protocol["pairs"][:8], stored_legacy, strict=True):
        run = _reconstruct_saved_run(
            declared,
            stored,
            paired_workload_contract=None,
            expected_replay_root=LEGACY_REMAINING_REPLAY_DIR,
        )
        compatibility_by_task = {}
        for task_id in declared["task_ids"]:
            requests = []
            replay_actions_by_arm = {}
            for arm_data in run["arms"]:
                attempt = Path(
                    next(
                        task["attempt_dir"]
                        for task in arm_data["tasks"]
                        if task["task_id"] == task_id
                    )
                )
                request = json.loads(
                    (attempt / "openclaw_host_replay_request.json").read_text(
                        encoding="utf-8"
                    )
                )
                requests.append(request["source_actions"])
                replay_actions_by_arm[arm_data["arm"]] = [
                    record
                    for record in (
                        json.loads(line)
                        for line in (attempt / "openclaw_host_replay.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                        if line.strip()
                    )
                    if record.get("type") == "action"
                ]
            if requests[0] != requests[1]:
                raise AssertionError(f"saved source actions differ for {task_id}")
            compatibility_by_task[task_id] = _legacy_contract_compatibility(
                requests[0], replay_actions_by_arm
            )
        run["provenance"] = {
            "kind": "legacy_contract_compatible",
            "source_partial": str(LEGACY_REMAINING_RESULT_DIR / "partial.json"),
            "compatibility_by_task": compatibility_by_task,
        }
        runs.append(run)

    preflight = json.loads(
        (PAIR09_PREFLIGHT_RESULT_DIR / "result.json").read_text(encoding="utf-8")
    )
    if (
        preflight.get("status") != "go"
        or preflight.get("protocol") != _protocol("pair09_contract_v2")
        or len(preflight.get("pairs", [])) != 1
    ):
        raise AssertionError("pair-9 contract preflight is not a valid GO")
    stored_pair9 = {
        "pair": 9,
        "arms": preflight["pairs"][0]["arms"],
    }
    pair9 = _reconstruct_saved_run(
        protocol["pairs"][8],
        stored_pair9,
        paired_workload_contract=True,
        expected_replay_root=PAIR09_PREFLIGHT_REPLAY_DIR,
    )
    pair9["provenance"] = {
        "kind": "contract_v2_preflight",
        "source_result": str(PAIR09_PREFLIGHT_RESULT_DIR / "result.json"),
    }
    runs.append(pair9)
    return runs


def _aggregate(protocol: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair = {item["pair"]: item for item in runs}
    decision_unit = str(protocol.get("decision_unit", "pair"))
    plural = f"{decision_unit}s"
    pairs: list[dict[str, Any]] = []
    for declared in protocol["pairs"]:
        pair = declared["pair"]
        arms = {item["arm"]: item for item in by_pair[pair]["arms"]}
        hard = arms["hard_two"]
        burst = arms["burstable_two"]
        if hard["action_sequences"] != burst["action_sequences"]:
            raise AssertionError(f"action identity differs for pair {pair}")
        hard_contracts = {
            task["task_id"]: task["replay_action_contract"] for task in hard["tasks"]
        }
        burst_contracts = {
            task["task_id"]: task["replay_action_contract"] for task in burst["tasks"]
        }
        if hard_contracts != burst_contracts:
            raise AssertionError(f"replay contract differs for pair {pair}")
        improvement = (hard["pair_makespan_s"] - burst["pair_makespan_s"]) / hard[
            "pair_makespan_s"
        ]
        pair_result = {
            decision_unit: pair,
            "task_ids": declared["task_ids"],
            "arm_order": declared["arm_order"],
            "hard_two_makespan_s": hard["pair_makespan_s"],
            "burstable_two_makespan_s": burst["pair_makespan_s"],
            "improvement_fraction": improvement,
            "arms": by_pair[pair]["arms"],
        }
        if "provenance" in by_pair[pair]:
            pair_result["provenance"] = by_pair[pair]["provenance"]
        pairs.append(pair_result)
    improvements = np.array([item["improvement_fraction"] for item in pairs])
    mean = float(improvements.mean())
    improving = int((improvements > 0).sum())
    minimum_improving = int(protocol["minimum_improving_pairs"])
    if protocol.get("mechanism_only"):
        return {
            "schema": protocol["schema"],
            "status": "go",
            "protocol": protocol,
            "comparison": {
                "claim": "replay_contract_validity_only",
                "mean_improvement_fraction": mean,
                "median_improvement_fraction": float(statistics.median(improvements)),
                "improving_pairs": improving,
                "pair_count": len(pairs),
                "gate": {"all_validity_checks_passed": True},
            },
            "pairs": pairs,
        }
    if decision_unit == "queue":
        gate = {
            "makespan_improvement_at_least_5_percent": mean >= 0.05,
            "all_validity_checks_passed": True,
        }
        result = {
            "schema": protocol["schema"],
            "status": "go" if all(gate.values()) else "no_go",
            "protocol": protocol,
            "comparison": {
                "makespan_improvement_fraction": mean,
                "queue_count": len(pairs),
                "gate": gate,
            },
            plural: pairs,
        }
    else:
        rng = np.random.Generator(np.random.PCG64(protocol["bootstrap_seed"]))
        draws = improvements[
            rng.integers(0, len(improvements), size=(10_000, len(improvements)))
        ].mean(axis=1)
        interval = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
        gate = {
            "mean_improvement_at_least_5_percent": mean >= 0.05,
            "bootstrap_lower_above_zero": interval[0] > 0.0,
            f"at_least_{minimum_improving}_of_{len(pairs)}_{plural}_improve": (
                improving >= minimum_improving
            ),
            "all_validity_checks_passed": True,
        }
        result = {
            "schema": protocol["schema"],
            "status": "go" if all(gate.values()) else "no_go",
            "protocol": protocol,
            "comparison": {
                "mean_improvement_fraction": mean,
                "median_improvement_fraction": float(
                    statistics.median(improvements)
                ),
                f"improving_{plural}": improving,
                f"{decision_unit}_count": len(pairs),
                f"ci95_paired_{decision_unit}_bootstrap": interval,
                "bootstrap_draws": 10_000,
                "gate": gate,
            },
            plural: pairs,
        }
    if int(protocol.get("group_size", 2)) > 2:
        task_deltas = []
        hard_elapsed = []
        burst_elapsed = []
        for group in pairs:
            arms = {arm["arm"]: arm for arm in group["arms"]}
            hard = {
                str(task["agent_id"]): float(task["elapsed_s"])
                for task in arms["hard_two"]["task_stats"]
            }
            burst = {
                str(task["agent_id"]): float(task["elapsed_s"])
                for task in arms["burstable_two"]["task_stats"]
            }
            if set(hard) != set(burst):
                raise AssertionError(
                    f"task completion identity differs for {decision_unit} "
                    f"{group[decision_unit]}"
                )
            for task_id in group["task_ids"]:
                hard_s = hard[task_id]
                burst_s = burst[task_id]
                hard_elapsed.append(hard_s)
                burst_elapsed.append(burst_s)
                task_deltas.append(
                    {
                        decision_unit: group[decision_unit],
                        "task_id": task_id,
                        "hard_two_s": hard_s,
                        "burstable_two_s": burst_s,
                        "burstable_slowdown_fraction": (burst_s - hard_s) / hard_s,
                    }
                )
        slowdowns = np.array(
            [item["burstable_slowdown_fraction"] for item in task_deltas]
        )
        task_metric = "task_runtime" if decision_unit == "queue" else "task_completion"
        result[task_metric] = {
            "hard_two_mean_s": float(np.mean(hard_elapsed)),
            "burstable_two_mean_s": float(np.mean(burst_elapsed)),
            "hard_two_p95_s": float(np.quantile(hard_elapsed, 0.95)),
            "burstable_two_p95_s": float(np.quantile(burst_elapsed, 0.95)),
            "slowed_more_than_10_percent": int((slowdowns > 0.10).sum()),
            "slowed_more_than_25_percent": int((slowdowns > 0.25).sum()),
            "paired_task_deltas": task_deltas,
        }
    return result


def _load_resume_runs(
    protocol: dict[str, Any], result_dir: Path, replay_dir: Path
) -> list[dict[str, Any]]:
    partial_path = result_dir / "partial.json"
    if not partial_path.is_file():
        raise FileNotFoundError(f"resume requires {partial_path}")
    stored_protocol_path = result_dir / "protocol.json"
    if (
        not stored_protocol_path.is_file()
        or json.loads(stored_protocol_path.read_text(encoding="utf-8")) != protocol
    ):
        raise AssertionError("stored protocol differs from the frozen protocol")
    payload = json.loads(partial_path.read_text(encoding="utf-8"))
    if payload.get("protocol") != protocol:
        raise AssertionError("resume protocol differs from the frozen protocol")
    runs = payload.get("runs")
    arm_level_resume = bool(protocol.get("arm_level_resume", False))
    if (
        not isinstance(runs, list)
        or not runs
        or len(runs) > len(protocol["pairs"])
        or (len(runs) == len(protocol["pairs"]) and not arm_level_resume)
    ):
        raise AssertionError("resume requires a non-empty incomplete run prefix")

    for index, (declared, run) in enumerate(
        zip(protocol["pairs"][: len(runs)], runs, strict=True)
    ):
        pair = int(declared["pair"])
        if run.get("pair") != pair:
            raise AssertionError(f"resume pair order differs at pair {pair}")
        arms = run.get("arms")
        partial_last_arm = (
            arm_level_resume and index == len(runs) - 1 and len(arms or []) == 1
        )
        if not isinstance(arms, list) or (len(arms) != 2 and not partial_last_arm):
            raise AssertionError(f"resume pair {pair} is not complete")
        if [item.get("arm") for item in arms] != declared["arm_order"][: len(arms)]:
            raise AssertionError(f"resume arm order differs at pair {pair}")

        by_arm = {str(item["arm"]): item for item in arms}
        for arm, arm_data in by_arm.items():
            expected_dir = replay_dir / f"pair_{pair:02d}" / arm
            if Path(arm_data["output_dir"]).resolve() != expected_dir.resolve():
                raise AssertionError(f"resume output path differs at pair {pair} {arm}")
            trace_file = Path(arm_data["trace_file"])
            summary_path = Path(arm_data["summary_path"])
            if (
                trace_file.parent.resolve() != expected_dir.resolve()
                or not trace_file.is_file()
                or summary_path.resolve()
                != (expected_dir / "throughput_summary.json").resolve()
                or not summary_path.is_file()
            ):
                raise AssertionError(
                    f"resume output path is invalid at pair {pair} {arm}"
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected_tasks = len(declared["task_ids"])
            if summary.get("attempted_traces") != expected_tasks:
                raise AssertionError(f"resume summary is invalid at pair {pair} {arm}")
            summary_tasks = summary.get("tasks")
            if (
                not isinstance(summary_tasks, list)
                or len(summary_tasks) != expected_tasks
            ):
                raise AssertionError(
                    f"resume summary tasks are invalid at pair {pair} {arm}"
                )
            stats = {str(item["agent_id"]): item for item in summary_tasks}
            if set(stats) != set(declared["task_ids"]) or any(
                item["failed_action_count"] for item in stats.values()
            ):
                raise AssertionError(
                    f"resume summary tasks differ at pair {pair} {arm}"
                )
            stored_task_stats = arm_data.get("task_stats")
            if (
                not isinstance(stored_task_stats, list)
                or len(stored_task_stats) != expected_tasks
                or {
                    str(item["agent_id"]): item for item in stored_task_stats
                }
                != stats
            ):
                raise AssertionError(f"resume task stats differ at pair {pair} {arm}")
            if protocol.get("makespan_source") == "throughput_summary.wall_time_s":
                expected_summary = {
                    "concurrency": protocol["queue_concurrency"],
                    "effective_concurrency": min(
                        protocol["queue_concurrency"], expected_tasks
                    ),
                    "workers": protocol["queue_workers"],
                    "scheduler_mode": "bounded_queue",
                }
                for key, expected in expected_summary.items():
                    if summary.get(key) != expected:
                        raise AssertionError(
                            f"resume queue summary differs at pair {pair} {arm}: "
                            f"{key}={summary.get(key)!r}"
                        )
                makespan = float(summary["wall_time_s"])
            else:
                makespan = max(
                    float(item["elapsed_s"]) for item in stats.values()
                )
            if arm_data.get("pair_makespan_s") != makespan:
                raise AssertionError(f"resume makespan differs at pair {pair} {arm}")
            sequences = _action_sequences(trace_file)
            if arm_data.get("action_sequences") != sequences:
                raise AssertionError(
                    f"resume action record differs at pair {pair} {arm}"
                )
            for task_id in declared["task_ids"]:
                _task_artifacts(
                    expected_dir,
                    task_id,
                    arm,
                    paired_workload_contract=protocol["paired_workload_contract"],
                )
        if len(by_arm) == 2 and (
            by_arm["hard_two"]["action_sequences"]
            != by_arm["burstable_two"]["action_sequences"]
        ):
            raise AssertionError(f"resume action identity differs at pair {pair}")
    return runs


def _preserve_interruption_result(
    result_path: Path, protocol: dict[str, Any]
) -> list[Path]:
    preserved = sorted(result_path.parent.glob("interruption-*.json"))
    if not result_path.exists():
        return preserved
    if result_path.stat().st_size:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") != "invalid" or payload.get("protocol") != protocol:
            raise AssertionError("resume result is not the matching invalid run")
    destination = result_path.parent / f"interruption-{len(preserved) + 1:02d}.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    result_path.replace(destination)
    return [*preserved, destination]


def _load_compatible_resume_runs(
    protocol: dict[str, Any], result_dir: Path, replay_dir: Path
) -> list[dict[str, Any]]:
    stored_protocol = json.loads(
        (result_dir / "protocol.json").read_text(encoding="utf-8")
    )
    partial = json.loads((result_dir / "partial.json").read_text(encoding="utf-8"))
    if stored_protocol != protocol or partial.get("protocol") != protocol:
        raise AssertionError("compatible resume protocol differs")
    stored_runs = partial.get("runs")
    prefix_count = int(protocol["compatible_prefix_pairs"])
    if (
        not isinstance(stored_runs, list)
        or len(stored_runs) < prefix_count
        or len(stored_runs) >= len(protocol["pairs"])
    ):
        raise AssertionError("compatible resume requires an incomplete run prefix")
    runs = _load_compatible_prefix(protocol)
    if stored_runs[:prefix_count] != runs:
        raise AssertionError("compatible reused prefix changed")
    for declared, stored in zip(
        protocol["pairs"][prefix_count : len(stored_runs)],
        stored_runs[prefix_count:],
        strict=True,
    ):
        pair = int(declared["pair"])
        for arm_data in stored.get("arms", []):
            expected = replay_dir / f"pair_{pair:02d}" / str(arm_data.get("arm"))
            if Path(arm_data.get("output_dir", "")).resolve() != expected.resolve():
                raise AssertionError(f"compatible resume path differs at pair {pair}")
        run = _reconstruct_saved_run(
            declared,
            stored,
            paired_workload_contract=True,
            expected_replay_root=replay_dir,
        )
        run["provenance"] = {"kind": "fresh_contract_v2"}
        runs.append(run)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=COHORTS, default="initial24")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = COHORTS[args.cohort]
    result_dir = Path(config["result_dir"])
    replay_dir = Path(config["replay_dir"])
    protocol = _protocol(args.cohort)
    interruption_results: list[Path] = []
    if args.resume:
        if args.cohort not in {
            "remaining26",
            "remaining26_compatible_v2",
            "quartet48_contract_v2",
            "rolling48_contract_v1",
        }:
            raise ValueError("resume is not frozen for this cohort")
        result_path = result_dir / "result.json"
        if args.cohort == "remaining26_compatible_v2":
            runs = _load_compatible_resume_runs(protocol, result_dir, replay_dir)
        elif args.cohort in {"quartet48_contract_v2", "rolling48_contract_v1"}:
            runs = _load_resume_runs(protocol, result_dir, replay_dir)
            interruption_results = _preserve_interruption_result(result_path, protocol)
        elif not result_path.is_file() or result_path.stat().st_size != 0:
            raise FileExistsError(f"resume requires empty {result_path}")
        else:
            runs = _load_resume_runs(protocol, result_dir, replay_dir)
    else:
        if result_dir.exists() or replay_dir.exists():
            raise FileExistsError(f"refusing to overwrite {result_dir} or {replay_dir}")
        result_dir.mkdir(parents=True)
        replay_dir.mkdir(parents=True)
        (result_dir / "protocol.json").write_text(
            json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
        )
        runs = (
            _load_compatible_prefix(protocol)
            if protocol.get("compatible_prefix_pairs")
            else []
        )
        if runs:
            (result_dir / "partial.json").write_text(
                json.dumps({"protocol": protocol, "runs": runs}, indent=2) + "\n",
                encoding="utf-8",
            )
    resumed_after_pairs = sum(len(run["arms"]) == 2 for run in runs)
    resumed_after_arms = sum(len(run["arms"]) for run in runs)
    os.environ[OPENCLAW_EXEC_TIMEOUT_FLOOR_ENV] = str(EXEC_TIMEOUT_FLOOR_S)
    if protocol["paired_workload_contract"]:
        os.environ[OPENCLAW_PAIRED_WORKLOAD_CONTRACT_ENV] = "1"
    try:
        for index, declared in enumerate(protocol["pairs"]):
            pair = int(declared["pair"])
            task_ids = list(declared["task_ids"])
            if index < len(runs):
                pair_result = runs[index]
                if len(pair_result["arms"]) == 2:
                    continue
            else:
                pair_result = {"pair": pair, "arms": []}
                if protocol.get("compatible_prefix_pairs"):
                    pair_result["provenance"] = {"kind": "fresh_contract_v2"}
                runs.append(pair_result)
            for arm in declared["arm_order"][len(pair_result["arms"]) :]:
                manifest = result_dir / "manifests" / f"pair_{pair:02d}_{arm}.json"
                _write_manifest(manifest, task_ids)
                pair_result["arms"].append(
                    asyncio.run(
                        _run_arm(
                            pair=pair,
                            task_ids=task_ids,
                            arm=arm,
                            manifest=manifest,
                            replay_dir=replay_dir,
                            paired_workload_contract=protocol[
                                "paired_workload_contract"
                            ],
                            concurrency=protocol.get("queue_concurrency"),
                            workers=protocol.get("queue_workers"),
                            makespan_source=protocol.get(
                                "makespan_source", "max_task_elapsed_s"
                            ),
                            cleanup_images=bool(protocol.get("cleanup_images", False)),
                        )
                    )
                )
                (result_dir / "partial.json").write_text(
                    json.dumps({"protocol": protocol, "runs": runs}, indent=2) + "\n",
                    encoding="utf-8",
                )
        result = _aggregate(protocol, runs)
        if args.resume:
            result["recovery"] = {
                "resumed_after_completed_pairs": resumed_after_pairs,
                "resumed_after_completed_arms": resumed_after_arms,
            }
            if interruption_results:
                result["recovery"]["interruption_results"] = [
                    str(path) for path in interruption_results
                ]
            if args.cohort == "remaining26":
                result["recovery"]["reason"] = "disk_full_before_pair_4"
        (result_dir / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        (result_dir / "partial.json").unlink()
    except BaseException as exc:
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "schema": protocol["schema"],
                    "status": "invalid",
                    "protocol": protocol,
                    "runs": runs,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        os.environ.pop(OPENCLAW_EXEC_TIMEOUT_FLOOR_ENV, None)
        os.environ.pop(OPENCLAW_PAIRED_WORKLOAD_CONTRACT_ENV, None)


if __name__ == "__main__":
    main()
