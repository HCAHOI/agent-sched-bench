#!/usr/bin/env python3
"""Run the frozen paired SQLGlot hard-quota versus CPU-borrowing replay."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import functools
import heapq
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import traceback
from typing import Any

import numpy as np

from harness.container_image_prep import normalize_image_reference
from harness.container_runtime import image_exists_command
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
FRESH_FINAL48_RESULT_DIR = (
    ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot-final48-counterbalanced-rolling-exact-v1"
)
FRESH_FINAL48_REPLAY_DIR = (
    ROOT
    / "traces/swe-rebench/gpt-5.6-sol"
    / "sqlglot-final48-counterbalanced-rolling-exact-v1"
)
SAFETY_GUARD_REJECTION = (
    "Error: Command blocked by safety guard (dangerous pattern detected)\n\n"
    "[Analyze the error above and try a different approach.]"
)
_EXIT_CODE_RE = re.compile(r"(?:^|\n)Exit code:\s*(-?\d+)\s*(?:\n|$)")
_TIMEOUT_MARKERS = ("[timeout]", "[resource_timeout]", "[resource_stall_timeout]")
_OOM_RE = re.compile(r"\b(?:out of memory|oom[_ -]kill(?:ed)?)\b", re.IGNORECASE)
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
    "fresh_final48_counterbalanced_v1": {
        "preregistered_protocol": FRESH_FINAL48_RESULT_DIR / "protocol.json",
        "result_dir": FRESH_FINAL48_RESULT_DIR,
        "replay_dir": FRESH_FINAL48_REPLAY_DIR,
    },
}


def _protocol(cohort: str = "initial24") -> dict[str, Any]:
    config = COHORTS[cohort]
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    preregistered_path = config.get("preregistered_protocol")
    if preregistered_path is not None:
        protocol = json.loads(Path(preregistered_path).read_text(encoding="utf-8"))
        if protocol.get("cohort") != cohort:
            raise AssertionError("preregistered protocol cohort differs")
        selected = list(protocol["selected_task_ids"])
        excluded = set(protocol["excluded_task_ids"])
        expected = [task_id for task_id in split["final_test"] if task_id not in excluded]
        if selected != expected or len(selected) != len(set(selected)):
            raise AssertionError("preregistered final-task selection differs")
        missing = [
            str(SOURCE / task_id / "attempt_1" / "trace.jsonl")
            for task_id in selected
            if not (SOURCE / task_id / "attempt_1" / "trace.jsonl").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"selected source traces are missing: {missing}")
        return protocol
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
        "exec_timeout_floor_s": config.get(
            "exec_timeout_floor_s", EXEC_TIMEOUT_FLOOR_S
        ),
    }
    if "paired_workload_contract_version" in config:
        protocol["paired_workload_contract_version"] = int(
            config["paired_workload_contract_version"]
        )
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


@contextmanager
def _replay_environment(protocol: dict[str, Any]):
    keys = (
        OPENCLAW_EXEC_TIMEOUT_FLOOR_ENV,
        OPENCLAW_PAIRED_WORKLOAD_CONTRACT_ENV,
    )
    previous = {key: os.environ.get(key) for key in keys}
    floor = protocol["exec_timeout_floor_s"]
    paired = bool(protocol["paired_workload_contract"])
    version = int(protocol.get("paired_workload_contract_version", 1))
    if not paired and "paired_workload_contract_version" in protocol:
        raise ValueError("unpaired replay cannot select a paired contract version")
    if paired and version == 2 and floor is not None:
        raise ValueError("paired replay contract 2 forbids an exec timeout floor")
    try:
        if floor is None:
            os.environ.pop(OPENCLAW_EXEC_TIMEOUT_FLOOR_ENV, None)
        else:
            os.environ[OPENCLAW_EXEC_TIMEOUT_FLOOR_ENV] = str(floor)
        if paired:
            os.environ[OPENCLAW_PAIRED_WORKLOAD_CONTRACT_ENV] = str(version)
        else:
            os.environ.pop(OPENCLAW_PAIRED_WORKLOAD_CONTRACT_ENV, None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


@functools.cache
def _task_source_images_by_id() -> dict[str, str]:
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))
    images: dict[str, str] = {}
    for task in tasks:
        task_id = str(task["instance_id"])
        image = task.get("docker_image") or task.get("image_name")
        if image:
            images[task_id] = normalize_image_reference(str(image))
    return images


def _cached_source_images(
    task_ids: list[str], *, container_executable: str
) -> list[str]:
    images_by_id = _task_source_images_by_id()
    missing_metadata = [task_id for task_id in task_ids if task_id not in images_by_id]
    if missing_metadata:
        raise AssertionError(f"tasks have no source image metadata: {missing_metadata}")
    cached = []
    for image in dict.fromkeys(images_by_id[task_id] for task_id in task_ids):
        probe = subprocess.run(
            image_exists_command(image, container_executable=container_executable),
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        if probe.returncode == 0:
            cached.append(image)
            continue
        error = (probe.stderr or probe.stdout or "").strip()
        if not any(
            marker in error.lower()
            for marker in ("no such image", "no such object", "image not known")
        ):
            raise RuntimeError(f"source image probe failed for {image}: {error}")
    return cached


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


def _tool_terminal_outcome(action: dict[str, Any]) -> dict[str, Any]:
    if action.get("action_type") != "tool_exec":
        raise ValueError("terminal outcome requires a tool_exec action")
    data = action.get("data") or {}
    if data.get("tool_name") != "exec":
        return {
            "class": "tool_success" if data.get("success") is not False else "tool_error",
            "exit_code": None,
        }
    result = str(data.get("tool_result", data.get("result", "")) or "")
    if result.strip() == SAFETY_GUARD_REJECTION.strip():
        return {"class": "safety_rejection", "exit_code": None}
    exit_codes = _EXIT_CODE_RE.findall(result)
    exit_code = int(exit_codes[-1]) if exit_codes else None
    if any(marker in result for marker in _TIMEOUT_MARKERS):
        return {"class": "timeout", "exit_code": exit_code}
    if exit_code == 137 and _OOM_RE.search(result):
        return {"class": "explicit_oom", "exit_code": exit_code}
    if exit_code is not None:
        return {
            "class": "exit_zero" if exit_code == 0 else "exit_nonzero",
            "exit_code": exit_code,
        }
    if data.get("success") is False:
        return {"class": "tool_error", "exit_code": None}
    call_id = data.get("tool_call_id")
    raise ValueError(f"exec action {call_id!r} has no terminal status")


def _replay_tool_actions(trace_path: Path) -> list[dict[str, Any]]:
    return [
        record
        for record in (
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    ]


def _tool_identity(action: dict[str, Any]) -> tuple[str, str, str]:
    data = action.get("data") or {}
    return (
        str(action.get("action_id")),
        str(data.get("tool_name")),
        str(data.get("tool_call_id")),
    )


def _fixed_duration_queue_makespan(
    task_ids: list[str], durations: dict[str, float], concurrency: int
) -> float:
    if not task_ids:
        return 0.0
    if concurrency <= 0:
        raise ValueError("queue concurrency must be positive")
    slots = [0.0] * min(concurrency, len(task_ids))
    for task_id in task_ids:
        ready_s = heapq.heappop(slots)
        heapq.heappush(slots, ready_s + float(durations[task_id]))
    return max(slots)


def _arm_outcome_relation(
    burstable: dict[str, Any], hard: dict[str, Any]
) -> str:
    burstable_class = str(burstable["class"])
    hard_class = str(hard["class"])
    if burstable_class == hard_class:
        return "same_terminal_class"
    success_classes = {"exit_zero", "tool_success"}
    burstable_succeeded = burstable_class in success_classes
    hard_succeeded = hard_class in success_classes
    if burstable_succeeded and not hard_succeeded:
        return "burstable_success_hard_failure"
    if hard_succeeded and not burstable_succeeded:
        return "burstable_failure_hard_success"
    return "different_failure_class"


def _audit_existing_rolling_outcomes(result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    queues = result.get("queues") or []
    if len(queues) != 1:
        raise ValueError("outcome audit requires exactly one rolling queue")
    queue = queues[0]
    task_ids = [str(value) for value in queue["task_ids"]]
    arms = {str(item["arm"]): item for item in queue["arms"]}
    if set(arms) != {"burstable_two", "hard_two"}:
        raise ValueError("outcome audit requires burstable_two and hard_two arms")
    arm_tasks = {
        arm: {str(item["task_id"]): item for item in arm_data["tasks"]}
        for arm, arm_data in arms.items()
    }
    arm_elapsed = {
        arm: {
            str(item["agent_id"]): float(item["elapsed_s"])
            for item in arm_data["task_stats"]
        }
        for arm, arm_data in arms.items()
    }
    arm_startup: dict[str, dict[str, float]] = {arm: {} for arm in arms}
    mismatches: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    tool_call_count = 0
    stable_tool_call_count = 0
    arm_pair_mismatch_tool_call_count = 0
    arm_outcome_relation_counts = {
        "same_terminal_class": 0,
        "burstable_success_hard_failure": 0,
        "burstable_failure_hard_success": 0,
        "different_failure_class": 0,
    }
    source_clean_ids: list[str] = []
    source_clean_outcome_stable_ids: list[str] = []
    for task_id in task_ids:
        attempts = {
            arm: Path(arm_tasks[arm][task_id]["attempt_dir"]) for arm in arms
        }
        requests = {
            arm: json.loads(
                (attempt / "openclaw_host_replay_request.json").read_text(
                    encoding="utf-8"
                )
            )
            for arm, attempt in attempts.items()
        }
        source_actions = [
            action
            for action in requests["burstable_two"]["source_actions"]
            if action.get("action_type") == "tool_exec"
        ]
        hard_source_actions = [
            action
            for action in requests["hard_two"]["source_actions"]
            if action.get("action_type") == "tool_exec"
        ]
        if source_actions != hard_source_actions:
            raise AssertionError(f"source actions differ between arms for {task_id}")
        replay_actions = {
            arm: _replay_tool_actions(attempt / "openclaw_host_replay.jsonl")
            for arm, attempt in attempts.items()
        }
        source_identities = [_tool_identity(action) for action in source_actions]
        for arm, actions in replay_actions.items():
            if [_tool_identity(action) for action in actions] != source_identities:
                raise AssertionError(f"tool action identities differ for {task_id} {arm}")
        source_clean = all(
            (action.get("data") or {}).get("success") is not False
            for action in source_actions
        )
        if source_clean:
            source_clean_ids.append(task_id)
        task_mismatch_count = 0
        task_arm_pair_mismatch_count = 0
        for source_action, burst_action, hard_action in zip(
            source_actions,
            replay_actions["burstable_two"],
            replay_actions["hard_two"],
            strict=True,
        ):
            tool_call_count += 1
            outcomes = {
                "source": _tool_terminal_outcome(source_action),
                "burstable_two": _tool_terminal_outcome(burst_action),
                "hard_two": _tool_terminal_outcome(hard_action),
            }
            relation = _arm_outcome_relation(
                outcomes["burstable_two"], outcomes["hard_two"]
            )
            arm_outcome_relation_counts[relation] += 1
            if len({item["class"] for item in outcomes.values()}) == 1:
                stable_tool_call_count += 1
                continue
            task_mismatch_count += 1
            if outcomes["burstable_two"]["class"] != outcomes["hard_two"]["class"]:
                arm_pair_mismatch_tool_call_count += 1
                task_arm_pair_mismatch_count += 1
            source_data = source_action.get("data") or {}
            mismatches.append(
                {
                    "task_id": task_id,
                    "tool_call_id": str(source_data.get("tool_call_id")),
                    "tool_name": str(source_data.get("tool_name")),
                    **outcomes,
                    "duration_s": {
                        "burstable_two": float(
                            (burst_action.get("data") or {}).get("duration_ms", 0)
                        )
                        / 1000.0,
                        "hard_two": float(
                            (hard_action.get("data") or {}).get("duration_ms", 0)
                        )
                        / 1000.0,
                    },
                    "tool_args": source_data.get("tool_args"),
                }
            )
        if source_clean and task_mismatch_count == 0:
            source_clean_outcome_stable_ids.append(task_id)
        task_rows.append(
            {
                "task_id": task_id,
                "source_clean": source_clean,
                "tool_call_count": len(source_actions),
                "outcome_mismatch_count": task_mismatch_count,
                "arm_pair_outcome_mismatch_count": task_arm_pair_mismatch_count,
            }
        )
        for arm, attempt in attempts.items():
            startup = json.loads(
                (attempt / "container_startup.json").read_text(encoding="utf-8")
            )
            arm_startup[arm][task_id] = float(startup["elapsed_s"])

    concurrency = int(result["protocol"]["queue_concurrency"])

    def schedule(include_recorded_startup: bool) -> dict[str, float]:
        makespans = {}
        for arm in ("burstable_two", "hard_two"):
            durations = {
                task_id: arm_elapsed[arm][task_id]
                + (arm_startup[arm][task_id] if include_recorded_startup else 0.0)
                for task_id in source_clean_outcome_stable_ids
            }
            makespans[arm] = _fixed_duration_queue_makespan(
                source_clean_outcome_stable_ids, durations, concurrency
            )
        hard = makespans["hard_two"]
        return {
            "burstable_two_makespan_s": makespans["burstable_two"],
            "hard_two_makespan_s": hard,
            "improvement_fraction": (hard - makespans["burstable_two"]) / hard,
        }

    return {
        "schema": "sqlglot-rolling-outcome-audit-v1",
        "status": "diagnostic_only_no_verdict",
        "frozen_result_status": result.get("status"),
        "source_result": str(result_path),
        "policy": {
            "exact_output_match_required": False,
            "outcome_comparison": "coarse terminal class",
            "terminal_classes": [
                "exit_zero",
                "exit_nonzero",
                "timeout",
                "explicit_oom",
                "safety_rejection",
                "tool_success",
                "tool_error",
            ],
            "source_clean_definition": "no source tool_exec action has success=false",
            "diagnostic_subset": (
                "source-clean tasks with zero source/burstable/hard terminal-class "
                "mismatches"
            ),
            "recorded_startup_scope": (
                "container_startup.json elapsed_s only; excludes artifact restore, "
                "finalization, container stop, and image cleanup"
            ),
            "selection_timing": "post-hoc after rolling result exposure",
        },
        "counts": {
            "task_count": len(task_ids),
            "source_clean_task_count": len(source_clean_ids),
            "source_failed_task_count": len(task_ids) - len(source_clean_ids),
            "source_clean_outcome_stable_task_count": len(
                source_clean_outcome_stable_ids
            ),
            "source_clean_outcome_mismatch_task_count": len(source_clean_ids)
            - len(source_clean_outcome_stable_ids),
            "tool_call_count": tool_call_count,
            "outcome_stable_tool_call_count": stable_tool_call_count,
            "outcome_mismatch_tool_call_count": len(mismatches),
            "outcome_mismatch_task_count": sum(
                row["outcome_mismatch_count"] > 0 for row in task_rows
            ),
            "arm_pair_outcome_mismatch_tool_call_count": (
                arm_pair_mismatch_tool_call_count
            ),
            "arm_pair_outcome_mismatch_task_count": sum(
                row["arm_pair_outcome_mismatch_count"] > 0 for row in task_rows
            ),
        },
        "tasks": task_rows,
        "mismatches": mismatches,
        "arm_outcome_quality": {
            "unit": "tool_call",
            "success_classes": ["exit_zero", "tool_success"],
            "relation_counts": arm_outcome_relation_counts,
            "burstable_quality_regression_count": arm_outcome_relation_counts[
                "burstable_failure_hard_success"
            ],
            "selection_timing": "post-hoc descriptive audit",
        },
        "source_clean_outcome_stable_fixed_duration_diagnostic": {
            "task_count": len(source_clean_outcome_stable_ids),
            "queue_concurrency": concurrency,
            "execution_only": schedule(False),
            "recorded_container_startup_plus_replay": schedule(True),
        },
    }


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


def _source_replay_tool_outcomes(
    request: dict[str, Any], trace_path: Path
) -> list[dict[str, Any]]:
    source = [
        action
        for action in request["source_actions"]
        if action.get("action_type") == "tool_exec"
    ]
    replay = _replay_tool_actions(trace_path)
    if [_tool_identity(action) for action in replay] != [
        _tool_identity(action) for action in source
    ]:
        raise AssertionError("source and replay tool identities differ")
    outcomes = []
    for source_action, replay_action in zip(source, replay, strict=True):
        action_id, tool_name, tool_call_id = _tool_identity(source_action)
        outcomes.append(
            {
                "action_id": action_id,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "source": _tool_terminal_outcome(source_action),
                "replay": _tool_terminal_outcome(replay_action),
            }
        )
    return outcomes


def _task_attempt_dir(
    arm_dir: Path, task_id: str, *, use_latest_attempt: bool
) -> Path:
    instance_dir = arm_dir / task_id
    if not use_latest_attempt:
        return instance_dir / "attempt_1"
    attempts = []
    for path in instance_dir.glob("attempt_*"):
        match = re.fullmatch(r"attempt_(\d+)", path.name)
        if path.is_dir() and match:
            attempts.append((int(match.group(1)), path))
    if not attempts:
        return instance_dir / "attempt_1"
    return max(attempts)[1]


def _resource_sample_action_window(
    samples: list[dict[str, Any]], trace_path: Path
) -> tuple[float, float, float, float]:
    sample_epochs = [float(sample["epoch"]) for sample in samples]
    action_bounds = [
        (float(record["ts_start"]), float(record["ts_end"]))
        for record in (
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if record.get("type") == "action"
    ]
    if not action_bounds:
        raise AssertionError(f"replay trace has no action window: {trace_path}")
    return (
        min(sample_epochs),
        max(sample_epochs),
        min(start for start, _ in action_bounds),
        max(end for _, end in action_bounds),
    )


def _task_artifacts(
    arm_dir: Path,
    task_id: str,
    arm: str,
    *,
    paired_workload_contract: bool | None,
    exec_timeout_floor_s: float | None = EXEC_TIMEOUT_FLOOR_S,
    paired_workload_contract_version: int | None = None,
    require_telemetry_integrity: bool = False,
    telemetry_boundary_tolerance_s: float = 0.0,
    use_latest_attempt: bool = False,
) -> dict[str, Any]:
    attempt = _task_attempt_dir(
        arm_dir, task_id, use_latest_attempt=use_latest_attempt
    )
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
    if request.get("exec_timeout_floor_s") != exec_timeout_floor_s:
        raise AssertionError(f"wrong timeout floor for {task_id} {arm}")
    if paired_workload_contract is None:
        if "paired_workload_contract" in request:
            raise AssertionError(
                f"unexpected replay contract field for {task_id} {arm}"
            )
    elif request.get("paired_workload_contract") is not paired_workload_contract:
        raise AssertionError(f"wrong replay contract for {task_id} {arm}")
    if (
        paired_workload_contract_version is not None
        and request.get("paired_workload_contract_version")
        != paired_workload_contract_version
    ):
        raise AssertionError(f"wrong replay contract version for {task_id} {arm}")
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
    if timeout_count and replay_action_contract.get(
        "require_source_outcome_match", True
    ):
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
    samples = resources.get("samples") or []
    monitoring = summary.get("monitoring") or {}
    if require_telemetry_integrity and (
        status.get("telemetry_quality") != "ok"
        or status.get("telemetry_integrity_failed") is not False
        or status.get("telemetry_errors") != []
        or not samples
        or summary.get("sample_count") != len(samples)
        or monitoring.get("status") != "collected"
        or monitoring.get("resource_enabled") is not True
        or monitoring.get("per_task_resource_enabled") is not True
    ):
        raise AssertionError(f"invalid resource telemetry for {task_id} {arm}")
    telemetry_window = None
    if require_telemetry_integrity:
        if telemetry_boundary_tolerance_s < 0:
            raise ValueError("telemetry boundary tolerance must be non-negative")
        try:
            telemetry_window = _resource_sample_action_window(
                samples, attempt / "openclaw_host_replay.jsonl"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AssertionError(
                f"invalid resource sample timestamps for {task_id} {arm}"
            ) from exc
        sample_start, sample_end, action_start, action_end = telemetry_window
        sample_duration = sample_end - sample_start
        if abs(float(summary.get("duration_seconds")) - sample_duration) > 1e-6:
            raise AssertionError(
                f"resource sample duration differs for {task_id} {arm}"
            )
        if (
            sample_start > action_start + telemetry_boundary_tolerance_s
            or sample_end < action_end - telemetry_boundary_tolerance_s
        ):
            raise AssertionError(
                f"resource samples do not cover replay actions for {task_id} {arm}"
            )
    artifact = {
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
        "resource_sample_count": len(samples),
        "cpu_controls": expected_controls,
        "exec_timeout_floor_s": request["exec_timeout_floor_s"],
        "replay_action_contract": replay_action_contract,
        "source_success_replay_timeout_count": timeout_count,
        "final_container_state": final_state,
    }
    if require_telemetry_integrity:
        assert telemetry_window is not None
        artifact["telemetry_integrity"] = {
            "telemetry_quality": status["telemetry_quality"],
            "telemetry_integrity_failed": status["telemetry_integrity_failed"],
            "telemetry_errors": status["telemetry_errors"],
            "resource_monitoring_status": monitoring["status"],
            "sample_epoch_start": telemetry_window[0],
            "sample_epoch_end": telemetry_window[1],
            "action_epoch_start": telemetry_window[2],
            "action_epoch_end": telemetry_window[3],
            "boundary_tolerance_s": telemetry_boundary_tolerance_s,
        }
    if paired_workload_contract_version == 2:
        artifact["tool_outcomes"] = _source_replay_tool_outcomes(
            request, attempt / "openclaw_host_replay.jsonl"
        )
    return artifact


async def _run_arm(
    *,
    pair: int,
    task_ids: list[str],
    arm: str,
    manifest: Path,
    replay_dir: Path,
    paired_workload_contract: bool,
    exec_timeout_floor_s: float | None = EXEC_TIMEOUT_FLOOR_S,
    paired_workload_contract_version: int | None = None,
    concurrency: int | None = None,
    workers: int | None = None,
    makespan_source: str = "max_task_elapsed_s",
    cleanup_images: bool = False,
    require_cold_source_images: bool = False,
    require_telemetry_integrity: bool = False,
    telemetry_boundary_tolerance_s: float = 0.0,
    use_latest_attempt: bool = False,
) -> dict[str, Any]:
    arm_dir = replay_dir / f"pair_{pair:02d}" / arm
    concurrency = len(task_ids) if concurrency is None else concurrency
    workers = concurrency if workers is None else workers
    if require_cold_source_images:
        cached_before = _cached_source_images(
            task_ids, container_executable="docker"
        )
        if cached_before:
            raise AssertionError(
                f"source images cached before arm {pair} {arm}: {cached_before}"
            )
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
    if require_cold_source_images:
        cached_after = _cached_source_images(task_ids, container_executable="docker")
        if cached_after:
            raise AssertionError(
                f"source images cached after arm {pair} {arm}: {cached_after}"
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
                exec_timeout_floor_s=exec_timeout_floor_s,
                paired_workload_contract_version=paired_workload_contract_version,
                require_telemetry_integrity=require_telemetry_integrity,
                telemetry_boundary_tolerance_s=telemetry_boundary_tolerance_s,
                use_latest_attempt=use_latest_attempt,
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


def _cross_arm_outcome_quality(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    relation_counts = {
        "same_terminal_class": 0,
        "burstable_success_hard_failure": 0,
        "burstable_failure_hard_success": 0,
        "different_failure_class": 0,
    }
    differences = []
    source_differences = []
    regression_task_ids: list[str] = []
    improvement_task_ids: list[str] = []
    incomparable_task_ids: list[str] = []
    for pair in pairs:
        arms = {arm["arm"]: arm for arm in pair["arms"]}
        arm_tasks = {
            arm: {task["task_id"]: task for task in arm_data["tasks"]}
            for arm, arm_data in arms.items()
        }
        for task_id in pair["task_ids"]:
            hard = arm_tasks["hard_two"][task_id]["tool_outcomes"]
            burst = arm_tasks["burstable_two"][task_id]["tool_outcomes"]
            hard_by_identity = {
                (row["action_id"], row["tool_name"], row["tool_call_id"]): row
                for row in hard
            }
            burst_by_identity = {
                (row["action_id"], row["tool_name"], row["tool_call_id"]): row
                for row in burst
            }
            if hard_by_identity.keys() != burst_by_identity.keys():
                raise AssertionError(f"arm tool outcomes differ for {task_id}")
            task_relations: set[str] = set()
            for identity, hard_row in hard_by_identity.items():
                burst_row = burst_by_identity[identity]
                if hard_row["source"] != burst_row["source"]:
                    raise AssertionError(f"arm source outcomes differ for {task_id}")
                for arm, row in (
                    ("hard_two", hard_row),
                    ("burstable_two", burst_row),
                ):
                    if row["source"]["class"] != row["replay"]["class"]:
                        source_differences.append(
                            {
                                "queue": pair["queue"],
                                "task_id": task_id,
                                "action_id": identity[0],
                                "tool_name": identity[1],
                                "tool_call_id": identity[2],
                                "arm": arm,
                                "source": row["source"],
                                "replay": row["replay"],
                            }
                        )
                relation = _arm_outcome_relation(
                    burst_row["replay"], hard_row["replay"]
                )
                relation_counts[relation] += 1
                if relation == "same_terminal_class":
                    continue
                task_relations.add(relation)
                differences.append(
                    {
                        "queue": pair["queue"],
                        "task_id": task_id,
                        "action_id": identity[0],
                        "tool_name": identity[1],
                        "tool_call_id": identity[2],
                        "source": hard_row["source"],
                        "burstable_two": burst_row["replay"],
                        "hard_two": hard_row["replay"],
                        "relation": relation,
                    }
                )
            if "burstable_failure_hard_success" in task_relations:
                regression_task_ids.append(task_id)
            if "burstable_success_hard_failure" in task_relations:
                improvement_task_ids.append(task_id)
            if "different_failure_class" in task_relations:
                incomparable_task_ids.append(task_id)
    return {
        "unit": "tool_call_terminal_class",
        "success_classes": ["exit_zero", "tool_success"],
        "relation_counts": relation_counts,
        "arm_terminal_class_difference_count": len(differences),
        "arm_terminal_class_difference_task_count": len(
            {row["task_id"] for row in differences}
        ),
        "source_terminal_class_difference_count": len(source_differences),
        "source_terminal_class_difference_task_ids": list(
            dict.fromkeys(row["task_id"] for row in source_differences)
        ),
        "quality_regression_task_ids": regression_task_ids,
        "quality_improvement_task_ids": improvement_task_ids,
        "incomparable_task_ids": incomparable_task_ids,
        "timing_gate_eligible": not differences and not source_differences,
        "differences": differences,
        "source_differences": source_differences,
    }


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
    if protocol.get("quality_gate"):
        quality = _cross_arm_outcome_quality(pairs)
        gate = None
        if quality["timing_gate_eligible"]:
            timing_gate = protocol["timing_gate"]
            threshold = float(timing_gate["mean_improvement_at_least_fraction"])
            minimum_improving = int(timing_gate["minimum_improving_queues"])
            expected_queue_count = int(timing_gate["queue_count"])
            if len(pairs) != expected_queue_count:
                raise AssertionError("timing gate queue count differs")
            gate = {
                "mean_improvement_at_least_5_percent": mean >= threshold,
                "at_least_3_of_4_queues_improve": improving >= minimum_improving,
            }
            status = str(
                timing_gate["go_verdict"]
                if all(gate.values())
                else timing_gate["failure_verdict"]
            )
        else:
            status = str(protocol["quality_gate"]["arm_difference_verdict"])
        result = {
            "schema": protocol["schema"],
            "status": status,
            "protocol": protocol,
            "comparison": {
                "mean_improvement_fraction": mean,
                "median_improvement_fraction": float(
                    statistics.median(improvements)
                ),
                "improving_queues": improving,
                "queue_count": len(pairs),
                "gate": gate,
            },
            "arm_outcome_quality": quality,
            plural: pairs,
        }
    elif decision_unit == "queue":
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
        or (not runs and not arm_level_resume)
        or len(runs) > len(protocol["pairs"])
        or (len(runs) == len(protocol["pairs"]) and not arm_level_resume)
    ):
        raise AssertionError("resume requires an incomplete run prefix")

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
            validated_tasks = []
            for task_id in declared["task_ids"]:
                validated_tasks.append(_task_artifacts(
                    expected_dir,
                    task_id,
                    arm,
                    paired_workload_contract=protocol["paired_workload_contract"],
                    exec_timeout_floor_s=protocol["exec_timeout_floor_s"],
                    paired_workload_contract_version=protocol.get(
                        "paired_workload_contract_version"
                    ),
                    require_telemetry_integrity=bool(
                        (protocol.get("framework_gate") or {}).get(
                            "telemetry_integrity_required", False
                        )
                    ),
                    telemetry_boundary_tolerance_s=float(
                        (protocol.get("framework_gate") or {}).get(
                            "maximum_resource_sample_boundary_gap_s", 0.0
                        )
                    ),
                    use_latest_attempt=bool(protocol.get("arm_level_resume", False)),
                ))
            if protocol.get("quality_gate"):
                arm_data["tasks"] = validated_tasks
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


def _resume_requested_run(
    cohort: str,
    protocol: dict[str, Any],
    result_dir: Path,
    replay_dir: Path,
) -> tuple[list[dict[str, Any]], list[Path]]:
    result_path = result_dir / "result.json"
    if cohort == "remaining26_compatible_v2":
        return _load_compatible_resume_runs(protocol, result_dir, replay_dir), []
    if cohort in {
        "quartet48_contract_v2",
        "rolling48_contract_v1",
        "fresh_final48_counterbalanced_v1",
    }:
        runs = _load_resume_runs(protocol, result_dir, replay_dir)
        return runs, _preserve_interruption_result(result_path, protocol)
    if not result_path.is_file() or result_path.stat().st_size != 0:
        raise FileExistsError(f"resume requires empty {result_path}")
    return _load_resume_runs(protocol, result_dir, replay_dir), []


def _prepare_new_run_directories(
    config: dict[str, Any], protocol: dict[str, Any]
) -> None:
    result_dir = Path(config["result_dir"])
    replay_dir = Path(config["replay_dir"])
    preregistered_path = config.get("preregistered_protocol")
    if preregistered_path is None:
        if result_dir.exists() or replay_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite {result_dir} or {replay_dir}"
            )
        result_dir.mkdir(parents=True)
        replay_dir.mkdir(parents=True)
        (result_dir / "protocol.json").write_text(
            json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
        )
        if protocol.get("arm_level_resume"):
            (result_dir / "partial.json").write_text(
                json.dumps({"protocol": protocol, "runs": []}, indent=2) + "\n",
                encoding="utf-8",
            )
        return

    protocol_path = Path(preregistered_path)
    if protocol_path != result_dir / "protocol.json" or not protocol_path.is_file():
        raise FileNotFoundError("preregistered protocol is not in the result directory")
    unexpected = [path for path in result_dir.iterdir() if path != protocol_path]
    if unexpected:
        raise FileExistsError(f"unexpected preregistration artifact: {unexpected}")
    if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise AssertionError("preregistered protocol differs from generated protocol")
    if replay_dir.exists():
        raise FileExistsError(f"refusing to overwrite {replay_dir}")
    replay_dir.mkdir(parents=True)
    if protocol.get("arm_level_resume"):
        (result_dir / "partial.json").write_text(
            json.dumps({"protocol": protocol, "runs": []}, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=COHORTS, default="initial24")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-existing-outcomes", action="store_true")
    args = parser.parse_args()
    config = COHORTS[args.cohort]
    result_dir = Path(config["result_dir"])
    replay_dir = Path(config["replay_dir"])
    if args.audit_existing_outcomes:
        if args.resume or args.cohort != "rolling48_contract_v1":
            raise ValueError(
                "existing outcome audit requires rolling48_contract_v1 without resume"
            )
        result_path = result_dir / "result.json"
        audit_path = result_dir / "outcome-audit.json"
        if audit_path.exists():
            raise FileExistsError(f"refusing to overwrite {audit_path}")
        audit = _audit_existing_rolling_outcomes(result_path)
        audit_path.write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"artifact": str(audit_path), **audit["counts"]}, indent=2))
        return
    protocol = _protocol(args.cohort)
    interruption_results: list[Path] = []
    if args.resume:
        if args.cohort not in {
            "remaining26",
            "remaining26_compatible_v2",
            "quartet48_contract_v2",
            "rolling48_contract_v1",
            "fresh_final48_counterbalanced_v1",
        }:
            raise ValueError("resume is not frozen for this cohort")
        runs, interruption_results = _resume_requested_run(
            args.cohort, protocol, result_dir, replay_dir
        )
    else:
        _prepare_new_run_directories(config, protocol)
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
    replay_environment = _replay_environment(protocol)
    replay_environment.__enter__()
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
                            exec_timeout_floor_s=protocol[
                                "exec_timeout_floor_s"
                            ],
                            paired_workload_contract_version=protocol.get(
                                "paired_workload_contract_version"
                            ),
                            concurrency=protocol.get("queue_concurrency"),
                            workers=protocol.get("queue_workers"),
                            makespan_source=protocol.get(
                                "makespan_source", "max_task_elapsed_s"
                            ),
                            cleanup_images=bool(protocol.get("cleanup_images", False)),
                            require_cold_source_images=bool(
                                (protocol.get("framework_gate") or {}).get(
                                    "source_images_absent_before_and_after_each_arm",
                                    False,
                                )
                            ),
                            require_telemetry_integrity=bool(
                                (protocol.get("framework_gate") or {}).get(
                                    "telemetry_integrity_required", False
                                )
                            ),
                            telemetry_boundary_tolerance_s=float(
                                (protocol.get("framework_gate") or {}).get(
                                    "maximum_resource_sample_boundary_gap_s", 0.0
                                )
                            ),
                            use_latest_attempt=bool(
                                protocol.get("arm_level_resume", False)
                            ),
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
        replay_environment.__exit__(None, None, None)


if __name__ == "__main__":
    main()
