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
from trace_collect.simulate_openclaw import OPENCLAW_EXEC_TIMEOUT_FLOOR_ENV


ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "analysis/development/sqlglot-relational-task-split.json"
SOURCE = (
    ROOT
    / "traces/swe-rebench/gpt-5.6-sol"
    / "sqlglot-prev100-c2-fast-requested-ebpf-20260804"
)
TASKS = ROOT / "data/swe-rebench/tasks.json"
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
}


def _protocol(cohort: str = "initial24") -> dict[str, Any]:
    config = COHORTS[cohort]
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    task_ids = np.array(sorted(split["validation"]), dtype=object)
    np.random.Generator(np.random.PCG64(SELECTION_SEED)).shuffle(task_ids)
    start = int(config["selection_start"])
    stop = start + int(config["selection_count"])
    selected = [str(value) for value in task_ids[start:stop]]
    pairs = [selected[index : index + 2] for index in range(0, len(selected), 2)]
    arm_rng = np.random.Generator(np.random.PCG64(config["arm_order_seed"]))
    arm_orders: list[list[str]] = []
    for _pair in pairs:
        order = np.array(["hard_two", "burstable_two"], dtype=object)
        arm_rng.shuffle(order)
        arm_orders.append([str(value) for value in order])
    traces = {
        task_id: SOURCE / task_id / "attempt_1" / "trace.jsonl" for task_id in selected
    }
    missing = [str(path) for path in traces.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"selected source traces are missing: {missing}")
    return {
        "cohort": cohort,
        "selection_seed": SELECTION_SEED,
        "selection_range": [start, stop],
        "arm_order_seed": config["arm_order_seed"],
        "bootstrap_seed": config["bootstrap_seed"],
        "minimum_improving_pairs": config["minimum_improving_pairs"],
        "schema": config["schema"],
        "selected_task_ids": selected,
        "pairs": [
            {
                "pair": index + 1,
                "task_ids": pair,
                "arm_order": arm_orders[index],
            }
            for index, pair in enumerate(pairs)
        ],
        "source_dir": str(SOURCE),
        "task_source": str(TASKS),
        "exec_timeout_floor_s": EXEC_TIMEOUT_FLOOR_S,
    }


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


def _task_artifacts(arm_dir: Path, task_id: str, arm: str) -> dict[str, Any]:
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
        "unexpected_replay_failed_actions": 0,
        "missing_source_action_count": 0,
        "action_sequence_matches": True,
        "error": None,
    }
    for key, expected in required_status.items():
        if status.get(key) != expected:
            raise AssertionError(
                f"invalid replay status for {task_id} {arm}: {key}={status.get(key)!r}"
            )
    if status.get("emitted_actions") != status.get("expected_actions"):
        raise AssertionError(f"incomplete replay for {task_id} {arm}")

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
        "final_container_state": final_state,
    }


async def _run_arm(
    *,
    pair: int,
    task_ids: list[str],
    arm: str,
    manifest: Path,
    replay_dir: Path,
) -> dict[str, Any]:
    arm_dir = replay_dir / f"pair_{pair:02d}" / arm
    trace_file = await simulate(
        manifest=manifest,
        output_dir=arm_dir,
        concurrency=2,
        workers=2,
        prep_concurrency=2,
        container_executable="docker",
        network_mode="host",
        command_timeout_s=3_600.0,
        replay_speed=20.0,
        resource_monitoring="on",
        pmu_monitoring="off",
        memory_bandwidth_monitoring="off",
        container_start_extra_args=ARM_ARGS[arm],
    )
    summary_path = arm_dir / "throughput_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("attempted_traces") != 2:
        raise AssertionError(f"invalid throughput summary for pair {pair} {arm}")
    stats = {str(item["agent_id"]): item for item in summary["tasks"]}
    if set(stats) != set(task_ids):
        raise AssertionError(f"wrong tasks for pair {pair} {arm}")
    if any(item["failed_action_count"] for item in stats.values()):
        raise AssertionError(f"replay action failure for pair {pair} {arm}")
    return {
        "arm": arm,
        "output_dir": str(arm_dir),
        "trace_file": str(trace_file),
        "summary_path": str(summary_path),
        "pair_makespan_s": max(float(item["elapsed_s"]) for item in stats.values()),
        "tasks": [_task_artifacts(arm_dir, task_id, arm) for task_id in task_ids],
        "task_stats": [stats[task_id] for task_id in task_ids],
        "action_sequences": _action_sequences(trace_file),
    }


def _aggregate(protocol: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair = {item["pair"]: item for item in runs}
    pairs: list[dict[str, Any]] = []
    for declared in protocol["pairs"]:
        pair = declared["pair"]
        arms = {item["arm"]: item for item in by_pair[pair]["arms"]}
        hard = arms["hard_two"]
        burst = arms["burstable_two"]
        if hard["action_sequences"] != burst["action_sequences"]:
            raise AssertionError(f"action identity differs for pair {pair}")
        improvement = (hard["pair_makespan_s"] - burst["pair_makespan_s"]) / hard[
            "pair_makespan_s"
        ]
        pairs.append(
            {
                "pair": pair,
                "task_ids": declared["task_ids"],
                "arm_order": declared["arm_order"],
                "hard_two_makespan_s": hard["pair_makespan_s"],
                "burstable_two_makespan_s": burst["pair_makespan_s"],
                "improvement_fraction": improvement,
                "arms": by_pair[pair]["arms"],
            }
        )
    improvements = np.array([item["improvement_fraction"] for item in pairs])
    rng = np.random.Generator(np.random.PCG64(protocol["bootstrap_seed"]))
    draws = improvements[
        rng.integers(0, len(improvements), size=(10_000, len(improvements)))
    ].mean(axis=1)
    mean = float(improvements.mean())
    interval = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
    improving = int((improvements > 0).sum())
    minimum_improving = int(protocol["minimum_improving_pairs"])
    gate = {
        "mean_improvement_at_least_5_percent": mean >= 0.05,
        "bootstrap_lower_above_zero": interval[0] > 0.0,
        f"at_least_{minimum_improving}_of_{len(pairs)}_pairs_improve": (
            improving >= minimum_improving
        ),
        "all_validity_checks_passed": True,
    }
    return {
        "schema": protocol["schema"],
        "status": "go" if all(gate.values()) else "no_go",
        "protocol": protocol,
        "comparison": {
            "mean_improvement_fraction": mean,
            "median_improvement_fraction": float(statistics.median(improvements)),
            "improving_pairs": improving,
            "pair_count": len(pairs),
            "ci95_paired_pair_bootstrap": interval,
            "bootstrap_draws": 10_000,
            "gate": gate,
        },
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=COHORTS, default="initial24")
    args = parser.parse_args()
    config = COHORTS[args.cohort]
    result_dir = Path(config["result_dir"])
    replay_dir = Path(config["replay_dir"])
    if result_dir.exists() or replay_dir.exists():
        raise FileExistsError(f"refusing to overwrite {result_dir} or {replay_dir}")
    protocol = _protocol(args.cohort)
    result_dir.mkdir(parents=True)
    replay_dir.mkdir(parents=True)
    (result_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    runs: list[dict[str, Any]] = []
    os.environ[OPENCLAW_EXEC_TIMEOUT_FLOOR_ENV] = str(EXEC_TIMEOUT_FLOOR_S)
    try:
        for declared in protocol["pairs"]:
            pair = int(declared["pair"])
            task_ids = list(declared["task_ids"])
            pair_result: dict[str, Any] = {"pair": pair, "arms": []}
            runs.append(pair_result)
            for arm in declared["arm_order"]:
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
                        )
                    )
                )
                (result_dir / "partial.json").write_text(
                    json.dumps({"protocol": protocol, "runs": runs}, indent=2) + "\n",
                    encoding="utf-8",
                )
        result = _aggregate(protocol, runs)
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


if __name__ == "__main__":
    main()
