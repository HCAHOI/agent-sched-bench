#!/usr/bin/env python3
"""Evaluate the frozen live keep-versus-deadline A/B."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spike.vllm_connector import percentile  # noqa: E402


RequestKey = tuple[int, int]
_POLICIES = ("keep", "deadline", "deadline", "keep")
_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_DEVICE = "NVIDIA A100 80GB PCIe"
_WORKLOAD = "swe-rebench-277-development-exposed"
_REPLAY_ROOT = "traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200"
_PROFILE_ROOT = "traces/swe-rebench/qwen3.7-max/20260624T162037"
_TASK_IDS_FILE = Path(
    "analysis/serving/w5-multitenant/inputs/swe-rebench-fresh277-task-ids.txt"
)


def _request_key(row: dict[str, Any]) -> RequestKey:
    return int(row["program_index"]), int(row["turn_index"])


def _request_maps(
    cell: dict[str, Any],
) -> tuple[dict[RequestKey, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_key: dict[RequestKey, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for row in cell["requests"]:
        key = _request_key(row)
        request_id = str(row["request_id"])
        if key in by_key or request_id in by_id:
            raise ValueError("duplicate live request key or id")
        by_key[key] = row
        by_id[request_id] = row
    return by_key, by_id


def _program_map(cell: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {int(row["program_index"]): row for row in cell["programs"]}
    if len(rows) != len(cell["programs"]):
        raise ValueError("duplicate live program index")
    return rows


def validate_output_parity(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> None:
    ref_requests, _ = _request_maps(reference)
    cand_requests, _ = _request_maps(candidate)
    if set(ref_requests) != set(cand_requests):
        raise ValueError("output parity request-set mismatch")
    fields = (
        "task_id",
        "messages_in",
        "prompt_token_ids_sha256",
        "output_token_ids",
        "finish_reason",
    )
    for key, ref in ref_requests.items():
        cand = cand_requests[key]
        if any(ref[field] != cand[field] for field in fields):
            raise ValueError(f"output parity mismatch at request {key}")

    ref_programs = _program_map(reference)
    cand_programs = _program_map(candidate)
    if set(ref_programs) != set(cand_programs):
        raise ValueError("output parity program-set mismatch")
    for key, ref in ref_programs.items():
        cand = cand_programs[key]
        if (ref["task_id"], ref["status"]) != (cand["task_id"], cand["status"]):
            raise ValueError(f"output parity mismatch at program {key}")


def validate_frozen_cells(cells: Sequence[dict[str, Any]]) -> None:
    if len(cells) != 4:
        raise ValueError("frozen live evaluation requires four ABBA cells")
    first = cells[0]
    expected_serving = {
        "model": _MODEL,
        "kv_cache_dtype": "auto",
        "quantization": None,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 131_072,
        "max_num_seqs": 32,
        "enforce_eager": True,
        "transfer_mode": "staged",
        "block_size": 16,
        "block_dim": 1,
        "transfer_max_blocks": 128,
        "transfer_chunk_bytes": 0,
    }
    current_sha = _git_sha()
    expected_task_ids = _TASK_IDS_FILE.read_text(encoding="utf-8").splitlines()
    for cell, policy in zip(cells, _POLICIES, strict=True):
        if cell["status"] != "complete":
            raise ValueError("live cell is not complete")
        if cell["policy"] != policy or int(cell["load"]) != 32:
            raise ValueError("live cells must use frozen keep/deadline ABBA at load 32")
        if int(cell["program_count"]) != 277 or int(cell["request_count"]) != 13_048:
            raise ValueError("live cell does not contain the frozen full workload")
        if cell["limit_programs"] is not None or cell["max_turns"] is not None:
            raise ValueError("formal live cells cannot use subset flags")
        if cell["git_sha"] != current_sha or cell["git_sha"] != first["git_sha"]:
            raise ValueError("live cell code provenance differs from the evaluator")
        if cell["vllm_version"] != "0.11.2":
            raise ValueError("live cell must use frozen vLLM 0.11.2")
        if cell["runtime"]["device_name"] != _DEVICE:
            raise ValueError("live cell did not run on the frozen A100 device")
        if cell["runtime"] != first["runtime"]:
            raise ValueError("live cells did not run on the same host and device")
        if cell["config"] != first["config"] or cell["workload"] != first["workload"]:
            raise ValueError("live cell config or workload provenance differs")
        if int(cell["config"]["seed"]) != 0:
            raise ValueError("live cell must use frozen seed 0")
        serving = cell["config"]["serving"]
        if any(serving.get(key) != value for key, value in expected_serving.items()):
            raise ValueError("live cell serving config differs from the frozen config")
        workload = cell["workload"]
        if (
            workload.get("name") != _WORKLOAD
            or workload.get("corpus_role") != "development_exposed"
            or workload.get("replay_trace_root") != _REPLAY_ROOT
            or workload.get("profile_trace_root") != _PROFILE_ROOT
            or workload.get("task_ids_file") != str(_TASK_IDS_FILE)
            or workload.get("expected_task_count") != 277
            or workload.get("expected_profile_task_count") != 50
            or workload.get("deadline_ms") != 5_000.0
        ):
            raise ValueError("live cell workload differs from the frozen workload")
        if cell.get("replay_task_ids") != expected_task_ids or {
            row["task_id"] for row in cell["programs"]
        } != set(expected_task_ids):
            raise ValueError("live cell task IDs differ from the frozen manifest")

    for keep in (cells[0], cells[3]):
        if any(
            row.get("phase") in {"retention_offload", "retention_restore"}
            for row in keep["transfers"]
        ):
            raise ValueError("keep cell unexpectedly transferred retained KV")

    intervals = [
        (
            dt.datetime.fromisoformat(cell["cell_started_at"]),
            dt.datetime.fromisoformat(cell["cell_finished_at"]),
        )
        for cell in cells
    ]
    if any(start >= finish for start, finish in intervals):
        raise ValueError("live cell has an invalid execution interval")
    if any(
        previous[1] > following[0]
        for previous, following in zip(intervals, intervals[1:])
    ):
        raise ValueError("live cells did not execute sequentially in ABBA order")


def causal_reuse_events(cell: dict[str, Any]) -> list[dict[str, Any]]:
    """Return exact freed-block reuse windows from one deadline cell."""
    _, requests = _request_maps(cell)
    transfers = cell["transfers"]
    scheduler_events = cell["retention_events"]
    restores = [row for row in transfers if row.get("phase") == "retention_restore"]
    frees = [
        row
        for row in scheduler_events
        if row.get("phase") == "retention_blocks_freed"
    ]
    admissions = [
        row
        for row in scheduler_events
        if row.get("phase") == "retention_request_admitted"
    ]
    output: list[dict[str, Any]] = []
    for offload in (
        row for row in transfers if row.get("phase") == "retention_offload"
    ):
        old_request_id = str(offload["request_id"])
        old_request = requests[old_request_id]
        owner_program, owner_turn = _request_key(old_request)
        offload_completed = float(offload["completed_monotonic_s"])
        restore_candidates = [
            row
            for row in restores
            if row.get("program_id") == offload.get("program_id")
            and row.get("request_id") in requests
            and _request_key(requests[str(row["request_id"])])
            == (owner_program, owner_turn + 1)
            and float(row["started_monotonic_s"]) > offload_completed
        ]
        if not restore_candidates:
            continue
        restore = min(
            restore_candidates, key=lambda row: float(row["started_monotonic_s"])
        )
        restore_started = float(restore["started_monotonic_s"])
        free_candidates = [
            row
            for row in frees
            if row.get("request_id") == old_request_id
            and float(row["monotonic_s"]) >= offload_completed
            and int(row["free_blocks_after"]) > int(row["free_blocks_before"])
            and float(row["monotonic_s"]) < restore_started
        ]
        if not free_candidates:
            continue
        freed = min(free_candidates, key=lambda row: float(row["monotonic_s"]))
        freed_at = float(freed["monotonic_s"])
        freed_blocks = {int(block_id) for block_id in freed["block_ids"]}
        for admitted in admissions:
            admitted_at = float(admitted["monotonic_s"])
            request_id = str(admitted["request_id"])
            if not freed_at < admitted_at < restore_started or request_id not in requests:
                continue
            admitted_request = requests[request_id]
            admitted_key = _request_key(admitted_request)
            if admitted_key[0] == owner_program:
                continue
            reused = sorted(
                freed_blocks & {int(block_id) for block_id in admitted["block_ids"]}
            )
            if not reused:
                continue
            reused_block_seconds = 0.0
            for block_id in reused:
                released_again = min(
                    (
                        float(row["monotonic_s"])
                        for row in frees
                        if row.get("request_id") == request_id
                        and float(row["monotonic_s"]) > admitted_at
                        and block_id in {int(value) for value in row["block_ids"]}
                    ),
                    default=restore_started,
                )
                reused_block_seconds += max(
                    0.0, min(restore_started, released_again) - admitted_at
                )
            output.append(
                {
                    "offload_request_id": old_request_id,
                    "owner_program_index": owner_program,
                    "owner_turn_index": owner_turn,
                    "admitted_request_key": list(admitted_key),
                    "freed_at_monotonic_s": freed_at,
                    "admitted_at_monotonic_s": admitted_at,
                    "restore_started_monotonic_s": restore_started,
                    "freed_block_ids": sorted(freed_blocks),
                    "reused_block_ids": reused,
                    "freed_block_seconds": len(freed_blocks)
                    * (restore_started - freed_at),
                    "reused_block_seconds": reused_block_seconds,
                }
            )
    return output


def _mean_program_jct(cell: dict[str, Any]) -> float:
    programs = _program_map(cell)
    if any(row["status"] != "replayed_complete" for row in programs.values()):
        raise ValueError("live cell contains an incomplete program")
    return statistics.fmean(float(row["jct_ms"]) for row in programs.values())


def _p99(rows: Sequence[dict[str, Any]], keys: set[RequestKey] | None = None) -> float:
    values = [
        float(row["ttft_ms"])
        for row in rows
        if keys is None or _request_key(row) in keys
    ]
    return percentile(values, 99.0)


def _transfer_summary(cell: dict[str, Any]) -> dict[str, int]:
    rows = [row for row in cell["transfers"] if "bytes_moved" in row]
    return {
        "copy_count": len(rows),
        "bytes_moved": sum(int(row["bytes_moved"]) for row in rows),
        "offload_count": sum(row.get("phase") == "retention_offload" for row in rows),
        "restore_count": sum(row.get("phase") == "retention_restore" for row in rows),
    }


def evaluate(
    keep_first: dict[str, Any],
    deadline_first: dict[str, Any],
    deadline_second: dict[str, Any],
    keep_second: dict[str, Any],
) -> dict[str, Any]:
    cells = (keep_first, deadline_first, deadline_second, keep_second)
    parity_error: str | None = None
    try:
        validate_frozen_cells(cells)
        for cell in cells[1:]:
            validate_output_parity(keep_first, cell)
    except ValueError as error:
        parity_error = str(error)

    reuse_first = causal_reuse_events(deadline_first)
    reuse_second = causal_reuse_events(deadline_second)
    reuse_rows = (reuse_first, reuse_second)
    pairs = ((keep_first, deadline_first), (keep_second, deadline_second))
    pair_rows = []
    action_gates = []
    direction_gates = []
    tail_gates = []
    for index, ((keep, deadline), reuse) in enumerate(
        zip(pairs, reuse_rows, strict=True), start=1
    ):
        owner_count = len({int(row["owner_program_index"]) for row in reuse})
        affected = {
            tuple(map(int, row["admitted_request_key"])) for row in reuse
        }
        keep_jct = _mean_program_jct(keep)
        deadline_jct = _mean_program_jct(deadline)
        overall_keep_p99 = _p99(keep["requests"])
        overall_deadline_p99 = _p99(deadline["requests"])
        affected_keep_p99 = _p99(keep["requests"], affected) if affected else None
        affected_deadline_p99 = (
            _p99(deadline["requests"], affected) if affected else None
        )
        action_go = owner_count >= 20
        direction_go = deadline_jct < keep_jct
        tail_go = (
            affected_keep_p99 is not None
            and affected_deadline_p99 is not None
            and overall_deadline_p99 <= 1.05 * overall_keep_p99
            and affected_deadline_p99 <= 1.05 * affected_keep_p99
        )
        action_gates.append(action_go)
        direction_gates.append(direction_go)
        tail_gates.append(tail_go)
        pair_rows.append(
            {
                "repetition": index,
                "reuse_owner_program_count": owner_count,
                "affected_request_count": len(affected),
                "keep_mean_program_jct_ms": keep_jct,
                "deadline_mean_program_jct_ms": deadline_jct,
                "jct_reduction_fraction": (keep_jct - deadline_jct) / keep_jct,
                "keep_all_request_p99_ttft_ms": overall_keep_p99,
                "deadline_all_request_p99_ttft_ms": overall_deadline_p99,
                "keep_affected_p99_ttft_ms": affected_keep_p99,
                "deadline_affected_p99_ttft_ms": affected_deadline_p99,
                "action_go": action_go,
                "direction_go": direction_go,
                "tail_go": tail_go,
            }
        )

    keep_programs = (_program_map(keep_first), _program_map(keep_second))
    deadline_programs = (
        _program_map(deadline_first),
        _program_map(deadline_second),
    )
    program_ids = sorted(keep_programs[0])
    keep_mean = statistics.fmean(
        statistics.fmean(float(rows[index]["jct_ms"]) for rows in keep_programs)
        for index in program_ids
    )
    deadline_mean = statistics.fmean(
        statistics.fmean(float(rows[index]["jct_ms"]) for rows in deadline_programs)
        for index in program_ids
    )
    aggregate_reduction = (keep_mean - deadline_mean) / keep_mean
    effect_go = aggregate_reduction >= 0.05
    validity_go = parity_error is None
    live_go = (
        validity_go
        and all(action_gates)
        and effect_go
        and all(direction_gates)
        and all(tail_gates)
    )

    return {
        "schema_version": 1,
        "status": "go" if live_go else "no_go",
        "protocol": "tool-resource-canonical-objective.md Section 5.3",
        "validity": {"output_parity": validity_go, "error": parity_error},
        "aggregate": {
            "keep_mean_program_jct_ms": keep_mean,
            "deadline_mean_program_jct_ms": deadline_mean,
            "jct_reduction_fraction": aggregate_reduction,
            "freed_block_seconds": [
                sum(
                    next(
                        row["freed_block_seconds"]
                        for row in reuse
                        if row["offload_request_id"] == request_id
                    )
                    for request_id in {
                        row["offload_request_id"] for row in reuse
                    }
                )
                for reuse in reuse_rows
            ],
            "reused_block_seconds": [
                sum(row["reused_block_seconds"] for row in reuse)
                for reuse in reuse_rows
            ],
        },
        "pairs": pair_rows,
        "transfers": {
            "keep_first": _transfer_summary(keep_first),
            "deadline_first": _transfer_summary(deadline_first),
            "deadline_second": _transfer_summary(deadline_second),
            "keep_second": _transfer_summary(keep_second),
        },
        "gates": {
            "validity_go": validity_go,
            "action_go_each_repetition": action_gates,
            "effect_go": effect_go,
            "direction_go_each_repetition": direction_gates,
            "tail_go_each_repetition": tail_gates,
            "live_go": live_go,
        },
        "causal_reuse_events": [reuse_first, reuse_second],
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-first", type=Path, required=True)
    parser.add_argument("--deadline-first", type=Path, required=True)
    parser.add_argument("--deadline-second", type=Path, required=True)
    parser.add_argument("--keep-second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = (
        args.keep_first,
        args.deadline_first,
        args.deadline_second,
        args.keep_second,
    )
    result = evaluate(
        *(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    )
    result.update(
        {
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "inputs": [str(path) for path in paths],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gates": result["gates"]}, indent=2))


if __name__ == "__main__":
    main()
