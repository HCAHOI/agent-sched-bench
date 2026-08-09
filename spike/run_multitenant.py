#!/usr/bin/env python3
"""Run one real-trace multi-tenant serving cell on a GPU host."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
import zlib
from collections.abc import Awaitable, Callable
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from spike.multitenant import (
    RetentionPlan,
    ToolSpan,
    TraceTurn,
    build_continuum_profile,
    build_retention_plan,
    load_prefill_cost_profile,
    load_trace_programs,
    policy_provenance,
    prepare_llama_chat_messages,
    validate_serving_cell,
)
from spike.trigger_table import load_trigger_table
from spike.vllm_connector import OffloadControl, percentile
from spike.vllm_connector.core import continuum_priority

MULTITENANT_RESULT_SCHEMA_VERSION = 1


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_config(
    path: Path, workload_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("workloads"), list):
        raise ValueError(f"{path}: workloads must be a list")
    matches = [row for row in config["workloads"] if row.get("name") == workload_name]
    if len(matches) != 1:
        raise ValueError(
            f"{path}: expected exactly one workload named {workload_name!r}"
        )
    return config, matches[0]


def _read_task_ids(paths: str | list[str] | None) -> list[str] | None:
    if paths is None:
        return None
    source_paths = [paths] if isinstance(paths, str) else paths
    if not source_paths or not all(isinstance(path, str) for path in source_paths):
        raise ValueError("task_ids_file must be a path or non-empty path list")
    task_ids = [
        task_id
        for path in source_paths
        for task_id in Path(path).read_text(encoding="utf-8").splitlines()
    ]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{source_paths}: task IDs must be non-empty and unique")
    return task_ids


def _validate_corpus_role(
    workload: dict[str, Any], replay_task_ids: set[str], profile_task_ids: set[str]
) -> None:
    role = workload.get("corpus_role")
    if role not in {"development", "development_exposed", "heldout_eval"}:
        raise ValueError(
            "corpus_role must be 'development', 'development_exposed', or "
            "'heldout_eval'"
        )
    expected = workload.get("expected_task_count")
    if not isinstance(expected, int) or expected <= 0:
        raise ValueError("expected_task_count must be a positive integer")
    if len(replay_task_ids) != expected:
        raise ValueError(
            f"workload expects {expected} replay tasks, loaded {len(replay_task_ids)}"
        )
    if role == "heldout_eval" and replay_task_ids & profile_task_ids:
        overlap = sorted(replay_task_ids & profile_task_ids)
        raise ValueError(f"held-out replay/profile task overlap: {overlap[:5]}")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def _read_transfer_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _transfer_totals(events: list[dict[str, Any]]) -> tuple[int, int]:
    copies = [row for row in events if "bytes_moved" in row]
    return len(copies), sum(row["bytes_moved"] for row in copies)


def _gpu_sample() -> dict[str, Any]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in proc.stdout.splitlines():
        timestamp, memory_mib, utilization = (part.strip() for part in line.split(","))
        rows.append(
            {
                "timestamp": timestamp,
                "memory_used_mib": float(memory_mib),
                "utilization_percent": float(utilization),
            }
        )
    return {"monotonic_s": time.perf_counter(), "gpus": rows}


def _runtime_device() -> tuple[str, str]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    devices = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(devices) != 1:
        raise RuntimeError("multi-tenant GPU run requires exactly one visible GPU")
    return platform.node(), devices[0]


async def _sample_gpu(stop: asyncio.Event, interval_s: float) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    while not stop.is_set():
        samples.append(await asyncio.to_thread(_gpu_sample))
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            pass
    return samples


async def _replay_tool_lifecycles(
    turn: TraceTurn,
    *,
    gap_started_s: float,
    sleep_until: Callable[[float], Awaitable[None]],
    reveal: Callable[[int, str, str], Awaitable[float | None]],
    fire: Callable[[int], Awaitable[None]],
) -> None:
    """Reveal starts and cancel unfired triggers at causal tool-end events."""

    async def replay_one(tool_index: int, tool: ToolSpan) -> None:
        start_at = gap_started_s + tool.start_offset_ms / 1000.0
        end_at = gap_started_s + tool.end_offset_ms / 1000.0
        await sleep_until(start_at)
        trigger_ms = await reveal(tool_index, tool.tool_name, tool.command)
        if trigger_ms is None:
            await sleep_until(end_at)
            return
        if (
            isinstance(trigger_ms, bool)
            or not isinstance(trigger_ms, (int, float))
            or not math.isfinite(trigger_ms)
            or trigger_ms < 0
        ):
            raise ValueError("tool trigger must be finite and >= 0")
        trigger_offset_ms = tool.start_offset_ms + trigger_ms
        if trigger_offset_ms >= tool.end_offset_ms or trigger_offset_ms >= turn.gap_ms:
            await sleep_until(end_at)
            return
        trigger_task = asyncio.create_task(
            sleep_until(gap_started_s + trigger_offset_ms / 1000.0)
        )
        await trigger_task
        await fire(tool_index)
        await sleep_until(end_at)

    await asyncio.gather(
        *(replay_one(index, tool) for index, tool in enumerate(turn.tools))
    )


def _causal_trigger_delay_ms(policy: str, plan: RetentionPlan | None) -> float | None:
    if plan is None:
        return 0.0 if policy == "continuum" else None
    return plan.expire_ms


def _request_metrics(
    output: Any, submitted_at: float, first_token_at: float
) -> dict[str, float]:
    metrics = output.metrics
    if metrics is None:
        raise ValueError("vLLM output lacks request metrics")
    queued = float(metrics.queued_ts)
    scheduled = float(metrics.scheduled_ts)
    first_token = float(metrics.first_token_ts)
    finished = float(metrics.last_token_ts)
    if not 0.0 < queued <= scheduled <= first_token <= finished:
        raise ValueError("vLLM request metrics have invalid timestamp order")
    return {
        "queue_ms": (scheduled - queued) * 1000.0,
        "prefill_ms": (first_token - scheduled) * 1000.0,
        "ttft_ms": (first_token_at - submitted_at) * 1000.0,
        "latency_ms": (time.perf_counter() - submitted_at) * 1000.0,
    }


async def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    cell_started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    config, workload = _load_config(args.config, args.workload)
    if args.policy not in config["policies"]:
        raise ValueError(f"policy {args.policy!r} is not enabled by the config")
    if args.load not in config["load_levels"]:
        raise ValueError(f"load {args.load} is not enabled by the config")
    if args.limit_programs is not None and args.limit_programs <= 0:
        raise ValueError("--limit-programs must be > 0")
    if args.max_turns is not None and args.max_turns <= 0:
        raise ValueError("--max-turns must be > 0")
    if workload["corpus_role"] == "heldout_eval" and not getattr(args, "final", False):
        raise ValueError("heldout_eval workloads require --final")
    if args.limit_programs is not None and workload["corpus_role"] == "heldout_eval":
        raise ValueError("--limit-programs is forbidden for heldout_eval workloads")
    if args.max_turns is not None and workload["corpus_role"] == "heldout_eval":
        raise ValueError("--max-turns is forbidden for heldout_eval workloads")
    validate_serving_cell(workload["corpus_role"], args.policy)
    serving = config["serving"]
    runtime_host, runtime_device = _runtime_device()
    prefill_profile = (
        load_prefill_cost_profile(
            config["continuum_prefill_profile"],
            expected_model=serving["model"],
            expected_kv_cache_dtype=serving["kv_cache_dtype"],
            expected_quantization=serving["quantization"],
        )
        if args.policy == "continuum"
        else None
    )
    if prefill_profile is not None:
        if (
            prefill_profile.host_name != runtime_host
            or prefill_profile.device_name != runtime_device
        ):
            raise ValueError(
                "Continuum prefill profile device "
                f"{(prefill_profile.host_name, prefill_profile.device_name)!r} "
                f"does not match serving device {(runtime_host, runtime_device)!r}"
            )

    replay_ids = _read_task_ids(workload.get("task_ids_file"))
    replay_programs = load_trace_programs(
        workload["replay_trace_root"], task_ids=replay_ids, seed=config["seed"]
    )
    profile_programs = load_trace_programs(
        workload["profile_trace_root"], seed=config["seed"]
    )
    _validate_corpus_role(
        workload,
        {program.task_id for program in replay_programs},
        {program.task_id for program in profile_programs},
    )
    if args.limit_programs is not None:
        replay_programs = replay_programs[: args.limit_programs]
    continuum_profile = build_continuum_profile(profile_programs)
    trigger_table = load_trigger_table(config["trigger_table"])
    out_json = args.output
    runtime_dir = out_json.with_suffix("").with_name(out_json.stem + "-runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    control_path = runtime_dir / "offload-control.json"
    retention_path = runtime_dir / "retention-control.json"
    transfer_path = runtime_dir / "transfers.jsonl"
    retention_events_path = runtime_dir / "retention-events.jsonl"
    OffloadControl(control_path).clear()
    _atomic_json(retention_path, {})
    transfer_path.unlink(missing_ok=True)
    retention_events_path.unlink(missing_ok=True)

    queue_window = int(config["continuum_queue_window"])
    if queue_window <= 0:
        raise ValueError("continuum_queue_window must be > 0")
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    KVConnectorFactory.register_connector(
        "SelectiveOffloadConnector",
        "spike.vllm_connector.gpu",
        "SelectiveOffloadConnector",
    )

    kv_config = KVTransferConfig(
        kv_connector="SelectiveOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "control_path": str(control_path),
            "timing_path": str(transfer_path),
            "retention_path": str(retention_path),
            "retention_events_path": str(retention_events_path),
            "transfer_mode": serving["transfer_mode"],
            "block_dim": serving["block_dim"],
            "max_blocks": serving["transfer_max_blocks"],
            "continuum_priority_stride": (
                len(replay_programs) if args.policy == "continuum" else 0
            ),
            "chunk_bytes": serving["transfer_chunk_bytes"],
            "thunderagent_buffer_tokens": policy_provenance()["thunderagent"][
                "buffer_per_program_tokens"
            ],
        },
    )
    engine_args = AsyncEngineArgs(
        model=serving["model"],
        kv_cache_dtype=serving["kv_cache_dtype"],
        quantization=serving["quantization"],
        gpu_memory_utilization=serving["gpu_memory_utilization"],
        tensor_parallel_size=serving["tensor_parallel_size"],
        max_model_len=serving["max_model_len"],
        max_num_seqs=serving["max_num_seqs"],
        block_size=serving["block_size"],
        scheduling_policy=("priority" if args.policy == "continuum" else "fcfs"),
        enable_prefix_caching=True,
        enforce_eager=serving["enforce_eager"],
        seed=config["seed"],
        scheduler_cls="spike.vllm_connector.scheduler.RetentionScheduler",
        kv_transfer_config=kv_config,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = await engine.get_tokenizer()

    retention_specs: dict[str, Any] = {}
    request_rows: list[dict[str, Any]] = []
    prerestore_rows: list[dict[str, Any]] = []
    program_rows: list[dict[str, Any]] = []
    observed_queue_ms: deque[float] = deque(maxlen=queue_window)
    semaphore = asyncio.Semaphore(args.load)
    result_lock = asyncio.Lock()
    stop_gpu = asyncio.Event()
    gpu_task = asyncio.create_task(
        _sample_gpu(stop_gpu, serving["gpu_sample_interval_s"])
    )
    run_started = time.perf_counter()

    async def sleep_until(deadline_s: float) -> None:
        await asyncio.sleep(max(0.0, deadline_s - time.monotonic()))

    async def run_program(program_index: int) -> None:
        program = replay_programs[program_index]
        async with semaphore:
            program_started = time.perf_counter()
            previous_retained_tokens = 0
            turn_limit = len(program.turns)
            if args.max_turns is not None:
                turn_limit = min(turn_limit, args.max_turns)
            truncated = turn_limit < len(program.turns)
            source_failed = (
                args.max_turns is None
                and turn_limit == len(program.turns)
                and program.omitted_terminal_llm_calls > 0
            )
            for turn_index, turn in enumerate(program.turns[:turn_limit]):
                request_id = f"{args.policy}:{program_index}:{turn_index}"
                token_ids = tokenizer.apply_chat_template(
                    prepare_llama_chat_messages(turn.messages),
                    tokenize=True,
                    add_generation_prompt=True,
                )
                max_tokens = turn.completion_tokens
                if max_tokens <= 0:
                    raise ValueError(
                        f"{program.trace_path}: turn {turn_index} has no completion tokens"
                    )
                if len(token_ids) + max_tokens > serving["max_model_len"]:
                    raise ValueError(
                        f"{program.task_id} turn {turn_index} needs "
                        f"{len(token_ids) + max_tokens} tokens, over max_model_len"
                    )
                prefill_reload_ms = (
                    prefill_profile.estimate_ms(len(token_ids))
                    if prefill_profile is not None
                    else 0.0
                )
                queue_estimate = (
                    statistics.fmean(observed_queue_ms) if observed_queue_ms else 0.0
                )
                final = turn_index + 1 == turn_limit and not source_failed
                provisional_action = None
                if not final:
                    provisional_action = {
                        "cache": None,
                        "keep": "offload",
                        "deadline": "offload",
                        "ours": "offload",
                        "continuum": "release",
                        "thunderagent": "pressure",
                    }[args.policy]
                retention_specs[request_id] = {
                    "program_id": f"program:{program_index}",
                    "policy": args.policy,
                    "program_arrival_s": program_started,
                    "program_index": program_index,
                    "priority_stride": len(replay_programs),
                    "action": provisional_action,
                    "expire_ms": None,
                    "final": final,
                }
                _atomic_json(retention_path, retention_specs)

                sampling = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    ignore_eos=True,
                    seed=zlib.crc32(
                        f"{config['seed']}:{program_index}:{turn_index}".encode()
                    ),
                )
                request_priority = continuum_priority(
                    program_index,
                    len(replay_programs),
                    ttl_hit=False,
                )
                submitted = time.perf_counter()
                first_token_at: float | None = None
                final_output = None
                async for output in engine.generate(
                    {"prompt_token_ids": token_ids},
                    sampling,
                    request_id,
                    priority=(request_priority if args.policy == "continuum" else 0),
                ):
                    if first_token_at is None and output.outputs[0].token_ids:
                        first_token_at = time.perf_counter()
                    final_output = output
                if final_output is None or first_token_at is None:
                    raise RuntimeError(f"request {request_id} produced no output")
                gap_started = time.monotonic()
                timing = _request_metrics(final_output, submitted, first_token_at)
                cached_tokens = int(
                    getattr(final_output.metrics, "num_cached_tokens", 0) or 0
                )
                continuum_retention_hit = (
                    previous_retained_tokens > 0
                    and cached_tokens >= previous_retained_tokens
                )
                if (
                    args.policy == "continuum"
                    and turn_index > 0
                    and not continuum_retention_hit
                ):
                    observed_queue_ms.append(timing["queue_ms"])
                previous_retained_tokens = (
                    max(0, len(token_ids) - 1) // serving["block_size"]
                ) * serving["block_size"]
                completion = final_output.outputs[0]
                plan = None
                tool_reveals: list[dict[str, Any]] = []
                row = {
                    "request_id": request_id,
                    "program_index": program_index,
                    "task_id": program.task_id,
                    "trace_path": program.trace_path,
                    "turn_index": turn_index,
                    "scheduler_priority": None,
                    "continuum_ttl_priority": None,
                    "continuum_preempted_priority": None,
                    "continuum_retention_hit": (
                        continuum_retention_hit if args.policy == "continuum" else None
                    ),
                    "messages_in": list(turn.messages),
                    "prompt_token_ids_sha256": hashlib.sha256(
                        json.dumps(token_ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "output_text": completion.text,
                    "output_token_ids": completion.token_ids,
                    "finish_reason": completion.finish_reason,
                    "recorded_prompt_tokens": turn.recorded_prompt_tokens,
                    "actual_prompt_tokens": len(token_ids),
                    "recorded_completion_tokens": turn.completion_tokens,
                    "actual_completion_tokens": len(completion.token_ids),
                    "recorded_gap_after_ms": turn.gap_ms,
                    "tool_spans": [asdict(span) for span in turn.tools],
                    "tool_reveals": tool_reveals,
                    "retention_plan": None,
                    **timing,
                    "cached_tokens": cached_tokens,
                }
                async with result_lock:
                    request_rows.append(row)
                if not final:
                    gap_deadline = gap_started + turn.gap_ms / 1000.0
                    arrival = asyncio.create_task(sleep_until(gap_deadline))
                    revealed_tools: list[ToolSpan] = []
                    armed_expiry: float | None = None

                    async def reveal(
                        tool_index: int, tool_name: str, command: str
                    ) -> float | None:
                        nonlocal plan
                        revealed_tools.append(ToolSpan(tool_name, command, 0.0, 0.0))
                        revealed_turn = TraceTurn(
                            messages=(),
                            recorded_prompt_tokens=0,
                            completion_tokens=0,
                            gap_ms=0.0,
                            tools=(
                                tuple(revealed_tools)
                                if args.policy == "continuum"
                                else (revealed_tools[-1],)
                            ),
                        )
                        plan = build_retention_plan(
                            args.policy,
                            revealed_turn,
                            deadline_ms=workload["deadline_ms"],
                            trigger_table=trigger_table,
                            continuum_profile=continuum_profile,
                            queue_delay_ms=queue_estimate,
                            prefill_reload_ms=prefill_reload_ms,
                        )
                        revealed_at = time.monotonic()
                        trigger_delay_ms = _causal_trigger_delay_ms(args.policy, plan)
                        tool_reveals.append(
                            {
                                "tool_index": tool_index,
                                "actual_reveal_ms": (revealed_at - gap_started)
                                * 1000.0,
                                "plan_source": None if plan is None else plan.source,
                                "trigger_delay_ms": trigger_delay_ms,
                                "trigger_fired": False,
                            }
                        )
                        row["retention_plan"] = None if plan is None else asdict(plan)
                        return trigger_delay_ms

                    async def fire(tool_index: int) -> None:
                        nonlocal armed_expiry
                        fired_at = time.monotonic()
                        reveal_row = next(
                            item
                            for item in tool_reveals
                            if item["tool_index"] == tool_index
                        )
                        reveal_row["trigger_fired"] = True
                        reveal_row["actual_trigger_ms"] = (
                            fired_at - gap_started
                        ) * 1000.0
                        if armed_expiry is None or fired_at < armed_expiry:
                            armed_expiry = fired_at
                            reveal_row["armed_expiry"] = True
                            retention_specs[request_id]["expire_at_monotonic_s"] = (
                                fired_at
                            )
                            _atomic_json(retention_path, retention_specs)
                            await engine.engine_core.abort_requests_async(
                                [f"retention-tick:{request_id}:{tool_index}"]
                            )
                        else:
                            reveal_row["armed_expiry"] = False

                    tool_events = asyncio.create_task(
                        _replay_tool_lifecycles(
                            turn,
                            gap_started_s=gap_started,
                            sleep_until=sleep_until,
                            reveal=reveal,
                            fire=fire,
                        )
                    )
                    await arrival
                    if tool_events.done():
                        await tool_events
                    else:
                        tool_events.cancel()
                        await asyncio.gather(tool_events, return_exceptions=True)
                retention_specs.pop(request_id, None)
                _atomic_json(retention_path, retention_specs)
            if source_failed:
                releases = retention_specs.setdefault("__release_programs__", [])
                releases.append(f"program:{program_index}")
                _atomic_json(retention_path, retention_specs)
                await engine.engine_core.abort_requests_async(
                    [f"terminal-failure:{program_index}"]
                )
            async with result_lock:
                program_rows.append(
                    {
                        "program_index": program_index,
                        "task_id": program.task_id,
                        "turns": turn_limit,
                        "source_llm_call_count": program.source_llm_call_count,
                        "omitted_terminal_llm_calls": (
                            program.omitted_terminal_llm_calls if source_failed else 0
                        ),
                        "status": (
                            "source_terminal_llm_failure"
                            if source_failed
                            else "truncated"
                            if truncated
                            else "replayed_complete"
                        ),
                        "jct_ms": (time.perf_counter() - program_started) * 1000.0,
                    }
                )

    try:
        await asyncio.gather(
            *(run_program(index) for index in range(len(replay_programs)))
        )
        run_finished = time.perf_counter()
    finally:
        stop_gpu.set()
    gpu_samples = await gpu_task
    request_rows.sort(key=lambda row: (row["program_index"], row["turn_index"]))
    prerestore_rows.sort(
        key=lambda row: (row["program_index"], row["after_turn_index"])
    )
    program_rows.sort(key=lambda row: row["program_index"])
    transfers = _read_transfer_events(transfer_path)
    retention_events = _read_transfer_events(retention_events_path)
    transfer_count, transfer_bytes = _transfer_totals(transfers)
    if args.policy == "continuum":
        priority_events = {
            row["request_id"]: row
            for row in transfers
            if row.get("phase") == "continuum_priority"
        }
        for request_row in request_rows:
            priority = priority_events.get(request_row["request_id"])
            if priority is None:
                raise RuntimeError(
                    f"missing live Continuum priority for {request_row['request_id']}"
                )
            request_row["scheduler_priority"] = priority["priority"]
            request_row["continuum_ttl_priority"] = priority["ttl_hit"]
            request_row["continuum_preempted_priority"] = priority["preempted"]
    total_tokens = sum(row["actual_completion_tokens"] for row in request_rows)
    elapsed_s = run_finished - run_started
    completed_programs = [
        row for row in program_rows if row["status"] == "replayed_complete"
    ]
    summary = {
        "program_jct_ms": _percentiles([row["jct_ms"] for row in completed_programs]),
        "request_latency_ms": _percentiles([row["latency_ms"] for row in request_rows]),
        "queue_ms": _percentiles([row["queue_ms"] for row in request_rows]),
        "ttft_ms": _percentiles([row["ttft_ms"] for row in request_rows]),
        "completion_throughput_tokens_per_s": total_tokens / elapsed_s,
        "program_status_counts": {
            status: sum(row["status"] == status for row in program_rows)
            for status in (
                "replayed_complete",
                "source_terminal_llm_failure",
                "truncated",
            )
        },
        "prerestore": {
            "timer_count": len(prerestore_rows),
            "started_count": sum(
                row["status"] == "completed" for row in prerestore_rows
            ),
            "completed_before_arrival_count": sum(
                row.get("completed_before_arrival") is True for row in prerestore_rows
            ),
            "host_restore_count": sum(
                row.get("host_restore_observed") is True for row in prerestore_rows
            ),
        },
        "transfer_count": transfer_count,
        "transfer_bytes": transfer_bytes,
        "peak_gpu_memory_mib": max(
            gpu["memory_used_mib"] for sample in gpu_samples for gpu in sample["gpus"]
        ),
    }
    import vllm

    result = {
        "schema_version": MULTITENANT_RESULT_SCHEMA_VERSION,
        "status": "complete",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "vllm_version": vllm.__version__,
        "cell_started_at": cell_started_at,
        "runtime": {
            "host_name": runtime_host,
            "device_name": runtime_device,
        },
        "config_path": str(args.config),
        "config": config,
        "policy": args.policy,
        "load": args.load,
        "workload": workload,
        "replay_task_ids": replay_ids,
        "corpus_role": workload["corpus_role"],
        "limit_programs": args.limit_programs,
        "max_turns": args.max_turns,
        "policy_provenance": policy_provenance(),
        "retained_scope": (
            "stock vLLM prefix cache; finished blocks are evictable"
            if args.policy == "cache"
            else (
                "complete prompt blocks excluding the final prompt token; "
                "partial-prompt and generated-completion KV are freed"
            )
        ),
        "continuum_profile": {
            "program_count": len(profile_programs),
            "gap_count": len(continuum_profile.global_gap_ms),
            "memoryfulness": continuum_profile.memoryfulness,
            "queue_estimator": (
                f"last {queue_window} observed eviction/miss queue delays"
            ),
            "prefill_estimator": (
                None
                if prefill_profile is None
                else {
                    "path": config["continuum_prefill_profile"],
                    "model": prefill_profile.model,
                    "kv_cache_dtype": prefill_profile.kv_cache_dtype,
                    "quantization": prefill_profile.quantization,
                    "device_name": prefill_profile.device_name,
                    "host_name": prefill_profile.host_name,
                    "max_context_tokens": prefill_profile.max_context_tokens,
                    "coefficients": prefill_profile.coefficients,
                    "overhead_floor_ms": prefill_profile.overhead_floor_ms,
                }
            ),
        },
        "prerestore_profile": {
            "enabled": False,
            "optimizer": None,
            "mechanism": (
                "disabled until pre-restore planning consumes only causally "
                "revealed tool events"
            ),
        },
        "program_count": len(replay_programs),
        "request_count": len(request_rows),
        "elapsed_s": elapsed_s,
        "summary": summary,
        "programs": program_rows,
        "requests": request_rows,
        "prerestore_events": prerestore_rows,
        "transfers": transfers,
        "retention_events": retention_events,
        "gpu_samples": gpu_samples,
    }
    result["cell_finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _atomic_json(out_json, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--load", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-programs", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--final", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = asyncio.run(run_cell(args))
    print(
        json.dumps({"output": str(args.output), "summary": result["summary"]}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
