"""Trace-driven CacheWise prefix-scheduling x C100 KV diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable, Mapping

import numpy as np

from spike.multitenant import _trace_program
from tool_resource_eval.cachewise_reproduction import (
    ClusterModel,
    Gap,
    _histories,
    _predict,
    fit_clusters,
    load_gaps,
)
from trace_collect.trace_data import TraceData


BLOCK_SIZE_TOKENS = 16
CAPACITY_TOKENS = 800_000
CAPACITY_BLOCKS = CAPACITY_TOKENS // BLOCK_SIZE_TOKENS
LOAD = 40
SEEDS = tuple(range(32))
BOOTSTRAP_DRAWS = 10_000
ARMS = ("fcfs_lru", "prefix_lru", "fcfs_c100", "prefix_c100")


@dataclass(frozen=True)
class Turn:
    prompt_tokens: int
    completion_tokens: int
    service_s: float
    gap: Gap | None


@dataclass(frozen=True)
class Program:
    task_id: str
    turns: tuple[Turn, ...]


@dataclass
class Session:
    program: Program
    seed_rank: int
    turn_index: int = 0
    arrival_s: float = 0.0
    resident_blocks: int = 0
    last_access_s: float = -math.inf
    gap_started_s: float | None = None
    request_latency_s: float = 0.0
    finished: bool = False


RemainingPredictor = Callable[[Session, float], float]
RequestKey = tuple[str, int]


def _canonical_traces(root: Path) -> dict[str, Path]:
    rows = [
        json.loads(line)
        for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    traces: dict[str, Path] = {}
    for row in rows:
        if row.get("success") is not True:
            raise ValueError("factorial diagnostic requires successful canonical rows")
        task_id = row.get("instance_id")
        raw_path = row.get("trace_file")
        if not isinstance(task_id, str) or not isinstance(raw_path, str):
            raise ValueError("results row lacks instance_id or trace_file")
        path = Path(raw_path).resolve()
        if task_id in traces or not path.is_file():
            raise ValueError(f"duplicate task or missing trace: {task_id}")
        traces[task_id] = path
    return traces


def _program(path: Path) -> Program:
    replay = _trace_program(path, command_field="command")
    trace = TraceData.load(path)
    llm_actions = sorted(
        (action for action in trace.actions if action.get("action_type") == "llm_call"),
        key=lambda action: float(action["ts_start"]),
    )
    replayable = []
    for action in llm_actions:
        data = action.get("data") or {}
        if not data.get("prompt_tokens") or not data.get("completion_tokens"):
            break
        replayable.append(action)
    gaps = load_gaps(path)
    if len(replayable) != len(replay.turns) or len(gaps) != len(replay.turns) - 1:
        raise ValueError(f"{path}: LLM turns and CacheWise gaps do not align")

    turns: list[Turn] = []
    for index, (turn, action) in enumerate(zip(replay.turns, replayable, strict=True)):
        service_s = float(action["ts_end"]) - float(action["ts_start"])
        if service_s <= 0:
            raise ValueError(f"{path}: non-positive LLM service time")
        gap = gaps[index] if index < len(gaps) else None
        if gap is not None and not math.isclose(
            gap.end - gap.start,
            turn.gap_ms / 1000.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"{path}: replay and CacheWise gap durations differ")
        turns.append(
            Turn(
                prompt_tokens=turn.recorded_prompt_tokens,
                completion_tokens=turn.completion_tokens,
                service_s=service_s,
                gap=gap,
            )
        )
    return Program(replay.task_id, tuple(turns))


def _full_blocks(tokens: int) -> int:
    return tokens // BLOCK_SIZE_TOKENS


def _allocated_blocks(tokens: int) -> int:
    return math.ceil(tokens / BLOCK_SIZE_TOKENS)


def _reusable_blocks(session: Session) -> int:
    if session.turn_index == 0:
        return 0
    prior = session.program.turns[session.turn_index - 1]
    current = session.program.turns[session.turn_index]
    prior_blocks = _full_blocks(prior.prompt_tokens + prior.completion_tokens)
    return min(prior_blocks, _full_blocks(current.prompt_tokens))


def _additional_blocks(session: Session) -> int:
    prompt = session.program.turns[session.turn_index].prompt_tokens
    return _allocated_blocks(prompt) - min(
        session.resident_blocks, _reusable_blocks(session)
    )


def _choose_session(queued: list[Session], scheduler: str) -> Session:
    def arrival_key(session: Session) -> tuple[float, int, str]:
        return session.arrival_s, session.seed_rank, session.program.task_id

    if scheduler == "fcfs":
        return min(queued, key=arrival_key)
    if scheduler != "prefix":
        raise ValueError(f"unknown scheduler {scheduler}")
    return min(
        queued,
        key=lambda session: (_additional_blocks(session), *arrival_key(session)),
    )


def _release_arrived_suffixes(
    sessions: list[Session],
    now_s: float,
    total_blocks: int,
    *,
    exclude: Session | None = None,
) -> int:
    for session in sessions:
        if (
            session is exclude
            or session.finished
            or session.arrival_s > now_s + 1e-12
        ):
            continue
        reusable = _reusable_blocks(session)
        if session.resident_blocks > reusable:
            total_blocks -= session.resident_blocks - reusable
            session.resident_blocks = reusable
    return total_blocks


def _predicted_remaining(
    session: Session,
    now_s: float,
    global_history: np.ndarray,
    tool_history: dict[str, np.ndarray],
    clusters: dict[int, dict[str, ClusterModel]],
    label_cache: dict[tuple[int, str, str], int],
) -> float:
    if session.arrival_s <= now_s or session.turn_index == 0:
        return 0.0
    prior = session.program.turns[session.turn_index - 1]
    if prior.gap is None or session.gap_started_s is None:
        return 0.0
    return _predict(
        prior.gap,
        max(0.0, now_s - session.gap_started_s),
        "c100",
        global_history,
        tool_history,
        clusters,
        label_cache,
    )


def _make_room(
    add_blocks: int,
    *,
    total_blocks: int,
    current: Session,
    sessions: list[Session],
    now_s: float,
    eviction: str,
    global_history: np.ndarray,
    tool_history: dict[str, np.ndarray],
    clusters: dict[int, dict[str, ClusterModel]],
    label_cache: dict[tuple[int, str, str], int],
    remaining_predictor: RemainingPredictor | None = None,
    next_request_rank: Mapping[RequestKey, int] | None = None,
) -> tuple[int, int, int]:
    required = max(0, total_blocks + add_blocks - CAPACITY_BLOCKS)
    if required == 0:
        return total_blocks, 0, 0
    evicted = events = 0
    while required > 0:
        candidates = [
            session
            for session in sessions
            if session is not current and session.resident_blocks > 0
        ]
        if not candidates:
            raise ValueError("request cannot fit in the frozen KV capacity")
        if eviction == "lru":
            victim = min(
                candidates,
                key=lambda session: (session.last_access_s, session.program.task_id),
            )
        elif eviction == "c100":
            victim = min(
                candidates,
                key=lambda session: (
                    -_predicted_remaining(
                        session,
                        now_s,
                        global_history,
                        tool_history,
                        clusters,
                        label_cache,
                    ),
                    session.last_access_s,
                    session.program.task_id,
                ),
            )
        elif eviction == "predicted" and remaining_predictor is not None:
            victim = min(
                candidates,
                key=lambda session: (
                    -remaining_predictor(session, now_s),
                    session.last_access_s,
                    session.program.task_id,
                ),
            )
            limit = victim.resident_blocks
        elif eviction == "belady" and next_request_rank is not None:
            free_suffix = [
                (session.resident_blocks - _reusable_blocks(session), session)
                for session in candidates
                if session.resident_blocks > _reusable_blocks(session)
            ]
            if free_suffix:
                limit, victim = max(
                    free_suffix,
                    key=lambda item: (
                        next_request_rank[
                            (item[1].program.task_id, item[1].turn_index)
                        ],
                        item[1].program.task_id,
                    ),
                )
            else:
                victim = max(
                    candidates,
                    key=lambda session: (
                        next_request_rank[
                            (session.program.task_id, session.turn_index)
                        ],
                        session.program.task_id,
                    ),
                )
                limit = victim.resident_blocks
        else:
            raise ValueError(f"unknown eviction policy {eviction}")
        if eviction not in {"predicted", "belady"}:
            limit = victim.resident_blocks
        count = min(required, limit)
        victim.resident_blocks -= count
        required -= count
        total_blocks -= count
        evicted += count
        events += 1
    return total_blocks, evicted, events


def simulate(
    programs: list[Program],
    *,
    scheduler: str,
    eviction: str,
    global_history: np.ndarray,
    tool_history: dict[str, np.ndarray],
    clusters: dict[int, dict[str, ClusterModel]],
    label_cache: dict[tuple[int, str, str], int],
    remaining_predictor: RemainingPredictor | None = None,
    next_request_rank: Mapping[RequestKey, int] | None = None,
) -> dict[str, float | int]:
    sessions = [Session(program, rank) for rank, program in enumerate(programs)]
    total_blocks = evicted_blocks = eviction_events = pressure_events = 0
    miss_blocks = request_count = 0
    request_latencies: list[float] = []
    now_s = 0.0

    while not all(session.finished for session in sessions):
        total_blocks = _release_arrived_suffixes(sessions, now_s, total_blocks)
        queued = [
            session
            for session in sessions
            if not session.finished and session.arrival_s <= now_s + 1e-12
        ]
        if not queued:
            now_s = min(
                session.arrival_s for session in sessions if not session.finished
            )
            continue

        current = _choose_session(queued, scheduler)
        turn = current.program.turns[current.turn_index]
        reusable = _reusable_blocks(current)
        miss_blocks += max(0, reusable - current.resident_blocks)
        prompt_blocks = _allocated_blocks(turn.prompt_tokens)
        add_blocks = prompt_blocks - current.resident_blocks
        before = total_blocks
        total_blocks, evicted, events = _make_room(
            add_blocks,
            total_blocks=total_blocks,
            current=current,
            sessions=sessions,
            now_s=now_s,
            eviction=eviction,
            global_history=global_history,
            tool_history=tool_history,
            clusters=clusters,
            label_cache=label_cache,
            remaining_predictor=remaining_predictor,
            next_request_rank=next_request_rank,
        )
        pressure_events += total_blocks < before
        evicted_blocks += evicted
        eviction_events += events
        current.resident_blocks = prompt_blocks
        total_blocks += add_blocks

        finish_s = now_s + turn.service_s
        total_blocks = _release_arrived_suffixes(
            sessions, finish_s, total_blocks, exclude=current
        )
        desired_blocks = _allocated_blocks(
            turn.prompt_tokens + turn.completion_tokens
        )
        growth = desired_blocks - current.resident_blocks
        if growth > 0:
            before = total_blocks
            total_blocks, evicted, events = _make_room(
                growth,
                total_blocks=total_blocks,
                current=current,
                sessions=sessions,
                now_s=finish_s,
                eviction=eviction,
                global_history=global_history,
                tool_history=tool_history,
                clusters=clusters,
                label_cache=label_cache,
                remaining_predictor=remaining_predictor,
                next_request_rank=next_request_rank,
            )
            pressure_events += total_blocks < before
            evicted_blocks += evicted
            eviction_events += events
        total_blocks += growth
        current.resident_blocks = desired_blocks
        current.last_access_s = finish_s
        latency = finish_s - current.arrival_s
        current.request_latency_s += latency
        request_latencies.append(latency)
        request_count += 1
        current.turn_index += 1

        if current.turn_index == len(current.program.turns):
            total_blocks -= current.resident_blocks
            current.resident_blocks = 0
            current.finished = True
        else:
            if turn.gap is None:
                raise ValueError("non-final turn lacks a CacheWise tool gap")
            current.gap_started_s = finish_s
            current.arrival_s = finish_s + turn.gap.end - turn.gap.start
        now_s = finish_s

    session_latencies = [session.request_latency_s for session in sessions]
    return {
        "request_count": request_count,
        "evicted_blocks": evicted_blocks,
        "eviction_events": eviction_events,
        "capacity_pressure_events": pressure_events,
        "recomputed_prefix_blocks": miss_blocks,
        "mean_request_latency_s": statistics.fmean(request_latencies),
        "mean_session_llm_latency_s": statistics.fmean(session_latencies),
        "p95_session_llm_latency_s": float(np.quantile(session_latencies, 0.95)),
        "makespan_s": now_s,
    }


def request_ranks(
    programs: list[Program], *, scheduler: str = "fcfs"
) -> dict[RequestKey, int]:
    """Return the fixed request order when cache misses do not affect service."""

    sessions = [Session(program, rank) for rank, program in enumerate(programs)]
    ranks: dict[RequestKey, int] = {}
    now_s = 0.0
    while not all(session.finished for session in sessions):
        queued = [
            session
            for session in sessions
            if not session.finished and session.arrival_s <= now_s + 1e-12
        ]
        if not queued:
            now_s = min(
                session.arrival_s for session in sessions if not session.finished
            )
            continue
        current = _choose_session(queued, scheduler)
        key = (current.program.task_id, current.turn_index)
        ranks[key] = len(ranks)
        turn = current.program.turns[current.turn_index]
        finish_s = now_s + turn.service_s
        current.turn_index += 1
        if current.turn_index == len(current.program.turns):
            current.finished = True
        else:
            if turn.gap is None:
                raise ValueError("non-final turn lacks a CacheWise tool gap")
            current.arrival_s = finish_s + turn.gap.end - turn.gap.start
        now_s = finish_s
    return ranks


def _paper_sized(
    prefix_delta: float,
    c100_delta: float,
    c100_under_prefix_delta: float,
    combined_ratio: float,
) -> bool:
    return (
        prefix_delta < 0
        and c100_delta < 0
        and c100_under_prefix_delta < 0
        and combined_ratio >= 2.0
    )


def _bootstrap(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    draws = array[rng.integers(0, len(array), (BOOTSTRAP_DRAWS, len(array)))].mean(
        axis=1
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "ci95_paired_seed_bootstrap": [float(low), float(high)],
        "draws": BOOTSTRAP_DRAWS,
    }


def _bootstrap_ratio(
    numerators: list[float], denominators: list[float]
) -> dict[str, Any]:
    numerator = np.asarray(numerators, dtype=float)
    denominator = np.asarray(denominators, dtype=float)
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(numerator), (BOOTSTRAP_DRAWS, len(numerator)))
    numerator_means = numerator[indices].mean(axis=1)
    denominator_means = denominator[indices].mean(axis=1)
    zero_draws = int(np.count_nonzero(denominator_means == 0))
    ratios = np.divide(
        numerator_means,
        denominator_means,
        out=np.full_like(numerator_means, np.inf),
        where=denominator_means != 0,
    )
    low, high = np.quantile(ratios, [0.025, 0.975], method="nearest")
    numerator_mean = float(numerator.mean())
    denominator_mean = float(denominator.mean())
    return {
        "ratio_of_means": (
            numerator_mean / denominator_mean if denominator_mean else None
        ),
        "ratio_is_infinite": denominator_mean == 0 and numerator_mean > 0,
        "ci95_paired_seed_bootstrap": [
            float(low) if math.isfinite(low) else None,
            float(high) if math.isfinite(high) else None,
        ],
        "zero_denominator_draws": zero_draws,
        "draws": BOOTSTRAP_DRAWS,
    }


def run(fit_root: Path, eval_root: Path, corpus_config: Path) -> dict[str, Any]:
    config = json.loads(corpus_config.read_text(encoding="utf-8"))
    fit_ids = list(config["fit"]["task_ids"])
    fit_paths = _canonical_traces(fit_root)
    eval_paths = _canonical_traces(eval_root)
    if not set(fit_ids) <= fit_paths.keys() or not set(fit_ids) <= eval_paths.keys():
        raise ValueError("frozen fit tasks are absent from a trace package")

    fit_gaps = [
        gap
        for task_id in fit_ids
        for gap in load_gaps(fit_paths[task_id])
    ]
    global_history, tool_history = _histories(fit_gaps)
    clusters, occupied = fit_clusters(fit_gaps, cluster_counts=(100,))
    programs = {
        task_id: _program(path)
        for task_id, path in eval_paths.items()
        if task_id not in set(fit_ids)
    }
    if len(programs) != 76:
        raise ValueError(f"expected 76 evaluation programs, found {len(programs)}")

    label_cache: dict[tuple[int, str, str], int] = {}
    schedule_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        task_ids = sorted(programs)
        np.random.default_rng(seed).shuffle(task_ids)
        selected = task_ids[:LOAD]
        row: dict[str, Any] = {"seed": seed, "task_ids": selected, "arms": {}}
        for arm in ARMS:
            scheduler, eviction = arm.split("_")
            row["arms"][arm] = simulate(
                [programs[task_id] for task_id in selected],
                scheduler=scheduler,
                eviction=eviction,
                global_history=global_history,
                tool_history=tool_history,
                clusters=clusters,
                label_cache=label_cache,
            )
        request_counts = {
            metrics["request_count"] for metrics in row["arms"].values()
        }
        if len(request_counts) != 1:
            raise ValueError("factorial arms evaluated different request rows")
        schedule_rows.append(row)

    means = {
        arm: {
            metric: float(
                np.mean([row["arms"][arm][metric] for row in schedule_rows])
            )
            for metric in schedule_rows[0]["arms"][arm]
        }
        for arm in ARMS
    }

    def comparison(candidate: str, baseline: str) -> dict[str, Any]:
        values = [
            float(row["arms"][candidate]["evicted_blocks"])
            - float(row["arms"][baseline]["evicted_blocks"])
            for row in schedule_rows
        ]
        return {
            "candidate": candidate,
            "baseline": baseline,
            "metric": "evicted_blocks; lower is better",
            **_bootstrap(values),
        }

    prefix = comparison("prefix_lru", "fcfs_lru")
    c100 = comparison("fcfs_c100", "fcfs_lru")
    c100_under_prefix = comparison("prefix_c100", "prefix_lru")
    ratio = _bootstrap_ratio(
        [float(row["arms"]["fcfs_lru"]["evicted_blocks"]) for row in schedule_rows],
        [
            float(row["arms"]["prefix_c100"]["evicted_blocks"])
            for row in schedule_rows
        ],
    )
    aggregate_ratio = (
        math.inf
        if ratio["ratio_is_infinite"]
        else float(ratio["ratio_of_means"] or 0.0)
    )
    paper_sized = _paper_sized(
        prefix["mean"],
        c100["mean"],
        c100_under_prefix["mean"],
        aggregate_ratio,
    )
    return {
        "status": "development-exposed trace-driven KV mechanism diagnostic",
        "question": "Do prefix-aware scheduling and C100 eviction reproduce CacheWise-sized KV eviction reductions on SQLGlot100?",
        "protocol": {
            "fit_root": str(fit_root.resolve()),
            "eval_root": str(eval_root.resolve()),
            "corpus_config": str(corpus_config.resolve()),
            "fit_tasks": 24,
            "evaluation_pool_tasks": 76,
            "load": LOAD,
            "seeds": list(SEEDS),
            "block_size_tokens": BLOCK_SIZE_TOKENS,
            "capacity_tokens": CAPACITY_TOKENS,
            "capacity_blocks": CAPACITY_BLOCKS,
            "arms": list(ARMS),
            "service_model": "recorded LLM service durations fixed across arms; cache misses do not feed back",
            "primary": "evicted KV blocks and eviction-induced reusable-prefix miss blocks",
            "paper_sized_criterion": "prefix_lru < fcfs_lru, fcfs_c100 < fcfs_lru, prefix_c100 < prefix_lru, and mean(fcfs_lru evicted blocks) / mean(prefix_c100) >= 2.0",
        },
        "data": {
            "fit_gaps": len(fit_gaps),
            "evaluation_programs": len(programs),
            "evaluation_turns": sum(len(program.turns) for program in programs.values()),
            "schedule_repetitions": len(schedule_rows),
            "mean_requests_per_schedule": float(
                np.mean(
                    [row["arms"]["fcfs_lru"]["request_count"] for row in schedule_rows]
                )
            ),
        },
        "cluster_model": {
            "c100_tool_batch_keys": len(clusters[100]),
            "c100_occupied_clusters": sum(occupied[100].values()),
        },
        "mean_metrics": means,
        "comparisons": {
            "prefix_alone": prefix,
            "c100_alone": c100,
            "c100_incremental_under_prefix": c100_under_prefix,
            "full_baseline_to_combined_ratio": {
                "ratio_of_mean_evicted_blocks": ratio["ratio_of_means"],
                "ratio_is_infinite": ratio["ratio_is_infinite"],
                "ci95_paired_seed_bootstrap": ratio[
                    "ci95_paired_seed_bootstrap"
                ],
                "zero_denominator_draws": ratio["zero_denominator_draws"],
                "draws": ratio["draws"],
                "paper_lower_bound": 2.0,
                "gap_to_paper_lower_bound": (
                    0.0 if math.isinf(aggregate_ratio) else max(0.0, 2.0 - aggregate_ratio)
                ),
            },
        },
        "paper_sized_effect": paper_sized,
        "decision": (
            "PAPER-SIZED in the frozen block model; live smoke still requires approval"
            if paper_sized
            else "BELOW PAPER-SIZED; stop before live CacheWise integration"
        ),
        "schedule_results": schedule_rows,
        "limitations": [
            "The cohort, prior C100 result, and simulator are development-exposed.",
            "The 800,000-token capacity is a paper-scale estimate, not a measured vLLM block count.",
            "The simulator is serial and omits continuous batching, chunked prefill, exact vLLM allocation, transfer contention, and cache-miss service feedback.",
            "Simulated latency cannot be compared numerically with CacheWise's live 2.7-3.5x session-time result.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-root",
        type=Path,
        default=Path(
            "traces/swe-rebench/gpt-5.6-sol/"
            "sqlglot-48-c2-fast-requested-ebpf-8a72722-20260803"
        ),
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path(
            "traces/swe-rebench/gpt-5.6-sol/"
            "sqlglot-100-c2-fast-requested-ebpf-a0419d9-20260803"
        ),
    )
    parser.add_argument(
        "--corpus-config",
        type=Path,
        default=Path("configs/corpora/swe-sqlglot-48-gpt56-ebpf.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/results/cachewise-sqlglot100-kv-factorial-20260804/"
            "result.json"
        ),
    )
    args = parser.parse_args()
    result = run(args.fit_root, args.eval_root, args.corpus_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
