"""Development-only prequential updates for the latency-prefix profile.

The initial profile and its gate are fitted offline.  Evaluation then compares
three information schedules over a task-disjoint stream:

* ``frozen`` never changes the initial profile;
* ``task`` publishes every observation only after its logical task resolves;
* ``call`` publishes an observation after the tool returns and a measured
  single-worker update finishes.

A decision reads only fully published versions.  Calls with the same start time
share one snapshot, and an observation can never update its own decision.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from time import perf_counter_ns
from typing import Any, Literal

from trace_collect.command_features import make_row_command_prefix_keys
from trace_collect.latency_validation import (
    required_nonnegative_float,
    required_text,
)
from trace_collect.tool_latency_profiled import (
    build_latency_prior,
    latency_prior_hierarchy,
    validate_profile_eval_disjoint,
)
from trace_collect.tool_latency_utility_clock import (
    robust_prior_nodes,
    robust_utility_trigger_stats,
)

UpdateMode = Literal["frozen", "task", "call"]
_UPDATE_MODES = frozenset({"frozen", "task", "call"})


@dataclass(frozen=True)
class UpdateBenchmark:
    """Measured cost of publishing one completed tool observation."""

    sample_id: str
    task_id: str
    runtime_ms: float

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "runtime_ms": self.runtime_ms,
        }


class EvolvingLatencyProfile:
    """An exclusively owned latency prior with incremental sorted inserts."""

    def __init__(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        row_group_keys: Callable[[dict[str, Any]], tuple[str, ...]] | None,
    ) -> None:
        profile_rows = list(rows)
        self._row_group_keys = row_group_keys
        self.prior = build_latency_prior(
            profile_rows,
            row_group_keys=row_group_keys,
        )
        self._sample_ids: set[str] = set()
        self._task_by_source: dict[str, str] = {}
        payloads = []
        for index, row in enumerate(profile_rows):
            payload = self._payload(row, source=f"initial profile row {index}")
            sample_id = payload["sample_id"]
            if sample_id in self._sample_ids:
                raise ValueError(f"duplicate initial profile sample_id: {sample_id!r}")
            self._sample_ids.add(sample_id)
            self._record_source_task(payload, source=f"initial profile row {index}")
            payloads.append(payload)
        digest = hashlib.sha256()
        for payload in sorted(payloads, key=lambda item: item["sample_id"]):
            digest.update(_canonical_json(payload))
            digest.update(b"\n")
        self.version = 0
        self.state_hash = digest.hexdigest()

    @property
    def row_count(self) -> int:
        return len(self.prior.global_values)

    @property
    def task_count(self) -> int:
        return len(self.prior.values_by_task)

    def observe(self, row: dict[str, Any]) -> dict[str, Any]:
        """Publish one completed observation and return its audit metadata."""

        update, _ = self.observe_with_runtime(row)
        return update

    def observe_with_runtime(self, row: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """Publish one observation and time only decision-relevant work."""

        start_ns = perf_counter_ns()
        payload = self._payload(row, source="online update")
        sample_id = payload["sample_id"]
        if sample_id in self._sample_ids:
            raise ValueError(f"sample_id already present in profile: {sample_id!r}")
        self._record_source_task(payload, source="online update")
        self._sample_ids.add(sample_id)

        tool_name = payload["tool_name"]
        task_id = payload["task_id"]
        latency_ms = payload["latency_ms"]
        group_keys = tuple(payload["group_keys"])
        prior = self.prior
        insort(prior.global_values, latency_ms)
        insort(prior.values_by_tool.setdefault(tool_name, []), latency_ms)
        insort(prior.values_by_task.setdefault(task_id, []), latency_ms)
        insort(
            prior.values_by_task_by_tool.setdefault(tool_name, {}).setdefault(
                task_id, []
            ),
            latency_ms,
        )
        if prior.values_by_group is not None:
            assert prior.values_by_task_by_group is not None
            for group_key in group_keys:
                insort(prior.values_by_group.setdefault(group_key, []), latency_ms)
                insort(
                    prior.values_by_task_by_group.setdefault(group_key, {}).setdefault(
                        task_id, []
                    ),
                    latency_ms,
                )
        runtime_ms = (perf_counter_ns() - start_ns) / 1_000_000.0

        self.version += 1
        digest = hashlib.sha256()
        digest.update(bytes.fromhex(self.state_hash))
        digest.update(_canonical_json(payload))
        self.state_hash = digest.hexdigest()
        return (
            {
                **payload,
                "model_version": self.version,
                "model_state_hash": self.state_hash,
                "profile_row_count": self.row_count,
                "profile_task_count": self.task_count,
            },
            runtime_ms,
        )

    def _payload(self, row: dict[str, Any], *, source: str) -> dict[str, Any]:
        sample_id = required_text(row, "sample_id", source=source)
        source_trace = required_text(row, "source_trace", source=source)
        task_id = required_text(row, "task_id", source=source)
        tool_name = required_text(row, "tool_name", source=source)
        latency_ms = required_nonnegative_float(row, "latency_ms", source=source)
        group_keys = (
            self._row_group_keys(row) if self._row_group_keys is not None else ()
        )
        return {
            "sample_id": sample_id,
            "source_trace": source_trace,
            "task_id": task_id,
            "tool_name": tool_name,
            "latency_ms": latency_ms,
            "group_keys": list(group_keys),
        }

    def _record_source_task(self, payload: Mapping[str, Any], *, source: str) -> None:
        source_trace = str(payload["source_trace"])
        task_id = str(payload["task_id"])
        existing = self._task_by_source.setdefault(source_trace, task_id)
        if existing != task_id:
            raise ValueError(
                f"{source}: source trace {source_trace!r} maps to multiple tasks: "
                f"{existing!r} and {task_id!r}"
            )


def benchmark_profile_updates(
    profile_rows: Iterable[dict[str, Any]],
    eval_rows: Iterable[dict[str, Any]],
    *,
    task_order: Sequence[str],
    command_field: str | None,
    max_prefix_depth: int,
    skip_leading_cd: bool,
) -> list[UpdateBenchmark]:
    """Measure the actual incremental update path once per stream observation."""

    row_group_keys = _row_group_keys(
        command_field=command_field,
        max_prefix_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
    )
    profile = EvolvingLatencyProfile(profile_rows, row_group_keys=row_group_keys)
    rows_by_task = _rows_by_task(eval_rows, task_order=task_order)
    timings: list[UpdateBenchmark] = []
    for task_id in task_order:
        for row in _rows_by_completion(rows_by_task[task_id]):
            _, runtime_ms = profile.observe_with_runtime(row)
            timings.append(
                UpdateBenchmark(
                    sample_id=str(row["sample_id"]),
                    task_id=task_id,
                    runtime_ms=runtime_ms,
                )
            )
    return timings


def evaluate_prequential_updates(
    eval_rows: Iterable[dict[str, Any]],
    *,
    profile_rows: Iterable[dict[str, Any]],
    task_order: Sequence[str],
    update_mode: UpdateMode,
    update_runtime_ms: Mapping[str, float],
    kv_costs_ms: Sequence[float],
    guard_ms: float,
    selected_guard_normalized: float | None,
    min_tool_history: int,
    min_profile_tasks: int,
    command_field: str | None,
    max_prefix_depth: int,
    skip_leading_cd: bool,
    restore_cost_fraction: float,
) -> dict[str, Any]:
    """Score one frozen or causally evolving profile over a task stream."""

    if update_mode not in _UPDATE_MODES:
        raise ValueError(f"unknown update mode: {update_mode!r}")
    if not math.isfinite(guard_ms) or guard_ms < 0.0:
        raise ValueError("guard_ms must be finite and non-negative")
    if not math.isfinite(restore_cost_fraction) or restore_cost_fraction < 0.0:
        raise ValueError("restore_cost_fraction must be finite and non-negative")
    costs = _positive_floats(kv_costs_ms, label="kv_costs_ms")
    row_group_keys = _row_group_keys(
        command_field=command_field,
        max_prefix_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
    )
    profile_list = list(profile_rows)
    eval_list = list(eval_rows)
    profile = EvolvingLatencyProfile(profile_list, row_group_keys=row_group_keys)
    validate_profile_eval_disjoint(eval_list, prior=profile.prior)
    rows_by_task = _rows_by_task(eval_list, task_order=task_order)
    expected_samples = {str(row["sample_id"]) for row in eval_list}
    if update_mode == "frozen":
        runtime_by_sample: dict[str, float] = {}
    else:
        runtime_by_sample = {
            sample_id: _finite_nonnegative(runtime, label=f"runtime {sample_id}")
            for sample_id, runtime in update_runtime_ms.items()
        }
        if set(runtime_by_sample) != expected_samples:
            raise ValueError(
                "update runtime panel differs from eval samples: "
                f"missing={sorted(expected_samples - set(runtime_by_sample))}, "
                f"unexpected={sorted(set(runtime_by_sample) - expected_samples)}"
            )

    decisions: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    readiness_eligible = 0
    readiness_ready = 0
    trigger_cache: dict[tuple[Any, ...], tuple[float, float]] = {}

    for task_position, task_id in enumerate(task_order):
        task_rows = rows_by_task[task_id]
        publication = (
            _publication_times(task_rows, runtime_by_sample)
            if update_mode == "call"
            else {}
        )
        if update_mode == "call":
            eligible, ready = _next_call_readiness(task_rows, publication)
            readiness_eligible += eligible
            readiness_ready += ready
        pending: list[dict[str, Any]] = []
        row_index = 0
        ordered = _rows_by_start(task_rows)
        while row_index < len(ordered):
            bucket_start = row_index
            start_ts = float(ordered[bucket_start]["tool_ts_start"])
            while (
                row_index < len(ordered)
                and float(ordered[row_index]["tool_ts_start"]) == start_ts
            ):
                row_index += 1
            bucket = ordered[bucket_start:row_index]

            if update_mode == "call":
                ready_rows = [
                    row
                    for row in pending
                    if publication[str(row["sample_id"])] <= start_ts
                ]
                pending = [
                    row
                    for row in pending
                    if publication[str(row["sample_id"])] > start_ts
                ]
                for row in sorted(
                    ready_rows,
                    key=lambda item: (
                        publication[str(item["sample_id"])],
                        str(item["sample_id"]),
                    ),
                ):
                    update = profile.observe(row)
                    updates.append(
                        {
                            **update,
                            "arm": update_mode,
                            "task_position": task_position,
                            "published_ts": publication[str(row["sample_id"])],
                            "update_runtime_ms": runtime_by_sample[
                                str(row["sample_id"])
                            ],
                        }
                    )

            snapshot_version = profile.version
            snapshot_hash = profile.state_hash
            for row in bucket:
                decisions.extend(
                    _score_row(
                        row,
                        profile=profile,
                        arm=update_mode,
                        task_position=task_position,
                        model_version=snapshot_version,
                        model_state_hash=snapshot_hash,
                        kv_costs=costs,
                        guard_ms=guard_ms,
                        selected_guard_normalized=selected_guard_normalized,
                        min_tool_history=min_tool_history,
                        min_profile_tasks=min_profile_tasks,
                        row_group_keys=row_group_keys,
                        restore_cost_fraction=restore_cost_fraction,
                        trigger_cache=trigger_cache,
                    )
                )
            if update_mode == "call":
                pending.extend(bucket)

        if update_mode == "call":
            for row in sorted(
                pending,
                key=lambda item: (
                    publication[str(item["sample_id"])],
                    str(item["sample_id"]),
                ),
            ):
                update = profile.observe(row)
                updates.append(
                    {
                        **update,
                        "arm": update_mode,
                        "task_position": task_position,
                        "published_ts": publication[str(row["sample_id"])],
                        "update_runtime_ms": runtime_by_sample[str(row["sample_id"])],
                    }
                )
        elif update_mode == "task":
            for row in _rows_by_completion(task_rows):
                update = profile.observe(row)
                updates.append(
                    {
                        **update,
                        "arm": update_mode,
                        "task_position": task_position,
                        "published_ts": None,
                        "update_runtime_ms": runtime_by_sample[str(row["sample_id"])],
                    }
                )

    return {
        "arm": update_mode,
        "initial_profile_row_count": len(profile_list),
        "initial_profile_task_count": len(
            {str(row["task_id"]) for row in profile_list}
        ),
        "final_profile_row_count": profile.row_count,
        "final_profile_task_count": profile.task_count,
        "final_model_version": profile.version,
        "final_model_state_hash": profile.state_hash,
        "call_update_readiness": {
            "eligible_update_count": readiness_eligible,
            "ready_before_next_eligible_call_count": readiness_ready,
            "fraction": (
                readiness_ready / readiness_eligible if readiness_eligible else None
            ),
        },
        "decisions": decisions,
        "updates": updates,
    }


def _score_row(
    row: dict[str, Any],
    *,
    profile: EvolvingLatencyProfile,
    arm: UpdateMode,
    task_position: int,
    model_version: int,
    model_state_hash: str,
    kv_costs: Sequence[float],
    guard_ms: float,
    selected_guard_normalized: float | None,
    min_tool_history: int,
    min_profile_tasks: int,
    row_group_keys: Callable[[dict[str, Any]], tuple[str, ...]] | None,
    restore_cost_fraction: float,
    trigger_cache: dict[tuple[Any, ...], tuple[float, float]],
) -> list[dict[str, Any]]:
    source = f"prequential sample {row.get('sample_id')!r}"
    sample_id = required_text(row, "sample_id", source=source)
    task_id = required_text(row, "task_id", source=source)
    tool_name = required_text(row, "tool_name", source=source)
    latency_ms = required_nonnegative_float(row, "latency_ms", source=source)
    group_keys = row_group_keys(row) if row_group_keys is not None else ()
    score_start_ns = perf_counter_ns()
    hierarchy = latency_prior_hierarchy(
        profile.prior,
        tool_name,
        group_keys,
        min_tool_history=min_tool_history,
        min_profile_tasks=min_profile_tasks,
    )
    robust_node, robust_parent = robust_prior_nodes(hierarchy)
    output: list[dict[str, Any]] = []
    for cost_ms in kv_costs:
        threshold_ms = cost_ms + guard_ms
        restore_cost_ms = restore_cost_fraction * cost_ms
        cache_key = (
            id(robust_node.values),
            len(robust_node.values),
            id(robust_parent.values) if robust_parent is not None else None,
            len(robust_parent.values) if robust_parent is not None else None,
            threshold_ms,
            cost_ms,
            restore_cost_ms,
        )
        cached = trigger_cache.get(cache_key)
        if cached is None:
            stats = robust_utility_trigger_stats(
                robust_node,
                parent=robust_parent,
                threshold_ms=threshold_ms,
                kv_cost_ms=cost_ms,
                restore_cost_ms=restore_cost_ms,
            )
            cached = (stats.trigger_ms, stats.normalized_advantage)
            trigger_cache[cache_key] = cached
        candidate_ms, margin = cached
        trigger_ms = (
            candidate_ms
            if selected_guard_normalized is not None
            and candidate_ms < threshold_ms
            and margin > selected_guard_normalized
            else threshold_ms
        )
        output.append(
            {
                "arm": arm,
                "sample_id": sample_id,
                "task_id": task_id,
                "source_trace": str(row["source_trace"]),
                "task_position": task_position,
                "tool_name": tool_name,
                "tool_ts_start": float(row["tool_ts_start"]),
                "tool_ts_end": float(row["tool_ts_end"]),
                "latency_ms": latency_ms,
                "kv_cost_ms": cost_ms,
                "threshold_ms": threshold_ms,
                "restore_cost_ms": restore_cost_ms,
                "candidate_trigger_ms": candidate_ms,
                "candidate_margin_normalized": margin,
                "trigger_ms": trigger_ms,
                "prior_source": robust_node.source,
                "prior_group_key": robust_node.group_key,
                "prior_task_count": len(robust_node.values_by_task),
                "model_version": model_version,
                "model_state_hash": model_state_hash,
                "profile_row_count": profile.row_count,
                "profile_task_count": profile.task_count,
            }
        )
    elapsed_ms = (perf_counter_ns() - score_start_ns) / 1_000_000.0
    for decision in output:
        decision["score_panel_runtime_ms"] = elapsed_ms
    return output


def _publication_times(
    rows: Sequence[dict[str, Any]],
    runtime_by_sample: Mapping[str, float],
) -> dict[str, float]:
    """Single-worker asynchronous publication times in trace-clock seconds."""

    publication: dict[str, float] = {}
    worker_available = -math.inf
    for row in _rows_by_completion(rows):
        sample_id = str(row["sample_id"])
        tool_end = float(row["tool_ts_end"])
        start = max(tool_end, worker_available)
        finish = start + runtime_by_sample[sample_id] / 1000.0
        publication[sample_id] = finish
        worker_available = finish
    return publication


def _next_call_readiness(
    rows: Sequence[dict[str, Any]],
    publication: Mapping[str, float],
) -> tuple[int, int]:
    starts = sorted({float(row["tool_ts_start"]) for row in rows})
    eligible = 0
    ready = 0
    for row in rows:
        own_start = float(row["tool_ts_start"])
        end = float(row["tool_ts_end"])
        index = max(bisect_left(starts, end), bisect_left(starts, own_start) + 1)
        if index >= len(starts):
            continue
        eligible += 1
        if publication[str(row["sample_id"])] <= starts[index]:
            ready += 1
    return eligible, ready


def _rows_by_task(
    rows: Iterable[dict[str, Any]], *, task_order: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    if len(task_order) != len(set(task_order)):
        raise ValueError("task_order contains duplicates")
    rows_by_task = {task_id: [] for task_id in task_order}
    sample_ids: set[str] = set()
    for index, row in enumerate(rows):
        source = f"eval row {index}"
        task_id = required_text(row, "task_id", source=source)
        sample_id = required_text(row, "sample_id", source=source)
        start = required_nonnegative_float(row, "tool_ts_start", source=source)
        end = required_nonnegative_float(row, "tool_ts_end", source=source)
        if end < start:
            raise ValueError(f"{source}: tool_ts_end < tool_ts_start")
        if task_id not in rows_by_task:
            raise ValueError(f"eval row has undeclared task_id: {task_id!r}")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate eval sample_id: {sample_id!r}")
        sample_ids.add(sample_id)
        rows_by_task[task_id].append(row)
    empty = [task_id for task_id, task_rows in rows_by_task.items() if not task_rows]
    if empty:
        raise ValueError(f"task_order contains tasks without latency rows: {empty}")
    return rows_by_task


def _rows_by_start(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["tool_ts_start"]),
            str(row["sample_id"]),
        ),
    )


def _rows_by_completion(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["tool_ts_end"]),
            float(row["tool_ts_start"]),
            str(row["sample_id"]),
        ),
    )


def _row_group_keys(
    *,
    command_field: str | None,
    max_prefix_depth: int,
    skip_leading_cd: bool,
) -> Callable[[dict[str, Any]], tuple[str, ...]] | None:
    if max_prefix_depth < 1:
        raise ValueError("max_prefix_depth must be positive")
    return (
        make_row_command_prefix_keys(
            command_field,
            max_depth=max_prefix_depth,
            skip_leading_cd=skip_leading_cd,
        )
        if command_field is not None
        else None
    )


def _positive_floats(values: Sequence[float], *, label: str) -> list[float]:
    output = sorted({float(value) for value in values})
    if not output or any(not math.isfinite(value) or value <= 0.0 for value in output):
        raise ValueError(f"{label} must contain finite positive values")
    return output


def _finite_nonnegative(value: float, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = [
    "EvolvingLatencyProfile",
    "UpdateBenchmark",
    "benchmark_profile_updates",
    "evaluate_prequential_updates",
]
