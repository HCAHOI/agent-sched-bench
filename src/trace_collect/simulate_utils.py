from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trace_collect.attempt_pipeline import sanitize_path_segment
from trace_collect.simulate_types import LoadedTraceSession, SimulateError, SleepDrift

logger = logging.getLogger(__name__)
_DEFAULT_PREP_CONCURRENCY = 20


def _source_model(loaded: LoadedTraceSession) -> str:
    summary = loaded.summary or {}
    for key in ("model", "source_model"):
        summary_model = summary.get(key)
        if summary_model:
            return str(summary_model)
    metadata = loaded.metadata or {}
    for key in ("model", "source_model"):
        metadata_model = metadata.get(key)
        if metadata_model:
            return str(metadata_model)
    return "unknown"


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _exception_payload(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _iteration_count(actions: list[dict[str, Any]]) -> int:
    return len({int(action.get("iteration", 0)) for action in actions})


def _sanitize_run_label(value: str) -> str:
    return sanitize_path_segment(value).replace(" ", "-")


def _coerce_timestamp(
    value: Any,
    *,
    field: str,
    source_trace: Path,
    action_id: str,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SimulateError(
            f"{source_trace} action {action_id!r} is missing a numeric {field}"
        ) from exc


def _coerce_action_bounds(
    action: dict[str, Any],
    *,
    source_trace: Path,
) -> tuple[float, float]:
    action_id = str(action.get("action_id", ""))
    ts_start = _coerce_timestamp(
        action.get("ts_start"),
        field="ts_start",
        source_trace=source_trace,
        action_id=action_id,
    )
    ts_end = _coerce_timestamp(
        action.get("ts_end"),
        field="ts_end",
        source_trace=source_trace,
        action_id=action_id,
    )
    return ts_start, ts_end


def _resolve_prep_concurrency(requested: int, num_sessions: int) -> int:
    """Resolve the system-wide concurrent container preparation limit."""
    if requested < 0:
        raise ValueError("prep_concurrency must be >= 0")
    if num_sessions < 1:
        raise ValueError("num_sessions must be >= 1")
    return min(requested or _DEFAULT_PREP_CONCURRENCY, num_sessions)


def _sleep_drift_metrics(
    *,
    source_gap: SleepDrift | None,
    action_sleep: SleepDrift | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if source_gap is not None:
        payload["source_gap_sleep"] = source_gap.to_dict()
    if action_sleep is not None:
        payload["action_sleep"] = action_sleep.to_dict()
    return payload


def _summarize_sleep_drifts(drifts: list[SleepDrift]) -> dict[str, Any]:
    if not drifts:
        return {
            "sample_count": 0,
            "expected_total_s": 0.0,
            "actual_total_s": 0.0,
            "drift_s": {"min": 0.0, "max": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0},
            "by_phase": {},
        }
    drift_values = [drift.drift_s for drift in drifts]
    by_phase: dict[str, list[SleepDrift]] = {}
    for drift in drifts:
        by_phase.setdefault(drift.phase, []).append(drift)
    return {
        "sample_count": len(drifts),
        "expected_total_s": round(sum(drift.expected_s for drift in drifts), 6),
        "actual_total_s": round(sum(drift.actual_s for drift in drifts), 6),
        "drift_s": _summarize_float_values(drift_values),
        "by_phase": {
            phase: {
                "sample_count": len(items),
                "expected_total_s": round(sum(item.expected_s for item in items), 6),
                "actual_total_s": round(sum(item.actual_s for item in items), 6),
                "drift_s": _summarize_float_values([item.drift_s for item in items]),
            }
            for phase, items in sorted(by_phase.items())
        },
    }


def _summarize_float_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0}
    sorted_values = sorted(values)
    return {
        "min": round(sorted_values[0], 6),
        "max": round(sorted_values[-1], 6),
        "avg": round(sum(sorted_values) / len(sorted_values), 6),
        "p50": round(_nearest_rank_percentile(sorted_values, 50), 6),
        "p95": round(_nearest_rank_percentile(sorted_values, 95), 6),
    }


def _nearest_rank_percentile(sorted_values: list[float], percentile: int) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    index = max(0, min(len(sorted_values) - 1, (percentile * len(sorted_values) + 99) // 100 - 1))
    return sorted_values[index]


def _resolve_docker_image(loaded: LoadedTraceSession) -> str | None:
    """Resolve an explicit, task, or recorded benchmark container image."""
    metadata = loaded.metadata or {}
    explicit = (
        loaded.docker_image_override
        or loaded.task.get("image_name")
        or loaded.task.get("docker_image")
    )
    if explicit:
        return str(explicit)
    return None


def _execution_environment(loaded: LoadedTraceSession) -> str:
    metadata = loaded.metadata or {}
    value = metadata.get("execution_environment")
    if value is None or value == "":
        # Backward compat for legacy traces that predate the
        # execution_environment field: host_controller agents always ran on
        # the host, so infer "host" from agent_runtime_mode before falling
        # back to the container default.
        if metadata.get("agent_runtime_mode") == "host_controller":
            logger.warning(
                "%s has no execution_environment metadata; inferring host "
                "from agent_runtime_mode=host_controller",
                loaded.source_trace,
            )
            return "host"
        logger.warning(
            "%s has no execution_environment metadata; assuming container",
            loaded.source_trace,
        )
        return "container"
    return str(value)


def _is_host_mode(loaded: LoadedTraceSession) -> bool:
    return _execution_environment(loaded) == "host"


def _is_terminal_bench_registry_task(loaded: LoadedTraceSession) -> bool:
    return loaded.task.get("task_source_kind") == "terminal_bench_registry"


def _requires_task_container(loaded: LoadedTraceSession) -> bool:
    return not _is_host_mode(loaded) or _is_terminal_bench_registry_task(loaded)
