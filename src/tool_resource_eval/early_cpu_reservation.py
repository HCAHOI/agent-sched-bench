"""Model causal command-level CPU feedback from exec timelines."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

from trace_collect.resource_timeline import valid_resource_timeline


THROUGHPUT_CPU_PAGES = (2, 4, 8)
CONTROL_PAGE = 8
SAMPLE_INTERVAL_S = 0.5
SAMPLE_AVAILABILITY_PAD_S = 0.05
CPU_UPDATE_P95_S = 0.09132007875
CPU_UPDATE_DELAY_S = SAMPLE_AVAILABILITY_PAD_S + CPU_UPDATE_P95_S


def _number(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _page(cpu_rate: float) -> int:
    return next(page for page in THROUGHPUT_CPU_PAGES if cpu_rate <= page + 1e-12)


def _modeled_interval(
    dt_s: float,
    cpu_core_s: float,
    requests: list[tuple[float, int]],
) -> tuple[float, float, bool]:
    """Charge uniformly partitioned interval work under piecewise CPU pages."""

    duration_s = reserved_core_s = 0.0
    rate = cpu_core_s / dt_s
    throttled = False
    for segment_s, request in requests:
        work = rate * segment_s
        modeled_s = max(segment_s, work / request)
        duration_s += modeled_s
        reserved_core_s += request * modeled_s
        throttled |= rate > request + 1e-12
    return duration_s, reserved_core_s, throttled


def feedback_action_row(action: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Model fixed, probe-then-two, and one-interval CPU feedback actions."""

    data = action.get("data")
    start = _number(action.get("ts_start"))
    end = _number(action.get("ts_end"))
    if (
        not isinstance(data, dict)
        or data.get("tool_name") != "exec"
        or start is None
        or end is None
        or end <= start
    ):
        raise ValueError("feedback row requires a timed exec action")
    duration = end - start
    timeline = valid_resource_timeline(data.get("resource_timeline"))
    interval = _number(timeline.get("sample_interval_s")) if timeline else None
    samples = timeline.get("samples") if timeline else None
    if (
        interval is None
        or not math.isclose(interval, SAMPLE_INTERVAL_S, abs_tol=1e-9)
        or not isinstance(samples, list)
    ):
        fallback = {
            "service_s": duration,
            "reserved_cpu_core_s": CONTROL_PAGE * duration,
            "added_service_s": 0.0,
            "request_counts": {str(CONTROL_PAGE): 1},
            "throttled_samples": 0,
        }
        return {
            "tool_call_id": str(data.get("tool_call_id") or action.get("action_id") or ""),
            "recorded_duration_s": duration,
            "timeline_cpu_core_s": None,
            "raw_timeline_cpu_core_s": None,
            "clipped_samples": 0,
            "clipped_cpu_core_s": 0.0,
            "eligible": False,
            "arms": {name: dict(fallback) for name in ("fixed8", "probe_then_two", "feedback")},
        }, "no_valid_timeline"

    parsed: list[tuple[float, float]] = []
    raw_cpu_work = clipped_cpu_work = 0.0
    clipped_samples = 0
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("resource timeline contains a non-object sample")
        dt = _number(sample.get("dt_s"))
        cpu = _number(sample.get("cpu_core_s"))
        quota = _number(sample.get("cpu_quota_cores"))
        if dt is None or dt <= 0.0 or cpu is None or quota is None or quota <= 0.0:
            raise ValueError("resource timeline sample lacks CPU work or duration")
        if not math.isclose(quota, CONTROL_PAGE, abs_tol=1e-9):
            raise ValueError("feedback source timeline was not collected at eight cores")
        raw_cpu_work += cpu
        physical_cpu = min(cpu, quota * dt)
        clipped_cpu_work += physical_cpu
        clipped_samples += physical_cpu < cpu
        parsed.append((dt, physical_cpu))
    if not parsed:
        raise ValueError("valid resource timeline has no samples")
    sample_wall = sum(dt for dt, _ in parsed)
    residual = duration - sample_wall
    if residual < -SAMPLE_AVAILABILITY_PAD_S or residual > SAMPLE_AVAILABILITY_PAD_S:
        raise ValueError("resource timeline does not cover the exec duration")
    residual = max(0.0, residual)
    first_full = next(
        (index for index, (dt, _) in enumerate(parsed) if dt >= SAMPLE_INTERVAL_S),
        None,
    )
    if first_full is None:
        fallback = {
            "service_s": duration,
            "reserved_cpu_core_s": CONTROL_PAGE * duration,
            "added_service_s": 0.0,
            "request_counts": {str(CONTROL_PAGE): len(parsed)},
            "throttled_samples": 0,
        }
        return {
            "tool_call_id": str(data.get("tool_call_id") or action.get("action_id") or ""),
            "recorded_duration_s": duration,
            "timeline_cpu_core_s": sum(cpu for _, cpu in parsed),
            "raw_timeline_cpu_core_s": raw_cpu_work,
            "clipped_samples": clipped_samples,
            "clipped_cpu_core_s": raw_cpu_work - clipped_cpu_work,
            "eligible": False,
            "arms": {name: dict(fallback) for name in ("fixed8", "probe_then_two", "feedback")},
        }, "no_full_decision_sample"

    arms: dict[str, dict[str, Any]] = {}
    for arm in ("fixed8", "probe_then_two", "feedback"):
        current = CONTROL_PAGE
        pending: int | None = None
        delay = 0.0
        service = reserved = 0.0
        throttled_samples = 0
        counts: Counter[int] = Counter()
        probe_has_acted = False
        for index, (dt, cpu) in enumerate(parsed):
            requests: list[tuple[float, int]] = []
            delayed = min(dt, delay)
            if delayed:
                requests.append((delayed, current))
                delay -= delayed
            remaining = dt - delayed
            if delay <= 1e-12 and pending is not None:
                current = pending
                pending = None
            if remaining:
                requests.append((remaining, current))
            interval_service, interval_reserved, throttled = _modeled_interval(
                dt, cpu, requests
            )
            service += interval_service
            reserved += interval_reserved
            throttled_samples += throttled
            for _, request in requests:
                counts[request] += 1

            if index < first_full or dt < SAMPLE_INTERVAL_S or arm == "fixed8":
                continue
            if arm == "probe_then_two":
                if probe_has_acted:
                    continue
                target = 2
                probe_has_acted = True
            else:
                target = CONTROL_PAGE if throttled else _page(cpu / dt)
            pending = target
            delay = CPU_UPDATE_DELAY_S

        if residual:
            service += residual
            reserved += current * residual
            counts[current] += 1
        arms[arm] = {
            "service_s": service,
            "reserved_cpu_core_s": reserved,
            "added_service_s": service - duration,
            "request_counts": {str(key): value for key, value in sorted(counts.items())},
            "throttled_samples": throttled_samples,
        }
    return {
        "tool_call_id": str(data.get("tool_call_id") or action.get("action_id") or ""),
        "recorded_duration_s": duration,
        "timeline_cpu_core_s": sum(cpu for _, cpu in parsed),
        "raw_timeline_cpu_core_s": raw_cpu_work,
        "clipped_samples": clipped_samples,
        "clipped_cpu_core_s": raw_cpu_work - clipped_cpu_work,
        "eligible": True,
        "arms": arms,
    }, "eligible"
