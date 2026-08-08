"""Measure the hindsight CPU-reservation ceiling from exec timelines."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from trace_collect.resource_timeline import valid_resource_timeline
from trace_collect.trace_data import TraceData


CPU_PAGES = (1, 2, 4, 8)
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


def action_row(
    action: dict[str, Any],
    *,
    task_id: str,
    trace_path: Path,
) -> tuple[dict[str, Any] | None, str]:
    """Return one eligible oracle row and its exclusion reason."""

    data = action.get("data")
    if not isinstance(data, dict) or data.get("tool_name") != "exec":
        return None, "not_exec"
    timeline = valid_resource_timeline(data.get("resource_timeline"))
    if timeline is None:
        return None, "no_valid_timeline"
    interval = _number(timeline.get("sample_interval_s"))
    if interval is None or not math.isclose(interval, SAMPLE_INTERVAL_S, abs_tol=1e-9):
        return None, "wrong_sample_interval"
    start = _number(action.get("ts_start"))
    end = _number(action.get("ts_end"))
    if start is None or end is None or end <= start:
        return None, "invalid_action_time"
    duration = end - start

    samples = [sample for sample in timeline["samples"] if isinstance(sample, dict)]
    last_offset = _number(samples[-1].get("offset_s")) if samples else None
    if last_offset is None or not 0 <= duration - last_offset <= SAMPLE_AVAILABILITY_PAD_S:
        return None, "timeline_overhead_exceeds_pad"

    decision_sample = next(
        (
            sample
            for sample in samples
            if (_number(sample.get("dt_s")) or 0.0) >= SAMPLE_INTERVAL_S
        ),
        None,
    )
    if decision_sample is None:
        return None, "no_full_decision_sample"
    decision_dt = _number(decision_sample.get("dt_s"))
    decision_offset = _number(decision_sample.get("offset_s"))
    decision_cpu = _number(decision_sample.get("cpu_core_s"))
    decision_quota = _number(decision_sample.get("cpu_quota_cores"))
    if (
        decision_dt is None
        or decision_offset is None
        or decision_cpu is None
        or decision_quota is None
        or decision_quota <= 0
    ):
        return None, "missing_decision_cpu"
    effective_offset = (
        decision_offset + SAMPLE_AVAILABILITY_PAD_S + CPU_UPDATE_P95_S
    )
    if duration - effective_offset < SAMPLE_INTERVAL_S:
        return None, "no_post_actuation_interval"

    future_rates: list[float] = []
    for sample in samples:
        dt = _number(sample.get("dt_s"))
        offset = _number(sample.get("offset_s"))
        cpu = _number(sample.get("cpu_core_s"))
        quota = _number(sample.get("cpu_quota_cores"))
        if (
            dt is None
            or offset is None
            or dt < SAMPLE_INTERVAL_S
            or offset - dt < effective_offset - 1e-12
        ):
            continue
        if dt <= 0 or cpu is None or quota is None or quota <= 0:
            return None, "missing_future_cpu"
        future_rates.append(min(cpu / dt, quota, float(CONTROL_PAGE)))
    if not future_rates:
        return None, "no_future_cpu"

    future_peak = max(future_rates)
    oracle_page = next(page for page in CPU_PAGES if page + 1e-12 >= future_peak)
    remaining_s = duration - effective_offset
    control_core_s = CONTROL_PAGE * remaining_s
    oracle_core_s = oracle_page * remaining_s
    false_shrink = any(rate > oracle_page + 1e-12 for rate in future_rates)
    data_id = str(data.get("tool_call_id") or action.get("action_id") or "")
    return {
        "task_id": task_id,
        "tool_call_id": data_id,
        "trace": str(trace_path),
        "effective_offset_s": effective_offset,
        "remaining_s": remaining_s,
        "prefix_cpu_rate_cores": decision_cpu / decision_dt,
        "future_peak_cpu_rate_cores": future_peak,
        "oracle_page_cores": oracle_page,
        "control_reserved_core_s": control_core_s,
        "oracle_reserved_core_s": oracle_core_s,
        "saved_reserved_core_s": control_core_s - oracle_core_s,
        "false_shrink": false_shrink,
    }, "eligible"


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


def model_cpu_page_policy(
    action: dict[str, Any], *, initial_page: int, feedback: bool
) -> tuple[dict[str, Any], str]:
    """Model a BeginCall CPU page followed by the existing interval controller."""

    if initial_page not in THROUGHPUT_CPU_PAGES:
        raise ValueError("initial CPU page must be 2, 4, or 8 cores")
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
        raise ValueError("CPU page policy requires a timed exec action")
    duration = end - start
    timeline = valid_resource_timeline(data.get("resource_timeline"))
    interval = _number(timeline.get("sample_interval_s")) if timeline else None
    samples = timeline.get("samples") if timeline else None
    if (
        interval is None
        or not math.isclose(interval, SAMPLE_INTERVAL_S, abs_tol=1e-9)
        or not isinstance(samples, list)
    ):
        return {
            "service_s": duration,
            "reserved_cpu_core_s": CONTROL_PAGE * duration,
            "added_service_s": 0.0,
            "request_counts": {str(CONTROL_PAGE): 1},
            "throttled_samples": 0,
            "feedback_updates": 0,
        }, "no_valid_timeline"

    parsed: list[tuple[float, float]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("resource timeline contains a non-object sample")
        dt = _number(sample.get("dt_s"))
        cpu = _number(sample.get("cpu_core_s"))
        quota = _number(sample.get("cpu_quota_cores"))
        if dt is None or dt <= 0.0 or cpu is None or quota is None or quota <= 0.0:
            raise ValueError("resource timeline sample lacks CPU work or duration")
        if not math.isclose(quota, CONTROL_PAGE, abs_tol=1e-9):
            raise ValueError("CPU page source timeline was not collected at eight cores")
        parsed.append((dt, min(cpu, quota * dt)))
    if not parsed:
        raise ValueError("valid resource timeline has no samples")
    residual = duration - sum(dt for dt, _cpu in parsed)
    if residual < -SAMPLE_AVAILABILITY_PAD_S or residual > SAMPLE_AVAILABILITY_PAD_S:
        raise ValueError("resource timeline does not cover the exec duration")
    residual = max(0.0, residual)
    first_full = next(
        (index for index, (dt, _cpu) in enumerate(parsed) if dt >= SAMPLE_INTERVAL_S),
        None,
    )

    profiles = [(dt, cpu / dt) for dt, cpu in parsed]
    if residual:
        profiles.append((residual, 0.0))
    feedback_enabled = feedback and first_full is not None
    current = initial_page
    pending: int | None = None
    pending_at = math.inf
    next_observation = SAMPLE_INTERVAL_S if feedback_enabled else math.inf
    profile_index = 0
    profile_remaining = profiles[0][0]
    wall = reserved = observed_cpu = 0.0
    observed_throttling = False
    throttled_samples = feedback_updates = 0
    counts: Counter[int] = Counter()
    while profile_index < len(profiles):
        rate = profiles[profile_index][1]
        progress_rate = 1.0 if rate <= current else current / rate
        source_done_in = profile_remaining / progress_rate
        step = min(
            source_done_in,
            pending_at - wall,
            next_observation - wall,
        )
        if step < -1e-12:
            raise ValueError("CPU feedback event order moved backwards")
        step = max(0.0, step)
        source_progress = step * progress_rate
        profile_remaining -= source_progress
        cpu_rate = min(rate, float(current))
        observed_cpu += cpu_rate * step
        observed_throttling |= rate > current + 1e-12
        reserved += current * step
        counts[current] += step > 0.0
        wall += step

        if profile_remaining <= 1e-12:
            profile_index += 1
            if profile_index < len(profiles):
                profile_remaining = profiles[profile_index][0]
        if pending is not None and pending_at <= wall + 1e-12:
            current = pending
            pending = None
            pending_at = math.inf
        if (
            feedback_enabled
            and profile_index < len(profiles)
            and next_observation <= wall + 1e-12
        ):
            pending = (
                CONTROL_PAGE
                if observed_throttling
                else _page(observed_cpu / SAMPLE_INTERVAL_S)
            )
            pending_at = wall + CPU_UPDATE_DELAY_S
            next_observation += SAMPLE_INTERVAL_S
            throttled_samples += observed_throttling
            feedback_updates += 1
            observed_cpu = 0.0
            observed_throttling = False
    throttled_samples += observed_throttling
    return {
        "service_s": wall,
        "reserved_cpu_core_s": reserved,
        "added_service_s": wall - duration,
        "request_counts": {str(key): value for key, value in sorted(counts.items())},
        "throttled_samples": throttled_samples,
        "feedback_updates": feedback_updates,
    }, "eligible" if first_full is not None else "no_full_decision_sample"


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


def _selected_traces(root: Path) -> list[tuple[str, Path]]:
    results_path = root / "results.jsonl"
    selected: list[tuple[str, Path]] = []
    seen_tasks: set[str] = set()
    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            result = json.loads(line)
            if result.get("success") is not True:
                raise ValueError("Phase 1 requires every selected SQLGlot task to succeed")
            task_id = str(result.get("instance_id") or "")
            raw_path = result.get("trace_file")
            if not task_id or not isinstance(raw_path, str):
                raise ValueError("results.jsonl row lacks instance_id or trace_file")
            trace_path = Path(raw_path)
            if not trace_path.is_absolute():
                trace_path = Path.cwd() / trace_path
            trace_path = trace_path.resolve()
            if task_id in seen_tasks:
                raise ValueError(f"duplicate selected task {task_id}")
            if not trace_path.is_file():
                raise FileNotFoundError(trace_path)
            seen_tasks.add(task_id)
            selected.append((task_id, trace_path))
    if len(selected) != 100:
        raise ValueError(f"Phase 1 requires 100 selected traces, found {len(selected)}")
    return selected


def evaluate(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    exec_actions = timeline_actions = 0
    selected = _selected_traces(root)
    for task_id, trace_path in selected:
        trace = TraceData.load(trace_path)
        for action in trace.actions:
            data = action.get("data")
            if not isinstance(data, dict) or data.get("tool_name") != "exec":
                continue
            exec_actions += 1
            if valid_resource_timeline(data.get("resource_timeline")) is not None:
                timeline_actions += 1
            row, reason = action_row(
                action,
                task_id=task_id,
                trace_path=trace_path,
            )
            if row is None:
                exclusions[reason] += 1
            else:
                rows.append(row)

    action_counts = Counter(int(row["oracle_page_cores"]) for row in rows)
    changed_rows = [row for row in rows if row["oracle_page_cores"] < CONTROL_PAGE]
    control_core_s = sum(float(row["control_reserved_core_s"]) for row in rows)
    oracle_core_s = sum(float(row["oracle_reserved_core_s"]) for row in rows)
    saved_core_s = control_core_s - oracle_core_s
    saving_fraction = saved_core_s / control_core_s if control_core_s else 0.0
    false_shrinks = sum(bool(row["false_shrink"]) for row in rows)
    changed_tasks = len({str(row["task_id"]) for row in changed_rows})
    go = (
        len(changed_rows) >= 20
        and changed_tasks >= 10
        and false_shrinks == 0
        and saving_fraction >= 0.10
    )
    result = {
        "status": "development_exposed_phase1_oracle",
        "question": "Can perfect post-prefix information reduce EAR CPU reservations without lowering any later observed interval demand?",
        "protocol": {
            "trace_root": str(root.resolve()),
            "consumer": "EAR shared CPU lease pool",
            "ear_policy_source": {
                "commit": "244d1f58ed5ea88147d2601a45f2889ce243da64",
                "path": "configs/policies/docker_elastic_mixed_burst.yaml",
            },
            "pages_cores": list(CPU_PAGES),
            "control_page_cores": CONTROL_PAGE,
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "sample_availability_pad_s": SAMPLE_AVAILABILITY_PAD_S,
            "cpu_update_p95_s": CPU_UPDATE_P95_S,
            "effective_time": "first full sample endpoint + pad + update p95",
            "oracle": "smallest page >= maximum later interval CPU rate, capped by recorded quota",
            "primary": "post-decision reserved CPU core-seconds",
            "go_criterion": "zero modeled false shrink, >=20 changed commands across >=10 tasks, and >=10% reservation reduction",
        },
        "data": {
            "selected_traces": len(selected),
            "exec_actions": exec_actions,
            "valid_timeline_actions": timeline_actions,
            "eligible_actions": len(rows),
            "eligible_tasks": len({str(row["task_id"]) for row in rows}),
            "exclusions": dict(sorted(exclusions.items())),
        },
        "actions": {
            "counts_by_page": {
                str(page): action_counts.get(page, 0) for page in CPU_PAGES
            },
            "changed_commands": len(changed_rows),
            "changed_tasks": changed_tasks,
            "modeled_false_shrinks": false_shrinks,
        },
        "reservation": {
            "control_core_s": control_core_s,
            "oracle_core_s": oracle_core_s,
            "saved_core_s": saved_core_s,
            "saving_fraction": saving_fraction,
            "saving_percent": 100.0 * saving_fraction,
        },
        "decision": "GO to Phase 2 predictor" if go else "STOP before predictor",
        "gate_pass": go,
        "evidence_boundary": [
            "The oracle reads future CPU samples and is only an action-space upper bound.",
            "Interval-average demand can miss sub-0.5-second bursts.",
            "Reserved core-seconds are meaningful for the EAR lease consumer but do not prove lower latency or higher throughput.",
            "SQLGlot100 and the EAR action pages are development-exposed.",
        ],
    }
    return result, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=Path(
            "traces/swe-rebench/gpt-5.6-sol/"
            "sqlglot-100-c2-fast-requested-ebpf-a0419d9-20260803"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/results/early-execution-resource-control-20260804/"
            "phase1-cpu-reservation-oracle.json"
        ),
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=Path(
            "analysis/results/early-execution-resource-control-20260804/"
            "phase1-cpu-reservation-oracle-rows.jsonl"
        ),
    )
    args = parser.parse_args()
    result, rows = evaluate(args.trace_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.rows_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
