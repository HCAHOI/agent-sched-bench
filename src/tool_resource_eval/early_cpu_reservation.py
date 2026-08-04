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
CONTROL_PAGE = 8
SAMPLE_INTERVAL_S = 0.5
SAMPLE_AVAILABILITY_PAD_S = 0.05
CPU_UPDATE_P95_S = 0.09132007875


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
