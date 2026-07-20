from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scripts.evaluate_utility_clock_policy import main as utility_clock_main
from trace_collect.tool_latency_profiled import LatencyPriorNode
from trace_collect.tool_latency_utility_clock import (
    evaluate_utility_clock_policy,
    robust_utility_trigger_ms,
    robust_utility_trigger_stats,
)


def test_robust_clock_can_fire_early_when_long_calls_are_a_minority() -> None:
    node = _node(
        {
            "task-a": [80.0, 80.0, 150.0],
            "task-b": [80.0, 80.0, 150.0],
        }
    )

    trigger = robust_utility_trigger_ms(
        node,
        parent=None,
        threshold_ms=100.0,
        kv_cost_ms=100.0,
    )

    # Each task has only one long call in three, so p(long)=1/3. Acting at
    # k=0 still has +20ms mean utility per call and every LOTO fit prefers it
    # to waiting until the 100ms deadline.
    assert trigger == 0.0


def test_robust_clock_reports_weakest_normalized_advantage() -> None:
    node = _node(
        {
            "task-a": [80.0, 80.0, 150.0],
            "task-b": [80.0, 80.0, 150.0],
        }
    )

    stats = robust_utility_trigger_stats(
        node,
        parent=None,
        threshold_ms=100.0,
        kv_cost_ms=100.0,
    )

    assert stats.trigger_ms == robust_utility_trigger_ms(
        node,
        parent=None,
        threshold_ms=100.0,
        kv_cost_ms=100.0,
    )
    assert stats.normalized_advantage == pytest.approx(1.0 / 15.0)


def test_robust_clock_waits_when_one_loto_task_model_vetoes_early_action() -> None:
    node = _node({"short-task": [10.0], "long-task": [150.0]})

    trigger = robust_utility_trigger_ms(
        node,
        parent=None,
        threshold_ms=100.0,
        kv_cost_ms=100.0,
    )

    assert trigger == 100.0


def test_robust_clock_single_task_evidence_cannot_fire_early() -> None:
    node = _node({"only-task": [80.0, 80.0, 150.0]})

    assert (
        robust_utility_trigger_ms(
            node,
            parent=None,
            threshold_ms=100.0,
            kv_cost_ms=100.0,
        )
        == 100.0
    )
    assert (
        robust_utility_trigger_stats(
            node,
            parent=None,
            threshold_ms=100.0,
            kv_cost_ms=100.0,
        ).normalized_advantage
        == 0.0
    )


@pytest.mark.parametrize(
    ("threshold_ms", "kv_cost_ms", "message"),
    [
        (0.0, 100.0, "threshold_ms"),
        (100.0, float("inf"), "kv_cost_ms"),
    ],
)
def test_robust_clock_rejects_invalid_costs(
    threshold_ms: float,
    kv_cost_ms: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        robust_utility_trigger_ms(
            _node({"task-a": [100.0], "task-b": [100.0]}),
            parent=None,
            threshold_ms=threshold_ms,
            kv_cost_ms=kv_cost_ms,
        )


def test_robust_clock_parent_can_delay_overconfident_child() -> None:
    child = _node(
        {
            "task-a": [80.0, 150.0],
            "task-b": [80.0, 150.0],
        },
        source="prior_group",
        group_key="exec:work",
    )
    parent = _node(
        {
            "task-a": [80.0, 150.0],
            "task-b": [80.0, 150.0],
            "task-c": [99.0] * 200,
        },
        source="prior_tool",
    )

    trigger = robust_utility_trigger_ms(
        child,
        parent=parent,
        threshold_ms=100.0,
        kv_cost_ms=100.0,
    )

    # The child alone prefers k=0. The parent waits until its repeated 99ms
    # calls have exited, then captures the remaining positive long-call value.
    assert trigger == 99.0


def test_utility_clock_policies_do_not_depend_on_binary_majority() -> None:
    profile_rows = _profile_rows_with_minority_long_tail()
    eval_rows = [
        _row("eval-long", 150.0, source_trace="eval-a", task_id="eval-a"),
        _row("eval-short", 80.0, source_trace="eval-b", task_id="eval-b"),
    ]

    summary = evaluate_utility_clock_policy(
        eval_rows,
        profile_rows=profile_rows,
        kv_costs_ms=[100.0],
        guard_ms=0.0,
    )

    assert summary["policies"] == [
        "deadline_only",
        "mean_hazard",
        "robust_clock",
    ]
    assert summary["profile_task_count"] == 2
    (point,) = summary["points"]
    deadline = point["policies"]["deadline_only"]
    assert deadline["trigger_count"] == 1
    assert deadline["early_trigger_count"] == 0
    assert deadline["absorbed_on_long_ms_total"] == 50.0
    assert deadline["exposed_ms_total"] == 50.0
    assert deadline["net_saved_ms"] == 0.0

    for policy in ("mean_hazard", "robust_clock"):
        row = point["policies"][policy]
        assert row["trigger_count"] == 2
        assert row["early_trigger_count"] == 2
        assert row["early_trigger_on_short_count"] == 1
        assert row["absorbed_on_long_ms_total"] == 100.0
        assert row["exposed_ms_total"] == 20.0
        assert row["net_saved_ms"] == 80.0

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    assert decisions["eval-long"]["robust_trigger_ms"] == 0.0
    assert decisions["eval-short"]["robust_trigger_ms"] == 0.0


def test_single_task_prefix_backs_off_for_robust_clock() -> None:
    profile_rows = [
        _row(
            "a-work-long",
            150.0,
            source_trace="profile-a",
            task_id="task-a",
            command="work --special",
        ),
        _row(
            "a-work-short",
            80.0,
            source_trace="profile-a",
            task_id="task-a",
            command="work --special",
        ),
        _row(
            "b-other-short",
            10.0,
            source_trace="profile-b",
            task_id="task-b",
            command="other",
        ),
    ]
    eval_rows = [
        _row(
            "eval",
            150.0,
            source_trace="eval",
            task_id="eval",
            command="work --special",
        )
    ]

    summary = evaluate_utility_clock_policy(
        eval_rows,
        profile_rows=profile_rows,
        kv_costs_ms=[100.0],
        guard_ms=0.0,
        command_field="command",
        max_prefix_depth=4,
    )

    (decision,) = summary["decisions"]
    assert decision["prior_source"] == "prior_group"
    assert decision["prior_task_count"] == 1
    assert decision["robust_source"] == "prior_tool"
    assert decision["robust_task_count"] == 2


def test_utility_clock_cli_writes_summary_and_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    output_path = tmp_path / "summary.json"
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(profile_path, _profile_rows_with_minority_long_tail())
    _write_jsonl(
        eval_path,
        [_row("eval", 150.0, source_trace="eval", task_id="eval")],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_utility_clock_policy.py",
            "--profile-latencies",
            str(profile_path),
            "--eval-latencies",
            str(eval_path),
            "--kv-costs-ms",
            "100",
            "--output",
            str(output_path),
            "--decisions-output",
            str(decisions_path),
        ],
    )

    utility_clock_main()

    assert (
        f"Evaluated 1 kv points over 1 rows -> {output_path}" in capsys.readouterr().out
    )
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert "decisions" not in summary
    decisions = _read_jsonl(decisions_path)
    assert len(decisions) == 1
    assert decisions[0]["robust_trigger_ms"] == 0.0


def test_utility_clock_rejects_profile_eval_task_overlap() -> None:
    rows = _profile_rows_with_minority_long_tail()

    with pytest.raises(ValueError, match="disjoint logical tasks"):
        evaluate_utility_clock_policy(
            [
                _row(
                    "eval",
                    150.0,
                    source_trace="other-trace",
                    task_id="task-a",
                )
            ],
            profile_rows=rows,
            kv_costs_ms=[100.0],
            guard_ms=0.0,
        )


def test_utility_clock_reports_deadline_exposure_for_barely_long_call() -> None:
    summary = evaluate_utility_clock_policy(
        [
            _row(
                "barely-long",
                160.0,
                source_trace="eval",
                task_id="eval",
            )
        ],
        profile_rows=[
            _row(
                "short-a",
                10.0,
                source_trace="profile-a",
                task_id="task-a",
            ),
            _row(
                "short-b",
                10.0,
                source_trace="profile-b",
                task_id="task-b",
            ),
        ],
        kv_costs_ms=[100.0],
        guard_ms=50.0,
    )

    (point,) = summary["points"]
    deadline = point["policies"]["deadline_only"]
    assert deadline["absorbed_on_long_ms_total"] == 10.0
    assert deadline["exposed_ms_total"] == 90.0
    assert deadline["net_saved_ms"] == -80.0


def test_oof_aggregator_recomputes_policy_metrics_from_raw_decisions(
    tmp_path: Path,
) -> None:
    for fold in range(1, 6):
        rows = []
        for cost in (500.0, 1000.0, 2000.0):
            rows.append(
                {
                    "sample_id": f"sample-{fold}",
                    "task_id": f"task-{fold}",
                    "tool_name": "exec",
                    "latency_ms": 1500.0,
                    "kv_cost_ms": cost,
                    "threshold_ms": cost,
                    "label_exceeds_threshold": 1500.0 > cost,
                    "prior_source": "prior_tool",
                    "prior_task_count": 4,
                    "robust_source": "prior_tool",
                    "robust_task_count": 4,
                    "deadline_trigger_ms": cost,
                    "mean_hazard_trigger_ms": 0.0,
                    "robust_trigger_ms": cost,
                }
            )
        _write_jsonl(tmp_path / f"f{fold}_decisions.jsonl", rows)

    result = _load_aggregate_module().aggregate(tmp_path)

    assert result["oof_sample_count"] == 5
    at_500 = result["points"]["500.0"]["policies"]
    assert at_500["deadline_only"]["net_saved_ms"] == 2500.0
    assert at_500["mean_hazard"]["early_trigger_count"] == 5
    assert at_500["mean_hazard"]["mean_fired_trigger_ms"] == 0.0
    assert result["points"]["500.0"]["prior_task_count_histogram"] == {"4": 5}
    at_1000 = result["points"]["1000.0"]["policies"]
    assert at_1000["deadline_only"]["net_saved_ms"] == 0.0
    assert at_1000["mean_hazard"]["net_saved_ms"] == 5000.0
    at_2000 = result["points"]["2000.0"]["policies"]
    assert at_2000["deadline_only"]["trigger_count"] == 0
    assert at_2000["mean_hazard"]["net_saved_ms"] == -2500.0


def _load_aggregate_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / ".omc/artifacts/tool-time-utility-clock-probe-20260711/aggregate_cv.py"
    )
    spec = importlib.util.spec_from_file_location("utility_clock_aggregate_cv", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load aggregate module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile_rows_with_minority_long_tail() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_id in ("task-a", "task-b"):
        for index, latency in enumerate((80.0, 80.0, 150.0)):
            rows.append(
                _row(
                    f"{task_id}-{index}",
                    latency,
                    source_trace=f"profile-{task_id}",
                    task_id=task_id,
                )
            )
    return rows


def _node(
    values_by_task: dict[str, list[float]],
    *,
    source: str = "prior_tool",
    group_key: str | None = None,
) -> LatencyPriorNode:
    values = sorted(
        value for task_values in values_by_task.values() for value in task_values
    )
    return LatencyPriorNode(
        values=values,
        values_by_task={
            task: sorted(values) for task, values in values_by_task.items()
        },
        source=source,
        group_key=group_key,
    )


def _row(
    sample_id: str,
    latency_ms: float,
    *,
    source_trace: str,
    task_id: str,
    command: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample_id,
        "source_trace": source_trace,
        "task_id": task_id,
        "tool_name": "exec",
        "tool_ts_start": 0.0,
        "tool_ts_end": latency_ms / 1000.0,
        "latency_ms": latency_ms,
    }
    if command is not None:
        row["tool_args"] = {"command": command}
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
