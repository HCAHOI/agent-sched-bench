from __future__ import annotations

import pytest

from tool_time.prior import LatencyPriorNode
from tool_time.policy import (
    evaluate_utility_clock_policy,
    robust_utility_trigger_ms,
    robust_utility_trigger_stats,
    trigger_policy_utility_ms,
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


def test_trigger_policy_utility_restore_cost_charges_unnecessary_swap() -> None:
    kwargs = {"threshold_ms": 100.0, "kv_cost_ms": 50.0}

    # A swap fully hidden inside a short call is free without a restore cost:
    # the swap-back the deadline policy never pays is invisible to the metric.
    assert trigger_policy_utility_ms(80.0, 0.0, **kwargs) == 0.0
    assert (
        trigger_policy_utility_ms(80.0, 0.0, restore_cost_ms=30.0, **kwargs) == -30.0
    )
    # Long calls fire under the deadline policy too, so no differential charge.
    assert (
        trigger_policy_utility_ms(150.0, 0.0, restore_cost_ms=30.0, **kwargs) == 50.0
    )
    # A trigger the call never survives costs nothing at any restore cost.
    assert (
        trigger_policy_utility_ms(80.0, 80.0, restore_cost_ms=30.0, **kwargs) == 0.0
    )
    with pytest.raises(ValueError, match="restore_cost_ms"):
        trigger_policy_utility_ms(80.0, 0.0, restore_cost_ms=-1.0, **kwargs)


def test_robust_clock_restore_cost_delays_early_trigger() -> None:
    node = _node(
        {
            "task-a": [80.0, 80.0, 150.0],
            "task-b": [80.0, 80.0, 150.0],
        }
    )
    kwargs = {"parent": None, "threshold_ms": 100.0, "kv_cost_ms": 100.0}

    assert robust_utility_trigger_ms(node, **kwargs) == 0.0
    # At rho=35 each short fire costs 20 exposed + 35 restore, so k=0 earns
    # (100 - 2*55)/3 < the k=80 curve value of 40/3; the clock moves to k=80,
    # after the short calls have finished.
    assert robust_utility_trigger_ms(node, restore_cost_ms=35.0, **kwargs) == 80.0


def test_utility_clock_policy_reports_restore_accounting() -> None:
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
        restore_cost_fraction=0.05,
    )

    assert summary["restore_cost_fraction"] == 0.05
    (point,) = summary["points"]
    assert point["restore_cost_ms"] == 5.0
    deadline = point["policies"]["deadline_only"]
    assert deadline["restore_ms_total"] == 0.0
    assert deadline["net_saved_ms"] == 0.0
    for policy in ("mean_hazard", "robust_clock"):
        row = point["policies"][policy]
        assert row["early_trigger_on_short_count"] == 1
        assert row["restore_ms_total"] == 5.0
        assert row["net_saved_ms"] == 75.0


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
