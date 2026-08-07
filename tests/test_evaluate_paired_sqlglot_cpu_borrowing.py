import json

import pytest

from scripts.evaluation import evaluate_paired_sqlglot_cpu_borrowing as experiment


def test_frozen_cohorts_partition_validation_tasks() -> None:
    initial = experiment._protocol("initial24")
    remaining = experiment._protocol("remaining26")
    split = json.loads(experiment.SPLIT.read_text(encoding="utf-8"))

    initial_ids = set(initial["selected_task_ids"])
    remaining_ids = set(remaining["selected_task_ids"])
    assert initial["selection_range"] == [0, 24]
    assert remaining["selection_range"] == [24, 50]
    assert len(initial_ids) == 24
    assert len(remaining_ids) == 26
    assert initial_ids.isdisjoint(remaining_ids)
    assert initial_ids | remaining_ids == set(split["validation"])
    assert len(initial["pairs"]) == 12
    assert len(remaining["pairs"]) == 13
    frozen = json.loads(
        (
            experiment.ROOT
            / "analysis/results/tool-resource-5-3-3-3-20260804"
            / "sqlglot24-paired-cpu-borrowing-timeout-floor-v2/protocol.json"
        ).read_text(encoding="utf-8")
    )
    assert initial["selected_task_ids"] == frozen["selected_task_ids"]
    assert initial["pairs"] == frozen["pairs"]
    assert remaining["arm_order_seed"] == 20_260_810
    assert remaining["bootstrap_seed"] == 20_260_811
    assert [(pair["task_ids"], pair["arm_order"]) for pair in remaining["pairs"]] == [
        (
            ["tobymao__sqlglot-3620", "tobymao__sqlglot-3333"],
            ["hard_two", "burstable_two"],
        ),
        (
            ["tobymao__sqlglot-3238", "tobymao__sqlglot-2439"],
            ["hard_two", "burstable_two"],
        ),
        (
            ["tobymao__sqlglot-2857", "tobymao__sqlglot-2739"],
            ["hard_two", "burstable_two"],
        ),
        (
            ["tobymao__sqlglot-2754", "tobymao__sqlglot-3284"],
            ["hard_two", "burstable_two"],
        ),
        (
            ["tobymao__sqlglot-3182", "tobymao__sqlglot-3394"],
            ["hard_two", "burstable_two"],
        ),
        (
            ["tobymao__sqlglot-3734", "tobymao__sqlglot-3440"],
            ["burstable_two", "hard_two"],
        ),
        (
            ["tobymao__sqlglot-2794", "tobymao__sqlglot-2619"],
            ["burstable_two", "hard_two"],
        ),
        (
            ["tobymao__sqlglot-3417", "tobymao__sqlglot-3515"],
            ["hard_two", "burstable_two"],
        ),
        (
            ["tobymao__sqlglot-2443", "tobymao__sqlglot-3230"],
            ["burstable_two", "hard_two"],
        ),
        (
            ["tobymao__sqlglot-3179", "tobymao__sqlglot-2337"],
            ["hard_two", "burstable_two"],
        ),
        (
            ["tobymao__sqlglot-3198", "tobymao__sqlglot-3566"],
            ["burstable_two", "hard_two"],
        ),
        (
            ["tobymao__sqlglot-3637", "tobymao__sqlglot-2822"],
            ["burstable_two", "hard_two"],
        ),
        (
            ["tobymao__sqlglot-3630", "tobymao__sqlglot-3204"],
            ["burstable_two", "hard_two"],
        ),
    ]


def _runs(protocol: dict, improvements: list[float]) -> list[dict]:
    runs = []
    for pair, improvement in zip(protocol["pairs"], improvements, strict=True):
        pair_number = pair["pair"]
        arms = [
            {
                "arm": "hard_two",
                "pair_makespan_s": 1.0,
                "action_sequences": {"task": [["agent", "tool_exec", "tool"]]},
            },
            {
                "arm": "burstable_two",
                "pair_makespan_s": 1.0 - improvement,
                "action_sequences": {"task": [["agent", "tool_exec", "tool"]]},
            },
        ]
        runs.append({"pair": pair_number, "arms": arms})
    return runs


def test_remaining_cohort_requires_ten_improving_pairs() -> None:
    protocol = experiment._protocol("remaining26")
    runs = _runs(protocol, [0.1] * 10 + [0.0] * 3)

    result = experiment._aggregate(protocol, runs)

    assert result["status"] == "go"
    assert result["comparison"]["improving_pairs"] == 10
    assert result["comparison"]["gate"]["at_least_10_of_13_pairs_improve"]

    runs = _runs(protocol, [0.1] * 9 + [0.0] * 4)
    result = experiment._aggregate(protocol, runs)
    assert result["status"] == "no_go"
    assert result["comparison"]["improving_pairs"] == 9
    assert not result["comparison"]["gate"]["at_least_10_of_13_pairs_improve"]


def test_remaining_cohort_freezes_bootstrap_and_effect_gates() -> None:
    protocol = experiment._protocol("remaining26")
    varied = [
        0.31,
        0.23,
        0.18,
        0.14,
        0.11,
        0.09,
        0.07,
        0.052,
        0.031,
        0.013,
        -0.017,
        -0.083,
        -0.271,
    ]
    result = experiment._aggregate(protocol, _runs(protocol, varied))
    comparison = result["comparison"]
    assert comparison["bootstrap_draws"] == 10_000
    assert comparison["ci95_paired_pair_bootstrap"] == pytest.approx(
        [-0.01361923076923077, 0.13892307692307695]
    )
    assert comparison["gate"]["mean_improvement_at_least_5_percent"]
    assert not comparison["gate"]["bootstrap_lower_above_zero"]
    assert result["status"] == "no_go"

    result = experiment._aggregate(protocol, _runs(protocol, [0.05] * 13))
    assert result["status"] == "go"
    result = experiment._aggregate(protocol, _runs(protocol, [0.049999] * 13))
    assert not result["comparison"]["gate"]["mean_improvement_at_least_5_percent"]
    assert result["comparison"]["gate"]["bootstrap_lower_above_zero"]
    assert result["status"] == "no_go"
