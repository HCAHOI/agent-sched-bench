from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluation.evaluate_tabular_resource import (
    _balanced_accuracy_uncertainty,
    _binary_metrics,
    _build_out_of_repo_fit_dataset,
    _clamp_prediction_pair,
    _inverse_prevalence_weights,
    _predict_binary_probabilities,
    _standardize_fit_eval,
    _write_dump_rows,
    quantile_exceedance,
    repo_cluster_key,
)


def test_scaler_uses_fit_split_only() -> None:
    fit = np.asarray([[0.0], [2.0]])
    eval_a = np.asarray([[100.0]])
    eval_b = np.asarray([[-100.0]])

    fit_a, scaled_a, mean_a, std_a = _standardize_fit_eval(fit, eval_a)
    fit_b, scaled_b, mean_b, std_b = _standardize_fit_eval(fit, eval_b)

    np.testing.assert_array_equal(mean_a, mean_b)
    np.testing.assert_array_equal(std_a, std_b)
    np.testing.assert_array_equal(fit_a, fit_b)
    assert scaled_a[0, 0] == pytest.approx(99.0)
    assert scaled_b[0, 0] == pytest.approx(-101.0)


def test_prediction_clamp_is_symmetric() -> None:
    model, baseline = _clamp_prediction_pair(
        np.asarray([[-1.0, 20.0, 200.0]]),
        np.asarray([[-2.0, 30.0, 300.0]]),
        fit_max=10.0,
    )

    np.testing.assert_array_equal(model, [[0.0, 20.0, 100.0]])
    np.testing.assert_array_equal(baseline, [[0.0, 30.0, 100.0]])


def test_quantile_exceedance_interpolates_hand_built_case() -> None:
    scores = quantile_exceedance(
        np.asarray([[10.0, 20.0, 30.0]]),
        [5.0, 15.0, 25.0, 35.0],
    )

    np.testing.assert_allclose(scores, [[0.50, 0.30, 0.075, 0.05]])


def test_repo_cluster_key_strips_only_trailing_issue_number() -> None:
    assert repo_cluster_key("owner__repo-name-123") == "owner__repo-name"
    assert repo_cluster_key("owner__repo-name") == "owner__repo-name"
    assert repo_cluster_key("owner__repo-name-12x") == "owner__repo-name-12x"


def test_fit_prior_features_leave_current_repo_out() -> None:
    first = _sample("owner__first-1", "first", 0.1)
    second = _sample("owner__second-1", "second", 0.9)
    dataset = _build_out_of_repo_fit_dataset(
        {
            "owner__first-1": [first],
            "owner__second-1": [second],
        },
        None,
        [],
    )
    p50_by_task = dict(
        zip(
            dataset.task_ids,
            dataset.features["prior_latency_p50_ms"],
            strict=True,
        )
    )

    assert p50_by_task["owner__first-1"] == pytest.approx(900.0)
    assert p50_by_task["owner__second-1"] == pytest.approx(100.0)


def test_prevalence_matching_is_permutation_invariant_under_ties() -> None:
    scores = np.asarray([0.5, 0.5])
    first = _binary_metrics(np.asarray([True, False]), scores, 0.5)
    reversed_rows = _binary_metrics(np.asarray([False, True]), scores, 0.5)

    assert first == reversed_rows
    assert first["balanced_accuracy"] == pytest.approx(0.5)
    assert first["minority_recall"] == pytest.approx(0.5)
    assert first["minority_precision"] == pytest.approx(0.5)


def test_fit_prevalence_cut_does_not_use_eval_positive_count() -> None:
    labels = np.asarray([True, True, True, True, True, True, False, False])
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

    metrics = _binary_metrics(labels, scores, fit_prevalence=0.25)

    assert metrics["prevalence_matched_positive_count"] == 2
    assert metrics["oracle_prevalence_cut"]["prevalence_matched_positive_count"] == 6
    assert metrics["decision_cut_source"] == "fit_label_prevalence"


def test_inverse_prevalence_weights_balance_imbalanced_classes() -> None:
    labels = np.asarray([True, False, False, False])
    weights = _inverse_prevalence_weights(labels)

    assert weights[0] == pytest.approx(2.0)
    assert weights[1] == pytest.approx(2.0 / 3.0)
    assert weights[labels].sum() == pytest.approx(weights[~labels].sum())


def test_binary_probabilities_are_raw_sigmoid_outputs() -> None:
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(math.log(4.0))

    probabilities = _predict_binary_probabilities(
        model, np.zeros((2, 1), dtype=np.float32), torch.device("cpu")
    )

    np.testing.assert_allclose(probabilities, [0.8, 0.8])
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))
    assert np.all(probabilities > 0.5)


def test_repo_confusion_bootstrap_is_seed_reproducible() -> None:
    labels = np.asarray([True, False, True, False, True, False])
    model_scores = np.asarray([0.9, 0.2, 0.7, 0.4, 0.6, 0.1])
    baseline_scores = np.asarray([0.8, 0.3, 0.5, 0.6, 0.4, 0.2])
    task_ids = [
        "owner__a-1",
        "owner__a-2",
        "owner__b-1",
        "owner__b-2",
        "owner__c-1",
        "owner__c-2",
    ]

    first = _balanced_accuracy_uncertainty(
        labels,
        model_scores,
        baseline_scores,
        task_ids,
        fit_prevalence=0.5,
        seed=7,
    )
    second = _balanced_accuracy_uncertainty(
        labels,
        model_scores,
        baseline_scores,
        task_ids,
        fit_prevalence=0.5,
        seed=7,
    )

    assert first == second
    assert first["model"]["point"] == pytest.approx(1.0)
    assert first["model_vs_baseline_difference"]["point"] > 0.0


def test_dump_rows_writes_each_eligible_row_without_mutating_metrics(
    tmp_path,
) -> None:
    metrics = {"targets": {"cpu_peak": {"eligible_eval_rows": 2}}}
    metrics_before = json.dumps(metrics, sort_keys=True)
    rows = [
        {"sample_id": "first", "observed": 1.0},
        {"sample_id": "second", "observed": 2.0},
    ]
    path = tmp_path / "rows.jsonl"

    _write_dump_rows(path, rows)

    dumped = [json.loads(line) for line in path.read_text().splitlines()]
    assert path.exists()
    assert len(dumped) == metrics["targets"]["cpu_peak"]["eligible_eval_rows"]
    assert dumped == rows
    assert json.dumps(metrics, sort_keys=True) == metrics_before


def _sample(task_id: str, sample_id: str, end: float) -> SimpleNamespace:
    row = {
        "sample_id": sample_id,
        "source_trace": f"trace-{sample_id}",
        "task_id": task_id,
        "tool_name": "exec",
        "tool_args": {"command": "pytest -q"},
    }
    return SimpleNamespace(
        **row,
        tool_ts_start=0.0,
        tool_ts_end=end,
        censored=False,
        peak_cpu_cores=None,
        peak_cpu_cores_eligible=False,
        peak_memory_mb=None,
        peak_memory_mb_eligible=False,
        to_json_obj=lambda: dict(row),
    )
