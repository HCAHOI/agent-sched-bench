from copy import deepcopy

import pytest

from scripts.evaluation import evaluate_confidence_gated_composition as confidence


def _pmfs() -> dict[str, list[float]]:
    return {
        "latency": [1.0, 0.0, 0.0, 0.0, 0.0],
        "peak_cpu_cores": [0.6, 0.4, 0.0],
        "sampled_peak_rss_mb": [1.0, 0.0, 0.0],
        "disk_read_write_bytes_total": [0.6, 0.4, 0.0],
    }


def _row(task: int, *, candidate_correct: bool, feature: float) -> dict:
    current_pmfs = _pmfs()
    candidate_pmfs = deepcopy(current_pmfs)
    for target in confidence.SELECTED_TARGETS:
        candidate_pmfs[target] = [0.4 - feature, 0.6 + feature, 0.0]
    labels = {
        "latency": 0,
        "peak_cpu_cores": 1 if candidate_correct else 0,
        "sampled_peak_rss_mb": 0,
        "disk_read_write_bytes_total": 1 if candidate_correct else 0,
    }
    current = {
        "latency": 0,
        "peak_cpu_cores": "low",
        "sampled_peak_rss_mb": "low",
        "disk_read_write_bytes_total": "low",
    }
    candidate = {
        **current,
        "peak_cpu_cores": "medium",
        "disk_read_write_bytes_total": "medium",
    }
    return {
        "sample_id": f"task-{task}:0",
        "task_id": f"task-{task}",
        "command": "python -m pytest",
        "labels": labels,
        "current_dynamic": current,
        "current_probability_by_bucket": current_pmfs,
        "arms": {
            "composition": {
                "candidate": candidate,
                "candidate_probability_by_bucket": candidate_pmfs,
            }
        },
    }


def test_threshold_erm_and_conservative_tie_break() -> None:
    rows = [
        _row(0, candidate_correct=False, feature=0.01),
        _row(1, candidate_correct=True, feature=0.10),
        _row(2, candidate_correct=True, feature=0.20),
    ]

    assert confidence._fit_threshold(rows, "peak_cpu_cores") == pytest.approx(0.10)

    tied = [
        _row(0, candidate_correct=True, feature=0.10),
        _row(1, candidate_correct=False, feature=0.20),
    ]
    assert confidence._fit_threshold(tied, "peak_cpu_cores") == float("inf")


def test_clustered_evaluation_preserves_rows_and_unchanged_targets() -> None:
    rows = [
        _row(index, candidate_correct=index % 3 != 0, feature=(index % 5) / 20)
        for index in range(50)
    ]

    result, output = confidence.evaluate(rows)

    assert len(output) == 50
    assert [row["sample_id"] for row in output] == [row["sample_id"] for row in rows]
    assert {row["fold"] for row in output} == set(range(5))
    assert result["row_identity"] is True
    assert result["latency_rss_bit_identical_to_composition"] is True
    assert all(
        row["candidate_probability_by_bucket"][target]
        == row["base_probability_by_bucket"][target]
        for row in output
        for target in confidence.UNCHANGED_TARGETS
    )

    unavailable = deepcopy(rows[0])
    unavailable["current_dynamic"] = dict.fromkeys(confidence.ALL_TARGETS)
    unavailable["current_probability_by_bucket"] = dict.fromkeys(confidence.ALL_TARGETS)
    unavailable["arms"]["composition"]["candidate"] = dict.fromkeys(
        confidence.ALL_TARGETS
    )
    unavailable["arms"]["composition"]["candidate_probability_by_bucket"] = (
        dict.fromkeys(confidence.ALL_TARGETS)
    )
    assert confidence._apply(
        unavailable,
        0,
        {target: 0.0 for target in confidence.SELECTED_TARGETS},
    )["candidate"] == dict.fromkeys(confidence.ALL_TARGETS)


def test_held_fold_labels_cannot_change_its_thresholds_or_predictions(
    monkeypatch,
) -> None:
    rows = []
    for task in range(50):
        for call in range(2):
            row = _row(
                task,
                candidate_correct=(task + call) % 3 != 0,
                feature=((task + call) % 7) / 20,
            )
            row["sample_id"] = f"task-{task}:{call}"
            rows.append(row)
    changed = deepcopy(rows)
    for row in changed:
        task = int(row["task_id"].split("-")[1])
        if task % 5 == 0:
            for target in confidence.SELECTED_TARGETS:
                row["labels"][target] = 1 - row["labels"][target]

    calls = []
    fit_threshold = confidence._fit_threshold

    def recording_fit(training_rows, target):
        calls.append((target, {row["task_id"] for row in training_rows}))
        return fit_threshold(training_rows, target)

    monkeypatch.setattr(confidence, "_fit_threshold", recording_fit)
    result, output = confidence.evaluate(rows)
    assert len(calls) == 10
    for fold in range(5):
        expected_training = {f"task-{task}" for task in range(50) if task % 5 != fold}
        for target_index, target in enumerate(confidence.SELECTED_TARGETS):
            assert calls[fold * 2 + target_index] == (target, expected_training)

    calls.clear()
    changed_result, changed_output = confidence.evaluate(changed)

    assert all(row["fold"] == int(row["task_id"].split("-")[1]) % 5 for row in output)
    assert result["fold_thresholds"]["0"] == changed_result["fold_thresholds"]["0"]
    held = {
        row["sample_id"]: (row["candidate"], row["selector"])
        for row in output
        if row["fold"] == 0
    }
    changed_held = {
        row["sample_id"]: (row["candidate"], row["selector"])
        for row in changed_output
        if row["fold"] == 0
    }
    assert held == changed_held
