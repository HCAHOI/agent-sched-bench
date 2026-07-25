from __future__ import annotations

import pytest

from scripts.evaluation.evaluate_clause_resource_kb import (
    ScoredRow,
    _metrics,
    _validate_partition,
)


def _row(truth: bool, prediction: bool | None) -> ScoredRow:
    return ScoredRow(
        target="cpu_heavy_2cores",
        sample_id="sample",
        task_id="owner__repo-1",
        repo="owner__repo",
        command="pytest",
        truth=truth,
        prediction=prediction,
        layer="public",
        key_kind="bin",
        evidence_count=1,
        command_structure="single",
        mapping_evidence="bin_exact",
    )


def test_metrics_keep_unknown_out_of_confusion_but_in_coverage() -> None:
    metrics = _metrics(
        [_row(True, True), _row(False, False), _row(True, None), _row(False, True)]
    )
    assert metrics == {
        "eligible": 4,
        "positive": 2,
        "prevalence": 0.5,
        "known": 3,
        "unknown": 1,
        "coverage": 0.75,
        "balanced_accuracy": 0.75,
        "recall": 1.0,
        "precision": 0.5,
        "tp": 1,
        "tn": 1,
        "fp": 1,
        "fn": 0,
    }


def test_partition_overlap_fails_closed() -> None:
    with pytest.raises(ValueError, match="fit/eval task overlap"):
        _validate_partition(["owner__repo-1"], ["owner__repo-1"])
