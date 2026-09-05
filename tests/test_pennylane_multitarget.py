from scripts.evaluation.evaluate_pennylane_multitarget import (
    EXPOSED_VALIDATION_TASK,
    _gate,
    _split,
)
from scripts.evaluation.evaluate_doc_tool_semantics import _changes


def test_validation_split_excludes_only_exposed_task() -> None:
    task_ids, warmup_count, _path = _split("validation")

    assert warmup_count == 16
    assert len(task_ids) == 31
    assert EXPOSED_VALIDATION_TASK not in task_ids


def test_transfer_gate_requires_all_fixed_conditions() -> None:
    metrics = {
        "clause_kb": {
            "equal_weight_accuracy": 0.7,
            "equal_weight_severe_underprediction_rate": 0.1,
        },
        "task_aware": {
            "equal_weight_accuracy": 0.71,
            "equal_weight_severe_underprediction_rate": 0.1,
        },
    }
    changes = {"helpful": 3, "harmful": 2, "helpful_task_ids": ["a", "b"]}

    assert _gate(metrics, changes)["go"] is True
    changes["harmful"] = 3
    assert _gate(metrics, changes)["go"] is False


def test_change_counts_ignore_unavailable_labels() -> None:
    targets = (
        "latency",
        "peak_cpu_cores",
        "sampled_peak_rss_mb",
        "disk_read_write_bytes_total",
    )
    rows = [
        {
            "sample_id": "task:0",
            "task_id": "task",
            "command": "true",
            "labels": {target: None for target in targets},
            "arms": {
                "left": {
                    "prediction": {target: None for target in targets},
                    "probability_by_bucket": {target: None for target in targets},
                },
                "right": {
                    "prediction": {target: 0 for target in targets},
                    "probability_by_bucket": {target: None for target in targets},
                },
            },
        }
    ]

    changes = _changes(rows, "left", "right")

    assert changes["changed"] == changes["helpful"] == changes["harmful"] == 0
