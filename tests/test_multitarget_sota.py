from copy import deepcopy

import pytest

from scripts.evaluation.evaluate_multitarget_sota import DISK, _compose, run


def _rows() -> tuple[dict, dict, dict]:
    current = {
        "latency": 0,
        "peak_cpu_cores": "low",
        "sampled_peak_rss_mb": "low",
        DISK: "low",
    }
    pmfs = {
        "latency": [1, 0, 0, 0, 0],
        "peak_cpu_cores": [1, 0, 0],
        "sampled_peak_rss_mb": [1, 0, 0],
        DISK: [1, 0, 0],
    }
    shared = {
        "sample_id": "task:0",
        "task_id": "task",
        "command": "pytest",
        "labels": {target: 0 for target in current},
        "current_dynamic": current,
        "current_probability_by_bucket": pmfs,
    }
    semantic_arm = {
        "candidate": dict(current),
        "candidate_probability_by_bucket": deepcopy(pmfs),
        "provenance": {target: {"source": "current"} for target in current},
    }
    semantic_arm["candidate"]["sampled_peak_rss_mb"] = "high"
    semantic_arm["candidate_probability_by_bucket"]["sampled_peak_rss_mb"] = [0, 0, 1]
    semantic = {**shared, "arms": {"semantic_work_units": semantic_arm}}
    phase = {
        **shared,
        "phase_applied_targets": ["latency", "peak_cpu_cores", "sampled_peak_rss_mb"],
        "full_test_phase": 2,
        "candidate": {
            **current,
            "latency": 4,
            "peak_cpu_cores": "high",
            "sampled_peak_rss_mb": "medium",
        },
        "candidate_probability_by_bucket": {
            **pmfs,
            "latency": [0, 0, 0, 0, 1],
            "peak_cpu_cores": [0, 0, 1],
            "sampled_peak_rss_mb": [0, 1, 0],
        },
    }
    exact_arm = deepcopy(semantic_arm)
    exact_arm["candidate"][DISK] = "medium"
    exact_arm["candidate_probability_by_bucket"][DISK] = [0, 1, 0]
    exact_arm["provenance"][DISK] = {"source": "exact", "support": 2}
    exact = {**shared, "arms": {"exact": exact_arm}}
    return semantic, phase, exact


def test_compose_uses_target_heads_at_one_decision_time() -> None:
    semantic, phase, exact = _rows()

    row = _compose(semantic, phase, exact)

    assert row["candidate"] == {
        "latency": 4,
        "peak_cpu_cores": "high",
        "sampled_peak_rss_mb": "high",
        DISK: "medium",
    }
    assert row["phase_applied_targets"] == ["latency", "peak_cpu_cores"]
    assert row["provenance"][DISK] == {"source": "exact", "support": 2}

    exact["task_id"] = "other"
    with pytest.raises(ValueError, match="task_id"):
        _compose(semantic, phase, exact)


def test_run_rejects_duplicate_frozen_rows() -> None:
    semantic, phase, exact = _rows()

    with pytest.raises(ValueError, match="duplicate sample IDs"):
        run([semantic, semantic], [phase, phase], [exact, exact])
