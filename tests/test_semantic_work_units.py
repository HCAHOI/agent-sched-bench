import json

import pytest

from scripts.evaluation.evaluate_command_history_residual import BUCKETS, DISK, TARGETS, Row
from scripts.evaluation.evaluate_semantic_work_units import (
    FROZEN_FIT_ROWS,
    FROZEN_VALIDATION_ROWS,
    SPLIT_MANIFEST,
    _pip_query,
    run,
)


def _row(sample: str, task: str, command: str, label: int) -> Row:
    return Row(
        sample,
        task,
        command,
        {target: label for target in BUCKETS},
        {
            "latency": 0,
            "peak_cpu_cores": "low",
            "sampled_peak_rss_mb": "low",
            DISK: "low",
        },
        {
            target: tuple([1.0, *([0.0] * (buckets - 1))])
            for target, buckets in BUCKETS.items()
        },
    )


def test_pip_query_is_narrow_and_canonicalizes_package_units() -> None:
    direct = _pip_query("pip install Pandas numpy==2")
    reordered = _pip_query("pip install numpy==3 pandas")

    assert direct is not None and reordered is not None
    assert direct[0] == reordered[0] == ("pip", "direct", ())
    assert direct[1] == reordered[1] == frozenset({"numpy", "pandas"})
    assert _pip_query("python3 -m pip install numpy")[0] == (
        "python3",
        "python-module",
        (),
    )
    assert _pip_query("pip install -r requirements.txt") is None
    assert _pip_query("pytest -m pip install numpy") is None
    assert _pip_query("bash -m pip install numpy") is None
    assert _pip_query("pip install numpy && pip install pandas") is None


def test_pip_overlap_is_weighted_and_partitions_invocation() -> None:
    fit = [
        _row("fit-a:0", "fit-a", "pip install alpha", 2),
        _row("fit-b:0", "fit-b", "pip install beta", 0),
    ]
    validation = [
        _row("t1:0", "t1", "pip install alpha beta", 0),
        _row("t1:1", "t1", "python3 -m pip install alpha beta", 0),
    ]

    result, rows = run(fit, validation)

    primary = rows[0]["arms"]["semantic_work_units"]
    assert primary["candidate_probability_by_bucket"]["latency"] == pytest.approx(
        [0.5, 0.0, 0.5, 0.0, 0.0]
    )
    assert primary["candidate"]["latency"] == 0
    for target in TARGETS:
        assert primary["provenance"][target] == {
            "source": "pip_package_overlap",
            "support": 2,
            "weight_sum": 1.0,
        }
        assert rows[0]["arms"]["pytest_only"]["provenance"][target] == {
            "source": "current",
            "support": 0,
        }
        assert rows[1]["arms"]["semantic_work_units"]["provenance"][target] == {
            "source": "current",
            "support": 0,
        }
    assert result["coverage"] == {
        "pip_eligible_rows": 2,
        "pip_without_prior_hierarchy_evidence": 2,
        "pip_rows_with_overlap_prediction": 1,
    }


def test_same_task_labels_are_hidden_until_settlement() -> None:
    validation = [
        _row("t1:0", "t1", "pip install omega", 2),
        _row("t1:1", "t1", "pip install omega delta", 2),
        _row("t2:0", "t2", "pip install omega epsilon", 2),
    ]

    _result, rows = run([], validation)

    sources = [
        row["arms"]["semantic_work_units"]["provenance"]["latency"]["source"]
        for row in rows
    ]
    assert sources == ["current", "current", "pip_package_overlap"]
    assert rows[2]["arms"]["semantic_work_units"]["provenance"]["latency"][
        "support"
    ] == 2


def test_exact_and_pytest_precede_pip_and_disk_is_always_current() -> None:
    fit = [
        _row("fit-pip:0", "fit-pip", "pip install alpha", 2),
        _row("fit-test:0", "fit-test", "pytest tests/a.py::test_one", 2),
    ]
    validation = [
        _row("t1:0", "t1", "pip install alpha", 2),
        _row("t2:0", "t2", "pytest tests/a.py::test_two", 2),
    ]

    result, rows = run(fit, validation)

    assert rows[0]["arms"]["semantic_work_units"]["provenance"]["latency"] == {
        "source": "exact",
        "support": 1,
    }
    assert rows[1]["arms"]["semantic_work_units"]["provenance"]["latency"][
        "source"
    ] == "pytest_target_overlap"
    for row in rows:
        for arm in ("pytest_only", "semantic_work_units"):
            candidate = row["arms"][arm]
            assert candidate["candidate"][DISK] == row["current_dynamic"][DISK]
            assert candidate["candidate_probability_by_bucket"][DISK] == row[
                "current_probability_by_bucket"
            ][DISK]
            assert candidate["provenance"][DISK] == {
                "source": "current",
                "support": 0,
            }
    assert result["gate"]["disk_bit_identical_to_current"] is True
    assert result["arms"]["semantic_work_units"][DISK][
        "delta_percentage_points"
    ] == 0.0


def test_gate_and_row_identity_cover_all_changed_targets() -> None:
    fit = [_row("fit:0", "fit", "pip install alpha", 2)]
    validation = [_row("t1:0", "t1", "pip install alpha beta", 2)]

    result, rows = run(fit, validation)

    assert len(rows) == 1
    assert result["gate"] == {
        "go": False,
        "minimum_gain_percentage_points_each_changed_target": 5.0,
        "all_changed_targets_meet_gain": True,
        "no_severe_underprediction_regression": True,
        "helpful": 3,
        "harmful": 0,
        "helpful_tasks": 1,
        "minimum_helpful_tasks": 10,
        "row_identity": True,
        "disk_bit_identical_to_current": True,
    }
    assert result["protocol"]["changed_targets"] == list(TARGETS)
    assert result["protocol"]["unchanged_target"] == DISK


def test_frozen_inputs_and_split_are_present() -> None:
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))

    assert FROZEN_FIT_ROWS.name == "rows.jsonl" and FROZEN_FIT_ROWS.is_file()
    assert FROZEN_VALIDATION_ROWS.name == "rows.jsonl" and FROZEN_VALIDATION_ROWS.is_file()
    assert len(split["development"]) == 100
    assert len(split["validation"]) == 50
    assert len(split["final_test"]) == 50
