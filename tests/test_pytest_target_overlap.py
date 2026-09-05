import json

import pytest

from scripts.evaluation.evaluate_command_history_residual import BUCKETS, Row
from scripts.evaluation.evaluate_pytest_target_overlap import (
    FROZEN_FIT_ROWS,
    FROZEN_VALIDATION_ROWS,
    SPLIT_MANIFEST,
    _pytest_query,
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
            "disk_read_write_bytes_total": "low",
        },
        {
            target: tuple([1.0, *([0.0] * (buckets - 1))])
            for target, buckets in BUCKETS.items()
        },
    )


def test_pytest_query_extracts_normalized_file_and_node_units() -> None:
    query = _pytest_query(
        "python3 -m pytest -q --maxfail 2 ./tests/a.py::Suite::test_x tests/b.py"
    )

    assert query is not None
    signature, units = query
    assert signature.maxfail == 2
    assert units == frozenset(
        {
            "file:tests/a.py",
            "node:tests/a.py::Suite::test_x",
            "file:tests/b.py",
        }
    )
    assert _pytest_query("pytest --unknown tests/a.py") is None
    assert _pytest_query("pytest tests/a.py && pytest tests/b.py") is None


def test_target_overlap_is_causal_and_does_not_collapse_unrelated_targets() -> None:
    fit = [_row("fit:0", "fit", "pytest tests/a.py::test_one", 2)]
    validation = [
        _row("t1:0", "t1", "pytest -q tests/a.py::test_two", 2),
        _row("t1:1", "t1", "pytest -q tests/b.py::test_two", 0),
        _row("t2:0", "t2", "pytest -q tests/a.py::test_three", 2),
    ]

    result, rows = run(fit, validation)

    assert [
        row["arms"]["target_overlap"]["candidate"]["latency"] for row in rows
    ] == [2, 0, 2]
    assert rows[0]["arms"]["target_overlap"]["provenance"]["latency"] == {
        "source": "pytest_target_overlap",
        "support": 1,
        "weight_sum": 1 / 3,
    }
    assert rows[1]["arms"]["target_overlap"]["provenance"]["latency"] == {
        "source": "current",
        "support": 0,
    }
    assert rows[1]["arms"]["collapsed_signature"]["candidate"]["latency"] == 2
    assert rows[2]["arms"]["target_overlap"]["provenance"]["latency"][
        "support"
    ] == 2
    assert rows[0]["arms"]["target_overlap"]["candidate"] == {
        "latency": 2,
        "peak_cpu_cores": "high",
        "sampled_peak_rss_mb": "high",
        "disk_read_write_bytes_total": "high",
    }
    assert result["gate"]["minimum_gain_percentage_points_each"] == 5.0
    assert result["gate"]["minimum_helpful_tasks"] == 10
    assert result["gate"]["row_identity"] is True
    assert result["gate"] == {
        "go": False,
        "minimum_gain_percentage_points_each": 5.0,
        "all_targets_meet_gain": True,
        "no_severe_underprediction_regression": True,
        "helpful": 8,
        "harmful": 0,
        "helpful_tasks": 2,
        "minimum_helpful_tasks": 10,
        "row_identity": True,
    }


def test_exact_precedes_overlap_and_modifiers_partition_evidence() -> None:
    fit = [
        _row("fit-exact:0", "fit-exact", "pytest tests/a.py::test_one", 2),
        _row("fit-file:0", "fit-file", "pytest tests/a.py", 0),
    ]
    validation = [
        _row("t1:0", "t1", "pytest tests/a.py::test_one", 2),
        _row("t1:1", "t1", "pytest --collect-only tests/a.py::test_two", 0),
    ]

    _result, rows = run(fit, validation)

    assert rows[0]["arms"]["target_overlap"]["provenance"]["latency"] == {
        "source": "exact",
        "support": 1,
    }
    assert rows[0]["arms"]["target_overlap"]["candidate"]["latency"] == 2
    assert rows[1]["arms"]["target_overlap"]["provenance"]["latency"] == {
        "source": "current",
        "support": 0,
    }


def test_same_task_is_hidden_and_weighted_pmf_uses_jaccard() -> None:
    fit = [
        _row("fit-node:0", "fit-node", "pytest tests/a.py::test_one", 2),
        _row("fit-file:0", "fit-file", "pytest tests/a.py", 0),
    ]
    validation = [
        _row("t1:0", "t1", "pytest tests/c.py::test_one", 2),
        _row("t1:1", "t1", "pytest tests/c.py::test_two", 2),
        _row("t2:0", "t2", "pytest tests/c.py::test_three", 2),
        _row("t2:1", "t2", "pytest tests/a.py::test_query", 0),
    ]

    _result, rows = run(fit, validation)

    assert [
        rows[index]["arms"]["target_overlap"]["provenance"]["latency"][
            "source"
        ]
        for index in range(3)
    ] == ["current", "current", "pytest_target_overlap"]
    assert rows[2]["arms"]["target_overlap"]["provenance"]["latency"][
        "support"
    ] == 2
    weighted = rows[3]["arms"]["target_overlap"]
    assert weighted["candidate_probability_by_bucket"]["latency"] == pytest.approx([
        0.6,
        0.0,
        0.4,
        0.0,
        0.0,
    ])
    assert weighted["candidate_probability_by_bucket"][
        "peak_cpu_cores"
    ] == pytest.approx([0.6, 0.0, 0.4])
    assert weighted["candidate"]["latency"] == 0
    assert weighted["candidate"]["peak_cpu_cores"] == "low"


def test_frozen_inputs_and_split_are_present() -> None:
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))

    assert FROZEN_FIT_ROWS.name == "rows.jsonl" and FROZEN_FIT_ROWS.is_file()
    assert FROZEN_VALIDATION_ROWS.name == "rows.jsonl" and FROZEN_VALIDATION_ROWS.is_file()
    assert len(split["development"]) == 100
    assert len(split["validation"]) == 50
    assert len(split["final_test"]) == 50
