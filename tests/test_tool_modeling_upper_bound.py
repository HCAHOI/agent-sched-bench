from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.evaluation.evaluate_tool_modeling_upper_bound import (
    _materialize_candidates,
    build_oracle_rows,
    summarize,
)
from scripts.evaluation.evaluate_command_history_residual import Row as HistoryRow


TARGETS = (
    "latency",
    "peak_cpu_cores",
    "sampled_peak_rss_mb",
    "disk_read_write_bytes_total",
)


def test_cli_starts_from_the_repository_root() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluation/evaluate_tool_modeling_upper_bound.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[2],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--profile-scored-tasks" in completed.stdout


def _pmfs(latency_bucket: int) -> dict[str, list[float]]:
    latency = [0.0] * 5
    latency[latency_bucket] = 1.0
    return {
        "latency": latency,
        "peak_cpu_cores": [1.0, 0.0, 0.0],
        "sampled_peak_rss_mb": [1.0, 0.0, 0.0],
        "disk_read_write_bytes_total": [1.0, 0.0, 0.0],
    }


def _predictions(latency_bucket: int) -> dict[str, int | str]:
    return {
        "latency": latency_bucket,
        "peak_cpu_cores": "low",
        "sampled_peak_rss_mb": "low",
        "disk_read_write_bytes_total": "low",
    }


def _baseline(
    command: str,
    *,
    sample_id: str = "task-1:0",
    task_bucket: int = 0,
    clause_bucket: int = 0,
) -> dict:
    return {
        "sample_id": sample_id,
        "task_id": sample_id.partition(":")[0],
        "command": command,
        "labels": {target: 0 for target in TARGETS} | {"latency": 1},
        "arms": {
            "task_aware": {
                "prediction": _predictions(task_bucket),
                "probability_by_bucket": _pmfs(task_bucket),
                "provenance": {},
            },
            "clause_kb": {
                "prediction": _predictions(clause_bucket),
                "probability_by_bucket": _pmfs(clause_bucket),
                "provenance": {},
            },
        },
    }


def _candidate_rows(baseline: dict, *, exact_bucket: int = 1) -> tuple[dict, dict]:
    shared = {
        key: deepcopy(baseline[key])
        for key in ("sample_id", "task_id", "command", "labels")
    }
    exact = {
        "candidate": _predictions(exact_bucket),
        "candidate_probability_by_bucket": _pmfs(exact_bucket),
        "provenance": {
            target: {"source": "exact", "support": 1}
            for target in TARGETS
        },
    }
    pytest_row = {
        **shared,
        "arms": {
            "exact": exact,
            "collapsed_signature": deepcopy(exact),
            "target_overlap": deepcopy(exact),
        },
    }
    semantic_row = {
        **shared,
        "arms": {"semantic_work_units": deepcopy(exact)},
    }
    return pytest_row, semantic_row


def test_existing_candidate_and_perfect_tool_oracles_use_only_their_allowed_pmfs() -> None:
    baseline = _baseline("pytest tests/a.py")
    pytest_row, semantic_row = _candidate_rows(baseline)

    row = build_oracle_rows([baseline], [pytest_row], [semantic_row])[0]

    assert row["tools"] == ["pytest"]
    assert row["arms"]["static_candidate_oracle"]["prediction"]["latency"] == 1
    assert row["arms"]["static_candidate_oracle"]["probability_by_bucket"][
        "latency"
    ] == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert row["arms"]["static_candidate_oracle"]["provenance"]["latency"][
        "selected_source"
    ] == "exact_complete_command"
    assert row["static_candidates"]["latency"][0]["source"] == "clause_kb"
    assert row["arms"]["perfect_tool_oracle"]["prediction"]["latency"] == 1
    assert row["arms"]["perfect_tool_oracle"]["probability_by_bucket"][
        "latency"
    ] == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert row["arms"]["static_candidate_oracle"]["prediction"][
        "peak_cpu_cores"
    ] == "low"


def test_non_tool_rows_are_bit_identical_to_task_aware() -> None:
    baseline = _baseline("git status")
    pytest_row, semantic_row = _candidate_rows(baseline)

    row = build_oracle_rows([baseline], [pytest_row], [semantic_row])[0]

    assert row["tools"] == []
    assert row["arms"]["static_candidate_oracle"] == row["arms"]["task_aware"]
    assert row["arms"]["perfect_tool_oracle"] == row["arms"]["task_aware"]


def test_pip_overlap_is_an_allowed_static_candidate() -> None:
    baseline = _baseline("python -m pip install sqlparse")
    pytest_row, semantic_row = _candidate_rows(baseline, exact_bucket=0)
    semantic_row["arms"]["semantic_work_units"]["candidate"] = _predictions(1)
    semantic_row["arms"]["semantic_work_units"][
        "candidate_probability_by_bucket"
    ] = _pmfs(1)
    semantic_row["arms"]["semantic_work_units"]["provenance"]["latency"] = {
        "source": "pip_package_overlap",
        "support": 2,
    }

    row = build_oracle_rows([baseline], [pytest_row], [semantic_row])[0]

    assert row["tools"] == ["pip"]
    assert row["arms"]["static_candidate_oracle"]["prediction"]["latency"] == 1
    assert row["arms"]["static_candidate_oracle"]["provenance"]["latency"][
        "selected_source"
    ] == "pip_package_overlap"


def test_component_identity_rejects_a_candidate_from_another_task() -> None:
    baseline = _baseline("pytest tests/a.py")
    pytest_row, semantic_row = _candidate_rows(baseline)
    pytest_row["sample_id"] = "later-task:0"
    pytest_row["task_id"] = "later-task"

    with pytest.raises(ValueError, match="component rows disagree"):
        build_oracle_rows([baseline], [pytest_row], [semantic_row])


def test_unrelated_fallback_cannot_enter_the_exact_memory_pool() -> None:
    baseline = _baseline("python -m pip install sqlparse")
    pytest_row, semantic_row = _candidate_rows(baseline)
    pytest_row["arms"]["exact"]["provenance"] = {
        target: {"source": "unrelated_fallback", "support": 9}
        for target in TARGETS
    }

    row = build_oracle_rows([baseline], [pytest_row], [semantic_row])[0]

    assert all(
        candidate["source"] != "exact_complete_command"
        for candidate in row["static_candidates"]["latency"]
    )
    assert row["arms"]["static_candidate_oracle"]["prediction"]["latency"] == 0


def test_candidate_materialization_updates_only_after_task_settlement() -> None:
    labels = {target: 0 for target in TARGETS} | {"latency": 1}
    current = _predictions(0)
    pmfs = {target: tuple(value) for target, value in _pmfs(0).items()}
    fit = [HistoryRow("fit:0", "fit", "git status", labels, current, pmfs)]
    validation = [
        HistoryRow("first:0", "first", "pytest tests/a.py", labels, current, pmfs),
        HistoryRow(
            "first:1",
            "first",
            "pytest tests/a.py::test_x",
            labels,
            current,
            pmfs,
        ),
        HistoryRow(
            "later:0",
            "later",
            "pytest tests/a.py::test_y",
            labels,
            current,
            pmfs,
        ),
    ]

    pytest_rows, _semantic_rows = _materialize_candidates(fit, validation)

    assert pytest_rows[0]["arms"]["target_overlap"]["provenance"]["latency"][
        "source"
    ] == "current"
    assert pytest_rows[1]["arms"]["target_overlap"]["provenance"]["latency"][
        "source"
    ] == "current"
    assert pytest_rows[2]["arms"]["target_overlap"]["provenance"]["latency"][
        "source"
    ] == "pytest_target_overlap"


def test_summary_applies_the_frozen_coverage_and_materiality_gate() -> None:
    baselines = []
    pytest_rows = []
    semantic_rows = []
    for index in range(20):
        task_id = f"task-{index // 2}"
        baseline = _baseline(
            "pytest tests/a.py",
            sample_id=f"{task_id}:{index % 2}",
        )
        pytest_row, semantic_row = _candidate_rows(baseline, exact_bucket=0)
        for arm in ("exact", "collapsed_signature"):
            pytest_row["arms"][arm]["provenance"] = {
                target: {"source": "current", "support": 0} for target in TARGETS
            }
        pytest_row["arms"]["target_overlap"]["candidate"] = _predictions(1)
        pytest_row["arms"]["target_overlap"][
            "candidate_probability_by_bucket"
        ] = _pmfs(1)
        pytest_row["arms"]["target_overlap"]["provenance"] = {
            target: {"source": "pytest_target_overlap", "support": 2}
            for target in TARGETS
        }
        baselines.append(baseline)
        pytest_rows.append(pytest_row)
        semantic_rows.append(semantic_row)

    rows = build_oracle_rows(baselines, pytest_rows, semantic_rows)
    result = summarize(rows)

    assert result["coverage"]["pytest"] == {
        "commands": 20,
        "tasks": 10,
        "non_exact_commands": 20,
        "non_exact_tasks": 10,
        "distinct_semantic_candidate_commands": 20,
        "gate_pass": True,
    }
    assert result["comparisons"]["static_candidate_oracle"][
        "macro_gain_percentage_points"
    ] == pytest.approx(25.0)
    assert result["comparisons"]["static_candidate_oracle"]["materially_positive"] is True
    assert result["tool_decisions"]["pytest"] == "keep_static_family"


def test_uncovered_tool_cannot_determine_a_covered_tools_verdict() -> None:
    baselines = []
    pytest_rows = []
    semantic_rows = []
    for index in range(20):
        baseline = _baseline(
            "pytest tests/a.py",
            sample_id=f"pytest-{index // 2}:{index % 2}",
            task_bucket=1,
        )
        pytest_row, semantic_row = _candidate_rows(baseline, exact_bucket=0)
        for arm in ("exact", "collapsed_signature"):
            pytest_row["arms"][arm]["provenance"] = {
                target: {"source": "current", "support": 0} for target in TARGETS
            }
        pytest_row["arms"]["target_overlap"]["provenance"] = {
            target: {"source": "pytest_target_overlap", "support": 2}
            for target in TARGETS
        }
        baselines.append(baseline)
        pytest_rows.append(pytest_row)
        semantic_rows.append(semantic_row)
    for index in range(10):
        baseline = _baseline(
            "python -m pip install sqlparse",
            sample_id=f"pip-{index}:0",
        )
        pytest_row, semantic_row = _candidate_rows(baseline, exact_bucket=0)
        pytest_row["arms"]["exact"]["provenance"] = {
            target: {"source": "current", "support": 0} for target in TARGETS
        }
        semantic_row["arms"]["semantic_work_units"]["candidate"] = _predictions(1)
        semantic_row["arms"]["semantic_work_units"][
            "candidate_probability_by_bucket"
        ] = _pmfs(1)
        semantic_row["arms"]["semantic_work_units"]["provenance"] = {
            target: {"source": "pip_package_overlap", "support": 2}
            for target in TARGETS
        }
        baselines.append(baseline)
        pytest_rows.append(pytest_row)
        semantic_rows.append(semantic_row)

    result = summarize(build_oracle_rows(baselines, pytest_rows, semantic_rows))

    assert result["coverage"]["pip"]["gate_pass"] is False
    assert result["tool_decisions"]["pip"] == "uninformative_coverage"
    assert result["tool_decisions"]["pytest"] == "close_tool_specific_modeling_on_sqlglot"
    assert result["decision"] == "mixed_by_tool"


def test_coverage_uses_all_scored_tasks_not_only_non_exact_tasks() -> None:
    baselines = []
    pytest_rows = []
    semantic_rows = []
    for index in range(20):
        baseline = _baseline(
            "pytest tests/a.py",
            sample_id=f"nonexact-{index % 9}:{index}",
        )
        pytest_row, semantic_row = _candidate_rows(baseline, exact_bucket=0)
        for arm in ("exact", "collapsed_signature"):
            pytest_row["arms"][arm]["provenance"] = {
                target: {"source": "current", "support": 0} for target in TARGETS
            }
        pytest_row["arms"]["target_overlap"]["candidate"] = _predictions(1)
        pytest_row["arms"]["target_overlap"][
            "candidate_probability_by_bucket"
        ] = _pmfs(1)
        pytest_row["arms"]["target_overlap"]["provenance"] = {
            target: {"source": "pytest_target_overlap", "support": 2}
            for target in TARGETS
        }
        baselines.append(baseline)
        pytest_rows.append(pytest_row)
        semantic_rows.append(semantic_row)
    exact_baseline = _baseline("pytest tests/a.py", sample_id="exact-task:0")
    exact_pytest, exact_semantic = _candidate_rows(exact_baseline)
    baselines.append(exact_baseline)
    pytest_rows.append(exact_pytest)
    semantic_rows.append(exact_semantic)

    result = summarize(build_oracle_rows(baselines, pytest_rows, semantic_rows))

    assert result["coverage"]["pytest"]["non_exact_tasks"] == 9
    assert result["coverage"]["pytest"]["tasks"] == 10
    assert result["coverage"]["pytest"]["gate_pass"] is True
