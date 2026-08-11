import pytest

import scripts.evaluation.evaluate_pennylane_xdist_rss_positive_control as evaluation
from scripts.evaluation.evaluate_pennylane_xdist_rss_positive_control import (
    fit_clause_kb,
    max_bucket_convolution,
    parse_pytest_workers,
    pytest_scope,
    scaled_rss_pmf,
)
from scripts.evaluation.evaluate_clause_resource_classes import CommandRow, Row


def test_worker_parser_and_rss_composition() -> None:
    assert parse_pytest_workers(("pytest", "-n4")) == 4
    assert parse_pytest_workers(("/opt/env/bin/python", "-m", "pytest", "--numprocesses=auto")) == 8
    assert parse_pytest_workers(("python3", "-m", "pytest", "-q")) == "serial"
    assert parse_pytest_workers(("grep", "-n", "text")) is None
    assert parse_pytest_workers(("pytest", "-n", "4", "-n8")) == "invalid"
    assert pytest_scope(("pytest", "-n4")) == "full"
    assert pytest_scope(("pytest", "tests", "-n4")) == "broad"
    assert pytest_scope(("pytest", "tests/test_one.py", "-n4")) == "narrow"
    assert pytest_scope(("pytest", "--unknown", "-n4")) is None
    assert scaled_rss_pmf((100.0, 600.0), 4) == (0.5, 0.0, 0.5)
    assert max_bucket_convolution((0.5, 0.5, 0.0), (0.5, 0.0, 0.5)) == (
        0.25,
        0.25,
        0.5,
    )


def test_fit_evidence_is_repository_local_at_first_replay_query() -> None:
    row = Row(
        task_id="owner__repo-1",
        repo="owner/repo",
        manifest_index=0,
        bin="python",
        argv=("python", "-m", "pytest", "-q"),
        latency_ms=100.0,
        peak_cpu_cores=1.0,
        sampled_peak_rss_mb=100.0,
        disk_read_write_bytes_total=0.0,
    )

    prediction = fit_clause_kb((row,)).predict_command_resource_buckets(
        "owner/repo", "python -m pytest -q", 3.0
    ).classifications["sampled_peak_rss_mb"]

    assert prediction is not None
    assert prediction.scope == "repo"
    assert prediction.key_kind == "exact_clause"


def test_coverage_failure_precedes_label_access(monkeypatch) -> None:
    fit = Row(
        task_id="owner__repo-1",
        repo="owner/repo",
        manifest_index=0,
        bin="pytest",
        argv=("pytest", "tests"),
        latency_ms=100.0,
        peak_cpu_cores=1.0,
        sampled_peak_rss_mb=100.0,
        disk_read_write_bytes_total=0.0,
    )
    carrier = Row(
        task_id="replay",
        repo="owner/repo",
        manifest_index=1,
        bin="pytest",
        argv=("pytest", "tests", "-n4"),
        latency_ms=100.0,
        peak_cpu_cores=1.0,
        sampled_peak_rss_mb=100.0,
        disk_read_write_bytes_total=0.0,
    )
    command = CommandRow(
        "replay", "owner/repo", 1, 0, "call", "pytest tests -n4", 100.0, (carrier,)
    )
    monkeypatch.setattr(evaluation, "_EXPECTED_TASK_IDS", ("replay",))
    monkeypatch.setattr(evaluation, "_clean_head", lambda: "head")
    monkeypatch.setattr(
        evaluation,
        "_load_tasks",
        lambda: ([fit], {"replay": []}, {"replay": [command]}),
    )
    monkeypatch.setattr(
        evaluation,
        "command_resource_bucket_label",
        lambda *_args: pytest.fail("label read before coverage gate"),
    )

    with pytest.raises(ValueError, match="coverage misses"):
        evaluation.run()
