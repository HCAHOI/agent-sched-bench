import pytest

from scripts.evaluation.evaluate_clause_latency_buckets import evaluate_pytest_semantics
from scripts.evaluation.evaluate_clause_resource_classes import CommandRow, Row
from tool_resource.pytest_semantics import is_pytest_invocation, parse_pytest


def test_pytest_signature_normalizes_work_shape_and_fails_closed() -> None:
    direct = parse_pytest(
        (
            "pytest",
            "tests/test_a.py::test_one",
            "tests/test_b.py",
            "-x",
            "-n",
            "auto",
            "--dist=loadscope",
            "-k",
            "slow and not network",
            "-m=unit or smoke",
            "-q",
        )
    )
    module = parse_pytest(
        (
            "/usr/bin/python3",
            "-m",
            "pytest",
            "other/test_x.py",
            "other/test_y.py::test_two",
            "--maxfail=1",
            "--numprocesses=auto",
            "--dist",
            "loadscope",
            "-k",
            "fast and not remote",
            "-m",
            "core or quick",
            "--no-header",
        )
    )
    assert direct == module
    assert direct is not None
    assert direct.target_shapes == (("file", 1), ("nodeid", 1))
    assert direct.k_shape == ("ATOM", "and", "not", "ATOM")
    assert direct.k_atoms == 2
    assert direct.m_shape == ("ATOM", "or", "ATOM")
    assert direct.m_atoms == 2

    assert is_pytest_invocation(("python", "-m", "pytest", "tests"))
    assert parse_pytest(("pytest", "--collect-only", "tests")) is not None
    assert parse_pytest(("pytest", "--sw-skip", "-kfoo", "-ra", "tests")) is not None
    assert parse_pytest(("pytest", "--dist", "--", "tests")) is None
    assert parse_pytest(("pytest", "-k", "--unknown", "tests")) is None
    assert parse_pytest(("pytest", "--tb", "--", "tests")) is None
    assert parse_pytest(("pytest", "--dist=project-specific", "tests")) is None
    assert parse_pytest(("pytest", "--unknown-plugin-option", "tests")) is None
    assert parse_pytest(("pytest", "--", "-literal-target")) is None
    assert parse_pytest(("python", "-m", "unittest")) is None


def test_pytest_evaluator_uses_only_settled_prior_tasks() -> None:
    public = [
        Row("public__repo-1", "public__repo", 0, "pytest", ("pytest", "tests"), 9000.0, None, None, None),
        Row("public__repo-2", "public__repo", 1, "echo", ("echo", "ok"), 1000.0, None, None, None),
    ]
    task_ids = ["target__repo-1", "target__repo-2", "target__repo-3"]
    clauses = [
        Row(task_ids[0], "target__repo", 0, "pytest", ("pytest", "tests/test_a.py", "-q"), 1000.0, None, None, None),
        Row(task_ids[1], "target__repo", 1, "pytest", ("pytest", "other/test_b.py", "-v"), 9000.0, None, None, None),
        Row(task_ids[1], "target__repo", 1, "echo", ("echo", "ok"), 1000.0, None, None, None),
        Row(task_ids[2], "target__repo", 2, "echo", ("echo", "ok"), 1000.0, None, None, None),
    ]
    commands = [
        CommandRow(task_ids[0], "target__repo", 0, 0, "call_1", "pytest tests/test_a.py -q", 1000.0, (clauses[0],)),
        CommandRow(task_ids[1], "target__repo", 1, 0, "call_2", "pytest other/test_b.py -v && echo ok", 9000.0, (clauses[1], clauses[2])),
        CommandRow(task_ids[2], "target__repo", 2, 0, "call_3", "echo ok", 1000.0, (clauses[3],)),
    ]
    result, sidecar = evaluate_pytest_semantics(
        public,
        task_ids,
        clauses,
        commands,
        {},
    )
    assert result["counts"]["nonexact_carrier_commands"] == 1
    assert result["row_identity"]["nonpytest_probability_vectors_bit_identical"]
    assert result["gates"]["coverage"]["pass"] is False
    assert sidecar[0]["carrier"] is False
    assert sidecar[1]["carrier"] is True
    assert sidecar[1]["clauses"][0]["contributing_task_ids"] == [task_ids[0]]
    with pytest.raises(ValueError, match="frozen pre-label gate"):
        evaluate_pytest_semantics(
            public,
            task_ids,
            clauses,
            commands,
            {},
            expected_coverage=(398, 339, 88, 125),
        )
