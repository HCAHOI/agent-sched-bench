from dataclasses import replace

import pytest

from scripts.evaluation.evaluate_clause_latency_buckets import (
    _PipSemanticKB,
    PipExecEvent,
    _pip_contexts_by_call,
    _predict_pip_resources,
    _row_resource_value,
    evaluate_pip_resources,
    evaluate_pip_semantics,
)
from scripts.evaluation.evaluate_clause_resource_classes import CommandRow, Row
from tool_resource.clause_parser import parse_command_clauses
from tool_resource.pip_semantics import PipQueryState, PipTaskState, parse_pip_install
from tool_resource.runtime_kb import (
    CANONICAL_RESOURCE_HEAVY_THRESHOLDS,
    ClauseResourceKB,
)


def test_pip_signature_and_causal_task_state() -> None:
    left = parse_pip_install(
        ("python3", "-m", "pip", "install", "--break-system-packages", "pandas", "numpy")
    )
    right = parse_pip_install(
        ("python3", "-m", "pip", "install", "numpy", "pandas", "--break-system-packages")
    )
    assert left == right
    assert left is not None
    assert left.package_names == ("numpy", "pandas")
    assert parse_pip_install(("pip", "install", "-e", ".[dev]")) is not None
    assert parse_pip_install(("pip", "install", "-r", "requirements.txt")) is None
    assert parse_pip_install(("pytest", "-q")) is None

    state = PipTaskState()
    assert state.query(left).availability == "unknown"
    state.observe(
        "python3 -m pip install pandas numpy",
        "/usr/bin/python3: No module named pip\n\nExit code: 1",
    )
    assert state.query(left).availability == "absent"
    state.observe(
        "apt-get install -y python3-pip",
        "Setting up python3-pip\n\nExit code: 0",
    )
    assert state.query(left).availability == "present"
    state.observe(
        "python3 -m pip install numpy",
        "Successfully installed numpy-2.0\n\nExit code: 0",
    )
    assert state.query(left).remaining_packages == ("pandas",)
    state.observe(
        "python3 -m pip install pandas",
        "network failed\n\nExit code: 1",
    )
    assert state.query(left).remaining_packages == ("pandas",)
    assert PipTaskState().query(left).availability == "unknown"


def test_pip_evaluator_preserves_rows_and_task_boundaries() -> None:
    public = [
        Row("public__repo-1", "public__repo", 0, "python3", ("python3", "-m", "pip", "install", "alpha"), 1000.0, None, None, None),
        Row("public__repo-2", "public__repo", 1, "pytest", ("pytest", "-q"), 9000.0, None, None, None),
    ]
    task_ids = ["target__repo-1", "target__repo-2", "target__repo-3"]
    clauses = [
        Row(task_ids[0], "target__repo", 0, "python3", ("python3", "-m", "pip", "install", "alpha"), 1000.0, None, None, None),
        Row(task_ids[0], "target__repo", 0, "pytest", ("pytest", "-q"), 9000.0, None, None, None),
        Row(task_ids[1], "target__repo", 1, "apt-get", ("apt-get", "install", "python3-pip"), 1000.0, None, None, None),
        Row(task_ids[1], "target__repo", 1, "python3", ("python3", "-m", "pip", "install", "alpha"), 9000.0, None, None, None),
        Row(task_ids[2], "target__repo", 2, "apt-get", ("apt-get", "install", "python3-pip"), 1000.0, None, None, None),
        Row(task_ids[2], "target__repo", 2, "python3", ("python3", "-m", "pip", "install", "alpha", "beta"), 9000.0, None, None, None),
    ]
    commands = [
        CommandRow(task_ids[0], "target__repo", 0, 0, "call_pip_1", "python3 -m pip install alpha", 1000.0, (clauses[0],)),
        CommandRow(task_ids[0], "target__repo", 0, 1, "call_test", "pytest -q", 9000.0, (clauses[1],)),
        CommandRow(task_ids[1], "target__repo", 1, 0, "call_apt", "apt-get install python3-pip", 1000.0, (clauses[2],)),
        CommandRow(task_ids[1], "target__repo", 1, 1, "call_pip_2", "python3 -m pip install alpha", 9000.0, (clauses[3],)),
        CommandRow(task_ids[2], "target__repo", 2, 0, "call_compound", "apt-get install python3-pip && python3 -m pip install alpha beta", 9000.0, (clauses[4], clauses[5])),
    ]
    events = {
        task_ids[0]: [
            PipExecEvent("call_pip_1", commands[0].command, "Successfully installed alpha\nExit code: 0"),
            PipExecEvent("call_test", commands[1].command, "failed\nExit code: 1"),
        ],
        task_ids[1]: [
            PipExecEvent("call_apt", commands[2].command, "ok\nExit code: 0"),
            PipExecEvent("call_pip_2", commands[3].command, "Using cached alpha.whl\nExit code: 0"),
        ],
        task_ids[2]: [
            PipExecEvent("call_compound", commands[4].command, "Using cached alpha.whl\nExit code: 0"),
        ],
    }
    contexts = _pip_contexts_by_call(events[task_ids[2]])
    assert contexts["call_compound"][1].availability == "present"

    result, sidecar = evaluate_pip_semantics(
        public,
        task_ids,
        clauses,
        commands,
        events,
        {},
    )
    assert result["row_identity"]["nonpip_probability_vectors_bit_identical"]
    assert result["counts"]["pip_commands"] == 3
    assert result["counts"]["stored_target_pip_observations"] == 3
    compound = next(row for row in sidecar if row["call_id"] == "call_compound")
    assert (
        compound["arms"]["pip_semantic"]["probability_by_bucket"]
        != compound["arms"]["pip_semantic_state"]["probability_by_bucket"]
    )
    assert result["case_study"]["pip_execution_modes"]["success_cache_only"]["commands"] == 2
    assert len(sidecar) == len(commands)

    def with_resources(row: Row) -> Row:
        heavy = row.latency_ms > 8000.0
        return replace(
            row,
            peak_cpu_cores=3.0 if heavy else 1.0,
            sampled_peak_rss_mb=600.0 if heavy else 100.0,
            disk_read_write_bytes_total=200_000_000.0 if heavy else 1.0,
        )

    resource_public = [with_resources(row) for row in public]
    resource_clauses = [with_resources(row) for row in clauses]
    by_original_id = {
        id(original): updated
        for original, updated in zip(clauses, resource_clauses, strict=True)
    }
    resource_commands = [
        replace(
            command,
            clauses=tuple(by_original_id[id(clause)] for clause in command.clauses),
        )
        for command in commands
    ]
    result["status"] = "development_exposed_pip_latency_go"
    result["gates"]["latency"]["go"] = True
    result["gates"]["resource_evaluation"]["go"] = True
    resource_result, resource_sidecar = evaluate_pip_resources(
        resource_public,
        task_ids,
        resource_clauses,
        resource_commands,
        events,
        {},
        result,
    )
    assert resource_result["row_identity"][
        "nonpip_probability_vectors_bit_identical"
    ]
    assert len(resource_sidecar) == len(commands)
    assert all(
        metric["all"]["arms"]["current"]["eligible_n"] == len(commands)
        for metric in resource_result["resources"].values()
    )
    assert any(
        row["arms"]["pip_semantic"][resource]["probability_heavy"]
        != row["arms"]["pip_semantic_state"][resource]["probability_heavy"]
        for row in resource_sidecar
        for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS
    )
    with pytest.raises(ValueError, match="latency GO"):
        evaluate_pip_resources(
            resource_public,
            task_ids,
            resource_clauses,
            resource_commands,
            events,
            {"target_run_dir": "wrong"},
            result,
        )


def test_pip_resource_compound_physics_and_short_null() -> None:
    public = [
        Row("public__repo-1", "public__repo", 0, "python3", ("python3", "-m", "pip", "install", package), 1000.0, 1.1, 300.0, 60_000_000.0)
        for package in ("alpha", "beta")
    ]
    current = ClauseResourceKB.fit_public(
        row.observation(0.0, 1.0) for row in public
    )
    memories = {
        resource: _PipSemanticKB(
            [replace(row, latency_ms=getattr(row, resource)) for row in public]
        )
        for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS
    }

    def predict(command: str) -> dict[str, str | None]:
        parsed = parse_command_clauses(command)
        row = CommandRow(
            "target__repo-1",
            "target__repo",
            0,
            0,
            "call_compound",
            command,
            1000.0,
            tuple(public),
        )
        baseline = current.predict_command_resource_classes_from_clauses(
            row.repo,
            parsed["clauses"],
            3.0,
            command=command,
        )
        predictions, _diagnostics = _predict_pip_resources(
            memories,
            current,
            row,
            parsed,
            (PipQueryState("unknown", ("alpha",)), PipQueryState("unknown", ("beta",))),
            baseline.classifications,
            use_state=True,
        )
        return {
            resource: None if prediction is None else prediction.label
            for resource, prediction in predictions.items()
        }

    pipeline = predict(
        "python3 -m pip install alpha | python3 -m pip install beta"
    )
    sequential = predict(
        "python3 -m pip install alpha && python3 -m pip install beta"
    )
    assert pipeline == {
        "peak_cpu_cores": "heavy",
        "sampled_peak_rss_mb": "heavy",
        "disk_read_write_bytes_total": "heavy",
    }
    assert sequential == {
        "peak_cpu_cores": "light",
        "sampled_peak_rss_mb": "light",
        "disk_read_write_bytes_total": "heavy",
    }
    assert _row_resource_value(
        replace(public[0], latency_ms=100.0, sampled_peak_rss_mb=None),
        "sampled_peak_rss_mb",
    ) == 0.0
