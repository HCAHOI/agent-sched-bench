from __future__ import annotations

import hashlib
import json

import pytest

from scripts.evaluation.evaluate_clause_latency_buckets import CommandRow, PipExecEvent
from scripts.evaluation.evaluate_clause_resource_classes import Row
from scripts.evaluation.evaluate_declarative_family_state import (
    EVALUATION_SCHEMA,
    GENERATION_PROMPT,
    GENERATION_SCHEMA,
    MAX_COMMAND_CHARS,
    MAX_PRIOR_EVENTS,
    BoundExceeded,
    SpecError,
    _collapsed_outputs,
    _coverage,
    _fresh_gate,
    _json_bytes,
    _paired_correctness,
    _trim,
    build_evidence,
    build_generation_input,
    extract_query,
    invocation_key,
    match_scope,
    select_primary_contrasts,
    _validate_generation_binding,
    _validation_authorizes_final,
    validate_development_support,
    validate_spec,
)
from scripts.evaluation.evaluate_offline_agent_extractor import _fit_pmfs
from tool_resource.runtime_kb import parse_command_clauses


def _response(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "abstain": False,
        "family_id": "suite",
        "scopes": [
            {"scope_id": "full", "patterns": [r"run\s{1,8}suite"]},
            {
                "scope_id": "targeted",
                "patterns": [r"run\s{1,8}suite\s{1,8}[a-z][a-z0-9]{1,8}"],
            },
        ],
        "blocker_patterns": [r"missing\s{1,8}(?P<id>[a-z][a-z0-9]{1,20})"],
        "remediation_patterns": [r"install\s{1,8}(?P<id>[a-z][a-z0-9]{1,20})"],
        "explanation": "Relate named blockers across targeted and full suite runs.",
    }
    value.update(changes)
    return value


def _event(
    index: int, command: str, exit_code: int, result: str = ""
) -> dict[str, object]:
    return {
        "event_index": index,
        "command": command,
        "exit_code": exit_code,
        "result_excerpt": result,
    }


def _query(*events: dict[str, object], command: str = "run suite") -> dict[str, object]:
    return {
        "current_command": command,
        "parsed_clauses": parse_command_clauses(command),
        "prior_events": list(events),
    }


def _raw(call: str, command: str, exit_code: int, result: str = "") -> PipExecEvent:
    return PipExecEvent(call, command, f"{result}\nExit code: {exit_code}")


def test_family_verifier_crosses_scope_but_keeps_current_scope() -> None:
    spec = validate_spec(_response())
    assert spec is not None
    failed = _event(0, "run suite item0", 1, "missing alpha")
    installed = _event(1, "install alpha", 0)

    candidate, bounded = extract_query(spec, _query(failed, installed))
    assert not bounded and candidate is not None
    assert candidate["scope_id"] == "full"
    assert candidate["dependency_state"] == "closure_candidate"

    verified, _ = extract_query(
        spec,
        _query(failed, installed, _event(2, "run suite item0", 0)),
    )
    assert verified is not None
    assert verified["state"] == "full::closure_verified"

    partial, _ = extract_query(
        spec,
        _query(
            _event(0, "run suite item0", 1, "missing alpha missing beta"),
            installed,
        ),
    )
    surfaced, _ = extract_query(
        spec,
        _query(
            failed,
            installed,
            _event(2, "run suite item0", 1, "missing beta"),
        ),
    )
    assert partial is not None and partial["dependency_state"] == "partial_remediation"
    assert surfaced is not None and surfaced["dependency_state"] == "newly_surfaced"


def test_schema_and_scope_overlap_fail_closed() -> None:
    assert (
        validate_spec(
            _response(
                abstain=True,
                family_id="",
                scopes=[],
                blocker_patterns=[],
                remediation_patterns=[],
            )
        )
        is None
    )
    for change in (
        {"scopes": [_response()["scopes"][0]]},
        {"family_id": "Bad Family"},
        {"blocker_patterns": [r"missing (?P<id>[a-z]{1,8})|lost"]},
    ):
        try:
            validate_spec(_response(**change))
        except SpecError:
            pass
        else:
            raise AssertionError(
                "invalid declarative family specification was accepted"
            )

    overlap = validate_spec(
        _response(
            scopes=[
                {"scope_id": "one", "patterns": [r"run\s{1,8}suite"]},
                {"scope_id": "two", "patterns": [r"run[\t ]{1,8}suite"]},
            ]
        )
    )
    assert overlap is not None
    try:
        match_scope(overlap, "run suite")
    except SpecError:
        pass
    else:
        raise AssertionError("overlapping scopes were accepted")

    for response in (
        _response(explanation="A SQLGlot-specific suite rule."),
        _response(
            scopes=[
                {"scope_id": "full", "patterns": [r"sqlglot\s{1,8}test"]},
                _response()["scopes"][1],
            ]
        ),
        _response(
            scopes=[
                {"scope_id": "full", "patterns": [r"sql{1}glot\s{1,8}test"]},
                _response()["scopes"][1],
            ]
        ),
        _response(explanation="Use task alias T000."),
    ):
        with pytest.raises(SpecError, match="opaque"):
            validate_spec(response, ("tobymao__sqlglot-3785", "T000"))


def test_query_and_span_bounds_fail_closed() -> None:
    spec = validate_spec(_response())
    assert spec is not None
    value, bounded = extract_query(spec, _query(command="x" * (MAX_COMMAND_CHARS + 1)))
    assert value is None and bounded

    blockers = " ".join(f"missing item{index}" for index in range(17))
    value, bounded = extract_query(
        spec, _query(_event(0, "run suite item0", 1, blockers))
    )
    assert value is None and bounded


def test_label_free_evidence_is_deterministic_and_bounded() -> None:
    task_ids = [f"task-{index}" for index in range(18)]
    events = {}
    for index, task in enumerate(task_ids):
        runner = "pytest" if index < 13 else "python -m unittest"
        middle = (
            [_raw(f"{task}-{step}", f"echo {task} {step}", 0) for step in range(1, 51)]
            if index == 0
            else [_raw(f"{task}-1", f"install {task} item{index}", 0)]
        )
        events[task] = [
            _raw(f"{task}-0", runner, 1, f"missing {task} item{index}"),
            *middle,
            _raw(f"{task}-last", runner, 0),
        ]
    evidence = build_evidence(task_ids, events)
    assert evidence == build_evidence(task_ids, events)
    assert [row["invocation_key"] for row in evidence["candidates"]] == [
        "pytest",
        "unittest",
    ]
    assert evidence["candidates"][0]["qualifying_task_count"] == 13
    assert len(evidence["candidates"][0]["episodes"]) == 12
    long_episode = evidence["candidates"][0]["episodes"][0]["events"]
    assert len(long_episode) == 48
    assert [row["event_index"] for row in long_episode][23:25] == [23, 28]
    assert not any(task.casefold() in str(evidence).casefold() for task in task_ids)
    assert "T000" in str(evidence)
    assert invocation_key("/opt/python -m pytest -q") == "pytest"
    assert len(_trim("x" * 1_000, 800)) == 800
    assert "\n<omitted>\n" in _trim("x" * 1_000, 800)


def test_generation_input_rejects_insufficient_candidates_before_call() -> None:
    tasks = [f"task-{index}" for index in range(5)]
    events = {
        task: [_raw(f"{task}-0", "pytest", 1), _raw(f"{task}-1", "pytest", 0)]
        for task in tasks
    }
    with pytest.raises(SpecError, match="fewer than two"):
        build_generation_input(tasks, events)


def test_development_support_uses_both_scopes_and_relation_ids() -> None:
    spec = validate_spec(_response())
    assert spec is not None
    tasks = [f"task-{index}" for index in range(5)]
    queries = {
        task: _query(
            _event(0, f"run suite item{index}", 1, f"missing item{index}"),
            _event(1, f"install item{index}", 0),
        )
        for index, task in enumerate(tasks)
    }
    report = validate_development_support(spec, queries, {task: task for task in tasks})
    assert report["scopes"]["full"]["tasks"] == 5
    assert report["scopes"]["targeted"]["tasks"] == 5
    assert report["blocker"][0]["identifiers"] == 5
    assert report["remediation"][0]["identifiers"] == 5

    oversized = {
        "oversized": _query(
            *[_event(index, "true", 0) for index in range(MAX_PRIOR_EVENTS + 1)]
        )
    }
    with pytest.raises(BoundExceeded):
        validate_development_support(spec, oversized, {"oversized": "oversized"})


def test_primary_contrasts_require_relation_and_scope_modes() -> None:
    rows = []
    outputs = {}
    cells = (
        ("full", "blocked"),
        ("full", "closure_verified"),
        ("targeted", "blocked"),
    )
    for cell_index, (scope, state) in enumerate(cells):
        for index in range(5):
            task = f"task-{cell_index}-{index}"
            rows.append(CommandRow(task, "repo", len(rows), 0, task, "run", 1.0, ()))
            outputs[f"{task}:0"] = {
                "rule_id": "suite",
                "state": f"{scope}::{state}",
                "scope_id": scope,
                "dependency_state": state,
                "evidence_event_indices": [0],
            }
    pmfs = {
        (("suite", "full::blocked"), "latency"): (1.0, 0.0, 0.0, 0.0, 0.0),
        (("suite", "full::closure_verified"), "latency"): (0.0, 0.0, 1.0, 0.0, 0.0),
        (("suite", "targeted::blocked"), "latency"): (0.0, 1.0, 0.0, 0.0, 0.0),
    }
    contrasts = select_primary_contrasts(rows, outputs, pmfs)
    assert contrasts is not None
    assert contrasts["relation"]["scope_id"] == "full"
    assert contrasts["relation"]["states"] == ["blocked", "closure_verified"]
    assert contrasts["scope"]["dependency_state"] == "blocked"
    assert contrasts["scope"]["scopes"] == ["full", "targeted"]


def test_collapsed_arms_keep_identical_carrier_and_fit_separately() -> None:
    tasks = [f"task-{index}" for index in range(5)]
    rows = [
        CommandRow(
            task,
            "repo",
            index,
            0,
            task,
            "run",
            1_000.0,
            (
                Row(
                    task,
                    "repo",
                    index,
                    "run",
                    ("run",),
                    1_000.0,
                    1.0,
                    100.0,
                    0.0,
                ),
            ),
        )
        for index, task in enumerate(tasks)
    ]
    full = {
        f"{task}:0": {
            "rule_id": "suite",
            "state": "full::blocked",
            "scope_id": "full",
            "dependency_state": "blocked",
            "evidence_event_indices": [0],
        }
        for task in tasks
    }
    full["absent:0"] = None
    family = _collapsed_outputs(full, dimension="family")
    scope = _collapsed_outputs(full, dimension="scope")
    assert (
        {key for key, value in full.items() if value is not None}
        == {key for key, value in family.items() if value is not None}
        == {key for key, value in scope.items() if value is not None}
    )
    full.pop("absent:0")
    family.pop("absent:0")
    scope.pop("absent:0")
    full_pmfs, _ = _fit_pmfs(tasks, rows, full, warmup_tasks=5)
    family_pmfs, _ = _fit_pmfs(tasks, rows, family, warmup_tasks=5)
    scope_pmfs, _ = _fit_pmfs(tasks, rows, scope, warmup_tasks=5)
    assert (("suite", "full::blocked"), "latency") in full_pmfs
    assert (("suite", "__family__"), "latency") in family_pmfs
    assert (("suite", "scope::full"), "latency") in scope_pmfs


def test_fresh_coverage_requires_both_frozen_contrasts() -> None:
    primary = {
        "relation": {
            "family_id": "suite",
            "scope_id": "full",
            "states": ["blocked", "closure_verified"],
            "target": "latency",
        },
        "scope": {
            "family_id": "suite",
            "dependency_state": "blocked",
            "scopes": ["full", "targeted"],
            "target": "latency",
        },
    }
    cells = [
        ("full", "blocked"),
        ("full", "closure_verified"),
        ("targeted", "blocked"),
        ("targeted", "closure_verified"),
    ]
    rows = []
    outputs = []
    for cell_index, (scope, state) in enumerate(cells):
        for index in range(5):
            task = f"task-{cell_index}-{index % 3}"
            rows.append(
                CommandRow(
                    task, "repo", len(rows), index, str(len(rows)), "run", 0.0, ()
                )
            )
            outputs.append(
                {
                    "rule_id": "suite",
                    "state": f"{scope}::{state}",
                    "scope_id": scope,
                    "dependency_state": state,
                }
            )
    assert _coverage(rows, outputs, primary, 0)["passed"]
    assert not _coverage(rows, outputs, primary, 1)["passed"]


def test_paired_requirements_and_final_authorization() -> None:
    def row(state: str, truth: int, prediction: int) -> dict[str, object]:
        return {
            "generated_signature": {"rule_id": "suite", "state": state},
            "labels": {"latency": truth},
            "candidate": {"latency": prediction},
        }

    selectors = [row("full::blocked", 0, 0), row("full::closure_verified", 1, 1)]
    worse = [row("full::blocked", 0, 1), row("full::closure_verified", 1, 0)]
    pair = _paired_correctness(
        selectors,
        selectors,
        worse,
        {
            "family_id": "suite",
            "scope_id": "full",
            "states": ["blocked", "closure_verified"],
            "target": "latency",
        },
        kind="relation",
    )
    assert pair["strictly_better"] and pair["commands"] == 2
    scope_selectors = [row("full::blocked", 0, 0), row("targeted::blocked", 1, 1)]
    scope_pair = _paired_correctness(
        scope_selectors,
        scope_selectors,
        worse,
        {
            "family_id": "suite",
            "dependency_state": "blocked",
            "scopes": ["full", "targeted"],
            "target": "latency",
        },
        kind="scope",
    )
    assert scope_pair["strictly_better"] and scope_pair["commands"] == 2

    validation = {
        "schema": EVALUATION_SCHEMA,
        "role": "validation",
        "status": "validation_go",
        "claim_bearing": False,
        "artifact_sha256": "a" * 64,
        "split_manifest_sha256": "b" * 64,
        "labels_scored": True,
        "coverage": {"passed": True},
        "gate": {"go": True},
        "row_identity": {"identical_rows": True},
        "rows_sha256": "c" * 64,
    }
    assert _validation_authorizes_final(validation, "a" * 64, "b" * 64)
    validation["gate"] = {"go": False}
    assert not _validation_authorizes_final(validation, "a" * 64, "b" * 64)


def test_fresh_gate_requires_relation_and_scope_contributions() -> None:
    baseline = {
        "latency": {"current_dynamic": {"severe_underprediction_rate": 0.5}},
        "resources": {
            target: {"current_dynamic": {"severe_underprediction_rate": 0.5}}
            for target in (
                "peak_cpu_cores",
                "sampled_peak_rss_mb",
                "disk_read_write_bytes_total",
            )
        },
    }

    def score(severe: dict[str, float]) -> dict[str, object]:
        return {
            "metrics": {
                target: {
                    "exact_class_accuracy" if target == "latency" else "accuracy": 1.0,
                    "severe_underprediction_rate": severe[target],
                }
                for target in (
                    "latency",
                    "peak_cpu_cores",
                    "sampled_peak_rss_mb",
                    "disk_read_write_bytes_total",
                )
            },
            "gate": {
                "no_accuracy_regression": True,
                "no_severe_underprediction_regression": True,
                "helpful": 4,
                "harmful": 0,
                "helpful_tasks": 3,
            },
        }

    severe = {
        "latency": 0.4,
        "peak_cpu_cores": 0.4,
        "sampled_peak_rss_mb": 0.5,
        "disk_read_write_bytes_total": 0.5,
    }
    full_rows = [
        {
            "generated_signature": {"rule_id": "suite", "state": state},
            "labels": {"latency": truth},
            "candidate": {"latency": prediction},
        }
        for state, truth, prediction in (
            ("full::blocked", 0, 0),
            ("full::closure_verified", 1, 1),
            ("targeted::blocked", 1, 1),
        )
    ]

    def predictions(values: list[int]) -> list[dict[str, object]]:
        return [{"candidate": {"latency": value}} for value in values]

    family_rows = predictions([1, 0, 0])
    scope_rows = predictions([1, 0, 1])
    primary = {
        "relation": {
            "family_id": "suite",
            "scope_id": "full",
            "states": ["blocked", "closure_verified"],
            "target": "latency",
        },
        "scope": {
            "family_id": "suite",
            "dependency_state": "blocked",
            "scopes": ["full", "targeted"],
            "target": "latency",
        },
    }
    passing = _fresh_gate(
        baseline,
        score(severe),
        score(severe),
        score(severe),
        full_rows,
        family_rows,
        scope_rows,
        primary,
    )
    assert passing["go"]
    failing = _fresh_gate(
        baseline,
        score(severe),
        score(severe),
        score(severe),
        full_rows,
        family_rows,
        family_rows,
        primary,
    )
    assert not failing["go"] and not failing["scope_pair"]["strictly_better"]


def test_generation_transcript_binding(tmp_path) -> None:
    response = _response()
    evidence = {"schema": "label-free-family-episodes-v1", "candidates": []}
    prompt = GENERATION_PROMPT + json.dumps(evidence, separators=(",", ":"))
    (tmp_path / "generation.prompt.txt").write_text(prompt)
    (tmp_path / "generation.schema.json").write_text(json.dumps(GENERATION_SCHEMA))
    (tmp_path / "generation.response.json").write_text(json.dumps(response))
    artifact = {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "development_evidence_sha256": hashlib.sha256(
            _json_bytes(evidence)
        ).hexdigest(),
        "generation": response,
        "cost": {"prompt_bytes": len(prompt.encode())},
    }
    _validate_generation_binding(tmp_path, artifact, response, ("task-0",))
    artifact["cost"] = {"prompt_bytes": 0}
    with pytest.raises(ValueError, match="differs"):
        _validate_generation_binding(tmp_path, artifact, response, ("task-0",))
