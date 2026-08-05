from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.evaluation.evaluate_clause_latency_buckets import (
    CANONICAL_RESOURCE_BUCKET_EDGES,
    RESOURCE_BUCKET_LABELS,
    CommandRow,
)
from scripts.evaluation.evaluate_clause_resource_classes import Row
from scripts.evaluation.evaluate_declarative_relational_state import (
    MAX_COMMAND_CHARS,
    GENERATION_PROMPT,
    GENERATION_SCHEMA,
    SPLIT_MANIFEST,
    SpecError,
    _fit_fingerprint,
    _fresh_gate,
    _freeze,
    _frozen_baseline,
    _load_split_manifest,
    _json_bytes,
    _validate_host_identity,
    _validate_generation_binding,
    _validation_authorizes_final,
    _compile_pattern,
    extract_query,
    scope_only,
    validate_development_support,
    validate_spec,
)
from scripts.evaluation.evaluate_offline_agent_extractor import (
    TARGETS,
    _apply_candidate,
    _fit_pmfs,
)
from tool_resource.runtime_kb import parse_command_clauses


def _response(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "abstain": False,
        "rule_id": "suite_blocker",
        "scope_patterns": [r"run\s{1,8}suite"],
        "blocker_patterns": [r"missing\s{1,8}(?P<id>[a-z][a-z0-9_-]{1,20})"],
        "remediation_patterns": [r"install\s{1,8}(?P<id>[a-z][a-z0-9_-]{1,20})"],
        "explanation": "Track exact blocker identifiers across verifier attempts.",
    }
    value.update(changes)
    return value


def _event(index: int, command: str, exit_code: int, result: str = "") -> dict[str, object]:
    return {
        "event_index": index,
        "command": command,
        "exit_code": exit_code,
        "result_excerpt": result,
    }


def _query(*events: dict[str, object]) -> dict[str, object]:
    return {
        "current_command": "run suite",
        "parsed_clauses": parse_command_clauses("run suite"),
        "prior_events": list(events),
    }


def test_declarative_spec_drives_host_relations_and_ablations() -> None:
    spec = validate_spec(_response())
    assert spec is not None
    failed = _event(0, "run suite", 1, "missing alpha and missing beta")
    installed = _event(1, "install alpha", 0)

    graph, bounded = extract_query(spec, _query(failed, installed))
    assert not bounded and graph is not None
    assert graph["state"] == "partial_remediation"
    assert graph["blockers"][0]["identifier"] == "alpha"
    assert graph["remediations"][0]["addresses"] == [[0, [8, 13]]]
    assert scope_only(spec, _query()) == {
        "rule_id": "suite_blocker",
        "state": "__scope_only__",
        "evidence_event_indices": [],
    }

    closed, _ = extract_query(
        spec,
        _query(
            _event(0, "run suite", 1, "missing alpha"),
            _event(1, "install alpha", 0),
            _event(2, "run suite", 0),
        ),
    )
    surfaced, _ = extract_query(
        spec,
        _query(
            _event(0, "run suite", 1, "missing alpha"),
            _event(1, "install alpha", 0),
            _event(2, "run suite", 1, "missing beta"),
        ),
    )
    assert closed is not None and closed["state"] == "closure_verified"
    assert surfaced is not None and surfaced["state"] == "newly_surfaced"
    blocked, _ = extract_query(
        spec, _query(_event(0, "run suite", 1, "missing alpha"))
    )
    candidate, _ = extract_query(
        spec,
        _query(
            _event(0, "run suite", 1, "missing alpha"),
            _event(1, "install alpha", 0),
        ),
    )
    assert blocked is not None and blocked["state"] == "blocked"
    assert candidate is not None and candidate["state"] == "closure_candidate"


def test_schema_and_regex_adversaries_fail_closed() -> None:
    assert validate_spec(
        _response(
            abstain=True,
            rule_id="",
            scope_patterns=[],
            blocker_patterns=[],
            remediation_patterns=[],
        )
    ) is None
    for change in (
        {"extra": True},
        {"rule_id": "Bad Rule"},
        {"scope_patterns": []},
        {"scope_patterns": [r"run\s{1,8}suite"] * 2},
        {"explanation": ""},
    ):
        with pytest.raises(SpecError):
            validate_spec(_response(**change))

    for pattern in (
        r"missing (?P<id>a|b)",
        r"missing (?P<id>.+)",
        r"missing (?P<id>[a-z]+)",
        r"missing (?P<id>(?:[a-z]{1,8}))",
        r"missing (?P<id>[a-z]{1,129})",
        r"(?i)missing (?P<id>[a-z]{1,8})",
        r"missing (?P<id>[a-z]{0,8})",
        r"missing (?P<id>[a-z]{1,8}) (?P<other>[a-z]{1,8})",
        "missing (?P<id>\\N{LATIN SMALL LETTER E WITH ACUTE})",
        "a{0,128}" * 12 + "b",
    ):
        with pytest.raises(SpecError):
            _compile_pattern(
                pattern,
                kind="scope" if pattern.endswith("b") else "blocker",
            )
    assert _compile_pattern(r"missing\|(?P<id>[a-z]{2,8})", kind="blocker")
    with pytest.raises(SpecError):
        validate_spec(
            _response(scope_patterns=[r"\x73\x71\x6c\x67\x6c\x6f\x74"]),
            forbidden_ids=("tobymao__sqlglot-1234",),
        )
    for obfuscated in (
        r"[sS][qQ][lL][gG][lL][oO][tT]",
        r"s\Bq\Bl\Bg\Bl\Bo\Bt",
        r"s{1,2}q{1,2}l{1,2}g{1,2}l{1,2}o{1,2}t{1,2}",
    ):
        with pytest.raises(SpecError):
            validate_spec(
                _response(scope_patterns=[obfuscated]),
                forbidden_ids=("tobymao__sqlglot-1234",),
            )


def test_bounds_and_identifier_spans_fall_back() -> None:
    spec = validate_spec(_response())
    assert spec is not None
    long_query = _query()
    long_query["current_command"] = "x" * (MAX_COMMAND_CHARS + 1)
    assert extract_query(spec, long_query) == (None, True)

    too_many = " ".join(f"missing a{index}" for index in range(17))
    assert extract_query(spec, _query(_event(0, "run suite", 1, too_many))) == (
        None,
        True,
    )
    path_spec = validate_spec(
        _response(
            blocker_patterns=[r"missing\s{1,8}(?P<id>[a-z/]{2,20})"],
        )
    )
    assert path_spec is not None
    assert extract_query(
        path_spec, _query(_event(0, "run suite", 1, "missing a/b"))
    ) == (None, True)

    overflow_spec = validate_spec(
        _response(
            blocker_patterns=[
                rf"p{prefix}(?P<id>{letter}[a-z]{{1}}\d{{2}})"
                for prefix, letter in enumerate("abcde")
            ]
        )
    )
    assert overflow_spec is not None
    blockers = " ".join(
        f"p{prefix}{letter}a{index:02d}"
        for prefix, letter in enumerate("abcde")
        for index in range(13)
    )
    assert len(blockers) <= 500
    assert extract_query(
        overflow_spec, _query(_event(0, "run suite", 1, blockers))
    ) == (None, True)


def test_development_support_matches_verifier_semantics() -> None:
    spec = validate_spec(_response())
    assert spec is not None
    queries = {}
    task_by_sample = {}
    for index in range(5):
        sample = f"task-{index}:0"
        queries[sample] = _query(
            _event(0, "other suite", 1, f"missing ignored{index}"),
            _event(1, "run suite", 1, f"missing item{index}"),
            _event(2, f"install item{index}", 0),
        )
        task_by_sample[sample] = f"task-{index}"
    report = validate_development_support(spec, queries, task_by_sample)
    assert report["scope"][0]["tasks"] == 5
    assert report["blocker"][0]["identifiers"] == 5
    assert report["remediation"][0]["identifiers"] == 5

    specific_spec = validate_spec(
        _response(
            blocker_patterns=[
                r"item0\s{1,8}missing\s{1,8}(?P<id>[a-z][a-z0-9]{1,20})"
            ]
        )
    )
    assert specific_spec is not None
    specific_queries = {}
    for index in range(5):
        specific_queries[f"task-{index}:0"] = _query(
            _event(0, "run suite", 1, f"item0 missing error{index}"),
            _event(1, "pip install item0", 0),
            _event(2, f"pip install fix{index}", 0),
        )
    with pytest.raises(SpecError, match="specific positional literal"):
        validate_development_support(spec=specific_spec, queries=specific_queries, task_by_sample=task_by_sample)


def test_fit_identity_current_fallback_and_fresh_gate() -> None:
    command = CommandRow("task", "repo", 0, 0, "call", "run suite", 1.0, ())
    changed = CommandRow("task", "repo", 0, 0, "call", "run suite", 2.0, ())
    assert _fit_fingerprint(["task"], [], [command]) != _fit_fingerprint(
        ["task"], [], [changed]
    )
    fit_tasks = [f"task-{index}" for index in range(6)]
    clauses = [
        Row(task, "repo", index, "run", ("run", "suite"), 1.0, 0.0, 0.0, 0.0)
        for index, task in enumerate(fit_tasks)
    ]
    fit_commands = [
        CommandRow(
            task,
            "repo",
            index,
            0,
            f"call-{index}",
            "run suite",
            1.0,
            (clauses[index],),
        )
        for index, task in enumerate(fit_tasks)
    ]
    fit_commands[-1] = CommandRow(
        fit_tasks[-1],
        "repo",
        5,
        0,
        "call-5",
        "run suite",
        1_000_000.0,
        (clauses[-1],),
    )
    outputs = {
        f"{task}:0": {"rule_id": "rule", "state": "fit", "evidence_event_indices": []}
        for task in fit_tasks
    }
    pmfs, _support = _fit_pmfs(
        fit_tasks, fit_commands, outputs, warmup_tasks=5
    )
    assert pmfs[(('rule', 'fit'), 'latency')] == (1.0, 0.0, 0.0, 0.0, 0.0)

    current = {
        "latency": 2,
        **{target: RESOURCE_BUCKET_LABELS[1] for target in CANONICAL_RESOURCE_BUCKET_EDGES},
    }
    probabilities = {
        "latency": [0.0, 0.0, 1.0, 0.0, 0.0],
        **{
            target: [0.0, 1.0, 0.0]
            for target in CANONICAL_RESOURCE_BUCKET_EDGES
        },
    }
    applied = _apply_candidate(
        [
            {
                "sample_id": "task:0",
                "current_dynamic": {**current, "probability_by_bucket": probabilities},
                "latency_label": 2,
                "resource_labels": {
                    target: 1 for target in CANONICAL_RESOURCE_BUCKET_EDGES
                },
            }
        ],
        {"task:0": None},
        {},
    )[0]
    assert applied["candidate"] == current
    assert applied["candidate_probability_by_bucket"] == probabilities

    frozen, frozen_rows = _frozen_baseline(
        {
            "latency": {"current_dynamic": "online", "frozen_at_80": "frozen"},
            "resources": {
                target: {"current_dynamic": "online", "frozen_at_80": "frozen"}
                for target in CANONICAL_RESOURCE_BUCKET_EDGES
            },
        },
        [{"current_dynamic": "online", "frozen_at_80": {"latency": 1}}],
    )
    assert frozen["latency"]["current_dynamic"] == "frozen"
    assert frozen_rows[0]["current_dynamic"] == {"latency": 1}

    baseline = {
        "latency": {"current_dynamic": {"severe_underprediction_rate": 0.5}},
        "resources": {
            target: {"current_dynamic": {"severe_underprediction_rate": 0.5}}
            for target in CANONICAL_RESOURCE_BUCKET_EDGES
        },
    }

    def score(accuracy: float, severe: float) -> dict[str, object]:
        return {
            "metrics": {
                target: {
                    "exact_class_accuracy" if target == "latency" else "accuracy": accuracy,
                    "severe_underprediction_rate": severe,
                }
                for target in TARGETS
            },
            "gate": {
                "no_accuracy_regression": True,
                "no_severe_underprediction_regression": True,
                "helpful": 4,
                "harmful": 0,
                "helpful_tasks": 3,
            },
        }

    signature = {"rule_id": "rule", "state": "left"}
    labels = {target: 1 for target in TARGETS}
    relational_row = {
        "generated_signature": signature,
        "labels": labels,
        "candidate": {
            target: 1 if target == "latency" else RESOURCE_BUCKET_LABELS[1]
            for target in TARGETS
        },
    }
    ablation_row = {
        **relational_row,
        "candidate": {
            target: 0 if target == "latency" else RESOURCE_BUCKET_LABELS[0]
            for target in TARGETS
        },
    }
    gate = _fresh_gate(
        baseline,
        score(0.8, 0.4),
        score(0.8, 0.4),
        score(0.8, 0.4),
        [relational_row],
        [ablation_row],
        [ablation_row],
        {"rule_id": "rule", "states": ["left", "right"], "target": "latency"},
    )
    assert gate["go"]


def test_final_authorization_and_host_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    validation = {
        "schema": "declarative-relational-fresh-evaluation-v1",
        "role": "validation",
        "status": "validation_go",
        "claim_bearing": False,
        "artifact_sha256": "artifact",
        "split_manifest_sha256": "split",
        "labels_scored": True,
        "coverage": {"passed": True},
        "gate": {"go": True},
        "row_identity": {"identical_rows": True},
        "rows_sha256": "a" * 64,
    }
    assert _validation_authorizes_final(validation, "artifact", "split")
    validation["gate"] = {"go": False}
    assert not _validation_authorizes_final(validation, "artifact", "split")

    class Result:
        stdout = ""
        returncode = 0

    def changed_host(*arguments: str) -> Result:
        result = Result()
        if arguments[0] == "diff":
            result.returncode = 1
        return result

    monkeypatch.setattr(
        "scripts.evaluation.evaluate_declarative_relational_state._git", changed_host
    )
    with pytest.raises(ValueError, match="differs"):
        _validate_host_identity(
            {"commit": "a" * 40, "paths": ["scripts/evaluation", "src", "pyproject.toml", "uv.lock"]}
        )


def test_transcript_split_and_run_path_are_bound(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _response()
    evidence = {"selected_rows": [{"sample_id": "S00001"}], "causal_prefixes": {}}
    prompt = GENERATION_PROMPT + json.dumps(evidence, separators=(",", ":"))
    (tmp_path / "generation.prompt.txt").write_text(prompt)
    (tmp_path / "generation.schema.json").write_text(json.dumps(GENERATION_SCHEMA))
    (tmp_path / "generation.response.json").write_text(json.dumps(response))
    artifact = {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "generation": response,
        "development_evidence_sha256": hashlib.sha256(_json_bytes(evidence)).hexdigest(),
        "development_selected_sample_ids": ["S00001"],
    }
    _validate_generation_binding(tmp_path, artifact, response)
    with pytest.raises(ValueError, match="model response"):
        _validate_generation_binding(tmp_path, artifact, _response(rule_id="changed"))

    manifest = {
        "development": [f"dev-{index}" for index in range(100)],
        "validation": [f"val-{index}" for index in range(50)],
        "final_test": [f"final-{index}" for index in range(50)],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    _load_split_manifest(manifest_path)
    manifest["final_test"][0] = manifest["validation"][0]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disjointness"):
        _load_split_manifest(manifest_path)

    monkeypatch.setattr(
        "scripts.evaluation.evaluate_declarative_relational_state._committed_file",
        lambda _path: (b"", "a" * 40),
    )
    monkeypatch.setattr(
        "scripts.evaluation.evaluate_declarative_relational_state._load_split_manifest",
        lambda _path: ({"development_run": str(tmp_path / "good")}, "split"),
    )
    monkeypatch.setattr(
        "scripts.evaluation.evaluate_declarative_relational_state.load_run_rows",
        lambda _path: pytest.fail("load_run_rows ran before path rejection"),
    )
    with pytest.raises(ValueError, match="development run differs"):
        _freeze(
            SimpleNamespace(
                out_dir=tmp_path / "out",
                split_manifest=SPLIT_MANIFEST,
                run_dir=tmp_path / "wrong",
            )
        )
