from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.evaluation.evaluate_clause_resource_classes import CommandRow
from scripts.evaluation.evaluate_relational_agent_state import (
    _collapsed_outputs,
    _derive_state,
    _load_split_manifest,
    _select_primary_contrast,
    _validate_spans,
    validate_source,
)


SOURCE = '''
def scope(current_command, parsed_clauses):
    return "suite" if current_command == "run suite" else None

def blocker_spans(result_excerpt):
    spans = []
    for name in ("alpha", "beta", "gamma"):
        start = result_excerpt.find(name)
        if start >= 0:
            spans.append([start, start + len(name)])
    return sorted(spans)

def remediation_spans(command):
    spans = []
    for name in ("alpha", "beta", "gamma"):
        start = command.find(name)
        if start >= 0:
            spans.append([start, start + len(name)])
    return sorted(spans)
'''


def _namespace() -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(SOURCE, namespace)
    return namespace


def _event(index: int, command: str, exit_code: int, result: str = "") -> dict[str, object]:
    return {
        "event_index": index,
        "command": command,
        "exit_code": exit_code,
        "result_excerpt": result,
    }


def _state(*events: dict[str, object]) -> dict[str, object] | None:
    return _derive_state(
        {
            "current_command": "run suite",
            "parsed_clauses": {},
            "prior_events": list(events),
        },
        _namespace(),
    )


def test_host_derives_all_relational_states() -> None:
    failed_both = _event(0, "run suite", 1, "missing alpha and beta")
    install_alpha = _event(1, "install alpha", 0)

    assert _state(_event(0, "run suite", 1, "missing alpha"))["state"] == "blocked"
    assert _state(failed_both, install_alpha)["state"] == "partial_remediation"
    assert (
        _state(_event(0, "run suite", 1, "missing alpha"), install_alpha)["state"]
        == "closure_candidate"
    )
    assert (
        _state(
            _event(0, "run suite", 1, "missing alpha"),
            install_alpha,
            _event(2, "run suite", 0),
        )["state"]
        == "closure_verified"
    )
    surfaced = _state(
        _event(0, "run suite", 1, "missing alpha"),
        install_alpha,
        _event(2, "run suite", 1, "missing beta"),
    )
    assert surfaced is not None and surfaced["state"] == "newly_surfaced"
    resolved_new = _state(
        _event(0, "run suite", 1, "missing alpha and gamma"),
        install_alpha,
        _event(2, "run suite", 1, "missing beta"),
        _event(3, "install beta", 0),
    )
    assert resolved_new is not None and resolved_new["state"] == "partial_remediation"


def test_only_successful_exact_remediation_resolves() -> None:
    blocked = _event(0, "run suite", 1, "missing alpha")
    assert _state(blocked, _event(1, "install alpha", 1))["state"] == "blocked"
    assert _state(blocked, _event(1, "install gamma", 0))["state"] == "blocked"
    assert _state(_event(0, "other suite", 1, "missing alpha")) is None


def test_spans_and_source_fail_closed() -> None:
    assert _validate_spans([[8, 13]], "missing alpha", "test")[0]["identifier"] == "alpha"
    for spans in ([[8, 13], [8, 13]], [[8, 13], [14, 19]], [[-1, 2]], [[0, 99]]):
        with pytest.raises(ValueError):
            _validate_spans(spans, "missing alpha alpha", "test")
    with pytest.raises(ValueError):
        _validate_spans([[0, 5]], "a/b/c", "test")

    validate_source(SOURCE)
    with pytest.raises(ValueError):
        validate_source(SOURCE + "\nprint('side effect')\n")
    with pytest.raises(ValueError):
        validate_source(SOURCE.replace("def scope(current_command, parsed_clauses):", "@staticmethod\ndef scope(current_command, parsed_clauses):"))
    with pytest.raises(ValueError):
        validate_source(
            SOURCE.replace(
                'return "suite" if current_command == "run suite" else None',
                'blocker_spans.context = current_command\n    return "suite"',
            )
        )

    cache_channel = '''
def scope(current_command, parsed_clauses):
    re.compile(current_command)
    return "suite"

def blocker_spans(result_excerpt):
    return (
        [[result_excerpt.find("alpha"), sum((result_excerpt.find("alpha"), 5))]]
        if "run suite" in "{0._cache}".format(re)
        and result_excerpt.find("alpha") >= 0
        else []
    )

def remediation_spans(command):
    return []
'''
    validate_source(cache_channel)
    cache_namespace: dict[str, object] = {"re": re}
    exec(cache_channel, cache_namespace)
    assert (
        _derive_state(
            {
                "current_command": "run suite",
                "parsed_clauses": {},
                "prior_events": [_event(0, "run suite", 1, "missing alpha")],
            },
            cache_namespace,
        )
        is None
    )
    with pytest.raises(ValueError):
        validate_source(
            SOURCE.replace(
                'return "suite" if current_command == "run suite" else None',
                '__builtins__[0] = current_command\n    return "suite"',
            )
        )


def test_split_and_primary_contrast_are_mechanical(tmp_path: Path) -> None:
    manifest = {
        "development": [f"dev-{index}" for index in range(100)],
        "validation": [f"val-{index}" for index in range(50)],
        "final_test": [f"test-{index}" for index in range(50)],
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    loaded, digest = _load_split_manifest(path)
    assert loaded == manifest and len(digest) == 64
    manifest["final_test"][0] = manifest["validation"][0]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        _load_split_manifest(path)

    commands = [
        CommandRow(f"task-{index}", "repo", index, 0, f"call-{index}", "cmd", 1.0, ())
        for index in range(10)
    ]
    outputs = {
        f"task-{index}:0": {
            "rule_id": "rule",
            "state": "left" if index < 5 else "right",
            "evidence_event_indices": [0],
        }
        for index in range(10)
    }
    pmfs = {
        (("rule", "left"), "latency"): (1.0, 0.0, 0.0, 0.0, 0.0),
        (("rule", "right"), "latency"): (0.0, 1.0, 0.0, 0.0, 0.0),
    }
    primary, state_tasks = _select_primary_contrast(commands, outputs, pmfs)
    assert primary == {
        "rule_id": "rule",
        "states": ["left", "right"],
        "target": "latency",
        "task_support": [5, 5],
    }
    assert len(state_tasks["rule::left"]) == 5
    assert all(
        value is None or value["state"] == "__collapsed__"
        for value in _collapsed_outputs(outputs).values()
    )
