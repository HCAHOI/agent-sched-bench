from __future__ import annotations

import pytest

import scripts.evaluation.evaluate_typed_catalog_state as typed
from scripts.evaluation.evaluate_clause_latency_buckets import PipExecEvent
from scripts.evaluation.evaluate_typed_catalog_state import (
    CatalogConfig,
    SpecError,
    TokenTemplate,
    _comparison_changes,
    _config_json,
    _occurrences,
    build_catalog,
    build_generation_input,
    extract_query,
    match_scope,
    scope_shape,
    template_spans,
    validate_selection,
)
from scripts.evaluation.evaluate_offline_agent_extractor import TARGETS
from tool_resource.runtime_kb import parse_command_clauses


def _raw(call: str, command: str, exit_code: int, result: str = "") -> PipExecEvent:
    return PipExecEvent(call, command, f"{result}\nExit code: {exit_code}")


def _config(identifier: str = "C004") -> CatalogConfig:
    return CatalogConfig(
        identifier,
        "unittest",
        ("no-target", "dotted"),
        (("no-target", 12), ("dotted", 5)),
        TokenTemplate("whole", ("module", "named")),
        TokenTemplate("whole", ("break-system-packages",)),
        7,
        2,
    )


def _event(index: int, command: str, exit_code: int, result: str = "") -> dict[str, object]:
    return {
        "event_index": index,
        "command": command,
        "exit_code": exit_code,
        "result_excerpt": result,
    }


def _query(*events: dict[str, object], command: str = "python -m unittest") -> dict[str, object]:
    return {
        "current_command": command,
        "parsed_clauses": parse_command_clauses(command),
        "prior_events": list(events),
    }


def test_templates_enumerate_contexts_and_exact_suffix_spans() -> None:
    text = "pip install --break-system-packages python3_foo"
    occurrences = _occurrences(text, suffixes=True)
    assert (
        TokenTemplate("whole", ("break-system-packages",)),
        "python3-foo",
        (36, 47),
    ) in occurrences
    suffix = TokenTemplate("suffix", ("install", "break-system-packages"))
    assert (suffix, "foo", (44, 47)) in occurrences
    assert template_spans(suffix, text) == [[44, 47]]
    trailing = "pip install python3-alpha-"
    trailing_suffix = TokenTemplate("suffix", ("pip", "install"))
    assert template_spans(trailing_suffix, trailing) == [[20, 25]]
    assert not any(template.capture_mode == "suffix" for template, _id, _span in _occurrences(text, suffixes=False))


def test_scope_requires_invocation_and_uses_frozen_shape_precedence() -> None:
    config = _config()
    assert scope_shape("python -m unittest", "unittest") == "no-target"
    assert scope_shape("python -m unittest suite.case", "unittest") == "dotted"
    assert scope_shape("pytest a.py b/c::case", "pytest") == "multi-nodeid+python-file"
    assert match_scope(config, "python -m unittest") == "no-target"
    assert match_scope(config, "pytest") is None
    assert match_scope(config, "echo") is None


def test_catalog_is_deterministic_causal_and_scope_restricted() -> None:
    tasks = [f"task-{index}" for index in range(5)]
    events = {}
    for index, task in enumerate(tasks):
        dependency = "alpha" if index < 3 else "beta"
        events[task] = [
            _raw(f"{task}-0", "python -m unittest", 1, f"No module named {dependency}"),
            _raw(
                f"{task}-1",
                f"pip install --break-system-packages {dependency}",
                0,
            ),
            _raw(f"{task}-2", "python -m unittest suite.case", 0),
        ]
    catalog = build_catalog(tasks, events)
    assert catalog == build_catalog(tasks, events)
    assert len(catalog) == 4
    assert all(config.invocation_key == "unittest" for config in catalog)
    assert all(config.causal_task_count == 5 for config in catalog)
    assert all(config.identifier_count == 2 for config in catalog)
    assert catalog[0].configuration_id == "C000"

    unrelated = {
        task: [
            _raw(f"{task}-0", "python -m unittest unknown extra", 1, "No module named alpha"),
            _raw(f"{task}-1", "pip install --break-system-packages alpha", 0),
            _raw(f"{task}-2", "python -m unittest suite.case", 0),
        ]
        for task in tasks
    }
    assert build_catalog(tasks, unrelated) == ()


def test_generation_is_finite_choice_and_validates_abstention() -> None:
    catalog = (_config("C000"), _config("C001"))
    payload, prompt, schema = build_generation_input(catalog)
    assert [row["configuration_id"] for row in payload["configurations"]] == ["C000", "C001"]
    assert schema["properties"]["configuration_id"]["enum"] == ["", "C000", "C001"]
    assert "captured_identifiers" in prompt and "alpha" not in prompt
    assert validate_selection(
        {"abstain": False, "configuration_id": "C001", "explanation": "Install changes state."},
        catalog,
    ) == "C001"
    assert validate_selection(
        {"abstain": True, "configuration_id": "", "explanation": "No credible relation."},
        catalog,
    ) is None
    with pytest.raises(SpecError):
        validate_selection(
            {"abstain": False, "configuration_id": "C999", "explanation": "Unknown."},
            catalog,
        )
    assert _config_json(catalog[0])["blocker_template"]["context_tokens"] == ["module", "named"]


def test_runtime_derives_all_five_states_and_empty_history_is_null() -> None:
    config = _config()
    alpha = _event(0, "python -m unittest", 1, "No module named alpha")
    install = _event(1, "pip install --break-system-packages alpha", 0)

    blocked, bounded = extract_query(config, _query(alpha))
    candidate, _ = extract_query(config, _query(alpha, install))
    partial, _ = extract_query(
        config,
        _query(
            _event(0, "python -m unittest", 1, "No module named alpha No module named beta"),
            install,
        ),
    )
    surfaced, _ = extract_query(
        config,
        _query(alpha, install, _event(2, "python -m unittest", 1, "No module named beta")),
    )
    verified, _ = extract_query(
        config,
        _query(alpha, install, _event(2, "python -m unittest", 0)),
    )
    empty, empty_bounded = extract_query(config, _query())

    assert not bounded and blocked is not None and blocked["dependency_state"] == "blocked"
    assert candidate is not None and candidate["dependency_state"] == "closure_candidate"
    assert partial is not None and partial["dependency_state"] == "partial_remediation"
    assert surfaced is not None and surfaced["dependency_state"] == "newly_surfaced"
    assert verified is not None and verified["dependency_state"] == "closure_verified"
    assert empty is None and not empty_bounded


def test_support_only_comparison_counts_helpful_harmful_and_identical() -> None:
    def row(task: str, candidate: int, truth: int) -> dict[str, object]:
        return {
            "task_id": task,
            "candidate": {
                "latency": candidate,
                **{target: "Light" for target in TARGETS if target != "latency"},
            },
            "labels": {
                "latency": truth,
                **{target: 0 for target in TARGETS if target != "latency"},
            },
        }

    agent = [row("a", 2, 2), row("b", 0, 1), row("c", 1, 0)]
    support = [row("a", 1, 2), row("b", 0, 1), row("c", 0, 0)]
    changes = _comparison_changes(agent, support)
    assert changes["changed"] == 2
    assert changes["helpful"] == 1
    assert changes["harmful"] == 1
    assert changes["helpful_tasks"] == 1
    assert _comparison_changes(support, support)["changed"] == 0


def test_fresh_gate_requires_nonidentical_safe_gain_over_support_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(typed, "_family_fresh_gate", lambda *_args: {"go": True})

    def score(accuracy: float, severe: float) -> dict[str, object]:
        return {
            "metrics": {
                target: {
                    "exact_class_accuracy" if target == "latency" else "accuracy": accuracy,
                    "severe_underprediction_rate": severe,
                }
                for target in TARGETS
            }
        }

    def row(task: str, latency: int, truth: int) -> dict[str, object]:
        return {
            "task_id": task,
            "candidate": {
                "latency": latency,
                **{target: "Light" for target in TARGETS if target != "latency"},
            },
            "labels": {
                "latency": truth,
                **{target: 0 for target in TARGETS if target != "latency"},
            },
        }

    agent_rows = [row(task, 1, 1) for task in ("a", "b", "c")]
    support_rows = [row(task, 0, 1) for task in ("a", "b", "c")]
    passed = typed._fresh_gate(
        {},
        score(0.8, 0.1),
        {},
        {},
        score(0.8, 0.1),
        agent_rows,
        [],
        [],
        support_rows,
        {},
    )
    assert passed["go"]

    identical = typed._fresh_gate(
        {},
        score(0.8, 0.1),
        {},
        {},
        score(0.8, 0.1),
        support_rows,
        [],
        [],
        support_rows,
        {},
    )
    assert not identical["go"]

    unsafe = typed._fresh_gate(
        {},
        score(0.7, 0.2),
        {},
        {},
        score(0.8, 0.1),
        agent_rows,
        [],
        [],
        support_rows,
        {},
    )
    assert not unsafe["go"]
