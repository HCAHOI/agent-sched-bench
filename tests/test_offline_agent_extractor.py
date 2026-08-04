from __future__ import annotations

import pytest

from scripts.evaluation.evaluate_clause_latency_buckets import PipExecEvent
from scripts.evaluation.evaluate_clause_resource_classes import CommandRow, Row
from scripts.evaluation.evaluate_offline_agent_extractor import (
    _catalog_rows,
    _excerpt,
    _fit_pmfs,
    _validated_usage,
    build_catalog,
    build_queries,
    run_source,
    validate_source,
    validate_source_semantics,
)


def _command(task: int, call: int, *, latency_ms: float = 1_000.0) -> CommandRow:
    task_id = f"target__repo-{task}"
    clause = Row(
        task_id=task_id,
        repo="target__repo",
        manifest_index=task,
        bin="runner",
        argv=("runner", "test"),
        latency_ms=latency_ms,
        peak_cpu_cores=1.0,
        sampled_peak_rss_mb=100.0,
        disk_read_write_bytes_total=0.0,
    )
    return CommandRow(
        task_id=task_id,
        repo="target__repo",
        manifest_index=task,
        call_index=call,
        call_id=f"call-{task}-{call}",
        command="runner test",
        duration_ms=latency_ms,
        clauses=(clause,),
    )


def test_generic_queries_expose_only_causal_prefix() -> None:
    commands = [_command(0, 0), _command(0, 1)]
    events = {
        "target__repo-0": [
            PipExecEvent("call-0-0", "runner test", "missing\nExit code: 1", 0.0, 1.0),
            PipExecEvent("call-0-1", "runner test", "passed\nExit code: 0", 2.0, 3.0),
        ]
    }
    queries = build_queries(["target__repo-0"], commands, events)

    assert queries["target__repo-0:0"]["prior_events"] == []
    assert queries["target__repo-0:1"]["prior_events"] == [
        {
            "event_index": 0,
            "command": "runner test",
            "exit_code": 1,
            "result_excerpt": "missing\nExit code: 1",
        }
    ]
    assert "passed" not in str(queries["target__repo-0:1"])


def test_generated_source_is_restricted_and_replayable() -> None:
    source = """
def extract(query):
    for event in reversed(query.get("prior_events", [])):
        if event.get("exit_code") != 0:
            return {
                "rule_id": "prior_failure",
                "state": "blocked",
                "evidence_event_indices": [event["event_index"]],
            }
    return None
"""
    query = {
        "current_command": "runner test",
        "parsed_clauses": {},
        "prior_events": [
            {
                "event_index": 0,
                "command": "runner test",
                "exit_code": 1,
                "result_excerpt": "missing",
            }
        ],
    }
    outputs, durations = run_source(source, [query])

    assert outputs == [
        {
            "rule_id": "prior_failure",
            "state": "blocked",
            "evidence_event_indices": [0],
        }
    ]
    assert durations[0] > 0
    with pytest.raises(ValueError, match="forbidden Import"):
        validate_source("import os\ndef extract(query): return None")
    with pytest.raises(ValueError, match="opaque training ID"):
        validate_source("def extract(query): return 'S00001'", ["S00001"])
    with pytest.raises(ValueError, match="file-specific"):
        validate_source('def extract(query): return "tests/test_x.py"')
    with pytest.raises(ValueError, match="file-specific"):
        validate_source('def extract(query): return "/tmp/cache"')
    with pytest.raises(ValueError, match="assigns a matching literal"):
        validate_source(
            'def extract(query):\n needle = "numpy"\n return needle in query["current_command"]'
        )
    assert len(_excerpt("x" * 600)) == 500


def test_generated_source_semantics_reject_count_and_specific_literal() -> None:
    samples = [f"task-{index}:0" for index in range(5)]
    queries = {
        sample: {
            "current_command": "runner install numpy",
            "parsed_clauses": {
                "clauses": [{"argv": ["runner", "install", "numpy"]}]
            },
            "prior_events": [
                {
                    "event_index": 0,
                    "command": "runner probe",
                    "exit_code": 1,
                    "result_excerpt": "missing dependency",
                }
            ],
        }
        for sample in samples
    }
    task_by_sample = {sample: sample.split(":", 1)[0] for sample in samples}
    count_value = {
        "rule_id": "count",
        "state": "later",
        "evidence_event_indices": [0],
    }
    actual = dict.fromkeys(samples, count_value)
    empty = dict.fromkeys(samples)
    neutral = dict(actual)
    with pytest.raises(ValueError, match="history count alone"):
        validate_source_semantics(
            "def extract(query): return None",
            queries,
            task_by_sample,
            actual,
            empty,
            neutral,
        )
    package_source = '''
def extract(query):
    if "numpy" in query["current_command"]:
        return {"rule_id": "scope", "state": "match", "evidence_event_indices": []}
    return None
'''
    with pytest.raises(ValueError, match="package/test-specific"):
        validate_source_semantics(
            package_source,
            queries,
            task_by_sample,
            empty,
            empty,
            empty,
        )
    for bypass_source in (
        '''
def extract(query):
    if any(pattern in query["current_command"] for pattern in ("numpy", "runner")):
        return {"rule_id": "scope", "state": "match", "evidence_event_indices": []}
    return None
''',
        '''
def package_name():
    return "numpy"
def extract(query):
    if package_name() in query["current_command"]:
        return {"rule_id": "scope", "state": "match", "evidence_event_indices": []}
    return None
''',
        '''
def scope():
    return {"rule_id": "numpy", "state": "helper", "evidence_event_indices": []}
def extract(query):
    if scope()["rule_id"] in query["current_command"]:
        return {"rule_id": "scope", "state": "match", "evidence_event_indices": []}
    return None
''',
    ):
        with pytest.raises(
            ValueError, match="package/test-specific|lacks five-task support"
        ):
            validate_source_semantics(
                bypass_source,
                queries,
                task_by_sample,
                empty,
                empty,
                empty,
            )

    regex_source = '''
def extract(query):
    if re.fullmatch(r"runner\\s+install\\s+\\w+", query["current_command"]):
        return {"rule_id": "scope", "state": "match", "evidence_event_indices": []}
    return None
'''
    validate_source_semantics(
        regex_source,
        queries,
        task_by_sample,
        empty,
        empty,
        empty,
    )
    specific_regex = regex_source.replace(r"\w+", "numpy")
    with pytest.raises(
        ValueError, match="regex (contains|depends on) package/test"
    ):
        validate_source_semantics(
            specific_regex,
            queries,
            task_by_sample,
            empty,
            empty,
            empty,
        )
    obfuscated_regex = regex_source.replace(r"\w+", r"num(py)")
    with pytest.raises(ValueError, match="depends on package/test/file"):
        validate_source_semantics(
            obfuscated_regex,
            queries,
            task_by_sample,
            empty,
            empty,
            empty,
        )


def test_token_usage_is_required() -> None:
    usage = {"input_tokens": 10, "cached_input_tokens": 2, "output_tokens": 3}
    assert _validated_usage([{"usage": usage}], "fixture") == usage
    with pytest.raises(ValueError, match="exactly one"):
        _validated_usage([], "fixture")


def test_catalog_is_opaque_and_signature_support_is_task_counted() -> None:
    commands = [_command(task, 0, latency_ms=40_000.0) for task in range(6)]
    task_ids = [command.task_id for command in commands]
    queries = {
        f"{command.task_id}:0": {
            "current_command": command.command,
            "parsed_clauses": {"parse_failed": False, "clauses": []},
            "prior_events": [],
        }
        for command in commands
    }
    current = {
        sample_id: {
            "hard": {
                "latency": 0,
                "peak_cpu_cores": 0,
                "sampled_peak_rss_mb": 0,
                "disk_read_write_bytes_total": 0,
            },
            "pmf": {},
        }
        for sample_id in queries
    }
    catalog, private, _tasks = build_catalog(
        task_ids, commands, queries, current, warmup_tasks=5
    )
    decoded = _catalog_rows(catalog)

    assert len(decoded) == 5
    assert decoded[0]["task_id"] == "T001"
    assert decoded[0]["sample_id"] == "S00001"
    assert "current_pmf" not in decoded[0]
    assert private["S00001"] == "target__repo-0:0"

    outputs = {
        f"{command.task_id}:0": {
            "rule_id": "suite",
            "state": "ready",
            "evidence_event_indices": [],
        }
        for command in commands
    }
    pmfs, report = _fit_pmfs(
        task_ids, commands, outputs, warmup_tasks=5
    )
    assert pmfs[(('suite', 'ready'), 'latency')] == (0.0, 0.0, 0.0, 0.0, 1.0)
    assert report["suite::ready::latency"]["distinct_tasks"] == 5
