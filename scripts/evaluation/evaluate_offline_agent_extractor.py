#!/usr/bin/env python3
"""Generate and evaluate one offline agent-written causal feature function."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.classify_full_test_states import (  # noqa: E402
    _is_tool_free_event,
)
from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    CANONICAL_LATENCY_BUCKETS,
    CANONICAL_RESOURCE_BUCKET_EDGES,
    RESOURCE_BUCKET_LABELS,
    ClauseResourceKB,
    CommandRow,
    PipExecEvent,
    Row,
    _argmax_probabilities,
    _phase_changes,
    _result_exit_code,
    _sidecar_hard_metrics,
    command_resource_bucket_label,
    evaluate_full_test_phase,
    evaluate_prequential_commands,
    parse_command_clauses,
)
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    load_rows,
    load_run_rows,
)
from scripts.evaluation.evaluate_command_prequential import (  # noqa: E402
    _load_exec_events,
)
from tool_resource_eval.labels import repo_of  # noqa: E402

MODEL = "gpt-5.6-sol"
SCHEMA = "offline-agent-causal-extractor-v1"
WARMUP_TASKS = 80
MINIMUM_SUPPORT_TASKS = 5
MAX_SELECTED_SAMPLES = 12
MAX_SELECTED_TASKS = 12
MAX_COMBINED_PROMPT_BYTES = 800_000
TARGETS = ("latency", *CANONICAL_RESOURCE_BUCKET_EDGES)

SELECTION_PROMPT = """You are selecting evidence for one reusable causal resource-prediction
feature. The JSON catalog contains every eligible command from 80 settled
training tasks. Labels and Current predictions are training outcomes, not
runtime inputs.

Find one repeated opportunity where Current mixes execution modes that a pure
function could distinguish from current command structure and/or causally
earlier command text, exit status, and bounded result excerpts in the same task.
Do not assume any named tool or choose a pattern because this prompt suggests
it. Select at most 12 catalog sample IDs from at most 12 tasks, including both
errors and controls when possible. Prefer evidence repeated across at least five
tasks. Select none if the catalog does not justify such a mechanism. Do not call
tools. Return only the required JSON.

CATALOG:
"""

GENERATION_PROMPT = """Generate one small, auditable Python feature function from the selected
training evidence below. The function must be named extract and accept exactly
one dict query with current_command, parsed_clauses, and prior_events. Each prior
event has event_index, command, exit_code, and result_excerpt and is causally
earlier than the current command.

Return None when the discovered mechanism does not apply. Otherwise return a
dict with exactly rule_id, state, and evidence_event_indices. rule_id and state
are reusable categorical strings; they are not resource buckets. The host will
fit latency, CPU, RSS, and Disk probability distributions. The function must
not predict a bucket, contain task/sample/repository/file/test/package-specific
literals, or infer from event count alone when semantic evidence is absent.

The source may define small helper functions but may not import modules; `re`
is provided. It may use only the query, ordinary pure Python, and no filesystem,
network, subprocess, clock, randomness, labels, Current prediction, timing,
telemetry, current result, or future event. Generate one source or abstain. Do
not assign or construct string matching literals indirectly; put them directly
in the comparison that uses them, and put rule_id/state constants directly in
the returned dict. Do not call tools. Return only the required JSON.

SELECTED TRAINING EVIDENCE:
"""

SELECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hypothesis", "selected_sample_ids", "why_agent_needed"],
    "properties": {
        "hypothesis": {"type": "string"},
        "selected_sample_ids": {
            "type": "array",
            "maxItems": MAX_SELECTED_SAMPLES,
            "items": {"type": "string"},
        },
        "why_agent_needed": {"type": "string"},
    },
}

GENERATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["abstain", "source", "explanation"],
    "properties": {
        "abstain": {"type": "boolean"},
        "source": {"type": "string"},
        "explanation": {"type": "string"},
    },
}

_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
_FORBIDDEN_NAMES = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}
_INTERFACE_STRINGS = {
    "current_command",
    "parsed_clauses",
    "prior_events",
    "event_index",
    "command",
    "exit_code",
    "result_excerpt",
    "rule_id",
    "state",
    "evidence_event_indices",
}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _excerpt(text: str) -> str:
    if len(text) <= 500:
        return text
    marker = "\n<omitted>\n"
    head = (500 - len(marker)) // 2
    return text[:head] + marker + text[-(500 - len(marker) - head) :]


def _compact_parsed(parsed: Mapping[str, Any]) -> list[Any]:
    return [
        bool(parsed.get("parse_failed")),
        [
            [
                clause.get(key)
                for key in (
                    "bin",
                    "in_loop",
                    "in_pipe",
                    "in_subst",
                    "pipeline_position",
                )
            ]
            for clause in parsed.get("clauses", ())
        ],
    ]


def build_queries(
    task_ids: Sequence[str],
    command_rows: Sequence[CommandRow],
    events_by_task: Mapping[str, Sequence[PipExecEvent]],
) -> dict[str, dict[str, Any]]:
    """Build one label-free causal query for every eligible command."""

    if list(events_by_task) != list(task_ids):
        raise ValueError("raw exec event tasks differ from accepted task order")
    commands = {(row.task_id, row.call_id): row for row in command_rows}
    queries: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        events = events_by_task[task_id]
        if len({event.call_id for event in events}) != len(events):
            raise ValueError(f"{task_id}: duplicate raw exec call id")
        for event_index, event in enumerate(events):
            row = commands.get((task_id, event.call_id))
            if row is None:
                continue
            if row.command != event.command:
                raise ValueError(f"{row.call_id}: eligible command lacks matching raw event")
            prior_events = []
            for prior_index, prior in enumerate(events[:event_index]):
                if (
                    prior.ts_end is not None
                    and event.ts_start is not None
                    and prior.ts_end >= event.ts_start
                ):
                    raise ValueError(f"{row.call_id}: prior exec event is not causal")
                prior_events.append(
                    {
                        "event_index": prior_index,
                        "command": prior.command,
                        "exit_code": _result_exit_code(prior.tool_result),
                        "result_excerpt": _excerpt(prior.tool_result),
                    }
                )
            sample_id = f"{task_id}:{row.call_index}"
            queries[sample_id] = {
                "current_command": row.command,
                "parsed_clauses": parse_command_clauses(row.command),
                "prior_events": prior_events,
            }
    expected = {f"{row.task_id}:{row.call_index}" for row in command_rows}
    if set(queries) != expected:
        raise ValueError("eligible commands differ from raw exec events")
    return queries


def _current_predictions(
    public_rows: Sequence[Row],
    task_ids: Sequence[str],
    clause_rows: Sequence[Row],
    command_rows: Sequence[CommandRow],
) -> dict[str, dict[str, Any]]:
    clauses_by_task: dict[str, list[Row]] = defaultdict(list)
    commands_by_task: dict[str, list[CommandRow]] = defaultdict(list)
    for row in clause_rows:
        clauses_by_task[row.task_id].append(row)
    for row in command_rows:
        commands_by_task[row.task_id].append(row)
    kb = ClauseResourceKB.fit_public(
        row.observation(0.0, 1.0)
        for row in public_rows
        if row.structure_known and row.pipeline_position <= 0
    )
    predictions: dict[str, dict[str, Any]] = {}
    for ordinal, task_id in enumerate(task_ids):
        query_ts = float(ordinal * 2 + 3)
        for row in commands_by_task[task_id]:
            latency = kb.predict_command_latency_bucket(
                row.repo, row.command, query_ts, CANONICAL_LATENCY_BUCKETS
            ).prediction
            resources = kb.predict_command_resource_buckets(
                row.repo, row.command, query_ts
            ).classifications
            predictions[f"{task_id}:{row.call_index}"] = {
                "hard": {
                    "latency": (
                        None
                        if latency is None
                        else _argmax_probabilities(latency.probability_by_bucket)
                    ),
                    **{
                        target: (
                            None
                            if resources[target] is None
                            else RESOURCE_BUCKET_LABELS.index(resources[target].label)
                        )
                        for target in CANONICAL_RESOURCE_BUCKET_EDGES
                    },
                },
                "pmf": {
                    "latency": (
                        None if latency is None else list(latency.probability_by_bucket)
                    ),
                    **{
                        target: (
                            None
                            if resources[target] is None
                            else list(resources[target].probability_by_bucket)
                        )
                        for target in CANONICAL_RESOURCE_BUCKET_EDGES
                    },
                },
            }
        settle_ts = query_ts + 0.5
        for row in clauses_by_task[task_id]:
            kb.observe_completed_clause(row.observation(query_ts, settle_ts))
    return predictions


def build_catalog(
    task_ids: Sequence[str],
    command_rows: Sequence[CommandRow],
    queries: Mapping[str, Mapping[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
    *,
    warmup_tasks: int = WARMUP_TASKS,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Return the generic warm-up catalog and private opaque-ID maps."""

    task_opaque = {
        task_id: f"T{index + 1:03d}"
        for index, task_id in enumerate(task_ids[:warmup_tasks])
    }
    sample_private: dict[str, str] = {}
    rows = []
    for row in command_rows:
        if row.task_id not in task_opaque:
            continue
        real_sample = f"{row.task_id}:{row.call_index}"
        sample_id = f"S{len(rows) + 1:05d}"
        sample_private[sample_id] = real_sample
        labels = {"latency": CANONICAL_LATENCY_BUCKETS.bucket_id(row.duration_ms)}
        for target in CANONICAL_RESOURCE_BUCKET_EDGES:
            labels[target], _source = command_resource_bucket_label(row, target)
        hard = current[real_sample]["hard"]
        rows.append(
            [
                sample_id,
                task_opaque[row.task_id],
                len(queries[real_sample]["prior_events"]),
                row.command,
                _compact_parsed(queries[real_sample]["parsed_clauses"]),
                [labels[target] for target in TARGETS],
                [hard[target] for target in TARGETS],
            ]
        )
    catalog = {
        "targets": list(TARGETS),
        "class_ids": {
            "latency": "0..4 low to high",
            "resources": "0=Low, 1=Medium, 2=High",
            "null": "label unavailable",
        },
        "parsed_clauses_fields": [
            "parse_failed",
            ["bin", "in_loop", "in_pipe", "in_subst", "pipeline_position"],
        ],
        "row_fields": [
            "sample_id",
            "task_id",
            "event_index",
            "command",
            "parsed_clauses",
            "labels",
            "current_hard",
        ],
        "rows": rows,
    }
    return catalog, sample_private, task_opaque


def _catalog_rows(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = catalog.get("row_fields")
    rows = catalog.get("rows")
    if not isinstance(fields, list) or not isinstance(rows, list):
        raise ValueError("catalog fields or rows are invalid")
    return [dict(zip(fields, row, strict=True)) for row in rows]


def _codex_call(
    prompt: str,
    schema: Mapping[str, Any],
    directory: Path,
    stem: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    schema_path = directory / f"{stem}.schema.json"
    response_path = directory / f"{stem}.response.json"
    schema_path.write_text(json.dumps(schema, sort_keys=True))
    started = time.monotonic()
    completed = subprocess.run(
        [
            "codex",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--cd",
            str(directory),
            "--model",
            MODEL,
            "--config",
            'service_tier="fast"',
            "--config",
            'model_reasoning_effort="medium"',
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(response_path),
            "--json",
            "-",
        ],
        input=prompt,
        text=True,
        capture_output=True,
    )
    wall_seconds = time.monotonic() - started
    (directory / f"{stem}.prompt.txt").write_text(prompt)
    (directory / f"{stem}.events.jsonl").write_text(completed.stdout)
    (directory / f"{stem}.stderr.txt").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"{stem}: codex failed: {completed.stderr[-2_000:]}")
    events = [json.loads(line) for line in completed.stdout.splitlines() if line]
    if not events or any(not _is_tool_free_event(event) for event in events):
        raise ValueError(f"{stem}: agent attempted forbidden tool use")
    usage = _validated_usage(events, stem)
    return json.loads(response_path.read_text()), {
        "prompt_bytes": len(prompt.encode()),
        "wall_seconds": wall_seconds,
        "usage": usage,
    }


def _validated_usage(events: Sequence[Mapping[str, Any]], stem: str) -> dict[str, int]:
    usage_records = [
        event["usage"]
        for event in events
        if isinstance(event, dict) and isinstance(event.get("usage"), dict)
    ]
    if len(usage_records) != 1:
        raise ValueError(f"{stem}: expected exactly one token-usage record")
    usage = usage_records[0]
    if any(
        not isinstance(usage.get(key), int)
        or isinstance(usage[key], bool)
        or usage[key] < 0
        for key in ("input_tokens", "cached_input_tokens", "output_tokens")
    ):
        raise ValueError(f"{stem}: token usage is incomplete")
    return {key: int(usage[key]) for key in ("input_tokens", "cached_input_tokens", "output_tokens")}


def _validate_selection(
    selection: Any,
    catalog: Mapping[str, Any],
) -> list[str]:
    if not isinstance(selection, dict) or set(selection) != {
        "hypothesis",
        "selected_sample_ids",
        "why_agent_needed",
    }:
        raise ValueError("selection differs from the frozen schema")
    if not all(
        isinstance(selection[key], str) and selection[key].strip()
        for key in ("hypothesis", "why_agent_needed")
    ) or not isinstance(selection["selected_sample_ids"], list):
        raise ValueError("selection explanation is invalid")
    selected = selection["selected_sample_ids"]
    if (
        len(selected) > MAX_SELECTED_SAMPLES
        or len(set(selected)) != len(selected)
        or any(not isinstance(sample_id, str) for sample_id in selected)
    ):
        raise ValueError("selected sample IDs are invalid")
    rows = {row["sample_id"]: row for row in _catalog_rows(catalog)}
    if any(sample_id not in rows for sample_id in selected):
        raise ValueError("selection contains an unknown sample ID")
    if len({rows[sample_id]["task_id"] for sample_id in selected}) > MAX_SELECTED_TASKS:
        raise ValueError("selection exceeds the frozen task limit")
    return selected


def _selected_evidence(
    selected: Sequence[str],
    catalog: Mapping[str, Any],
    sample_private: Mapping[str, str],
    queries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = {row["sample_id"]: row for row in _catalog_rows(catalog)}
    return {
        "selected_rows": [rows[sample_id] for sample_id in selected],
        "causal_prefixes": {
            sample_id: queries[sample_private[sample_id]]["prior_events"]
            for sample_id in selected
        },
        "runtime_query_schema": {
            "current_command": "str",
            "parsed_clauses": "parser output",
            "prior_events": [
                {
                    "event_index": "int",
                    "command": "str",
                    "exit_code": "int|null",
                    "result_excerpt": "str",
                }
            ],
        },
    }


def validate_source(source: str, forbidden_ids: Sequence[str] = ()) -> None:
    if not source.strip() or len(source.encode()) > 30_000:
        raise ValueError("generated source is empty or too large")
    if any(identifier in source for identifier in forbidden_ids):
        raise ValueError("generated source contains an opaque training ID")
    tree = ast.parse(source)
    extract_defs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "extract"
    ]
    if len(extract_defs) != 1 or isinstance(extract_defs[0], ast.AsyncFunctionDef):
        raise ValueError("generated source must define one synchronous extract")
    function = extract_defs[0]
    if (
        len(function.args.args) != 1
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.kwonlyargs
    ):
        raise ValueError("extract must accept exactly one positional query")
    forbidden_nodes = (
        ast.AsyncFunctionDef,
        ast.Await,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Nonlocal,
        ast.Raise,
        ast.Try,
        ast.While,
        ast.With,
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError(f"generated source contains forbidden {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"generated source uses forbidden name {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("generated source accesses a private attribute")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literal = node.value
            if "/" in literal or re.search(
                r"(?:^|[/\\])[^/\\]+\.[A-Za-z0-9]{1,8}$", literal
            ):
                raise ValueError("generated source contains a file-specific literal")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            value = node.value
            if any(
                isinstance(descendant, ast.Constant)
                and isinstance(descendant.value, str)
                and descendant.value not in _INTERFACE_STRINGS
                for descendant in ast.walk(value)
            ):
                raise ValueError("generated source assigns a matching literal indirectly")
        if isinstance(node, (ast.BinOp, ast.JoinedStr)) and any(
            isinstance(descendant, ast.Constant)
            and isinstance(descendant.value, str)
            for descendant in ast.walk(node)
        ):
            raise ValueError("generated source constructs a string dynamically")


def _matching_literals(source: str) -> set[str]:
    tree = ast.parse(source)
    output_values: set[int] = set()
    docstrings: set[int] = set()
    for owner in (tree, *(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))):
        if (
            owner.body
            and isinstance(owner.body[0], ast.Expr)
            and isinstance(owner.body[0].value, ast.Constant)
            and isinstance(owner.body[0].value.value, str)
        ):
            docstrings.add(id(owner.body[0].value))
    extract = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "extract"
    )
    direct_returns: list[ast.Return] = []

    def collect_returns(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.Lambda)) and child is not extract:
                continue
            if isinstance(child, ast.Return):
                direct_returns.append(child)
            else:
                collect_returns(child)

    collect_returns(extract)
    for node in direct_returns:
        if not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value in {"rule_id", "state"}
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                output_values.add(id(value))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value
        and node.value not in _INTERFACE_STRINGS
        and id(node) not in output_values
        and id(node) not in docstrings
    }


def _regex_literals(source: str) -> dict[str, set[str]]:
    methods: dict[str, set[str]] = defaultdict(set)
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re"
            and node.func.attr in {"fullmatch", "match", "search"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            methods[node.args[0].value].add(node.func.attr)
    return methods


def _specific_command_tokens(command: str) -> set[str]:
    parsed = parse_command_clauses(command)
    if parsed.get("parse_failed"):
        return set()
    specific: set[str] = set()
    for clause in parsed.get("clauses", ()):
        argv = [str(value) for value in clause.get("argv", ())]
        if not argv:
            continue
        generic_indices = {0}
        generic_indices.update(
            index for index, value in enumerate(argv) if value.startswith("-")
        )
        if "-m" in argv[:-1]:
            module_index = argv.index("-m")
            generic_indices.add(module_index + 1)
        else:
            subcommand = next(
                (index for index in range(1, len(argv)) if not argv[index].startswith("-")),
                None,
            )
            if subcommand is not None:
                generic_indices.add(subcommand)
        specific.update(
            value
            for index, value in enumerate(argv)
            if index not in generic_indices and not value.startswith("-")
        )
    return specific


def validate_source_semantics(
    source: str,
    training_queries: Mapping[str, Mapping[str, Any]],
    task_by_sample: Mapping[str, str],
    actual: Mapping[str, Mapping[str, Any] | None],
    empty_history: Mapping[str, Mapping[str, Any] | None],
    neutral_history: Mapping[str, Mapping[str, Any] | None],
) -> None:
    """Reject unsupported literals and history-count proxies before scoring."""

    material_by_task: dict[str, str] = defaultdict(str)
    texts_by_task: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    generic_tokens: set[str] = set()
    positional_tokens: set[str] = set()
    for sample_id, query in training_queries.items():
        task_id = task_by_sample[sample_id]
        material_by_task[task_id] += "\n" + json.dumps(query, sort_keys=True).lower()
        texts_by_task[task_id].append((str(query["current_command"]), True))
        for event in query["prior_events"]:
            texts_by_task[task_id].extend(
                (
                    (str(event["command"]), True),
                    (str(event["result_excerpt"]), False),
                )
            )
        for clause in query["parsed_clauses"].get("clauses", ()):
            argv = [str(value).lower() for value in clause.get("argv", ())]
            if not argv:
                continue
            generic_tokens.add(argv[0])
            module_index = argv.index("-m") if "-m" in argv[:-1] else None
            if module_index is not None:
                generic_tokens.add(argv[module_index + 1])
            elif len(argv) > 1 and not argv[1].startswith("-"):
                generic_tokens.add(argv[1])
            generic_tokens.update(value for value in argv if value.startswith("-"))
            positional_tokens.update(value for value in argv[1:] if not value.startswith("-"))
    regex_literals = _regex_literals(source)
    for literal in _matching_literals(source):
        lowered = literal.lower()
        if literal in regex_literals:
            try:
                compiled = re.compile(literal)
            except re.error as error:
                raise ValueError(f"generated regex is invalid: {literal!r}") from error
            def matches(text: str) -> bool:
                return any(
                    getattr(compiled, method)(text) is not None
                    for method in regex_literals[literal]
                )

            matched = [
                (text, is_command)
                for texts in texts_by_task.values()
                for text, is_command in texts
                if matches(text)
            ]
            support = sum(
                any(matches(text) for text, _is_command in texts)
                for texts in texts_by_task.values()
            )
            dependent_specific_tokens = sorted(
                {
                    token
                    for text, is_command in matched
                    if is_command
                    for token in _specific_command_tokens(text)
                    if token in text and not matches(text.replace(token, "__ARG__"))
                }
            )
            if dependent_specific_tokens:
                raise ValueError(
                    "generated regex depends on package/test/file arguments: "
                    f"{dependent_specific_tokens}"
                )
            regex_tokens = {
                token.lower()
                for token in re.findall(r"[A-Za-z0-9_.-]{2,}", literal)
            }
            specific = sorted(
                regex_tokens & positional_tokens - generic_tokens
            )
            if specific:
                raise ValueError(
                    f"generated regex contains package/test-specific tokens: {specific}"
                )
        else:
            support = sum(
                lowered in material for material in material_by_task.values()
            )
        if support < MINIMUM_SUPPORT_TASKS:
            raise ValueError(
                f"generated matching literal lacks five-task support: {literal!r}"
            )
        if (
            re.fullmatch(r"[A-Za-z0-9_.+-]+", literal)
            and lowered in positional_tokens
            and lowered not in generic_tokens
        ):
            raise ValueError(
                f"generated source contains a package/test-specific literal: {literal!r}"
            )
    for sample_id in training_queries:
        actual_value = actual[sample_id]
        empty_value = empty_history[sample_id]
        neutral_value = neutral_history[sample_id]
        if _signature(actual_value) == _signature(empty_value):
            continue
        if actual_value is not None and not actual_value["evidence_event_indices"]:
            raise ValueError("history-dependent signature lacks cited semantic evidence")
        if _signature(actual_value) == _signature(neutral_value):
            raise ValueError("generated source infers state from history count alone")


def _validate_extraction(value: Any, query: Mapping[str, Any]) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "rule_id",
        "state",
        "evidence_event_indices",
    }:
        raise ValueError("extract returned the wrong fields")
    if any(
        not isinstance(value[key], str)
        or not value[key].strip()
        or len(value[key]) > 80
        for key in ("rule_id", "state")
    ):
        raise ValueError("extract returned an invalid signature")
    evidence = value["evidence_event_indices"]
    available = {event["event_index"] for event in query["prior_events"]}
    if (
        not isinstance(evidence, list)
        or len(set(evidence)) != len(evidence)
        or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index not in available
            for index in evidence
        )
    ):
        raise ValueError("extract returned invalid causal evidence")
    return value


def _worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = str(payload["source"])
    queries = payload["queries"]
    validate_source(source)
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, "re": re}
    exec(compile(source, "<generated_extractor>", "exec"), namespace)
    extract = namespace["extract"]
    outputs = []
    durations_ns = []
    for query in queries:
        started = time.perf_counter_ns()
        value = _validate_extraction(extract(query), query)
        durations_ns.append(time.perf_counter_ns() - started)
        outputs.append(value)
    return {"outputs": outputs, "durations_ns": durations_ns}


def run_source(
    source: str,
    queries: Sequence[Mapping[str, Any]],
    *,
    timeout_seconds: float = 30.0,
) -> tuple[list[dict[str, Any] | None], list[int]]:
    payload = {"source": source, "queries": list(queries)}
    completed = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    if completed.returncode:
        raise RuntimeError(f"generated extractor failed: {completed.stderr[-2_000:]}")
    result = json.loads(completed.stdout)
    return result["outputs"], result["durations_ns"]


def _signature(value: Mapping[str, Any] | None) -> tuple[str, str] | None:
    return None if value is None else (str(value["rule_id"]), str(value["state"]))


def _fit_pmfs(
    task_ids: Sequence[str],
    command_rows: Sequence[CommandRow],
    outputs: Mapping[str, Mapping[str, Any] | None],
    *,
    warmup_tasks: int = WARMUP_TASKS,
) -> tuple[dict[tuple[tuple[str, str], str], tuple[float, ...]], dict[str, Any]]:
    warmup = set(task_ids[:warmup_tasks])
    counts: dict[tuple[tuple[str, str], str], Counter[int]] = defaultdict(Counter)
    support: dict[tuple[tuple[str, str], str], set[str]] = defaultdict(set)
    for row in command_rows:
        if row.task_id not in warmup:
            continue
        signature = _signature(outputs[f"{row.task_id}:{row.call_index}"])
        if signature is None:
            continue
        labels: dict[str, int | None] = {
            "latency": CANONICAL_LATENCY_BUCKETS.bucket_id(row.duration_ms)
        }
        for target in CANONICAL_RESOURCE_BUCKET_EDGES:
            labels[target], _source = command_resource_bucket_label(row, target)
        for target, label in labels.items():
            if label is not None:
                counts[signature, target][label] += 1
                support[signature, target].add(row.task_id)
    pmfs = {}
    report = {}
    for key, target_counts in sorted(counts.items()):
        signature, target = key
        tasks = support[key]
        bucket_count = CANONICAL_LATENCY_BUCKETS.bucket_count if target == "latency" else 3
        usable = len(tasks) >= MINIMUM_SUPPORT_TASKS
        if usable:
            total = sum(target_counts.values())
            pmfs[key] = tuple(target_counts[index] / total for index in range(bucket_count))
        report[f"{signature[0]}::{signature[1]}::{target}"] = {
            "distinct_tasks": len(tasks),
            "label_counts": dict(sorted(target_counts.items())),
            "usable": usable,
            "pmf": None if not usable else list(pmfs[key]),
        }
    return pmfs, report


def _apply_candidate(
    baseline_rows: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Any] | None],
    pmfs: Mapping[tuple[tuple[str, str], str], tuple[float, ...]],
) -> list[dict[str, Any]]:
    rows = []
    for base in baseline_rows:
        current = dict(base["current_dynamic"])
        current_pmfs = current.pop("probability_by_bucket")
        candidate = dict(current)
        candidate_pmfs = dict(current_pmfs)
        extracted = outputs[base["sample_id"]]
        signature = _signature(extracted)
        applied = []
        if signature is not None:
            for target in TARGETS:
                pmf = pmfs.get((signature, target))
                if pmf is None:
                    continue
                bucket = _argmax_probabilities(pmf)
                candidate[target] = (
                    bucket if target == "latency" else RESOURCE_BUCKET_LABELS[bucket]
                )
                candidate_pmfs[target] = list(pmf)
                applied.append(target)
        rows.append(
            {
                **base,
                "labels": {"latency": base["latency_label"], **base["resource_labels"]},
                "current_dynamic": current,
                "current_probability_by_bucket": current_pmfs,
                "generated_signature": (
                    None
                    if extracted is None
                    else {key: extracted[key] for key in ("rule_id", "state")}
                ),
                "generated_evidence_event_indices": (
                    None if extracted is None else extracted["evidence_event_indices"]
                ),
                "generated_applied_targets": applied,
                "candidate": candidate,
                "candidate_probability_by_bucket": candidate_pmfs,
            }
        )
    return rows


def _assert_current_alignment(
    baseline_rows: Sequence[Mapping[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
) -> None:
    for base in baseline_rows:
        expected = current[base["sample_id"]]
        actual = base["current_dynamic"]
        hard = {
            "latency": actual["latency"],
            **{
                target: RESOURCE_BUCKET_LABELS.index(actual[target])
                for target in CANONICAL_RESOURCE_BUCKET_EDGES
            },
        }
        if hard != expected["hard"] or actual["probability_by_bucket"] != expected["pmf"]:
            raise AssertionError("catalog Current predictions differ from evaluator Current")


def _score_candidate(
    baseline: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = {target: _sidecar_hard_metrics(rows, target) for target in TARGETS}
    changes = {
        target: _phase_changes(rows, target, reference="current_dynamic")
        for target in TARGETS
    }
    reference = {
        "latency": baseline["latency"]["current_dynamic"],
        **{
            target: baseline["resources"][target]["current_dynamic"]
            for target in CANONICAL_RESOURCE_BUCKET_EDGES
        },
    }
    accuracy_ok = all(
        metrics[target][key] is not None
        and reference[target][key] is not None
        and metrics[target][key] >= reference[target][key]
        for target in TARGETS
        for key in ("exact_class_accuracy" if target == "latency" else "accuracy",)
    )
    severe_ok = all(
        metrics[target]["severe_underprediction_rate"] is not None
        and reference[target]["severe_underprediction_rate"] is not None
        and metrics[target]["severe_underprediction_rate"]
        <= reference[target]["severe_underprediction_rate"]
        for target in TARGETS
    )
    helpful = sum(change["helpful"] for change in changes.values())
    harmful = sum(change["harmful"] for change in changes.values())
    helpful_tasks = {
        task_id
        for change in changes.values()
        for task_id in change["helpful_task_ids"]
    }
    return {
        "metrics": metrics,
        "changes": changes,
        "gate": {
            "go": accuracy_ok and severe_ok and helpful > harmful and len(helpful_tasks) >= 3,
            "no_accuracy_regression": accuracy_ok,
            "no_severe_underprediction_regression": severe_ok,
            "helpful": helpful,
            "harmful": harmful,
            "helpful_tasks": len(helpful_tasks),
            "requires_helpful_tasks_at_least": 3,
        },
    }


def _percentile(values: Sequence[int], fraction: float) -> float:
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))])


def _run(args: argparse.Namespace) -> None:
    replay = args.replay_frozen_artifact
    if args.out_dir.exists() and not replay:
        raise FileExistsError("output directory already exists; use a fresh path")
    if replay and not args.out_dir.exists():
        raise FileNotFoundError("frozen artifact directory does not exist")
    if replay and any((args.out_dir / name).exists() for name in ("result.json", "rows.jsonl")):
        raise FileExistsError("frozen artifact was already scored")
    task_ids, clauses, commands = load_run_rows(args.run_dir)
    if len(task_ids) != 100:
        raise ValueError("offline extractor requires the frozen 100-task development run")
    excluded = {repo_of(task_id) for task_id in task_ids}
    public = [row for path in args.public_telemetry for row in load_rows(path)]
    public = [row for row in public if row.repo not in excluded]
    if not public:
        raise ValueError("public evidence is empty after target-repository exclusion")
    events = _load_exec_events(args.run_dir, task_ids)
    queries = build_queries(task_ids, commands, events)
    current = _current_predictions(public, task_ids, clauses, commands)
    catalog, sample_private, _task_opaque = build_catalog(
        task_ids, commands, queries, current
    )
    args.out_dir.mkdir(parents=True, exist_ok=replay)
    catalog_path = args.out_dir / "training-catalog.json"
    if replay:
        if json.loads(catalog_path.read_text()) != catalog:
            raise ValueError("replay catalog differs from the frozen artifact")
    else:
        catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")

    selection_prompt = SELECTION_PROMPT + json.dumps(catalog, separators=(",", ":"))
    if len(selection_prompt.encode()) >= MAX_COMBINED_PROMPT_BYTES:
        raise ValueError("selection prompt alone exceeds the frozen cost ceiling")
    if replay:
        artifact = json.loads((args.out_dir / "agent-artifact.json").read_text())
        if artifact.get("schema") != SCHEMA or artifact.get("model") != MODEL:
            raise ValueError("frozen agent artifact has the wrong schema or model")
        selection = artifact.get("selection")
        selected = _validate_selection(selection, catalog)
        evidence = _selected_evidence(selected, catalog, sample_private, queries)
        generation_prompt = GENERATION_PROMPT + json.dumps(
            evidence, separators=(",", ":")
        )
        combined_bytes = len(selection_prompt.encode()) + len(generation_prompt.encode())
        generation = artifact.get("generation")
        if not isinstance(generation, dict) or combined_bytes > MAX_COMBINED_PROMPT_BYTES:
            raise ValueError("frozen generation or prompt cost is invalid")
        replay_source = generation.get("source")
        if not isinstance(replay_source, str):
            raise ValueError("frozen generation source is invalid")
        expected_hashes = {
            "selection": hashlib.sha256(selection_prompt.encode()).hexdigest(),
            "generation": hashlib.sha256(generation_prompt.encode()).hexdigest(),
        }
        if (
            artifact.get("catalog_sha256") != hashlib.sha256(_json_bytes(catalog)).hexdigest()
            or artifact.get("prompt_sha256") != expected_hashes
            or artifact.get("source_sha256")
            != (
                None
                if not replay_source.strip()
                else hashlib.sha256(replay_source.encode()).hexdigest()
            )
            or artifact.get("cost", {}).get("combined_prompt_bytes") != combined_bytes
        ):
            raise ValueError("frozen artifact hashes or cost differ from replay inputs")
        stages = ("selection", "generation") if selected else ("selection",)
        for stage in stages:
            _validated_usage(
                [{"usage": artifact.get("cost", {}).get(stage, {}).get("usage")}],
                stage,
            )
        feature_path = args.out_dir / "generated_feature.py"
        expected_feature = replay_source.rstrip().encode() + b"\n"
        if replay_source.strip() and feature_path.read_bytes() != expected_feature:
            raise ValueError("generated feature file differs from the frozen source")
    else:
        with tempfile.TemporaryDirectory(prefix="offline-agent-") as temporary:
            work = Path(temporary)
            selection, selection_cost = _codex_call(
                selection_prompt, SELECTION_SCHEMA, work, "selection"
            )
            for path in work.iterdir():
                if path.is_file():
                    (args.out_dir / path.name).write_bytes(path.read_bytes())
            selected = _validate_selection(selection, catalog)
            evidence = _selected_evidence(selected, catalog, sample_private, queries)
            generation_prompt = GENERATION_PROMPT + json.dumps(
                evidence, separators=(",", ":")
            )
            combined_bytes = len(selection_prompt.encode()) + len(generation_prompt.encode())
            if combined_bytes > MAX_COMBINED_PROMPT_BYTES:
                raise ValueError(
                    f"combined prompts exceed frozen cost ceiling: {combined_bytes}"
                )
            if selected:
                generation, generation_cost = _codex_call(
                    generation_prompt, GENERATION_SCHEMA, work, "generation"
                )
            else:
                generation = {
                    "abstain": True,
                    "source": "",
                    "explanation": "selection agent abstained",
                }
                generation_cost = {"prompt_bytes": 0, "wall_seconds": 0.0, "usage": {}}
            for path in work.iterdir():
                if path.is_file():
                    (args.out_dir / path.name).write_bytes(path.read_bytes())
        artifact = None

    if not isinstance(generation, dict):
        raise ValueError("generation differs from the frozen schema")
    source = generation.get("source")
    if (
        set(generation) != {"abstain", "source", "explanation"}
        or not isinstance(generation.get("abstain"), bool)
        or not isinstance(source, str)
        or not isinstance(generation.get("explanation"), str)
        or (generation["abstain"] and source.strip())
        or (not generation["abstain"] and not source.strip())
    ):
        raise ValueError("generation differs from the frozen schema")
    if artifact is None:
        artifact = {
            "schema": SCHEMA,
            "model": MODEL,
            "service_tier": "fast",
            "reasoning_effort": "medium",
            "codex_version": subprocess.run(
                ["codex", "--version"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "catalog_sha256": hashlib.sha256(_json_bytes(catalog)).hexdigest(),
            "prompt_sha256": {
                "selection": hashlib.sha256(selection_prompt.encode()).hexdigest(),
                "generation": hashlib.sha256(generation_prompt.encode()).hexdigest(),
            },
            "source_sha256": (
                None if not source.strip() else hashlib.sha256(source.encode()).hexdigest()
            ),
            "selection": selection,
            "generation": generation,
            "cost": {
                "selection": selection_cost,
                "generation": generation_cost,
                "combined_prompt_bytes": combined_bytes,
            },
        }
        (args.out_dir / "agent-artifact.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n"
        )
    ordered_samples = [f"{row.task_id}:{row.call_index}" for row in commands]
    ordered_queries = [queries[sample_id] for sample_id in ordered_samples]
    if generation["abstain"]:
        durations: list[int] = []
        outputs = dict.fromkeys(ordered_samples)
        empty_outputs = dict(outputs)
    else:
        validate_source(
            source,
            (*sample_private, *{row["task_id"] for row in _catalog_rows(catalog)}),
        )
        (args.out_dir / "generated_feature.py").write_text(source.rstrip() + "\n")
        extracted, durations = run_source(source, ordered_queries)
        outputs = dict(zip(ordered_samples, extracted, strict=True))
        empty_queries = [{**query, "prior_events": []} for query in ordered_queries]
        empty_extracted, _empty_durations = run_source(source, empty_queries)
        empty_outputs = dict(zip(ordered_samples, empty_extracted, strict=True))
        warmup = set(task_ids[:WARMUP_TASKS])
        training_samples = [
            sample_id
            for sample_id, row in zip(ordered_samples, commands, strict=True)
            if row.task_id in warmup
        ]
        neutral_queries = [
            {
                **queries[sample_id],
                "prior_events": [
                    {
                        **event,
                        "command": "",
                        "exit_code": None,
                        "result_excerpt": "",
                    }
                    for event in queries[sample_id]["prior_events"]
                ],
            }
            for sample_id in training_samples
        ]
        neutral_extracted, _neutral_durations = run_source(source, neutral_queries)
        validate_source_semantics(
            source,
            {sample_id: queries[sample_id] for sample_id in training_samples},
            {
                f"{row.task_id}:{row.call_index}": row.task_id
                for row in commands
                if row.task_id in warmup
            },
            {sample_id: outputs[sample_id] for sample_id in training_samples},
            {sample_id: empty_outputs[sample_id] for sample_id in training_samples},
            dict(zip(training_samples, neutral_extracted, strict=True)),
        )
    pmfs, support = _fit_pmfs(task_ids, commands, outputs)

    baseline, baseline_rows = evaluate_prequential_commands(
        public,
        task_ids,
        clauses,
        commands,
        {"target_run_dir": str(args.run_dir.resolve())},
        warmup_task_count=WARMUP_TASKS,
    )
    _assert_current_alignment(baseline_rows, current)
    rows = _apply_candidate(baseline_rows, outputs, pmfs)
    score = _score_candidate(baseline, rows)

    empty_pmfs, empty_support = _fit_pmfs(task_ids, commands, empty_outputs)
    empty_rows = _apply_candidate(baseline_rows, empty_outputs, empty_pmfs)
    empty_score = _score_candidate(baseline, empty_rows)
    phase_result, _phase_rows = evaluate_full_test_phase(
        public,
        task_ids,
        clauses,
        commands,
        events,
        {"target_run_dir": str(args.run_dir.resolve())},
        warmup_task_count=WARMUP_TASKS,
    )
    result = {
        "status": (
            "development_exposed_agent_extractor_abstained"
            if generation["abstain"]
            else "development_exposed_offline_agent_extractor_go"
            if score["gate"]["go"]
            else "development_exposed_offline_agent_extractor_no_go"
        ),
        "claim_bearing": False,
        "objective": "offline_agent_generated_causal_command_signature",
        "protocol": {
            "warmup_tasks": WARMUP_TASKS,
            "test_tasks": 20,
            "minimum_distinct_warmup_tasks_per_signature_target": MINIMUM_SUPPORT_TASKS,
            "runtime_agent_calls": 0,
            "fallback": "Current per unsupported signature-target",
        },
        "artifact": artifact,
        "coverage": {
            "training_catalog_commands": len(catalog["rows"]),
            "selected_samples": len(selected),
            "selected_tasks": len(
                {
                    row["task_id"]
                    for row in _catalog_rows(catalog)
                    if row["sample_id"] in selected
                }
            ),
            "warmup_non_abstain": sum(
                outputs[sample_id] is not None
                for sample_id in ordered_samples
                if sample_id.split(":", 1)[0] in set(task_ids[:WARMUP_TASKS])
            ),
            "test_non_abstain": sum(
                outputs[row["sample_id"]] is not None for row in baseline_rows
            ),
            "test_any_target_applied": sum(
                bool(row["generated_applied_targets"]) for row in rows
            ),
        },
        "support": support,
        "extractor_runtime": {
            "queries": len(durations),
            "p50_ms": None if not durations else statistics.median(durations) / 1_000_000,
            "p95_ms": None if not durations else _percentile(durations, 0.95) / 1_000_000,
        },
        "baseline": {
            "latency": baseline["latency"]["current_dynamic"],
            "resources": {
                target: baseline["resources"][target]["current_dynamic"]
                for target in CANONICAL_RESOURCE_BUCKET_EDGES
            },
        },
        "candidate": score,
        "diagnostics": {
            "empty_history": {"support": empty_support, **empty_score},
            "full_test_phase": {
                "latency": phase_result["latency"]["arms"]["full_test_phase"],
                "resources": {
                    target: phase_result["resources"][target]["arms"]["full_test_phase"]
                    for target in CANONICAL_RESOURCE_BUCKET_EDGES
                },
                "gate": phase_result["gate"],
            },
        },
        "row_identity": {
            "test_commands": len(rows),
            "identical_sample_ids_and_labels": [row["sample_id"] for row in rows]
            == [row["sample_id"] for row in baseline_rows],
        },
    }
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def main() -> None:
    if sys.argv[1:] == ["--worker"]:
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        sys.stdout.write(json.dumps(_worker(json.load(sys.stdin))))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--public-telemetry", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--replay-frozen-artifact", action="store_true")
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
