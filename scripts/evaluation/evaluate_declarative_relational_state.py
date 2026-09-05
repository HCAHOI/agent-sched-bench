#!/usr/bin/env python3
"""Freeze and evaluate one declarative blocker/remediation specification."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    CANONICAL_RESOURCE_BUCKET_EDGES,
    RESOURCE_BUCKET_LABELS,
    evaluate_prequential_commands,
)
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    load_rows,
    load_run_rows,
)
from scripts.evaluation.evaluate_command_prequential import (  # noqa: E402
    _load_exec_events,
)
from scripts.evaluation.evaluate_offline_agent_extractor import (  # noqa: E402
    MODEL,
    TARGETS,
    _apply_candidate,
    _codex_call,
    _fit_pmfs,
    _score_candidate,
    _specific_command_tokens,
    build_queries,
)
from scripts.evaluation.evaluate_relational_agent_state import (  # noqa: E402
    _all_command_queries,
    _attempt_records,
    _collapse_whitespace,
    _collapsed_outputs,
    _coverage,
    _derive_state,
    _frozen_baseline,
    _json_bytes,
    _label_free_selected_evidence,
    _load_split_manifest,
    _normalize_identifier,
    _offset_rows,
    _select_primary_contrast,
    _telemetry_valid_records,
    _validate_spans,
    _write_result_view,
)
from tool_resource.runtime_kb import parse_command_clauses  # noqa: E402
from tool_resource_eval.labels import repo_of  # noqa: E402

SCHEMA = "offline-agent-declarative-relational-v1"
EVALUATION_SCHEMA = "declarative-relational-fresh-evaluation-v1"
MAX_PROMPT_BYTES = 200_000
MAX_SCOPE_PATTERNS = 4
MAX_RELATION_PATTERNS = 6
MAX_SCOPE_PATTERN_CHARS = 256
MAX_RELATION_PATTERN_CHARS = 512
MAX_REPEAT = 128
MAX_REGEX_AMBIGUITY = 1_024
MAX_COMMAND_CHARS = 8_192
MAX_RESULT_CHARS = 500
MAX_PRIOR_EVENTS = 256
MAX_SPANS_PER_PATTERN = 16
MAX_BLOCKERS = 64
MAX_EDGES = 64
MAX_P95_MS = 5.0
MINIMUM_SUPPORT_TASKS = 5
REGEX_FLAGS = re.ASCII | re.IGNORECASE
HOST_PATHS = ("scripts/evaluation", "src", "pyproject.toml", "uv.lock")
SPLIT_MANIFEST = _REPO_ROOT / "analysis/development/sqlglot-relational-task-split.json"
PREREGISTRATION_PATHS = (
    SPLIT_MANIFEST,
    _REPO_ROOT / "analysis/development/clause-interaction-kb-plan.md",
    _REPO_ROOT / "analysis/development/tool-resource-canonical-objective.md",
)

GENERATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "abstain",
        "rule_id",
        "scope_patterns",
        "blocker_patterns",
        "remediation_patterns",
        "explanation",
    ],
    "properties": {
        "abstain": {"type": "boolean"},
        "rule_id": {"type": "string", "maxLength": 40},
        "scope_patterns": {
            "type": "array",
            "maxItems": MAX_SCOPE_PATTERNS,
            "items": {"type": "string", "maxLength": MAX_SCOPE_PATTERN_CHARS},
        },
        "blocker_patterns": {
            "type": "array",
            "maxItems": MAX_RELATION_PATTERNS,
            "items": {"type": "string", "maxLength": MAX_RELATION_PATTERN_CHARS},
        },
        "remediation_patterns": {
            "type": "array",
            "maxItems": MAX_RELATION_PATTERNS,
            "items": {"type": "string", "maxLength": MAX_RELATION_PATTERN_CHARS},
        },
        "explanation": {"type": "string", "maxLength": 1_000},
    },
}

GENERATION_PROMPT = r"""Generate one declarative blocker/remediation specification
from the label-free development examples below. Return only the required JSON.

The host, not you, will scan every causal event, extract exact spans, join a
successful remediation to an earlier blocker only on exact normalized identifier
equality, derive blocked/partial/closure/new-blocker states, fit resource PMFs,
and fall back to Current. You cannot output code, buckets, states, relations,
weights, thresholds, or per-query decisions.

scope_patterns are case-insensitive ASCII regular expressions with zero capture
groups. The host full-matches them against the whitespace-collapsed current
command and separately requires one simple parsed clause. blocker_patterns and
remediation_patterns each contain exactly one named capture (?P<id>...) and no
other group. They run with finditer on one failed-verifier result excerpt or one
successful command, respectively; the id capture alone defines the span.

Allowed regex features: ASCII literals, escaped literals, character classes and
ranges, standard character categories, string/word anchors only at the two outer
pattern boundaries, and bounded repeats
with upper bound at most 128. Do not use alternation, wildcard dot, lookaround,
backreferences, conditional/noncapturing groups, inline flags, unbounded or
nested repeats. Use multiple patterns instead of alternation and {1,32} instead
of +. Do not spell one case-insensitive literal as a character class. Each
pattern must repeat across at least five development tasks;
blocker/remediation captures must yield at least two different identifiers.
Do not include task/sample/repo IDs or file, test, or package-specific literals.

Return abstain=true with empty rule_id and empty arrays if this contract cannot
express one reusable rule. Otherwise use 1-4 scope patterns, 1-6 blocker
patterns, and 1-6 remediation patterns. Do not call tools.

LABEL-FREE DEVELOPMENT EVIDENCE:
"""


class SpecError(ValueError):
    """The generated declarative specification violates its frozen contract."""


class BoundExceeded(SpecError):
    """A frozen input, span, or relation cardinality bound was exceeded."""


@dataclass(frozen=True)
class CompiledSpec:
    rule_id: str
    scope: tuple[re.Pattern[str], ...]
    blocker: tuple[re.Pattern[str], ...]
    remediation: tuple[re.Pattern[str], ...]


def _validate_regex_tree(tree: Any, *, inside_repeat: bool = False) -> int:
    constants = re._constants
    simple = {
        constants.LITERAL,
        constants.NOT_LITERAL,
        constants.CATEGORY,
        constants.AT,
    }
    class_tokens = {
        constants.LITERAL,
        constants.RANGE,
        constants.CATEGORY,
        constants.NEGATE,
    }
    ambiguity = 1
    for operation, value in tree:
        if operation in simple:
            if operation in {constants.LITERAL, constants.NOT_LITERAL} and value > 127:
                raise SpecError("regex contains a non-ASCII literal")
            continue
        if operation is constants.IN:
            if any(child not in class_tokens for child, _item in value):
                raise SpecError("regex character class contains a forbidden operation")
            if any(
                (child is constants.LITERAL and item > 127)
                or (child is constants.RANGE and item[1] > 127)
                for child, item in value
            ):
                raise SpecError("regex character class contains non-ASCII values")
            literal_choices = [
                chr(item).casefold()
                for child, item in value
                if child is constants.LITERAL
            ]
            if len(literal_choices) == len(value) and len(set(literal_choices)) == 1:
                raise SpecError("regex uses a character class to encode one literal")
            continue
        if operation is constants.SUBPATTERN:
            group, add_flags, delete_flags, body = value
            if group != 1 or add_flags or delete_flags or inside_repeat:
                raise SpecError("regex contains a forbidden group or inline flag")
            ambiguity *= _validate_regex_tree(body)
            continue
        if operation in {constants.MAX_REPEAT, constants.MIN_REPEAT}:
            minimum, maximum, body = value
            if (
                inside_repeat
                or maximum == constants.MAXREPEAT
                or maximum > MAX_REPEAT
                or minimum > maximum
            ):
                raise SpecError("regex repeat is unbounded, nested, or too large")
            ambiguity *= (maximum - minimum + 1) * _validate_regex_tree(
                body, inside_repeat=True
            )
            if ambiguity > MAX_REGEX_AMBIGUITY:
                raise SpecError("regex exceeds the backtracking ambiguity budget")
            continue
        raise SpecError(f"regex contains forbidden operation {operation}")
    return ambiguity


def _validate_anchor_positions(tree: Any, *, top_level: bool = True) -> None:
    constants = re._constants
    values = list(tree)
    for index, (operation, value) in enumerate(values):
        if operation is constants.AT:
            if (
                not top_level
                or value == constants.AT_NON_BOUNDARY
                or index not in {0, len(values) - 1}
            ):
                raise SpecError("regex anchor is not at an outer boundary")
        elif operation is constants.SUBPATTERN:
            _validate_anchor_positions(value[3], top_level=False)
        elif operation in {constants.MAX_REPEAT, constants.MIN_REPEAT}:
            _validate_anchor_positions(value[2], top_level=False)


def _contains_unescaped(pattern: str, character: str) -> bool:
    escaped = False
    for value in pattern:
        if value == character and not escaped:
            return True
        if value == "\\":
            escaped = not escaped
        else:
            escaped = False
    return False


def _regex_literal_runs(pattern: re.Pattern[str]) -> set[str]:
    constants = re._constants

    def exact(tree: Any) -> str | None:
        parts = []
        for operation, value in tree:
            if operation is constants.LITERAL:
                parts.append(chr(value))
            elif operation in {constants.MAX_REPEAT, constants.MIN_REPEAT}:
                minimum, maximum, body = value
                child = exact(body)
                if minimum != maximum or child is None:
                    return None
                parts.append(child * maximum)
            else:
                return None
        return "".join(parts)

    def collect(tree: Any) -> set[str]:
        runs: set[str] = set()
        current = []
        for operation, value in tree:
            fragment = chr(value) if operation is constants.LITERAL else None
            nested = None
            if operation in {constants.MAX_REPEAT, constants.MIN_REPEAT}:
                minimum, maximum, body = value
                child = exact(body)
                fragment = child * minimum if minimum and child is not None else None
                nested = body
            elif operation is constants.SUBPATTERN:
                nested = value[3]
            if fragment is None:
                if current:
                    runs.add("".join(current))
                    current = []
                if nested is not None:
                    runs.update(collect(nested))
            else:
                current.append(fragment)
        if current:
            runs.add("".join(current))
        return runs

    return collect(re._parser.parse(pattern.pattern, REGEX_FLAGS))


def _compile_pattern(pattern: str, *, kind: str) -> re.Pattern[str]:
    limit = MAX_SCOPE_PATTERN_CHARS if kind == "scope" else MAX_RELATION_PATTERN_CHARS
    if not pattern or len(pattern) > limit or not pattern.isascii():
        raise SpecError(f"{kind} regex is empty, non-ASCII, or too long")
    if _contains_unescaped(pattern, "|"):
        raise SpecError(f"{kind} regex contains forbidden alternation")
    if (kind == "scope" and "(?" in pattern) or (
        kind != "scope"
        and (pattern.count("(?") != 1 or pattern.count("(?P<id>") != 1)
    ):
        raise SpecError(f"{kind} regex contains a forbidden group construct")
    try:
        parsed = re._parser.parse(pattern, REGEX_FLAGS)
        compiled = re.compile(pattern, REGEX_FLAGS)
    except re.error as error:
        raise SpecError(f"{kind} regex does not compile") from error
    _validate_regex_tree(parsed)
    _validate_anchor_positions(parsed)
    if kind == "scope":
        if compiled.groups or compiled.groupindex:
            raise SpecError("scope regex contains a capture group")
        if compiled.fullmatch("") is not None:
            raise SpecError("scope regex may not match an empty command")
    elif compiled.groups != 1 or compiled.groupindex != {"id": 1}:
        raise SpecError(f"{kind} regex must contain exactly the named id capture")
    elif kind != "scope":
        capture_widths = [
            value[3].getwidth()[0]
            for operation, value in parsed
            if operation is re._constants.SUBPATTERN and value[0] == 1
        ]
        if len(capture_widths) != 1 or capture_widths[0] == 0:
            raise SpecError(f"{kind} id capture may be empty")
    return compiled


def validate_spec(value: Any, forbidden_ids: Sequence[str] = ()) -> CompiledSpec | None:
    if not isinstance(value, dict) or set(value) != set(GENERATION_SCHEMA["required"]):
        raise SpecError("generation differs from the frozen JSON schema")
    if (
        not isinstance(value["abstain"], bool)
        or not isinstance(value["rule_id"], str)
        or not isinstance(value["explanation"], str)
        or not all(
            isinstance(value[field], list)
            and all(isinstance(pattern, str) for pattern in value[field])
            for field in ("scope_patterns", "blocker_patterns", "remediation_patterns")
        )
    ):
        raise SpecError("generation has invalid JSON value types")
    serialized = json.dumps(value, sort_keys=True).casefold()
    if any(identifier.casefold() in serialized for identifier in forbidden_ids):
        raise SpecError("generation contains an opaque development ID")
    arrays = tuple(
        value[field]
        for field in ("scope_patterns", "blocker_patterns", "remediation_patterns")
    )
    if not value["explanation"].strip() or len(value["explanation"]) > 1_000:
        raise SpecError("explanation is empty or too long")
    if value["abstain"]:
        if value["rule_id"] or any(arrays):
            raise SpecError("abstention contains a rule or pattern")
        return None
    if re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", value["rule_id"]) is None:
        raise SpecError("rule_id is invalid")
    if not 1 <= len(arrays[0]) <= MAX_SCOPE_PATTERNS or any(
        not 1 <= len(patterns) <= MAX_RELATION_PATTERNS for patterns in arrays[1:]
    ):
        raise SpecError("pattern array cardinality is invalid")
    if any(len(patterns) != len(set(patterns)) for patterns in arrays):
        raise SpecError("pattern arrays contain duplicates")
    compiled = CompiledSpec(
        rule_id=value["rule_id"],
        scope=tuple(_compile_pattern(pattern, kind="scope") for pattern in arrays[0]),
        blocker=tuple(_compile_pattern(pattern, kind="blocker") for pattern in arrays[1]),
        remediation=tuple(
            _compile_pattern(pattern, kind="remediation") for pattern in arrays[2]
        ),
    )
    literal_tokens = {
        token.casefold()
        for pattern in (*compiled.scope, *compiled.blocker, *compiled.remediation)
        for run in _regex_literal_runs(pattern)
        for token in re.findall(r"[A-Za-z0-9_.+-]{4,}", run)
    }
    literal_tokens.update(
        token.casefold()
        for token in re.findall(
            r"[A-Za-z0-9_.+-]{4,}", f"{value['rule_id']} {value['explanation']}"
        )
    )
    if any(
        token in identifier.casefold()
        for token in literal_tokens
        for identifier in forbidden_ids
    ):
        raise SpecError("generation encodes an opaque development ID")
    return compiled


def _scope(spec: CompiledSpec, command: str, parsed: Mapping[str, Any]) -> str | None:
    clauses = parsed.get("clauses", ())
    if (
        parsed.get("parse_failed")
        or len(clauses) != 1
        or any(
            clauses[0].get(field)
            for field in ("in_pipe", "in_loop", "in_subst")
        )
        or not any(pattern.fullmatch(_collapse_whitespace(command)) for pattern in spec.scope)
    ):
        return None
    return spec.rule_id


def _pattern_spans(
    patterns: Sequence[re.Pattern[str]],
    text: str,
    owner: str,
) -> list[list[int]]:
    spans: set[tuple[int, int]] = set()
    for pattern in patterns:
        matches = list(pattern.finditer(text))
        if len(matches) > MAX_SPANS_PER_PATTERN:
            raise BoundExceeded(f"{owner} pattern exceeded its span bound")
        for match in matches:
            start, end = match.span("id")
            if start < 0:
                raise SpecError(f"{owner} id capture did not participate")
            spans.add((start, end))
    ordered = [list(span) for span in sorted(spans)]
    try:
        _validate_spans(ordered, text, owner)
    except ValueError as error:
        raise SpecError(str(error)) from error
    return ordered


def _namespace(spec: CompiledSpec) -> dict[str, Any]:
    return {
        "scope": lambda command, parsed: _scope(spec, command, parsed),
        "blocker_spans": lambda text: _pattern_spans(spec.blocker, text, "blocker"),
        "remediation_spans": lambda command: _pattern_spans(
            spec.remediation, command, "remediation"
        ),
    }


def extract_query(
    spec: CompiledSpec,
    query: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    command = str(query["current_command"])
    parsed = query["parsed_clauses"]
    events = query["prior_events"]
    if (
        len(command) > MAX_COMMAND_CHARS
        or len(events) > MAX_PRIOR_EVENTS
        or any(
            len(str(event["command"])) > MAX_COMMAND_CHARS
            or len(str(event["result_excerpt"])) > MAX_RESULT_CHARS
            for event in events
        )
    ):
        return None, True
    if _scope(spec, command, parsed) is None:
        return None, False
    try:
        graph = _derive_state(query, _namespace(spec))
    except SpecError:
        return None, True
    if graph is None:
        return None, False
    edge_count = sum(len(item["addresses"]) for item in graph["remediations"])
    if len(graph["blockers"]) > MAX_BLOCKERS or edge_count > MAX_EDGES:
        return None, True
    return graph, False


def scope_only(spec: CompiledSpec, query: Mapping[str, Any]) -> dict[str, Any] | None:
    command = str(query["current_command"])
    if len(command) > MAX_COMMAND_CHARS:
        return None
    rule_id = _scope(spec, command, query["parsed_clauses"])
    return (
        None
        if rule_id is None
        else {"rule_id": rule_id, "state": "__scope_only__", "evidence_event_indices": []}
    )


def run_queries(
    spec: CompiledSpec,
    queries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any] | None], list[int], int]:
    outputs = []
    durations = []
    bounded = 0
    for query in queries:
        started = time.perf_counter_ns()
        output, exceeded = extract_query(spec, query)
        durations.append(time.perf_counter_ns() - started)
        outputs.append(output)
        bounded += exceeded
    return outputs, durations, bounded


def _pattern_support(
    pattern: re.Pattern[str],
    texts_by_task: Mapping[str, Sequence[str]],
    *,
    scope: bool,
) -> tuple[set[str], set[str]]:
    tasks = set()
    identifiers = set()
    for task_id, texts in texts_by_task.items():
        for text in texts:
            if scope:
                if pattern.fullmatch(_collapse_whitespace(text)) is not None:
                    tasks.add(task_id)
                continue
            spans = _pattern_spans((pattern,), text, "development_support")
            if spans:
                tasks.add(task_id)
            identifiers.update(
                identifier
                for start, end in spans
                if (identifier := _normalize_identifier(text[start:end]))
            )
    return tasks, identifiers


def _relation_specific_tokens(command: str) -> set[str]:
    """Return positional arguments while treating invocation/subcommand as syntax."""

    parsed = parse_command_clauses(command)
    if parsed.get("parse_failed"):
        return set()
    specific: set[str] = set()
    for clause in parsed.get("clauses", ()):
        argv = [str(value) for value in clause.get("argv", ())]
        if not argv:
            continue
        generic = {0, *(index for index, value in enumerate(argv) if value.startswith("-"))}
        start = 1
        if "-m" in argv[:-1]:
            module = argv.index("-m")
            generic.add(module + 1)
            start = module + 2
        subcommand = next(
            (index for index in range(start, len(argv)) if not argv[index].startswith("-")),
            None,
        )
        if subcommand is not None:
            generic.add(subcommand)
        specific.update(
            value
            for index, value in enumerate(argv)
            if index not in generic and not value.startswith("-")
        )
    return specific


def validate_development_support(
    spec: CompiledSpec,
    queries: Mapping[str, Mapping[str, Any]],
    task_by_sample: Mapping[str, str],
) -> dict[str, Any]:
    commands: dict[str, set[str]] = defaultdict(set)
    all_commands: dict[str, set[str]] = defaultdict(set)
    failed_results: dict[str, set[str]] = defaultdict(set)
    successful_commands: dict[str, set[str]] = defaultdict(set)
    for sample_id, query in queries.items():
        task_id = task_by_sample[sample_id]
        current = str(query["current_command"])
        all_commands[task_id].add(current)
        parsed = query["parsed_clauses"]
        clauses = parsed.get("clauses", ())
        if (
            not parsed.get("parse_failed")
            and len(clauses) == 1
            and not any(
                clauses[0].get(field) for field in ("in_pipe", "in_loop", "in_subst")
            )
        ):
            commands[task_id].add(current)
        for event in query["prior_events"]:
            all_commands[task_id].add(str(event["command"]))
            if event["exit_code"] == 0:
                successful_commands[task_id].add(str(event["command"]))
            elif (
                event["exit_code"] is not None
                and _collapse_whitespace(str(event["command"]))
                == _collapse_whitespace(current)
            ):
                failed_results[task_id].add(str(event["result_excerpt"]))
    report: dict[str, Any] = {"scope": [], "blocker": [], "remediation": []}
    for kind, patterns, texts in (
        ("scope", spec.scope, commands),
        ("blocker", spec.blocker, failed_results),
        ("remediation", spec.remediation, successful_commands),
    ):
        for pattern in patterns:
            tasks, identifiers = _pattern_support(
                pattern, texts, scope=kind == "scope"
            )
            if len(tasks) < MINIMUM_SUPPORT_TASKS:
                raise SpecError(f"{kind} pattern lacks five-task support")
            if kind != "scope" and len(identifiers) < 2:
                raise SpecError(f"{kind} pattern captures fewer than two identifiers")
            report[kind].append(
                {
                    "pattern": pattern.pattern,
                    "tasks": len(tasks),
                    "identifiers": len(identifiers),
                }
            )
    for pattern in spec.scope:
        for task_commands in commands.values():
            for command in task_commands:
                if pattern.fullmatch(_collapse_whitespace(command)) is None:
                    continue
                for token in _specific_command_tokens(command):
                    masked = _collapse_whitespace(command.replace(token, "__ARG__"))
                    if pattern.fullmatch(masked) is None:
                        raise SpecError("scope pattern depends on a specific argument")
    for pattern in spec.remediation:
        for task_commands in successful_commands.values():
            for command in task_commands:
                matches = list(pattern.finditer(command))
                if not matches:
                    continue
                identifiers = {
                    _normalize_identifier(match.group("id")) for match in matches
                }
                for token in _relation_specific_tokens(command):
                    if _normalize_identifier(token) in identifiers:
                        continue
                    if not list(pattern.finditer(command.replace(token, "__ARG__"))):
                        raise SpecError(
                            "remediation pattern depends on a specific argument"
                        )
    specific: set[str] = set()
    generic: set[str] = set()
    for task_commands in all_commands.values():
        for command in task_commands:
            command_specific = {
                token.casefold() for token in _relation_specific_tokens(command)
            }
            command_argv = {
                str(token).casefold()
                for clause in parse_command_clauses(command).get("clauses", ())
                for token in clause.get("argv", ())
            }
            specific.update(command_specific)
            generic.update(command_argv - command_specific)
    for pattern in (*spec.scope, *spec.blocker, *spec.remediation):
        for run in _regex_literal_runs(pattern):
            for token in re.findall(r"[A-Za-z0-9_.+-]{2,}", run):
                if token.casefold() in specific and token.casefold() not in generic:
                    raise SpecError("pattern contains a specific positional literal")
    return report


def _encode_pmfs(
    pmfs: Mapping[tuple[tuple[str, str], str], tuple[float, ...]],
) -> list[dict[str, Any]]:
    return [
        {
            "rule_id": signature[0],
            "state": signature[1],
            "target": target,
            "pmf": list(pmf),
        }
        for (signature, target), pmf in sorted(pmfs.items())
    ]


def _decode_pmfs(
    rows: Any,
) -> dict[tuple[tuple[str, str], str], tuple[float, ...]]:
    if not isinstance(rows, list):
        raise ValueError("frozen PMFs are missing")
    decoded = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"rule_id", "state", "target", "pmf"}:
            raise ValueError("frozen PMF row is invalid")
        target = row["target"]
        pmf = tuple(float(value) for value in row["pmf"])
        expected = 5 if target == "latency" else 3
        if (
            target not in TARGETS
            or len(pmf) != expected
            or any(value < 0.0 or value > 1.0 for value in pmf)
            or abs(sum(pmf) - 1.0) > 1e-9
        ):
            raise ValueError("frozen PMF value is invalid")
        key = ((str(row["rule_id"]), str(row["state"])), str(target))
        if key in decoded:
            raise ValueError("frozen PMF row is duplicated")
        decoded[key] = pmf
    return decoded


def _accuracy(score: Mapping[str, Any], target: str) -> float:
    key = "exact_class_accuracy" if target == "latency" else "accuracy"
    return float(score["metrics"][target][key])


def _prediction_bucket(value: Any, target: str) -> int | None:
    if value is None:
        return None
    return int(value) if target == "latency" else RESOURCE_BUCKET_LABELS.index(str(value))


def _fresh_gate(
    baseline: Mapping[str, Any],
    relational: Mapping[str, Any],
    collapsed: Mapping[str, Any],
    scope: Mapping[str, Any],
    relational_rows: Sequence[Mapping[str, Any]],
    collapsed_rows: Sequence[Mapping[str, Any]],
    scope_rows: Sequence[Mapping[str, Any]],
    primary: Mapping[str, Any],
) -> dict[str, Any]:
    reference_severe = {
        "latency": baseline["latency"]["current_dynamic"][
            "severe_underprediction_rate"
        ],
        **{
            target: baseline["resources"][target]["current_dynamic"][
                "severe_underprediction_rate"
            ]
            for target in CANONICAL_RESOURCE_BUCKET_EDGES
        },
    }
    severe_improved = sum(
        relational["metrics"][target]["severe_underprediction_rate"]
        < reference_severe[target]
        for target in TARGETS
    )
    no_worse_collapsed = all(
        _accuracy(relational, target) >= _accuracy(collapsed, target)
        for target in TARGETS
    )
    no_worse_scope = all(
        _accuracy(relational, target) >= _accuracy(scope, target) for target in TARGETS
    )
    pair = {
        "target": str(primary["target"]),
        "commands": 0,
        "relational_correct": 0,
        "collapsed_correct": 0,
        "scope_only_correct": 0,
    }
    states = set(primary["states"])
    for row, collapsed_row, scope_row in zip(
        relational_rows, collapsed_rows, scope_rows, strict=True
    ):
        signature = row["generated_signature"]
        if (
            signature is None
            or signature["rule_id"] != primary["rule_id"]
            or signature["state"] not in states
        ):
            continue
        truth = row["labels"][pair["target"]]
        if truth is None:
            continue
        pair["commands"] += 1
        pair["relational_correct"] += (
            _prediction_bucket(row["candidate"][pair["target"]], pair["target"])
            == truth
        )
        pair["collapsed_correct"] += (
            _prediction_bucket(
                collapsed_row["candidate"][pair["target"]], pair["target"]
            )
            == truth
        )
        pair["scope_only_correct"] += (
            _prediction_bucket(scope_row["candidate"][pair["target"]], pair["target"])
            == truth
        )
    pair["strictly_better_than_collapsed"] = (
        pair["relational_correct"] > pair["collapsed_correct"]
    )
    pair["strictly_better_than_scope_only"] = (
        pair["relational_correct"] > pair["scope_only_correct"]
    )
    gate = {
        "no_accuracy_regression": bool(relational["gate"]["no_accuracy_regression"]),
        "no_severe_underprediction_regression": bool(
            relational["gate"]["no_severe_underprediction_regression"]
        ),
        "severe_underprediction_improved_targets": severe_improved,
        "requires_severe_improvement_targets": 2,
        "helpful": relational["gate"]["helpful"],
        "harmful": relational["gate"]["harmful"],
        "helpful_tasks": relational["gate"]["helpful_tasks"],
        "no_accuracy_regression_vs_collapsed": no_worse_collapsed,
        "no_accuracy_regression_vs_scope_only": no_worse_scope,
        "primary_pair": pair,
    }
    gate["go"] = (
        gate["no_accuracy_regression"]
        and gate["no_severe_underprediction_regression"]
        and severe_improved >= 2
        and gate["helpful"] > gate["harmful"]
        and gate["helpful_tasks"] >= 3
        and no_worse_collapsed
        and no_worse_scope
        and pair["strictly_better_than_collapsed"]
        and pair["strictly_better_than_scope_only"]
    )
    return gate


def _public_inputs(paths: Sequence[Path]) -> list[dict[str, str]]:
    return [
        {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]


def _frozen_public_inputs(
    split: Mapping[str, Any], paths: Sequence[Path]
) -> list[dict[str, str]]:
    declared = split.get("declarative_public_telemetry")
    if not isinstance(declared, list) or len(declared) != len(paths):
        raise ValueError("public evidence differs from the frozen split")
    expected = []
    for row in declared:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("frozen public evidence declaration is invalid")
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = _REPO_ROOT / path
        expected.append({"path": str(path.resolve()), "sha256": str(row["sha256"])})
    if [str(path.resolve()) for path in paths] != [row["path"] for row in expected]:
        raise ValueError("public evidence paths differ from the frozen split")
    actual = _public_inputs(paths)
    if actual != expected:
        raise ValueError("public evidence bytes differ from the frozen split")
    return actual


def _fit_fingerprint(task_ids: Sequence[str], clauses: Sequence[Any], commands: Sequence[Any]) -> str:
    return hashlib.sha256(
        _json_bytes(
            {
                "task_ids": list(task_ids),
                "clauses": [asdict(row) for row in clauses],
                "commands": [asdict(row) for row in commands],
            }
        )
    ).hexdigest()


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _host_identity() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--", *HOST_PATHS)
    if commit.returncode or dirty.returncode or dirty.stdout:
        raise ValueError("result-affecting host code is not committed and clean")
    return {"commit": commit.stdout.strip(), "paths": list(HOST_PATHS)}


def _validate_host_identity(value: Any) -> str:
    if (
        not isinstance(value, dict)
        or value.get("paths") != list(HOST_PATHS)
        or re.fullmatch(r"[0-9a-f]{40}", str(value.get("commit", ""))) is None
    ):
        raise ValueError("frozen host identity is invalid")
    commit = str(value["commit"])
    available = _git("cat-file", "-e", f"{commit}^{{commit}}")
    changed = _git("diff", "--quiet", commit, "--", *HOST_PATHS)
    dirty = _git("status", "--porcelain", "--", *HOST_PATHS)
    if (
        available.returncode
        or changed.returncode
        or dirty.returncode
        or dirty.stdout
    ):
        raise ValueError("result-affecting host code differs from the frozen commit")
    return commit


def _validate_preregistration_commit(value: Any) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", str(value)) is None:
        raise ValueError("frozen preregistration commit is invalid")
    for path in PREREGISTRATION_PATHS:
        relative = path.relative_to(_REPO_ROOT).as_posix()
        blob = subprocess.run(
            ["git", "show", f"{value}:{relative}"],
            cwd=_REPO_ROOT,
            check=False,
            capture_output=True,
        )
        dirty = _git("status", "--porcelain", "--", relative)
        if (
            blob.returncode
            or blob.stdout != path.read_bytes()
            or dirty.returncode
            or dirty.stdout
        ):
            raise ValueError("preregistration differs from the frozen commit")
    return str(value)


def _committed_file(path: Path) -> tuple[bytes, str]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("authorization file is outside the repository") from error
    tracked = _git("ls-files", "--error-unmatch", "--", relative)
    dirty = _git("status", "--porcelain", "--", relative)
    head = _git("rev-parse", "HEAD")
    blob = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
    )
    current = resolved.read_bytes()
    if (
        tracked.returncode
        or dirty.returncode
        or dirty.stdout
        or head.returncode
        or blob.returncode
        or blob.stdout != current
    ):
        raise ValueError("validation authorization is not an unchanged committed file")
    return current, head.stdout.strip()


def _prior_inputs(
    split: Mapping[str, Any], prior_dir: Path
) -> list[dict[str, str]]:
    declared = Path(str(split.get("declarative_prior_dir", "")))
    if not declared.is_absolute():
        declared = _REPO_ROOT / declared
    expected = split.get("declarative_prior_files")
    if prior_dir.resolve() != declared.resolve() or not isinstance(expected, dict):
        raise ValueError("development evidence directory differs from the frozen split")
    rows = []
    for name in ("agent-artifact.json", "training-catalog.json"):
        path = prior_dir / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected.get(name) != digest:
            raise ValueError("development evidence bytes differ from the frozen split")
        rows.append({"path": str(path.resolve()), "sha256": digest})
    if set(expected) != {row["path"].rsplit("/", 1)[-1] for row in rows}:
        raise ValueError("development evidence file set differs from the frozen split")
    return rows


def _artifact_base(
    response: Mapping[str, Any],
    prompt: str,
    split_sha256: str,
    development_fit_sha256: str,
    public_inputs: Sequence[Mapping[str, str]],
    prior_inputs: Sequence[Mapping[str, str]],
    evidence_sha256: str,
    selected_sample_ids: Sequence[str],
    preregistration_commit: str,
    host_identity: Mapping[str, Any],
    generation_files: Sequence[Mapping[str, str]],
    cost: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "claim_bearing": False,
        "model": MODEL,
        "service_tier": "fast",
        "reasoning_effort": "medium",
        "codex_version": subprocess.run(
            ["codex", "--version"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "spec_sha256": hashlib.sha256(_json_bytes(response)).hexdigest(),
        "split_manifest_sha256": split_sha256,
        "development_fit_sha256": development_fit_sha256,
        "public_inputs": list(public_inputs),
        "prior_inputs": list(prior_inputs),
        "development_evidence_sha256": evidence_sha256,
        "development_selected_sample_ids": list(selected_sample_ids),
        "preregistration_commit": preregistration_commit,
        "host_identity": dict(host_identity),
        "generation_files": list(generation_files),
        "generation": dict(response),
        "cost": dict(cost),
        "validation_consumed": False,
        "final_test_consumed": False,
    }


def _write_artifact(out_dir: Path, artifact: Mapping[str, Any]) -> None:
    (out_dir / "artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )


def _validate_generation_binding(
    artifact_dir: Path, artifact: Mapping[str, Any], response: Any
) -> None:
    prompt = (artifact_dir / "generation.prompt.txt").read_text()
    if (
        hashlib.sha256(prompt.encode()).hexdigest() != artifact.get("prompt_sha256")
        or not prompt.startswith(GENERATION_PROMPT)
        or json.loads((artifact_dir / "generation.schema.json").read_text())
        != GENERATION_SCHEMA
    ):
        raise ValueError("frozen prompt or generation schema differs")
    generated = json.loads((artifact_dir / "generation.response.json").read_text())
    if response != generated or artifact.get("generation") != generated:
        raise ValueError("evaluated specification differs from the model response")
    evidence = json.loads(prompt[len(GENERATION_PROMPT) :])
    selected = [str(row["sample_id"]) for row in evidence["selected_rows"]]
    if (
        hashlib.sha256(_json_bytes(evidence)).hexdigest()
        != artifact.get("development_evidence_sha256")
        or selected != artifact.get("development_selected_sample_ids")
    ):
        raise ValueError("frozen prompt evidence differs from the recorded selection")


def _freeze(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    if args.split_manifest.resolve() != SPLIT_MANIFEST.resolve():
        raise ValueError("split manifest path differs from the preregistration")
    preregistration_commits = {
        _committed_file(path)[1] for path in PREREGISTRATION_PATHS
    }
    if len(preregistration_commits) != 1:
        raise ValueError("preregistration files come from different commits")
    split, split_sha256 = _load_split_manifest(args.split_manifest)
    declared = Path(str(split.get("development_run", "")))
    if not declared.is_absolute():
        declared = _REPO_ROOT / declared
    if args.run_dir.resolve() != declared.resolve():
        raise ValueError("development run differs from split manifest")
    host_identity = _host_identity()
    prior_inputs = _prior_inputs(split, args.prior_dir)
    task_ids, clauses, commands = load_run_rows(args.run_dir)
    if task_ids != split["development"]:
        raise ValueError("development task order differs from split manifest")
    development_fit_sha256 = _fit_fingerprint(task_ids, clauses, commands)
    public_inputs = _frozen_public_inputs(split, args.public_telemetry)
    events = _load_exec_events(args.run_dir, list(task_ids))
    queries = build_queries(task_ids, commands, events)
    evidence, private, _prior_artifact = _label_free_selected_evidence(
        task_ids, commands, queries, args.prior_dir
    )
    prompt = GENERATION_PROMPT + json.dumps(evidence, separators=(",", ":"))
    evidence_sha256 = hashlib.sha256(_json_bytes(evidence)).hexdigest()
    if len(prompt.encode()) > MAX_PROMPT_BYTES:
        raise ValueError("declarative generation prompt exceeds its frozen budget")
    args.out_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="declarative-relational-") as directory:
        work = Path(directory)
        response, cost = _codex_call(prompt, GENERATION_SCHEMA, work, "generation")
        for path in work.iterdir():
            if path.is_file():
                (args.out_dir / path.name).write_bytes(path.read_bytes())
    generation_files = [
        {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(args.out_dir.glob("generation.*"))
    ]
    artifact = _artifact_base(
        response,
        prompt,
        split_sha256,
        development_fit_sha256,
        public_inputs,
        prior_inputs,
        evidence_sha256,
        [str(row["sample_id"]) for row in evidence["selected_rows"]],
        next(iter(preregistration_commits)),
        host_identity,
        generation_files,
        cost,
    )
    (args.out_dir / "spec.json").write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n"
    )
    try:
        spec = validate_spec(
            response,
            (*private, *(item["task_id"] for item in evidence["selected_rows"])),
        )
        if spec is None:
            artifact["status"] = "development_structural_no_go_abstained"
            _write_artifact(args.out_dir, artifact)
            return
        samples = [f"{row.task_id}:{row.call_index}" for row in commands]
        task_by_sample = {
            f"{row.task_id}:{row.call_index}": row.task_id for row in commands
        }
        support = validate_development_support(spec, queries, task_by_sample)
        ordered_queries = [queries[sample] for sample in samples]
        extracted, durations, bounded = run_queries(spec, ordered_queries)
        if bounded:
            raise BoundExceeded("development queries exceeded a frozen bound")
        empty, _empty_durations, empty_bounded = run_queries(
            spec, [{**query, "prior_events": []} for query in ordered_queries]
        )
        if empty_bounded or any(value is not None for value in empty):
            raise SpecError("empty history produced a relational state")
        if (
            _scope(spec, "", parse_command_clauses("")) is not None
            or _pattern_spans(spec.blocker, "", "blank_blocker")
            or _pattern_spans(spec.remediation, "", "blank_remediation")
        ):
            raise SpecError("blank semantics matched the specification")
        outputs = dict(zip(samples, extracted, strict=True))
        collapsed_outputs = _collapsed_outputs(outputs)
        scope_outputs = {
            sample: scope_only(spec, queries[sample]) for sample in samples
        }
        relational_pmfs, relational_support = _fit_pmfs(
            task_ids, commands, outputs, warmup_tasks=100
        )
        collapsed_pmfs, collapsed_support = _fit_pmfs(
            task_ids, commands, collapsed_outputs, warmup_tasks=100
        )
        scope_pmfs, scope_support = _fit_pmfs(
            task_ids, commands, scope_outputs, warmup_tasks=100
        )
        primary, state_tasks = _select_primary_contrast(
            commands, outputs, relational_pmfs
        )
        p95_ms = (
            sorted(durations)[min(len(durations) - 1, int(0.95 * len(durations)))]
            / 1_000_000
        )
        artifact.update(
            {
                "status": (
                    "development_structural_go"
                    if primary is not None and p95_ms <= MAX_P95_MS
                    else "development_structural_no_go_no_contrast_or_runtime"
                ),
                "pattern_support": support,
                "primary_contrast": primary,
                "state_tasks": state_tasks,
                "relational_pmfs": _encode_pmfs(relational_pmfs),
                "collapsed_pmfs": _encode_pmfs(collapsed_pmfs),
                "scope_only_pmfs": _encode_pmfs(scope_pmfs),
                "relational_support": relational_support,
                "collapsed_support": collapsed_support,
                "scope_only_support": scope_support,
                "coverage": {
                    "commands": len(commands),
                    "relational_commands": sum(value is not None for value in extracted),
                    "scope_only_commands": sum(
                        value is not None for value in scope_outputs.values()
                    ),
                    "relational_tasks": len(
                        {
                            row.task_id
                            for row, value in zip(commands, extracted, strict=True)
                            if value is not None
                        }
                    ),
                },
                "runtime": {
                    "queries": len(durations),
                    "p50_ms": statistics.median(durations) / 1_000_000,
                    "p95_ms": p95_ms,
                    "p95_limit_ms": MAX_P95_MS,
                },
            }
        )
        (args.out_dir / "fit-graphs.jsonl").write_text(
            "".join(
                json.dumps({"sample_id": sample, "graph": outputs[sample]}, sort_keys=True)
                + "\n"
                for sample in samples
            )
        )
    except SpecError as error:
        artifact.update(
            {
                "status": "development_structural_no_go_invalid_spec",
                "structural_error": str(error),
            }
        )
    _write_artifact(args.out_dir, artifact)


def _load_artifact(
    artifact_dir: Path,
    split_sha256: str,
) -> tuple[dict[str, Any], CompiledSpec, str, str]:
    artifact_bytes = (artifact_dir / "artifact.json").read_bytes()
    artifact = json.loads(artifact_bytes)
    spec_bytes = (artifact_dir / "spec.json").read_bytes()
    response = json.loads(spec_bytes)
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("status") != "development_structural_go"
        or artifact.get("split_manifest_sha256") != split_sha256
        or artifact.get("spec_sha256") != hashlib.sha256(_json_bytes(response)).hexdigest()
    ):
        raise ValueError("frozen declarative artifact is incomplete or differs")
    expected_generation_files = artifact.get("generation_files")
    actual_generation_files = [
        {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(artifact_dir.glob("generation.*"))
    ]
    if expected_generation_files != actual_generation_files:
        raise ValueError("frozen generation transcript or schema differs")
    _validate_generation_binding(artifact_dir, artifact, response)
    artifact_commit = None
    for path in (
        artifact_dir / "artifact.json",
        artifact_dir / "spec.json",
        artifact_dir / "fit-graphs.jsonl",
        *(artifact_dir / row["name"] for row in actual_generation_files),
    ):
        _bytes, commit = _committed_file(path)
        if artifact_commit not in (None, commit):
            raise ValueError("frozen artifact files come from different commits")
        artifact_commit = commit
    host_commit = _validate_host_identity(artifact.get("host_identity"))
    preregistration_commit = _validate_preregistration_commit(
        artifact.get("preregistration_commit")
    )
    if host_commit != preregistration_commit:
        raise ValueError("host and preregistration were not frozen in one commit")
    spec = validate_spec(response)
    if spec is None or artifact.get("primary_contrast") is None:
        raise ValueError("frozen declarative artifact has no usable specification")
    for field in ("relational_pmfs", "collapsed_pmfs", "scope_only_pmfs"):
        _decode_pmfs(artifact.get(field))
    assert artifact_commit is not None
    return artifact, spec, hashlib.sha256(artifact_bytes).hexdigest(), artifact_commit


def _validation_authorizes_final(
    validation: Mapping[str, Any], artifact_sha256: str, split_sha256: str
) -> bool:
    return bool(
        validation.get("schema") == EVALUATION_SCHEMA
        and validation.get("role") == "validation"
        and validation.get("status") == "validation_go"
        and validation.get("claim_bearing") is False
        and validation.get("artifact_sha256") == artifact_sha256
        and validation.get("split_manifest_sha256") == split_sha256
        and validation.get("labels_scored") is True
        and isinstance(validation.get("coverage"), dict)
        and validation["coverage"].get("passed") is True
        and isinstance(validation.get("gate"), dict)
        and validation["gate"].get("go") is True
        and validation.get("row_identity", {}).get("identical_rows") is True
        and re.fullmatch(r"[0-9a-f]{64}", str(validation.get("rows_sha256", "")))
        is not None
    )


def _evaluate(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    if args.split_manifest.resolve() != SPLIT_MANIFEST.resolve():
        raise ValueError("split manifest path differs from the preregistration")
    _committed_file(args.split_manifest)
    split, split_sha256 = _load_split_manifest(args.split_manifest)
    declared_reserved = Path(str(split.get("reserved_run", "")))
    if not declared_reserved.is_absolute():
        declared_reserved = _REPO_ROOT / declared_reserved
    if args.run_dir.resolve() != declared_reserved.resolve():
        raise ValueError("reserved run differs from split manifest")
    artifact, spec, artifact_sha256, artifact_commit = _load_artifact(
        args.artifact_dir, split_sha256
    )
    validation: dict[str, Any] | None = None
    validation_sha256 = None
    validation_commit = None
    if args.role == "final_test":
        if args.validation_result is None:
            raise ValueError("final_test requires a passing validation result")
        validation_bytes, validation_commit = _committed_file(args.validation_result)
        loaded_validation = json.loads(validation_bytes)
        if not isinstance(loaded_validation, dict):
            raise ValueError("validation authorization is not a JSON object")
        validation = loaded_validation
        if not _validation_authorizes_final(
            validation, artifact_sha256, split_sha256
        ):
            raise ValueError("final_test is not authorized by matching validation GO")
        validation_rows, rows_commit = _committed_file(
            args.validation_result.with_name("rows.jsonl")
        )
        if (
            rows_commit != validation_commit
            or hashlib.sha256(validation_rows).hexdigest()
            != validation["rows_sha256"]
        ):
            raise ValueError("validation rows differ from committed authorization")
        validation_sha256 = hashlib.sha256(validation_bytes).hexdigest()
    elif args.validation_result is not None:
        raise ValueError("validation must not consume a prior validation result")

    development_ids, development_clauses, development_commands = load_run_rows(
        args.development_run
    )
    declared_development = Path(str(split.get("development_run", "")))
    if not declared_development.is_absolute():
        declared_development = _REPO_ROOT / declared_development
    development_fit_sha256 = _fit_fingerprint(
        development_ids, development_clauses, development_commands
    )
    public_inputs = _frozen_public_inputs(split, args.public_telemetry)
    declared_prior = Path(str(split.get("declarative_prior_dir", "")))
    if not declared_prior.is_absolute():
        declared_prior = _REPO_ROOT / declared_prior
    prior_inputs = _prior_inputs(split, declared_prior)
    if (
        development_ids != split["development"]
        or args.development_run.resolve() != declared_development.resolve()
        or artifact.get("development_fit_sha256") != development_fit_sha256
        or artifact.get("public_inputs") != public_inputs
        or artifact.get("prior_inputs") != prior_inputs
    ):
        raise ValueError("frozen development or public evidence differs")
    if validation is not None and (
        validation.get("development_fit_sha256") != development_fit_sha256
        or validation.get("development_run") != str(args.development_run.resolve())
        or validation.get("public_inputs") != public_inputs
    ):
        raise ValueError("final_test Current evidence differs from validation")

    role_ids = list(split[args.role])
    records, statuses = _attempt_records(args.run_dir, role_ids)
    valid_records = _telemetry_valid_records(records, statuses)
    valid_ids = [str(record["instance_id"]) for record in valid_records]
    with tempfile.TemporaryDirectory(prefix="declarative-role-") as directory:
        view = Path(directory) / "results.jsonl"
        _write_result_view(valid_records, view)
        events = _load_exec_events(args.run_dir, valid_ids, results_path=view)
        coverage_commands, coverage_queries = _all_command_queries(valid_ids, events)
        ordered_coverage = [
            coverage_queries[f"{row.task_id}:{row.call_index}"]
            for row in coverage_commands
        ]
        coverage_outputs, _coverage_durations, coverage_bounded = run_queries(
            spec, ordered_coverage
        )
        coverage = _coverage(
            valid_ids,
            coverage_commands,
            coverage_outputs,
            artifact["primary_contrast"],
        )
        coverage["bounded_fallbacks"] = coverage_bounded
        if not coverage["passed"]:
            args.out_dir.mkdir(parents=True)
            result = {
                "schema": EVALUATION_SCHEMA,
                "status": f"{args.role}_coverage_no_go",
                "claim_bearing": False,
                "role": args.role,
                "split_manifest_sha256": split_sha256,
                "artifact_sha256": artifact_sha256,
                "artifact_commit": artifact_commit,
                "validation_result_sha256": validation_sha256,
                "validation_result_commit": validation_commit,
                "development_run": str(args.development_run.resolve()),
                "development_fit_sha256": development_fit_sha256,
                "public_inputs": public_inputs,
                "task_statuses": statuses,
                "coverage": coverage,
                "labels_scored": False,
            }
            (args.out_dir / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            return
        if not valid_ids:
            raise ValueError("coverage passed without an evidence-valid task")
        role_task_ids, role_clauses, role_commands = load_run_rows(
            args.run_dir, results_path=view
        )
        if role_task_ids != valid_ids:
            raise AssertionError("role task order differs after telemetry validation")
        role_queries = build_queries(
            role_task_ids,
            role_commands,
            {task_id: events[task_id] for task_id in role_task_ids},
        )
        role_samples = [f"{row.task_id}:{row.call_index}" for row in role_commands]
        extracted, _durations, scored_bounded = run_queries(
            spec, [role_queries[sample] for sample in role_samples]
        )
        outputs = dict(zip(role_samples, extracted, strict=True))
        collapsed_outputs = _collapsed_outputs(outputs)
        scope_outputs = {
            sample: scope_only(spec, role_queries[sample]) for sample in role_samples
        }

        adjusted_clauses, adjusted_commands = _offset_rows(
            role_clauses, role_commands, len(development_ids)
        )
        public = [row for path in args.public_telemetry for row in load_rows(path)]
        excluded_repos = {repo_of(task_id) for task_id in (*development_ids, *role_ids)}
        public = [row for row in public if row.repo not in excluded_repos]
        if not public or {row.task_id for row in public} & set((*development_ids, *role_ids)):
            raise ValueError("public evidence is empty or overlaps SQLGlot tasks")
        baseline, baseline_rows = evaluate_prequential_commands(
            public,
            [*development_ids, *role_task_ids],
            [*development_clauses, *adjusted_clauses],
            [*development_commands, *adjusted_commands],
            {
                "development_run": str(args.development_run.resolve()),
                "reserved_run": str(args.run_dir.resolve()),
                "role": args.role,
            },
            warmup_task_count=100,
        )
    frozen_baseline, frozen_rows = _frozen_baseline(baseline, baseline_rows)
    relational_pmfs = _decode_pmfs(artifact["relational_pmfs"])
    collapsed_pmfs = _decode_pmfs(artifact["collapsed_pmfs"])
    scope_pmfs = _decode_pmfs(artifact["scope_only_pmfs"])
    relational_rows = _apply_candidate(frozen_rows, outputs, relational_pmfs)
    collapsed_rows = _apply_candidate(frozen_rows, collapsed_outputs, collapsed_pmfs)
    scope_rows = _apply_candidate(frozen_rows, scope_outputs, scope_pmfs)
    relational_score = _score_candidate(frozen_baseline, relational_rows)
    collapsed_score = _score_candidate(frozen_baseline, collapsed_rows)
    scope_score = _score_candidate(frozen_baseline, scope_rows)
    gate = _fresh_gate(
        frozen_baseline,
        relational_score,
        collapsed_score,
        scope_score,
        relational_rows,
        collapsed_rows,
        scope_rows,
        artifact["primary_contrast"],
    )
    rows = [
        {
            **row,
            "role": args.role,
            "collapsed_candidate": collapsed_row["candidate"],
            "collapsed_probability_by_bucket": collapsed_row[
                "candidate_probability_by_bucket"
            ],
            "scope_only_candidate": scope_row["candidate"],
            "scope_only_probability_by_bucket": scope_row[
                "candidate_probability_by_bucket"
            ],
        }
        for row, collapsed_row, scope_row in zip(
            relational_rows, collapsed_rows, scope_rows, strict=True
        )
    ]
    rows_bytes = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    ).encode()
    args.out_dir.mkdir(parents=True)
    result = {
        "schema": EVALUATION_SCHEMA,
        "status": f"{args.role}_{'go' if gate['go'] else 'no_go'}",
        "claim_bearing": args.role == "final_test" and gate["go"],
        "role": args.role,
        "split_manifest_sha256": split_sha256,
        "artifact_sha256": artifact_sha256,
        "artifact_commit": artifact_commit,
        "validation_result_sha256": validation_sha256,
        "validation_result_commit": validation_commit,
        "development_run": str(args.development_run.resolve()),
        "development_fit_sha256": development_fit_sha256,
        "public_inputs": public_inputs,
        "task_statuses": statuses,
        "coverage": coverage,
        "scored_bounded_fallbacks": scored_bounded,
        "labels_scored": True,
        "primary_contrast": artifact["primary_contrast"],
        "baseline": {
            "latency": frozen_baseline["latency"]["current_dynamic"],
            "resources": {
                target: frozen_baseline["resources"][target]["current_dynamic"]
                for target in CANONICAL_RESOURCE_BUCKET_EDGES
            },
        },
        "relational": relational_score,
        "collapsed_state": collapsed_score,
        "scope_only": scope_score,
        "gate": gate,
        "row_identity": {
            "identical_rows": [row["sample_id"] for row in relational_rows]
            == [row["sample_id"] for row in collapsed_rows]
            == [row["sample_id"] for row in scope_rows],
            "commands": len(rows),
            "evidence_valid_tasks": len(valid_ids),
        },
        "rows_sha256": hashlib.sha256(rows_bytes).hexdigest(),
    }
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "rows.jsonl").write_bytes(rows_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--run-dir", type=Path, required=True)
    freeze.add_argument("--prior-dir", type=Path, required=True)
    freeze.add_argument("--public-telemetry", type=Path, action="append", required=True)
    freeze.add_argument("--split-manifest", type=Path, required=True)
    freeze.add_argument("--out-dir", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--role", choices=("validation", "final_test"), required=True)
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--development-run", type=Path, required=True)
    evaluate.add_argument(
        "--public-telemetry", type=Path, action="append", required=True
    )
    evaluate.add_argument("--split-manifest", type=Path, required=True)
    evaluate.add_argument("--artifact-dir", type=Path, required=True)
    evaluate.add_argument("--validation-result", type=Path)
    evaluate.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        _freeze(args)
    else:
        _evaluate(args)


if __name__ == "__main__":
    main()
