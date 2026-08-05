#!/usr/bin/env python3
"""Freeze one declarative command-family and dependency-state candidate."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    CANONICAL_RESOURCE_BUCKET_EDGES,
    CommandRow,
    PipExecEvent,
    RESOURCE_BUCKET_LABELS,
    _argmax_probabilities,
    evaluate_prequential_commands,
)
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    load_rows,
    load_run_rows,
)
from scripts.evaluation.evaluate_command_prequential import (  # noqa: E402
    _load_exec_events,
)
from scripts.evaluation.evaluate_declarative_relational_state import (  # noqa: E402
    MAX_BLOCKERS,
    MAX_COMMAND_CHARS,
    MAX_EDGES,
    MAX_PRIOR_EVENTS,
    MAX_RESULT_CHARS,
    MAX_P95_MS,
    MINIMUM_SUPPORT_TASKS,
    SPLIT_MANIFEST,
    BoundExceeded,
    SpecError,
    _committed_file,
    _compile_pattern,
    _decode_pmfs,
    _encode_pmfs,
    _fit_fingerprint,
    _frozen_public_inputs,
    _host_identity,
    _json_bytes,
    _load_split_manifest,
    _pattern_spans,
    _pattern_support,
    _regex_literal_runs,
    _relation_specific_tokens,
    _validate_host_identity,
    _validate_preregistration_commit,
    _write_artifact,
)
from scripts.evaluation.evaluate_offline_agent_extractor import (  # noqa: E402
    MODEL,
    TARGETS,
    _apply_candidate,
    _codex_call,
    _excerpt,
    _fit_pmfs,
    _result_exit_code,
    _score_candidate,
    build_queries,
)
from scripts.evaluation.evaluate_relational_agent_state import (  # noqa: E402
    _all_command_queries,
    _attempt_records,
    _collapse_whitespace,
    _derive_state,
    _frozen_baseline,
    _offset_rows,
    _telemetry_valid_records,
    _write_result_view,
)
from tool_resource.runtime_kb import parse_command_clauses  # noqa: E402
from tool_resource_eval.labels import repo_of  # noqa: E402

SCHEMA = "offline-agent-declarative-family-state-v1"
EVALUATION_SCHEMA = "declarative-family-state-fresh-evaluation-v1"
MAX_PROMPT_BYTES = 180_000
MAX_CANDIDATE_FAMILIES = 4
MAX_EVIDENCE_TASKS = 12
MAX_EPISODE_EVENTS = 48
MAX_SCOPES = 4
MAX_PATTERNS_PER_SCOPE = 4
MAX_SCOPE_PATTERNS = 8
MAX_RELATION_PATTERNS = 6
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
        "family_id",
        "scopes",
        "blocker_patterns",
        "remediation_patterns",
        "explanation",
    ],
    "properties": {
        "abstain": {"type": "boolean"},
        "family_id": {"type": "string", "maxLength": 40},
        "scopes": {
            "type": "array",
            "maxItems": MAX_SCOPES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["scope_id", "patterns"],
                "properties": {
                    "scope_id": {"type": "string", "maxLength": 40},
                    "patterns": {
                        "type": "array",
                        "maxItems": MAX_PATTERNS_PER_SCOPE,
                        "items": {"type": "string", "maxLength": 256},
                    },
                },
            },
        },
        "blocker_patterns": {
            "type": "array",
            "maxItems": MAX_RELATION_PATTERNS,
            "items": {"type": "string", "maxLength": 512},
        },
        "remediation_patterns": {
            "type": "array",
            "maxItems": MAX_RELATION_PATTERNS,
            "items": {"type": "string", "maxLength": 512},
        },
        "explanation": {"type": "string", "maxLength": 1_000},
    },
}

GENERATION_PROMPT = r"""Generate one reusable command-family specification
from the label-free causal episodes below. Return only the required JSON.

The host, not you, will parse every event, match the current work scope, scan
all earlier family verifiers, join a successful remediation to a blocker only
on exact normalized identifier equality, derive dependency states, fit resource
PMFs, and fall back to Current. You cannot output code, buckets, states,
relations, weights, thresholds, or per-query decisions.

Choose one family visible in the evidence. Define 2-4 mutually exclusive work
scopes that preserve a meaningful difference in requested work, such as a
bounded probe versus broader execution. The union of the scopes is the family.
Each scope pattern must repeat across five development tasks. Do not include
task aliases or file, test, package, or repository-specific literals.

Scope patterns are case-insensitive ASCII regular expressions with no capture
groups and are full-matched against whitespace-collapsed commands.
blocker_patterns and remediation_patterns each contain exactly one named
capture (?P<id>...) and no other group. They run with finditer on one failed
family-verifier result excerpt or one successful command. Each relation pattern
must repeat across five tasks and capture at least two distinct identifiers.

Allowed regex features are ASCII literals, escaped literals, character classes
and ranges, standard character categories, outer string/word anchors, and
bounded repeats with upper bound at most 128. Do not use alternation, wildcard
dot, lookaround, backreferences, inline flags, unbounded repeats, nested
repeats, noncapturing groups, or any group other than (?P<id>...). Use multiple
patterns instead of alternation. Do not spell one case-insensitive literal as a
character class.

Return abstain=true with empty family_id and arrays when the contract cannot
express a reusable family with two work scopes. Do not call tools.

LABEL-FREE COMPRESSED DEVELOPMENT EPISODES:
"""


@dataclass(frozen=True)
class ScopeSpec:
    scope_id: str
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    scopes: tuple[ScopeSpec, ...]
    blocker: tuple[re.Pattern[str], ...]
    remediation: tuple[re.Pattern[str], ...]


def _normalize_key(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "-", value.casefold()).strip("-")


def invocation_key(command: str) -> str | None:
    parsed = parse_command_clauses(command)
    clauses = parsed.get("clauses", ())
    if (
        parsed.get("parse_failed")
        or len(clauses) != 1
        or any(clauses[0].get(field) for field in ("in_pipe", "in_loop", "in_subst"))
    ):
        return None
    argv = [str(value) for value in clauses[0].get("argv", ())]
    if not argv:
        return None
    value = (
        argv[2] if len(argv) >= 3 and argv[1] == "-m" else argv[0].rsplit("/", 1)[-1]
    )
    return _normalize_key(value) or None


def _trim(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n<omitted>\n"
    head = (limit - len(marker)) // 2
    return value[:head] + marker + value[-(limit - len(marker) - head) :]


def _episode(events: Sequence[PipExecEvent], key: str) -> list[dict[str, Any]] | None:
    keys = [invocation_key(event.command) for event in events]
    failures = [
        index
        for index, event in enumerate(events)
        if keys[index] == key
        and _result_exit_code(event.tool_result) not in (0, None)
        and key in keys[index + 1 :]
    ]
    if not failures:
        return None
    start = failures[0]
    end = max(index for index, value in enumerate(keys) if value == key)
    selected = [
        (index, event)
        for index, event in enumerate(events[start : end + 1], start)
        if keys[index] == key or _result_exit_code(event.tool_result) == 0
    ]
    if len(selected) > MAX_EPISODE_EVENTS:
        selected = [*selected[:24], *selected[-24:]]
    rows = []
    for index, event in selected:
        exit_code = _result_exit_code(event.tool_result)
        verifier = keys[index] == key
        rows.append(
            {
                "event_index": index,
                "kind": "family_verifier" if verifier else "successful_action",
                "command": _trim(event.command, 800),
                "exit_code": exit_code,
                "result_excerpt": (
                    _trim(_excerpt(event.tool_result), 500)
                    if verifier and exit_code not in (0, None)
                    else ""
                ),
            }
        )
    return rows


def _redact_task_ids(
    value: str, task_ids: Sequence[str], aliases: Mapping[str, str]
) -> str:
    for task_id in sorted(task_ids, key=lambda item: (-len(item), item)):
        value = re.sub(
            re.escape(task_id),
            aliases.get(task_id, "<task>"),
            value,
            flags=re.IGNORECASE,
        )
    return value


def build_evidence(
    task_ids: Sequence[str],
    events_by_task: Mapping[str, Sequence[PipExecEvent]],
) -> dict[str, Any]:
    if list(events_by_task) != list(task_ids):
        raise ValueError("event tasks differ from the committed development order")
    episodes: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for task_id in task_ids:
        keys = sorted(
            {
                key
                for event in events_by_task[task_id]
                if (key := invocation_key(event.command))
            }
        )
        for key in keys:
            episode = _episode(events_by_task[task_id], key)
            if episode is not None:
                episodes[key].append((task_id, episode))
    candidates = sorted(
        (
            (key, rows)
            for key, rows in episodes.items()
            if len(rows) >= MINIMUM_SUPPORT_TASKS
        ),
        key=lambda item: (-len(item[1]), item[0]),
    )[:MAX_CANDIDATE_FAMILIES]
    if len(candidates) < 2:
        raise SpecError(
            "fewer than two invocation families have five-task episode support"
        )
    retained_tasks = []
    for _key, rows in candidates:
        for task_id, _episode_rows in rows[:MAX_EVIDENCE_TASKS]:
            if task_id not in retained_tasks:
                retained_tasks.append(task_id)
    task_aliases = {
        task_id: f"T{index:03d}" for index, task_id in enumerate(retained_tasks)
    }
    output = []
    for family_index, (key, rows) in enumerate(candidates):
        rendered = []
        for task_id, episode in rows[:MAX_EVIDENCE_TASKS]:
            rendered.append(
                {
                    "task_alias": task_aliases[task_id],
                    "events": [
                        {
                            **event,
                            "command": _redact_task_ids(
                                event["command"], task_ids, task_aliases
                            ),
                            "result_excerpt": _redact_task_ids(
                                event["result_excerpt"], task_ids, task_aliases
                            ),
                        }
                        for event in episode
                    ],
                }
            )
        output.append(
            {
                "candidate_alias": f"F{family_index:03d}",
                "invocation_key": key,
                "qualifying_task_count": len(rows),
                "episodes": rendered,
            }
        )
    evidence = {
        "schema": "label-free-family-episodes-v1",
        "candidates": output,
        "omitted": ["durations", "telemetry", "resource_labels", "current_predictions"],
    }
    serialized = json.dumps(evidence, sort_keys=True).casefold()
    if any(task_id.casefold() in serialized for task_id in task_ids):
        raise SpecError("compressed evidence retains an original task ID")
    return evidence


def build_generation_input(
    task_ids: Sequence[str],
    events_by_task: Mapping[str, Sequence[PipExecEvent]],
) -> tuple[dict[str, Any], str]:
    evidence = build_evidence(task_ids, events_by_task)
    prompt = GENERATION_PROMPT + json.dumps(evidence, separators=(",", ":"))
    if len(prompt.encode()) > MAX_PROMPT_BYTES:
        raise SpecError("family generation prompt exceeds its frozen byte budget")
    return evidence, prompt


def _opaque_literals(forbidden_ids: Sequence[str]) -> set[str]:
    literals = {identifier.casefold() for identifier in forbidden_ids}
    for identifier in forbidden_ids:
        repository = repo_of(identifier)
        if "__" not in repository:
            continue
        literals.update(
            token.casefold() for token in re.findall(r"[A-Za-z]{4,}", repository)
        )
    return literals


def validate_spec(value: Any, forbidden_ids: Sequence[str] = ()) -> FamilySpec | None:
    required = set(GENERATION_SCHEMA["required"])
    if not isinstance(value, dict) or set(value) != required:
        raise SpecError("generation differs from the frozen JSON schema")
    if (
        not isinstance(value["abstain"], bool)
        or not isinstance(value["family_id"], str)
        or not isinstance(value["scopes"], list)
        or not isinstance(value["blocker_patterns"], list)
        or not isinstance(value["remediation_patterns"], list)
        or not isinstance(value["explanation"], str)
    ):
        raise SpecError("generation has invalid JSON value types")
    serialized = json.dumps(value, sort_keys=True).casefold()
    if any(literal in serialized for literal in _opaque_literals(forbidden_ids)):
        raise SpecError("generation contains an opaque development or repository ID")
    arrays_empty = (
        not value["scopes"]
        and not value["blocker_patterns"]
        and not value["remediation_patterns"]
    )
    if not 1 <= len(value["explanation"]) <= 1_000:
        raise SpecError("explanation is empty or too long")
    if value["abstain"]:
        if value["family_id"] or not arrays_empty:
            raise SpecError("abstention contains a family or pattern")
        return None
    if re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", value["family_id"]) is None:
        raise SpecError("family_id is invalid")
    if not 2 <= len(value["scopes"]) <= MAX_SCOPES:
        raise SpecError("scope cardinality is invalid")
    scopes = []
    scope_ids = set()
    scope_pattern_count = 0
    all_scope_patterns = []
    for row in value["scopes"]:
        if not isinstance(row, dict) or set(row) != {"scope_id", "patterns"}:
            raise SpecError("scope differs from the frozen JSON schema")
        scope_id = row["scope_id"]
        patterns = row["patterns"]
        if (
            not isinstance(scope_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", scope_id) is None
            or scope_id in scope_ids
            or not isinstance(patterns, list)
            or not 1 <= len(patterns) <= MAX_PATTERNS_PER_SCOPE
            or any(not isinstance(pattern, str) for pattern in patterns)
            or len(patterns) != len(set(patterns))
        ):
            raise SpecError("scope ID or pattern array is invalid")
        scope_ids.add(scope_id)
        scope_pattern_count += len(patterns)
        all_scope_patterns.extend(patterns)
        scopes.append(
            ScopeSpec(
                scope_id,
                tuple(_compile_pattern(pattern, kind="scope") for pattern in patterns),
            )
        )
    if scope_pattern_count > MAX_SCOPE_PATTERNS or len(all_scope_patterns) != len(
        set(all_scope_patterns)
    ):
        raise SpecError("scope patterns exceed the total bound or overlap by identity")
    relation = []
    for field, kind in (
        ("blocker_patterns", "blocker"),
        ("remediation_patterns", "remediation"),
    ):
        patterns = value[field]
        if (
            not 1 <= len(patterns) <= MAX_RELATION_PATTERNS
            or any(not isinstance(pattern, str) for pattern in patterns)
            or len(patterns) != len(set(patterns))
        ):
            raise SpecError(f"{kind} pattern array is invalid")
        relation.append(
            tuple(_compile_pattern(pattern, kind=kind) for pattern in patterns)
        )
    spec = FamilySpec(value["family_id"], tuple(scopes), relation[0], relation[1])
    opaque = _opaque_literals(forbidden_ids)
    if any(
        literal in run.casefold()
        for pattern in (
            *(item for scope in spec.scopes for item in scope.patterns),
            *spec.blocker,
            *spec.remediation,
        )
        for run in _regex_literal_runs(pattern)
        for literal in opaque
    ):
        raise SpecError("generation encodes an opaque development or repository ID")
    return spec


def match_scope(spec: FamilySpec, command: str) -> str | None:
    parsed = parse_command_clauses(command)
    clauses = parsed.get("clauses", ())
    if (
        parsed.get("parse_failed")
        or len(clauses) != 1
        or any(clauses[0].get(field) for field in ("in_pipe", "in_loop", "in_subst"))
    ):
        return None
    collapsed = _collapse_whitespace(command)
    matched = {
        scope.scope_id
        for scope in spec.scopes
        if any(pattern.fullmatch(collapsed) for pattern in scope.patterns)
    }
    if len(matched) > 1:
        raise SpecError("command matches more than one work scope")
    return next(iter(matched), None)


def _query_exceeds_bounds(query: Mapping[str, Any]) -> bool:
    return (
        len(str(query["current_command"])) > MAX_COMMAND_CHARS
        or len(query["prior_events"]) > MAX_PRIOR_EVENTS
        or any(
            len(str(event["command"])) > MAX_COMMAND_CHARS
            or len(str(event["result_excerpt"])) > MAX_RESULT_CHARS
            for event in query["prior_events"]
        )
    )


def extract_query(
    spec: FamilySpec, query: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, bool]:
    command = str(query["current_command"])
    if _query_exceeds_bounds(query):
        return None, True
    scope_id = match_scope(spec, command)
    if scope_id is None:
        return None, False
    namespace = {
        "scope": lambda _command, _parsed: spec.family_id,
        "blocker_spans": lambda text: _pattern_spans(spec.blocker, text, "blocker"),
        "remediation_spans": lambda text: _pattern_spans(
            spec.remediation, text, "remediation"
        ),
    }
    try:
        graph = _derive_state(
            query,
            namespace,
            verifier_match=lambda prior: match_scope(spec, prior) is not None,
        )
    except SpecError:
        return None, True
    if graph is None:
        return None, False
    edge_count = sum(len(item["addresses"]) for item in graph["remediations"])
    if len(graph["blockers"]) > MAX_BLOCKERS or edge_count > MAX_EDGES:
        return None, True
    dependency_state = str(graph["state"])
    return {
        **graph,
        "scope_id": scope_id,
        "dependency_state": dependency_state,
        "state": f"{scope_id}::{dependency_state}",
    }, False


def run_queries(
    spec: FamilySpec, queries: Sequence[Mapping[str, Any]]
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


def _collapsed_outputs(
    outputs: Mapping[str, Mapping[str, Any] | None], *, dimension: str
) -> dict[str, dict[str, Any] | None]:
    return {
        sample: (
            None
            if value is None
            else {
                **value,
                "state": (
                    "__family__"
                    if dimension == "family"
                    else f"scope::{value['scope_id']}"
                ),
            }
        )
        for sample, value in outputs.items()
    }


def validate_development_support(
    spec: FamilySpec,
    queries: Mapping[str, Mapping[str, Any]],
    task_by_sample: Mapping[str, str],
) -> dict[str, Any]:
    commands: dict[str, set[str]] = defaultdict(set)
    all_commands: dict[str, set[str]] = defaultdict(set)
    failed_results: dict[str, set[str]] = defaultdict(set)
    successful_commands: dict[str, set[str]] = defaultdict(set)
    scope_tasks: dict[str, set[str]] = defaultdict(set)
    for sample_id, query in queries.items():
        if _query_exceeds_bounds(query):
            raise BoundExceeded("development support query exceeds a frozen bound")
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
        scope_id = match_scope(spec, current)
        if scope_id is not None:
            scope_tasks[scope_id].add(task_id)
        for event in query["prior_events"]:
            command = str(event["command"])
            all_commands[task_id].add(command)
            prior_scope = match_scope(spec, command)
            if prior_scope is not None:
                commands[task_id].add(command)
                scope_tasks[prior_scope].add(task_id)
                if event["exit_code"] not in (0, None):
                    failed_results[task_id].add(str(event["result_excerpt"]))
            if event["exit_code"] == 0:
                successful_commands[task_id].add(command)
    report: dict[str, Any] = {"scopes": {}, "blocker": [], "remediation": []}
    for scope in spec.scopes:
        rows = []
        for pattern in scope.patterns:
            tasks, _identifiers = _pattern_support(pattern, commands, scope=True)
            if len(tasks) < MINIMUM_SUPPORT_TASKS:
                raise SpecError("scope pattern lacks five-task support")
            rows.append({"pattern": pattern.pattern, "tasks": len(tasks)})
        if len(scope_tasks[scope.scope_id]) < MINIMUM_SUPPORT_TASKS:
            raise SpecError("scope lacks five-task support")
        report["scopes"][scope.scope_id] = {
            "tasks": len(scope_tasks[scope.scope_id]),
            "patterns": rows,
        }
    for kind, patterns, texts in (
        ("blocker", spec.blocker, failed_results),
        ("remediation", spec.remediation, successful_commands),
    ):
        for pattern in patterns:
            tasks, identifiers = _pattern_support(pattern, texts, scope=False)
            if len(tasks) < MINIMUM_SUPPORT_TASKS:
                raise SpecError(f"{kind} pattern lacks five-task support")
            if len(identifiers) < 2:
                raise SpecError(f"{kind} pattern captures fewer than two identifiers")
            report[kind].append(
                {
                    "pattern": pattern.pattern,
                    "tasks": len(tasks),
                    "identifiers": len(identifiers),
                }
            )
    for pattern in spec.remediation:
        for task_commands in successful_commands.values():
            for command in task_commands:
                matches = list(pattern.finditer(command))
                if not matches:
                    continue
                identifiers = {_normalize_key(match.group("id")) for match in matches}
                for token in _relation_specific_tokens(command):
                    if _normalize_key(token) in identifiers:
                        continue
                    if not list(pattern.finditer(command.replace(token, "__ARG__"))):
                        raise SpecError(
                            "remediation pattern depends on a specific argument"
                        )
    specific = set()
    generic = set()
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
    patterns = (
        *(pattern for scope in spec.scopes for pattern in scope.patterns),
        *spec.blocker,
        *spec.remediation,
    )
    for pattern in patterns:
        for run in _regex_literal_runs(pattern):
            for token in re.findall(r"[A-Za-z0-9_.+-]{2,}", run):
                if token.casefold() in specific and token.casefold() not in generic:
                    raise SpecError("pattern contains a specific positional literal")
    return report


def select_primary_contrasts(
    commands: Sequence[CommandRow],
    outputs: Mapping[str, Mapping[str, Any] | None],
    pmfs: Mapping[tuple[tuple[str, str], str], tuple[float, ...]],
) -> dict[str, Any] | None:
    family_ids = {str(value["rule_id"]) for value in outputs.values() if value}
    if len(family_ids) != 1:
        return None
    family_id = next(iter(family_ids))
    tasks: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in commands:
        value = outputs[f"{row.task_id}:{row.call_index}"]
        if value is not None:
            tasks[(str(value["scope_id"]), str(value["dependency_state"]))].add(
                row.task_id
            )

    def differing(
        scope: str, state: str, other_scope: str, other_state: str
    ) -> list[str]:
        left = (family_id, f"{scope}::{state}")
        right = (family_id, f"{other_scope}::{other_state}")
        return [
            target
            for target in TARGETS
            if (left, target) in pmfs
            and (right, target) in pmfs
            and _argmax_probabilities(pmfs[(left, target)])
            != _argmax_probabilities(pmfs[(right, target)])
        ]

    relation_candidates = []
    by_scope: dict[str, list[str]] = defaultdict(list)
    by_state: dict[str, list[str]] = defaultdict(list)
    for scope, state in tasks:
        if len(tasks[(scope, state)]) >= MINIMUM_SUPPORT_TASKS:
            by_scope[scope].append(state)
            by_state[state].append(scope)
    for scope, states in by_scope.items():
        for left, right in itertools.combinations(sorted(set(states)), 2):
            targets = differing(scope, left, scope, right)
            if targets:
                support = (len(tasks[(scope, left)]), len(tasks[(scope, right)]))
                relation_candidates.append(
                    (
                        -min(support),
                        -sum(support),
                        family_id,
                        scope,
                        left,
                        right,
                        targets[0],
                        support,
                    )
                )
    scope_candidates = []
    for state, scopes in by_state.items():
        for left, right in itertools.combinations(sorted(set(scopes)), 2):
            targets = differing(left, state, right, state)
            if targets:
                support = (len(tasks[(left, state)]), len(tasks[(right, state)]))
                scope_candidates.append(
                    (
                        -min(support),
                        -sum(support),
                        family_id,
                        state,
                        left,
                        right,
                        targets[0],
                        support,
                    )
                )
    if not relation_candidates or not scope_candidates:
        return None
    relation = min(relation_candidates)
    scope = min(scope_candidates)
    return {
        "relation": {
            "family_id": relation[2],
            "scope_id": relation[3],
            "states": [relation[4], relation[5]],
            "target": relation[6],
            "task_support": list(relation[7]),
        },
        "scope": {
            "family_id": scope[2],
            "dependency_state": scope[3],
            "scopes": [scope[4], scope[5]],
            "target": scope[6],
            "task_support": list(scope[7]),
        },
    }


def _artifact_base(
    response: Mapping[str, Any],
    prompt: str,
    evidence: Mapping[str, Any],
    split_sha256: str,
    preregistration_commit: str,
    host_identity: Mapping[str, Any],
    generation_files: Sequence[Mapping[str, str]],
    cost: Mapping[str, Any],
    public_inputs: Sequence[Mapping[str, str]],
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
        "development_evidence_sha256": hashlib.sha256(
            _json_bytes(evidence)
        ).hexdigest(),
        "split_manifest_sha256": split_sha256,
        "preregistration_commit": preregistration_commit,
        "host_identity": dict(host_identity),
        "generation_files": list(generation_files),
        "generation": dict(response),
        "cost": dict(cost),
        "public_inputs": list(public_inputs),
        "validation_consumed": False,
        "final_test_consumed": False,
    }


def freeze(args: argparse.Namespace) -> None:
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
    declared = Path(str(split["development_run"]))
    if not declared.is_absolute():
        declared = _REPO_ROOT / declared
    if args.run_dir.resolve() != declared.resolve():
        raise ValueError("development run differs from the frozen split")
    host_identity = _host_identity()
    public_inputs = _frozen_public_inputs(split, args.public_telemetry)
    events = _load_exec_events(args.run_dir, list(split["development"]))
    evidence, prompt = build_generation_input(split["development"], events)

    args.out_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="declarative-family-") as directory:
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
        evidence,
        split_sha256,
        next(iter(preregistration_commits)),
        host_identity,
        generation_files,
        cost,
        public_inputs,
    )
    artifact["development_task_ids_sha256"] = hashlib.sha256(
        _json_bytes(split["development"])
    ).hexdigest()
    (args.out_dir / "spec.json").write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n"
    )
    try:
        evidence_aliases = [
            str(episode["task_alias"])
            for candidate in evidence["candidates"]
            for episode in candidate["episodes"]
        ] + [str(candidate["candidate_alias"]) for candidate in evidence["candidates"]]
        spec = validate_spec(response, (*split["development"], *evidence_aliases))
        if spec is None:
            artifact["status"] = "development_structural_no_go_abstained"
            _write_artifact(args.out_dir, artifact)
            return
        task_ids, clauses, commands = load_run_rows(args.run_dir)
        if task_ids != split["development"]:
            raise ValueError("development task order differs from the frozen split")
        artifact["development_fit_sha256"] = _fit_fingerprint(
            task_ids, clauses, commands
        )
        queries = build_queries(task_ids, commands, events)
        samples = [f"{row.task_id}:{row.call_index}" for row in commands]
        task_by_sample = {
            f"{row.task_id}:{row.call_index}": row.task_id for row in commands
        }
        support = validate_development_support(spec, queries, task_by_sample)
        ordered = [queries[sample] for sample in samples]
        extracted, durations, bounded = run_queries(spec, ordered)
        if bounded:
            raise BoundExceeded("development queries exceeded a frozen bound")
        empty, _empty_durations, empty_bounded = run_queries(
            spec, [{**query, "prior_events": []} for query in ordered]
        )
        if empty_bounded or any(value is not None for value in empty):
            raise SpecError("empty history produced a family relation state")
        outputs = dict(zip(samples, extracted, strict=True))
        family_outputs = _collapsed_outputs(outputs, dimension="family")
        scope_outputs = _collapsed_outputs(outputs, dimension="scope")
        full_pmfs, full_support = _fit_pmfs(
            task_ids, commands, outputs, warmup_tasks=100
        )
        family_pmfs, family_support = _fit_pmfs(
            task_ids, commands, family_outputs, warmup_tasks=100
        )
        scope_pmfs, scope_support = _fit_pmfs(
            task_ids, commands, scope_outputs, warmup_tasks=100
        )
        contrasts = select_primary_contrasts(commands, outputs, full_pmfs)
        p95_ms = (
            sorted(durations)[min(len(durations) - 1, int(0.95 * len(durations)))]
            / 1_000_000
        )
        artifact.update(
            {
                "status": (
                    "development_structural_go"
                    if contrasts is not None and p95_ms <= MAX_P95_MS
                    else "development_structural_no_go_no_contrasts_or_runtime"
                ),
                "pattern_support": support,
                "primary_contrasts": contrasts,
                "full_pmfs": _encode_pmfs(full_pmfs),
                "family_only_pmfs": _encode_pmfs(family_pmfs),
                "scope_only_pmfs": _encode_pmfs(scope_pmfs),
                "full_support": full_support,
                "family_only_support": family_support,
                "scope_only_support": scope_support,
                "coverage": {
                    "commands": len(commands),
                    "relation_commands": sum(value is not None for value in extracted),
                    "relation_tasks": len(
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
                json.dumps(
                    {"sample_id": sample, "graph": outputs[sample]}, sort_keys=True
                )
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


def _validate_generation_binding(
    artifact_dir: Path,
    artifact: Mapping[str, Any],
    response: Any,
    development_ids: Sequence[str],
) -> None:
    prompt = (artifact_dir / "generation.prompt.txt").read_text()
    schema = json.loads((artifact_dir / "generation.schema.json").read_text())
    generated = json.loads((artifact_dir / "generation.response.json").read_text())
    if (
        not prompt.startswith(GENERATION_PROMPT)
        or hashlib.sha256(prompt.encode()).hexdigest() != artifact.get("prompt_sha256")
        or schema != GENERATION_SCHEMA
        or generated != response
        or generated != artifact.get("generation")
        or artifact.get("cost", {}).get("prompt_bytes") != len(prompt.encode())
    ):
        raise ValueError("frozen generation prompt, schema, response, or cost differs")
    evidence = json.loads(prompt[len(GENERATION_PROMPT) :])
    serialized = json.dumps(evidence, sort_keys=True).casefold()
    if hashlib.sha256(_json_bytes(evidence)).hexdigest() != artifact.get(
        "development_evidence_sha256"
    ) or any(task_id.casefold() in serialized for task_id in development_ids):
        raise ValueError("frozen compressed evidence differs or exposes a task ID")


def _load_artifact(
    artifact_dir: Path,
    split: Mapping[str, Any],
    split_sha256: str,
) -> tuple[dict[str, Any], FamilySpec, str, str]:
    artifact_bytes = (artifact_dir / "artifact.json").read_bytes()
    artifact = json.loads(artifact_bytes)
    response = json.loads((artifact_dir / "spec.json").read_text())
    expected_task_hash = hashlib.sha256(_json_bytes(split["development"])).hexdigest()
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("status") != "development_structural_go"
        or artifact.get("split_manifest_sha256") != split_sha256
        or artifact.get("development_task_ids_sha256") != expected_task_hash
        or artifact.get("spec_sha256")
        != hashlib.sha256(_json_bytes(response)).hexdigest()
        or not isinstance(artifact.get("primary_contrasts"), dict)
    ):
        raise ValueError("frozen family artifact is incomplete or differs")
    actual_generation = [
        {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(artifact_dir.glob("generation.*"))
    ]
    if artifact.get("generation_files") != actual_generation:
        raise ValueError("frozen generation transcript differs")
    _validate_generation_binding(artifact_dir, artifact, response, split["development"])
    aliases = []
    evidence = json.loads(
        (artifact_dir / "generation.prompt.txt").read_text()[len(GENERATION_PROMPT) :]
    )
    for candidate in evidence["candidates"]:
        aliases.append(str(candidate["candidate_alias"]))
        aliases.extend(str(episode["task_alias"]) for episode in candidate["episodes"])
    spec = validate_spec(response, (*split["development"], *aliases))
    if spec is None:
        raise ValueError("frozen family artifact contains an abstention")
    for field in ("full_pmfs", "family_only_pmfs", "scope_only_pmfs"):
        _decode_pmfs(artifact.get(field))
    artifact_commit = None
    for path in (
        artifact_dir / "artifact.json",
        artifact_dir / "spec.json",
        artifact_dir / "fit-graphs.jsonl",
        *(artifact_dir / row["name"] for row in actual_generation),
    ):
        _bytes, commit = _committed_file(path)
        if artifact_commit not in (None, commit):
            raise ValueError("frozen artifact files come from different commits")
        artifact_commit = commit
    _validate_host_identity(artifact.get("host_identity"))
    _validate_preregistration_commit(artifact.get("preregistration_commit"))
    assert artifact_commit is not None
    return artifact, spec, hashlib.sha256(artifact_bytes).hexdigest(), artifact_commit


def _coverage(
    commands: Sequence[CommandRow],
    outputs: Sequence[Mapping[str, Any] | None],
    primary: Mapping[str, Any],
    bounded: int,
) -> dict[str, Any]:
    carrier = [
        (row, value)
        for row, value in zip(commands, outputs, strict=True)
        if value is not None
    ]
    relation = primary["relation"]
    relation_counts = {}
    for state in relation["states"]:
        selected = [
            row
            for row, value in carrier
            if value["rule_id"] == relation["family_id"]
            and value["scope_id"] == relation["scope_id"]
            and value["dependency_state"] == state
        ]
        relation_counts[state] = {
            "commands": len(selected),
            "tasks": len({row.task_id for row in selected}),
        }
    scope = primary["scope"]
    scope_counts = {}
    for scope_id in scope["scopes"]:
        selected = [
            row
            for row, value in carrier
            if value["rule_id"] == scope["family_id"]
            and value["scope_id"] == scope_id
            and value["dependency_state"] == scope["dependency_state"]
        ]
        scope_counts[scope_id] = {
            "commands": len(selected),
            "tasks": len({row.task_id for row in selected}),
        }
    carrier_tasks = {row.task_id for row, _value in carrier}
    cells = [*relation_counts.values(), *scope_counts.values()]
    passed = (
        bounded == 0
        and len(carrier) >= 20
        and len(carrier_tasks) >= 5
        and all(cell["commands"] >= 5 and cell["tasks"] >= 3 for cell in cells)
    )
    return {
        "passed": passed,
        "all_exec_commands": len(commands),
        "carrier_commands": len(carrier),
        "carrier_tasks": len(carrier_tasks),
        "relation_counts": relation_counts,
        "scope_counts": scope_counts,
        "bounded_fallbacks": bounded,
        "minimum_carrier_commands": 20,
        "minimum_carrier_tasks": 5,
        "minimum_cell_commands": 5,
        "minimum_cell_tasks": 3,
    }


def _accuracy(score: Mapping[str, Any], target: str) -> float:
    key = "exact_class_accuracy" if target == "latency" else "accuracy"
    return float(score["metrics"][target][key])


def _prediction_bucket(value: Any, target: str) -> int:
    return (
        int(value) if target == "latency" else RESOURCE_BUCKET_LABELS.index(str(value))
    )


def _paired_correctness(
    selector_rows: Sequence[Mapping[str, Any]],
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    contrast: Mapping[str, Any],
    *,
    kind: str,
) -> dict[str, Any]:
    target = str(contrast["target"])
    if kind == "relation":
        selected_states = {
            f"{contrast['scope_id']}::{state}" for state in contrast["states"]
        }
    else:
        selected_states = {
            f"{scope_id}::{contrast['dependency_state']}"
            for scope_id in contrast["scopes"]
        }
    result = {"target": target, "commands": 0, "left_correct": 0, "right_correct": 0}
    for selector, left, right in zip(selector_rows, left_rows, right_rows, strict=True):
        signature = selector["generated_signature"]
        truth = selector["labels"][target]
        if (
            signature is None
            or signature["rule_id"] != contrast["family_id"]
            or signature["state"] not in selected_states
            or truth is None
        ):
            continue
        result["commands"] += 1
        result["left_correct"] += _prediction_bucket(
            left["candidate"][target], target
        ) == int(truth)
        result["right_correct"] += _prediction_bucket(
            right["candidate"][target], target
        ) == int(truth)
    result["strictly_better"] = result["left_correct"] > result["right_correct"]
    return result


def _fresh_gate(
    baseline: Mapping[str, Any],
    full: Mapping[str, Any],
    family: Mapping[str, Any],
    scope: Mapping[str, Any],
    full_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
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
        full["metrics"][target]["severe_underprediction_rate"]
        < reference_severe[target]
        for target in TARGETS
    )
    no_worse_family = all(
        _accuracy(full, target) >= _accuracy(family, target) for target in TARGETS
    )
    no_worse_scope = all(
        _accuracy(full, target) >= _accuracy(scope, target) for target in TARGETS
    )
    relation_pair = _paired_correctness(
        full_rows,
        full_rows,
        scope_rows,
        primary["relation"],
        kind="relation",
    )
    scope_pair = _paired_correctness(
        full_rows,
        scope_rows,
        family_rows,
        primary["scope"],
        kind="scope",
    )
    gate = {
        "no_accuracy_regression": bool(full["gate"]["no_accuracy_regression"]),
        "no_severe_underprediction_regression": bool(
            full["gate"]["no_severe_underprediction_regression"]
        ),
        "severe_underprediction_improved_targets": severe_improved,
        "requires_severe_improvement_targets": 2,
        "helpful": full["gate"]["helpful"],
        "harmful": full["gate"]["harmful"],
        "helpful_tasks": full["gate"]["helpful_tasks"],
        "no_accuracy_regression_vs_family_only": no_worse_family,
        "no_accuracy_regression_vs_scope_only": no_worse_scope,
        "relation_pair": relation_pair,
        "scope_pair": scope_pair,
    }
    gate["go"] = (
        gate["no_accuracy_regression"]
        and gate["no_severe_underprediction_regression"]
        and severe_improved >= 2
        and gate["helpful"] > gate["harmful"]
        and gate["helpful_tasks"] >= 3
        and no_worse_family
        and no_worse_scope
        and relation_pair["strictly_better"]
        and scope_pair["strictly_better"]
    )
    return gate


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
        and validation.get("coverage", {}).get("passed") is True
        and validation.get("gate", {}).get("go") is True
        and validation.get("row_identity", {}).get("identical_rows") is True
        and re.fullmatch(r"[0-9a-f]{64}", str(validation.get("rows_sha256", "")))
        is not None
    )


def _reserved_role_complete(run_dir: Path, task_ids: Sequence[str]) -> bool:
    return (run_dir / "results.jsonl").is_file() and all(
        any(
            (attempt / "results.json").is_file()
            for attempt in (run_dir / task_id).glob("attempt_*")
        )
        for task_id in task_ids
    )


def evaluate(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    if args.split_manifest.resolve() != SPLIT_MANIFEST.resolve():
        raise ValueError("split manifest path differs from the preregistration")
    _committed_file(args.split_manifest)
    split, split_sha256 = _load_split_manifest(args.split_manifest)
    declared_reserved = Path(str(split["reserved_run"]))
    if not declared_reserved.is_absolute():
        declared_reserved = _REPO_ROOT / declared_reserved
    if args.run_dir.resolve() != declared_reserved.resolve():
        raise ValueError("reserved run differs from the frozen split")
    artifact, spec, artifact_sha256, artifact_commit = _load_artifact(
        args.artifact_dir, split, split_sha256
    )
    validation = None
    validation_sha256 = None
    validation_commit = None
    if args.role == "final_test":
        if args.validation_result is None:
            raise ValueError("final_test requires a passing validation result")
        validation_bytes, validation_commit = _committed_file(args.validation_result)
        validation = json.loads(validation_bytes)
        if not _validation_authorizes_final(validation, artifact_sha256, split_sha256):
            raise ValueError("final_test is not authorized by matching validation GO")
        validation_rows, rows_commit = _committed_file(
            args.validation_result.with_name("rows.jsonl")
        )
        if (
            rows_commit != validation_commit
            or hashlib.sha256(validation_rows).hexdigest() != validation["rows_sha256"]
        ):
            raise ValueError("validation rows differ from committed authorization")
        validation_sha256 = hashlib.sha256(validation_bytes).hexdigest()
    elif args.validation_result is not None:
        raise ValueError("validation must not consume a prior result")

    development_ids, development_clauses, development_commands = load_run_rows(
        args.development_run
    )
    declared_development = Path(str(split["development_run"]))
    if not declared_development.is_absolute():
        declared_development = _REPO_ROOT / declared_development
    development_fit_sha256 = _fit_fingerprint(
        development_ids, development_clauses, development_commands
    )
    public_inputs = _frozen_public_inputs(split, args.public_telemetry)
    if (
        args.development_run.resolve() != declared_development.resolve()
        or development_ids != split["development"]
        or artifact.get("development_fit_sha256") != development_fit_sha256
        or artifact.get("public_inputs") != public_inputs
    ):
        raise ValueError("frozen development or public evidence differs")
    if validation is not None and (
        validation.get("development_fit_sha256") != development_fit_sha256
        or validation.get("development_run") != str(args.development_run.resolve())
        or validation.get("public_inputs") != public_inputs
    ):
        raise ValueError("final_test Current evidence differs from validation")

    role_ids = list(split[args.role])
    if not _reserved_role_complete(args.run_dir, role_ids):
        raise RuntimeError("reserved role is incomplete; no outcome was read")
    records, statuses = _attempt_records(args.run_dir, role_ids)
    valid_records = _telemetry_valid_records(records, statuses)
    valid_ids = [str(record["instance_id"]) for record in valid_records]
    with tempfile.TemporaryDirectory(prefix="family-role-") as directory:
        view = Path(directory) / "results.jsonl"
        _write_result_view(valid_records, view)
        events = _load_exec_events(args.run_dir, valid_ids, results_path=view)
        coverage_commands, coverage_queries = _all_command_queries(valid_ids, events)
        coverage_outputs, _durations, coverage_bounded = run_queries(
            spec,
            [
                coverage_queries[f"{row.task_id}:{row.call_index}"]
                for row in coverage_commands
            ],
        )
        coverage = _coverage(
            coverage_commands,
            coverage_outputs,
            artifact["primary_contrasts"],
            coverage_bounded,
        )
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
        role_task_ids, role_clauses, role_commands = load_run_rows(
            args.run_dir, results_path=view
        )
        if role_task_ids != valid_ids:
            raise AssertionError("role task order differs after telemetry validation")
        role_queries = build_queries(role_task_ids, role_commands, events)
        role_samples = [f"{row.task_id}:{row.call_index}" for row in role_commands]
        extracted, _durations, scored_bounded = run_queries(
            spec, [role_queries[sample] for sample in role_samples]
        )
        if scored_bounded:
            raise BoundExceeded("scored queries exceeded a frozen bound")
        outputs = dict(zip(role_samples, extracted, strict=True))
        family_outputs = _collapsed_outputs(outputs, dimension="family")
        scope_outputs = _collapsed_outputs(outputs, dimension="scope")
        adjusted_clauses, adjusted_commands = _offset_rows(
            role_clauses, role_commands, len(development_ids)
        )
        public = [row for path in args.public_telemetry for row in load_rows(path)]
        excluded_repos = {repo_of(task_id) for task_id in (*development_ids, *role_ids)}
        public = [row for row in public if row.repo not in excluded_repos]
        if not public or {row.task_id for row in public} & set(
            (*development_ids, *role_ids)
        ):
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
    full_rows = _apply_candidate(
        frozen_rows, outputs, _decode_pmfs(artifact["full_pmfs"])
    )
    family_rows = _apply_candidate(
        frozen_rows, family_outputs, _decode_pmfs(artifact["family_only_pmfs"])
    )
    scope_rows = _apply_candidate(
        frozen_rows, scope_outputs, _decode_pmfs(artifact["scope_only_pmfs"])
    )
    full_score = _score_candidate(frozen_baseline, full_rows)
    family_score = _score_candidate(frozen_baseline, family_rows)
    scope_score = _score_candidate(frozen_baseline, scope_rows)
    gate = _fresh_gate(
        frozen_baseline,
        full_score,
        family_score,
        scope_score,
        full_rows,
        family_rows,
        scope_rows,
        artifact["primary_contrasts"],
    )
    rows = [
        {
            **full,
            "role": args.role,
            "family_only_candidate": family["candidate"],
            "family_only_probability_by_bucket": family[
                "candidate_probability_by_bucket"
            ],
            "scope_only_candidate": scope["candidate"],
            "scope_only_probability_by_bucket": scope[
                "candidate_probability_by_bucket"
            ],
        }
        for full, family, scope in zip(full_rows, family_rows, scope_rows, strict=True)
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
        "labels_scored": True,
        "primary_contrasts": artifact["primary_contrasts"],
        "baseline": {
            "latency": frozen_baseline["latency"]["current_dynamic"],
            "resources": {
                target: frozen_baseline["resources"][target]["current_dynamic"]
                for target in CANONICAL_RESOURCE_BUCKET_EDGES
            },
        },
        "full": full_score,
        "family_only": family_score,
        "scope_only": scope_score,
        "gate": gate,
        "row_identity": {
            "identical_rows": [row["sample_id"] for row in full_rows]
            == [row["sample_id"] for row in family_rows]
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--run-dir", type=Path, required=True)
    freeze_parser.add_argument("--out-dir", type=Path, required=True)
    freeze_parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST)
    freeze_parser.add_argument(
        "--public-telemetry", type=Path, action="append", required=True
    )
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--role", choices=("validation", "final_test"), required=True
    )
    evaluate_parser.add_argument("--run-dir", type=Path, required=True)
    evaluate_parser.add_argument("--development-run", type=Path, required=True)
    evaluate_parser.add_argument(
        "--public-telemetry", type=Path, action="append", required=True
    )
    evaluate_parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST)
    evaluate_parser.add_argument("--artifact-dir", type=Path, required=True)
    evaluate_parser.add_argument("--validation-result", type=Path)
    evaluate_parser.add_argument("--out-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    freeze(arguments) if arguments.command == "freeze" else evaluate(arguments)
