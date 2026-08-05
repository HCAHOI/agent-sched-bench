#!/usr/bin/env python3
"""Freeze and replay one agent-generated blocker/remediation parser."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import itertools
import json
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    CANONICAL_RESOURCE_BUCKET_EDGES,
    RESOURCE_BUCKET_LABELS,
    CommandRow,
    _argmax_probabilities,
    evaluate_prequential_commands,
)
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    Row,
    load_rows,
    load_run_rows,
)
from scripts.evaluation.evaluate_command_prequential import (  # noqa: E402
    _load_exec_events,
)
from scripts.evaluation.evaluate_offline_agent_extractor import (  # noqa: E402
    MODEL,
    TARGETS,
    _SAFE_BUILTINS,
    _apply_candidate,
    _catalog_rows,
    _codex_call,
    _fit_pmfs,
    _regex_literals,
    _score_candidate,
    _specific_command_tokens,
    build_queries,
)
from tool_resource_eval.labels import repo_of  # noqa: E402

SCHEMA = "offline-agent-relational-state-v1"
MINIMUM_SUPPORT_TASKS = 5
MAX_PROMPT_BYTES = 200_000
STATE_ORDER = (
    "newly_surfaced",
    "partial_remediation",
    "closure_verified",
    "closure_candidate",
    "blocked",
)
ENTRY_POINTS = {
    "scope": 2,
    "blocker_spans": 1,
    "remediation_spans": 1,
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
GENERATION_PROMPT = """Generate one small auditable Python parser from the label-free
development examples below. Define exactly three synchronous pure functions:

scope(current_command, parsed_clauses) -> reusable_rule_id_or_None
blocker_spans(result_excerpt) -> list of [start, end] character spans
remediation_spans(command) -> list of [start, end] character spans

blocker_spans must enumerate every specific named entity that prevents the
scoped command from beginning its requested work. remediation_spans must
enumerate every explicitly requested entity in a command that could address such
a blocker. Return spans into the exact input string, not copied text. Return no
span for generic error words, paths, test names, or ambiguous text. The host
will call each span function on one string at a time, require exact normalized
identifier equality and a successful remediation exit code, scan every causal
event, build the graph, derive state, and fit resource PMFs.

The source may use re, which is provided, and ordinary pure Python. It may not
import modules, read files or network, execute commands, use task/sample/repo
identity, contain file/test/package-specific literals, or return a resource
bucket. Do not assign or construct matching string literals indirectly. The
three functions must return None/strings/spans exactly as specified. Generate
one source or abstain. Do not call tools. Return only the required JSON.

LABEL-FREE DEVELOPMENT EVIDENCE:
"""

_FORBIDDEN_NAMES = {
    "__builtins__",
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


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _normalize_identifier(text: str) -> str:
    trimmed = text.casefold().strip(" \t\r\n'\"`[](){}<>:;,.")
    return re.sub(r"[^0-9a-z]+", "-", trimmed).strip("-")


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _validate_spans(value: Any, text: str, owner: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{owner} did not return a list")
    rows: list[dict[str, Any]] = []
    previous_end = -1
    identifiers: set[str] = set()
    for span in value:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in span)
        ):
            raise ValueError(f"{owner} returned a malformed span")
        start, end = span
        if start < 0 or end <= start or end > len(text) or start < previous_end:
            raise ValueError(f"{owner} returned an overlapping or out-of-range span")
        raw = text[start:end]
        identifier = _normalize_identifier(raw)
        if (
            not 2 <= len(identifier) <= 80
            or "/" in raw
            or "\\" in raw
            or not any(character.isalnum() for character in identifier)
        ):
            raise ValueError(f"{owner} returned an empty or path-like identifier")
        if identifier in identifiers:
            raise ValueError(f"{owner} returned a duplicate identifier")
        rows.append({"span": [start, end], "raw": raw, "identifier": identifier})
        identifiers.add(identifier)
        previous_end = end
    return rows


def validate_source(source: str, forbidden_ids: Sequence[str] = ()) -> None:
    """Validate the frozen three-function parser before isolated execution."""

    if not source.strip() or len(source.encode()) > 30_000:
        raise ValueError("generated source is empty or too large")
    if any(identifier in source for identifier in forbidden_ids):
        raise ValueError("generated source contains an opaque training ID")
    tree = ast.parse(source)
    top_level = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if {node.name for node in top_level} != set(ENTRY_POINTS) or len(top_level) != 3:
        raise ValueError("generated source must define exactly the three entry points")
    if any(not isinstance(node, (ast.FunctionDef, ast.Expr)) for node in tree.body):
        raise ValueError("generated source has executable top-level statements")
    if any(
        isinstance(node, ast.Expr)
        and (
            node is not tree.body[0]
            or not isinstance(node.value, ast.Constant)
            or not isinstance(node.value.value, str)
        )
        for node in tree.body
    ):
        raise ValueError("generated source has a non-docstring top-level expression")
    for function in top_level:
        if (
            len(function.args.args) != ENTRY_POINTS[function.name]
            or function.args.vararg is not None
            or function.args.kwarg is not None
            or function.args.kwonlyargs
            or function.args.defaults
            or function.args.kw_defaults
            or function.decorator_list
            or function.returns is not None
            or any(argument.annotation is not None for argument in function.args.args)
        ):
            raise ValueError(f"{function.name} has the wrong arguments")
    forbidden_nodes = (
        ast.AsyncFunctionDef,
        ast.Await,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.Nonlocal,
        ast.Raise,
        ast.Try,
        ast.While,
        ast.With,
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError(f"generated source contains forbidden {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and node not in top_level:
            raise ValueError("generated source defines a nested or helper function")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"generated source uses forbidden name {node.id}")
        if isinstance(node, ast.Name) and node.id in ENTRY_POINTS:
            raise ValueError("generated entry points may not call or mutate one another")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("generated source accesses a private attribute")
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            raise ValueError("generated source mutates an object attribute")
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            raise ValueError("generated source mutates an indexed object")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "/" in node.value or "\\" in node.value:
                raise ValueError("generated source contains a path-like literal")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and any(
            isinstance(descendant, ast.Constant)
            and isinstance(descendant.value, str)
            for descendant in ast.walk(node.value)
        ):
            raise ValueError("generated source assigns a matching literal indirectly")
        if isinstance(node, (ast.BinOp, ast.JoinedStr)) and any(
            isinstance(descendant, ast.Constant)
            and isinstance(descendant.value, str)
            for descendant in ast.walk(node)
        ):
            raise ValueError("generated source constructs a string dynamically")


def _scope_output_literal_ids(tree: ast.Module) -> set[int]:
    scope = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "scope")
    return {
        id(node.value)
        for node in ast.walk(scope)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _matching_literals(source: str) -> set[str]:
    tree = ast.parse(source)
    output_ids = _scope_output_literal_ids(tree)
    docstring_ids: set[int] = set()
    for owner in (tree, *(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))):
        if (
            owner.body
            and isinstance(owner.body[0], ast.Expr)
            and isinstance(owner.body[0].value, ast.Constant)
            and isinstance(owner.body[0].value.value, str)
        ):
            docstring_ids.add(id(owner.body[0].value))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.strip()
        and id(node) not in output_ids
        and id(node) not in docstring_ids
    }


def _relational_regex_literals(source: str) -> dict[str, set[str]]:
    methods = _regex_literals(source)
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re"
            and node.func.attr in {"findall", "finditer"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            methods[node.args[0].value].add(node.func.attr)
    return methods


def _regex_matches(pattern: re.Pattern[str], methods: set[str], text: str) -> bool:
    return any(
        (
            bool(pattern.findall(text))
            if method == "findall"
            else any(pattern.finditer(text))
            if method == "finditer"
            else getattr(pattern, method)(text) is not None
        )
        for method in methods
    )


def validate_source_literals(
    source: str,
    training_queries: Mapping[str, Mapping[str, Any]],
    task_by_sample: Mapping[str, str],
) -> None:
    """Reject literals without repeated support or depending on specific args."""

    material_by_task: dict[str, str] = defaultdict(str)
    texts_by_task: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    positional_tokens: set[str] = set()
    generic_tokens: set[str] = set()
    for sample_id, query in training_queries.items():
        task_id = task_by_sample[sample_id]
        material_by_task[task_id] += "\n" + json.dumps(query, sort_keys=True).lower()
        texts_by_task[task_id].append((str(query["current_command"]), True))
        for event in query["prior_events"]:
            texts_by_task[task_id].extend(
                ((str(event["command"]), True), (str(event["result_excerpt"]), False))
            )
        for clause in query["parsed_clauses"].get("clauses", ()):
            argv = [str(value).lower() for value in clause.get("argv", ())]
            if not argv:
                continue
            generic_tokens.add(argv[0])
            if "-m" in argv[:-1]:
                module_index = argv.index("-m")
                generic_tokens.add(argv[module_index + 1])
            elif len(argv) > 1 and not argv[1].startswith("-"):
                generic_tokens.add(argv[1])
            generic_tokens.update(value for value in argv if value.startswith("-"))
            positional_tokens.update(value for value in argv[1:] if not value.startswith("-"))
    regex_literals = _relational_regex_literals(source)
    for literal in _matching_literals(source):
        lowered = literal.lower()
        if literal in regex_literals:
            try:
                pattern = re.compile(literal)
            except re.error as error:
                raise ValueError(f"generated regex is invalid: {literal!r}") from error
            matched = [
                (text, is_command)
                for texts in texts_by_task.values()
                for text, is_command in texts
                if _regex_matches(pattern, regex_literals[literal], text)
            ]
            support = sum(
                any(_regex_matches(pattern, regex_literals[literal], text) for text, _ in texts)
                for texts in texts_by_task.values()
            )
            dependent_specific = sorted(
                {
                    token
                    for text, is_command in matched
                    if is_command
                    for token in _specific_command_tokens(text)
                    if token in text
                    and not _regex_matches(
                        pattern, regex_literals[literal], text.replace(token, "__ARG__")
                    )
                }
            )
            if dependent_specific:
                raise ValueError(
                    "generated regex depends on package/test/file arguments: "
                    f"{dependent_specific}"
                )
        else:
            support = sum(lowered in material for material in material_by_task.values())
        if support < MINIMUM_SUPPORT_TASKS:
            raise ValueError(f"generated matching literal lacks five-task support: {literal!r}")
        if (
            re.fullmatch(r"[A-Za-z0-9_.+-]+", literal)
            and lowered in positional_tokens
            and lowered not in generic_tokens
        ):
            raise ValueError(f"generated source contains a specific argument: {literal!r}")


def _call_parser(
    namespace: Mapping[str, Any],
    function: str,
    *args: Any,
) -> Any:
    re.purge()
    try:
        return namespace[function](*args)
    finally:
        re.purge()


def _derive_state(query: Mapping[str, Any], namespace: Mapping[str, Any]) -> dict[str, Any] | None:
    rule_id = _call_parser(
        namespace, "scope", query["current_command"], query["parsed_clauses"]
    )
    if rule_id is None:
        return None
    if (
        not isinstance(rule_id, str)
        or re.fullmatch(r"[a-z0-9_-]{1,80}", rule_id) is None
    ):
        raise ValueError("scope returned an invalid rule id")
    current = _collapse_whitespace(str(query["current_command"]))
    blockers: list[dict[str, Any]] = []
    remediations: list[dict[str, Any]] = []
    verifier_indices: list[int] = []
    known_ids: set[str] = set()
    resolved_ids: set[str] = set()
    newly_surfaced_ids: set[str] = set()
    last_remediation = -1
    successful_verifiers: list[int] = []
    for event in query["prior_events"]:
        event_index = int(event["event_index"])
        is_verifier = _collapse_whitespace(str(event["command"])) == current
        if is_verifier:
            verifier_indices.append(event_index)
            if event["exit_code"] == 0:
                successful_verifiers.append(event_index)
            elif event["exit_code"] is not None:
                found = _validate_spans(
                    _call_parser(
                        namespace, "blocker_spans", event["result_excerpt"]
                    ),
                    str(event["result_excerpt"]),
                    "blocker_spans",
                )
                prior_known = set(known_ids)
                for item in found:
                    blocker = {
                        **item,
                        "event_index": event_index,
                        "resolved_by_event_index": None,
                    }
                    blockers.append(blocker)
                    known_ids.add(item["identifier"])
                    if resolved_ids and item["identifier"] not in prior_known:
                        newly_surfaced_ids.add(item["identifier"])
        if event["exit_code"] != 0:
            continue
        tokens = _validate_spans(
            _call_parser(namespace, "remediation_spans", event["command"]),
            str(event["command"]),
            "remediation_spans",
        )
        for token in tokens:
            addressed = [
                blocker
                for blocker in blockers
                if blocker["identifier"] == token["identifier"]
                and blocker["resolved_by_event_index"] is None
            ]
            if not addressed:
                continue
            for blocker in addressed:
                blocker["resolved_by_event_index"] = event_index
            resolved_ids.add(token["identifier"])
            last_remediation = event_index
            remediations.append(
                {
                    **token,
                    "event_index": event_index,
                    "addresses": [
                        [blocker["event_index"], blocker["span"]] for blocker in addressed
                    ],
                }
            )
    if not blockers:
        return None
    active = [item for item in blockers if item["resolved_by_event_index"] is None]
    resolved = [item for item in blockers if item["resolved_by_event_index"] is not None]
    if newly_surfaced_ids & {item["identifier"] for item in active}:
        state = "newly_surfaced"
    elif active and resolved:
        state = "partial_remediation"
    elif not active and resolved and any(index > last_remediation for index in successful_verifiers):
        state = "closure_verified"
    elif not active and resolved:
        state = "closure_candidate"
    elif active:
        state = "blocked"
    else:
        return None
    evidence = sorted(
        {
            *(item["event_index"] for item in blockers),
            *(item["event_index"] for item in remediations),
            *(index for index in successful_verifiers if index > last_remediation),
        }
    )
    return {
        "rule_id": rule_id,
        "state": state,
        "evidence_event_indices": evidence,
        "blockers": blockers,
        "remediations": remediations,
        "verifier_event_indices": verifier_indices,
    }


def _worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = str(payload["source"])
    validate_source(source)
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, "re": re}
    exec(compile(source, "<relational_parser>", "exec"), namespace)
    outputs = []
    durations_ns = []
    for query in payload["queries"]:
        started = time.perf_counter_ns()
        outputs.append(_derive_state(query, namespace))
        durations_ns.append(time.perf_counter_ns() - started)
    return {"outputs": outputs, "durations_ns": durations_ns}


def run_source(
    source: str,
    queries: Sequence[Mapping[str, Any]],
    *,
    timeout_seconds: float = 30.0,
) -> tuple[list[dict[str, Any] | None], list[int]]:
    completed = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
        input=json.dumps({"source": source, "queries": list(queries)}),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    if completed.returncode:
        raise RuntimeError(f"relational parser failed: {completed.stderr[-2_000:]}")
    result = json.loads(completed.stdout)
    return result["outputs"], result["durations_ns"]


def _label_free_selected_evidence(
    task_ids: Sequence[str],
    commands: Sequence[CommandRow],
    queries: Mapping[str, Mapping[str, Any]],
    prior_dir: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    catalog = json.loads((prior_dir / "training-catalog.json").read_text())
    artifact = json.loads((prior_dir / "agent-artifact.json").read_text())
    catalog_rows = _catalog_rows(catalog)
    prior_training_tasks = set(task_ids[:80])
    training_commands = [row for row in commands if row.task_id in prior_training_tasks]
    if len(training_commands) != len(catalog_rows):
        raise ValueError("prior catalog differs from development command order")
    private: dict[str, str] = {}
    for item, command in zip(catalog_rows, training_commands, strict=True):
        sample_id = f"{command.task_id}:{command.call_index}"
        if item["command"] != command.command:
            raise ValueError("prior catalog command differs from development trace")
        private[str(item["sample_id"])] = sample_id
    selection = artifact.get("selection", {})
    selected = selection.get("selected_sample_ids")
    if not isinstance(selected, list) or len(selected) != 12 or len(set(selected)) != 12:
        raise ValueError("prior selection is not the frozen twelve samples")
    by_id = {str(item["sample_id"]): item for item in catalog_rows}
    if any(sample_id not in private for sample_id in selected):
        raise ValueError("prior selection contains an unknown sample")
    rows = []
    prefixes = {}
    for sample_id in selected:
        item = by_id[sample_id]
        real_id = private[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "task_id": item["task_id"],
                "event_index": item["event_index"],
                "command": item["command"],
                "parsed_clauses": queries[real_id]["parsed_clauses"],
            }
        )
        prefixes[sample_id] = queries[real_id]["prior_events"]
    evidence = {
        "selected_rows": rows,
        "causal_prefixes": prefixes,
        "labels_and_current_predictions": "omitted",
    }
    return evidence, private, artifact


def _select_primary_contrast(
    commands: Sequence[CommandRow],
    outputs: Mapping[str, Mapping[str, Any] | None],
    pmfs: Mapping[tuple[tuple[str, str], str], tuple[float, ...]],
) -> tuple[dict[str, Any] | None, dict[str, list[str]]]:
    tasks_by_state: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in commands:
        value = outputs[f"{row.task_id}:{row.call_index}"]
        if value is not None:
            tasks_by_state[(str(value["rule_id"]), str(value["state"]))].add(row.task_id)
    candidates = []
    by_rule: dict[str, list[str]] = defaultdict(list)
    for rule_id, state in tasks_by_state:
        if len(tasks_by_state[(rule_id, state)]) >= MINIMUM_SUPPORT_TASKS:
            by_rule[rule_id].append(state)
    for rule_id, states in by_rule.items():
        for left, right in itertools.combinations(sorted(set(states)), 2):
            differing_targets = [
                target
                for target in TARGETS
                if ((rule_id, left), target) in pmfs
                and ((rule_id, right), target) in pmfs
                and _argmax_probabilities(pmfs[((rule_id, left), target)])
                != _argmax_probabilities(pmfs[((rule_id, right), target)])
            ]
            if not differing_targets:
                continue
            left_support = len(tasks_by_state[(rule_id, left)])
            right_support = len(tasks_by_state[(rule_id, right)])
            candidates.append(
                (
                    -min(left_support, right_support),
                    -(left_support + right_support),
                    rule_id,
                    left,
                    right,
                    differing_targets[0],
                )
            )
    state_tasks = {
        f"{rule_id}::{state}": sorted(tasks)
        for (rule_id, state), tasks in sorted(tasks_by_state.items())
    }
    if not candidates:
        return None, state_tasks
    _, _, rule_id, left, right, target = min(candidates)
    return {
        "rule_id": rule_id,
        "states": [left, right],
        "target": target,
        "task_support": [
            len(tasks_by_state[(rule_id, left)]),
            len(tasks_by_state[(rule_id, right)]),
        ],
    }, state_tasks


def _load_split_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    expected = {"development", "validation", "final_test"}
    if not isinstance(manifest, dict) or any(
        not isinstance(manifest.get(role), list) for role in expected
    ):
        raise ValueError("split manifest lacks the three task-ID partitions")
    partitions = {role: list(manifest[role]) for role in expected}
    if (
        len(partitions["development"]) != 100
        or len(partitions["validation"]) != 50
        or len(partitions["final_test"]) != 50
        or any(
            not isinstance(task_id, str)
            for task_ids in partitions.values()
            for task_id in task_ids
        )
        or any(len(set(task_ids)) != len(task_ids) for task_ids in partitions.values())
        or any(
            set(partitions[left]) & set(partitions[right])
            for left, right in itertools.combinations(sorted(expected), 2)
        )
    ):
        raise ValueError("split manifest counts, IDs, or disjointness are invalid")
    return manifest, hashlib.sha256(raw).hexdigest()


def _collapsed_outputs(
    outputs: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, dict[str, Any] | None]:
    return {
        sample_id: (
            None
            if value is None
            else {**value, "state": "__collapsed__"}
        )
        for sample_id, value in outputs.items()
    }


def _decode_pmfs(
    artifact: Mapping[str, Any],
    field: str,
) -> dict[tuple[tuple[str, str], str], tuple[float, ...]]:
    encoded = artifact.get(field)
    if not isinstance(encoded, dict):
        raise ValueError(f"frozen artifact lacks {field}")
    decoded = {}
    for key, values in encoded.items():
        parts = key.split("::")
        if field == "pmfs" and len(parts) == 3:
            rule_id, state, target = parts
        elif field == "collapsed_pmfs" and len(parts) == 2:
            rule_id, target = parts
            state = "__collapsed__"
        else:
            raise ValueError(f"frozen {field} key is invalid: {key}")
        if target not in TARGETS or not isinstance(values, list):
            raise ValueError(f"frozen {field} value is invalid: {key}")
        pmf = tuple(float(value) for value in values)
        expected = 5 if target == "latency" else 3
        if (
            len(pmf) != expected
            or any(value < 0.0 or value > 1.0 for value in pmf)
            or abs(sum(pmf) - 1.0) > 1e-9
        ):
            raise ValueError(f"frozen {field} PMF is invalid: {key}")
        decoded[((rule_id, state), target)] = pmf
    return decoded


def _load_frozen_artifact(
    artifact_dir: Path,
    split_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    artifact_path = artifact_dir / "artifact.json"
    artifact_bytes = artifact_path.read_bytes()
    artifact = json.loads(artifact_bytes)
    source = str(artifact.get("generation", {}).get("source", ""))
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("status") != "development_structural_go"
        or artifact.get("split_manifest_sha256") != split_sha256
        or artifact.get("source_sha256") != source_sha256
        or (artifact_dir / "generated_parser.py").read_bytes()
        != source.rstrip().encode() + b"\n"
        or not isinstance(artifact.get("primary_contrast"), dict)
    ):
        raise ValueError("frozen relational artifact is incomplete or differs")
    validate_source(source)
    _decode_pmfs(artifact, "pmfs")
    _decode_pmfs(artifact, "collapsed_pmfs")
    return artifact, source, hashlib.sha256(artifact_bytes).hexdigest()


def _attempt_records(
    run_dir: Path,
    task_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not (run_dir / "results.jsonl").is_file():
        raise RuntimeError("reserved collection is not complete; no result was consumed")
    records = []
    statuses = []
    for task_id in task_ids:
        task_dir = (run_dir / task_id).resolve()
        if task_dir.parent != run_dir.resolve():
            raise ValueError(f"task path escapes reserved run: {task_id}")
        attempts = sorted(
            task_dir.glob("attempt_*"),
            key=lambda path: int(path.name.removeprefix("attempt_")),
        )
        completed = []
        for attempt in attempts:
            path = attempt / "results.json"
            if path.is_file():
                result = json.loads(path.read_text())
                if result.get("instance_id") != task_id:
                    raise ValueError(f"{path}: task identity differs from manifest")
                completed.append((attempt, result))
        successful = [item for item in completed if item[1].get("success") is True]
        if successful:
            attempt, _result = successful[-1]
            records.append(
                {
                    "instance_id": task_id,
                    "attempt_dir": str(attempt.resolve()),
                    "success": True,
                }
            )
            statuses.append({"task_id": task_id, "status": "accepted"})
        else:
            statuses.append({"task_id": task_id, "status": "failed"})
    return records, statuses


def _write_result_view(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _all_command_queries(
    task_ids: Sequence[str],
    events_by_task: Mapping[str, Sequence[Any]],
) -> tuple[list[CommandRow], dict[str, dict[str, Any]]]:
    commands = [
        CommandRow(
            task_id=task_id,
            repo=repo_of(task_id),
            manifest_index=manifest_index,
            call_index=event_index,
            call_id=event.call_id,
            command=event.command,
            duration_ms=0.0,
            clauses=(),
        )
        for manifest_index, task_id in enumerate(task_ids)
        for event_index, event in enumerate(events_by_task[task_id])
    ]
    return commands, build_queries(task_ids, commands, events_by_task)


def _coverage(
    task_ids: Sequence[str],
    commands: Sequence[CommandRow],
    extracted: Sequence[Mapping[str, Any] | None],
    primary: Mapping[str, Any],
) -> dict[str, Any]:
    non_null = [
        (row, value)
        for row, value in zip(commands, extracted, strict=True)
        if value is not None
    ]
    carrier_tasks = {row.task_id for row, _value in non_null}
    primary_states = tuple(primary["states"])
    state_counts = {}
    for state in primary_states:
        selected = [
            row
            for row, value in non_null
            if value["rule_id"] == primary["rule_id"] and value["state"] == state
        ]
        state_counts[state] = {
            "commands": len(selected),
            "tasks": len({row.task_id for row in selected}),
        }
    passed = (
        len(non_null) >= 20
        and len(carrier_tasks) >= 5
        and all(
            counts["commands"] >= 5 and counts["tasks"] >= 3
            for counts in state_counts.values()
        )
    )
    return {
        "passed": passed,
        "all_exec_commands": len(commands),
        "carrier_commands": len(non_null),
        "carrier_tasks": len(carrier_tasks),
        "primary_state_counts": state_counts,
        "minimum_carrier_commands": 20,
        "minimum_carrier_tasks": 5,
        "minimum_primary_state_commands": 5,
        "minimum_primary_state_tasks": 3,
    }


def _offset_rows(
    clauses: Sequence[Row],
    commands: Sequence[CommandRow],
    offset: int,
) -> tuple[list[Row], list[CommandRow]]:
    adjusted_clauses = [
        replace(row, manifest_index=row.manifest_index + offset) for row in clauses
    ]
    adjusted_commands = [
        replace(
            row,
            manifest_index=row.manifest_index + offset,
            clauses=tuple(
                replace(clause, manifest_index=clause.manifest_index + offset)
                for clause in row.clauses
            ),
        )
        for row in commands
    ]
    return adjusted_clauses, adjusted_commands


def _frozen_baseline(
    baseline: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen = copy.deepcopy(baseline)
    frozen["latency"]["current_dynamic"] = frozen["latency"]["frozen_at_80"]
    for target in CANONICAL_RESOURCE_BUCKET_EDGES:
        frozen["resources"][target]["current_dynamic"] = frozen["resources"][target][
            "frozen_at_80"
        ]
    frozen_rows = []
    for row in rows:
        item = dict(row)
        item["current_dynamic"] = copy.deepcopy(row["frozen_at_80"])
        frozen_rows.append(item)
    return frozen, frozen_rows


def _accuracy(score: Mapping[str, Any], target: str) -> float:
    key = "exact_class_accuracy" if target == "latency" else "accuracy"
    return float(score["metrics"][target][key])


def _fresh_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    collapsed: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    collapsed_rows: Sequence[Mapping[str, Any]],
    primary: Mapping[str, Any],
) -> dict[str, Any]:
    no_accuracy_regression = bool(candidate["gate"]["no_accuracy_regression"])
    no_severe_regression = bool(
        candidate["gate"]["no_severe_underprediction_regression"]
    )
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
        candidate["metrics"][target]["severe_underprediction_rate"]
        < reference_severe[target]
        for target in TARGETS
    )
    no_worse_than_collapsed = all(
        _accuracy(candidate, target) >= _accuracy(collapsed, target)
        for target in TARGETS
    )
    states = set(primary["states"])
    target = str(primary["target"])
    relational_correct = 0
    collapsed_correct = 0
    primary_commands = 0
    for row, collapsed_row in zip(candidate_rows, collapsed_rows, strict=True):
        signature = row["generated_signature"]
        if (
            signature is None
            or signature["rule_id"] != primary["rule_id"]
            or signature["state"] not in states
        ):
            continue
        truth = row["labels"][target]
        if truth is None:
            continue
        primary_commands += 1
        relational = row["candidate"][target]
        collapsed_value = collapsed_row["candidate"][target]
        if target != "latency":
            relational = RESOURCE_BUCKET_LABELS.index(relational)
            collapsed_value = RESOURCE_BUCKET_LABELS.index(collapsed_value)
        relational_correct += relational == truth
        collapsed_correct += collapsed_value == truth
    pair_better = relational_correct > collapsed_correct
    go = (
        no_accuracy_regression
        and no_severe_regression
        and severe_improved >= 2
        and candidate["gate"]["helpful"] > candidate["gate"]["harmful"]
        and candidate["gate"]["helpful_tasks"] >= 3
        and no_worse_than_collapsed
        and pair_better
    )
    return {
        "go": go,
        "no_accuracy_regression": no_accuracy_regression,
        "no_severe_underprediction_regression": no_severe_regression,
        "severe_underprediction_improved_targets": severe_improved,
        "requires_severe_improvement_targets": 2,
        "helpful": candidate["gate"]["helpful"],
        "harmful": candidate["gate"]["harmful"],
        "helpful_tasks": candidate["gate"]["helpful_tasks"],
        "no_accuracy_regression_vs_collapsed": no_worse_than_collapsed,
        "primary_pair": {
            "target": target,
            "commands": primary_commands,
            "relational_correct": relational_correct,
            "collapsed_correct": collapsed_correct,
            "strictly_better": pair_better,
        },
    }


def _telemetry_valid_records(
    records: Sequence[Mapping[str, Any]],
    statuses: list[dict[str, str]],
) -> list[Mapping[str, Any]]:
    valid = []
    by_task = {row["task_id"]: row for row in statuses}
    required = {
        "collection_validity": "valid",
        "workload_execution": "completed",
        "telemetry_quality": "ok",
        "cleanup": "ok",
    }
    for record in records:
        task_id = str(record["instance_id"])
        artifact = json.loads(
            (Path(str(record["attempt_dir"])) / "resource_observations.json").read_text()
        )
        if any(artifact.get(key) != value for key, value in required.items()):
            by_task[task_id]["status"] = "telemetry_invalid"
        else:
            valid.append(record)
    return valid


def _evaluate(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    split, split_sha256 = _load_split_manifest(args.split_manifest)
    role = args.role
    role_ids = list(split[role])
    declared_run = Path(str(split.get("reserved_run", "")))
    if not declared_run.is_absolute():
        declared_run = _REPO_ROOT / declared_run
    if declared_run.resolve() != args.run_dir.resolve():
        raise ValueError("reserved run differs from the frozen split manifest")
    artifact, source, artifact_sha256 = _load_frozen_artifact(
        args.artifact_dir, split_sha256
    )
    if artifact.get("development_task_ids_sha256") != hashlib.sha256(
        _json_bytes(split["development"])
    ).hexdigest():
        raise ValueError("frozen artifact development tasks differ from split manifest")
    validation: dict[str, Any] | None = None
    validation_sha256 = None
    if role == "final_test":
        if args.validation_result is None:
            raise ValueError("final_test requires the passing validation result")
        validation_bytes = args.validation_result.read_bytes()
        validation = json.loads(validation_bytes)
        if (
            validation.get("schema") != "relational-fresh-evaluation-v1"
            or validation.get("role") != "validation"
            or validation.get("status") != "validation_go"
            or validation.get("artifact_sha256") != artifact_sha256
            or validation.get("split_manifest_sha256") != split_sha256
        ):
            raise ValueError("final_test is not authorized by matching validation GO")
        validation_sha256 = hashlib.sha256(validation_bytes).hexdigest()
    elif args.validation_result is not None:
        raise ValueError("validation must not consume a prior validation result")

    development_ids, development_clauses, development_commands = load_run_rows(
        args.development_run
    )
    declared_development = Path(str(split.get("development_run", "")))
    if not declared_development.is_absolute():
        declared_development = _REPO_ROOT / declared_development
    if (
        development_ids != split["development"]
        or args.development_run.resolve() != declared_development.resolve()
    ):
        raise ValueError("development run differs from split manifest")
    development_fit_sha256 = hashlib.sha256(
        _json_bytes([asdict(row) for row in development_clauses])
    ).hexdigest()
    public_inputs = [
        {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in args.public_telemetry
    ]
    if validation is not None and (
        validation.get("public_inputs") != public_inputs
        or validation.get("development_fit_sha256") != development_fit_sha256
        or validation.get("development_run") != str(args.development_run.resolve())
    ):
        raise ValueError("final_test Current evidence differs from validation")

    records, statuses = _attempt_records(args.run_dir, role_ids)
    valid_records = _telemetry_valid_records(records, statuses)
    accepted_ids = [str(record["instance_id"]) for record in valid_records]
    with tempfile.TemporaryDirectory(prefix="relational-role-") as directory:
        view = Path(directory) / "results.jsonl"
        _write_result_view(valid_records, view)
        events = _load_exec_events(
            args.run_dir,
            accepted_ids,
            results_path=view,
        )
        coverage_commands, coverage_queries = _all_command_queries(accepted_ids, events)
        ordered_coverage = [
            coverage_queries[f"{row.task_id}:{row.call_index}"]
            for row in coverage_commands
        ]
        extracted_coverage, _durations = run_source(source, ordered_coverage)
        coverage = _coverage(
            accepted_ids,
            coverage_commands,
            extracted_coverage,
            artifact["primary_contrast"],
        )
        if not coverage["passed"]:
            args.out_dir.mkdir(parents=True)
            result = {
                "schema": "relational-fresh-evaluation-v1",
                "status": f"{role}_coverage_no_go",
                "claim_bearing": False,
                "role": role,
                "split_manifest_sha256": split_sha256,
                "artifact_sha256": artifact_sha256,
                "validation_result_sha256": validation_sha256,
                "task_statuses": statuses,
                "coverage": coverage,
                "labels_scored": False,
            }
            (args.out_dir / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            return

        if not accepted_ids:
            raise ValueError("coverage passed but no evidence-valid task remains")
        role_task_ids, role_clauses, role_commands = load_run_rows(
            args.run_dir,
            results_path=view,
        )
        if role_task_ids != accepted_ids:
            raise AssertionError("role task order differs after telemetry validation")
        role_events = {task_id: events[task_id] for task_id in accepted_ids}
        role_queries = build_queries(role_task_ids, role_commands, role_events)
        role_samples = [f"{row.task_id}:{row.call_index}" for row in role_commands]
        extracted, _runtime = run_source(
            source, [role_queries[sample_id] for sample_id in role_samples]
        )
        outputs = dict(zip(role_samples, extracted, strict=True))

        adjusted_clauses, adjusted_commands = _offset_rows(
            role_clauses, role_commands, len(development_ids)
        )
        public = [row for path in args.public_telemetry for row in load_rows(path)]
        excluded_repos = {repo_of(task_id) for task_id in (*development_ids, *role_ids)}
        public = [row for row in public if row.repo not in excluded_repos]
        if not public or {row.task_id for row in public} & set((*development_ids, *role_ids)):
            raise ValueError("public evidence is empty or overlaps SQLGlot tasks")
        combined_ids = [*development_ids, *role_task_ids]
        combined_clauses = [*development_clauses, *adjusted_clauses]
        combined_commands = [*development_commands, *adjusted_commands]
        baseline, baseline_rows = evaluate_prequential_commands(
            public,
            combined_ids,
            combined_clauses,
            combined_commands,
            {
                "development_run": str(args.development_run.resolve()),
                "reserved_run": str(args.run_dir.resolve()),
                "role": role,
            },
            warmup_task_count=100,
        )
    frozen_baseline, frozen_rows = _frozen_baseline(baseline, baseline_rows)
    pmfs = _decode_pmfs(artifact, "pmfs")
    collapsed_pmfs = _decode_pmfs(artifact, "collapsed_pmfs")
    candidate_rows = _apply_candidate(frozen_rows, outputs, pmfs)
    collapsed_rows = _apply_candidate(
        frozen_rows,
        _collapsed_outputs(outputs),
        collapsed_pmfs,
    )
    candidate = _score_candidate(frozen_baseline, candidate_rows)
    collapsed = _score_candidate(frozen_baseline, collapsed_rows)
    gate = _fresh_gate(
        frozen_baseline,
        candidate,
        collapsed,
        candidate_rows,
        collapsed_rows,
        artifact["primary_contrast"],
    )
    rows = []
    for row, collapsed_row in zip(candidate_rows, collapsed_rows, strict=True):
        rows.append(
            {
                **row,
                "role": role,
                "collapsed_candidate": collapsed_row["candidate"],
                "collapsed_probability_by_bucket": collapsed_row[
                    "candidate_probability_by_bucket"
                ],
            }
        )
    args.out_dir.mkdir(parents=True)
    result = {
        "schema": "relational-fresh-evaluation-v1",
        "status": f"{role}_{'go' if gate['go'] else 'no_go'}",
        "claim_bearing": role == "final_test" and gate["go"],
        "role": role,
        "split_manifest_sha256": split_sha256,
        "artifact_sha256": artifact_sha256,
        "validation_result_sha256": validation_sha256,
        "development_run": str(args.development_run.resolve()),
        "development_fit_sha256": development_fit_sha256,
        "task_statuses": statuses,
        "coverage": coverage,
        "labels_scored": True,
        "primary_contrast": artifact["primary_contrast"],
        "public_inputs": public_inputs,
        "baseline": {
            "latency": frozen_baseline["latency"]["current_dynamic"],
            "resources": {
                target: frozen_baseline["resources"][target]["current_dynamic"]
                for target in CANONICAL_RESOURCE_BUCKET_EDGES
            },
        },
        "relational": candidate,
        "collapsed_state": collapsed,
        "gate": gate,
        "row_identity": {
            "identical_rows": [row["sample_id"] for row in candidate_rows]
            == [row["sample_id"] for row in collapsed_rows],
            "commands": len(rows),
            "evidence_valid_tasks": len(accepted_ids),
        },
    }
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def _freeze(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    task_ids, _clauses, commands = load_run_rows(args.run_dir)
    if len(task_ids) != 100:
        raise ValueError("relational fit requires exactly 100 development tasks")
    split, split_sha256 = _load_split_manifest(args.split_manifest)
    if task_ids != split["development"]:
        raise ValueError("development run order differs from the frozen split manifest")
    declared_run = Path(str(split.get("development_run", "")))
    if not declared_run.is_absolute():
        declared_run = _REPO_ROOT / declared_run
    if declared_run.resolve() != args.run_dir.resolve():
        raise ValueError("development run differs from the frozen split manifest")
    events = _load_exec_events(args.run_dir, list(task_ids))
    queries = build_queries(task_ids, commands, events)
    evidence, private, prior_artifact = _label_free_selected_evidence(
        task_ids, commands, queries, args.prior_dir
    )
    prior_source = str(prior_artifact.get("generation", {}).get("source", ""))
    if (
        prior_artifact.get("source_sha256")
        != hashlib.sha256(prior_source.encode()).hexdigest()
        or (args.prior_dir / "generated_feature.py").read_bytes()
        != prior_source.rstrip().encode() + b"\n"
    ):
        raise ValueError("prior frozen source or hash differs")
    prompt = GENERATION_PROMPT + json.dumps(evidence, separators=(",", ":"))
    if len(prompt.encode()) > MAX_PROMPT_BYTES:
        raise ValueError("relational generation prompt exceeds the frozen cost ceiling")
    args.out_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="relational-agent-") as directory:
        work = Path(directory)
        generation, cost = _codex_call(prompt, GENERATION_SCHEMA, work, "generation")
        for path in work.iterdir():
            if path.is_file():
                (args.out_dir / path.name).write_bytes(path.read_bytes())
    if (
        not isinstance(generation, dict)
        or set(generation) != {"abstain", "source", "explanation"}
        or not isinstance(generation.get("abstain"), bool)
        or not isinstance(generation.get("source"), str)
        or not isinstance(generation.get("explanation"), str)
    ):
        raise ValueError("generation differs from the frozen schema")
    source = generation["source"]
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "model": MODEL,
        "service_tier": "fast",
        "reasoning_effort": "medium",
        "source_sha256": None if not source.strip() else hashlib.sha256(source.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "evidence_sha256": hashlib.sha256(_json_bytes(evidence)).hexdigest(),
        "split_manifest_sha256": split_sha256,
        "development_task_ids_sha256": hashlib.sha256(
            _json_bytes(split["development"])
        ).hexdigest(),
        "prior_source_sha256": prior_artifact.get("source_sha256"),
        "generation": generation,
        "cost": cost,
    }
    if generation["abstain"]:
        if source.strip():
            raise ValueError("abstention contains source")
        artifact["status"] = "development_structural_no_go_abstained"
        (args.out_dir / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        return
    if not source.strip():
        raise ValueError("non-abstention contains no source")
    validate_source(source, (*private, *(item["task_id"] for item in evidence["selected_rows"])))
    training_samples = [f"{row.task_id}:{row.call_index}" for row in commands]
    validate_source_literals(
        source,
        {sample_id: queries[sample_id] for sample_id in training_samples},
        {f"{row.task_id}:{row.call_index}": row.task_id for row in commands},
    )
    ordered_queries = [queries[sample_id] for sample_id in training_samples]
    extracted, durations = run_source(source, ordered_queries)
    outputs = dict(zip(training_samples, extracted, strict=True))
    empty_outputs, _ = run_source(
        source, [{**query, "prior_events": []} for query in ordered_queries]
    )
    if any(value is not None for value in empty_outputs):
        raise ValueError("empty causal history produced a relational state")
    blank_queries = [
        {
            "current_command": "",
            "parsed_clauses": {},
            "prior_events": [
                {**event, "command": "", "exit_code": None, "result_excerpt": ""}
                for event in query["prior_events"]
            ],
        }
        for query in ordered_queries
    ]
    blank_outputs, _ = run_source(source, blank_queries)
    if any(value is not None for value in blank_outputs):
        raise ValueError("blank semantics produced a relational state")
    pmfs, support = _fit_pmfs(task_ids, commands, outputs, warmup_tasks=100)
    collapsed_pmfs, collapsed_support = _fit_pmfs(
        task_ids,
        commands,
        _collapsed_outputs(outputs),
        warmup_tasks=100,
    )
    primary, state_tasks = _select_primary_contrast(commands, outputs, pmfs)
    artifact.update(
        {
            "status": (
                "development_structural_go"
                if primary is not None
                else "development_structural_no_go_no_contrast"
            ),
            "minimum_support_tasks": MINIMUM_SUPPORT_TASKS,
            "primary_contrast": primary,
            "state_tasks": state_tasks,
            "pmfs": {
                f"{signature[0]}::{signature[1]}::{target}": list(pmf)
                for (signature, target), pmf in sorted(pmfs.items())
            },
            "support": support,
            "collapsed_pmfs": {
                f"{signature[0]}::{target}": list(pmf)
                for (signature, target), pmf in sorted(collapsed_pmfs.items())
            },
            "collapsed_support": collapsed_support,
            "coverage": {
                "commands": len(commands),
                "non_null": sum(value is not None for value in extracted),
                "tasks": len(
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
                "p95_ms": sorted(durations)[min(len(durations) - 1, int(0.95 * len(durations)))] / 1_000_000,
            },
        }
    )
    (args.out_dir / "generated_parser.py").write_text(source.rstrip() + "\n")
    (args.out_dir / "fit-graphs.jsonl").write_text(
        "".join(
            json.dumps({"sample_id": sample_id, "graph": outputs[sample_id]}, sort_keys=True) + "\n"
            for sample_id in training_samples
        )
    )
    (args.out_dir / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if sys.argv[1:] == ["--worker"]:
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        sys.stdout.write(json.dumps(_worker(json.load(sys.stdin))))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--run-dir", type=Path, required=True)
    freeze.add_argument("--prior-dir", type=Path, required=True)
    freeze.add_argument("--split-manifest", type=Path, required=True)
    freeze.add_argument("--out-dir", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--role", choices=("validation", "final_test"), required=True)
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--development-run", type=Path, required=True)
    evaluate.add_argument("--public-telemetry", type=Path, action="append", required=True)
    evaluate.add_argument("--split-manifest", type=Path, required=True)
    evaluate.add_argument("--artifact-dir", type=Path, required=True)
    evaluate.add_argument("--validation-result", type=Path)
    evaluate.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        _freeze(args)
    elif args.command == "evaluate":
        _evaluate(args)


if __name__ == "__main__":
    main()
