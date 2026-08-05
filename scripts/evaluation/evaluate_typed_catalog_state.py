#!/usr/bin/env python3
"""Select and evaluate one deterministic typed relational configuration."""

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
    evaluate_prequential_commands,
)
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    load_rows,
    load_run_rows,
)
from scripts.evaluation.evaluate_command_prequential import _load_exec_events  # noqa: E402
from scripts.evaluation.evaluate_declarative_family_state import (  # noqa: E402
    _collapsed_outputs,
    _episode,
    _fresh_gate as _family_fresh_gate,
    _query_exceeds_bounds,
    _reserved_role_complete,
    invocation_key,
    select_primary_contrasts,
)
from scripts.evaluation.evaluate_declarative_relational_state import (  # noqa: E402
    MAX_BLOCKERS,
    MAX_EDGES,
    MAX_P95_MS,
    MAX_SPANS_PER_PATTERN,
    MINIMUM_SUPPORT_TASKS,
    SPLIT_MANIFEST,
    BoundExceeded,
    SpecError,
    _committed_file,
    _decode_pmfs,
    _encode_pmfs,
    _fit_fingerprint,
    _frozen_public_inputs,
    _host_identity,
    _json_bytes,
    _load_split_manifest,
    _validate_host_identity,
    _validate_preregistration_commit,
    _write_artifact,
)
from scripts.evaluation.evaluate_offline_agent_extractor import (  # noqa: E402
    MODEL,
    TARGETS,
    _apply_candidate,
    _codex_call,
    _fit_pmfs,
    _score_candidate,
    build_queries,
)
from scripts.evaluation.evaluate_relational_agent_state import (  # noqa: E402
    _all_command_queries,
    _attempt_records,
    _derive_state,
    _frozen_baseline,
    _offset_rows,
    _telemetry_valid_records,
    _write_result_view,
)
from tool_resource.runtime_kb import parse_command_clauses  # noqa: E402
from tool_resource_eval.labels import repo_of  # noqa: E402

SCHEMA = "offline-agent-typed-catalog-state-v1"
EVALUATION_SCHEMA = "typed-catalog-state-fresh-evaluation-v1"
MAX_PROMPT_BYTES = 50_000
MAX_CANDIDATE_FAMILIES = 4
MAX_SCOPES = 4
MAX_CONFIGS_PER_FAMILY = 4
MAX_CONFIGS = 16
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}", re.ASCII)
PREREGISTRATION_PATHS = (
    SPLIT_MANIFEST,
    _REPO_ROOT / "analysis/development/clause-interaction-kb-plan.md",
    _REPO_ROOT / "analysis/development/tool-resource-canonical-objective.md",
)

GENERATION_PROMPT = """Select one configuration from the finite, host-generated catalog below.
Return only the required JSON and do not call tools.

Prefer a configuration whose failed-command context describes a real blocker,
whose successful-command context is an environment-changing remediation, and
whose work scopes represent meaningfully different requested work. Reject
diagnostic commands that merely inspect state. You cannot emit or modify code,
regex, literals, templates, scopes, relations, states, thresholds, weights,
features, buckets, or predictions. Abstain if none is semantically credible.

LABEL-FREE AGGREGATE CATALOG:
"""


@dataclass(frozen=True, order=True)
class TokenTemplate:
    capture_mode: str
    context_tokens: tuple[str, ...]


@dataclass(frozen=True)
class CatalogConfig:
    configuration_id: str
    invocation_key: str
    scopes: tuple[str, ...]
    scope_support: tuple[tuple[str, int], ...]
    blocker: TokenTemplate
    remediation: TokenTemplate
    causal_task_count: int
    identifier_count: int


def _normalized_token(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "-", value.casefold()).strip("-")


def _tokens(text: str) -> list[tuple[str, str, int, int]]:
    return [
        (match.group(), _normalized_token(match.group()), *match.span())
        for match in TOKEN_RE.finditer(text)
    ]


def _occurrences(
    text: str, *, suffixes: bool
) -> set[tuple[TokenTemplate, str, tuple[int, int]]]:
    tokens = _tokens(text)
    output: set[tuple[TokenTemplate, str, tuple[int, int]]] = set()
    for index, (raw, normalized, start, end) in enumerate(tokens):
        captures = [("whole", normalized, (start, end))]
        suffix = re.search(r"[-_]([A-Za-z0-9]+)[-_]*$", raw)
        if suffixes and suffix is not None:
            suffix_id = _normalized_token(suffix.group(1))
            captures.append(
                (
                    "suffix",
                    suffix_id,
                    (start + suffix.start(1), start + suffix.end(1)),
                )
            )
        for width in range(1, min(3, index) + 1):
            context = tuple(token[1] for token in tokens[index - width : index])
            if not all(context):
                continue
            for mode, identifier, span in captures:
                if len(identifier) >= 2:
                    output.add((TokenTemplate(mode, context), identifier, span))
    return output


def template_spans(template: TokenTemplate, text: str) -> list[list[int]]:
    matches = {
        (identifier, span)
        for candidate, identifier, span in _occurrences(
            text, suffixes=template.capture_mode == "suffix"
        )
        if candidate == template
    }
    if len(matches) > MAX_SPANS_PER_PATTERN:
        raise BoundExceeded("typed template exceeded its span bound")
    by_identifier: dict[str, tuple[int, int]] = {}
    for identifier, span in sorted(matches, key=lambda item: item[1]):
        by_identifier.setdefault(identifier, span)
    return [list(span) for span in sorted(by_identifier.values())]


def scope_shape(command: str, key: str) -> str | None:
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
    offset = (
        3
        if len(argv) >= 3 and argv[1] == "-m" and _normalized_token(argv[2]) == key
        else 1
    )
    positionals = [value for value in argv[offset:] if not value.startswith("-")]
    kinds = []
    for value in positionals:
        if "::" in value:
            kind = "nodeid"
        elif value.casefold().endswith(".py"):
            kind = "python-file"
        elif "/" in value:
            kind = "path"
        elif "." in value:
            kind = "dotted"
        else:
            kind = "word"
        kinds.append(kind)
    if not kinds:
        return "no-target"
    distinct = sorted(set(kinds))
    return distinct[0] if len(kinds) == 1 else "multi-" + "+".join(distinct)


def _candidate_episodes(
    task_ids: Sequence[str], events_by_task: Mapping[str, Sequence[PipExecEvent]]
) -> list[tuple[str, list[tuple[str, list[dict[str, Any]]]]]]:
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
    return sorted(
        (
            (key, rows)
            for key, rows in episodes.items()
            if len(rows) >= MINIMUM_SUPPORT_TASKS
        ),
        key=lambda item: (-len(item[1]), item[0]),
    )[:MAX_CANDIDATE_FAMILIES]


def _config_rank(config: CatalogConfig) -> tuple[Any, ...]:
    supports = [count for _scope, count in config.scope_support]
    return (
        -config.causal_task_count,
        -config.identifier_count,
        -min(supports),
        -sum(supports),
        config.invocation_key,
        config.blocker,
        config.remediation,
    )


def build_catalog(
    task_ids: Sequence[str], events_by_task: Mapping[str, Sequence[PipExecEvent]]
) -> tuple[CatalogConfig, ...]:
    retained: list[CatalogConfig] = []
    for key, episodes in _candidate_episodes(task_ids, events_by_task):
        scope_tasks: dict[str, set[str]] = defaultdict(set)
        for task_id, episode in episodes:
            for row in episode:
                if row["kind"] != "family_verifier":
                    continue
                shape = scope_shape(str(row["command"]), key)
                if shape is not None:
                    scope_tasks[shape].add(task_id)
        scopes = sorted(
            (
                (shape, len(tasks))
                for shape, tasks in scope_tasks.items()
                if len(tasks) >= MINIMUM_SUPPORT_TASKS
            ),
            key=lambda item: (-item[1], item[0]),
        )[:MAX_SCOPES]
        if len(scopes) < 2:
            continue
        retained_scopes = {shape for shape, _count in scopes}
        blocker_tasks: dict[TokenTemplate, set[str]] = defaultdict(set)
        blocker_ids: dict[TokenTemplate, set[str]] = defaultdict(set)
        remediation_tasks: dict[TokenTemplate, set[str]] = defaultdict(set)
        remediation_ids: dict[TokenTemplate, set[str]] = defaultdict(set)
        edge_tasks: dict[tuple[TokenTemplate, TokenTemplate], set[str]] = defaultdict(set)
        edge_ids: dict[tuple[TokenTemplate, TokenTemplate], set[str]] = defaultdict(set)
        for task_id, episode in episodes:
            seen: dict[str, set[TokenTemplate]] = defaultdict(set)
            for row in episode:
                exit_code = row["exit_code"]
                if row["kind"] == "family_verifier" and exit_code not in (0, None):
                    if scope_shape(str(row["command"]), key) not in retained_scopes:
                        continue
                    for template, identifier, _span in _occurrences(
                        str(row["result_excerpt"]), suffixes=False
                    ):
                        blocker_tasks[template].add(task_id)
                        blocker_ids[template].add(identifier)
                        seen[identifier].add(template)
                if exit_code != 0:
                    continue
                for template, identifier, _span in _occurrences(
                    str(row["command"]), suffixes=True
                ):
                    remediation_tasks[template].add(task_id)
                    remediation_ids[template].add(identifier)
                    for blocker in seen.get(identifier, ()):
                        pair = (blocker, template)
                        edge_tasks[pair].add(task_id)
                        edge_ids[pair].add(identifier)
        family_configs = []
        for (blocker, remediation), causal_tasks in edge_tasks.items():
            identifiers = edge_ids[blocker, remediation]
            if (
                len(blocker_tasks[blocker]) < MINIMUM_SUPPORT_TASKS
                or len(blocker_ids[blocker]) < 2
                or len(remediation_tasks[remediation]) < MINIMUM_SUPPORT_TASKS
                or len(remediation_ids[remediation]) < 2
                or len(causal_tasks) < MINIMUM_SUPPORT_TASKS
                or len(identifiers) < 2
            ):
                continue
            family_configs.append(
                CatalogConfig(
                    configuration_id="",
                    invocation_key=key,
                    scopes=tuple(shape for shape, _count in scopes),
                    scope_support=tuple(scopes),
                    blocker=blocker,
                    remediation=remediation,
                    causal_task_count=len(causal_tasks),
                    identifier_count=len(identifiers),
                )
            )
        retained.extend(sorted(set(family_configs), key=_config_rank)[:MAX_CONFIGS_PER_FAMILY])
    ranked = sorted(set(retained), key=_config_rank)[:MAX_CONFIGS]
    return tuple(
        CatalogConfig(
            configuration_id=f"C{index:03d}",
            invocation_key=config.invocation_key,
            scopes=config.scopes,
            scope_support=config.scope_support,
            blocker=config.blocker,
            remediation=config.remediation,
            causal_task_count=config.causal_task_count,
            identifier_count=config.identifier_count,
        )
        for index, config in enumerate(ranked)
    )


def _template_json(template: TokenTemplate) -> dict[str, Any]:
    return {
        "capture_mode": template.capture_mode,
        "context_tokens": list(template.context_tokens),
    }


def _config_json(config: CatalogConfig) -> dict[str, Any]:
    return {
        "configuration_id": config.configuration_id,
        "invocation_key": config.invocation_key,
        "scopes": [
            {"shape": shape, "task_support": support}
            for shape, support in config.scope_support
        ],
        "blocker_template": _template_json(config.blocker),
        "remediation_template": _template_json(config.remediation),
        "causal_task_count": config.causal_task_count,
        "identifier_count": config.identifier_count,
    }


def _config_from_json(value: Any) -> CatalogConfig:
    if not isinstance(value, dict) or set(value) != {
        "configuration_id",
        "invocation_key",
        "scopes",
        "blocker_template",
        "remediation_template",
        "causal_task_count",
        "identifier_count",
    }:
        raise ValueError("frozen catalog configuration is invalid")

    def template(field: str) -> TokenTemplate:
        item = value[field]
        if not isinstance(item, dict) or set(item) != {"capture_mode", "context_tokens"}:
            raise ValueError("frozen token template is invalid")
        mode = str(item["capture_mode"])
        context = item["context_tokens"]
        if mode not in {"whole", "suffix"} or not isinstance(context, list) or not 1 <= len(context) <= 3:
            raise ValueError("frozen token template value is invalid")
        return TokenTemplate(mode, tuple(str(token) for token in context))

    scopes = value["scopes"]
    if not isinstance(scopes, list) or not 2 <= len(scopes) <= MAX_SCOPES:
        raise ValueError("frozen scope list is invalid")
    support = tuple((str(item["shape"]), int(item["task_support"])) for item in scopes)
    config = CatalogConfig(
        str(value["configuration_id"]),
        str(value["invocation_key"]),
        tuple(shape for shape, _count in support),
        support,
        template("blocker_template"),
        template("remediation_template"),
        int(value["causal_task_count"]),
        int(value["identifier_count"]),
    )
    if (
        re.fullmatch(r"C[0-9]{3}", config.configuration_id) is None
        or not config.invocation_key
        or any(count < MINIMUM_SUPPORT_TASKS for _shape, count in support)
        or config.causal_task_count < MINIMUM_SUPPORT_TASKS
        or config.identifier_count < 2
    ):
        raise ValueError("frozen catalog support is invalid")
    return config


def build_generation_input(catalog: Sequence[CatalogConfig]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if len(catalog) < 2:
        raise SpecError("fewer than two complete typed configurations")
    payload = {
        "schema": "typed-relational-catalog-v1",
        "configurations": [_config_json(config) for config in catalog],
        "omitted": [
            "raw_examples",
            "captured_identifiers",
            "durations",
            "telemetry",
            "resource_labels",
            "current_predictions",
        ],
    }
    prompt = GENERATION_PROMPT + json.dumps(payload, separators=(",", ":"), sort_keys=True)
    if len(prompt.encode()) > MAX_PROMPT_BYTES:
        raise SpecError("typed catalog prompt exceeds its frozen byte budget")
    identifiers = [config.configuration_id for config in catalog]
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["abstain", "configuration_id", "explanation"],
        "properties": {
            "abstain": {"type": "boolean"},
            "configuration_id": {"type": "string", "enum": ["", *identifiers]},
            "explanation": {"type": "string", "minLength": 1, "maxLength": 1_000},
        },
    }
    return payload, prompt, schema


def validate_selection(value: Any, catalog: Sequence[CatalogConfig]) -> str | None:
    if not isinstance(value, dict) or set(value) != {"abstain", "configuration_id", "explanation"}:
        raise SpecError("generation response has unexpected fields")
    abstain = value["abstain"]
    selected = value["configuration_id"]
    explanation = value["explanation"]
    if (
        not isinstance(abstain, bool)
        or not isinstance(selected, str)
        or not isinstance(explanation, str)
        or not 1 <= len(explanation) <= 1_000
    ):
        raise SpecError("generation response has invalid values")
    if abstain:
        if selected:
            raise SpecError("abstention must use an empty configuration ID")
        return None
    if selected not in {config.configuration_id for config in catalog}:
        raise SpecError("generation selected an unknown configuration")
    return selected


def match_scope(config: CatalogConfig, command: str) -> str | None:
    if invocation_key(command) != config.invocation_key:
        return None
    shape = scope_shape(command, config.invocation_key)
    return shape if shape in config.scopes else None


def extract_query(
    config: CatalogConfig, query: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, bool]:
    if _query_exceeds_bounds(query):
        return None, True
    shape = match_scope(config, str(query["current_command"]))
    if shape is None:
        return None, False
    namespace = {
        "scope": lambda _command, _parsed: config.configuration_id.casefold(),
        "blocker_spans": lambda text: template_spans(config.blocker, text),
        "remediation_spans": lambda text: template_spans(config.remediation, text),
    }
    try:
        graph = _derive_state(
            query,
            namespace,
            verifier_match=lambda command: match_scope(config, command) is not None,
        )
    except SpecError:
        return None, True
    if graph is None:
        return None, False
    edges = sum(len(item["addresses"]) for item in graph["remediations"])
    if len(graph["blockers"]) > MAX_BLOCKERS or edges > MAX_EDGES:
        return None, True
    dependency_state = str(graph["state"])
    return {
        **graph,
        "scope_id": shape,
        "dependency_state": dependency_state,
        "state": f"{shape}::{dependency_state}",
    }, False


def run_queries(
    config: CatalogConfig, queries: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any] | None], list[int], int]:
    outputs = []
    durations = []
    bounded = 0
    for query in queries:
        started = time.perf_counter_ns()
        output, exceeded = extract_query(config, query)
        durations.append(time.perf_counter_ns() - started)
        outputs.append(output)
        bounded += exceeded
    return outputs, durations, bounded


def _run_mapping(
    config: CatalogConfig,
    samples: Sequence[str],
    queries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any] | None], list[int], int]:
    outputs, durations, bounded = run_queries(config, [queries[sample] for sample in samples])
    return dict(zip(samples, outputs, strict=True)), durations, bounded


def _percentile_ms(values: Sequence[int]) -> float:
    return sorted(values)[min(len(values) - 1, int(0.95 * len(values)))] / 1_000_000


def _carrier_coverage(
    commands: Sequence[CommandRow], outputs: Sequence[Mapping[str, Any] | None], bounded: int
) -> dict[str, Any]:
    selected = [row for row, value in zip(commands, outputs, strict=True) if value is not None]
    report = {
        "commands": len(selected),
        "tasks": len({row.task_id for row in selected}),
        "bounded_fallbacks": bounded,
        "minimum_commands": 20,
        "minimum_tasks": 5,
    }
    report["passed"] = bounded == 0 and report["commands"] >= 20 and report["tasks"] >= 5
    return report


def _coverage(
    commands: Sequence[CommandRow],
    agent_outputs: Sequence[Mapping[str, Any] | None],
    support_outputs: Sequence[Mapping[str, Any] | None],
    primary: Mapping[str, Any],
    agent_bounded: int,
    support_bounded: int,
) -> dict[str, Any]:
    from scripts.evaluation.evaluate_declarative_family_state import _coverage as family_coverage

    agent = family_coverage(commands, agent_outputs, primary, agent_bounded)
    support = _carrier_coverage(commands, support_outputs, support_bounded)
    return {"passed": agent["passed"] and support["passed"], "agent": agent, "support_only": support}


def _comparison_changes(
    agent_rows: Sequence[Mapping[str, Any]], support_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_target = {}
    helpful_tasks: set[str] = set()
    for target in TARGETS:
        changed = helpful = harmful = 0
        for agent, support in zip(agent_rows, support_rows, strict=True):
            left = agent["candidate"][target]
            right = support["candidate"][target]
            truth = agent["labels"][target]
            if left == right:
                continue
            changed += 1
            if truth is None:
                continue
            left_bucket = (
                None
                if left is None
                else int(left)
                if target == "latency"
                else RESOURCE_BUCKET_LABELS.index(str(left))
            )
            right_bucket = (
                None
                if right is None
                else int(right)
                if target == "latency"
                else RESOURCE_BUCKET_LABELS.index(str(right))
            )
            if left_bucket == int(truth) and right_bucket != int(truth):
                helpful += 1
                helpful_tasks.add(str(agent["task_id"]))
            elif right_bucket == int(truth) and left_bucket != int(truth):
                harmful += 1
        by_target[target] = {"changed": changed, "helpful": helpful, "harmful": harmful}
    return {
        "by_target": by_target,
        "changed": sum(row["changed"] for row in by_target.values()),
        "helpful": sum(row["helpful"] for row in by_target.values()),
        "harmful": sum(row["harmful"] for row in by_target.values()),
        "helpful_tasks": len(helpful_tasks),
    }


def _accuracy(score: Mapping[str, Any], target: str) -> float:
    key = "exact_class_accuracy" if target == "latency" else "accuracy"
    return float(score["metrics"][target][key])


def _fresh_gate(
    baseline: Mapping[str, Any],
    full: Mapping[str, Any],
    family: Mapping[str, Any],
    scope: Mapping[str, Any],
    support: Mapping[str, Any],
    full_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    scope_rows: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
    primary: Mapping[str, Any],
) -> dict[str, Any]:
    gate = _family_fresh_gate(
        baseline, full, family, scope, full_rows, family_rows, scope_rows, primary
    )
    accuracy_safe = all(_accuracy(full, target) >= _accuracy(support, target) for target in TARGETS)
    severe_safe = all(
        full["metrics"][target]["severe_underprediction_rate"]
        <= support["metrics"][target]["severe_underprediction_rate"]
        for target in TARGETS
    )
    changes = _comparison_changes(full_rows, support_rows)
    gate.update(
        {
            "no_accuracy_regression_vs_support_only": accuracy_safe,
            "no_severe_underprediction_regression_vs_support_only": severe_safe,
            "support_only_changes": changes,
        }
    )
    gate["go"] = bool(
        gate["go"]
        and accuracy_safe
        and severe_safe
        and changes["changed"] > 0
        and changes["helpful"] > changes["harmful"]
        and changes["helpful_tasks"] >= 3
    )
    return gate


def _generation_files(directory: Path) -> list[dict[str, str]]:
    return [
        {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(directory.glob("generation.*"))
    ]


def _artifact_base(
    response: Mapping[str, Any],
    prompt: str,
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
    split_sha256: str,
    preregistration_commit: str,
    host_identity: Mapping[str, Any],
    cost: Mapping[str, Any],
    public_inputs: Sequence[Mapping[str, str]],
    generation_files: Sequence[Mapping[str, str]],
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
        "generation_schema_sha256": hashlib.sha256(_json_bytes(schema)).hexdigest(),
        "catalog_sha256": hashlib.sha256(_json_bytes(payload)).hexdigest(),
        "split_manifest_sha256": split_sha256,
        "preregistration_commit": preregistration_commit,
        "host_identity": dict(host_identity),
        "generation_files": list(generation_files),
        "generation": dict(response),
        "cost": dict(cost),
        "public_inputs": list(public_inputs),
        "support_only_configuration_id": "C000",
        "validation_consumed": False,
        "final_test_consumed": False,
    }


def freeze(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    if args.split_manifest.resolve() != SPLIT_MANIFEST.resolve():
        raise ValueError("split manifest path differs from the preregistration")
    preregistration_commits = {_committed_file(path)[1] for path in PREREGISTRATION_PATHS}
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
    catalog = build_catalog(split["development"], events)
    payload, prompt, schema = build_generation_input(catalog)

    args.out_dir.mkdir(parents=True)
    (args.out_dir / "catalog.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with tempfile.TemporaryDirectory(prefix="typed-catalog-") as directory:
        work = Path(directory)
        response, cost = _codex_call(prompt, schema, work, "generation")
        for path in work.iterdir():
            if path.is_file():
                (args.out_dir / path.name).write_bytes(path.read_bytes())
    artifact = _artifact_base(
        response,
        prompt,
        payload,
        schema,
        split_sha256,
        next(iter(preregistration_commits)),
        host_identity,
        cost,
        public_inputs,
        _generation_files(args.out_dir),
    )
    artifact["development_task_ids_sha256"] = hashlib.sha256(
        _json_bytes(split["development"])
    ).hexdigest()
    try:
        selected_id = validate_selection(response, catalog)
    except SpecError as error:
        artifact.update(
            {
                "status": "development_structural_no_go_invalid_selection",
                "structural_error": str(error),
            }
        )
        _write_artifact(args.out_dir, artifact)
        return
    artifact["selected_configuration_id"] = selected_id
    if selected_id is None:
        artifact["status"] = "development_structural_no_go_abstained"
        _write_artifact(args.out_dir, artifact)
        return
    if selected_id == "C000":
        artifact["status"] = "development_structural_no_go_support_baseline_selected"
        _write_artifact(args.out_dir, artifact)
        return

    try:
        selected = next(config for config in catalog if config.configuration_id == selected_id)
        support = catalog[0]
        task_ids, clauses, commands = load_run_rows(args.run_dir)
        if task_ids != split["development"]:
            raise ValueError("development task order differs from the frozen split")
        artifact["development_fit_sha256"] = _fit_fingerprint(task_ids, clauses, commands)
        queries = build_queries(task_ids, commands, events)
        samples = [f"{row.task_id}:{row.call_index}" for row in commands]
        selected_outputs, selected_durations, selected_bounded = _run_mapping(
            selected, samples, queries
        )
        support_outputs, support_durations, support_bounded = _run_mapping(
            support, samples, queries
        )
        if selected_bounded or support_bounded:
            raise BoundExceeded("development queries exceeded a frozen bound")
        for config in (selected, support):
            empty, _durations, bounded = run_queries(
                config, [{**queries[sample], "prior_events": []} for sample in samples]
            )
            if bounded or any(value is not None for value in empty):
                raise SpecError("empty history produced a typed relation state")
        family_outputs = _collapsed_outputs(selected_outputs, dimension="family")
        scope_outputs = _collapsed_outputs(selected_outputs, dimension="scope")
        full_pmfs, full_support = _fit_pmfs(
            task_ids, commands, selected_outputs, warmup_tasks=100
        )
        family_pmfs, family_support = _fit_pmfs(
            task_ids, commands, family_outputs, warmup_tasks=100
        )
        scope_pmfs, scope_support = _fit_pmfs(
            task_ids, commands, scope_outputs, warmup_tasks=100
        )
        support_pmfs, support_report = _fit_pmfs(
            task_ids, commands, support_outputs, warmup_tasks=100
        )
        contrasts = select_primary_contrasts(commands, selected_outputs, full_pmfs)
        selected_p95 = _percentile_ms(selected_durations)
        support_p95 = _percentile_ms(support_durations)
        artifact.update(
            {
                "status": (
                    "development_structural_go"
                    if contrasts is not None and selected_p95 <= MAX_P95_MS
                    else "development_structural_no_go_no_contrasts_or_runtime"
                ),
                "primary_contrasts": contrasts,
                "full_pmfs": _encode_pmfs(full_pmfs),
                "family_only_pmfs": _encode_pmfs(family_pmfs),
                "scope_only_pmfs": _encode_pmfs(scope_pmfs),
                "support_only_pmfs": _encode_pmfs(support_pmfs),
                "full_support": full_support,
                "family_only_support": family_support,
                "scope_only_support": scope_support,
                "support_only_support": support_report,
                "coverage": {
                    "commands": len(commands),
                    "agent_relation_commands": sum(
                        value is not None for value in selected_outputs.values()
                    ),
                    "support_relation_commands": sum(
                        value is not None for value in support_outputs.values()
                    ),
                },
                "runtime": {
                    "agent_p50_ms": statistics.median(selected_durations) / 1_000_000,
                    "agent_p95_ms": selected_p95,
                    "support_p50_ms": statistics.median(support_durations) / 1_000_000,
                    "support_p95_ms": support_p95,
                    "p95_limit_ms": MAX_P95_MS,
                },
            }
        )
        (args.out_dir / "fit-graphs.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "sample_id": sample,
                        "agent_graph": selected_outputs[sample],
                        "support_only_graph": support_outputs[sample],
                    },
                    sort_keys=True,
                )
                + "\n"
                for sample in samples
            )
        )
    except SpecError as error:
        artifact.update(
            {
                "status": "development_structural_no_go_invalid_runtime",
                "structural_error": str(error),
            }
        )
    _write_artifact(args.out_dir, artifact)


def _validate_generation_binding(
    artifact_dir: Path,
    artifact: Mapping[str, Any],
    catalog_payload: Mapping[str, Any],
) -> None:
    prompt = (artifact_dir / "generation.prompt.txt").read_text()
    schema = json.loads((artifact_dir / "generation.schema.json").read_text())
    response = json.loads((artifact_dir / "generation.response.json").read_text())
    _payload, expected_prompt, expected_schema = build_generation_input(
        tuple(_config_from_json(row) for row in catalog_payload["configurations"])
    )
    if (
        prompt != expected_prompt
        or schema != expected_schema
        or response != artifact.get("generation")
        or hashlib.sha256(prompt.encode()).hexdigest() != artifact.get("prompt_sha256")
        or hashlib.sha256(_json_bytes(schema)).hexdigest()
        != artifact.get("generation_schema_sha256")
        or artifact.get("cost", {}).get("prompt_bytes") != len(prompt.encode())
    ):
        raise ValueError("frozen generation prompt, schema, response, or cost differs")


def _load_artifact(
    artifact_dir: Path, split: Mapping[str, Any], split_sha256: str
) -> tuple[dict[str, Any], CatalogConfig, CatalogConfig, str, str]:
    artifact_bytes = (artifact_dir / "artifact.json").read_bytes()
    artifact = json.loads(artifact_bytes)
    payload = json.loads((artifact_dir / "catalog.json").read_text())
    configs = tuple(_config_from_json(row) for row in payload.get("configurations", ()))
    selected_id = artifact.get("selected_configuration_id")
    expected_task_hash = hashlib.sha256(_json_bytes(split["development"])).hexdigest()
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("status") != "development_structural_go"
        or artifact.get("split_manifest_sha256") != split_sha256
        or artifact.get("development_task_ids_sha256") != expected_task_hash
        or artifact.get("catalog_sha256") != hashlib.sha256(_json_bytes(payload)).hexdigest()
        or artifact.get("support_only_configuration_id") != "C000"
        or selected_id in (None, "C000")
        or not isinstance(artifact.get("primary_contrasts"), dict)
    ):
        raise ValueError("frozen typed-catalog artifact is incomplete or differs")
    selected = next((config for config in configs if config.configuration_id == selected_id), None)
    support = next((config for config in configs if config.configuration_id == "C000"), None)
    if selected is None or support is None:
        raise ValueError("selected or support-only configuration is absent")
    actual_generation = _generation_files(artifact_dir)
    if artifact.get("generation_files") != actual_generation:
        raise ValueError("frozen generation transcript differs")
    _validate_generation_binding(artifact_dir, artifact, payload)
    validate_selection(artifact["generation"], configs)
    for field in ("full_pmfs", "family_only_pmfs", "scope_only_pmfs", "support_only_pmfs"):
        _decode_pmfs(artifact.get(field))
    artifact_commit = None
    for path in (
        artifact_dir / "artifact.json",
        artifact_dir / "catalog.json",
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
    return artifact, selected, support, hashlib.sha256(artifact_bytes).hexdigest(), artifact_commit


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
    artifact, selected, support, artifact_sha256, artifact_commit = _load_artifact(
        args.artifact_dir, split, split_sha256
    )
    validation = None
    validation_sha256 = validation_commit = None
    if args.role == "final_test":
        if args.validation_result is None:
            raise ValueError("final_test requires a passing validation result")
        validation_bytes, validation_commit = _committed_file(args.validation_result)
        validation = json.loads(validation_bytes)
        if not _validation_authorizes_final(validation, artifact_sha256, split_sha256):
            raise ValueError("final_test is not authorized by matching validation GO")
        validation_rows, rows_commit = _committed_file(args.validation_result.with_name("rows.jsonl"))
        if rows_commit != validation_commit or hashlib.sha256(validation_rows).hexdigest() != validation["rows_sha256"]:
            raise ValueError("validation rows differ from committed authorization")
        validation_sha256 = hashlib.sha256(validation_bytes).hexdigest()
    elif args.validation_result is not None:
        raise ValueError("validation must not consume a prior result")

    development_ids, development_clauses, development_commands = load_run_rows(args.development_run)
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
    with tempfile.TemporaryDirectory(prefix="typed-role-") as directory:
        view = Path(directory) / "results.jsonl"
        _write_result_view(valid_records, view)
        events = _load_exec_events(args.run_dir, valid_ids, results_path=view)
        coverage_commands, coverage_queries = _all_command_queries(valid_ids, events)
        samples = [f"{row.task_id}:{row.call_index}" for row in coverage_commands]
        selected_coverage, _durations, selected_bounded = _run_mapping(
            selected, samples, coverage_queries
        )
        support_coverage, _durations, support_bounded = _run_mapping(
            support, samples, coverage_queries
        )
        coverage = _coverage(
            coverage_commands,
            list(selected_coverage.values()),
            list(support_coverage.values()),
            artifact["primary_contrasts"],
            selected_bounded,
            support_bounded,
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
            (args.out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            return
        role_task_ids, role_clauses, role_commands = load_run_rows(args.run_dir, results_path=view)
        if role_task_ids != valid_ids:
            raise AssertionError("role task order differs after telemetry validation")
        role_queries = build_queries(role_task_ids, role_commands, events)
        role_samples = [f"{row.task_id}:{row.call_index}" for row in role_commands]
        selected_outputs, _durations, selected_bounded = _run_mapping(
            selected, role_samples, role_queries
        )
        support_outputs, _durations, support_bounded = _run_mapping(
            support, role_samples, role_queries
        )
        if selected_bounded or support_bounded:
            raise BoundExceeded("scored queries exceeded a frozen bound")
        family_outputs = _collapsed_outputs(selected_outputs, dimension="family")
        scope_outputs = _collapsed_outputs(selected_outputs, dimension="scope")
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
    full_rows = _apply_candidate(frozen_rows, selected_outputs, _decode_pmfs(artifact["full_pmfs"]))
    family_rows = _apply_candidate(frozen_rows, family_outputs, _decode_pmfs(artifact["family_only_pmfs"]))
    scope_rows = _apply_candidate(frozen_rows, scope_outputs, _decode_pmfs(artifact["scope_only_pmfs"]))
    support_rows = _apply_candidate(frozen_rows, support_outputs, _decode_pmfs(artifact["support_only_pmfs"]))
    full_score = _score_candidate(frozen_baseline, full_rows)
    family_score = _score_candidate(frozen_baseline, family_rows)
    scope_score = _score_candidate(frozen_baseline, scope_rows)
    support_score = _score_candidate(frozen_baseline, support_rows)
    gate = _fresh_gate(
        frozen_baseline,
        full_score,
        family_score,
        scope_score,
        support_score,
        full_rows,
        family_rows,
        scope_rows,
        support_rows,
        artifact["primary_contrasts"],
    )
    rows = [
        {
            **full,
            "role": args.role,
            "family_only_candidate": family["candidate"],
            "family_only_probability_by_bucket": family["candidate_probability_by_bucket"],
            "scope_only_candidate": scope["candidate"],
            "scope_only_probability_by_bucket": scope["candidate_probability_by_bucket"],
            "support_only_candidate": support_row["candidate"],
            "support_only_probability_by_bucket": support_row["candidate_probability_by_bucket"],
        }
        for full, family, scope, support_row in zip(
            full_rows, family_rows, scope_rows, support_rows, strict=True
        )
    ]
    rows_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
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
        "support_only": support_score,
        "gate": gate,
        "row_identity": {
            "identical_rows": [row["sample_id"] for row in full_rows]
            == [row["sample_id"] for row in family_rows]
            == [row["sample_id"] for row in scope_rows]
            == [row["sample_id"] for row in support_rows],
            "commands": len(rows),
            "evidence_valid_tasks": len(valid_ids),
        },
        "rows_sha256": hashlib.sha256(rows_bytes).hexdigest(),
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "rows.jsonl").write_bytes(rows_bytes)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--run-dir", type=Path, required=True)
    freeze_parser.add_argument("--out-dir", type=Path, required=True)
    freeze_parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST)
    freeze_parser.add_argument("--public-telemetry", type=Path, action="append", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--role", choices=("validation", "final_test"), required=True)
    evaluate_parser.add_argument("--run-dir", type=Path, required=True)
    evaluate_parser.add_argument("--development-run", type=Path, required=True)
    evaluate_parser.add_argument("--public-telemetry", type=Path, action="append", required=True)
    evaluate_parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST)
    evaluate_parser.add_argument("--artifact-dir", type=Path, required=True)
    evaluate_parser.add_argument("--validation-result", type=Path)
    evaluate_parser.add_argument("--out-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    freeze(arguments) if arguments.command == "freeze" else evaluate(arguments)
