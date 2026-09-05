#!/usr/bin/env python3
"""Fit the fixed full-test phase feature and evaluate one fresh SQLGlot role."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    CANONICAL_LATENCY_BUCKETS,
    CANONICAL_RESOURCE_BUCKET_EDGES,
    FULL_TEST_PHASE_FEATURE_VERSION,
    CommandRow,
    PipExecEvent,
    RESOURCE_BUCKET_LABELS,
    _argmax_probabilities,
    _empirical_pmf,
    _full_test_phases,
    command_resource_bucket_label,
    evaluate_prequential_commands,
)
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    load_rows,
    load_run_rows,
)
from scripts.evaluation.evaluate_command_prequential import _load_exec_events  # noqa: E402
from scripts.evaluation.evaluate_declarative_family_state import (  # noqa: E402
    _reserved_role_complete,
)
from scripts.evaluation.evaluate_declarative_relational_state import (  # noqa: E402
    SPLIT_MANIFEST,
    _committed_file,
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
    _score_candidate,
)
from scripts.evaluation.evaluate_relational_agent_state import (  # noqa: E402
    _attempt_records,
    _frozen_baseline,
    _offset_rows,
    _telemetry_valid_records,
    _write_result_view,
)
from tool_resource_eval.labels import repo_of  # noqa: E402

SCHEMA = "full-test-phase-fresh-fit-v1"
EVALUATION_SCHEMA = "full-test-phase-fresh-evaluation-v1"
PHASE_TARGETS = ("latency", "peak_cpu_cores", "sampled_peak_rss_mb")
MINIMUM_FIT_TASKS = 5
MINIMUM_FRESH_TASKS = 3
PREREGISTRATION_PATHS = (
    SPLIT_MANIFEST,
    _REPO_ROOT / "analysis/development/clause-interaction-kb-plan.md",
    _REPO_ROOT / "analysis/development/tool-resource-canonical-objective.md",
)


def _phase_rows(
    task_ids: Sequence[str],
    commands: Sequence[CommandRow],
    events_by_task: Mapping[str, Sequence[PipExecEvent]],
) -> tuple[list[CommandRow], dict[str, dict[str, int]]]:
    if list(events_by_task) != list(task_ids):
        raise ValueError("raw event tasks differ from the committed task order")
    phases = {}
    for task_id in task_ids:
        events = events_by_task[task_id]
        event_by_call = {event.call_id: event for event in events}
        if len(event_by_call) != len(events):
            raise ValueError(f"{task_id}: duplicate raw exec call id")
        for row in (row for row in commands if row.task_id == task_id):
            event = event_by_call.get(row.call_id)
            if event is None or event.command != row.command:
                raise ValueError(f"{row.call_id}: command lacks matching raw event")
        phases[task_id] = _full_test_phases(events)
    selected = [row for row in commands if phases[row.task_id].get(row.call_id) == 2]
    return selected, phases


def fit_phase_pmfs(
    task_ids: Sequence[str],
    commands: Sequence[CommandRow],
    events_by_task: Mapping[str, Sequence[PipExecEvent]],
) -> tuple[dict[str, tuple[float, ...]], dict[str, Any]]:
    selected, phases = _phase_rows(task_ids, commands, events_by_task)
    counts = {target: Counter() for target in PHASE_TARGETS}
    support = {target: set() for target in PHASE_TARGETS}
    for row in selected:
        counts["latency"][CANONICAL_LATENCY_BUCKETS.bucket_id(row.duration_ms)] += 1
        support["latency"].add(row.task_id)
        for target in PHASE_TARGETS[1:]:
            label, _source = command_resource_bucket_label(row, target)
            if label is not None:
                counts[target][label] += 1
                support[target].add(row.task_id)
    if any(len(support[target]) < MINIMUM_FIT_TASKS for target in PHASE_TARGETS):
        raise ValueError("third-or-later phase lacks five-task fit support")
    pmfs = {
        target: _empirical_pmf(
            counts[target],
            CANONICAL_LATENCY_BUCKETS.bucket_count if target == "latency" else 3,
        )
        for target in PHASE_TARGETS
    }
    histogram = Counter(
        sum(value == 2 for value in phases[task_id].values()) for task_id in task_ids
    )
    return pmfs, {
        "phase_commands": len(selected),
        "phase_tasks": len({row.task_id for row in selected}),
        "phase_commands_per_task": dict(sorted(histogram.items())),
        "targets": {
            target: {
                "eligible_commands": sum(counts[target].values()),
                "task_support": len(support[target]),
                "label_counts": dict(sorted(counts[target].items())),
                "hard_bucket": _argmax_probabilities(pmfs[target]),
            }
            for target in PHASE_TARGETS
        },
    }


def phase_coverage(
    task_ids: Sequence[str], events_by_task: Mapping[str, Sequence[PipExecEvent]]
) -> dict[str, Any]:
    if list(events_by_task) != list(task_ids):
        raise ValueError("raw event tasks differ from the committed task order")
    phases = {
        task_id: _full_test_phases(events_by_task[task_id]) for task_id in task_ids
    }
    counts = {
        task_id: sum(value == 2 for value in phases[task_id].values())
        for task_id in task_ids
    }
    carrier_tasks = [task_id for task_id, count in counts.items() if count]
    report = {
        "phase_commands": sum(counts.values()),
        "phase_tasks": len(carrier_tasks),
        "phase_task_ids": carrier_tasks,
        "minimum_phase_tasks": MINIMUM_FRESH_TASKS,
    }
    report["passed"] = report["phase_tasks"] >= MINIMUM_FRESH_TASKS
    return report


def apply_phase_candidate(
    baseline_rows: Sequence[Mapping[str, Any]],
    commands: Sequence[CommandRow],
    events_by_task: Mapping[str, Sequence[PipExecEvent]],
    pmfs: Mapping[str, Sequence[float]],
) -> list[dict[str, Any]]:
    task_ids = list(dict.fromkeys(row.task_id for row in commands))
    _selected, phases = _phase_rows(task_ids, commands, events_by_task)
    by_sample = {f"{row.task_id}:{row.call_index}": row for row in commands}
    baseline_samples = [str(row["sample_id"]) for row in baseline_rows]
    if len(by_sample) != len(commands) or len(set(baseline_samples)) != len(
        baseline_rows
    ):
        raise AssertionError("phase candidate contains duplicate command rows")
    if set(by_sample) != set(baseline_samples):
        raise AssertionError("phase candidate differs from baseline command rows")
    rows = []
    for base in baseline_rows:
        sample_id = str(base["sample_id"])
        command = by_sample[sample_id]
        current = dict(base["current_dynamic"])
        current_pmfs = dict(current.pop("probability_by_bucket"))
        for target in CANONICAL_RESOURCE_BUCKET_EDGES:
            current.setdefault(target, None)
            current_pmfs.setdefault(target, None)
        candidate = dict(current)
        candidate_pmfs = dict(current_pmfs)
        applied = []
        phase = phases[command.task_id].get(command.call_id)
        if phase == 2:
            for target in PHASE_TARGETS:
                learned = _argmax_probabilities(
                    tuple(float(value) for value in pmfs[target])
                )
                current_value = current[target]
                current_bucket = (
                    None
                    if current_value is None
                    else int(current_value)
                    if target == "latency"
                    else RESOURCE_BUCKET_LABELS.index(str(current_value))
                )
                if current_bucket is not None and learned > current_bucket:
                    candidate[target] = (
                        learned
                        if target == "latency"
                        else RESOURCE_BUCKET_LABELS[learned]
                    )
                    candidate_pmfs[target] = list(pmfs[target])
                    applied.append(target)
        rows.append(
            {
                **base,
                "call_id": command.call_id,
                "full_test_phase": phase,
                "labels": {
                    "latency": base["latency_label"],
                    **base["resource_labels"],
                },
                "current_dynamic": current,
                "current_probability_by_bucket": current_pmfs,
                "candidate": candidate,
                "candidate_probability_by_bucket": candidate_pmfs,
                "phase_applied_targets": applied,
            }
        )
    return rows


def _validated_pmfs(value: Any) -> dict[str, tuple[float, ...]]:
    if not isinstance(value, dict) or set(value) != set(PHASE_TARGETS):
        raise ValueError("frozen phase PMFs are invalid")
    output = {}
    for target in PHASE_TARGETS:
        row = value[target]
        expected = CANONICAL_LATENCY_BUCKETS.bucket_count if target == "latency" else 3
        if (
            not isinstance(row, list)
            or len(row) != expected
            or any(
                not isinstance(item, (int, float)) or item < 0 or item > 1
                for item in row
            )
            or abs(sum(row) - 1.0) > 1e-9
        ):
            raise ValueError("frozen phase PMF value is invalid")
        output[target] = tuple(float(item) for item in row)
    return output


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
    task_ids, clauses, commands = load_run_rows(args.run_dir)
    if task_ids != split["development"]:
        raise ValueError("development task order differs from the frozen split")
    events = _load_exec_events(args.run_dir, task_ids)
    pmfs, support = fit_phase_pmfs(task_ids, commands, events)
    artifact = {
        "schema": SCHEMA,
        "status": "development_structural_go",
        "claim_bearing": False,
        "feature": FULL_TEST_PHASE_FEATURE_VERSION,
        "phase_definition": "third_or_later_recognized_full_suite",
        "targets": list(PHASE_TARGETS),
        "disk_policy": "bit_identical_current",
        "arbitration": "monotone_upward_override",
        "minimum_fit_tasks": MINIMUM_FIT_TASKS,
        "minimum_fresh_tasks": MINIMUM_FRESH_TASKS,
        "pmfs": {target: list(pmf) for target, pmf in pmfs.items()},
        "fit_support": support,
        "development_run": str(args.run_dir.resolve()),
        "development_task_ids_sha256": hashlib.sha256(
            _json_bytes(task_ids)
        ).hexdigest(),
        "development_fit_sha256": _fit_fingerprint(task_ids, clauses, commands),
        "split_manifest_sha256": split_sha256,
        "preregistration_commit": next(iter(preregistration_commits)),
        "host_identity": _host_identity(),
        "public_inputs": _frozen_public_inputs(split, args.public_telemetry),
        "validation_consumed": False,
        "final_test_consumed": False,
    }
    args.out_dir.mkdir(parents=True)
    _write_artifact(args.out_dir, artifact)


def _load_artifact(
    artifact_dir: Path, split: Mapping[str, Any], split_sha256: str
) -> tuple[dict[str, Any], dict[str, tuple[float, ...]], str, str]:
    artifact_bytes, artifact_commit = _committed_file(artifact_dir / "artifact.json")
    artifact = json.loads(artifact_bytes)
    expected_tasks = hashlib.sha256(_json_bytes(split["development"])).hexdigest()
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("status") != "development_structural_go"
        or artifact.get("claim_bearing") is not False
        or artifact.get("feature") != FULL_TEST_PHASE_FEATURE_VERSION
        or artifact.get("targets") != list(PHASE_TARGETS)
        or artifact.get("disk_policy") != "bit_identical_current"
        or artifact.get("arbitration") != "monotone_upward_override"
        or artifact.get("minimum_fit_tasks") != MINIMUM_FIT_TASKS
        or artifact.get("minimum_fresh_tasks") != MINIMUM_FRESH_TASKS
        or artifact.get("split_manifest_sha256") != split_sha256
        or artifact.get("development_task_ids_sha256") != expected_tasks
    ):
        raise ValueError("frozen full-test phase artifact is incomplete or differs")
    pmfs = _validated_pmfs(artifact.get("pmfs"))
    _validate_host_identity(artifact.get("host_identity"))
    _validate_preregistration_commit(artifact.get("preregistration_commit"))
    return artifact, pmfs, hashlib.sha256(artifact_bytes).hexdigest(), artifact_commit


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


def _result_gate(
    score: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    disk = "disk_read_write_bytes_total"
    disk_identical = all(
        row["candidate"][disk] == row["current_dynamic"][disk]
        and row["candidate_probability_by_bucket"][disk]
        == row["current_probability_by_bucket"][disk]
        for row in rows
    )
    gate = {**score["gate"], "disk_bit_identical": disk_identical}
    gate["go"] = bool(gate["go"] and disk_identical)
    return gate


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
    artifact, pmfs, artifact_sha256, artifact_commit = _load_artifact(
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
    development_fit_sha256 = _fit_fingerprint(
        development_ids, development_clauses, development_commands
    )
    public_inputs = _frozen_public_inputs(split, args.public_telemetry)
    if (
        args.development_run.resolve()
        != Path(str(artifact["development_run"])).resolve()
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
    with tempfile.TemporaryDirectory(prefix="full-test-phase-role-") as directory:
        view = Path(directory) / "results.jsonl"
        _write_result_view(valid_records, view)
        events = _load_exec_events(args.run_dir, valid_ids, results_path=view)
        coverage = phase_coverage(valid_ids, events)
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
    rows = apply_phase_candidate(frozen_rows, role_commands, events, pmfs)
    score = _score_candidate(frozen_baseline, rows)
    gate = _result_gate(score, rows)
    rows = [{**row, "role": args.role} for row in rows]
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
        "feature": FULL_TEST_PHASE_FEATURE_VERSION,
        "baseline": {
            "latency": frozen_baseline["latency"]["current_dynamic"],
            "resources": {
                target: frozen_baseline["resources"][target]["current_dynamic"]
                for target in CANONICAL_RESOURCE_BUCKET_EDGES
            },
        },
        "candidate": score,
        "gate": gate,
        "row_identity": {
            "identical_rows": [row["sample_id"] for row in rows]
            == [row["sample_id"] for row in frozen_rows],
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
