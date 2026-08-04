#!/usr/bin/env python3
"""Score command predictions after a causal same-repository warm-up."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    evaluate_interaction_commands,
    evaluate_poset_resources,
    evaluate_prequential_commands,
)
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    load_rows,
    load_run_rows,
)
from tool_resource_eval.labels import repo_of  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--public-telemetry", type=Path, action="append", required=True)
    parser.add_argument("--exclude-public-repo", action="append", default=[])
    parser.add_argument("--warmup-tasks", type=int, default=80)
    parser.add_argument(
        "--interaction-architectures",
        action="store_true",
        help="score the full causal stream with the two frozen non-trie candidates",
    )
    parser.add_argument(
        "--poset-resources-after-latency",
        type=Path,
        help="score resources only when this latency result contains a poset GO",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dump-rows", type=Path, required=True)
    args = parser.parse_args()

    task_ids, clauses, commands = load_run_rows(args.run_dir)
    excluded = {repo_of(task_id) for task_id in task_ids} | set(
        args.exclude_public_repo
    )
    public = [row for path in args.public_telemetry for row in load_rows(path)]
    raw_public_count = len(public)
    public = [row for row in public if row.repo not in excluded]
    overlap = {row.task_id for row in public} & set(task_ids)
    if not public or overlap:
        raise ValueError(
            "public evidence is empty or overlaps target tasks: "
            f"{sorted(overlap)[:3]}"
        )
    provenance = {
        "target_run_dir": str(args.run_dir.resolve()),
        "public_telemetry": [str(path.resolve()) for path in args.public_telemetry],
        "public_excluded_repositories": sorted(excluded),
        "public_clause_observations_before_repo_filter": raw_public_count,
        "public_clause_observations_after_repo_filter": len(public),
        "public_structure_unknown": sum(not row.structure_known for row in public),
        "public_online_eligible_clause_observations": sum(
            row.structure_known and row.pipeline_position <= 0 for row in public
        ),
        "public_structure_recovery": (
            "current parser validated against static_word_intent; earliest/latest "
            "ordered alignment must identify one static clause"
        ),
        "target_task_order_source": "results.jsonl successful final attempts",
    }
    if args.interaction_architectures and args.poset_resources_after_latency:
        raise ValueError("latency and gated resource modes are mutually exclusive")
    if args.poset_resources_after_latency:
        gate_path = args.poset_resources_after_latency.resolve()
        result, rows = evaluate_poset_resources(
            public,
            task_ids,
            clauses,
            commands,
            {**provenance, "latency_gate_result": str(gate_path)},
            json.loads(gate_path.read_text(encoding="utf-8")),
        )
    elif args.interaction_architectures:
        result, rows = evaluate_interaction_commands(
            public,
            task_ids,
            clauses,
            commands,
            provenance,
        )
    else:
        result, rows = evaluate_prequential_commands(
            public,
            task_ids,
            clauses,
            commands,
            provenance,
            warmup_task_count=args.warmup_tasks,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.dump_rows.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.dump_rows.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


if __name__ == "__main__":
    main()
