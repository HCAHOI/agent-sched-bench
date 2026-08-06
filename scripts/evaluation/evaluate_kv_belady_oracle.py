#!/usr/bin/env python3
"""Evaluate a block-aware Belady upper bound on the frozen KV schedules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from tool_resource_eval.cachewise_kv_factorial import (  # noqa: E402
    BLOCK_SIZE_TOKENS,
    CAPACITY_BLOCKS,
    CAPACITY_TOKENS,
    LOAD,
    SEEDS,
    _bootstrap,
    _program,
    request_ranks,
    simulate,
)


VERSION = "kv-block-belady-oracle-v1"
MINIMUM_REDUCTION = 0.10
RESULTS = _ROOT / "analysis/results/tool-resource-5-3-3-3-20260804"
BASELINE = RESULTS / "sqlglot50-kv-prediction-actionability-v1/result.json"
SPLIT = _ROOT / "analysis/development/sqlglot-relational-task-split.json"
VALIDATION_RUN = (
    _ROOT
    / "traces/swe-rebench/gpt-5.6-sol"
    / "sqlglot-prev100-c2-fast-requested-ebpf-20260804"
)


def _trace_path(task_id: str) -> Path:
    path = VALIDATION_RUN / task_id / "attempt_1/trace.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run() -> dict[str, Any]:
    baseline = json.loads(BASELINE.read_text())
    split = json.loads(SPLIT.read_text())
    validation_ids = list(split["validation"])
    protocol = baseline.get("protocol", {})
    schedule_rows = baseline.get("schedule_results", [])
    if (
        baseline.get("schema") != "kv-prediction-actionability-v1"
        or baseline.get("status") != "development_no_go"
        or protocol.get("scheduler") != "fcfs"
        or protocol.get("load") != LOAD
        or protocol.get("seeds") != list(SEEDS)
        or protocol.get("capacity_tokens") != CAPACITY_TOKENS
        or protocol.get("capacity_blocks") != CAPACITY_BLOCKS
        or protocol.get("block_size_tokens") != BLOCK_SIZE_TOKENS
        or [row.get("seed") for row in schedule_rows] != list(SEEDS)
        or len(validation_ids) != 50
    ):
        raise ValueError("committed KV baseline differs from the frozen protocol")

    programs = {task_id: _program(_trace_path(task_id)) for task_id in validation_ids}
    rows = []
    for baseline_row in schedule_rows:
        task_ids = list(baseline_row["task_ids"])
        if len(task_ids) != LOAD or not set(task_ids) <= set(validation_ids):
            raise ValueError("baseline schedule contains an invalid task selection")
        selected = [programs[task_id] for task_id in task_ids]
        ranks = request_ranks(selected)
        belady = simulate(
            selected,
            scheduler="fcfs",
            eviction="belady",
            global_history=np.asarray([1.0]),
            tool_history={},
            clusters={100: {}},
            label_cache={},
            next_request_rank=ranks,
        )
        c100 = baseline_row["arms"]["c100"]
        greedy = baseline_row["arms"]["next_reuse_oracle"]
        if belady["request_count"] != c100["request_count"]:
            raise ValueError("Belady and C100 evaluated different request rows")
        rows.append(
            {
                "seed": baseline_row["seed"],
                "task_ids": task_ids,
                "arms": {"c100": c100, "next_reuse_oracle": greedy, "belady": belady},
            }
        )

    metrics = ("recomputed_prefix_blocks", "evicted_blocks", "eviction_events")
    means = {
        arm: {
            metric: float(np.mean([row["arms"][arm][metric] for row in rows]))
            for metric in metrics
        }
        for arm in ("c100", "next_reuse_oracle", "belady")
    }
    c100_primary = means["c100"]["recomputed_prefix_blocks"]
    deltas = [
        float(row["arms"]["belady"]["recomputed_prefix_blocks"])
        - float(row["arms"]["c100"]["recomputed_prefix_blocks"])
        for row in rows
    ]
    reduction = (c100_primary - means["belady"]["recomputed_prefix_blocks"]) / c100_primary
    comparison = {
        "candidate": "belady",
        "baseline": "c100",
        "metric": "recomputed_prefix_blocks; lower is better",
        "relative_reduction_of_means": reduction,
        **_bootstrap(deltas),
    }
    gate = {
        "identical_schedules_and_requests": all(
            row["arms"]["belady"]["request_count"]
            == row["arms"]["c100"]["request_count"]
            for row in rows
        ),
        "reduction_at_least_10_percent": reduction >= MINIMUM_REDUCTION,
        "paired_ci_below_zero": comparison["ci95_paired_seed_bootstrap"][1] < 0.0,
        "evicted_blocks_no_worse_than_c100": means["belady"]["evicted_blocks"]
        <= means["c100"]["evicted_blocks"],
    }
    gate["go"] = all(gate.values())
    return {
        "schema": VERSION,
        "status": "development_go_to_causal_policy_design" if gate["go"] else "development_no_go_close_kv_eviction",
        "claim_bearing": False,
        "protocol": {
            "baseline": "committed kv-prediction-actionability-v1 schedule rows",
            "scheduler": "fcfs",
            "load": LOAD,
            "seeds": list(SEEDS),
            "capacity_tokens": CAPACITY_TOKENS,
            "capacity_blocks": CAPACITY_BLOCKS,
            "block_size_tokens": BLOCK_SIZE_TOKENS,
            "primary": "mean recomputed_prefix_blocks; lower is better",
            "guardrail": "mean evicted_blocks no worse than c100",
            "minimum_relative_reduction": MINIMUM_REDUCTION,
        },
        "coverage": {
            "schedules": len(rows),
            "tasks_per_schedule": LOAD,
            "validation_task_pool": len(programs),
            "mean_requests_per_schedule": float(
                np.mean([row["arms"]["belady"]["request_count"] for row in rows])
            ),
        },
        "mean_metrics": means,
        "primary_comparison": comparison,
        "gate": gate,
        "schedule_results": rows,
        "limitations": [
            "The workload and simulator are development-exposed.",
            "Belady uses future request order and next-turn reusable prefix length.",
            "The result is an upper bound only within the fixed serial simulator.",
            "Cache misses do not feed back into service time.",
        ],
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    result = run()
    result["inputs"] = {
        "baseline_result": str(BASELINE.resolve()),
        "validation_run": str(VALIDATION_RUN.resolve()),
        "split": str(SPLIT.resolve()),
        "git_sha": _git_sha(),
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
