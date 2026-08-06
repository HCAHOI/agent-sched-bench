#!/usr/bin/env python3
"""Factor KV oracle gains into request-rank and dead-suffix effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from tool_resource_eval.cachewise_kv_factorial import (  # noqa: E402
    BLOCK_SIZE_TOKENS,
    CAPACITY_BLOCKS,
    CAPACITY_TOKENS,
    LOAD,
    SEEDS,
    Session,
    _bootstrap,
    _program,
    request_ranks,
    simulate,
)


VERSION = "kv-oracle-mechanism-decomposition-v1"
MINIMUM_C100_SHARE = 0.01
MINIMUM_BELADY_GAP_SHARE = 0.20
RESULTS = _ROOT / "analysis/results/tool-resource-5-3-3-3-20260804"
BASELINE = RESULTS / "sqlglot50-kv-prediction-actionability-v1/result.json"
BELADY = RESULTS / "sqlglot50-kv-block-belady-oracle-v1/result.json"
SPLIT = _ROOT / "analysis/development/sqlglot-relational-task-split.json"
VALIDATION_RUN = (
    _ROOT
    / "traces/swe-rebench/gpt-5.6-sol"
    / "sqlglot-prev100-c2-fast-requested-ebpf-20260804"
)
ARMS = ("c100", "arrival_no_suffix", "rank_no_suffix", "arrival_suffix", "belady")


def _trace_path(task_id: str) -> Path:
    path = VALIDATION_RUN / task_id / "attempt_1/trace.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run() -> dict[str, Any]:
    baseline = json.loads(BASELINE.read_text())
    belady_result = json.loads(BELADY.read_text())
    validation_ids = list(json.loads(SPLIT.read_text())["validation"])
    baseline_rows = baseline.get("schedule_results", [])
    belady_rows = belady_result.get("schedule_results", [])
    baseline_protocol = baseline.get("protocol", {})
    belady_protocol = belady_result.get("protocol", {})
    if (
        baseline.get("schema") != "kv-prediction-actionability-v1"
        or baseline.get("status") != "development_no_go"
        or belady_result.get("schema") != "kv-block-belady-oracle-v1"
        or belady_result.get("status") != "development_no_go_close_kv_eviction"
        or baseline_protocol.get("scheduler") != "fcfs"
        or belady_protocol.get("scheduler") != "fcfs"
        or baseline_protocol.get("load") != LOAD
        or belady_protocol.get("load") != LOAD
        or baseline_protocol.get("capacity_tokens") != CAPACITY_TOKENS
        or belady_protocol.get("capacity_tokens") != CAPACITY_TOKENS
        or baseline_protocol.get("capacity_blocks") != CAPACITY_BLOCKS
        or belady_protocol.get("capacity_blocks") != CAPACITY_BLOCKS
        or baseline_protocol.get("block_size_tokens") != BLOCK_SIZE_TOKENS
        or belady_protocol.get("block_size_tokens") != BLOCK_SIZE_TOKENS
        or baseline_protocol.get("seeds") != list(SEEDS)
        or belady_protocol.get("seeds") != list(SEEDS)
        or [row.get("seed") for row in baseline_rows] != list(SEEDS)
        or [row.get("seed") for row in belady_rows] != list(SEEDS)
        or len(validation_ids) != 50
    ):
        raise ValueError("committed oracle corners differ from the frozen protocol")

    programs = {task_id: _program(_trace_path(task_id)) for task_id in validation_ids}
    rows = []
    for baseline_row, belady_row in zip(baseline_rows, belady_rows, strict=True):
        task_ids = list(baseline_row["task_ids"])
        if (
            task_ids != belady_row["task_ids"]
            or len(task_ids) != LOAD
            or not set(task_ids) <= set(validation_ids)
            or baseline_row["arms"]["c100"] != belady_row["arms"]["c100"]
            or baseline_row["arms"]["next_reuse_oracle"]
            != belady_row["arms"]["next_reuse_oracle"]
        ):
            raise ValueError("committed oracle corners use different schedules")
        selected = [programs[task_id] for task_id in task_ids]
        ranks = request_ranks(selected)

        def rank(session: Session, _now_s: float) -> float:
            return float(ranks[(session.program.task_id, session.turn_index)])

        common = {
            "programs": selected,
            "scheduler": "fcfs",
            "global_history": np.asarray([1.0]),
            "tool_history": {},
            "clusters": {100: {}},
            "label_cache": {},
        }
        rank_no_suffix = simulate(
            **common,
            eviction="predicted",
            remaining_predictor=rank,
        )
        arrival_suffix = simulate(**common, eviction="suffix_arrival")
        arms = {
            "c100": baseline_row["arms"]["c100"],
            "arrival_no_suffix": baseline_row["arms"]["next_reuse_oracle"],
            "rank_no_suffix": rank_no_suffix,
            "arrival_suffix": arrival_suffix,
            "belady": belady_row["arms"]["belady"],
        }
        if len({arm["request_count"] for arm in arms.values()}) != 1:
            raise ValueError("oracle decomposition evaluated different requests")
        rows.append({"seed": baseline_row["seed"], "task_ids": task_ids, "arms": arms})

    metrics = ("recomputed_prefix_blocks", "evicted_blocks", "eviction_events")
    means = {
        arm: {
            metric: float(np.mean([row["arms"][arm][metric] for row in rows]))
            for metric in metrics
        }
        for arm in ARMS
    }
    primary = "recomputed_prefix_blocks"
    per_seed_rank = [
        0.5
        * (
            row["arms"]["arrival_no_suffix"][primary]
            - row["arms"]["rank_no_suffix"][primary]
            + row["arms"]["arrival_suffix"][primary]
            - row["arms"]["belady"][primary]
        )
        for row in rows
    ]
    per_seed_suffix = [
        0.5
        * (
            row["arms"]["arrival_no_suffix"][primary]
            - row["arms"]["arrival_suffix"][primary]
            + row["arms"]["rank_no_suffix"][primary]
            - row["arms"]["belady"][primary]
        )
        for row in rows
    ]
    c100 = means["c100"][primary]
    total_gap = c100 - means["belady"][primary]

    def effect(name: str, values: list[float]) -> dict[str, Any]:
        mean = float(np.mean(values))
        return {
            "factor": name,
            "mean_blocks_saved": mean,
            "share_of_c100": mean / c100,
            "share_of_c100_to_belady_gap": mean / total_gap,
            "positive": mean > 0.0,
            "passes_minimums": mean > 0.0
            and mean / c100 >= MINIMUM_C100_SHARE
            and mean / total_gap >= MINIMUM_BELADY_GAP_SHARE,
            "paired_seed_bootstrap": _bootstrap(values),
        }

    effects = {
        "request_rank": effect("request_rank", per_seed_rank),
        "dead_suffix": effect("dead_suffix", per_seed_suffix),
    }
    passing = [value for value in effects.values() if value["passes_minimums"]]
    selected = (
        max(passing, key=lambda value: value["mean_blocks_saved"])["factor"]
        if passing
        else None
    )
    return {
        "schema": VERSION,
        "status": "development_go_to_one_causal_proxy" if selected else "development_no_go_close_kv_mechanism",
        "claim_bearing": False,
        "protocol": {
            "factors": {
                "next_use": ["arrival", "request_rank"],
                "dead_suffix": [False, True],
            },
            "arms": list(ARMS),
            "primary": "mean recomputed_prefix_blocks; lower is better",
            "minimum_factor_share_of_c100": MINIMUM_C100_SHARE,
            "minimum_factor_share_of_belady_gap": MINIMUM_BELADY_GAP_SHARE,
            "selection": "larger passing main effect",
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
        "factor_effects": effects,
        "interaction_blocks": (
            means["arrival_no_suffix"][primary]
            - means["rank_no_suffix"][primary]
            - means["arrival_suffix"][primary]
            + means["belady"][primary]
        ),
        "gate": {
            "authorized_factor": selected,
            "go": selected is not None,
        },
        "schedule_results": rows,
        "limitations": [
            "Every non-C100 arm uses hindsight unavailable to an online policy.",
            "The decomposition is development-exposed and cannot amend prior gates.",
            "Main effects need not sum to the total C100-to-Belady gap because C100 is not the no-factor oracle corner.",
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
        "belady_result": str(BELADY.resolve()),
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
