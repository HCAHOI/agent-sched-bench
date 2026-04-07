"""Compatibility shim for SWE-bench data utilities.

This module preserves the legacy public API so that all existing call sites
continue to work without modification.  **New code should use the benchmark
plugin directly:**

    from agents.benchmarks import get_benchmark_class
    cls = get_benchmark_class("swe-bench-verified")
    plugin = cls(config)
    tasks = plugin.load_tasks()

All implementations live in :mod:`agents.benchmarks.swe_bench_verified`.
This shim re-exports constants and provides thin wrappers around the
plugin's methods using a module-level default :class:`BenchmarkConfig`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from agents.benchmarks.base import BenchmarkConfig
from agents.benchmarks.swe_bench_verified import (
    SWEBenchVerified,
    CLASS_LEVEL_HEAVY_REPOS,
    CLASS_LEVEL_REPO_QUOTAS,
)

# ---------------------------------------------------------------------------
# Re-export constants at module level for backward compatibility.
# Tests and scripts import these directly from agents.swebench_data.
# ---------------------------------------------------------------------------

#: Repos known to have heavy test suites — re-exported from the plugin class.
HEAVY_REPOS: frozenset[str] = CLASS_LEVEL_HEAVY_REPOS

#: Target allocation per repo for a 32-task selection — re-exported from the plugin class.
REPO_QUOTAS: dict[str, int] = CLASS_LEVEL_REPO_QUOTAS

# ---------------------------------------------------------------------------
# Module-level default config used by the legacy wrapper functions below.
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = BenchmarkConfig(
    slug="swe-bench-verified",
    display_name="SWE-bench Verified",
    harness_dataset="princeton-nlp/SWE-bench_Verified",
    harness_split="test",
    data_root=Path("data/swebench_verified"),
    repos_root=None,
    trace_root=Path("traces/swebench_verified"),
    default_max_steps=80,
    selection_n=32,
    selection_seed=42,
    docker_namespace="swebench",
)


# ---------------------------------------------------------------------------
# Legacy public API — implementations kept inline for clarity and to avoid
# circular imports.  The plugin's select_subset() delegates back to
# select_tool_intensive_tasks(), so the body must live here.
# ---------------------------------------------------------------------------


def derive_test_cmd(task: dict[str, Any]) -> str:
    """Derive a pytest command from the FAIL_TO_PASS field.

    FAIL_TO_PASS may be a JSON-encoded string or a native Python list.

    We construct a pytest invocation that runs exactly those tests.
    """
    fail_to_pass_raw = task.get("FAIL_TO_PASS", "[]")
    if isinstance(fail_to_pass_raw, str):
        try:
            test_ids = json.loads(fail_to_pass_raw)
        except json.JSONDecodeError:
            test_ids = [fail_to_pass_raw]
    else:
        test_ids = list(fail_to_pass_raw)

    if not test_ids:
        return "python -m pytest --no-header -q"

    tests_str = " ".join(test_ids)
    return f"python -m pytest {tests_str} -x --no-header -q"


def _count_fail_to_pass(task: dict[str, Any]) -> int:
    """Count the number of tests in FAIL_TO_PASS."""
    raw = task.get("FAIL_TO_PASS", "[]")
    if isinstance(raw, str):
        try:
            return len(json.loads(raw))
        except json.JSONDecodeError:
            return 1 if raw.strip() else 0
    return len(raw)


def select_tool_intensive_tasks(
    tasks: list[dict[str, Any]],
    n: int = 32,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select *n* tool-intensive tasks from the full Verified dataset.

    Selection strategy:
    1. Prioritize large repos whose test suites are naturally slow
       (django, sympy, scikit-learn, matplotlib).
    2. Within each repo, rank by FAIL_TO_PASS test count (more tests ≈
       longer pytest runtime).
    3. Exclude trivial tasks (empty FAIL_TO_PASS or very short
       problem_statement suggesting doc/typo fixes).
    4. Stratified sampling ensures repo diversity.

    Args:
        tasks: Full list of SWE-bench Verified task dicts.
        n: Number of tasks to select.
        seed: Random seed for reproducibility.

    Returns:
        Selected subset of *n* tasks sorted by instance_id.
    """
    rng = random.Random(seed)

    # Filter out trivial tasks
    candidates = [
        t
        for t in tasks
        if _count_fail_to_pass(t) > 0 and len(t.get("problem_statement", "")) > 100
    ]

    # Group by repo
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for t in candidates:
        repo = t["repo"]
        by_repo.setdefault(repo, []).append(t)

    # Sort each repo's tasks by FAIL_TO_PASS count (descending)
    for repo_tasks in by_repo.values():
        repo_tasks.sort(key=_count_fail_to_pass, reverse=True)

    selected: list[dict[str, Any]] = []

    # Phase 1: Fill quotas from priority repos
    for repo, quota in REPO_QUOTAS.items():
        pool = by_repo.get(repo, [])
        take = min(quota, len(pool))
        selected.extend(pool[:take])

    # Phase 2: Fill remaining slots from other repos
    remaining = n - len(selected)
    if remaining > 0:
        other_repos = [r for r in by_repo if r not in REPO_QUOTAS]
        rng.shuffle(other_repos)
        other_pool: list[dict[str, Any]] = []
        for repo in other_repos:
            other_pool.extend(by_repo[repo])
        other_pool.sort(key=_count_fail_to_pass, reverse=True)
        selected.extend(other_pool[:remaining])

    # Trim to exactly n if we over-selected
    selected = selected[:n]

    # Stable output order
    selected.sort(key=lambda t: t["instance_id"])
    return selected


def load_swebench_verified() -> list[dict[str, Any]]:
    """Load the SWE-bench Verified dataset from HuggingFace.

    Returns a list of task dicts with official SWE-bench fields plus
    derived fields (test_cmd) for CodeAgent compatibility.

    Requires the ``datasets`` package::

        pip install datasets
    """
    plugin = SWEBenchVerified(_DEFAULT_CONFIG)
    return plugin.load_tasks()


def download_and_save(
    output_dir: str = "data/swebench_verified",
    n: int = 32,
    seed: int = 42,
) -> Path:
    """Download SWE-bench Verified, select tasks, and save to JSON.

    Returns the path to the saved tasks file.
    """
    import json as _json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tasks_file = output_path / "tasks.json"

    all_tasks = load_swebench_verified()
    selected = select_tool_intensive_tasks(all_tasks, n=n, seed=seed)

    tasks_file.write_text(
        _json.dumps(selected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return tasks_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and select SWE-bench tasks")
    parser.add_argument("--n", type=int, default=32, help="Number of tasks to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="data/swebench_verified", help="Output dir")
    args = parser.parse_args()

    path = download_and_save(output_dir=args.output, n=args.n, seed=args.seed)
    print(f"Saved {args.n} tasks to {path}")
