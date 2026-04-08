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
from pathlib import Path
from typing import Any

from agents.benchmarks.base import BenchmarkConfig
from agents.benchmarks._swebench_selection import (
    count_fail_to_pass as _shared_count_fail_to_pass,
    select_tool_intensive as _shared_select_tool_intensive,
)
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
# Legacy public API — keep these names stable, but delegate shared selection
# logic to the benchmark plugin so the shim does not carry a second copy.
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
    return _shared_count_fail_to_pass(task)


def select_tool_intensive_tasks(
    tasks: list[dict[str, Any]],
    n: int = 32,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select *n* tool-intensive tasks from the full Verified dataset."""
    return _shared_select_tool_intensive(tasks, repo_quotas=REPO_QUOTAS, n=n, seed=seed)


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
