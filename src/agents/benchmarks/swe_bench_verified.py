"""SWE-bench Verified benchmark plugin.

Implements the :class:`~agents.benchmarks.base.Benchmark` protocol for the
``princeton-nlp/SWE-bench_Verified`` dataset.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, ClassVar

from agents.benchmarks.base import Benchmark


# ---------------------------------------------------------------------------
# Repo-level constants — authoritative copies live here; the legacy shim
# re-exports these for backward compatibility.
# ---------------------------------------------------------------------------

#: Repos known to have heavy test suites (large codebase, slow pytest).
CLASS_LEVEL_HEAVY_REPOS: frozenset[str] = frozenset(
    {
        "django/django",
        "sympy/sympy",
        "scikit-learn/scikit-learn",
        "matplotlib/matplotlib",
        "sphinx-doc/sphinx",
        "pytest-dev/pytest",
    }
)

#: Target allocation per repo for a 32-task selection.
CLASS_LEVEL_REPO_QUOTAS: dict[str, int] = {
    "django/django": 10,
    "sympy/sympy": 6,
    "scikit-learn/scikit-learn": 5,
    "matplotlib/matplotlib": 4,
    # Remaining 7 slots filled from other repos
}


def _count_fail_to_pass(task: dict[str, Any]) -> int:
    """Count the number of tests in FAIL_TO_PASS (module-private helper)."""
    raw = task.get("FAIL_TO_PASS", "[]")
    if isinstance(raw, str):
        try:
            return len(json.loads(raw))
        except json.JSONDecodeError:
            return 1 if raw.strip() else 0
    return len(raw)


class SWEBenchVerified(Benchmark):
    """Benchmark plugin for SWE-bench Verified.

    Dataset: ``princeton-nlp/SWE-bench_Verified`` (HuggingFace).
    Task shape: ``swe_patch`` — the agent must produce a git patch.
    """

    slug: ClassVar[str] = "swe-bench-verified"
    task_shape: ClassVar[str] = "swe_patch"

    # Make the class-level constants accessible as instance-level attributes
    # so tests can reference SWEBenchVerified.CLASS_LEVEL_REPO_QUOTAS etc.
    CLASS_LEVEL_HEAVY_REPOS = CLASS_LEVEL_HEAVY_REPOS
    CLASS_LEVEL_REPO_QUOTAS = CLASS_LEVEL_REPO_QUOTAS

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def load_tasks(self) -> list[dict[str, Any]]:
        """Load all tasks from HuggingFace and normalize each row.

        Requires the ``datasets`` package (``pip install datasets``).
        """
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset(self.config.harness_dataset, split=self.config.harness_split)
        tasks: list[dict[str, Any]] = []
        for row in ds:
            tasks.append(self.normalize_task(dict(row)))
        return tasks

    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize a raw SWE-bench row by deriving ``test_cmd``."""
        task = dict(raw)
        task["test_cmd"] = self.derive_test_cmd(task)
        return task

    # ------------------------------------------------------------------
    # Override: task selection uses SWE-bench-specific repo quotas
    # ------------------------------------------------------------------

    def select_subset(
        self,
        tasks: list[dict[str, Any]],
        n: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Select *n* tool-intensive tasks using repo-quota stratification.

        Selection strategy:
        1. Prioritise large repos whose test suites are naturally slow.
        2. Within each repo, rank by FAIL_TO_PASS test count (more = longer).
        3. Exclude trivial tasks (empty FAIL_TO_PASS or very short statement).
        4. Stratified sampling ensures repo diversity.
        """
        effective_n = n if n is not None else self.config.selection_n
        effective_seed = seed if seed is not None else self.config.selection_seed
        return _select_tool_intensive(tasks, n=effective_n, seed=effective_seed)

    # ------------------------------------------------------------------
    # Override: build a SWE-bench runner for the given scaffold
    # ------------------------------------------------------------------

    def build_runner(
        self,
        *,
        scaffold: str,
        provider: Any,
        workspace_base: Path,
        max_iterations: int,
        context_window_tokens: int,
        model: str,
        **kwargs: Any,
    ) -> Any:
        """Return a :class:`~agents.openclaw.eval.runner.SWEBenchRunner`.

        The plugin self-injects ``repos_root`` and ``benchmark`` from its own
        config so the collector does not need to thread them through.
        """
        if scaffold != "openclaw":
            raise NotImplementedError(
                f"SWE-bench Verified does not support scaffold={scaffold!r}; "
                f"use scaffold='openclaw'."
            )
        from agents.openclaw.eval.runner import SWEBenchRunner

        return SWEBenchRunner(
            provider=provider,
            workspace_base=workspace_base,
            max_iterations=max_iterations,
            context_window_tokens=context_window_tokens,
            model=model,
            repos_root=self.config.repos_root,
            benchmark=self,
            **kwargs,
        )

# ---------------------------------------------------------------------------
# Module-level selection function (called by base.Benchmark.select_subset and
# by the legacy shim in agents.swebench_data)
# ---------------------------------------------------------------------------


def _select_tool_intensive(
    tasks: list[dict[str, Any]],
    n: int = 32,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Core stratified selection logic; shared by plugin and legacy shim."""
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
    for repo, quota in CLASS_LEVEL_REPO_QUOTAS.items():
        pool = by_repo.get(repo, [])
        take = min(quota, len(pool))
        selected.extend(pool[:take])

    # Phase 2: Fill remaining slots from other repos
    remaining = n - len(selected)
    if remaining > 0:
        other_repos = [r for r in by_repo if r not in CLASS_LEVEL_REPO_QUOTAS]
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
