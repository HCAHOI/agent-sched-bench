"""SWE-bench Verified benchmark plugin.

Implements the :class:`~agents.benchmarks.base.Benchmark` protocol for the
``princeton-nlp/SWE-bench_Verified`` dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from agents.benchmarks._swebench_selection import (
    select_tool_intensive as _select_tool_intensive,
)
from agents.benchmarks.base import Benchmark

# Repo-level constants — authoritative copies live here; the legacy shim
# re-exports these for backward compatibility.

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

    # Abstract method implementations

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
        task = dict(raw)
        task["test_cmd"] = self.derive_test_cmd(task)
        return task

    # Override: task selection uses SWE-bench-specific repo quotas

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
        return _select_tool_intensive(
            tasks,
            repo_quotas=CLASS_LEVEL_REPO_QUOTAS,
            n=effective_n,
            seed=effective_seed,
        )

    # Override: build a SWE-bench runner for the given scaffold

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

# Module-level selection function (called by base.Benchmark.select_subset and
# by the legacy shim in agents.swebench_data)
