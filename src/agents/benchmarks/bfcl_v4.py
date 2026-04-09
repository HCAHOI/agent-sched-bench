"""BFCL v4 (Berkeley Function-Calling Leaderboard v4) benchmark plugin.

``task_shape='function_call'``: tasks carry a JSON-Schema tool spec per
instance; scoring is pure-Python AST comparison (no Docker). See
``docs/benchmark_plugin_spec.md §10`` and ``configs/benchmarks/bfcl-v4.yaml``
for scope, data layout, and the v1 category coverage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from agents.benchmarks.base import Benchmark

logger = logging.getLogger(__name__)


class BFCLv4Benchmark(Benchmark):
    """Benchmark plugin for BFCL v4 (single-turn categories only in v1)."""

    slug: ClassVar[str] = "bfcl-v4"
    task_shape: ClassVar[str] = "function_call"

    #: Single-turn categories supported in v1. Names mirror the canonical
    #: BFCL v4 file stems (e.g. ``BFCL_v4_simple_python.json`` → ``simple_python``).
    _SUPPORTED_CATEGORIES: ClassVar[frozenset[str]] = frozenset(
        {
            "simple_python",
            "simple_java",
            "simple_javascript",
            "multiple",
            "parallel",
            "parallel_multiple",
            "live_simple",
            "live_multiple",
            "live_parallel",
            "live_parallel_multiple",
            "irrelevance",
            "live_relevance",
            "live_irrelevance",
        }
    )

    #: Categories deferred to v2 — require a stateful tool simulator.
    #: Filtered loudly in :meth:`load_tasks` with a WARN log.
    _DEFERRED_CATEGORIES: ClassVar[frozenset[str]] = frozenset(
        {
            "multi_turn_base",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
            "multi_turn_long_context",
            "memory",
            "web_search",
            "format_sensitivity",
        }
    )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def load_tasks(self) -> list[dict[str, Any]]:
        """Load BFCL v4 tasks from ``<data_root>/tasks.json`` (merged JSONL).

        Deferred-category rows are dropped with a WARN summary. Unknown-category
        rows are also dropped loudly — silently keeping them biases results
        (CLAUDE.md §5 completeness).
        """
        tasks_path = self.config.data_root / "tasks.json"
        if not tasks_path.exists():
            raise FileNotFoundError(
                f"BFCL v4 tasks.json not found at {tasks_path}; run "
                f"`make download-bfcl-v4` to populate data/bfcl-v4/."
            )

        supported: list[dict[str, Any]] = []
        deferred_counts: dict[str, int] = {}
        unknown_counts: dict[str, int] = {}

        with tasks_path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed JSON at %s:%d: %s",
                        tasks_path,
                        lineno,
                        exc,
                    )
                    continue

                category = row.get("category", "")
                if category in self._DEFERRED_CATEGORIES:
                    deferred_counts[category] = deferred_counts.get(category, 0) + 1
                    continue
                if category not in self._SUPPORTED_CATEGORIES:
                    unknown_counts[category] = unknown_counts.get(category, 0) + 1
                    continue
                supported.append(self.normalize_task(row))

        if deferred_counts:
            summary = ", ".join(
                f"{cat}={count}" for cat, count in sorted(deferred_counts.items())
            )
            logger.warning(
                "BFCL v4: dropped %d rows from deferred categories (%s). "
                "Multi-turn / memory / web-search / format-sensitivity "
                "categories require a stateful tool simulator (v2 work).",
                sum(deferred_counts.values()),
                summary,
            )
        if unknown_counts:
            unk_summary = ", ".join(
                f"{cat}={count}" for cat, count in sorted(unknown_counts.items())
            )
            logger.warning(
                "BFCL v4: dropped %d rows from UNKNOWN categories (%s). "
                "These categories appear in the dataset but are not "
                "classified as either supported or deferred in "
                "BFCLv4Benchmark. Classify them explicitly before "
                "running real experiments — silently keeping unknown "
                "rows is a research-integrity anti-pattern.",
                sum(unknown_counts.values()),
                unk_summary,
            )
        return supported

    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Map a BFCL row into the canonical task dict used by the runner.

        BFCL rows carry ``id`` (not ``instance_id``), ``function`` (not
        ``tools``), and a nested ``question`` turn list. The first user
        message is hoisted to ``problem_statement`` for logging; the full
        turn list is retained on the EvalTask for the runner.

        Intentionally leaves ``repo`` / ``base_commit`` unset so
        :attr:`EvalTask.needs_prepare` returns False and BFCL tasks skip
        the git-clone phase.
        """
        question = raw.get("question", [])
        # BFCL question is list[list[message]]; first turn, first message.
        first_user_content = ""
        if question and isinstance(question, list):
            first_turn = question[0] if isinstance(question[0], list) else []
            for msg in first_turn:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    first_user_content = str(msg.get("content", ""))
                    break

        # Prefer an already-normalized ``instance_id`` when present so
        # this function is idempotent (collector.collect_traces hoists
        # normalize_task before dispatch, and _collect_openclaw's
        # EvalTask.from_benchmark_instance calls it a second time).
        # When the raw row only has ``id`` (first-pass from setup
        # script), fall back to that.
        instance_id = raw.get("instance_id") or raw.get("id", "")
        return {
            "instance_id": str(instance_id),
            "problem_statement": first_user_content,
            "category": raw.get("category"),
            "tools": list(raw.get("function", raw.get("tools", []))),
            "question": list(question),
            "ground_truth": list(raw.get("ground_truth", [])),
        }

    # ------------------------------------------------------------------
    # Override: task selection is category-stratified
    # ------------------------------------------------------------------

    def select_subset(
        self,
        tasks: list[dict[str, Any]],
        n: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return a stratified sample across supported categories.

        Proportional allocation with leftover quota going to the largest
        categories. Deterministic order: sorted by ``(category, instance_id)``.
        ``seed`` is accepted for API compatibility but unused — proportional
        allocation with a deterministic tie-breaker needs no RNG.
        """
        del seed  # unused — deterministic by construction
        effective_n = n if n is not None else self.config.selection_n
        if effective_n >= len(tasks):
            return sorted(
                tasks,
                key=lambda t: (t.get("category", ""), t.get("instance_id", "")),
            )

        # Group by category.
        by_category: dict[str, list[dict[str, Any]]] = {}
        for t in tasks:
            cat = t.get("category", "")
            by_category.setdefault(cat, []).append(t)

        # Sort within each category for determinism.
        for cat_tasks in by_category.values():
            cat_tasks.sort(key=lambda t: t.get("instance_id", ""))

        # Proportional allocation.
        total = len(tasks)
        selected: list[dict[str, Any]] = []
        category_order = sorted(by_category.keys())
        quotas: dict[str, int] = {}
        for cat in category_order:
            pool = by_category[cat]
            share = round(effective_n * len(pool) / total)
            quotas[cat] = min(share, len(pool))

        # Distribute leftover slots to the largest under-quota categories.
        allocated = sum(quotas.values())
        leftover = effective_n - allocated
        if leftover > 0:
            largest = sorted(
                category_order,
                key=lambda c: len(by_category[c]) - quotas[c],
                reverse=True,
            )
            idx = 0
            while leftover > 0 and idx < len(largest):
                cat = largest[idx]
                if quotas[cat] < len(by_category[cat]):
                    quotas[cat] += 1
                    leftover -= 1
                idx = (idx + 1) % len(largest)
                if idx == 0 and all(
                    quotas[c] >= len(by_category[c]) for c in category_order
                ):
                    break

        for cat in category_order:
            selected.extend(by_category[cat][: quotas[cat]])
        selected.sort(key=lambda t: (t.get("category", ""), t.get("instance_id", "")))
        return selected

    # ------------------------------------------------------------------
    # Override: no pytest command, no SWE-bench harness
    # ------------------------------------------------------------------

    def derive_test_cmd(self, task: dict[str, Any]) -> str:
        raise NotImplementedError(
            "BFCL v4 has no test command; scoring is AST comparison against "
            "ground_truth performed in-process by BFCLRunner."
        )

    def image_name_for(self, task: dict[str, Any]) -> str | None:
        """BFCL v4 has no Docker image — tasks run in-process."""
        return None

    # ------------------------------------------------------------------
    # Override: scaffold refusal + runner construction
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
        """Return a :class:`BFCLRunner` for the openclaw scaffold.

        The ``miniswe`` scaffold is refused loudly: miniswe is a
        bash-in-repo scaffold and cannot emit structured function calls
        against a JSON-Schema tool spec.
        """
        if scaffold == "miniswe":
            raise NotImplementedError(
                "BFCL v4 (task_shape='function_call') requires scaffold='openclaw'; "
                "miniswe is bash-in-repo and cannot emit structured function calls."
            )
        if scaffold != "openclaw":
            raise NotImplementedError(
                f"BFCL v4 does not support scaffold={scaffold!r}; use scaffold='openclaw'."
            )

        # Lazy import to avoid circular dependency with the runner module.
        from agents.benchmarks.bfcl_runner import BFCLRunner

        return BFCLRunner(
            provider=provider,
            workspace_base=workspace_base,
            max_iterations=max_iterations,
            context_window_tokens=context_window_tokens,
            model=model,
            benchmark=self,
            **kwargs,
        )
