"""SWE-rebench benchmark plugin.

Implements the :class:`~agents.benchmarks.base.Benchmark` protocol for the
``nebius/SWE-rebench`` dataset (https://huggingface.co/datasets/nebius/SWE-rebench).

SWE-rebench is structurally a superset of SWE-Bench Verified but has three
concrete schema quirks that this plugin absorbs so the scaffolds don't need
to know about them:

1. ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` / ``FAIL_TO_FAIL`` / ``PASS_TO_FAIL``
   arrive as **native Python lists**, not JSON-encoded strings as in
   Verified. We keep them as native lists end-to-end — ``derive_test_cmd``
   and ``_count_fail_to_pass`` already handle both shapes, so no conversion
   is needed (this closes open question O2 from the ralplan consensus).
2. Each row carries an explicit ``docker_image`` URI (e.g.
   ``swerebench/sweb.eval.x86_64.django_1776_django-11734``). We pin it to
   ``task['image_name']`` in :meth:`normalize_task` so the harness uses the
   pre-built image rather than deriving one from the repo name.
3. ``meta.is_lite`` marks single-file "lite" tasks. Per CLAUDE.md §1 ("no
   benchmark-specific tuning") we do **not** filter them out by default.
   The ``exclude_lite`` knob on :class:`BenchmarkConfig` is opt-in and
   defaults to ``False``; experimenters who want lite exclusion must flip
   the YAML and document the rationale in the config comment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from agents.benchmarks.base import Benchmark


class SWERebenchBenchmark(Benchmark):
    """Benchmark plugin for ``nebius/SWE-rebench`` (filtered or test split).

    Dataset schema is a superset of SWE-Bench Verified; see the module
    docstring for the three schema quirks this plugin absorbs.
    """

    slug: ClassVar[str] = "swe-rebench"
    task_shape: ClassVar[str] = "swe_patch"

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def load_tasks(self) -> list[dict[str, Any]]:
        """Load all rows from ``nebius/SWE-rebench`` and normalize each.

        Requires the ``datasets`` package (``pip install datasets``).
        """
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset(self.config.harness_dataset, split=self.config.harness_split)
        tasks: list[dict[str, Any]] = []
        for row in ds:
            tasks.append(self.normalize_task(dict(row)))
        return tasks

    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize a raw SWE-rebench row.

        Steps:
        1. Copy the raw dict (avoid mutating the HF cache entry).
        2. Leave ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` / ``FAIL_TO_FAIL`` /
           ``PASS_TO_FAIL`` as native lists. No JSON round-trip.
        3. If the row carries an explicit ``docker_image`` URI, pin it to
           ``task['image_name']`` so :meth:`image_name_for` can return it
           without deriving from the repo.
        4. Derive ``test_cmd`` via the base helper (which handles both list
           and JSON-string shapes).
        """
        task = dict(raw)

        # Quirk 2: pin explicit docker image so the harness uses the pre-built
        # swerebench/sweb.eval.* image instead of deriving one.
        docker_image = raw.get("docker_image")
        if docker_image:
            task["image_name"] = docker_image

        # Derive test_cmd. The base helper delegates to
        # ``agents.swebench_data.derive_test_cmd`` which already handles
        # native lists (see its ``else: list(raw)`` branch). We do NOT
        # convert FAIL_TO_PASS to a JSON string — that would perpetuate a
        # lossy round-trip for no downstream benefit (see open question O2
        # in docs/CURRENT_PLAN.md).
        task["test_cmd"] = self.derive_test_cmd(task)
        return task

    # ------------------------------------------------------------------
    # Override: opt-in ``meta.is_lite`` filter via YAML knob
    # ------------------------------------------------------------------

    def select_subset(
        self,
        tasks: list[dict[str, Any]],
        n: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return a subset of tasks, honoring the ``exclude_lite`` config knob.

        When ``self.config.exclude_lite`` is ``True``, tasks whose
        ``meta.is_lite`` is truthy are dropped from the candidate pool
        **before** stratified selection runs. The default is ``False`` —
        we do not silently exclude "lite" tasks from research runs.
        """
        if self.config.exclude_lite:
            pool = [
                t
                for t in tasks
                if not (t.get("meta") or {}).get("is_lite", False)
            ]
        else:
            pool = list(tasks)
        return super().select_subset(pool, n=n, seed=seed)

    # ------------------------------------------------------------------
    # Override: SWE-rebench images are fully qualified — no namespace prefix
    # ------------------------------------------------------------------

    def build_harness_args(
        self,
        *,
        predictions_path: Path,
        run_id: str,
        max_workers: int = 1,
        timeout: int = 1800,
        report_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Return harness args with ``namespace=None``.

        SWE-rebench ships fully qualified ``swerebench/sweb.eval.*`` image
        URIs, so the harness's ``--namespace`` prefix must not be applied.
        The base implementation already reads ``namespace`` from
        ``self.config.docker_namespace`` (which is ``None`` in
        ``configs/benchmarks/swe-rebench.yaml``), but we force it here too
        as a belt-and-suspenders guard against someone flipping the YAML.
        """
        args = super().build_harness_args(
            predictions_path=predictions_path,
            run_id=run_id,
            max_workers=max_workers,
            timeout=timeout,
            report_dir=report_dir,
        )
        args["namespace"] = None
        return args

    def image_name_for(self, task: dict[str, Any]) -> str | None:
        """Return the ``docker_image`` URI pinned by :meth:`normalize_task`."""
        return task.get("image_name")

    # ------------------------------------------------------------------
    # Override: reuse the SWEBenchRunner for swe_patch tasks
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

        Reuses SWEBenchRunner (same ``swe_patch`` shape). The plugin
        self-injects ``repos_root`` and ``benchmark`` from its own config.
        """
        if scaffold != "openclaw":
            raise NotImplementedError(
                f"SWE-rebench does not support scaffold={scaffold!r}; "
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
