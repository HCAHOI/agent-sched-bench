"""Base classes and configuration for the benchmark plugin architecture.

New code should instantiate benchmarks via::

    from agents.benchmarks import get_benchmark_class
    cls = get_benchmark_class("swe-bench-verified")
    plugin = cls(config)
"""

from __future__ import annotations

import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark plugin.

    All path fields are stored as :class:`pathlib.Path` objects.
    """

    slug: str
    display_name: str
    harness_dataset: str
    harness_split: str
    data_root: Path
    repos_root: Path | None
    trace_root: Path
    default_max_steps: int
    selection_n: int
    selection_seed: int
    docker_namespace: str | None
    exclude_lite: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkConfig":
        """Load a :class:`BenchmarkConfig` from a YAML file.

        Path fields (``data_root``, ``repos_root``, ``trace_root``) are
        wrapped in :class:`pathlib.Path`.  ``repos_root`` is ``None`` when
        absent or explicitly set to ``null`` in the YAML.
        """
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

        repos_root_raw = raw.get("repos_root")
        repos_root: Path | None = Path(repos_root_raw) if repos_root_raw is not None else None

        return cls(
            slug=raw["slug"],
            display_name=raw["display_name"],
            harness_dataset=raw["harness_dataset"],
            harness_split=raw["harness_split"],
            data_root=Path(raw["data_root"]),
            repos_root=repos_root,
            trace_root=Path(raw["trace_root"]),
            default_max_steps=int(raw["default_max_steps"]),
            selection_n=int(raw["selection_n"]),
            selection_seed=int(raw["selection_seed"]),
            docker_namespace=raw.get("docker_namespace"),
            exclude_lite=bool(raw.get("exclude_lite", False)),
            extras=dict(raw.get("extras", {})),
        )


class Benchmark(ABC):
    """Abstract base class for all benchmark plugins.

    Subclasses must set :attr:`slug` and :attr:`task_shape` as class
    variables, and implement :meth:`load_tasks` and :meth:`normalize_task`.
    """

    slug: ClassVar[str]
    task_shape: ClassVar[Literal["swe_patch", "function_call"]] = "swe_patch"

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_tasks(self) -> list[dict[str, Any]]:
        """Load and return all tasks for this benchmark.

        Each task is a plain dict with at minimum an ``instance_id`` key.
        """
        ...

    @abstractmethod
    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize a raw task row into the canonical task dict format.

        Args:
            raw: A single raw row as returned by the upstream dataset.

        Returns:
            A normalized task dict suitable for use by scaffolds.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete defaults
    # ------------------------------------------------------------------

    def derive_test_cmd(self, task: dict[str, Any]) -> str:
        """Derive a pytest command string from ``task["FAIL_TO_PASS"]``.

        Delegates to :func:`agents.swebench_data.derive_test_cmd` so that
        the SWE-bench-specific logic lives in one place.
        """
        from agents.swebench_data import derive_test_cmd as _derive

        return _derive(task)

    def select_subset(
        self,
        tasks: list[dict[str, Any]],
        n: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return a stratified subset of *n* tasks.

        Delegates to :func:`agents.swebench_data.select_tool_intensive_tasks`
        using config defaults when *n* / *seed* are not provided.
        """
        from agents.swebench_data import select_tool_intensive_tasks

        return select_tool_intensive_tasks(
            tasks,
            n=n if n is not None else self.config.selection_n,
            seed=seed if seed is not None else self.config.selection_seed,
        )

    def build_harness_args(
        self,
        *,
        predictions_path: Path,
        run_id: str,
        max_workers: int = 1,
        timeout: int = 1800,
        report_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Return keyword arguments suitable for invoking the SWE-bench harness.

        Returns:
            Dict with ``dataset_name``, ``split``, ``namespace``,
            ``predictions_path``, ``run_id``, ``max_workers``, ``timeout``,
            and ``report_dir``.
        """
        return {
            "dataset_name": self.config.harness_dataset,
            "split": self.config.harness_split,
            "namespace": self.config.docker_namespace,
            "predictions_path": predictions_path,
            "run_id": run_id,
            "max_workers": max_workers,
            "timeout": timeout,
            "report_dir": report_dir,
        }

    def image_name_for(self, task: dict[str, Any]) -> str | None:
        """Return the Docker image name for *task*, or ``None`` if not set."""
        return task.get("image_name")

    def build_runner(self, *, scaffold: str, **kwargs: Any) -> Any:
        """Build and return a scaffold runner for this benchmark.

        The base implementation always raises :exc:`NotImplementedError`.
        Subclasses that support concrete scaffold integrations **must** override
        this method.

        Raises:
            NotImplementedError: Always — subclasses must override.
        """
        raise NotImplementedError(
            f"Benchmark {self.slug!r} (task_shape={self.task_shape!r}) does not implement "
            f"build_runner; subclasses must override this method for scaffold={scaffold!r}"
        )
