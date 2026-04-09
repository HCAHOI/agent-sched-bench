"""Base classes and configuration for the benchmark plugin architecture.

New code should instantiate benchmarks via::

    from agents.benchmarks import get_benchmark_class
    cls = get_benchmark_class("swe-bench-verified")
    plugin = cls(config)
"""

from __future__ import annotations

import json
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

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
    default_max_iterations: int
    selection_n: int
    selection_seed: int
    default_prompt_template: str = "default"
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
            default_max_iterations=int(raw["default_max_iterations"]),
            selection_n=int(raw["selection_n"]),
            selection_seed=int(raw["selection_seed"]),
            default_prompt_template=str(raw.get("default_prompt_template", "default")),
            exclude_lite=bool(raw.get("exclude_lite", False)),
            extras=dict(raw.get("extras", {})),
        )

class Benchmark(ABC):
    """Abstract base class for all benchmark plugins.

    Subclasses must set :attr:`slug` and implement :meth:`load_tasks`
    and :meth:`normalize_task`.
    """

    slug: ClassVar[str]

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    # Abstract interface

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

    # Concrete defaults

    def derive_test_cmd(self, task: dict[str, Any]) -> str:
        """Derive a pytest command from ``task["FAIL_TO_PASS"]``.

        Handles both native list (SWE-rebench) and JSON-encoded string
        (SWE-Bench Verified) forms.
        """
        raw = task.get("FAIL_TO_PASS", "[]")
        if isinstance(raw, str):
            try:
                test_ids = json.loads(raw)
            except json.JSONDecodeError:
                test_ids = [raw] if raw else []
        else:
            test_ids = list(raw)
        if not test_ids:
            return "python -m pytest --no-header -q"
        return f"python -m pytest {' '.join(test_ids)} -x --no-header -q"

    def select_subset(
        self,
        tasks: list[dict[str, Any]],
        n: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the first ``n`` tasks, sorted by instance_id for determinism.

        This is the simplest possible benchmark-agnostic default. Subclasses
        with specific selection needs (repo-stratified like SWE-Bench Verified,
        lite-filtering like SWE-rebench) MUST override this.
        """
        effective_n = n if n is not None else self.config.selection_n
        sorted_tasks = sorted(tasks, key=lambda t: t.get("instance_id", ""))
        return sorted_tasks[:effective_n]

    def image_name_for(self, task: dict[str, Any]) -> str | None:
        return task.get("image_name")

    def runtime_mode_for(self, scaffold: str) -> str:
        """Return the runtime strategy label for the given scaffold."""
        return "host_controller"

    def build_runner(self, *, scaffold: str, **kwargs: Any) -> Any:
        """Build and return a scaffold runner for this benchmark.

        The base implementation always raises :exc:`NotImplementedError`.
        Subclasses that support concrete scaffold integrations **must** override
        this method.

        Raises:
            NotImplementedError: Always — subclasses must override.
        """
        raise NotImplementedError(
            f"Benchmark {self.slug!r} does not implement build_runner; "
            f"subclasses must override this method for scaffold={scaffold!r}"
        )
