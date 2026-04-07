"""Benchmark plugin registry.

Usage::

    from agents.benchmarks import get_benchmark_class
    cls = get_benchmark_class("swe-bench-verified")
    plugin = cls(config)

New benchmarks register here by adding an entry to :data:`REGISTRY`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.benchmarks.base import Benchmark, BenchmarkConfig
from agents.benchmarks.swe_bench_verified import SWEBenchVerified

if TYPE_CHECKING:
    pass

__all__ = [
    "REGISTRY",
    "get_benchmark_class",
    "Benchmark",
    "BenchmarkConfig",
    "SWEBenchVerified",
]

#: Maps benchmark slug → concrete :class:`~agents.benchmarks.base.Benchmark` subclass.
REGISTRY: dict[str, type[Benchmark]] = {
    "swe-bench-verified": SWEBenchVerified,
}


def get_benchmark_class(slug: str) -> type[Benchmark]:
    """Return the :class:`~agents.benchmarks.base.Benchmark` subclass for *slug*.

    Args:
        slug: Benchmark identifier, e.g. ``"swe-bench-verified"``.

    Returns:
        The registered benchmark class.

    Raises:
        KeyError: If *slug* is not registered.  The error message lists known
            slugs so callers can diagnose typos quickly.
    """
    if slug not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(
            f"Benchmark slug {slug!r} is not registered. "
            f"Known slugs: {known}. "
            "To add swe-rebench or another benchmark, register its class in "
            "agents.benchmarks.REGISTRY."
        )
    return REGISTRY[slug]
