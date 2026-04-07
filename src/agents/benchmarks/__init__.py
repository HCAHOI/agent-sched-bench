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
from agents.benchmarks.bfcl_v4 import BFCLv4Benchmark
from agents.benchmarks.swe_bench_verified import SWEBenchVerified
from agents.benchmarks.swe_rebench import SWERebenchBenchmark

if TYPE_CHECKING:
    pass

__all__ = [
    "REGISTRY",
    "get_benchmark_class",
    "Benchmark",
    "BenchmarkConfig",
    "BFCLv4Benchmark",
    "SWEBenchVerified",
    "SWERebenchBenchmark",
]

#: Maps benchmark slug → concrete :class:`~agents.benchmarks.base.Benchmark` subclass.
REGISTRY: dict[str, type[Benchmark]] = {
    "swe-bench-verified": SWEBenchVerified,
    "swe-rebench": SWERebenchBenchmark,
    "bfcl-v4": BFCLv4Benchmark,
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
            "To add a new benchmark, create src/agents/benchmarks/<slug>.py "
            "and register its class in agents.benchmarks.REGISTRY."
        )
    return REGISTRY[slug]
