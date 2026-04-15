"""Scaffold/benchmark capability matrix derived from benchmark plugins."""

from __future__ import annotations


ALL_SCAFFOLDS = ("openclaw", "qwen-deep-research")


def _supported_scaffolds_for(cls: type[object]) -> set[str]:
    raw = getattr(cls, "SUPPORTED_SCAFFOLDS", None)
    if raw is None:
        return set()
    return {str(scaffold) for scaffold in raw}


def scaffold_benchmark_matrix() -> dict[str, set[str]]:
    """Return scaffold -> supported benchmark slugs."""
    from agents.benchmarks import REGISTRY

    matrix: dict[str, set[str]] = {scaffold: set() for scaffold in ALL_SCAFFOLDS}
    for slug, cls in REGISTRY.items():
        supported = _supported_scaffolds_for(cls)
        for scaffold in ALL_SCAFFOLDS:
            if scaffold in supported:
                matrix[scaffold].add(slug)
    return matrix


def validate_scaffold_benchmark(scaffold: str, benchmark_slug: str) -> None:
    """Raise ValueError when a scaffold/benchmark pair is unsupported."""
    from agents.benchmarks import REGISTRY

    if scaffold not in ALL_SCAFFOLDS:
        raise ValueError(
            f"Unknown scaffold {scaffold!r}; known scaffolds: {', '.join(ALL_SCAFFOLDS)}"
        )
    if benchmark_slug not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown benchmark {benchmark_slug!r}; known benchmarks: {known}"
        )
    if benchmark_slug not in scaffold_benchmark_matrix()[scaffold]:
        raise ValueError(
            f"scaffold={scaffold!r} does not support benchmark={benchmark_slug!r}"
        )

