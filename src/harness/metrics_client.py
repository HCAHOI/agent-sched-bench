"""Stable client wrapper for vLLM scheduler metrics access (Phase 2).

Wraps `harness.scheduler_hooks.get_snapshot()` so the simulator has a
single object to instantiate per run and call `get_snapshot()` on per
iteration. When `metrics_url` is None the client returns a fresh
all-None snapshot (the explicit opt-out path); HTTP errors propagate
as exceptions per CLAUDE.md "no silent fallbacks for real operations".
"""

from __future__ import annotations

from harness.scheduler_hooks import (
    PreemptionSnapshot,
    empty_snapshot,
    get_snapshot,
)


class VLLMMetricsClient:
    """Stateful client for repeated vLLM scheduler metric fetches."""

    def __init__(
        self,
        metrics_url: str | None,
        *,
        timeout_s: float = 5.0,
    ) -> None:
        self.metrics_url = metrics_url
        self.timeout_s = timeout_s

    @property
    def is_enabled(self) -> bool:
        return bool(self.metrics_url)

    def get_snapshot(self) -> PreemptionSnapshot:
        return get_snapshot(self.metrics_url, timeout_s=self.timeout_s)


__all__ = ["VLLMMetricsClient", "empty_snapshot", "get_snapshot"]
