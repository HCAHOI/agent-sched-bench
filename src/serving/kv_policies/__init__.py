"""KV cache eviction policies for the HF recording path.

Step 2 (scaffolding only) — `build_eviction_cache` is intentionally a stub.
Step 3+ wires concrete policies (random, streaming, h2o) into the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from serving.kv_policies.base import (
    BaseEvictionCache,
    EvictionDecision,
    EvictionPolicyConfig,
)
from serving.kv_policies.recorder import KVEvictionRecorder

if TYPE_CHECKING:
    pass


def build_eviction_cache(
    config: EvictionPolicyConfig,
    num_layers: int,
    recorder: KVEvictionRecorder | None = None,
) -> BaseEvictionCache:
    """Factory for the configured eviction cache. Populated in step 3+."""
    raise NotImplementedError(
        f"policy registry empty (requested {config.name!r}); step 3+ will populate"
    )


__all__ = [
    "BaseEvictionCache",
    "EvictionDecision",
    "EvictionPolicyConfig",
    "KVEvictionRecorder",
    "build_eviction_cache",
]
