"""KV cache eviction policies for the HF recording path.

Steps 3 + 4 wire `random` + `streaming`. `h2o` ships in step 6.
"""

from __future__ import annotations

from serving.kv_policies.base import (
    BaseEvictionCache,
    EvictionDecision,
    EvictionPolicyConfig,
)
from serving.kv_policies.recorder import KVEvictionRecorder


def build_eviction_cache(
    config: EvictionPolicyConfig,
    num_layers: int,
    recorder: KVEvictionRecorder | None = None,
) -> BaseEvictionCache:
    """Factory for the configured eviction cache.

    Callers must gate on `config is not None` upstream — the `"none"` policy
    is a CLI-layer concept (no cache subclass needed) so reaching this factory
    with `name="none"` indicates a wiring bug, not a runtime fallback.
    """
    name = config.name
    if name == "random":
        # Local import keeps `transformers.cache_utils` import out of module
        # load when no policy is in use.
        from serving.kv_policies.random_evict import RandomEvictCache

        return RandomEvictCache(config, num_layers, recorder=recorder)
    if name == "streaming":
        from serving.kv_policies.streaming import StreamingLLMCache

        return StreamingLLMCache(config, num_layers, recorder=recorder)
    if name == "none":
        raise ValueError(
            "build_eviction_cache should not be called when policy is disabled "
            "(config.name == 'none'); gate on `eviction_config is not None`"
        )
    raise NotImplementedError(
        f"policy {name!r} not yet registered (step 6 adds h2o)"
    )


__all__ = [
    "BaseEvictionCache",
    "EvictionDecision",
    "EvictionPolicyConfig",
    "KVEvictionRecorder",
    "build_eviction_cache",
]
