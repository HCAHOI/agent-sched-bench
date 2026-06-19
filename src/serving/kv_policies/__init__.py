"""KV cache eviction policies for the HF recording path.

Step 3-6 wire `random` + `streaming` + `h2o`; metadata residency and
position controls extend the same physical-drop surface.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from serving.kv_policies.base import (
    BaseEvictionCache,
    EvictionDecision,
    EvictionPolicyConfig,
)
from serving.kv_policies.recorder import KVEvictionRecorder

if TYPE_CHECKING:
    from serving.recording.attention_bus import AttentionBus

_POLICY_CLASS_PATHS: dict[str, tuple[str, str]] = {
    "random": ("serving.kv_policies.random_evict", "RandomEvictCache"),
    "streaming": ("serving.kv_policies.streaming", "StreamingLLMCache"),
    "h2o": ("serving.kv_policies.h2o", "H2OCache"),
    "metadata": ("serving.kv_policies.metadata", "MetadataResidencyCache"),
    "position_control": ("serving.kv_policies.metadata", "PositionControlCache"),
    "null_eviction": ("serving.kv_policies.metadata", "NullEvictionCache"),
}


def eviction_cache_class(config: EvictionPolicyConfig) -> type[BaseEvictionCache]:
    """Return the cache class registered for a resolved eviction config."""
    if config.name == "none":
        raise ValueError(
            "eviction_cache_class should not be called when policy is disabled "
            "(config.name == 'none'); gate on `eviction_config is not None`"
        )
    class_path = _POLICY_CLASS_PATHS.get(config.name)
    if class_path is None:
        raise NotImplementedError(f"policy {config.name!r} not registered")
    module_name, class_name = class_path
    cls = getattr(import_module(module_name), class_name)
    if not issubclass(cls, BaseEvictionCache):
        raise TypeError(
            f"registered policy class {class_name!r} is not a BaseEvictionCache"
        )
    return cls


def eviction_policy_requires_attention(config: EvictionPolicyConfig) -> bool:
    """Whether the configured policy needs post-softmax attention tensors."""
    return eviction_cache_class(config).requires_attention_backend()


def build_eviction_cache(
    config: EvictionPolicyConfig,
    num_layers: int,
    recorder: KVEvictionRecorder | None = None,
    *,
    attention_bus: "AttentionBus | None" = None,
    max_position_embeddings: int | None = None,
) -> BaseEvictionCache:
    """Factory for the configured eviction cache.

    Callers must gate on `config is not None` upstream — the `"none"` policy
    is a CLI-layer concept (no cache subclass needed) so reaching this factory
    with `name="none"` indicates a wiring bug, not a runtime fallback.

    `attention_bus` and `max_position_embeddings` are required for `h2o`
    (it subscribes to the bus at construction and pre-allocates a per-layer
    score buffer sized by `max_position_embeddings`); other policies ignore
    them.
    """
    cache_cls = eviction_cache_class(config)
    if cache_cls.requires_attention_backend():
        if attention_bus is None:
            raise ValueError(
                f"build_eviction_cache(name={config.name!r}) requires attention_bus; "
                "the policy subscribes at construction to share post-softmax "
                "tensors with LayerCapturer."
            )
        if max_position_embeddings is None:
            raise ValueError(
                f"build_eviction_cache(name={config.name!r}) requires "
                "max_position_embeddings."
            )
        return cache_cls(
            config,
            num_layers,
            recorder=recorder,
            attention_bus=attention_bus,
            max_position_embeddings=int(max_position_embeddings),
        )
    return cache_cls(config, num_layers, recorder=recorder)


__all__ = [
    "BaseEvictionCache",
    "EvictionDecision",
    "EvictionPolicyConfig",
    "KVEvictionRecorder",
    "build_eviction_cache",
    "eviction_cache_class",
    "eviction_policy_requires_attention",
]
