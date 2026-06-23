"""`supports_session_resume()` contract for eviction caches.

LCP-crop resume is only sound when every layer keeps the same token set.
Layer-divergent policies (random per-layer sampling, h2o per-layer scores)
must opt out, else a crop yields per-layer-different physical lengths and the
shared attention mask crashes the forward.
"""

from __future__ import annotations

import torch

from serving.kv_policies.base import EvictionPolicyConfig
from serving.kv_policies.h2o import H2OCache
from serving.kv_policies.metadata import MetadataResidencyCache
from serving.kv_policies.random_evict import RandomEvictCache
from serving.kv_policies.streaming import StreamingLLMCache
from serving.recording.attention_bus import AttentionBus


def test_resume_flag_per_policy() -> None:
    rnd = RandomEvictCache(
        EvictionPolicyConfig(name="random", budget=128), num_layers=2
    )
    strm = StreamingLLMCache(
        EvictionPolicyConfig(name="streaming", budget=128, sink_size=4, recent_window=124),
        num_layers=2,
    )
    meta = MetadataResidencyCache(
        EvictionPolicyConfig(
            name="metadata", budget=512, sink_size=4, recent_window=256, metadata_rung="rung4"
        ),
        num_layers=2,
    )
    h2o = H2OCache(
        EvictionPolicyConfig(name="h2o", budget=128, sink_size=4, recent_window=60),
        num_layers=2,
        attention_bus=AttentionBus(),
        max_position_embeddings=4096,
    )
    assert rnd.supports_session_resume() is False
    assert h2o.supports_session_resume() is False
    assert strm.supports_session_resume() is True
    assert meta.supports_session_resume() is True


def test_random_crop_diverges_streaming_does_not() -> None:
    """The invariant that justifies the opt-out: a logical-prefix crop leaves
    random's layers at different physical lengths, streaming's at equal ones.
    """
    def crop_lengths(cache, n_layers):
        for _ in range(200):
            for layer in range(n_layers):
                cache.update(torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4), layer)
        cache.crop_to_logical_length(100)
        return [cache.get_seq_length(layer) for layer in range(n_layers)]

    rnd = crop_lengths(
        RandomEvictCache(EvictionPolicyConfig(name="random", budget=64), num_layers=3), 3
    )
    strm = crop_lengths(
        StreamingLLMCache(
            EvictionPolicyConfig(name="streaming", budget=64, sink_size=4, recent_window=60),
            num_layers=3,
        ),
        3,
    )
    assert len(set(rnd)) > 1, f"random layers should diverge, got {rnd}"
    assert len(set(strm)) == 1, f"streaming layers should match, got {strm}"
