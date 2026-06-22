"""Batch-eviction margin tests for `BaseEvictionCache.update()`.

The base supports a soft cap: physically drop only once the cache reaches
`budget + eviction_margin`, then evict back to `budget`. This amortizes the
per-token `index_select` rebuild that stalls the FP8/triton decode forward.
`margin=0` must reproduce drop-on-every-over-budget-token exactly.

Driven through the real `update()` tensor path (StreamingLLMCache, which
needs no attention backend) so the gating, recording, and logical-index
compaction are all exercised together.
"""

from __future__ import annotations

import torch

from serving.kv_policies.base import EvictionDecision, EvictionPolicyConfig
from serving.kv_policies.streaming import StreamingLLMCache

_HEADS, _DIM = 2, 4


def _cache(*, margin: int, budget: int = 8, sink: int = 2, recent: int = 6):
    config = EvictionPolicyConfig(
        name="streaming",
        budget=budget,
        eviction_margin=margin,
        sink_size=sink,
        recent_window=recent,
        record=False,
    )
    return StreamingLLMCache(config, num_layers=1)


def _kv(n_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (1, _HEADS, n_tokens, _DIM)
    return torch.randn(shape), torch.randn(shape)


def _prefill(cache, n: int) -> int:
    k, v = _kv(n)
    keys, _ = cache.update(k, v, 0)
    return int(keys.shape[-2])


def _decode_one(cache) -> int:
    k, v = _kv(1)
    keys, _ = cache.update(k, v, 0)
    return int(keys.shape[-2])


def _decision(n_evict: int) -> EvictionDecision:
    # keep_indices are irrelevant to the gate; only the evict-set size matters.
    return EvictionDecision(
        keep_indices=[], evict_indices=list(range(n_evict)), reason="test"
    )


def test_evict_now_gates_on_evict_set_size_not_raw_length() -> None:
    """The drop gate fires on `len(evict) >= margin`, independent of cache
    length or off-budget reserved tokens.

    This is the property that keeps batch eviction working under
    `reserve_system_prompt`, where post-drop length is `budget + system_count`:
    a raw `pre_len >= budget + margin` gate would fire every token there.
    """
    cache = _cache(margin=16)
    assert not cache._evict_now(_decision(15))  # below margin -> defer
    assert cache._evict_now(_decision(16))  # at margin -> fire
    assert cache._evict_now(_decision(9999))  # far over -> fire

    zero = _cache(margin=0)
    assert zero._evict_now(_decision(1))  # margin 0: any pending evict fires
    assert zero._evict_now(_decision(0))  # len 0 >= 0; outer guard short-circuits


def test_margin_zero_matches_evict_every_token() -> None:
    """margin=0: every decode above budget snaps straight back to budget."""
    cache = _cache(margin=0)
    assert _prefill(cache, 8) == 8  # at budget, no eviction
    for _ in range(20):
        assert _decode_one(cache) == 8  # never exceeds budget


def test_margin_defers_then_drops_to_budget() -> None:
    """margin=4: KV grows to budget+margin, the next step drops to budget."""
    budget, margin = 8, 4
    cache = _cache(margin=margin)
    assert _prefill(cache, 8) == budget

    # Each decode appends one token; the resident length grows 9,10,11 and on
    # the step that would reach budget+margin=12 the drop fires inside the same
    # update(), so that step returns budget. Period = margin (3 grows + 1 drop).
    assert [_decode_one(cache) for _ in range(margin)] == [9, 10, 11, budget]
    # The cycle repeats indefinitely.
    assert [_decode_one(cache) for _ in range(margin)] == [9, 10, 11, budget]


def test_margin_keeps_kv_physically_bounded() -> None:
    """Physical length never exceeds budget + margin over a long decode."""
    budget, margin = 8, 16
    cache = _cache(margin=margin)
    _prefill(cache, 8)
    for _ in range(200):
        assert _decode_one(cache) <= budget + margin


def test_margin_preserves_sink_after_drop() -> None:
    """A deferred-then-fired drop still keeps the StreamingLLM sink+recent set.

    Logical-index tracking must survive the margin growth phase: after the
    drop the surviving originals are the sink (0,1) plus the recent window.
    """
    budget, margin, sink = 8, 4, 2
    cache = _cache(margin=margin, budget=budget, sink=sink, recent=budget - sink)
    _prefill(cache, 8)
    # `margin` decodes land exactly on the step where the batched drop fires.
    assert [_decode_one(cache) for _ in range(margin)][-1] == budget
    logical = cache._logical_indices_by_layer[0]
    assert len(logical) == budget
    assert logical[:sink] == [0, 1]  # sink preserved across the batched drop
    # Remaining kept slots are the most-recent originals, strictly increasing.
    assert logical == sorted(logical)
