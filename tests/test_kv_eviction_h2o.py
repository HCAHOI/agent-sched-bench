"""Unit tests for `H2OCache` (plan G16#2 + bus subscription contract).

Direct exercises of `observe()` and `_decide_evict()` without driving the
full `update()` path through transformers — that is covered by the smoke
script (`scripts/spikes/step6_h2o_smoke.py`).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from serving.kv_policies.base import EvictionPolicyConfig
from serving.kv_policies.h2o import H2OCache
from serving.recording.attention_bus import AttentionBus


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _make_cache(
    *,
    budget: int = 4,
    sink_size: int = 1,
    recent_window: int = 1,
    num_layers: int = 2,
    max_position_embeddings: int = 64,
    bus: AttentionBus | None = None,
) -> tuple[H2OCache, AttentionBus]:
    """Build an H2OCache + the bus it subscribes to."""
    config = EvictionPolicyConfig(
        name="h2o",
        budget=budget,
        sink_size=sink_size,
        recent_window=recent_window,
    )
    bus = bus or AttentionBus()
    cache = H2OCache(
        config,
        num_layers=num_layers,
        attention_bus=bus,
        max_position_embeddings=max_position_embeddings,
    )
    return cache, bus


def _attn_with_peaks(
    *,
    key_len: int,
    peak_positions: list[int],
    n_heads: int = 2,
    n_query_rows: int = 1,
    peak_value: float = 10.0,
) -> torch.Tensor:
    """Construct a (B=1, H, Q, K) attn tensor with mass concentrated on `peak_positions`.

    Each query row puts `peak_value` on each peak position and 0 elsewhere.
    Not a real softmax distribution (doesn't sum to 1) — but `H2OCache.observe`
    only sums and head-means, so the absolute scale is what matters for the
    decision contract.
    """
    attn = torch.zeros(1, n_heads, n_query_rows, key_len, dtype=torch.float32)
    for pos in peak_positions:
        attn[..., pos] = peak_value
    return attn


# ---------------------------------------------------------------------------
# G16#2: heavy-hitter eviction picks the right positions.
# ---------------------------------------------------------------------------


def test_h2o_evicts_lowest_score() -> None:
    """Plan G16#2: positions {3, 7} score 10, others 0; budget=4 sink=1 recent=1.

    Expected keep set:
      sink   = {0}                         (position 0)
      recent = {key_len - 1}               (position 9)
      heavy  = top 2 of middle [1, 9) by score = {3, 7}

    -> keep = sorted({0, 3, 7, 9}) = [0, 3, 7, 9], evict = [1, 2, 4, 5, 6, 8].
    """
    cache, _bus = _make_cache(
        budget=4, sink_size=1, recent_window=1, max_position_embeddings=16
    )
    key_len = 10
    attn = _attn_with_peaks(key_len=key_len, peak_positions=[3, 7])
    cache.observe(
        layer=0,
        attn=attn,
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=key_len,
        phase="prefill",
    )

    decision = cache._decide_evict(layer_idx=0, key_len=key_len)
    assert decision.reason == "over_budget"
    # Heavy-hitter set must include the two peaks; sink + recent always kept.
    assert set(decision.keep_indices) >= {0, 3, 7, 9}
    assert decision.keep_indices == sorted(decision.keep_indices)
    # Post-eviction length == budget.
    assert len(decision.keep_indices) == 4
    # Partition coverage.
    assert sorted(decision.keep_indices + decision.evict_indices) == list(range(key_len))
    # policy_state carries the topk diagnostics for npz.
    state = decision.policy_state
    assert state is not None
    assert set(state["score_topk_index"]) == {3, 7}
    # Score values match the cumulative head-mean we accumulated.
    # Each peak got 10.0 across all heads -> mean = 10.0 (one query row).
    assert all(abs(v - 10.0) < 1e-5 for v in state["score_topk_value"])


def test_h2o_under_budget_no_eviction() -> None:
    cache, _bus = _make_cache(
        budget=4, sink_size=1, recent_window=1, max_position_embeddings=16
    )
    decision = cache._decide_evict(layer_idx=0, key_len=4)
    assert decision.reason == "none"
    assert decision.evict_indices == []
    assert decision.keep_indices == list(range(4))


# ---------------------------------------------------------------------------
# Bus + suspend interaction.
# ---------------------------------------------------------------------------


def test_h2o_subscribes_with_always_active() -> None:
    """always_active=True so suspend gating cannot starve the score buffer."""
    bus = AttentionBus()
    cache, _ = _make_cache(bus=bus, max_position_embeddings=16)
    assert bus.n_consumers() == 1
    assert getattr(cache, "always_active") is True

    key_len = 8
    attn = _attn_with_peaks(key_len=key_len, peak_positions=[2, 5])
    bus.publish(
        layer=0,
        attn=attn,
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=key_len,
        phase="decode",
        suspended=True,  # gating ON
    )

    # Score buffer must have absorbed the publish despite the gate.
    buffer = cache._scores[0]
    assert buffer is not None
    # Peaks at 2 and 5 should match peak_value (head-mean of constant peak).
    assert float(buffer[2]) > 0
    assert float(buffer[5]) > 0
    # Off-peak slot stayed zero.
    assert float(buffer[1]) == 0


def test_h2o_unsubscribe_works() -> None:
    """Manual unsubscribe stops further observations (provider drives this)."""
    bus = AttentionBus()
    cache, _ = _make_cache(bus=bus, max_position_embeddings=16)
    assert bus.n_consumers() == 1

    bus.unsubscribe(cache)
    assert bus.n_consumers() == 0

    key_len = 8
    attn = _attn_with_peaks(key_len=key_len, peak_positions=[2])
    bus.publish(
        layer=0,
        attn=attn,
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=key_len,
        phase="prefill",
        suspended=False,
    )
    # observe() never ran -> score buffer is still None.
    assert cache._scores[0] is None


# ---------------------------------------------------------------------------
# Score buffer compaction after a triggered eviction.
# ---------------------------------------------------------------------------


def test_h2o_post_evict_compacts_scores() -> None:
    """After eviction, score buffer must mirror the keep set.

    Drive: observe -> _decide_evict (returns keep + heavy + recent) ->
    `_post_evict_hook(decision)`. Verify buffer slots match the
    `buffer[keep_indices]` permutation, with the tail zeroed.
    """
    cache, _bus = _make_cache(
        budget=4, sink_size=1, recent_window=1, max_position_embeddings=16
    )
    key_len = 10
    attn = _attn_with_peaks(key_len=key_len, peak_positions=[3, 7])
    cache.observe(
        layer=0,
        attn=attn,
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=key_len,
        phase="prefill",
    )

    # Snapshot buffer + decision (without going through the K/V path).
    buffer_before = cache._scores[0].clone()
    decision = cache._decide_evict(layer_idx=0, key_len=key_len)
    cache._post_evict_hook(layer_idx=0, decision=decision)

    buffer_after = cache._scores[0]
    n_keep = len(decision.keep_indices)
    # Live prefix matches the gathered slots.
    expected = buffer_before[torch.as_tensor(decision.keep_indices, dtype=torch.long)]
    assert torch.allclose(buffer_after[:n_keep], expected)
    # Tail past the keep set is zero (so future observe() into a now-empty
    # slot writes the correct cumulative sum from 0).
    assert torch.all(buffer_after[n_keep:] == 0)
    assert cache._score_lengths[0] == n_keep


# ---------------------------------------------------------------------------
# Construction / config validation.
# ---------------------------------------------------------------------------


def test_h2o_requires_attention_bus() -> None:
    config = EvictionPolicyConfig(
        name="h2o", budget=4, sink_size=1, recent_window=1
    )
    with pytest.raises(ValueError, match="attention_bus"):
        H2OCache(
            config,
            num_layers=2,
            attention_bus=None,  # type: ignore[arg-type]
            max_position_embeddings=64,
        )


def test_h2o_requires_max_position_embeddings() -> None:
    config = EvictionPolicyConfig(
        name="h2o", budget=4, sink_size=1, recent_window=1
    )
    bus = AttentionBus()
    with pytest.raises(ValueError, match="max_position_embeddings"):
        H2OCache(
            config,
            num_layers=2,
            attention_bus=bus,
            max_position_embeddings=None,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="max_position_embeddings"):
        H2OCache(
            config,
            num_layers=2,
            attention_bus=bus,
            max_position_embeddings=0,
        )


def test_h2o_validates_budget_floor() -> None:
    bus = AttentionBus()
    with pytest.raises(ValueError, match="budget >= sink_size \\+ recent_window"):
        H2OCache(
            EvictionPolicyConfig(
                name="h2o", budget=4, sink_size=4, recent_window=4
            ),
            num_layers=2,
            attention_bus=bus,
            max_position_embeddings=64,
        )


def test_h2o_unimplemented_aggregates_raise() -> None:
    bus = AttentionBus()
    for agg in ("mean", "ema"):
        with pytest.raises(NotImplementedError, match="aggregate="):
            H2OCache(
                EvictionPolicyConfig(
                    name="h2o",
                    budget=4,
                    sink_size=1,
                    recent_window=1,
                    aggregate=agg,
                ),
                num_layers=2,
                attention_bus=bus,
                max_position_embeddings=64,
            )


def test_h2o_observe_validates_shape() -> None:
    cache, _bus = _make_cache(max_position_embeddings=16)
    # Wrong dim count.
    with pytest.raises(ValueError, match="4-D attn"):
        cache.observe(
            layer=0,
            attn=torch.zeros(2, 8),
            query_positions=torch.tensor([0], dtype=torch.long),
            key_len=8,
            phase="prefill",
        )
    # Mismatched key_len.
    with pytest.raises(ValueError, match="published key_len"):
        cache.observe(
            layer=0,
            attn=torch.zeros(1, 2, 1, 8),
            query_positions=torch.tensor([0], dtype=torch.long),
            key_len=10,
            phase="prefill",
        )
    # Out-of-range layer.
    with pytest.raises(ValueError, match="outside"):
        cache.observe(
            layer=99,
            attn=torch.zeros(1, 2, 1, 8),
            query_positions=torch.tensor([0], dtype=torch.long),
            key_len=8,
            phase="prefill",
        )


def test_h2o_observe_rejects_overlong_key() -> None:
    cache, _bus = _make_cache(max_position_embeddings=8)
    with pytest.raises(ValueError, match="exceeds"):
        cache.observe(
            layer=0,
            attn=torch.zeros(1, 2, 1, 16),
            query_positions=torch.tensor([0], dtype=torch.long),
            key_len=16,
            phase="prefill",
        )


# ---------------------------------------------------------------------------
# Decide-evict guards: never run with no observation.
# ---------------------------------------------------------------------------


def test_h2o_decide_evict_without_observe_raises() -> None:
    """Score buffer absent -> raise instead of silently using zeros."""
    cache, _bus = _make_cache(
        budget=4, sink_size=1, recent_window=1, max_position_embeddings=16
    )
    with pytest.raises(RuntimeError, match="no score buffer"):
        cache._decide_evict(layer_idx=0, key_len=10)


def test_h2o_decide_evict_with_stale_middle_raises() -> None:
    """Score buffer that doesn't cover the middle window -> raise.

    The check is `score_lengths < middle_end` (not `< key_len`) because in
    HF generate the cache.update() runs BEFORE the softmax hook publishes
    for the same step — newly-appended positions sit in the recent window
    and don't need a score. But if the score buffer doesn't even cover the
    middle window, the bus wiring is broken.
    """
    cache, _bus = _make_cache(
        budget=4, sink_size=1, recent_window=1, max_position_embeddings=64
    )
    # Observe at key_len=2 only — middle for key_len=10 is [1, 9), so we
    # need score length >= 9, but only see 2.
    cache.observe(
        layer=0,
        attn=torch.zeros(1, 2, 1, 2),
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=2,
        phase="prefill",
    )
    with pytest.raises(RuntimeError, match="< middle_end"):
        cache._decide_evict(layer_idx=0, key_len=10)


# ---------------------------------------------------------------------------
# G16#5 echo: observe() must never call torch.softmax (consumer just reads).
# ---------------------------------------------------------------------------


def test_h2o_observe_does_not_resoftmax(monkeypatch) -> None:
    """H2O must consume the bus tensor as-is; any re-softmax breaks G16#5."""
    cache, _bus = _make_cache(max_position_embeddings=16)
    real = torch.softmax
    calls = {"n": 0}

    def counting_softmax(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "softmax", counting_softmax)

    cache.observe(
        layer=0,
        attn=torch.zeros(1, 2, 1, 8),
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=8,
        phase="decode",
    )
    assert calls["n"] == 0, (
        "H2OCache.observe must not call torch.softmax (G16#5); "
        f"saw {calls['n']} calls"
    )
