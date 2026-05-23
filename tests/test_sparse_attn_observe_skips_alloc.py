"""Observe-only must skip [1,1,Q,K] tensor allocation.

Regression for a 3.3x per-turn slowdown: in observe mode the pre-hook
was unconditionally calling build_additive_mask, which materialized a
full [1,1,Q,K] mask tensor (~5GB fp16 at Q=K~50K) per layer per call,
then immediately discarded it. The fix: each method short-circuits and
returns None when self.observe_only=True, AFTER updating record state.

This test pins two contracts:
1. build_additive_mask returns None for every method when observe_only=True
2. State that the recorder consumes (kept_count, record_metadata) is still
   correctly updated — observe-only semantics unchanged
"""

from __future__ import annotations

import pytest
import torch

from serving.recording.attention_bus import AttentionBus
from serving.sparse_attention import build_sparse_attention
from serving.sparse_attention.base import SparseAttentionConfig


@pytest.mark.parametrize("method_name", ["sliding", "streaming"])
def test_sliding_observe_only_returns_none_prefill(method_name: str) -> None:
    config = SparseAttentionConfig(
        name=method_name,
        sink_size=4,
        recent_window=8,
        record=True,
        observe_only=True,
    )
    method = build_sparse_attention(config, num_layers=1)
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=16,
        key_len=16,
        phase="prefill",
        decode_step=-1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask is None, f"{method_name} observe-only prefill must return None"
    # kept_count is a pure function of key_len for sliding — no state to check.
    assert method.kept_count(16) > 0


@pytest.mark.parametrize("method_name", ["sliding", "streaming"])
def test_sliding_observe_only_returns_none_decode(method_name: str) -> None:
    config = SparseAttentionConfig(
        name=method_name,
        sink_size=4,
        recent_window=8,
        record=True,
        observe_only=True,
    )
    method = build_sparse_attention(config, num_layers=1)
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=1,
        key_len=32,
        phase="decode",
        decode_step=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask is None
    assert method.kept_count(32) > 0


@pytest.mark.parametrize("method_name", ["block_topk", "quest", "heavy_hitter"])
def test_dynamic_observe_only_returns_none_prefill_phase_dense(method_name: str) -> None:
    """Prefill / phase_dense early return: must skip mask but still _record state."""
    bus = AttentionBus()
    config = SparseAttentionConfig(
        name=method_name,
        sink_size=4,
        recent_window=8,
        budget=16,
        block_size=4,
        record=True,
        observe_only=True,
        phase_scope="decode_only",
    )
    method = build_sparse_attention(config, num_layers=1, attention_bus=bus)
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=16,
        key_len=16,
        phase="prefill",
        decode_step=-1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask is None, f"{method_name} observe-only prefill must return None"
    md = method.record_metadata(layer_idx=0, phase="prefill", decode_step=-1)
    assert md.get("selection_reason") in {"phase_dense", "prefill_dense"}, (
        f"{method_name} must record phase_dense/prefill_dense reason; got {md}"
    )
    # kept_count must reflect the dense fallback (all key positions kept).
    assert method.kept_count(16) == 16, (
        f"{method_name} dense fallback must record kept_count==key_len; "
        f"got {method.kept_count(16)}"
    )


@pytest.mark.parametrize("method_name", ["block_topk", "quest", "heavy_hitter"])
def test_dynamic_observe_only_returns_none_empty_key_len(method_name: str) -> None:
    """key_len <= 0 early return: must skip mask, still _record reason=empty."""
    bus = AttentionBus()
    config = SparseAttentionConfig(
        name=method_name,
        sink_size=4,
        recent_window=8,
        budget=16,
        block_size=4,
        record=True,
        observe_only=True,
        phase_scope="decode_only",
    )
    method = build_sparse_attention(config, num_layers=1, attention_bus=bus)
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=1,
        key_len=0,
        phase="decode",
        decode_step=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask is None, f"{method_name} observe-only key_len=0 must return None"
    md = method.record_metadata(layer_idx=0, phase="decode", decode_step=0)
    assert md.get("selection_reason") == "empty", (
        f"{method_name} key_len=0 must record reason=empty; got {md}"
    )


def test_enforce_mode_still_returns_tensor() -> None:
    """Regression: enforce mode (observe_only=False) must still return a Tensor."""
    config = SparseAttentionConfig(
        name="sliding",
        sink_size=4,
        recent_window=8,
        record=False,
        observe_only=False,
    )
    method = build_sparse_attention(config, num_layers=1)
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=8,
        key_len=16,
        phase="prefill",
        decode_step=-1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask is not None and mask.shape == (1, 1, 8, 16)


@pytest.mark.parametrize("method_name", ["block_topk", "quest", "heavy_hitter"])
def test_enforce_mode_dynamic_still_returns_tensor_prefill(method_name: str) -> None:
    """Dynamic methods in enforce mode keep the dense-fallback tensor for prefill."""
    bus = AttentionBus()
    config = SparseAttentionConfig(
        name=method_name,
        sink_size=4,
        recent_window=8,
        budget=16,
        block_size=4,
        record=True,
        observe_only=False,
        phase_scope="decode_only",
    )
    method = build_sparse_attention(config, num_layers=1, attention_bus=bus)
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=8,
        key_len=16,
        phase="prefill",
        decode_step=-1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask is not None, f"{method_name} enforce prefill must return a tensor"
    assert mask.shape == (1, 1, 8, 16), (
        f"{method_name} enforce prefill shape: got {tuple(mask.shape)}"
    )
