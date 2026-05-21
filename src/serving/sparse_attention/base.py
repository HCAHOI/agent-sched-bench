"""Scaffolding for sparse attention methods.

`BaseSparseAttention` is a Protocol: implementations build a 4D additive
attention mask consumed by an HF pre-forward hook on each `self_attn` module.
Unlike `kv_policies/`, sparsity does NOT alter K/V cache contents — it
constrains which key positions each query may attend to within an otherwise
unmodified cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class SparseAttentionConfig:
    """User-facing config for a sparse attention method.

    Method-specific knobs sit alongside the shared ones (`name`, `record`).
    Currently only `sliding` consumes `sink_size` / `recent_window`; future
    methods may grow per-method fields here. Frozen to match
    `EvictionPolicyConfig`'s contract — providers stash a single config and
    pass it into the factory.
    """

    name: Literal["none", "sliding"]
    record: bool = False
    # Default sink_size=4 follows StreamingLLM (Xiao et al. 2024,
    # arXiv:2309.17453): the first few tokens carry the "attention sink"
    # that prevents softmax-distribution collapse when the middle is masked.
    sink_size: int = 4  # sliding
    # 256 is a reasonable default for short-context smoke tests; tune per
    # model and workload. Not anchored to any specific paper.
    recent_window: int = 256  # sliding


class BaseSparseAttention(Protocol):
    """Protocol every sparse attention method must satisfy.

    The HF pre-forward hook calls `build_additive_mask` per layer per forward,
    adds the returned tensor to any existing `attention_mask` kwarg, and then
    (when recording) appends one row of `record_metadata` to the recorder.

    `build_additive_mask` returns a tensor broadcastable to `[B, H, Q, K]`
    where 0 means "attend" and `-inf` means "mask". HF's upstream causal mask
    is applied separately by the SDPA path; implementations only encode the
    sparsity pattern, not causality.
    """

    name: str

    def build_additive_mask(
        self,
        *,
        layer_idx: int,
        query_len: int,
        key_len: int,
        phase: str,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Return additive sparsity mask broadcastable to [B, H, Q, K]."""

    def record_metadata(
        self,
        *,
        layer_idx: int,
        phase: str,
        decode_step: int,
    ) -> dict[str, Any]:
        """Per-call metadata recorded alongside the (layer, phase, step) row."""
