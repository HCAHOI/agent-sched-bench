"""Sliding-window sparse attention (sink prefix + recent tail).

Per-row keep set (query row q at absolute position `cached_len + q =
key_len - query_len + q`):
  k in [0, sink_size)                       -> 0    (attend)
  k in [key_len - recent_window, key_len)   -> 0    (attend)
  otherwise                                  -> -inf (mask)
  AND additionally: k > (key_len - query_len) + q -> -inf (causal)

At prefill (query_len > 1) the additive mask we return is `[1,1,Q,K]` with
the upper triangle (k > absolute query position) forced to -inf. This is
load-bearing: the recording pre-hook sets HF SDPA's `attention_mask`
kwarg to a non-None tensor, which disables SDPA's implicit `is_causal`
shortcut. Without the causal upper triangle here, query row 0 would see
the recent-window tail (future tokens) — a hindsight leak.

At decode (query_len == 1) the single query row sits at position
`key_len - 1`, which is >= every k in `[0, key_len)`. Causality is
trivially satisfied, so we keep the cheaper `[1,1,1,K]` key-uniform
mask shape and let SDPA broadcast.

When `sink_size + recent_window >= key_len`, no sparsity positions are
masked at decode; causal still applies at prefill via the per-row
upper-triangular cut.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from serving.sparse_attention.base import SparseAttentionConfig

if TYPE_CHECKING:
    import torch


class SlidingWindowSparseAttention:
    """Sink-prefix + recent-tail sparse attention."""

    name = "sliding"

    def __init__(self, sink_size: int, recent_window: int) -> None:
        if sink_size < 0:
            raise ValueError(
                f"SlidingWindowSparseAttention requires sink_size >= 0; "
                f"got {sink_size!r}"
            )
        if recent_window < 0:
            raise ValueError(
                f"SlidingWindowSparseAttention requires recent_window >= 0; "
                f"got {recent_window!r}"
            )
        if sink_size + recent_window <= 0:
            raise ValueError(
                "SlidingWindowSparseAttention requires sink_size + recent_window > 0; "
                f"got sink_size={sink_size!r}, recent_window={recent_window!r}"
            )
        self.sink_size = int(sink_size)
        self.recent_window = int(recent_window)

    @classmethod
    def from_config(cls, config: SparseAttentionConfig) -> "SlidingWindowSparseAttention":
        return cls(sink_size=config.sink_size, recent_window=config.recent_window)

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
        del layer_idx, phase  # method is head/layer-uniform
        import torch

        if query_len < 1:
            raise ValueError(f"query_len must be >= 1; got {query_len!r}")
        if key_len < 0:
            raise ValueError(f"key_len must be >= 0; got {key_len!r}")

        # Build a 1D key-mask first (same sparsity set for every query row).
        keep = torch.zeros(key_len, dtype=torch.bool, device=device)
        sink = min(self.sink_size, key_len)
        if sink > 0:
            keep[:sink] = True
        recent_start = max(0, key_len - self.recent_window)
        if self.recent_window > 0:
            keep[recent_start:] = True

        neg_inf = torch.finfo(dtype).min
        key_mask = torch.where(
            keep,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), neg_inf, dtype=dtype, device=device),
        )

        if query_len == 1:
            # Decode: the single query is at absolute position key_len-1,
            # which is >= every k in [0, key_len), so causal is trivially
            # satisfied. Keep the cheaper key-uniform shape and let SDPA
            # broadcast across the query dimension.
            return key_mask.view(1, 1, 1, key_len)

        # Prefill: per-row causal upper-triangular cut. The q-th query row's
        # absolute position is `cached_len + q = (key_len - query_len) + q`;
        # any k strictly greater is a future token and must be -inf,
        # because setting kwargs["attention_mask"] non-None disables HF
        # SDPA's implicit causal.
        offset = key_len - query_len
        q_idx = torch.arange(query_len, device=device).view(query_len, 1)
        k_idx = torch.arange(key_len, device=device).view(1, key_len)
        causal_keep = k_idx <= (offset + q_idx)  # [Q, K] bool
        mask_2d = torch.where(
            causal_keep,
            key_mask.view(1, key_len).expand(query_len, key_len),
            torch.full((), neg_inf, dtype=dtype, device=device),
        )
        return mask_2d.view(1, 1, query_len, key_len)

    def kept_count(self, key_len: int) -> int:
        """Number of unmasked key positions for the given `key_len`."""
        if key_len <= 0:
            return 0
        sink = min(self.sink_size, key_len)
        recent_start = max(0, key_len - self.recent_window)
        # Union of [0, sink) and [recent_start, key_len)
        if recent_start <= sink:
            return key_len
        return sink + (key_len - recent_start)

    def record_metadata(
        self,
        *,
        layer_idx: int,
        phase: str,
        decode_step: int,
    ) -> dict[str, Any]:
        del layer_idx, phase, decode_step
        # sink_size / recent_window already live in attempt-level meta.json
        # (the `sparse_attention` block); duplicating them per row in
        # extras_json would be pure bloat. Return an empty dict so the
        # recorder serialises the literal "{}" string and the column stays
        # schema-stable.
        return {}
