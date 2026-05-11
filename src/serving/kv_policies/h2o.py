"""H2O (Heavy-Hitter Oracle) KV eviction policy.

Reference: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
Inference of Large Language Models" (NeurIPS 2023, https://arxiv.org/abs/2306.14048).

Algorithm sketch (paper §4)
---------------------------

For each layer, maintain a per-key-position cumulative attention score. When
`key_len > budget` triggers eviction, partition `[0, key_len)` into

    sink   := [0, sink_size)
    recent := [key_len - recent_window, key_len)
    middle := [sink_size, key_len - recent_window)

Always keep `sink ∪ recent` (matches StreamingLLM semantics for sink + local
context). From `middle`, keep the top `budget - sink_size - recent_window`
positions ranked by accumulated attention score; evict the rest. The
post-eviction layer length is exactly `budget`.

Design choices (locked here so reviewers can audit later H2O variants against
the same baseline contract).

A. **Score buffer shape — head-mean at `observe()` time, NOT per-head store.**
   The decision uses head-averaged scores (paper convention; cheap and
   well-behaved for GQA). We accumulate the head-mean directly in
   `observe()` instead of storing per-(layer, head, key_pos) and reducing at
   decision time. Algebraically equivalent because both sum and head-mean are
   linear; the per-head store would cost `num_kv_heads`x more memory (≈ 8x
   for Qwen3) for zero behavioural difference. The published `attn` from the
   bus carries the per-query-head dim already, so we mean over that axis.

B. **`config.aggregate` — only `sum` is implemented.**
   The paper's default is cumulative sum across all queries, prefill + decode
   sharing one accumulator. `mean` and `ema` raise `NotImplementedError` here
   so a future step can wire them in without leaving silent placeholder
   semantics on disk.

C. **Score buffer pre-allocation.**
   We pre-allocate a single `(max_position_embeddings,)` fp32 tensor per
   layer at first `observe()` (lazy on attn.device). Decode is hot — every
   step would otherwise grow the buffer; an upfront fp32 vector at 128k pos
   is ~512 KiB per layer (32 layers ≈ 16 MiB). If a downstream model exceeds
   `max_position_embeddings`, we raise — silent truncation would corrupt the
   eviction decisions invisibly.

D. **Device.**
   The score buffer lives on the same device as the published `attn` tensor.
   First `observe()` for a layer pins the device; subsequent calls assume it
   stays put. This avoids per-call `.to(...)` traffic on the decode path.

E. **Post-eviction score buffer compaction.**
   `_physically_drop` shrinks the K/V tensors by `keep_indices`; the score
   buffer must follow the same permutation or the next `observe()` will
   write into stale slots. We override `_post_evict_hook` (added to
   `BaseEvictionCache` for this purpose) and rebuild the prefix as
   `buffer[keep_indices]`. The base class fires the hook *after* the K/V
   drop succeeds.

Lifecycle invariants (plan §H4 + Top Risk #4)
---------------------------------------------

* `always_active = True` (class-level): even when `LayerCapturer.suspend_attention()`
  is gating the bus, H2O still observes. A silent skip would corrupt the
  score buffer relative to the cache state and the next eviction would pick
  stale heavy hitters.
* `requires_attention()` returns True so the provider knows it must wire the
  bus.
* The provider owns subscribe/unsubscribe lifecycle: H2O subscribes itself
  on construction, but the provider must call `attention_bus.unsubscribe(cache)`
  in a `finally` block around `model.generate(...)`. Letting subscriptions
  accumulate across calls would (a) leak references to dead caches and
  (b) cause double-observation if a cache from a prior call is still
  registered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from serving.kv_policies.base import (
    BaseEvictionCache,
    EvictionDecision,
    EvictionPolicyConfig,
)

if TYPE_CHECKING:
    from serving.kv_policies.recorder import KVEvictionRecorder
    from serving.recording.attention_bus import AttentionBus


class H2OCache(BaseEvictionCache):
    """`BaseEvictionCache` with heavy-hitter (cumulative attention) eviction."""

    # Class-level: always observe even while LayerCapturer is suspended (plan
    # Top Risk #4). Per-instance overrides are not supported; H2O semantics
    # break if any decision is made on a partial accumulator.
    always_active: bool = True

    def __init__(
        self,
        config: EvictionPolicyConfig,
        num_layers: int,
        recorder: "KVEvictionRecorder | None" = None,
        *,
        attention_bus: "AttentionBus",
        max_position_embeddings: int,
    ) -> None:
        super().__init__(config, num_layers, recorder=recorder)
        if config.budget is None or config.budget <= 0:
            raise ValueError(
                f"H2OCache requires positive config.budget; got {config.budget!r}"
            )
        if config.sink_size < 0:
            raise ValueError(
                f"H2OCache requires sink_size >= 0; got {config.sink_size!r}"
            )
        if config.recent_window < 0:
            raise ValueError(
                f"H2OCache requires recent_window >= 0; got {config.recent_window!r}"
            )
        floor = int(config.sink_size) + int(config.recent_window)
        if int(config.budget) < floor:
            raise ValueError(
                f"H2OCache requires budget >= sink_size + recent_window "
                f"({config.budget!r} < {config.sink_size!r} + {config.recent_window!r} "
                f"= {floor})"
            )
        if config.aggregate != "sum":
            # mean / ema are paper-side ablations; deliberately not silently
            # falling back to sum — that would alter the score semantics
            # without the user noticing.
            raise NotImplementedError(
                f"aggregate={config.aggregate!r} not yet implemented; "
                "use sum (paper default)"
            )
        if attention_bus is None:
            raise ValueError("H2OCache requires an attention_bus to subscribe to")
        if max_position_embeddings is None or int(max_position_embeddings) <= 0:
            raise ValueError(
                "H2OCache requires positive max_position_embeddings; "
                f"got {max_position_embeddings!r}"
            )
        self._max_pos = int(max_position_embeddings)
        # Lazy per-layer score buffers: `_scores[layer]` is None until the
        # first observe() pins device + dtype.
        self._scores: list[torch.Tensor | None] = [None] * int(num_layers)
        self._score_lengths: list[int] = [0] * int(num_layers)
        # Subscribe AFTER all validation succeeds so a constructor failure
        # cannot leave a half-built cache wired to the bus.
        self._attention_bus: "AttentionBus" = attention_bus
        attention_bus.subscribe(self)

    def requires_attention(self) -> bool:
        return True

    # -- AttentionConsumer protocol -----------------------------------------

    def observe(
        self,
        *,
        layer: int,
        attn: torch.Tensor,
        query_positions: torch.Tensor,
        key_len: int,
        phase: str,
    ) -> None:
        """Accumulate head-mean attention into the per-layer score buffer.

        Shape contract from `AttentionBus`: `attn` is
        `(B, num_q_heads, n_query_rows, key_len)`. We sum across query rows
        and batch (each query contributes its full softmax row), then mean
        across heads to get a `(key_len,)` vector that we add into the
        running buffer at slots `[0:key_len]`.
        """
        del query_positions, phase  # unused: head-mean over heads; positions
        # already implicit in the key axis.
        if attn.ndim != 4:
            raise ValueError(
                f"H2OCache.observe expected 4-D attn (B,H,Q,K); got {tuple(attn.shape)}"
            )
        actual_key_len = int(attn.shape[-1])
        if actual_key_len != int(key_len):
            raise ValueError(
                "H2OCache.observe: attn key axis "
                f"({actual_key_len}) != published key_len ({key_len})"
            )
        if int(layer) < 0 or int(layer) >= self.num_layers:
            raise ValueError(
                f"H2OCache.observe: layer {layer} outside [0, {self.num_layers})"
            )
        if actual_key_len > self._max_pos:
            raise ValueError(
                f"H2OCache.observe: key_len {actual_key_len} exceeds "
                f"max_position_embeddings {self._max_pos}; bump the config "
                "or pick a smaller context"
            )

        # Sum across query-row axis (each query contributes its softmax row)
        # then sum across batch, then mean across heads. Order of reductions
        # is irrelevant (linear) but doing query-sum first matches the paper
        # cumulative-attention definition more naturally.
        per_head_per_key = attn.sum(dim=2).sum(dim=0)  # (H, K)
        per_key = per_head_per_key.mean(dim=0)  # (K,) — head-mean (choice A)

        buffer = self._scores[int(layer)]
        if buffer is None:
            # Choice C+D: pre-allocate fp32 buffer on the attn device on
            # first sight. fp32 keeps numerics stable across the cumulative
            # sum even when attn arrives in bf16/fp16.
            buffer = torch.zeros(self._max_pos, dtype=torch.float32, device=attn.device)
            self._scores[int(layer)] = buffer

        # Cast to fp32 in case attn is bf16/fp16; in-place add into the live
        # slots only.
        buffer[:actual_key_len].add_(per_key.to(dtype=buffer.dtype))
        if actual_key_len > self._score_lengths[int(layer)]:
            self._score_lengths[int(layer)] = actual_key_len

    # -- Eviction logic ------------------------------------------------------

    def _decide_evict(self, layer_idx: int, key_len: int) -> EvictionDecision:
        budget = int(self.config.budget)  # type: ignore[arg-type]
        if key_len <= budget:
            return EvictionDecision(
                keep_indices=list(range(key_len)),
                evict_indices=[],
                reason="none",
            )
        sink = int(self.config.sink_size)
        recent = int(self.config.recent_window)
        # Defensive: caller-side validation guarantees budget >= sink+recent,
        # so middle window is always non-empty when budget < key_len. But pin
        # the invariant explicitly so future config changes can't sneak past.
        if sink + recent > key_len:
            raise RuntimeError(
                f"H2OCache: sink({sink}) + recent({recent}) > key_len({key_len}); "
                "cannot partition without overlap"
            )
        n_heavy = budget - sink - recent
        middle_start = sink
        middle_end = key_len - recent
        middle_len = middle_end - middle_start
        if n_heavy < 0:
            raise RuntimeError(
                f"H2OCache: heavy_count budget({budget}) - sink({sink}) - recent({recent}) "
                f"< 0; constructor invariant breached"
            )

        buffer = self._scores[int(layer_idx)]
        # Order-of-operations note: in HF generate, the layer's
        # `past_key_values.update(...)` runs BEFORE softmax/capturer hook for
        # the same forward step. So at decision time the score buffer
        # reflects scores accumulated through the *previous* forward; the
        # newly-appended positions in `[key_len_prev, key_len)` have no
        # score yet. This is fine: they sit inside the recent window
        # (decode appends one token; recent_window >= 1 ensures the new
        # token is recent-kept and not eligible for heavy-hitter ranking).
        # Pre-condition we DO need: middle window must be covered by the
        # score buffer. If `_score_lengths[layer] < middle_end`, the bus
        # wiring is broken or the recent_window is too small relative to
        # how many new tokens this forward appends.
        if buffer is None:
            raise RuntimeError(
                f"H2OCache._decide_evict: no score buffer for layer {layer_idx}; "
                "AttentionBus.publish never reached this consumer — check that "
                "the cache is subscribed and LayerCapturer is publishing."
            )
        if self._score_lengths[int(layer_idx)] < middle_end:
            raise RuntimeError(
                f"H2OCache._decide_evict: score buffer length "
                f"{self._score_lengths[int(layer_idx)]} < middle_end "
                f"{middle_end} for layer {layer_idx}; observe() did not "
                "cover the heavy-hitter window. Either the bus subscription "
                "is broken or recent_window is too small for the per-forward "
                "query_len."
            )

        # Pick top-`n_heavy` positions in the middle window by accumulated
        # score. `topk` returns descending; sort the absolute indices for a
        # stable keep set.
        middle_scores = buffer[middle_start:middle_end]
        # n_heavy may be 0 (budget == sink + recent); topk(0) is degenerate
        # but valid in torch.
        n_heavy_eff = min(n_heavy, middle_len)
        if n_heavy_eff > 0:
            top_values, top_local_idx = torch.topk(middle_scores, n_heavy_eff)
            heavy_indices_abs = (top_local_idx + middle_start).tolist()
            score_topk_value = [float(v) for v in top_values.detach().cpu().tolist()]
        else:
            heavy_indices_abs = []
            score_topk_value = []
        score_topk_index = list(heavy_indices_abs)

        keep_set = set(range(0, sink))
        keep_set.update(heavy_indices_abs)
        keep_set.update(range(middle_end, key_len))
        keep = sorted(keep_set)
        evict = sorted(set(range(key_len)) - keep_set)

        return EvictionDecision(
            keep_indices=keep,
            evict_indices=evict,
            reason="over_budget",
            policy_state={
                "score_topk_index": score_topk_index,
                "score_topk_value": score_topk_value,
            },
        )

    # -- Post-eviction state compaction (choice E) --------------------------

    def _post_evict_hook(self, layer_idx: int, decision: EvictionDecision) -> None:
        """Compact the score buffer to mirror the K/V drop.

        Without this, the next `observe()` would `.add_()` into stale slots
        — old positions would remain in the buffer past the K/V boundary
        and the eviction decision after the next over-budget step would
        ghost-keep evicted positions.
        """
        buffer = self._scores[int(layer_idx)]
        if buffer is None:
            return
        keep = decision.keep_indices
        n_keep = len(keep)
        if n_keep == 0:
            buffer.zero_()
            self._score_lengths[int(layer_idx)] = 0
            return
        index = torch.as_tensor(keep, dtype=torch.long, device=buffer.device)
        # Read the live prefix into a fresh permutation; `index_select` is
        # safe vs aliasing where in-place gather would not be. Then scatter
        # back into the head of the same buffer and zero the tail.
        compact = buffer.index_select(0, index)
        buffer.zero_()
        buffer[:n_keep] = compact
        self._score_lengths[int(layer_idx)] = n_keep
