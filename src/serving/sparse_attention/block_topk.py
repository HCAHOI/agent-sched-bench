"""Query-key block top-k sparse attention."""

from __future__ import annotations

from typing import Any

from serving.sparse_attention.base import SparseAttentionConfig, SparseAttentionContext
from serving.sparse_attention.patterns import (
    additive_mask_from_keep,
    block_id_for_position,
    cap_middle_selection,
    dense_or_causal_mask,
    middle_indices,
    sink_recent_keep_indices,
    validate_sparse_budget,
)
from serving.sparse_attention.state import (
    current_query_states,
    full_key_states_for_pre_hook,
)


class BlockTopKSparseAttention:
    """Decode-time sparse mask selected by current Q/K block scores."""

    name = "block_topk"
    observe_only: bool

    def __init__(
        self,
        *,
        budget: int,
        sink_size: int,
        recent_window: int,
        block_size: int,
        score_reduction: str = "max",
        phase_scope: str = "decode_only",
        observe_only: bool = False,
    ) -> None:
        self.budget = validate_sparse_budget(
            method_name=self.name,
            budget=budget,
            sink_size=sink_size,
            recent_window=recent_window,
            block_size=block_size,
            score_reduction=score_reduction,
            phase_scope=phase_scope,
        )
        self.sink_size = int(sink_size)
        self.recent_window = int(recent_window)
        self.block_size = int(block_size)
        self.score_reduction = str(score_reduction)
        self.phase_scope = str(phase_scope)
        self.observe_only = bool(observe_only)
        self._last_kept_count = 0
        self._last_metadata: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: SparseAttentionConfig) -> "BlockTopKSparseAttention":
        return cls(
            budget=int(config.budget) if config.budget is not None else None,  # type: ignore[arg-type]
            sink_size=config.sink_size,
            recent_window=config.recent_window,
            block_size=config.block_size,
            score_reduction=config.score_reduction,
            phase_scope=config.phase_scope,
            observe_only=config.observe_only,
        )

    def build_additive_mask(
        self,
        *,
        layer_idx: int,
        query_len: int,
        key_len: int,
        phase: str,
        decode_step: int = -1,
        device: "Any",
        dtype: "Any",
        context: SparseAttentionContext | None = None,
    ) -> "Any":
        del decode_step
        if key_len <= 0:
            self._record(kept=[], reason="empty")
            return dense_or_causal_mask(
                query_len=query_len, key_len=key_len, device=device, dtype=dtype
            )
        if self.phase_scope == "decode_only" and phase != "decode":
            self._record(kept=range(key_len), reason="phase_dense")
            return dense_or_causal_mask(
                query_len=query_len, key_len=key_len, device=device, dtype=dtype
            )
        if query_len != 1:
            self._record(kept=range(key_len), reason="prefill_dense")
            return dense_or_causal_mask(
                query_len=query_len, key_len=key_len, device=device, dtype=dtype
            )
        if context is None or context.position_embeddings is None:
            raise ValueError(
                "block_topk requires position_embeddings in decode; "
                "cannot silently fall back to dense attention"
            )

        ranked_middle, selected_blocks = self._rank_middle_positions(
            context=context,
            layer_idx=layer_idx,
            key_len=key_len,
        )
        selected_middle = cap_middle_selection(
            key_len=key_len,
            budget=self.budget,
            sink_size=self.sink_size,
            recent_window=self.recent_window,
            ranked_middle=ranked_middle,
        )
        keep = sink_recent_keep_indices(
            key_len=key_len,
            sink_size=self.sink_size,
            recent_window=self.recent_window,
        )
        keep.update(selected_middle)
        self._record(
            kept=keep,
            reason="selected",
            selected_blocks=selected_blocks,
            selected_middle=selected_middle,
        )
        return additive_mask_from_keep(
            keep_indices=keep,
            query_len=query_len,
            key_len=key_len,
            device=device,
            dtype=dtype,
        )

    def _rank_middle_positions(
        self, *, context: SparseAttentionContext, layer_idx: int, key_len: int
    ) -> tuple[list[int], list[int]]:
        import torch

        q_states = current_query_states(
            module=context.module,
            hidden_states=context.hidden_states,
            position_embeddings=context.position_embeddings,  # type: ignore[arg-type]
        )
        key_states = full_key_states_for_pre_hook(
            module=context.module,
            layer_idx=int(layer_idx),
            hidden_states=context.hidden_states,
            position_embeddings=context.position_embeddings,  # type: ignore[arg-type]
            past_key_values=context.past_key_values,
        )
        if int(key_states.shape[-2]) != int(key_len):
            raise ValueError(
                f"block_topk key_len mismatch: computed {int(key_states.shape[-2])}, "
                f"hook reported {key_len}"
            )
        n_kv_heads = int(key_states.shape[1])
        n_query_heads = int(q_states.shape[1])
        if n_query_heads % n_kv_heads != 0:
            raise ValueError(
                f"query heads must be divisible by kv heads: {n_query_heads} vs {n_kv_heads}"
            )
        groups = n_query_heads // n_kv_heads
        grouped_q = q_states.reshape(
            q_states.shape[0], n_kv_heads, groups, q_states.shape[-2], q_states.shape[-1]
        )
        scores = torch.matmul(grouped_q, key_states.unsqueeze(2).transpose(-1, -2))
        scores = scores.reshape(q_states.shape[0], n_query_heads, q_states.shape[-2], key_len)
        scores = scores * float(getattr(context.module, "scaling", q_states.shape[-1] ** -0.5))
        token_scores = scores.amax(dim=(0, 1, 2))
        candidates = middle_indices(
            key_len=key_len, sink_size=self.sink_size, recent_window=self.recent_window
        )
        if not candidates:
            return [], []

        block_to_positions: dict[int, list[int]] = {}
        for pos in candidates:
            block_to_positions.setdefault(block_id_for_position(pos, self.block_size), []).append(pos)

        block_scores: list[tuple[float, int]] = []
        for block_id, positions in block_to_positions.items():
            idx = torch.as_tensor(positions, dtype=torch.long, device=token_scores.device)
            values = token_scores.index_select(0, idx)
            value = values.mean() if self.score_reduction == "mean" else values.max()
            block_scores.append((float(value.detach().cpu().item()), int(block_id)))
        block_scores.sort(key=lambda item: (-item[0], item[1]))
        ranked_positions: list[int] = []
        selected_blocks: list[int] = []
        for _score, block_id in block_scores:
            selected_blocks.append(block_id)
            ranked_positions.extend(block_to_positions[block_id])
        return ranked_positions, selected_blocks

    def _record(
        self,
        *,
        kept: Any,
        reason: str,
        selected_blocks: list[int] | None = None,
        selected_middle: list[int] | None = None,
    ) -> None:
        kept_list = [int(x) for x in kept]
        self._last_kept_count = len(set(kept_list))
        self._last_metadata = {
            "budget": self.budget,
            "block_size": self.block_size,
            "score_reduction": self.score_reduction,
            "phase_scope": self.phase_scope,
            "selection_reason": reason,
            "selected_blocks": list(selected_blocks or []),
            "selected_middle_count": len(selected_middle or []),
            "selected_middle_indices": [int(x) for x in (selected_middle or [])],
        }

    def kept_count(self, key_len: int) -> int:
        del key_len
        return int(self._last_kept_count)

    def record_metadata(
        self,
        *,
        layer_idx: int,
        phase: str,
        decode_step: int,
    ) -> dict[str, Any]:
        del layer_idx, phase, decode_step
        return dict(self._last_metadata)


__all__ = ["BlockTopKSparseAttention"]
