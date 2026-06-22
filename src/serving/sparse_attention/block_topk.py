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
    # Scores are recomputed at decode from current Q vs full cached + delta K,
    # so cache reuse is safe — the K state seen at decode is identical whether
    # we re-prefilled the full prompt or resumed from a strict-prefix delta.
    requires_full_prefill = False

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
        # Staged by _rank_middle_positions; consumed by recording hook.
        self._last_per_head_topk: dict[str, Any] | None = None
        # Vote distribution from last vote-reduction ranking; None for max/mean.
        self._last_vote_summary: dict[str, int] | None = None
        # Set by recording hook; > 0 triggers per-head [H, Nb] export each decode step.
        self.export_per_head_topk_rank: int = 0

    @classmethod
    def from_config(cls, config: SparseAttentionConfig) -> "BlockTopKSparseAttention":
        if config.budget is None:
            raise ValueError(
                f"{cls.__name__} requires SparseAttentionConfig.budget to be set "
                f"(method={config.name!r})"
            )
        return cls(
            budget=int(config.budget),
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
            if self.observe_only:
                return None
            return dense_or_causal_mask(
                query_len=query_len, key_len=key_len, device=device, dtype=dtype
            )
        if self.phase_scope == "decode_only" and phase != "decode":
            self._record(kept=range(key_len), reason="phase_dense")
            if self.observe_only:
                return None
            return dense_or_causal_mask(
                query_len=query_len, key_len=key_len, device=device, dtype=dtype
            )
        if query_len != 1:
            self._record(kept=range(key_len), reason="prefill_dense")
            if self.observe_only:
                return None
            return dense_or_causal_mask(
                query_len=query_len, key_len=key_len, device=device, dtype=dtype
            )
        if context is None or context.position_embeddings is None:
            raise ValueError(
                "block_topk requires position_embeddings in decode; "
                "cannot silently fall back to dense attention"
            )

        keep = sink_recent_keep_indices(
            key_len=key_len,
            sink_size=self.sink_size,
            recent_window=self.recent_window,
        )
        # Middle budget in token positions: how many middle slots this step can
        # fill after sink/recent claim their share. The vote path converts this
        # to a BLOCK budget (ceil(middle_slots / block_size)) at its call site.
        middle_slots = max(0, min(self.budget, key_len) - len(keep))
        ranked_middle, selected_blocks = self._rank_middle_positions(
            context=context,
            layer_idx=layer_idx,
            key_len=key_len,
            middle_slots=middle_slots,
        )
        selected_middle = cap_middle_selection(
            key_len=key_len,
            budget=self.budget,
            sink_size=self.sink_size,
            recent_window=self.recent_window,
            ranked_middle=ranked_middle,
        )
        keep.update(selected_middle)
        self._record(
            kept=keep,
            reason="selected",
            selected_blocks=selected_blocks,
            selected_middle=selected_middle,
        )
        if self.observe_only:
            return None
        return additive_mask_from_keep(
            keep_indices=keep,
            query_len=query_len,
            key_len=key_len,
            device=device,
            dtype=dtype,
        )

    def _rank_middle_positions(
        self,
        *,
        context: SparseAttentionContext,
        layer_idx: int,
        key_len: int,
        middle_slots: int,
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
        candidates = middle_indices(
            key_len=key_len, sink_size=self.sink_size, recent_window=self.recent_window
        )
        if not candidates:
            self._last_per_head_topk = None
            self._last_vote_summary = None
            return [], []

        # Build block_to_positions on CPU (needed for the final ranked_positions list)
        block_to_positions: dict[int, list[int]] = {}
        for pos in candidates:
            block_to_positions.setdefault(
                block_id_for_position(pos, self.block_size), []
            ).append(pos)

        candidates_t = torch.as_tensor(
            candidates, dtype=torch.long, device=scores.device
        )
        block_ids_t = candidates_t // int(self.block_size)               # [Nc]
        unique_blocks, inverse = torch.unique(
            block_ids_t, return_inverse=True
        )  # [Nb], [Nc]
        nb = int(unique_blocks.shape[0])

        need_per_head = self.score_reduction == "vote" or self.export_per_head_topk_rank > 0
        per_head_block_scores: Any = None
        cross_head_max_block_scores: Any = None
        if need_per_head:
            # token scores per head: fold batch + q_len with amax -> [H, K].
            token_scores_h = scores.amax(dim=(0, 2))                     # [H, K]
            cand_scores_h = token_scores_h.index_select(1, candidates_t)  # [H, Nc]
            per_head_block_scores = self._scatter_block_amax(
                cand_scores_h, inverse, nb
            )  # [H, Nb]
            cross_head_max_block_scores = per_head_block_scores.amax(dim=0)  # [Nb]

        if self.export_per_head_topk_rank > 0:
            self._last_per_head_topk = self._build_per_head_topk(
                per_head_block_scores=per_head_block_scores,
                unique_blocks=unique_blocks,
            )
        else:
            self._last_per_head_topk = None

        if self.score_reduction == "vote":
            # Vote budget is in BLOCKS: ceil(middle position slots / block_size)
            # = the number of middle blocks the final selection can keep. Raw
            # position slots would exceed the candidate count, let every head
            # vote for everything, and collapse vote onto the max tie-break.
            ordered = self._vote_order(
                per_head_block_scores=per_head_block_scores,
                cross_head_max_block_scores=cross_head_max_block_scores,
                unique_blocks=unique_blocks,
                top_b=-(-middle_slots // self.block_size),
            )
        else:
            if self.score_reduction == "max" and cross_head_max_block_scores is not None:
                block_scores_t = cross_head_max_block_scores
            else:
                cand_scores = scores.amax(dim=(0, 1, 2)).index_select(0, candidates_t)
                block_scores_t = self._reduce_block_scores(cand_scores, inverse, nb)
            scores_cpu = block_scores_t.detach().cpu().tolist()
            blocks_cpu = unique_blocks.detach().cpu().tolist()
            # Preserve tie-break: sort by score descending, block_id ascending on ties.
            ordered = sorted(
                zip(scores_cpu, blocks_cpu), key=lambda item: (-item[0], item[1])
            )
            ordered = [int(block_id) for _score, block_id in ordered]

        ranked_positions: list[int] = []
        selected_blocks: list[int] = []
        for block_id in ordered:
            selected_blocks.append(int(block_id))
            ranked_positions.extend(block_to_positions[int(block_id)])
        return ranked_positions, selected_blocks

    def _reduce_block_scores(self, cand_scores: Any, inverse: Any, nb: int) -> Any:
        """Reduce candidate token scores into per-block scores (max | mean)."""
        import torch

        if self.score_reduction == "max":
            block_scores_t = torch.full(
                (nb,), float("-inf"), device=cand_scores.device, dtype=cand_scores.dtype
            )
            return block_scores_t.scatter_reduce(
                0, inverse, cand_scores, reduce="amax", include_self=True
            )
        sums = torch.zeros((nb,), device=cand_scores.device, dtype=cand_scores.dtype)
        counts = torch.zeros((nb,), device=cand_scores.device, dtype=cand_scores.dtype)
        sums = sums.scatter_add(0, inverse, cand_scores)
        counts = counts.scatter_add(0, inverse, torch.ones_like(cand_scores))
        return sums / counts

    @staticmethod
    def _scatter_block_amax(cand_scores_h: Any, inverse: Any, nb: int) -> Any:
        """Per-head block amax: [H, Nc] candidate scores -> [H, Nb] block scores.

        Head dim is batched (no per-head Python loop); the same `inverse`
        (candidate -> unique-block index) maps every head's candidates in one
        scatter_reduce over the last axis.
        """
        import torch

        h = int(cand_scores_h.shape[0])
        index = inverse.unsqueeze(0).expand(h, -1)                       # [H, Nc]
        out = torch.full(
            (h, nb), float("-inf"), device=cand_scores_h.device, dtype=cand_scores_h.dtype
        )
        return out.scatter_reduce(1, index, cand_scores_h, reduce="amax", include_self=True)

    def _vote_order(
        self,
        *,
        per_head_block_scores: Any,
        cross_head_max_block_scores: Any,
        unique_blocks: Any,
        top_b: int,
    ) -> list[int]:
        """Rank blocks by cross-head votes.

        Each head votes for its own top-`top_b` candidate blocks; a block's vote
        count is the primary key. Ties break by cross-head max score (desc) then
        block_id (asc) for determinism. `top_b` is this step's middle budget in
        BLOCKS (ceil of position slots / block_size, capped to the candidate
        count below) — a head never votes for more blocks than the selection
        could keep. That cap is what gives votes discriminative power: with
        top_b >= nb every head votes for everything and the ranking degenerates
        to the tie-break.
        """
        import torch

        nb = int(per_head_block_scores.shape[1])
        b = max(0, min(int(top_b), nb))
        if b == 0:
            votes_t = torch.zeros(nb, dtype=torch.long, device=per_head_block_scores.device)
        else:
            # Per head, indices of its top-b blocks; scatter +1 vote each. One
            # bincount over the flattened [H*b] index set — no per-head sync.
            top_idx = per_head_block_scores.topk(b, dim=1).indices            # [H, b]
            votes_t = torch.bincount(top_idx.reshape(-1), minlength=nb)        # [Nb]
        votes_cpu = votes_t.detach().cpu().tolist()
        max_cpu = cross_head_max_block_scores.detach().cpu().tolist()
        blocks_cpu = unique_blocks.detach().cpu().tolist()
        ordered = sorted(
            zip(votes_cpu, max_cpu, blocks_cpu),
            key=lambda item: (-item[0], -item[1], item[2]),
        )
        nonzero = [v for v in votes_cpu if v > 0]
        self._last_vote_summary = {
            "n_candidate_blocks": int(nb),
            "vote_top_b": int(b),
            "blocks_with_votes": len(nonzero),
            "max_votes": max(votes_cpu) if votes_cpu else 0,
            # True when every head votes for every candidate (b >= nb): votes
            # carry no signal and the order falls back to the max tie-break.
            # Legitimate when candidates are scarcer than the budget; an offline
            # check on this flag catches any future config that re-saturates.
            "saturated": bool(b >= nb),
        }
        return [int(block_id) for _votes, _score, block_id in ordered]

    def _build_per_head_topk(
        self, *, per_head_block_scores: Any, unique_blocks: Any
    ) -> dict[str, list[list[int]] | list[list[float]]] | None:
        """Export per-head top-R (block_id, score) for offline counterfactuals.

        Returns ragged python lists (one inner list per head) so the recording
        hook can CSR-encode them; block ids map through `unique_blocks` back to
        absolute block ids. R is capped at the candidate-block count. One D2H of
        the [H, R] topk indices/values; no per-head loop on device.
        """
        if per_head_block_scores is None:
            return None
        h, nb = per_head_block_scores.shape
        r = max(0, min(int(self.export_per_head_topk_rank), int(nb)))
        if r == 0:
            return {"block_ids": [[] for _ in range(int(h))], "scores": [[] for _ in range(int(h))]}
        top = per_head_block_scores.topk(r, dim=1)                       # values/indices [H, R]
        abs_blocks = unique_blocks.index_select(0, top.indices.reshape(-1)).reshape(int(h), r)
        block_ids = abs_blocks.detach().cpu().tolist()
        scores = top.values.detach().cpu().tolist()
        return {"block_ids": block_ids, "scores": scores}

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
        # selected_blocks_kept: selected_blocks order (score rank) filtered to
        # blocks that have at least one position in selected_middle. This is the
        # correct input for the block-stats accumulator — selected_blocks may
        # contain more blocks than cap_middle_selection actually retained (budget
        # truncation), so using raw selected_blocks would include unretained keys.
        middle_set = set(int(p) for p in (selected_middle or []))
        kept_block_ids: set[int] = set()
        for p in middle_set:
            kept_block_ids.add(block_id_for_position(p, self.block_size))
        selected_blocks_kept = [
            b for b in (selected_blocks or []) if b in kept_block_ids
        ]
        self._last_metadata = {
            "budget": self.budget,
            "block_size": self.block_size,
            "score_reduction": self.score_reduction,
            "phase_scope": self.phase_scope,
            "selection_reason": reason,
            "selected_blocks": list(selected_blocks or []),
            "selected_blocks_kept": selected_blocks_kept,
            "selected_middle_count": len(selected_middle or []),
            "selected_middle_indices": [int(x) for x in (selected_middle or [])],
        }
        if self.score_reduction == "vote" and self._last_vote_summary is not None:
            self._last_metadata["vote_summary"] = dict(self._last_vote_summary)

    def reset_state(self) -> None:
        """No-op: block_topk computes its keep set fresh each decode step."""

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

    def per_head_topk_export(self) -> dict[str, Any] | None:
        """Per-head top-R (block_id, score) from the last decode ranking.

        Returns None when `export_per_head_topk_rank == 0` or no middle
        candidates existed. The recording hook reads this right after
        `build_additive_mask` and CSR-encodes it; block ids are absolute.
        """
        return self._last_per_head_topk


__all__ = ["BlockTopKSparseAttention"]
