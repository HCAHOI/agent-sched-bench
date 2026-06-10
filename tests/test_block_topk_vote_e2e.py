"""Torch-gated end-to-end tests for block_topk vote reduction + per-head export.

Pure-logic coverage (CSR, vote ranking via the analysis re-aggregator, config
validation) is in tests/test_per_head_topk_csr.py and runs on the torch-free
local .venv. These tests exercise the actual torch scoring path and so are
skipped where torch is unavailable (run on the GPU box).

The toy attention has 2 query heads, 1 KV head (head_dim=2): each head reads a
distinct 2-dim slice of the key, so per-head block preferences diverge by
construction — the property vote and per-head export are meant to surface.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from serving.sparse_attention.base import SparseAttentionContext
from serving.sparse_attention.block_topk import BlockTopKSparseAttention


class _TwoHeadAttention(nn.Module):
    """2 query heads, 1 KV head, head_dim=2 (identity projections)."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_idx = 0
        self.head_dim = 2
        self.scaling = 1.0
        self.q_proj = nn.Linear(4, 4, bias=False)  # 2 query heads * head_dim 2
        self.k_proj = nn.Linear(4, 2, bias=False)  # 1 KV head * head_dim 2
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(4))
            self.k_proj.weight.copy_(torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            ))


def _context(
    *, cached_keys: torch.Tensor, hidden: torch.Tensor, module: nn.Module | None = None
) -> SparseAttentionContext:
    # RoPE cos/sin over head_dim=2; identity rotation (cos=1, sin=0).
    cos = torch.ones((1, hidden.shape[-2], 2), dtype=hidden.dtype)
    sin = torch.zeros_like(cos)

    class _Cache:
        def __getitem__(self, layer_idx: int):
            if layer_idx != 0:
                raise KeyError(layer_idx)
            return cached_keys, None

        def get_seq_length(self, layer_idx: int = 0) -> int:
            if layer_idx != 0:
                raise KeyError(layer_idx)
            return int(cached_keys.shape[-2])

    return SparseAttentionContext(
        module=module if module is not None else _TwoHeadAttention(),
        hidden_states=hidden,
        position_embeddings=(cos, sin),
        past_key_values=_Cache(),
        attention_mask=None,
    )


def _kept(mask: torch.Tensor) -> set[int]:
    return {int(i) for i in torch.nonzero(mask[0, 0, 0] == 0.0).flatten().tolist()}


def _decode_mask(
    method: BlockTopKSparseAttention,
    cached: torch.Tensor,
    hidden: torch.Tensor,
    key_len: int,
    module: nn.Module | None = None,
):
    return method.build_additive_mask(
        layer_idx=0,
        query_len=1,
        key_len=key_len,
        phase="decode",
        decode_step=0,
        device=hidden.device,
        dtype=hidden.dtype,
        context=_context(cached_keys=cached, hidden=hidden, module=module),
    )


def test_vote_reduction_produces_valid_keep_set_and_metadata() -> None:
    # 8 cached keys (4 blocks of size 2) + 1 current = key_len 9.
    # Head 0 (dim 0) prefers block 3 (pos 6,7); head 1 (dim 1) prefers block 1.
    # hidden=[1,0,0,1] -> q_head0=[1,0] (reads key dim 0), q_head1=[0,1] (key dim 1).
    cached = torch.zeros((1, 1, 8, 2), dtype=torch.float32)
    cached[0, 0, 6, 0] = 9.0  # head0 -> block 3
    cached[0, 0, 2, 1] = 9.0  # head1 -> block 1
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]])

    method = BlockTopKSparseAttention(
        budget=6, sink_size=1, recent_window=1, block_size=2, score_reduction="vote"
    )
    mask = _decode_mask(method, cached, hidden, 9)
    assert mask.shape == (1, 1, 1, 9)
    kept = _kept(mask)
    # sink {0} + recent {8} always; middle budget = 6 - 2 = 4 positions.
    assert 0 in kept and 8 in kept
    assert len(kept) == 6
    meta = method.record_metadata(layer_idx=0, phase="decode", decode_step=0)
    assert meta["score_reduction"] == "vote"
    assert "vote_summary" in meta
    vs = meta["vote_summary"]
    assert vs["vote_top_b"] >= 1 and vs["n_candidate_blocks"] >= 1
    # Both heads' favored middle blocks (1 and 3) must be retained.
    assert {2, 3} <= kept  # block 1 -> positions {2,3}
    assert {6, 7} <= kept  # block 3 -> positions {6,7}


def test_vote_ranking_is_deterministic_across_runs() -> None:
    cached = torch.zeros((1, 1, 8, 2), dtype=torch.float32)
    cached[0, 0, 6, 0] = 9.0
    cached[0, 0, 2, 1] = 9.0
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]])
    method = BlockTopKSparseAttention(
        budget=6, sink_size=1, recent_window=1, block_size=2, score_reduction="vote"
    )
    _decode_mask(method, cached, hidden, 9)
    meta_a = dict(method.record_metadata(layer_idx=0, phase="decode", decode_step=0))
    _decode_mask(method, cached, hidden, 9)
    meta_b = dict(method.record_metadata(layer_idx=0, phase="decode", decode_step=0))
    assert meta_a["selected_blocks"] == meta_b["selected_blocks"]


def test_per_head_export_shape_and_ordering() -> None:
    cached = torch.zeros((1, 1, 8, 2), dtype=torch.float32)
    cached[0, 0, 6, 0] = 9.0
    cached[0, 0, 2, 1] = 9.0
    cached[0, 0, 4, 0] = 5.0
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]])

    method = BlockTopKSparseAttention(
        budget=6, sink_size=1, recent_window=1, block_size=2, score_reduction="max"
    )
    # Off by default: no export even though we run a decode.
    _decode_mask(method, cached, hidden, 9)
    assert method.per_head_topk_export() is None

    # Switch on (rank 3) — export must appear regardless of score_reduction=max.
    method.export_per_head_topk_rank = 3
    _decode_mask(method, cached, hidden, 9)
    export = method.per_head_topk_export()
    assert export is not None
    block_ids = export["block_ids"]
    scores = export["scores"]
    assert len(block_ids) == 2 and len(scores) == 2  # 2 query heads
    # Middle positions 1..7 (key_len 9, sink1 recent1) span blocks {0,1,2,3}:
    # pos 1 falls in block 0, so 4 candidate blocks total.
    n_candidates = 4
    for h in range(2):
        ids_h = block_ids[h]
        sc_h = scores[h]
        assert len(ids_h) == len(sc_h)
        assert len(ids_h) <= min(3, n_candidates)
        # scores descending; block ids within the middle candidate range.
        assert sc_h == sorted(sc_h, reverse=True)
        assert all(0 <= b <= 3 for b in ids_h)
    # Head 0's top block is 3 (pos 6 high on dim 0); head 1's is 1 (pos 2 dim 1).
    assert block_ids[0][0] == 3
    assert block_ids[1][0] == 1


def test_export_none_when_no_middle_candidates() -> None:
    # sink(2) + recent(3) cover all of key_len 5: middle_indices is empty, so the
    # per-head export is None even with the switch on (no candidate blocks exist).
    cached = torch.zeros((1, 1, 4, 2), dtype=torch.float32)
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]])
    method = BlockTopKSparseAttention(
        budget=5, sink_size=2, recent_window=3, block_size=2, score_reduction="vote"
    )
    method.export_per_head_topk_rank = 4
    _decode_mask(method, cached, hidden, 5)
    assert method.per_head_topk_export() is None


class _FourHeadAttention(nn.Module):
    """4 query heads, 1 KV head, head_dim=2 (identity projections)."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_idx = 0
        self.head_dim = 2
        self.scaling = 1.0
        self.q_proj = nn.Linear(8, 8, bias=False)  # 4 query heads * head_dim 2
        self.k_proj = nn.Linear(8, 2, bias=False)  # 1 KV head * head_dim 2
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(8))
            self.k_proj.weight.zero_()


def test_vote_differs_from_max_when_candidates_exceed_block_budget() -> None:
    """Regression for the position-vs-block vote budget bug.

    With nb (candidate blocks) > B (block budget), a 3-head consensus block must
    out-vote a single-head outlier that max aggregation ranks first, and no
    block may collect a vote from every head — the saturated-vote state in
    which vote silently degenerates to max.
    """
    # 12 cached keys (6 blocks of size 2) + 1 current = key_len 13.
    # sink=1, recent=1, budget=6 -> middle_slots=4 positions -> B=2 blocks;
    # candidate middle blocks cover positions 1..11 -> nb=6 > B.
    cached = torch.zeros((1, 1, 12, 2), dtype=torch.float32)
    cached[0, 0, 10, 0] = 10.0  # block 5: head0's huge outlier (key dim 0)
    cached[0, 0, 4, 1] = 3.0    # block 2: heads 1-3 consensus (key dim 1)
    cached[0, 0, 6, 1] = 2.0    # block 3: heads 1-3 second choice
    # head0 q=[1,0] reads key dim 0; heads 1-3 q=[0,1] read key dim 1.
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]])

    top2: dict[str, list[int]] = {}
    vote_summary: dict[str, int] = {}
    for reduction in ("max", "vote"):
        method = BlockTopKSparseAttention(
            budget=6, sink_size=1, recent_window=1, block_size=2,
            score_reduction=reduction,
        )
        _decode_mask(method, cached, hidden, 13, module=_FourHeadAttention())
        meta = method.record_metadata(layer_idx=0, phase="decode", decode_step=0)
        top2[reduction] = [int(b) for b in meta["selected_blocks"][:2]]
        if reduction == "vote":
            vote_summary = dict(meta["vote_summary"])

    # max ranks the outlier first; vote ranks the 3-vote consensus pair first.
    assert top2["max"][0] == 5
    assert top2["vote"] == [2, 3]
    assert top2["vote"] != top2["max"]
    # Votes must discriminate: budget B=2 < nb, and no block holds all 4 votes.
    assert vote_summary["vote_top_b"] == 2
    assert vote_summary["n_candidate_blocks"] > vote_summary["vote_top_b"]
    assert vote_summary["max_votes"] == 3  # consensus trio, strictly < 4 heads
