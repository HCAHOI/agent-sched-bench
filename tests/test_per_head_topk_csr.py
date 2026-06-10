"""Pure-numpy tests for per-head-topk CSR encode/decode + analysis helpers.

No torch: these exercise the recording CSR encoder, the loader's CSR row layout,
and the analysis script's pure-python metrics. The torch end-to-end path (vote
selection, per-head export shape) lives in tests/test_block_topk_vote_e2e.py and
is GPU-gated.
"""

from __future__ import annotations

import numpy as np

from serving.recording.hooks import LayerCapturer, _encode_ragged_csr
from serving.recording.recording import RecordingConfig


def test_build_save_load_per_head_topk_roundtrip(tmp_path) -> None:
    """Build arrays -> npz -> loader, exercising the ragged CSR (layer, step,
    head) layout with ragged per-layer T (no torch / no model)."""
    from scripts.recoding_figures.recording_loader import load_per_head_topk

    class _Stub:
        config = RecordingConfig(
            per_head_stats_layers=(0, 4),
            record_per_head_topk=True,
            per_head_topk_rank=3,
        )
        # layer 0: steps {0,1}; layer 4: step {0} -> ragged T padded with -1/empty.
        _per_head_topk_cache = {
            (0, 0): {"block_ids": [[5, 2], [9]], "scores": [[0.9, 0.5], [0.7]]},
            (0, 1): {"block_ids": [[3], []], "scores": [[0.4], []]},
            (4, 0): {
                "block_ids": [[1, 2, 3], [8, 7, 6]],
                "scores": [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]],
            },
        }

    arrays = LayerCapturer._build_per_head_topk_arrays(_Stub())
    iter_dir = tmp_path / "iter_0000"
    iter_dir.mkdir()
    np.savez_compressed(iter_dir / "attention.npz", **arrays)
    got = load_per_head_topk(iter_dir)

    assert got["per_head_topk_layers"].tolist() == [0, 4]
    assert got["per_head_topk_rank"] == 3 and got["per_head_topk_head_count"] == 2
    assert got["per_head_topk_decode_step"].tolist() == [[0, 1], [0, -1]]
    assert got["per_head_topk_decode_n"].tolist() == [2, 1]
    assert got["per_head_topk_n_candidate_blocks"].tolist() == [[2, 1], [3, 0]]

    off = got["per_head_topk_csr_offsets"]
    ids = got["per_head_topk_csr_block_ids"]
    L_s, T_max = got["per_head_topk_decode_step"].shape
    H = got["per_head_topk_head_count"]
    assert off.shape[0] - 1 == L_s * T_max * H

    def _row(li: int, ti: int, h: int) -> list[int]:
        r = (li * T_max + ti) * H + h
        return ids[off[r] : off[r + 1]].tolist()

    assert _row(0, 0, 0) == [5, 2] and _row(0, 0, 1) == [9]
    assert _row(0, 1, 0) == [3] and _row(0, 1, 1) == []  # padded empty head row
    assert _row(1, 0, 0) == [1, 2, 3] and _row(1, 0, 1) == [8, 7, 6]


def test_build_per_head_topk_disabled_is_shape_stable() -> None:
    """Disabled -> 0-size leading axes (mirrors empty head_span convention)."""

    class _Stub:
        config = RecordingConfig(per_head_stats_layers=(0, 4))  # record off
        _per_head_topk_cache: dict = {}

    arrays = LayerCapturer._build_per_head_topk_arrays(_Stub())
    assert arrays["per_head_topk_layers"].shape == (0,)
    assert arrays["per_head_topk_csr_offsets"].tolist() == [0]
    assert arrays["per_head_topk_csr_block_ids"].shape == (0,)
    assert int(arrays["per_head_topk_rank"]) == 0


def test_ragged_csr_roundtrip() -> None:
    rows = [
        np.array([5, 2, 9], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([7], dtype=np.int32),
    ]
    scores = [
        np.array([0.9, 0.5, 0.1], dtype=np.float16),
        np.array([], dtype=np.float16),
        np.array([0.3], dtype=np.float16),
    ]
    offsets, idx, sc = _encode_ragged_csr(rows, scores)
    assert offsets.tolist() == [0, 3, 3, 4]
    assert idx.dtype == np.int32 and sc.dtype == np.float16
    # Each row decodes back exactly (block ids order preserved).
    assert idx[offsets[0] : offsets[1]].tolist() == [5, 2, 9]
    assert idx[offsets[1] : offsets[2]].tolist() == []
    assert idx[offsets[2] : offsets[3]].tolist() == [7]


def test_ragged_csr_all_empty() -> None:
    offsets, idx, sc = _encode_ragged_csr(
        [np.zeros(0, dtype=np.int32)] * 3, [np.zeros(0, dtype=np.float16)] * 3
    )
    assert offsets.tolist() == [0, 0, 0, 0]
    assert idx.shape == (0,) and sc.shape == (0,)


def test_ragged_csr_length_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        _encode_ragged_csr([np.zeros(1, dtype=np.int32)], [])


def test_analysis_helpers_jaccard_n90_reaggregate() -> None:
    from scripts.recoding_figures.analyze_per_head_topk import (
        StepSelections,
        _jaccard,
        _n90,
        _reaggregate,
        _topk_set,
    )

    assert _jaccard({1, 2, 3}, {2, 3, 4}) == 2 / 4
    assert np.isnan(_jaccard(set(), set()))

    # n90 over softmax(scores): one dominant block -> 1 covers >=90%.
    assert _n90(np.array([10.0, 0.0, 0.0], dtype=np.float16)) == 1
    # uniform 4 -> need >= ceil(0.9*4)=4 blocks (each 25%).
    assert _n90(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float16)) == 4
    assert _n90(np.zeros(0, dtype=np.float16)) == 0

    assert _topk_set(np.array([3, 1, 2]), np.array([0.9, 0.5, 0.1]), 2) == {3, 1}

    # Two heads, B=2. Head0 top: blocks {10,11}; Head1 top: {10,12}.
    # max aggregation: cross-head max scores -> blocks 10,11,12 ; top-2 by score.
    # vote aggregation: block10 has 2 votes (both heads' top-2), 11 and 12 one each.
    # block_size=1 so middle_budget (positions) == B (blocks) == 2.
    step = StepSelections(
        layer=0,
        decode_step=0,
        head_blocks=[np.array([10, 11]), np.array([10, 12])],
        head_scores=[np.array([0.9, 0.8]), np.array([0.7, 0.6])],
        kept_blocks=frozenset({10, 11}),
        middle_budget=2,
    )
    max_set, vote_set = _reaggregate(step, block_size=1)
    # max top-2 by cross-head max score: 10 (0.9), 11 (0.8).
    assert max_set == {10, 11}
    # vote: 10 has 2 votes (always first), then tie between 11/12 at 1 vote ->
    # break by cross-head max score: 11 (0.8) > 12 (0.6).
    assert 10 in vote_set and len(vote_set) == 2 and vote_set == {10, 11}


def test_reaggregate_vote_drops_no_consensus_block() -> None:
    """A block only ONE head ranks (via a single outlier) loses to consensus."""
    from scripts.recoding_figures.analyze_per_head_topk import (
        StepSelections,
        _reaggregate,
    )

    # 3 heads, B=2 (block_size=1: positions == blocks). Block 99 has head0's
    # huge outlier score but no other head. Blocks 1 and 2 are each in 2 heads'
    # top-2 (consensus). vote should keep the consensus pair; max would be
    # polluted by the outlier 99.
    step = StepSelections(
        layer=0,
        decode_step=0,
        head_blocks=[
            np.array([99, 1]),
            np.array([1, 2]),
            np.array([2, 1]),
        ],
        head_scores=[
            np.array([100.0, 1.0]),
            np.array([2.0, 1.5]),
            np.array([2.0, 1.5]),
        ],
        kept_blocks=frozenset(),
        middle_budget=2,
    )
    max_set, vote_set = _reaggregate(step, block_size=1)
    assert 99 in max_set  # outlier dominates the max aggregation
    assert 99 not in vote_set  # but gets no consensus votes
    assert vote_set == {1, 2}  # the two consensus blocks
