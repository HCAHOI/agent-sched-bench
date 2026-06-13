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


def _brute_pairwise_jaccard(head_blocks, k):
    """Scalar reference: mean/median top-k Jaccard over non-empty-union pairs."""
    sets = [set(b[:k].tolist()) for b in head_blocks]
    H = len(sets)
    jac = []
    for i in range(H):
        for j in range(i + 1, H):
            a, b = sets[i], sets[j]
            if not (a or b):
                continue
            jac.append(len(a & b) / len(a | b))
    if not jac:
        return float("nan"), float("nan")
    return float(np.mean(jac)), float(np.median(jac))


def test_membership_jaccard_matches_bruteforce() -> None:
    """Vectorized membership Jaccard == scalar pairwise reference (random data)."""
    from scripts.recoding_figures.analyze_per_head_topk import (
        _pairwise_jaccard_stats,
        _topk_membership,
    )

    rng = np.random.default_rng(20260613)
    for _ in range(50):
        H = int(rng.integers(2, 33))
        R = int(rng.integers(1, 65))
        n_blocks = int(rng.integers(R, 4 * R + 1))
        head_blocks = []
        for _h in range(H):
            r = int(rng.integers(0, R + 1))  # allow empty heads
            blocks = rng.choice(n_blocks, size=r, replace=False).astype(np.int64)
            scores = rng.random(r)
            order = np.argsort(-scores)  # descending-score order, as recorded
            head_blocks.append(blocks[order])
        for k in (8, 16, 32, 64):
            membership, sizes, _ = _topk_membership(head_blocks, k)
            got = _pairwise_jaccard_stats(membership, sizes)
            ref = _brute_pairwise_jaccard(head_blocks, k)
            for g, r in zip(got, ref):
                assert (np.isnan(g) and np.isnan(r)) or g == r


def _old_step_rows(task, call_idx, step, roles, block_size):
    """Frozen pre-vectorization _step_rows for numerical-equivalence guarding."""
    import json as _json
    from collections import defaultdict

    from scripts.recoding_figures.analyze_per_head_topk import (
        CONSENSUS_FRAC,
        CONSENSUS_K,
        TOP_K_VALUES,
        _jaccard,
        _n90,
        _reaggregate,
        _role_counts,
        _topk_set,
    )

    n_heads = len(step.head_blocks)
    pairs = [(i, j) for i in range(n_heads) for j in range(i + 1, n_heads)]
    n90_vals = [_n90(sc) for sc in step.head_scores if sc.shape[0] > 0]
    n90_mean = float(np.mean(n90_vals)) if n90_vals else float("nan")
    votes_c = defaultdict(int)
    for blocks, scores in zip(step.head_blocks, step.head_scores):
        for blk in _topk_set(blocks, scores, CONSENSUS_K):
            votes_c[blk] += 1
    min_consensus = int(np.ceil(CONSENSUS_FRAC * n_heads))
    consensus_core = {blk for blk, v in votes_c.items() if v >= min_consensus}
    max_set, vote_set = _reaggregate(step, block_size)
    vote_max_jac = _jaccard(max_set, vote_set)
    only_vote = vote_set - max_set
    only_max = max_set - vote_set
    role_only_vote = _role_counts(only_vote, roles)
    role_only_max = _role_counts(only_max, roles)
    rows = []
    for k in TOP_K_VALUES:
        head_sets = [
            _topk_set(b, s, k) for b, s in zip(step.head_blocks, step.head_scores)
        ]
        jac = [
            _jaccard(head_sets[i], head_sets[j])
            for i, j in pairs
            if head_sets[i] or head_sets[j]
        ]
        union = set()
        for s in head_sets:
            union |= s
        kept = step.kept_blocks
        rows.append(
            {
                "task": task,
                "call_idx": call_idx,
                "layer": step.layer,
                "decode_step": step.decode_step,
                "k": k,
                "n_heads": n_heads,
                "jaccard_mean": float(np.mean(jac)) if jac else float("nan"),
                "jaccard_median": float(np.median(jac)) if jac else float("nan"),
                "union_abs": len(union),
                "kept_abs": len(kept),
                "union_and_kept": len(union & kept),
                "union_or_kept": len(union | kept),
                "union_minus_kept": len(union - kept),
                "kept_minus_union": len(kept - union),
                "n90_mean": n90_mean,
                "consensus_core_size": len(consensus_core),
                "middle_budget": step.middle_budget,
                "vote_vs_max_jaccard": vote_max_jac,
                "vote_only_count": len(only_vote),
                "max_only_count": len(only_max),
                "vote_only_roles": _json.dumps(role_only_vote, sort_keys=True),
                "max_only_roles": _json.dumps(role_only_max, sort_keys=True),
            }
        )
    return rows


def test_step_rows_matches_old_reference() -> None:
    """New vectorized _step_rows == frozen scalar reference on random fixtures."""
    from scripts.recoding_figures.analyze_per_head_topk import (
        StepSelections,
        _step_rows,
    )

    rng = np.random.default_rng(424242)
    for _ in range(40):
        H = int(rng.integers(2, 33))
        R = int(rng.integers(1, 65))
        n_blocks = int(rng.integers(R, 5 * R + 1))
        head_blocks, head_scores = [], []
        all_blocks: set[int] = set()
        for _h in range(H):
            r = int(rng.integers(0, R + 1))
            blocks = rng.choice(n_blocks, size=r, replace=False).astype(np.int32)
            scores = rng.random(r).astype(np.float16)
            order = np.argsort(-scores.astype(np.float32))
            blocks, scores = blocks[order], scores[order]
            head_blocks.append(blocks)
            head_scores.append(scores)
            all_blocks.update(int(b) for b in blocks)
        # kept overlaps the candidate union plus some out-of-union ids.
        kept_pool = list(all_blocks) + [n_blocks + i for i in range(5)]
        kept_size = int(rng.integers(0, len(kept_pool) + 1))
        kept = frozenset(
            int(x) for x in rng.choice(kept_pool, size=kept_size, replace=False)
        ) if kept_pool else frozenset()
        step = StepSelections(
            layer=int(rng.integers(0, 14)),
            decode_step=int(rng.integers(0, 500)),
            head_blocks=head_blocks,
            head_scores=head_scores,
            kept_blocks=kept,
            middle_budget=int(rng.integers(0, 4 * R + 1)),
        )
        roles = {b: ("tool_result" if b % 2 else "user") for b in range(n_blocks + 5)}
        new_rows = _step_rows("t", 0, step, roles, block_size=16)
        old_rows = _old_step_rows("t", 0, step, roles, block_size=16)
        assert len(new_rows) == len(old_rows)
        for nr, orow in zip(new_rows, old_rows):
            assert set(nr) == set(orow)
            for key in nr:
                a, b = nr[key], orow[key]
                if isinstance(a, float) and np.isnan(a):
                    assert np.isnan(b)
                else:
                    assert a == b, f"{key}: {a!r} != {b!r}"
