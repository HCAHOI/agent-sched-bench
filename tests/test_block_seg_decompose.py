"""Unit tests for `_segment_decode_stats`, the shared per-(head, segment) decode
aggregator that both head_span and the `block_span_seg_*` decomposition use.

The block-side use passes ``restrict=kept_bool`` so the stats cover only the
block_topk-selected (middle) positions; each kept key belongs to exactly one
segment, which is what makes a block straddling two segments split cleanly.

torch-gated: skipped where torch is unavailable (e.g. the local CPU .venv).
"""

import pytest

pytest.importorskip("torch")

import torch
from serving.recording.hooks import _segment_decode_stats


def _attn(values: list[float]):
    """Build a [H=2, 1, K] attn tensor; head 1 = head 0 * 10 (distinct per head)."""
    row = torch.tensor(values, dtype=torch.float32)
    return torch.stack([row, row * 10.0]).unsqueeze(1)


def test_full_segment_aggregation_no_restrict():
    attn = _attn([1.0, 3.0, 5.0, 7.0, 2.0, 4.0])  # K=6
    key_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    mean, var, kept = _segment_decode_stats(attn, key_ids, 3)

    assert mean.shape == (2, 3) and var.shape == (2, 3) and kept.shape == (3,)
    # seg means (head0): mean(1,3)=2, mean(5,7)=6, mean(2,4)=3
    assert torch.allclose(mean[0], torch.tensor([2.0, 6.0, 3.0]))
    assert torch.allclose(mean[1], torch.tensor([20.0, 60.0, 30.0]))  # head1 = *10
    # population variance of each 2-element segment is 1.0
    assert torch.allclose(var[0], torch.tensor([1.0, 1.0, 1.0]))
    assert kept.tolist() == [2, 2, 2]


def test_restrict_mask_splits_straddling_block_by_segment():
    # kept positions {1,2,4,5} straddle seg0/seg1/seg2 -> each key counted once
    # in its own segment (no "primary segment", no double counting).
    attn = _attn([1.0, 3.0, 5.0, 7.0, 2.0, 4.0])
    key_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    restrict = torch.tensor([False, True, True, False, True, True])

    mean, var, kept = _segment_decode_stats(attn, key_ids, 3, restrict=restrict)

    assert kept.tolist() == [1, 1, 2]  # seg0:{1}, seg1:{2}, seg2:{4,5}
    # seg0 = attn@1 = 3, seg1 = attn@2 = 5, seg2 = mean(2,4) = 3  (head0)
    assert torch.allclose(mean[0], torch.tensor([3.0, 5.0, 3.0]))
    # single-key segments have 0 variance; seg2 = pop var(2,4) = 1.0
    assert torch.allclose(var[0], torch.tensor([0.0, 0.0, 1.0]))


def test_segment_with_no_kept_key_stays_zero_count():
    # restrict keeps nothing in segment 1 -> kept_count 0, mean/var left at 0
    # (the build step NaN-fills cells whose kept_count == 0).
    attn = _attn([1.0, 3.0, 5.0, 7.0, 2.0, 4.0])
    key_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    restrict = torch.tensor([True, True, False, False, True, True])  # seg1 dropped

    mean, var, kept = _segment_decode_stats(attn, key_ids, 3, restrict=restrict)

    assert kept.tolist() == [2, 0, 2]
    assert mean[0, 1].item() == 0.0 and var[0, 1].item() == 0.0
