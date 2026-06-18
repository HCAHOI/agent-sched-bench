"""Pure helpers for metadata-residency validation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def mean_contiguity_displacement(kept_indices: Sequence[int]) -> float:
    """Mean displacement from contiguous renumbering for sorted survivors."""
    kept = [int(idx) for idx in kept_indices]
    if not kept:
        return 0.0
    sorted_kept = sorted(kept)
    return float(
        np.mean([original - dense_idx for dense_idx, original in enumerate(sorted_kept)])
    )


def phase_gap_count(kept_indices: Sequence[int]) -> int:
    """Count non-adjacent jumps among sorted survivor positions."""
    kept = sorted(int(idx) for idx in kept_indices)
    if len(kept) < 2:
        return 0
    return sum(1 for left, right in zip(kept, kept[1:], strict=True) if right != left + 1)


@dataclass(frozen=True)
class AffectedPositionResult:
    """Set-A and Set-B affected decode-row masks."""

    set_a_mask: np.ndarray
    set_b_mask: np.ndarray

    @property
    def set_a_indices(self) -> np.ndarray:
        return np.flatnonzero(self.set_a_mask)

    @property
    def set_b_indices(self) -> np.ndarray:
        return np.flatnonzero(self.set_b_mask)


def resolve_affected_positions(
    *,
    evicted_original_by_row: Sequence[Sequence[int]],
    topk_indices: np.ndarray,
    topk_weights: np.ndarray,
) -> AffectedPositionResult:
    """Resolve affected rows using Set B, not the weaker Set A.

    Set A marks rows that evicted at least one token. Set B marks rows where an
    evicted ORIGINAL token appears in the fresh full-KV top-k row with nonzero
    mass. The full attention source is top-k CSR materialized by the caller; no
    dense fallback or reused-recording reference is implied here.
    """
    if topk_indices.shape != topk_weights.shape:
        raise ValueError(
            f"topk_indices shape {topk_indices.shape} != weights {topk_weights.shape}"
        )
    n_rows = len(evicted_original_by_row)
    if topk_indices.shape[0] != n_rows:
        raise ValueError(
            f"topk rows {topk_indices.shape[0]} != evicted rows {n_rows}"
        )
    set_a = np.zeros(n_rows, dtype=bool)
    set_b = np.zeros(n_rows, dtype=bool)
    for row_idx, evicted in enumerate(evicted_original_by_row):
        evicted_set = {int(idx) for idx in evicted}
        if not evicted_set:
            continue
        set_a[row_idx] = True
        row_indices = topk_indices[row_idx].astype(np.int64, copy=False)
        row_weights = topk_weights[row_idx].astype(np.float64, copy=False)
        valid = (row_indices >= 0) & (row_weights > 0.0)
        attended = {int(idx) for idx in row_indices[valid]}
        set_b[row_idx] = bool(evicted_set.intersection(attended))
    return AffectedPositionResult(set_a_mask=set_a, set_b_mask=set_b)


def assert_teacher_forced_ids_equal(
    full_ids: Sequence[int] | np.ndarray,
    policy_ids: Sequence[int] | np.ndarray,
) -> None:
    """Hard-fail unless teacher-forced token-id sequences are byte-identical."""
    full = np.asarray(full_ids, dtype=np.int64)
    policy = np.asarray(policy_ids, dtype=np.int64)
    if full.shape != policy.shape or not np.array_equal(full, policy):
        raise AssertionError(
            "teacher-forced token ids differ; refusing to compute KL on "
            "non-identical conditioning sequences"
        )


def assert_logits_byte_identical(
    full_logits: np.ndarray,
    policy_logits: np.ndarray,
) -> None:
    """Hard-fail unless two logits arrays are bit-for-bit identical."""
    full = np.asarray(full_logits)
    policy = np.asarray(policy_logits)
    if full.shape != policy.shape:
        raise AssertionError(
            f"logit shape mismatch: full={full.shape}, policy={policy.shape}"
        )
    if full.dtype != policy.dtype:
        raise AssertionError(
            f"logit dtype mismatch: full={full.dtype}, policy={policy.dtype}"
        )
    if full.tobytes() != policy.tobytes():
        raise AssertionError("logits are not byte-identical")
