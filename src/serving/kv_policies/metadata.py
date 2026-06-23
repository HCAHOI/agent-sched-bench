"""Metadata-driven KV residency policies.

The metadata bridge is intentionally limited to information already present in
the prompt/message stream at the `notify_new_call` boundary. It does not read
task labels, ground truth, future generated text, or recording artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import yaml

from serving.kv_policies.base import (
    BaseEvictionCache,
    EvictionDecision,
    EvictionPolicyConfig,
)

if TYPE_CHECKING:
    from serving.kv_policies.recorder import KVEvictionRecorder


_SYSTEM_ROLE = "system"

_ROLE_PRIORITY: dict[str, int] = {
    _SYSTEM_ROLE: 7,
    "tool_result": 6,
    "user": 5,
    "assistant_call": 4,
    "assistant_message": 3,
    "gen_prompt": 2,
    "generation": 1,
}

_RUNG_LEVEL: dict[str, int] = {
    "rung1": 1,
    "rung2": 2,
    "rung3": 3,
    "rung4": 4,
}


@dataclass(frozen=True)
class TokenMetadata:
    """Metadata for one ORIGINAL token index.

    The table is keyed by original absolute token index. Cache compaction keeps
    a separate per-layer original-to-current remap so metadata lookup never
    treats compacted physical positions as original positions.
    """

    original_index: int
    segment_id: int
    role: str
    age: int
    offset: int
    segment_start: int
    segment_end: int
    exit_code: int | None = None
    tool_error: bool | None = None


def build_token_metadata_from_segments(
    segments: list[dict[str, Any]],
    *,
    input_token_count: int,
    call_idx: int,
) -> dict[int, TokenMetadata]:
    """Build ORIGINAL-index token metadata from prompt segments.

    `segments` is the same list passed into `LayerCapturer.recording_session`.
    It covers prompt tokens only. Generated tokens from the current call are not
    yet emitted at the boundary and therefore receive default non-reserved
    metadata until they re-enter a later prompt as ordinary messages.
    """
    if input_token_count < 0:
        raise ValueError(f"input_token_count must be non-negative, got {input_token_count}")
    table: dict[int, TokenMetadata] = {}
    for segment_id, segment in enumerate(segments):
        start = max(0, _optional_int(segment.get("token_start"), default=0))
        end = min(
            int(input_token_count),
            max(start, _optional_int(segment.get("token_end"), default=start)),
        )
        if start >= end:
            continue
        first_seen = _optional_int(segment.get("first_seen_call"), default=call_idx)
        age = max(0, int(call_idx) - first_seen)
        role = str(segment.get("role") or "unknown")
        exit_code = _none_or_int(segment.get("exit_code"))
        tool_error = _none_or_bool(segment.get("tool_error"))
        for original in range(start, end):
            table[original] = TokenMetadata(
                original_index=original,
                segment_id=int(segment_id),
                role=role,
                age=age,
                offset=original - start,
                segment_start=start,
                segment_end=end,
                exit_code=exit_code,
                tool_error=tool_error,
            )
    return table


def default_token_metadata(original_index: int) -> TokenMetadata:
    """Metadata for generated/unclassified tokens available only as K/V slots."""
    original = int(original_index)
    return TokenMetadata(
        original_index=original,
        segment_id=-1,
        role="generation",
        age=0,
        offset=0,
        segment_start=original,
        segment_end=original + 1,
        exit_code=None,
        tool_error=None,
    )


def _optional_int(value: Any, *, default: int) -> int:
    if value is None or value == "":
        return int(default)
    return int(value)


def _none_or_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _none_or_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _metadata_arrays_from_table(
    *,
    original_indices: list[int],
    metadata_table: dict[int, TokenMetadata],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    originals = np.asarray(original_indices, dtype=np.int64)
    n = int(originals.shape[0])
    role_rank = np.full(n, _ROLE_PRIORITY["generation"], dtype=np.int16)
    age = np.zeros(n, dtype=np.int32)
    offset = np.zeros(n, dtype=np.int32)
    segment_id = np.full(n, -1, dtype=np.int32)
    is_tool_result = np.zeros(n, dtype=bool)
    is_error = np.zeros(n, dtype=bool)
    is_system = np.zeros(n, dtype=bool)
    for physical_idx, original in enumerate(originals.tolist()):
        meta = metadata_table.get(int(original))
        if meta is None:
            continue
        role = str(meta.role)
        role_rank[physical_idx] = int(_ROLE_PRIORITY.get(role, 0))
        is_system[physical_idx] = role == _SYSTEM_ROLE
        age[physical_idx] = int(meta.age)
        offset[physical_idx] = int(meta.offset)
        segment_id[physical_idx] = int(meta.segment_id)
        if role == "tool_result":
            is_tool_result[physical_idx] = True
            is_error[physical_idx] = bool(meta.tool_error is True) or (
                meta.exit_code is not None and int(meta.exit_code) != 0
            )
    return (
        originals,
        role_rank,
        age,
        offset,
        segment_id,
        is_tool_result,
        is_error,
        is_system,
    )


@dataclass(frozen=True)
class MetadataSelection:
    """Keep/evict result in current physical-index space plus original indices."""

    keep_indices: list[int]
    evict_indices: list[int]
    original_kept_indices: list[int]
    original_evicted_indices: list[int]
    reason: str


@dataclass(frozen=True)
class PerLayerPriorityTable:
    """Frozen P0 metadata score table keyed by (layer, role, age)."""

    scores: dict[tuple[int, str, int], float]

    @classmethod
    def from_path(cls, path: str) -> "PerLayerPriorityTable":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            rows = raw.get("scores")
        else:
            rows = raw
        if not isinstance(rows, list):
            raise ValueError(
                "per-layer metadata table must be a YAML mapping with a "
                "`scores:` list or a top-level list of score rows"
            )
        scores: dict[tuple[int, str, int], float] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("each per-layer metadata score row must be a mapping")
            try:
                key = (
                    int(row["layer"]),
                    str(row["role"]),
                    int(row["age"]),
                )
                scores[key] = float(row["score"])
            except KeyError as exc:
                raise ValueError(
                    "per-layer metadata score rows require layer, role, age, score"
                ) from exc
        if not scores:
            raise ValueError("per-layer metadata table is empty")
        return cls(scores=scores)

    def score(self, *, layer_idx: int, meta: TokenMetadata) -> float:
        key = (int(layer_idx), str(meta.role), int(meta.age))
        try:
            return float(self.scores[key])
        except KeyError as exc:
            raise KeyError(
                "per-layer metadata table is missing score for "
                f"layer={key[0]}, role={key[1]!r}, age={key[2]}"
            ) from exc


class MetadataResidencySelector:
    """Pure keep-set selector shared by the cache and observe-only sidecar."""

    def __init__(self, config: EvictionPolicyConfig) -> None:
        if config.budget is None or int(config.budget) <= 0:
            raise ValueError(
                f"metadata residency requires positive config.budget; got {config.budget!r}"
            )
        if config.metadata_rung not in _RUNG_LEVEL:
            raise ValueError(
                f"metadata_rung={config.metadata_rung!r} unsupported; "
                f"choose one of {sorted(_RUNG_LEVEL)}"
            )
        if config.sink_size < 0:
            raise ValueError(
                f"metadata residency requires sink_size >= 0; got {config.sink_size!r}"
            )
        if config.recent_window < 0:
            raise ValueError(
                "metadata residency requires recent_window >= 0; "
                f"got {config.recent_window!r}"
            )
        if _RUNG_LEVEL[config.metadata_rung] >= 2:
            floor = int(config.sink_size) + int(config.recent_window)
            if int(config.budget) < floor:
                raise ValueError(
                    "metadata rung2+ requires budget >= sink_size + recent_window "
                    f"({config.budget!r} < {config.sink_size!r} + "
                    f"{config.recent_window!r})"
                )
        if config.per_layer_budget:
            raise ValueError(
                "per_layer_budget requires a frozen per-layer allocation rule "
                "from P0 and is not implemented in the P1 CPU path"
            )
        if config.per_layer_table:
            if not config.per_layer_table_path:
                raise ValueError(
                    "per_layer_table requires per_layer_table_path pointing to "
                    "a frozen P0 score table"
                )
            self._per_layer_table = PerLayerPriorityTable.from_path(
                str(config.per_layer_table_path)
            )
        else:
            self._per_layer_table = None
        self.config = config

    @property
    def layer_independent(self) -> bool:
        """True when a single keep-set can be shared across all layers."""
        return self._per_layer_table is None

    def select(
        self,
        *,
        layer_idx: int,
        key_len: int,
        original_indices: list[int],
        metadata_table: dict[int, TokenMetadata],
    ) -> MetadataSelection:
        if self._per_layer_table is None:
            arrays = _metadata_arrays_from_table(
                original_indices=original_indices,
                metadata_table=metadata_table,
            )
            return self.select_from_arrays(
                layer_idx=layer_idx,
                key_len=key_len,
                original_indices=arrays[0],
                role_rank=arrays[1],
                age=arrays[2],
                offset=arrays[3],
                segment_id=arrays[4],
                is_tool_result=arrays[5],
                is_error=arrays[6],
                is_system=arrays[7],
            )
        return self._select_python(
            layer_idx=layer_idx,
            key_len=key_len,
            original_indices=original_indices,
            metadata_table=metadata_table,
        )

    def _select_python(
        self,
        *,
        layer_idx: int,
        key_len: int,
        original_indices: list[int],
        metadata_table: dict[int, TokenMetadata],
    ) -> MetadataSelection:
        if key_len != len(original_indices):
            raise ValueError(
                f"key_len {key_len} != original index map length {len(original_indices)}"
            )
        budget = int(self.config.budget)  # type: ignore[arg-type]
        system_reserved = self._system_reserved(
            original_indices=original_indices,
            metadata_table=metadata_table,
        )
        if key_len - len(system_reserved) <= budget:
            return self._selection_from_keep(
                key_len=key_len,
                keep_indices=list(range(key_len)),
                original_indices=original_indices,
                reason="none",
            )

        rung = _RUNG_LEVEL[str(self.config.metadata_rung)]
        hard_reserved = self._sink_recent_reserved(key_len) if rung >= 2 else set()
        soft_reserved: set[int] = set()
        if rung >= 3:
            soft_reserved.update(
                self._recent_tool_result_reserved(
                    key_len=key_len,
                    original_indices=original_indices,
                    metadata_table=metadata_table,
                )
            )
        if rung >= 4:
            soft_reserved.update(
                self._error_reserved(
                    original_indices=original_indices,
                    metadata_table=metadata_table,
                )
            )
        hard_budget_count = len(hard_reserved - system_reserved)
        if hard_budget_count > budget:
            raise RuntimeError(
                "non-system hard metadata reservation exceeds budget; increase "
                "budget or reduce sink/recent reservation"
            )

        keep_set: set[int] = set(system_reserved)
        keep_set.update(hard_reserved)
        remaining_slots = budget - hard_budget_count
        candidates = [idx for idx in range(key_len) if idx not in keep_set]
        ranked = sorted(
            candidates,
            key=lambda idx: self._priority_key(
                layer_idx=int(layer_idx),
                physical_idx=idx,
                original_indices=original_indices,
                metadata_table=metadata_table,
                soft_reserved=soft_reserved,
            ),
            reverse=True,
        )
        keep_set.update(ranked[:remaining_slots])
        keep = sorted(keep_set)
        return self._selection_from_keep(
            key_len=key_len,
            keep_indices=keep,
            original_indices=original_indices,
            reason=str(self.config.metadata_rung),
        )

    def select_from_arrays(
        self,
        *,
        layer_idx: int,
        key_len: int,
        original_indices: np.ndarray,
        role_rank: np.ndarray,
        age: np.ndarray,
        offset: np.ndarray,
        segment_id: np.ndarray,
        is_tool_result: np.ndarray,
        is_error: np.ndarray,
        is_system: np.ndarray,
    ) -> MetadataSelection:
        """Vectorized layer-independent selector for the global table path."""
        del layer_idx
        if key_len != int(original_indices.shape[0]):
            raise ValueError(
                f"key_len {key_len} != original index map length "
                f"{int(original_indices.shape[0])}"
            )
        budget = int(self.config.budget)  # type: ignore[arg-type]
        system_reserved = (
            is_system.astype(bool, copy=True)
            if self.config.reserve_system_prompt
            else np.zeros(key_len, dtype=bool)
        )
        if key_len - int(system_reserved.sum()) <= budget:
            keep = np.arange(key_len, dtype=np.int32)
            return self._selection_from_numpy_keep(
                key_len=key_len,
                keep_indices=keep,
                original_indices=original_indices,
                reason="none",
            )

        rung = _RUNG_LEVEL[str(self.config.metadata_rung)]
        hard_reserved = (
            self._sink_recent_reserved_mask(key_len=key_len) if rung >= 2 else np.zeros(key_len, dtype=bool)
        )
        soft_reserved = np.zeros(key_len, dtype=bool)
        if rung >= 3:
            soft_reserved |= self._recent_tool_result_reserved_mask(
                key_len=key_len,
                segment_id=segment_id,
                is_tool_result=is_tool_result,
            )
        if rung >= 4:
            soft_reserved |= is_error
        hard_budget_count = int(np.logical_and(hard_reserved, ~system_reserved).sum())
        if hard_budget_count > budget:
            raise RuntimeError(
                "non-system hard metadata reservation exceeds budget; increase "
                "budget or reduce sink/recent reservation"
            )

        keep_mask = np.logical_or(system_reserved, hard_reserved)
        remaining_slots = budget - hard_budget_count
        if remaining_slots > 0:
            candidates = np.nonzero(~keep_mask)[0].astype(np.int32, copy=False)
            order = np.lexsort(
                (
                    original_indices[candidates],
                    offset[candidates],
                    -age[candidates],
                    -role_rank[candidates],
                    -soft_reserved[candidates].astype(np.int8),
                )
            )
            keep_mask[candidates[order[:remaining_slots]]] = True
        keep = np.nonzero(keep_mask)[0].astype(np.int32, copy=False)
        return self._selection_from_numpy_keep(
            key_len=key_len,
            keep_indices=keep,
            original_indices=original_indices,
            reason=str(self.config.metadata_rung),
        )

    def _selection_from_keep(
        self,
        *,
        key_len: int,
        keep_indices: list[int],
        original_indices: list[int],
        reason: str,
    ) -> MetadataSelection:
        keep_set = set(int(idx) for idx in keep_indices)
        evict = [idx for idx in range(key_len) if idx not in keep_set]
        return MetadataSelection(
            keep_indices=list(keep_indices),
            evict_indices=evict,
            original_kept_indices=[int(original_indices[idx]) for idx in keep_indices],
            original_evicted_indices=[int(original_indices[idx]) for idx in evict],
            reason=reason,
        )

    def _selection_from_numpy_keep(
        self,
        *,
        key_len: int,
        keep_indices: np.ndarray,
        original_indices: np.ndarray,
        reason: str,
    ) -> MetadataSelection:
        keep_mask = np.zeros(key_len, dtype=bool)
        keep_mask[keep_indices] = True
        evict_indices = np.nonzero(~keep_mask)[0].astype(np.int32, copy=False)
        return MetadataSelection(
            keep_indices=[int(idx) for idx in keep_indices.tolist()],
            evict_indices=[int(idx) for idx in evict_indices.tolist()],
            original_kept_indices=[
                int(idx) for idx in original_indices[keep_indices].tolist()
            ],
            original_evicted_indices=[
                int(idx) for idx in original_indices[evict_indices].tolist()
            ],
            reason=reason,
        )

    def _system_reserved(
        self,
        *,
        original_indices: list[int],
        metadata_table: dict[int, TokenMetadata],
    ) -> set[int]:
        """Physical positions for system/tool-schema prompt tokens.

        These tokens are inference-time context, not oracle information. When
        enabled, they are forced resident and do not consume the policy budget;
        the budget then measures residency quality over the growing
        conversation state.
        """
        if not self.config.reserve_system_prompt:
            return set()
        keep: set[int] = set()
        for physical_idx, original_idx in enumerate(original_indices):
            meta = metadata_table.get(
                int(original_idx), default_token_metadata(int(original_idx))
            )
            if meta.role == _SYSTEM_ROLE:
                keep.add(physical_idx)
        return keep

    def _sink_recent_reserved(self, key_len: int) -> set[int]:
        sink = min(int(self.config.sink_size), key_len)
        recent = min(int(self.config.recent_window), key_len)
        keep = set(range(sink))
        keep.update(range(max(0, key_len - recent), key_len))
        return keep

    def _sink_recent_reserved_mask(self, *, key_len: int) -> np.ndarray:
        sink = min(int(self.config.sink_size), key_len)
        recent = min(int(self.config.recent_window), key_len)
        keep = np.zeros(key_len, dtype=bool)
        if sink > 0:
            keep[:sink] = True
        if recent > 0:
            keep[max(0, key_len - recent) :] = True
        return keep

    def _recent_tool_result_reserved(
        self,
        *,
        key_len: int,
        original_indices: list[int],
        metadata_table: dict[int, TokenMetadata],
    ) -> set[int]:
        recent_window = int(self.config.recent_window)
        if recent_window <= 0:
            return set()
        recent_floor = max(0, key_len - recent_window)
        tool_positions_by_segment: dict[int, list[int]] = {}
        for physical_idx, original_idx in enumerate(original_indices):
            meta = metadata_table.get(
                int(original_idx), default_token_metadata(int(original_idx))
            )
            if meta.role == "tool_result":
                tool_positions_by_segment.setdefault(int(meta.segment_id), []).append(
                    physical_idx
                )

        keep: set[int] = set()
        for positions in tool_positions_by_segment.values():
            # Segment boundaries are evaluated in current physical cache space.
            # ORIGINAL segment_end is intentionally not used here: after KV
            # compaction it would keep stale tool outputs "recent" forever.
            current_segment_end = max(positions) + 1
            if current_segment_end >= recent_floor:
                keep.update(positions)
        return keep

    def _recent_tool_result_reserved_mask(
        self,
        *,
        key_len: int,
        segment_id: np.ndarray,
        is_tool_result: np.ndarray,
    ) -> np.ndarray:
        recent_window = int(self.config.recent_window)
        keep = np.zeros(key_len, dtype=bool)
        if recent_window <= 0:
            return keep
        tool_positions = np.nonzero(is_tool_result)[0].astype(np.int32, copy=False)
        if tool_positions.size == 0:
            return keep
        recent_floor = max(0, key_len - recent_window)
        tool_segments = segment_id[tool_positions]
        order = np.argsort(tool_segments, kind="stable")
        sorted_positions = tool_positions[order]
        sorted_segments = tool_segments[order]
        group_starts = np.r_[
            0, np.nonzero(sorted_segments[1:] != sorted_segments[:-1])[0] + 1
        ]
        group_ends = np.r_[group_starts[1:], sorted_segments.shape[0]]
        max_positions = np.maximum.reduceat(sorted_positions, group_starts)
        group_keep = (max_positions + 1) >= recent_floor
        repeated = np.repeat(group_keep, group_ends - group_starts)
        keep[sorted_positions[repeated]] = True
        return keep

    def _error_reserved(
        self,
        *,
        original_indices: list[int],
        metadata_table: dict[int, TokenMetadata],
    ) -> set[int]:
        keep: set[int] = set()
        for physical_idx, original_idx in enumerate(original_indices):
            meta = metadata_table.get(
                int(original_idx), default_token_metadata(int(original_idx))
            )
            if meta.role != "tool_result":
                continue
            if meta.tool_error is True:
                keep.add(physical_idx)
            elif meta.exit_code is not None and int(meta.exit_code) != 0:
                keep.add(physical_idx)
        return keep

    def _priority_key(
        self,
        *,
        layer_idx: int,
        physical_idx: int,
        original_indices: list[int],
        metadata_table: dict[int, TokenMetadata],
        soft_reserved: set[int],
    ) -> tuple[int, float, int, int, int]:
        original = int(original_indices[physical_idx])
        meta = metadata_table.get(original, default_token_metadata(original))
        score = self._metadata_score(meta=meta, layer_idx=int(layer_idx))
        age_rank = 0 if self._per_layer_table is not None else int(meta.age)
        # Tuple order: functional reservations first, then role, age, and
        # intra-segment offset. Original index is a deterministic tie-break.
        return (
            1 if physical_idx in soft_reserved else 0,
            score,
            age_rank,
            -int(meta.offset),
            -original,
        )

    def _metadata_score(self, *, meta: TokenMetadata, layer_idx: int) -> float:
        if self._per_layer_table is not None:
            return self._per_layer_table.score(layer_idx=layer_idx, meta=meta)
        return float(_ROLE_PRIORITY.get(meta.role, 0))


class MetadataResidencyCache(BaseEvictionCache):
    """Physical KV cache using metadata-derived residency decisions."""

    def supports_session_resume(self) -> bool:
        return self._selector.layer_independent

    def __init__(
        self,
        config: EvictionPolicyConfig,
        num_layers: int,
        recorder: "KVEvictionRecorder | None" = None,
    ) -> None:
        super().__init__(config, num_layers, recorder=recorder)
        self._selector = MetadataResidencySelector(config)
        self._metadata_table: dict[int, TokenMetadata] = {}
        self._original_indices_by_layer: dict[int, list[int]] = {}
        self._original_to_current_by_layer: dict[int, dict[int, int]] = {}
        self._next_original_by_layer: dict[int, int] = {}
        self._metadata_version = 0
        self._original_state_id_by_layer: dict[int, int] = {}
        self._state_transition_ids: dict[tuple[Any, ...], int] = {}
        self._next_state_id = 1
        self._role_rank_by_original = np.empty(0, dtype=np.int16)
        self._age_by_original = np.empty(0, dtype=np.int32)
        self._offset_by_original = np.empty(0, dtype=np.int32)
        self._segment_id_by_original = np.empty(0, dtype=np.int32)
        self._is_tool_result_by_original = np.empty(0, dtype=bool)
        self._is_error_by_original = np.empty(0, dtype=bool)
        self._is_system_by_original = np.empty(0, dtype=bool)
        self._selection_cache_key: tuple[int, int, int] | None = None
        self._selection_cache_originals: list[int] | None = None
        self._selection_cache_value: MetadataSelection | None = None
        self._selection_compute_count = 0
        self._selection_cache_hits = 0

    def notify_new_call(
        self,
        call_idx: int,
        *,
        segments: list[dict[str, Any]] | None = None,
        input_token_count: int | None = None,
    ) -> None:
        super().notify_new_call(
            call_idx, segments=segments, input_token_count=input_token_count
        )
        if segments is None or input_token_count is None:
            return
        new_metadata = build_token_metadata_from_segments(
            segments,
            input_token_count=int(input_token_count),
            call_idx=int(call_idx),
        )
        self._metadata_table.update(new_metadata)
        self._update_metadata_arrays(new_metadata)
        self._metadata_version += 1
        self._selection_cache_key = None
        self._selection_cache_originals = None
        self._selection_cache_value = None

    def original_indices_for_layer(
        self, layer_idx: int, key_len: int | None = None
    ) -> list[int]:
        """Return ORIGINAL token indices in current physical-slot order.

        `key_len` may include tokens being appended by the current attention
        call before `DynamicCache.update()` runs; this previews the same
        append-only original ids that `_decide_evict()` will later use.
        """
        length = self._current_layer_len(layer_idx) if key_len is None else int(key_len)
        return self._ensure_layer_state(int(layer_idx), length)

    def crop_to_logical_length(self, logical_length: int) -> None:
        length = int(logical_length)
        super().crop_to_logical_length(length)
        for layer, originals in list(self._original_indices_by_layer.items()):
            kept = [int(original) for original in originals if int(original) < length]
            self._original_indices_by_layer[layer] = kept
            self._original_to_current_by_layer[layer] = {
                int(original): current for current, original in enumerate(kept)
            }
            self._next_original_by_layer[layer] = length
            self._original_state_id_by_layer[layer] = self._intern_state_transition(
                ("logical_crop", self._original_state_id_by_layer.get(layer, 0), length)
            )
        self._selection_cache_key = None
        self._selection_cache_originals = None
        self._selection_cache_value = None

    def _decide_evict(self, layer_idx: int, key_len: int) -> EvictionDecision:
        layer = int(layer_idx)
        originals = self._ensure_layer_state(layer, int(key_len), copy=False)
        if self._selector.layer_independent:
            selection = self._cached_layer_independent_selection(
                key_len=int(key_len),
                originals=originals,
                state_id=self._original_state_id_by_layer.get(layer, 0),
            )
        else:
            self._selection_compute_count += 1
            selection = self._selector.select(
                layer_idx=layer,
                key_len=int(key_len),
                original_indices=originals,
                metadata_table=self._metadata_table,
            )
        return EvictionDecision(
            keep_indices=selection.keep_indices,
            evict_indices=selection.evict_indices,
            reason=selection.reason,
            policy_state={
                "original_kept_indices": selection.original_kept_indices,
                "original_evicted_indices": selection.original_evicted_indices,
            },
        )

    def _post_evict_hook(self, layer_idx: int, decision: EvictionDecision) -> None:
        layer = int(layer_idx)
        originals = self._original_indices_by_layer.get(layer)
        if originals is None:
            return
        self._original_indices_by_layer[layer] = [
            int(originals[idx]) for idx in decision.keep_indices
        ]
        self._original_to_current_by_layer[layer] = {
            int(original): current
            for current, original in enumerate(self._original_indices_by_layer[layer])
        }
        self._original_state_id_by_layer[layer] = self._intern_state_transition(
            (
                "keep",
                self._original_state_id_by_layer.get(layer, 0),
                tuple(int(idx) for idx in decision.keep_indices),
            )
        )

    def _ensure_layer_state(
        self, layer_idx: int, key_len: int, *, copy: bool = True
    ) -> list[int]:
        if key_len < 0:
            raise ValueError(f"key_len must be non-negative, got {key_len}")
        layer = int(layer_idx)
        originals = self._original_indices_by_layer.setdefault(layer, [])
        self._original_state_id_by_layer.setdefault(layer, 0)
        next_original = self._next_original_by_layer.get(layer, 0)
        if key_len < len(originals):
            raise RuntimeError(
                f"layer {layer} key_len {key_len} is shorter than tracked "
                f"physical origins {len(originals)}"
            )
        if key_len > len(originals):
            current_len = len(originals)
            delta = key_len - len(originals)
            new_originals = range(next_original, next_original + delta)
            originals.extend(new_originals)
            self._next_original_by_layer[layer] = next_original + delta
            mapping = self._original_to_current_by_layer.get(layer)
            if mapping is None or len(mapping) != current_len:
                mapping = {
                    int(original): current
                    for current, original in enumerate(originals[:current_len])
                }
                self._original_to_current_by_layer[layer] = mapping
            for current, original in enumerate(new_originals, start=current_len):
                mapping[int(original)] = current
            self._original_state_id_by_layer[layer] = self._intern_state_transition(
                (
                    "append",
                    self._original_state_id_by_layer.get(layer, 0),
                    next_original,
                    delta,
                )
            )
        elif layer not in self._original_to_current_by_layer:
            self._original_to_current_by_layer[layer] = {
                int(original): current for current, original in enumerate(originals)
            }
        return list(originals) if copy else originals

    def _current_layer_len(self, layer_idx: int) -> int:
        originals = self._original_indices_by_layer.get(int(layer_idx), [])
        return len(originals)

    def _cached_layer_independent_selection(
        self, *, key_len: int, originals: list[int], state_id: int | None = None
    ) -> MetadataSelection:
        cache_state_id = int(state_id) if state_id is not None else -1
        cache_key = (int(key_len), cache_state_id, int(self._metadata_version))
        if (
            self._selection_cache_key == cache_key
            and self._selection_cache_value is not None
            and (state_id is not None or self._selection_cache_originals == originals)
        ):
            self._selection_cache_hits += 1
            return self._selection_cache_value

        original_indices = np.asarray(originals, dtype=np.int64)
        if original_indices.size:
            self._ensure_metadata_array_capacity(int(original_indices.max()) + 1)
            role_rank = self._role_rank_by_original[original_indices]
            age = self._age_by_original[original_indices]
            offset = self._offset_by_original[original_indices]
            segment_id = self._segment_id_by_original[original_indices]
            is_tool_result = self._is_tool_result_by_original[original_indices]
            is_error = self._is_error_by_original[original_indices]
            is_system = self._is_system_by_original[original_indices]
        else:
            role_rank = np.empty(0, dtype=np.int16)
            age = np.empty(0, dtype=np.int32)
            offset = np.empty(0, dtype=np.int32)
            segment_id = np.empty(0, dtype=np.int32)
            is_tool_result = np.empty(0, dtype=bool)
            is_error = np.empty(0, dtype=bool)
            is_system = np.empty(0, dtype=bool)
        selection = self._selector.select_from_arrays(
            layer_idx=0,
            key_len=int(key_len),
            original_indices=original_indices,
            role_rank=role_rank,
            age=age,
            offset=offset,
            segment_id=segment_id,
            is_tool_result=is_tool_result,
            is_error=is_error,
            is_system=is_system,
        )
        self._selection_compute_count += 1
        self._selection_cache_key = cache_key
        self._selection_cache_originals = None if state_id is not None else list(originals)
        self._selection_cache_value = selection
        return selection

    def _intern_state_transition(self, key: tuple[Any, ...]) -> int:
        state_id = self._state_transition_ids.get(key)
        if state_id is not None:
            return state_id
        state_id = self._next_state_id
        self._next_state_id += 1
        self._state_transition_ids[key] = state_id
        return state_id

    def _ensure_metadata_array_capacity(self, size: int) -> None:
        current = int(self._role_rank_by_original.shape[0])
        if size <= current:
            return
        next_size = max(int(size), 16 if current == 0 else current * 2)
        grow = next_size - current
        self._role_rank_by_original = np.concatenate(
            [
                self._role_rank_by_original,
                np.full(grow, _ROLE_PRIORITY["generation"], dtype=np.int16),
            ]
        )
        self._age_by_original = np.concatenate(
            [self._age_by_original, np.zeros(grow, dtype=np.int32)]
        )
        self._offset_by_original = np.concatenate(
            [self._offset_by_original, np.zeros(grow, dtype=np.int32)]
        )
        self._segment_id_by_original = np.concatenate(
            [
                self._segment_id_by_original,
                np.full(grow, -1, dtype=np.int32),
            ]
        )
        self._is_tool_result_by_original = np.concatenate(
            [self._is_tool_result_by_original, np.zeros(grow, dtype=bool)]
        )
        self._is_error_by_original = np.concatenate(
            [self._is_error_by_original, np.zeros(grow, dtype=bool)]
        )
        self._is_system_by_original = np.concatenate(
            [self._is_system_by_original, np.zeros(grow, dtype=bool)]
        )

    def _update_metadata_arrays(self, metadata: dict[int, TokenMetadata]) -> None:
        if not metadata:
            return
        max_original = max(int(original) for original in metadata)
        self._ensure_metadata_array_capacity(max_original + 1)
        for original, meta in metadata.items():
            idx = int(original)
            role = str(meta.role)
            self._role_rank_by_original[idx] = int(_ROLE_PRIORITY.get(role, 0))
            self._age_by_original[idx] = int(meta.age)
            self._offset_by_original[idx] = int(meta.offset)
            self._segment_id_by_original[idx] = int(meta.segment_id)
            is_tool = role == "tool_result"
            self._is_tool_result_by_original[idx] = is_tool
            self._is_system_by_original[idx] = role == _SYSTEM_ROLE
            self._is_error_by_original[idx] = is_tool and (
                meta.tool_error is True
                or (meta.exit_code is not None and int(meta.exit_code) != 0)
            )


class NullEvictionCache(BaseEvictionCache):
    """Reserve-all identity cache used only for CPU/probe validation."""

    def __init__(
        self,
        config: EvictionPolicyConfig,
        num_layers: int,
        recorder: "KVEvictionRecorder | None" = None,
    ) -> None:
        super().__init__(config, num_layers, recorder=recorder)
        if config.budget is None or int(config.budget) <= 0:
            raise ValueError(
                f"NullEvictionCache requires positive config.budget; got {config.budget!r}"
            )
        self._original_indices_by_layer: dict[int, list[int]] = {}

    def crop_to_logical_length(self, logical_length: int) -> None:
        length = int(logical_length)
        super().crop_to_logical_length(length)
        for layer, originals in list(self._original_indices_by_layer.items()):
            self._original_indices_by_layer[layer] = [
                int(original) for original in originals if int(original) < length
            ]

    def _decide_evict(self, layer_idx: int, key_len: int) -> EvictionDecision:
        originals = self._ensure_layer_state(int(layer_idx), int(key_len))
        keep = list(range(int(key_len)))
        return EvictionDecision(
            keep_indices=keep,
            evict_indices=[],
            reason="null_identity",
            policy_state={
                "original_kept_indices": originals,
                "original_evicted_indices": [],
            },
        )

    def _ensure_layer_state(self, layer_idx: int, key_len: int) -> list[int]:
        layer = int(layer_idx)
        originals = self._original_indices_by_layer.setdefault(layer, [])
        if key_len > len(originals):
            originals.extend(range(len(originals), key_len))
        return list(originals[:key_len])


class PositionControlCache(BaseEvictionCache):
    """Non-metadata eviction controls for contiguity bracketing."""

    def supports_session_resume(self) -> bool:
        return self.config.position_control != "random"  # random -> per-layer divergent

    def __init__(
        self,
        config: EvictionPolicyConfig,
        num_layers: int,
        recorder: "KVEvictionRecorder | None" = None,
    ) -> None:
        super().__init__(config, num_layers, recorder=recorder)
        if config.budget is None or int(config.budget) <= 0:
            raise ValueError(
                f"PositionControlCache requires positive config.budget; got {config.budget!r}"
            )
        if config.position_control not in {"random", "middle", "structured"}:
            raise ValueError(
                "position_control must be one of random, middle, structured; "
                f"got {config.position_control!r}"
            )
        if config.sink_size < 0 or config.recent_window < 0:
            raise ValueError("position controls require non-negative sink/recent sizes")
        if config.position_control in {"middle", "structured"}:
            floor = int(config.sink_size) + int(config.recent_window)
            if int(config.budget) < floor:
                raise ValueError(
                    "position controls require budget >= sink_size + recent_window "
                    f"for {config.position_control!r}"
                )
        if config.position_control_stride <= 0:
            raise ValueError("position_control_stride must be > 0")
        if config.position_control_cluster_size <= 0:
            raise ValueError("position_control_cluster_size must be > 0")
        self._next_original_by_layer: dict[int, int] = {}
        self._original_indices_by_layer: dict[int, list[int]] = {}
        if config.position_control == "random":
            import torch

            self._generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
        else:
            self._generator = None

    def crop_to_logical_length(self, logical_length: int) -> None:
        length = int(logical_length)
        super().crop_to_logical_length(length)
        for layer, originals in list(self._original_indices_by_layer.items()):
            self._original_indices_by_layer[layer] = [
                int(original) for original in originals if int(original) < length
            ]
            self._next_original_by_layer[layer] = length

    def _decide_evict(self, layer_idx: int, key_len: int) -> EvictionDecision:
        originals = self._ensure_layer_state(int(layer_idx), int(key_len))
        budget = int(self.config.budget)  # type: ignore[arg-type]
        if key_len <= budget:
            keep = list(range(key_len))
            evict: list[int] = []
        elif self.config.position_control == "random":
            keep, evict = self._random_keep(key_len=key_len, budget=budget)
        elif self.config.position_control == "middle":
            keep = self._middle_keep(key_len=key_len, budget=budget)
            evict = sorted(set(range(key_len)) - set(keep))
        else:
            keep = self._structured_keep(key_len=key_len, budget=budget)
            evict = sorted(set(range(key_len)) - set(keep))
        return EvictionDecision(
            keep_indices=keep,
            evict_indices=evict,
            reason=f"pc_{self.config.position_control}",
            policy_state={
                "original_kept_indices": [originals[idx] for idx in keep],
                "original_evicted_indices": [originals[idx] for idx in evict],
            },
        )

    def _post_evict_hook(self, layer_idx: int, decision: EvictionDecision) -> None:
        layer = int(layer_idx)
        originals = self._original_indices_by_layer.get(layer)
        if originals is None:
            return
        self._original_indices_by_layer[layer] = [
            int(originals[idx]) for idx in decision.keep_indices
        ]

    def _ensure_layer_state(self, layer_idx: int, key_len: int) -> list[int]:
        layer = int(layer_idx)
        originals = self._original_indices_by_layer.setdefault(layer, [])
        next_original = self._next_original_by_layer.get(layer, 0)
        if key_len < len(originals):
            raise RuntimeError(
                f"layer {layer} key_len {key_len} is shorter than tracked "
                f"physical origins {len(originals)}"
            )
        if key_len > len(originals):
            delta = key_len - len(originals)
            originals.extend(range(next_original, next_original + delta))
            self._next_original_by_layer[layer] = next_original + delta
        return list(originals)

    def _random_keep(self, *, key_len: int, budget: int) -> tuple[list[int], list[int]]:
        import torch

        if self._generator is None:
            raise RuntimeError("random position control has no generator")
        n_evict = int(key_len) - int(budget)
        perm = torch.randperm(int(key_len), generator=self._generator).tolist()
        evict = sorted(perm[:n_evict])
        keep = sorted(perm[n_evict:])
        return keep, evict

    def _middle_keep(self, *, key_len: int, budget: int) -> list[int]:
        sink = min(int(self.config.sink_size), key_len)
        recent = min(int(self.config.recent_window), key_len)
        keep = set(range(sink))
        keep.update(range(max(0, key_len - recent), key_len))
        remaining = int(budget) - len(keep)
        if remaining <= 0:
            return sorted(keep)
        middle = [idx for idx in range(sink, max(sink, key_len - recent))]
        keep.update(middle[-remaining:])
        return sorted(keep)

    def _structured_keep(self, *, key_len: int, budget: int) -> list[int]:
        sink = min(int(self.config.sink_size), key_len)
        recent = min(int(self.config.recent_window), key_len)
        keep = set(range(sink))
        keep.update(range(max(0, key_len - recent), key_len))
        middle = [idx for idx in range(sink, max(sink, key_len - recent))]
        remaining = int(budget) - len(keep)
        if remaining <= 0 or not middle:
            return sorted(keep)
        for cluster in _clustered_middle_positions(
            middle,
            stride=int(self.config.position_control_stride),
            cluster_size=int(self.config.position_control_cluster_size),
        ):
            for idx in cluster:
                if len(keep) >= budget:
                    break
                keep.add(idx)
            if len(keep) >= budget:
                break
        return sorted(keep)


def _clustered_middle_positions(
    middle: list[int], *, stride: int, cluster_size: int
) -> Iterable[list[int]]:
    cursor = 0
    while cursor < len(middle):
        yield middle[cursor : cursor + cluster_size]
        cursor += stride


__all__ = [
    "MetadataResidencyCache",
    "MetadataResidencySelector",
    "MetadataSelection",
    "NullEvictionCache",
    "PerLayerPriorityTable",
    "PositionControlCache",
    "TokenMetadata",
    "build_token_metadata_from_segments",
    "default_token_metadata",
]
