"""Metadata-driven KV residency policies.

The metadata bridge is intentionally limited to information already present in
the prompt/message stream at the `notify_new_call` boundary. It does not read
task labels, ground truth, future generated text, or recording artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import yaml

from serving.kv_policies.base import (
    BaseEvictionCache,
    EvictionDecision,
    EvictionPolicyConfig,
)

if TYPE_CHECKING:
    from serving.kv_policies.recorder import KVEvictionRecorder


_ROLE_PRIORITY: dict[str, int] = {
    "system": 7,
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

    def select(
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
        if key_len <= budget:
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
        if len(hard_reserved) > budget:
            raise RuntimeError(
                "hard metadata reservation exceeds budget; constructor validation "
                "should have rejected this configuration"
            )

        keep_set: set[int] = set(hard_reserved)
        remaining_slots = budget - len(keep_set)
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

    def _sink_recent_reserved(self, key_len: int) -> set[int]:
        sink = min(int(self.config.sink_size), key_len)
        recent = min(int(self.config.recent_window), key_len)
        keep = set(range(sink))
        keep.update(range(max(0, key_len - recent), key_len))
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
        self._last_metadata_reads_by_layer: dict[int, list[int]] = {}

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
        self._metadata_table.update(
            build_token_metadata_from_segments(
                segments,
                input_token_count=int(input_token_count),
                call_idx=int(call_idx),
            )
        )

    def original_to_current(self, layer_idx: int) -> dict[int, int]:
        """Return the layer-local original->current physical slot map."""
        self._ensure_layer_state(int(layer_idx), self._current_layer_len(layer_idx))
        return dict(self._original_to_current_by_layer[int(layer_idx)])

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

    def assert_last_metadata_reads_within(
        self, layer_idx: int, *, original_limit: int
    ) -> None:
        """Assert the last decision did not read future ORIGINAL token metadata."""
        reads = self._last_metadata_reads_by_layer.get(int(layer_idx), [])
        bad = [idx for idx in reads if int(idx) >= int(original_limit)]
        if bad:
            raise AssertionError(
                f"metadata read future original indices {bad[:8]} "
                f">= limit {original_limit}"
            )

    def _decide_evict(self, layer_idx: int, key_len: int) -> EvictionDecision:
        originals = self._ensure_layer_state(int(layer_idx), int(key_len))
        self._last_metadata_reads_by_layer[int(layer_idx)] = list(originals)
        selection = self._selector.select(
            layer_idx=int(layer_idx),
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

    def _ensure_layer_state(self, layer_idx: int, key_len: int) -> list[int]:
        if key_len < 0:
            raise ValueError(f"key_len must be non-negative, got {key_len}")
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
        self._original_to_current_by_layer[layer] = {
            int(original): current for current, original in enumerate(originals)
        }
        return list(originals)

    def _current_layer_len(self, layer_idx: int) -> int:
        originals = self._original_indices_by_layer.get(int(layer_idx), [])
        return len(originals)


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
