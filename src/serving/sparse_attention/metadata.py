"""Observe-only sparse sidecar for metadata residency would-keep sets."""

from __future__ import annotations

from typing import Any

from serving.kv_policies.base import EvictionPolicyConfig
from serving.kv_policies.metadata import (
    MetadataResidencySelector,
    TokenMetadata,
    build_token_metadata_from_segments,
)
from serving.sparse_attention.base import SparseAttentionConfig, SparseAttentionContext


class MetadataResidencySparseAttention:
    """Record metadata-residency selections without enforcing a mask."""

    name = "metadata"
    observe_only = True
    requires_full_prefill = False

    def __init__(
        self,
        *,
        budget: int,
        sink_size: int,
        recent_window: int,
        metadata_rung: str,
    ) -> None:
        cfg = EvictionPolicyConfig(
            name="metadata",
            budget=int(budget),
            sink_size=int(sink_size),
            recent_window=int(recent_window),
            metadata_rung=metadata_rung,  # type: ignore[arg-type]
        )
        self._selector = MetadataResidencySelector(cfg)
        self.budget = int(budget)
        self.sink_size = int(sink_size)
        self.recent_window = int(recent_window)
        self.metadata_rung = str(metadata_rung)
        self._metadata_table: dict[int, TokenMetadata] = {}
        self._last_kept_count = 0
        self._last_metadata: dict[str, Any] = {}

    @classmethod
    def from_config(
        cls, config: SparseAttentionConfig
    ) -> "MetadataResidencySparseAttention":
        if not config.observe_only:
            raise ValueError("metadata sparse sidecar is observe-only")
        if config.budget is None:
            raise ValueError("metadata sparse sidecar requires budget")
        return cls(
            budget=int(config.budget),
            sink_size=int(config.sink_size),
            recent_window=int(config.recent_window),
            metadata_rung=str(config.metadata_rung),
        )

    def notify_new_call(
        self,
        *,
        call_idx: int,
        segments: list[dict[str, Any]] | None = None,
        input_token_count: int | None = None,
    ) -> None:
        if segments is None or input_token_count is None:
            return
        self._metadata_table.update(
            build_token_metadata_from_segments(
                segments,
                input_token_count=int(input_token_count),
                call_idx=int(call_idx),
            )
        )

    def build_additive_mask(
        self,
        *,
        layer_idx: int,
        query_len: int,
        key_len: int,
        phase: str,
        decode_step: int = -1,
        device: Any,
        dtype: Any,
        context: SparseAttentionContext | None = None,
    ) -> None:
        del query_len, device, dtype
        original_indices = self._original_indices_from_context(
            context=context,
            layer_idx=int(layer_idx),
            key_len=int(key_len),
        )
        selection = self._selector.select(
            layer_idx=int(layer_idx),
            key_len=int(key_len),
            original_indices=original_indices,
            metadata_table=self._metadata_table,
        )
        self._last_kept_count = len(selection.keep_indices)
        sink_recent = self._sink_recent_set(int(key_len))
        selected_middle = [
            idx for idx in selection.keep_indices if int(idx) not in sink_recent
        ]
        self._last_metadata = {
            "budget": self.budget,
            "sink_size": self.sink_size,
            "recent_window": self.recent_window,
            "metadata_rung": self.metadata_rung,
            "selection_reason": selection.reason,
            "selected_count": len(selection.keep_indices),
            "selected_indices": list(selection.keep_indices),
            "evicted_indices": list(selection.evict_indices),
            "original_selected_indices": list(selection.original_kept_indices),
            "original_evicted_indices": list(selection.original_evicted_indices),
            "selected_middle_indices": selected_middle,
            "phase": str(phase),
            "decode_step": int(decode_step),
        }
        return None

    def _original_indices_from_context(
        self,
        *,
        context: SparseAttentionContext | None,
        layer_idx: int,
        key_len: int,
    ) -> list[int]:
        if context is None or context.past_key_values is None:
            return list(range(key_len))
        cache = context.past_key_values
        original_indices_for_layer = getattr(cache, "original_indices_for_layer", None)
        if callable(original_indices_for_layer):
            originals = [
                int(idx)
                for idx in original_indices_for_layer(int(layer_idx), int(key_len))
            ]
            if len(originals) != int(key_len):
                raise RuntimeError(
                    "metadata sparse sidecar received an original-index map "
                    f"of length {len(originals)} for key_len={key_len}"
                )
            return originals

        cache_config = getattr(cache, "config", None)
        cache_name = getattr(cache_config, "name", None)
        if cache_name in {"streaming", "h2o", "random", "position_control"}:
            raise RuntimeError(
                "metadata sparse sidecar cannot run beside a compacting KV "
                f"policy without an original-index map (kv_policy={cache_name!r})"
            )
        return list(range(key_len))

    def _sink_recent_set(self, key_len: int) -> set[int]:
        sink = min(self.sink_size, key_len)
        recent = min(self.recent_window, key_len)
        keep = set(range(sink))
        keep.update(range(max(0, key_len - recent), key_len))
        return keep

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

    def reset_state(self) -> None:
        self._last_kept_count = 0
        self._last_metadata = {}


__all__ = ["MetadataResidencySparseAttention"]
