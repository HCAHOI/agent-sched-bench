"""Scaffolding for KV cache eviction policies.

Subclass `BaseEvictionCache` to implement a specific policy. The base provides
the recording hook and physical-drop plumbing; subclasses only fill
`_decide_evict()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from transformers.cache_utils import DynamicCache

if TYPE_CHECKING:
    import torch

    from serving.kv_policies.recorder import KVEvictionRecorder


@dataclass
class EvictionPolicyConfig:
    """User-facing config for a KV eviction policy.

    Fields mirror the plan's Key Interface Sketches table. Subclass-specific
    fields are tolerated as unused for policies that don't need them
    (e.g. `seed` is only consulted by random).
    """

    name: Literal[
        "none",
        "streaming",
        "h2o",
        "random",
        "metadata",
        "position_control",
        "null_eviction",
    ]
    budget: int | None  # required when name != "none"
    sink_size: int = 4  # streaming + h2o
    recent_window: int = 256  # streaming + h2o
    heavy_ratio: float = 0.5  # h2o
    aggregate: Literal["sum", "mean", "ema"] = "sum"  # h2o
    ema_decay: float = 0.9  # h2o (when aggregate="ema")
    seed: int = 0  # random
    record: bool = True  # F15: instrument toggle
    prefill_mode: Literal["sampled", "full"] = "full"  # h2o prefill fidelity
    metadata_rung: Literal["rung1", "rung2", "rung3", "rung4"] = "rung4"
    position_control: Literal["random", "middle", "structured"] = "random"
    position_control_stride: int = 16
    position_control_cluster_size: int = 8
    per_layer_table: bool = False
    per_layer_table_path: str | None = None
    per_layer_budget: bool = False
    # Metadata policy: force system/tool-schema prompt tokens resident outside
    # the competing token budget. This uses only role metadata available at
    # inference time; budget then applies to non-system conversation tokens.
    reserve_system_prompt: bool = True


@dataclass
class EvictionDecision:
    """One layer's keep/evict decision for one update() call.

    Uses plain Python lists so this module stays torch-free at the type level;
    concrete subclasses may carry tensors in `policy_state` for diagnostics.
    """

    keep_indices: list[int]
    evict_indices: list[int]
    reason: str
    policy_state: dict[str, Any] | None = None


class BaseEvictionCache(DynamicCache):
    """`DynamicCache` subclass that enforces a budget via `_decide_evict()`.

    Lifecycle per `update()` call:
      1. delegate to `super().update()` to append the new K/V slots
      2. call `_decide_evict(layer_idx, post_append_len)`
      3. give subclasses one chance to defer the decision when their score is
         only available after the current attention forward (H2O prefill)
      4. if any evictions, physically drop those positions from this layer's
         K/V tensors so `get_seq_length()` returns the post-eviction length
      5. if a recorder is attached and `config.record` is True, record the
         decision via `KVEvictionRecorder.append()`

    Step 2-only: `_decide_evict` is abstract. Step 3+ adds policy subclasses.
    """

    def __init__(
        self,
        config: EvictionPolicyConfig,
        num_layers: int,
        recorder: "KVEvictionRecorder | None" = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_layers = int(num_layers)
        self.recorder = recorder
        self._logical_indices_by_layer: dict[int, list[int]] = {}
        self._next_logical_by_layer: dict[int, int] = {}
        # Per-layer decode-step counter; prefill calls reset to -1.
        # We use a plain dict because the recording layer needs absolute step
        # ids that match attention.npz's record_decode_step semantics.
        self._step_counter: dict[int, int] = {}
        self._seen_prefill: set[int] = set()

    @classmethod
    def requires_attention_backend(cls) -> bool:
        """True if this policy needs post-softmax attention from AttentionBus."""
        return False

    def requires_attention(self) -> bool:
        """Instance-level mirror used for provider unsubscribe lifecycle."""
        return self.requires_attention_backend()

    def notify_new_call(
        self,
        call_idx: int,
        *,
        segments: list[dict[str, Any]] | None = None,
        input_token_count: int | None = None,
    ) -> None:
        """Mark a fresh chat() boundary.

        Resets per-call step counters so the next forward maps to `(prefill, -1)`
        and decode counts from 0 again. Physical KV slots, score buffers, and
        bus subscription persist across calls — only the recording labels reset.
        """
        del call_idx, segments, input_token_count
        self._step_counter.clear()
        self._seen_prefill.clear()

    def update(
        self,
        key_states: "torch.Tensor",
        value_states: "torch.Tensor",
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
        pre_len = int(keys.shape[-2])
        self._ensure_logical_state(int(layer_idx), pre_len)
        query_len = int(key_states.shape[-2])
        phase, step = self._advance_step(layer_idx, query_len=query_len)

        decision = self._decide_evict(layer_idx, pre_len)
        if self._defer_decision(
            layer_idx=layer_idx,
            phase=phase,
            step=step,
            pre_len=pre_len,
            decision=decision,
        ):
            return keys, values
        if decision.evict_indices:
            keys, values = self._physically_drop(layer_idx, decision.keep_indices)
            self._compact_logical_state(int(layer_idx), decision.keep_indices)
            # Subclasses with per-key-position state (H2O score buffer) need to
            # compact that state in lockstep with the K/V drop. Default no-op
            # for stateless policies (streaming, random).
            self._post_evict_hook(layer_idx, decision)
        post_len = int(keys.shape[-2])

        self._record_decision(
            layer_idx=layer_idx,
            phase=phase,
            step=step,
            pre_len=pre_len,
            post_len=post_len,
            decision=decision,
        )
        return keys, values

    def crop_to_logical_length(self, logical_length: int) -> None:
        """Retain only cached tokens with logical/original index < length.

        The provider uses this for true LCP resume-prefill: when a newly
        rendered prompt differs at position ``lcp``, cached K/V before ``lcp``
        remains valid and only the suffix needs prefill. For plain positional
        caches physical and logical indices coincide; eviction policies may
        have physically compacted slots, so the base tracks the logical index
        carried by each physical slot and drops by that mapping.
        """
        length = int(logical_length)
        if length < 0:
            raise ValueError(f"logical_length must be non-negative, got {length}")
        for layer_idx in self._materialized_layer_indices():
            keys, _values = self._get_layer_kv(layer_idx)
            key_len = int(keys.shape[-2])
            logical = self._ensure_logical_state(layer_idx, key_len, copy=False)
            keep = [idx for idx, original in enumerate(logical) if int(original) < length]
            if len(keep) < key_len:
                evict_set = set(keep)
                decision = EvictionDecision(
                    keep_indices=keep,
                    evict_indices=[idx for idx in range(key_len) if idx not in evict_set],
                    reason="logical_prefix_crop",
                )
                self._physically_drop(layer_idx, keep)
                self._compact_logical_state(layer_idx, keep)
                self._post_evict_hook(layer_idx, decision)
            self._next_logical_by_layer[layer_idx] = length

    def _defer_decision(
        self,
        *,
        layer_idx: int,
        phase: str,
        step: int,
        pre_len: int,
        decision: EvictionDecision,
    ) -> bool:
        """Return True when a subclass will finish and record this decision later."""
        del layer_idx, phase, step, pre_len, decision
        return False

    def _record_decision(
        self,
        *,
        layer_idx: int,
        phase: str,
        step: int,
        pre_len: int,
        post_len: int,
        decision: EvictionDecision,
    ) -> None:
        if self.recorder is None or not self.config.record:
            return
        # `policy_state` is the carry slot for policy-specific diagnostics the
        # recorder writes to npz; H2O fills `score_topk_index/value`, other
        # policies leave it None and the recorder fills sentinels.
        policy_state = decision.policy_state or {}
        self.recorder.append(
            step=step,
            layer=int(layer_idx),
            phase=phase,
            pre_len=pre_len,
            post_len=post_len,
            budget=int(self.config.budget) if self.config.budget is not None else -1,
            kept_indices=list(decision.keep_indices),
            evicted_indices=list(decision.evict_indices),
            evict_reason=decision.reason,
            score_topk_index=policy_state.get("score_topk_index"),
            score_topk_value=policy_state.get("score_topk_value"),
            score_evicted_index=policy_state.get("score_evicted_index"),
            score_evicted_value=policy_state.get("score_evicted_value"),
            original_kept_indices=policy_state.get("original_kept_indices"),
            original_evicted_indices=policy_state.get("original_evicted_indices"),
        )

    def _post_evict_hook(self, layer_idx: int, decision: EvictionDecision) -> None:
        """Hook fired after `_physically_drop` succeeds.

        Default: no-op. Subclasses with per-key-position state (H2O's score
        buffer) override to compact that state by `decision.keep_indices` so
        subsequent `observe()` calls write into the right slots.
        """
        del layer_idx, decision
        return

    def _materialized_layer_indices(self) -> list[int]:
        return [idx for idx, layer in enumerate(self.layers) if layer is not None]

    def _ensure_logical_state(
        self, layer_idx: int, key_len: int, *, copy: bool = True
    ) -> list[int]:
        layer = int(layer_idx)
        length = int(key_len)
        if length < 0:
            raise ValueError(f"key_len must be non-negative, got {length}")
        logical = self._logical_indices_by_layer.setdefault(layer, [])
        next_logical = self._next_logical_by_layer.get(layer, 0)
        if length < len(logical):
            raise RuntimeError(
                f"layer {layer} key_len {length} is shorter than tracked logical "
                f"origins {len(logical)}"
            )
        if length > len(logical):
            delta = length - len(logical)
            logical.extend(range(next_logical, next_logical + delta))
            self._next_logical_by_layer[layer] = next_logical + delta
        return list(logical) if copy else logical

    def _compact_logical_state(self, layer_idx: int, keep_indices: list[int]) -> None:
        layer = int(layer_idx)
        logical = self._logical_indices_by_layer.get(layer)
        if logical is None:
            return
        self._logical_indices_by_layer[layer] = [int(logical[idx]) for idx in keep_indices]

    def _decide_evict(self, layer_idx: int, key_len: int) -> EvictionDecision:
        """Return keep/evict decision for `layer_idx` given current `key_len`.

        Abstract — implemented by policy subclasses.
        """
        raise NotImplementedError(
            "BaseEvictionCache._decide_evict is abstract; policy subclasses "
            "(streaming/h2o/random) must implement."
        )

    def _physically_drop(
        self,
        layer_idx: int,
        keep_indices: list[int],
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Replace this layer's K/V tensors with the subset at `keep_indices`."""
        import torch

        keys, values = self._get_layer_kv(layer_idx)
        index = torch.as_tensor(keep_indices, dtype=torch.long, device=keys.device)
        new_keys = keys.index_select(-2, index)
        new_values = values.index_select(-2, index)
        self._set_layer_kv(layer_idx, new_keys, new_values)
        return new_keys, new_values

    def _get_layer_kv(
        self, layer_idx: int
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        layer = self.layers[layer_idx]
        return layer.keys, layer.values

    def _set_layer_kv(
        self,
        layer_idx: int,
        keys: "torch.Tensor",
        values: "torch.Tensor",
    ) -> None:
        layer = self.layers[layer_idx]
        layer.keys = keys
        layer.values = values

    def _advance_step(
        self, layer_idx: int, *, query_len: int
    ) -> tuple[str, int]:
        """Decide (phase, step) for this update() call.

        Phase rule: a multi-token query, or the first call on a layer, is
        prefill. All subsequent single-token queries on the same layer are
        decode steps numbered from 0. Matches LayerCapturer's decode_step
        semantics for cross-artifact alignment.
        """
        if query_len > 1 or layer_idx not in self._seen_prefill:
            self._seen_prefill.add(layer_idx)
            self._step_counter[layer_idx] = 0
            return "prefill", -1
        step = self._step_counter[layer_idx]
        self._step_counter[layer_idx] = step + 1
        return "decode", step
