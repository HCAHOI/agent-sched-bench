"""Pub/sub bus for post-softmax attention tensors.

Publishers emit ``attn`` shaped ``(B, num_q_heads, n_query_rows, key_len)``
where ``num_q_heads`` is query heads (GQA already expanded), ``n_query_rows``
is the number of sampled positions, and dtype matches the attention module
(bf16/fp16). ``query_positions`` is a LongTensor ``(n_query_rows,)`` of
absolute key-axis positions. ``key_len`` is the key axis size at this forward
(post-eviction). ``phase`` is ``"prefill"`` or ``"decode"``.

``full_prefill=True`` chunks are delivered only to consumers with
``prefill_observe_mode="full"`` (H2O paper-faithful prefill accumulation).

When ``suspended=True``, consumers with ``always_active=False`` are skipped;
H2O sets ``always_active=True`` so its score buffer never desyncs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class AttentionConsumer(Protocol):
    """Protocol for downstream consumers of post-softmax attention.

    Implementations expose a class- or instance-level ``always_active`` flag.
    When ``False`` (default), `AttentionBus.publish` skips this consumer while
    `LayerCapturer.suspend_attention()` is active. H2O sets this to ``True``
    so its score accumulator never silently desyncs from the cache state.
    """

    always_active: bool
    prefill_observe_mode: str

    def observe(
        self,
        *,
        layer: int,
        attn: "torch.Tensor",
        query_positions: "torch.Tensor",
        key_len: int,
        phase: str,
    ) -> None:
        ...


class AttentionBus:
    """Synchronous pub/sub for post-softmax attention tensors.

    Pure dispatcher: holds a list of consumers and forwards `publish(...)`
    calls to each. No tensor reshaping, no clones, no async — keeping the
    softmax-once invariant cheap and easy to reason about.
    """

    def __init__(self) -> None:
        self._consumers: list[AttentionConsumer] = []

    def subscribe(self, consumer: AttentionConsumer) -> None:
        self._consumers.append(consumer)

    def unsubscribe(self, consumer: AttentionConsumer) -> None:
        self._consumers.remove(consumer)

    def n_consumers(self) -> int:
        return len(self._consumers)

    def has_full_prefill_consumers(self) -> bool:
        return any(
            str(getattr(consumer, "prefill_observe_mode", "sampled")) == "full"
            for consumer in self._consumers
        )

    def publish(
        self,
        *,
        layer: int,
        attn: "torch.Tensor",
        query_positions: "torch.Tensor",
        key_len: int,
        phase: str,
        suspended: bool,
        full_prefill: bool = False,
    ) -> None:
        if not self._consumers:
            return
        for consumer in self._consumers:
            if phase == "prefill":
                mode = str(getattr(consumer, "prefill_observe_mode", "sampled"))
                if full_prefill and mode != "full":
                    continue
                if not full_prefill and mode == "full":
                    continue
            if suspended and not getattr(consumer, "always_active", False):
                continue
            consumer.observe(
                layer=layer,
                attn=attn,
                query_positions=query_positions,
                key_len=key_len,
                phase=phase,
            )


__all__ = ["AttentionBus", "AttentionConsumer"]
