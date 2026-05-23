"""Sparse attention methods for the HF recording path.

First-release method: `sliding` (sink prefix + recent tail). Mirrors
`serving.kv_policies` layout one-for-one but addresses an orthogonal axis
of the design space — sparse attention masks key positions per query
instead of physically dropping K/V slots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from serving.sparse_attention.base import (
    BaseSparseAttention,
    SparseAttentionConfig,
)
from serving.sparse_attention.recorder import SparseAttentionRecorder

if TYPE_CHECKING:
    pass


def build_sparse_attention(
    config: SparseAttentionConfig,
    num_layers: int,
    recorder: SparseAttentionRecorder | None = None,
) -> BaseSparseAttention:
    """Factory for the configured sparse attention method.

    Callers must gate on `config is not None` upstream — the `"none"`
    method is a CLI-layer concept (no method instance needed). The
    `recorder` is the per-call audit sink; the provider swaps it before
    each `recording_session()` and the pre-hook reads it via closure.
    The `num_layers` argument is reserved for future methods whose state
    is per-layer (Quest / DuoAttention); sliding ignores it.
    """
    del num_layers, recorder  # unused for sliding; reserved for future methods
    name = config.name
    if name == "sliding":
        from serving.sparse_attention.sliding import SlidingWindowSparseAttention

        method = SlidingWindowSparseAttention.from_config(config)
        return method
    if name == "none":
        raise ValueError(
            "build_sparse_attention should not be called when method is disabled "
            "(config.name == 'none'); gate on `sparse_attention_config is not None`"
        )
    raise NotImplementedError(f"sparse attention method {name!r} not registered")


__all__ = [
    "BaseSparseAttention",
    "SparseAttentionConfig",
    "SparseAttentionRecorder",
    "build_sparse_attention",
]
