"""vLLM connector KV-offload spike (W1).

Importing this package does NOT import torch/vllm -- only the CPU-safe control,
timing, and reporting logic. Import :mod:`spike.vllm_connector.gpu` explicitly
on the GPU box for the connector + CUDA transfer.
"""

from __future__ import annotations

from .control import ControlState, OffloadControl, OffloadPhase
from .core import (
    FakeTransferBackend,
    PauseBook,
    PauseResult,
    RepetitionResult,
    SavedKVRegistry,
    SaveConfirmationReader,
    SpikeReport,
    TransferBackend,
    TransferTiming,
    assert_saved_covers_tokens,
    cdiv,
    chunk_ranges,
    gbps,
    itl_summary,
    matched_tokens_delta,
    percentile,
    resume_load_block_ids,
    validate_block_ids,
    validate_staging_capacity,
)

__all__ = [
    "ControlState",
    "OffloadControl",
    "OffloadPhase",
    "FakeTransferBackend",
    "PauseBook",
    "PauseResult",
    "RepetitionResult",
    "SavedKVRegistry",
    "SaveConfirmationReader",
    "SpikeReport",
    "TransferBackend",
    "TransferTiming",
    "assert_saved_covers_tokens",
    "cdiv",
    "chunk_ranges",
    "gbps",
    "itl_summary",
    "matched_tokens_delta",
    "percentile",
    "resume_load_block_ids",
    "validate_block_ids",
    "validate_staging_capacity",
]
