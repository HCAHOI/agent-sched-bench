"""CPU-testable core for the vLLM connector offload spike.

Everything in this module is pure-Python (stdlib only): timing math, the
transfer-backend protocol, a fake backend for tests, inter-token-latency (ITL)
statistics, and the spike report schema. The GPU/vLLM code lives in ``gpu.py``
and imports from here, so all of the control/timing/reporting logic is unit
testable WITHOUT vllm or a GPU installed.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable


def gbps(bytes_moved: int, milliseconds: float) -> float:
    """Effective bandwidth in GB/s (10^9 bytes) for a transfer.

    Raises:
        ValueError: If ``milliseconds`` is not strictly positive (a zero-time
            transfer is a measurement bug, not a real event -- fail fast).
    """
    if milliseconds <= 0:
        raise ValueError(f"transfer time must be > 0 ms, got {milliseconds}")
    if bytes_moved < 0:
        raise ValueError(f"bytes_moved must be >= 0, got {bytes_moved}")
    return bytes_moved / (milliseconds / 1000.0) / 1e9


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (same convention as numpy default).

    ``pct`` is in [0, 100]. Used for the ITL distribution summary.

    Raises:
        ValueError: On empty input or out-of-range percentile.
    """
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def itl_summary(inter_token_latencies_ms: list[float]) -> dict[str, float]:
    """Summarize an ITL distribution (ms) for the interference measurement."""
    if not inter_token_latencies_ms:
        return {"count": 0}
    return {
        "count": len(inter_token_latencies_ms),
        "mean": sum(inter_token_latencies_ms) / len(inter_token_latencies_ms),
        "p50": percentile(inter_token_latencies_ms, 50),
        "p90": percentile(inter_token_latencies_ms, 90),
        "p99": percentile(inter_token_latencies_ms, 99),
        "max": max(inter_token_latencies_ms),
    }


@dataclass
class TransferTiming:
    """Result of one KV-block transfer (offload or restore direction)."""

    bytes_moved: int
    milliseconds: float
    num_blocks: int

    @property
    def effective_gbps(self) -> float:
        return gbps(self.bytes_moved, self.milliseconds)


@runtime_checkable
class TransferBackend(Protocol):
    """A KV-block transfer backend.

    The real backend (``gpu.CudaBlockTransfer``) copies paged KV blocks between
    the GPU cache and pinned host memory with CUDA-event timing. The fake
    backend below simulates it deterministically for CPU tests.
    """

    def offload(self, block_ids: list[int]) -> TransferTiming:
        """Move the given KV blocks GPU -> host, timed."""
        ...

    def restore(self, block_ids: list[int]) -> TransferTiming:
        """Move the given KV blocks host -> GPU, timed."""
        ...


@dataclass
class FakeTransferBackend:
    """Deterministic in-memory backend for CPU tests (no torch/GPU).

    Models a fixed device->host and host->device bandwidth so the timing/report
    math can be exercised end to end. ``bytes_per_block`` matches the real per
    -block KV byte size the connector would move.
    """

    bytes_per_block: int
    offload_gbps: float = 20.0  # D2H over PCIe gen4 x16, ballpark
    restore_gbps: float = 18.0  # H2D typically a touch slower
    offloaded: set[int] = field(default_factory=set)

    def _timing(self, block_ids: list[int], rate_gbps: float) -> TransferTiming:
        if not block_ids:
            raise ValueError("refusing to time a transfer of zero blocks")
        nbytes = len(block_ids) * self.bytes_per_block
        ms = (nbytes / (rate_gbps * 1e9)) * 1000.0
        return TransferTiming(bytes_moved=nbytes, milliseconds=ms, num_blocks=len(block_ids))

    def offload(self, block_ids: list[int]) -> TransferTiming:
        self.offloaded.update(block_ids)
        return self._timing(block_ids, self.offload_gbps)

    def restore(self, block_ids: list[int]) -> TransferTiming:
        missing = [b for b in block_ids if b not in self.offloaded]
        if missing:
            raise ValueError(f"restoring blocks never offloaded: {missing}")
        self.offloaded.difference_update(block_ids)
        return self._timing(block_ids, self.restore_gbps)


def validate_block_ids(num_blocks_in_dim: int, block_ids: list[int]) -> None:
    """Bounds-check block ids against the size of the (moved-to-front) block dim.

    Raises (not a bare ``assert``, so it survives ``python -O``):
        ValueError: If ``block_ids`` is empty or any id is outside
            ``[0, num_blocks_in_dim)`` -- an out-of-range id would silently read
            or overwrite another request's KV blocks.
    """
    if not block_ids:
        raise ValueError("no block ids provided")
    bad = [b for b in block_ids if b < 0 or b >= num_blocks_in_dim]
    if bad:
        raise ValueError(
            f"block ids out of range for dim size {num_blocks_in_dim}: {bad}"
        )


@dataclass
class RepetitionResult:
    """One offload/restore cycle under load.

    ``seed`` is provenance-only: the driver runs greedy decoding
    (``temperature=0.0``), which is deterministic and ignores the sampling
    seed. It is recorded so a rerun's request ids/prompts are reproducible,
    not because it affects the generated tokens.
    """

    repetition: int
    seed: int
    tool_duration_s: float
    num_blocks: int
    bytes_moved: int
    offload_ms: float
    restore_ms: float
    offload_gbps: float
    restore_gbps: float

    @classmethod
    def from_timings(
        cls,
        repetition: int,
        seed: int,
        tool_duration_s: float,
        offload: TransferTiming,
        restore: TransferTiming,
    ) -> RepetitionResult:
        if offload.num_blocks != restore.num_blocks:
            raise ValueError(
                f"offload/restore block count mismatch: "
                f"{offload.num_blocks} vs {restore.num_blocks}"
            )
        if offload.bytes_moved != restore.bytes_moved:
            raise ValueError("offload/restore byte count mismatch")
        return cls(
            repetition=repetition,
            seed=seed,
            tool_duration_s=tool_duration_s,
            num_blocks=offload.num_blocks,
            bytes_moved=offload.bytes_moved,
            offload_ms=offload.milliseconds,
            restore_ms=restore.milliseconds,
            offload_gbps=offload.effective_gbps,
            restore_gbps=restore.effective_gbps,
        )


def env_info() -> dict[str, object]:
    """Reproducibility metadata. GPU fields filled in by gpu.py when available."""
    return {"host": platform.node(), "python": platform.python_version()}


@dataclass
class SpikeReport:
    """Full spike output: config, per-repetition results, and interference."""

    model: str
    vllm_version: str | None
    kv_seam: str
    num_load_requests: int
    repetitions: list[RepetitionResult]
    # ITL (ms) of the co-running load requests, with and without an offload
    # event overlapping their generation -- the interference measurement.
    itl_with_offload_ms: list[float]
    itl_without_offload_ms: list[float]
    env: dict[str, object] = field(default_factory=env_info)

    def summary(self) -> dict[str, object]:
        reps = self.repetitions
        if not reps:
            raise ValueError("no repetitions to summarize")
        return {
            "offload_ms": {
                "median": percentile([r.offload_ms for r in reps], 50),
                "p99": percentile([r.offload_ms for r in reps], 99),
            },
            "restore_ms": {
                "median": percentile([r.restore_ms for r in reps], 50),
                "p99": percentile([r.restore_ms for r in reps], 99),
            },
            "offload_gbps_median": percentile([r.offload_gbps for r in reps], 50),
            "restore_gbps_median": percentile([r.restore_gbps for r in reps], 50),
            "bytes_moved": reps[0].bytes_moved,
            "interference": {
                "with_offload": itl_summary(self.itl_with_offload_ms),
                "without_offload": itl_summary(self.itl_without_offload_ms),
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "vllm_version": self.vllm_version,
            "kv_seam": self.kv_seam,
            "num_load_requests": self.num_load_requests,
            "env": self.env,
            "repetitions": [asdict(r) for r in self.repetitions],
            "itl_with_offload_ms": self.itl_with_offload_ms,
            "itl_without_offload_ms": self.itl_without_offload_ms,
            "summary": self.summary(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
