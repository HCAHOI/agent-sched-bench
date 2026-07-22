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
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Protocol, Sequence, runtime_checkable


def continuum_priority(
    program_index: int,
    stride: int,
    *,
    ttl_hit: bool,
    preempted: bool = False,
) -> int:
    """Continuum priority tuple encoded as one integer (smaller runs first)."""
    if stride <= 0 or not 0 <= program_index < stride:
        raise ValueError("program_index must be in [0, stride)")
    category = -1 if preempted else (0 if ttl_hit else 1)
    return category * stride + program_index


def thunderagent_reasoning_pauses(
    candidates: Sequence[tuple[str, int]],
    *,
    required_tokens: int,
    capacity_tokens: int,
    buffer_per_program: int,
) -> list[str]:
    """Mirror the pinned overflow loop: mark every REASONING candidate."""
    if min(required_tokens, capacity_tokens, buffer_per_program) < 0:
        raise ValueError("capacity accounting inputs must be >= 0")
    if len({program_id for program_id, _ in candidates}) != len(candidates):
        raise ValueError("reasoning candidates must have unique program ids")
    if any(not program_id or tokens < 0 for program_id, tokens in candidates):
        raise ValueError("reasoning candidates need non-empty ids and tokens >= 0")
    if required_tokens <= capacity_tokens:
        return []
    return [
        program_id
        for program_id, _ in sorted(candidates, key=lambda item: (item[1], item[0]))
    ]


def thunderagent_resume_admissions(
    candidates: Sequence[tuple[str, int]],
    *,
    capacity_tokens: int,
    buffer_per_program: int,
) -> list[str]:
    """Select the maximal small-first fit, then BFD-order admitted programs."""
    if min(capacity_tokens, buffer_per_program) < 0:
        raise ValueError("capacity accounting inputs must be >= 0")
    if len({program_id for program_id, _ in candidates}) != len(candidates):
        raise ValueError("resume candidates must have unique program ids")
    if any(not program_id or tokens < 0 for program_id, tokens in candidates):
        raise ValueError("resume candidates need non-empty ids and tokens >= 0")
    selected: list[tuple[str, int]] = []
    remaining = capacity_tokens
    for candidate in sorted(candidates, key=lambda item: (item[1], item[0])):
        required = candidate[1] + buffer_per_program
        if required <= remaining:
            selected.append(candidate)
            remaining -= required
    return [
        program_id
        for program_id, _ in sorted(selected, key=lambda item: (-item[1], item[0]))
    ]


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
    # Mirror the staged GPU backend's knobs so the same staging/chunk logic is
    # exercised off-GPU: ``max_blocks`` bounds the pre-allocated staging buffer,
    # ``chunk_bytes`` splits each transfer via ``chunk_ranges``.
    max_blocks: int | None = None
    chunk_bytes: int = 0
    offloaded: set[int] = field(default_factory=set)
    last_num_chunks: int = 0

    def _timing(self, block_ids: list[int], rate_gbps: float) -> TransferTiming:
        if not block_ids:
            raise ValueError("refusing to time a transfer of zero blocks")
        if self.max_blocks is not None:
            validate_staging_capacity(self.max_blocks, len(block_ids))
        self.last_num_chunks = len(
            chunk_ranges(len(block_ids), self.bytes_per_block, self.chunk_bytes)
        )
        nbytes = len(block_ids) * self.bytes_per_block
        ms = (nbytes / (rate_gbps * 1e9)) * 1000.0
        return TransferTiming(
            bytes_moved=nbytes, milliseconds=ms, num_blocks=len(block_ids)
        )

    # Dedicated per-request retained buffers (pause-save survives eviction and
    # is NOT recycled into the shared staging pool until the resume-load frees
    # it). Mirrors the GPU backend's ``save_retained`` / ``load_retained``.
    retained: dict[str, list[int]] = field(default_factory=dict)

    def offload(self, block_ids: list[int]) -> TransferTiming:
        self.offloaded.update(block_ids)
        return self._timing(block_ids, self.offload_gbps)

    def restore(self, block_ids: list[int]) -> TransferTiming:
        missing = [b for b in block_ids if b not in self.offloaded]
        if missing:
            raise ValueError(f"restoring blocks never offloaded: {missing}")
        self.offloaded.difference_update(block_ids)
        return self._timing(block_ids, self.restore_gbps)

    def save_retained(self, request_id: str, block_ids: list[int]) -> TransferTiming:
        if request_id in self.retained:
            raise ValueError(f"request {request_id!r} already has a retained save")
        self.retained[request_id] = list(block_ids)
        return self._timing(block_ids, self.offload_gbps)

    def load_retained(self, request_id: str, block_ids: list[int]) -> TransferTiming:
        if request_id not in self.retained:
            raise ValueError(f"no retained save for request {request_id!r}")
        saved = self.retained.pop(request_id)  # freed after resume-load
        if len(saved) != len(block_ids):
            raise ValueError(
                f"resume-load block count {len(block_ids)} != saved {len(saved)}"
            )
        return self._timing(block_ids, self.restore_gbps)

    def free_retained(self, request_id: str) -> None:
        """Drop a retained save without loading it.

        Used when the request finishes/aborts while paused, or resumes for
        free via the local prefix cache (no load needed) -- either way the
        buffer must not leak. No-op if nothing is retained for the id.
        """
        self.retained.pop(request_id, None)


def validate_staging_capacity(max_blocks: int, num_blocks: int) -> None:
    """Fail fast if a transfer would overrun the pre-allocated staging buffer.

    The staged transfer path sizes one pinned host buffer (and a device gather
    buffer) for ``max_blocks`` at ``register_kv_caches`` time. A request that
    resolved to more blocks than that would silently write past the buffer, so
    reject it rather than corrupt memory.

    Raises:
        ValueError: If ``num_blocks`` exceeds ``max_blocks`` or is non-positive.
    """
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
    if num_blocks > max_blocks:
        raise ValueError(
            f"transfer of {num_blocks} blocks exceeds staging capacity "
            f"{max_blocks}; raise --staging-max-blocks"
        )


def chunk_ranges(
    num_blocks: int, bytes_per_block: int, chunk_bytes: int
) -> list[tuple[int, int]]:
    """Split ``num_blocks`` into ``[start, stop)`` ranges of <= ``chunk_bytes``.

    The staged transfer copies blocks in these ranges so no single copy moves
    more than ``chunk_bytes`` (a rate limiter that lets co-tenant work slip
    between chunks). ``chunk_bytes <= 0`` disables chunking (one range). A block
    is the atom -- if one block alone exceeds ``chunk_bytes`` we still move a
    whole block per chunk rather than subdivide it.

    Raises:
        ValueError: On non-positive ``num_blocks``, or non-positive
            ``bytes_per_block`` when chunking is requested.
    """
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
    if chunk_bytes <= 0:
        return [(0, num_blocks)]
    if bytes_per_block <= 0:
        raise ValueError(f"bytes_per_block must be > 0, got {bytes_per_block}")
    per_chunk = max(1, chunk_bytes // bytes_per_block)
    return [
        (i, min(i + per_chunk, num_blocks)) for i in range(0, num_blocks, per_chunk)
    ]


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


# --- W1 offload/restore target tracking (GPU-validation bugfix) -------------
#
# GPU run found: staged mode's async transfer no longer stalls the forward
# pass (fix #2), so the offloaded request's own decode now races ahead
# unthrottled. Offload does NOT pause the request (documented limitation), so
# its live block-id list keeps growing after OFFLOAD -- and can go away
# entirely if the request finishes before RESTORE's control-file epoch is
# observed. build_connector_meta previously re-read that LIVE list for
# RESTORE too, instead of the exact blocks OFFLOAD actually staged host-side:
# wrong (possibly larger, or gone) block ids for restore. Symmetric with
# SavedKVRegistry below (P4 pause/resume solves the identical class of bug for
# the evict scenario) -- this is the W1 copy-scenario counterpart.


@dataclass
class OffloadedBlocks:
    """Connector-side registry of the exact block ids OFFLOAD last staged.

    ``record`` is called when an OFFLOAD directive is emitted; ``pop`` is
    called when a RESTORE directive is being built and returns the pinned
    list (one-shot) so RESTORE always operates on precisely what was staged,
    never a live/grown/absent list -- ``None`` means "nothing to restore"
    (never offloaded, already restored, or the request finished first).
    ``drop`` forgets a pending record without restoring (the request finished
    mid-offload -- its blocks are about to be freed/reassigned, restoring into
    them would corrupt a co-tenant's KV).
    """

    offloaded: dict[str, list[int]] = field(default_factory=dict)

    def record(self, request_id: str, block_ids: list[int]) -> None:
        self.offloaded[request_id] = list(block_ids)

    def pop(self, request_id: str) -> list[int] | None:
        return self.offloaded.pop(request_id, None)

    def drop(self, request_id: str) -> None:
        self.offloaded.pop(request_id, None)


def check_repetition_transfers(
    offload_rows: list[dict[str, object]],
    restore_rows: list[dict[str, object]],
    skip_rows: list[dict[str, object]],
    rep: int,
) -> None:
    """Fail fast on an incomplete repetition, with an accurate diagnosis.

    A ``restore_skipped`` row means the connector correctly refused to
    restore (the target already finished before the RESTORE trigger could
    fire -- the documented offload-does-not-pause liveness race, not a broken
    transfer seam). Anything else missing means the seam itself did not fire.

    Raises:
        RuntimeError: On an incomplete repetition, in either case above.
    """
    if offload_rows and restore_rows:
        return
    if skip_rows:
        reason = skip_rows[-1].get("reason", "unknown")
        raise RuntimeError(
            f"rep {rep}: restore skipped ({reason}) -- the target request "
            "finished before the restore trigger fired (offload does not "
            "pause the request; see README risk #2), not a broken seam. "
            "Raise --max-tokens or lower --tool-duration-s so the agent "
            "request cannot finish before restore fires."
        )
    raise RuntimeError(
        f"rep {rep}: connector recorded no transfer "
        f"(offload={len(offload_rows)}, restore={len(restore_rows)}); "
        "the seam did not fire -- see README risk section"
    )


# --- P4 pause/resume pure logic (scheduler + connector bookkeeping) ---------
#
# All GPU-free: the state machine, the delta-only matched-tokens math, the
# saved-KV registry lifecycle, and the file-based save-confirmation handshake.
# The vllm-dependent PausableScheduler (scheduler.py) and the connector
# (gpu.py) are thin wrappers over these, so the correctness-bearing logic is
# unit tested off-GPU.


def cdiv(a: int, b: int) -> int:
    """Ceiling division (blocks needed to cover ``a`` tokens of ``b``/block)."""
    if b <= 0:
        raise ValueError(f"block size must be > 0, got {b}")
    return -(-a // b)


def assert_saved_covers_tokens(
    num_tokens: int, block_count: int, block_size: int
) -> None:
    """Fail fast unless ``block_count`` blocks exactly cover ``num_tokens``.

    The pause instant's invariant (memo Q4): the KV we save must be exactly the
    request's ``num_computed_tokens`` -- no more, no less -- or resume mis-slices
    the sequence. ``block_count`` whole blocks cover ``num_tokens`` iff
    ``block_count == ceil(num_tokens / block_size)``.

    Raises:
        ValueError: on non-positive tokens or a block count that under- or
            over-covers the computed tokens.
    """
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be > 0 at pause, got {num_tokens}")
    expected = cdiv(num_tokens, block_size)
    if block_count != expected:
        raise ValueError(
            f"saved {block_count} blocks do not cover {num_tokens} tokens "
            f"(expected {expected} at block_size {block_size})"
        )


def resume_load_block_ids(
    new_block_ids: list[int], saved_block_count: int
) -> list[int]:
    """Slice the scheduler's full prefix block set down to the saved blocks.

    v0.11.2's ``update_state_after_alloc`` hands the connector the FULL prefix
    block set (``new_computed_blocks + new_blocks``), not just the external
    delta. Because the pause save is confirmed 1+ steps AFTER the pause trigger
    (the save-confirm lag -- see P4_EVICTION_DESIGN.md appendix), the request
    keeps running for that lag and can accrue extra computed tokens the save
    never covered. When that lag's suffix crosses a block boundary,
    ``len(new_block_ids) > saved_block_count`` -- the retained buffer only holds
    the saved prefix, so the resume-load must use only its first
    ``saved_block_count`` blocks; the scheduler recomputes the remaining
    (unsaved) suffix into the rest on its own.

    Raises:
        ValueError: if the scheduler allocated FEWER blocks than the save
            covers -- that would mean the connector is about to under-load,
            not a lag artifact.
    """
    if len(new_block_ids) < saved_block_count:
        raise ValueError(
            f"scheduler allocated {len(new_block_ids)} blocks, fewer than the "
            f"{saved_block_count} blocks the pause-save covers"
        )
    return new_block_ids[:saved_block_count]


def matched_tokens_delta(saved_tokens: int, num_local_computed: int) -> int:
    """Extra tokens the connector supplies BEYOND the local prefix-cache hit.

    ``get_num_new_matched_tokens`` must return only tokens beyond
    ``num_local_computed`` (scheduler.py:457) or the scheduler double-allocates
    (memo Q2 self-hit). Under low pressure the request's own just-freed blocks
    are still prefix-cached, so ``num_local_computed`` can already cover the
    whole save -> delta 0 (resume for free); under real pressure the delta is
    the whole save.
    """
    if saved_tokens < 0 or num_local_computed < 0:
        raise ValueError("token counts must be >= 0")
    return max(0, saved_tokens - num_local_computed)


@dataclass(frozen=True)
class RetainedPrefix:
    """One finished turn whose prompt KV is still reusable."""

    program_id: str
    request_id: str
    num_tokens: int
    resident_tokens: int
    block_ids: tuple[int, ...]
    block_hashes: tuple[object, ...]
    created_at: float
    expires_at: float | None
    action: str
    policy: str
    program_arrival_s: float = 0.0

    @property
    def block_count(self) -> int:
        return len(self.block_ids)


@dataclass(frozen=True)
class RetentionMatch:
    """Side-effect-free prefix lookup result for a follow-up request."""

    source: str
    record: RetainedPrefix | None
    local_blocks: int = 0

    @property
    def external_blocks(self) -> int:
        if self.source not in {"resident", "host"} or self.record is None:
            return 0
        return self.record.block_count - self.local_blocks


@dataclass
class FinishedRetentionBook:
    """Scheduler-side lifecycle for finished-turn GPU and host prefixes."""

    resident: dict[str, RetainedPrefix] = field(default_factory=dict)
    host: dict[str, RetainedPrefix] = field(default_factory=dict)

    def register(self, record: RetainedPrefix) -> None:
        if record.action not in {"offload", "release", "pressure"}:
            raise ValueError(f"unsupported retention action {record.action!r}")
        if not record.program_id or not record.request_id:
            raise ValueError("program_id and request_id must be non-empty")
        if record.num_tokens <= 0:
            raise ValueError("retained token count must be > 0")
        if record.resident_tokens < record.num_tokens:
            raise ValueError(
                "resident token capacity cannot be smaller than reusable KV"
            )
        if not record.block_ids or len(record.block_ids) != len(record.block_hashes):
            raise ValueError(
                "retained block ids and hashes must be non-empty and aligned"
            )
        if record.program_id in self.resident or record.program_id in self.host:
            raise ValueError(
                f"program {record.program_id!r} already has an unconsumed prefix"
            )
        self.resident[record.program_id] = record

    def due(
        self, now: float, *, waiting_program_ids: set[str] | None = None
    ) -> list[RetainedPrefix]:
        """Expired timer-based records whose program is not already waiting."""
        waiting_program_ids = waiting_program_ids or set()
        return sorted(
            (
                record
                for record in self.resident.values()
                if record.expires_at is not None
                and record.expires_at <= now
                and record.program_id not in waiting_program_ids
            ),
            key=lambda record: (record.expires_at, record.program_id),
        )

    def move_to_host(self, program_id: str) -> RetainedPrefix:
        record = self.resident.pop(program_id)
        if record.action != "offload":
            raise ValueError(
                f"cannot move {record.action!r} retention for {program_id!r} to host"
            )
        self.host[program_id] = record
        return record

    def drop(self, program_id: str) -> RetainedPrefix | None:
        record = self.resident.pop(program_id, None)
        return record if record is not None else self.host.pop(program_id, None)

    def match(
        self,
        program_id: str,
        block_hashes: list[object] | tuple[object, ...],
        *,
        num_local_tokens: int,
        block_size: int,
    ) -> RetentionMatch:
        """Match a follow-up prompt without mutating retained state."""
        if num_local_tokens < 0 or block_size <= 0:
            raise ValueError("num_local_tokens must be >= 0 and block_size must be > 0")
        source = "resident" if program_id in self.resident else "host"
        record = self.resident.get(program_id) or self.host.get(program_id)
        if record is None:
            return RetentionMatch("none", None)
        prefix = tuple(block_hashes[: record.block_count])
        if prefix != record.block_hashes:
            return RetentionMatch("mismatch", record)
        local_blocks = min(num_local_tokens // block_size, record.block_count)
        return RetentionMatch(source, record, local_blocks)

    def continuum_pressure_evictions(
        self,
        *,
        active_tokens: int,
        waiting_tokens: int,
        capacity_tokens: int,
        shared_tokens: int = 0,
        protected_program_ids: set[str] | None = None,
    ) -> list[RetainedPrefix]:
        """Unpin latest-arriving Continuum programs until capacity fits."""
        if min(active_tokens, waiting_tokens, capacity_tokens, shared_tokens) < 0:
            raise ValueError("capacity accounting inputs must be >= 0")
        all_retained = [
            record
            for record in self.resident.values()
            if record.policy == "continuum" and record.action == "release"
        ]
        candidates = [
            record
            for record in all_retained
            if protected_program_ids is None
            or record.program_id not in protected_program_ids
        ]
        required = max(
            0,
            active_tokens
            + waiting_tokens
            + sum(record.resident_tokens for record in all_retained)
            - shared_tokens,
        )
        evicted: list[RetainedPrefix] = []
        for record in sorted(
            candidates,
            key=lambda item: (
                -item.program_arrival_s,
                -item.created_at,
                item.program_id,
            ),
        ):
            if required <= capacity_tokens:
                break
            required -= record.resident_tokens
            evicted.append(record)
        return evicted

    def pressure_evictions(
        self,
        *,
        reasoning_tokens: int,
        reasoning_programs: int,
        capacity_tokens: int,
        buffer_per_program: int,
        shared_tokens: int = 0,
        reasoning_program_ids: set[str] | None = None,
        waiting_tokens: int = 0,
        waiting_programs: int = 0,
    ) -> list[RetainedPrefix]:
        """ThunderAgent's smallest-ACTING-first capacity rule."""
        if (
            min(
                reasoning_tokens,
                reasoning_programs,
                capacity_tokens,
                buffer_per_program,
                shared_tokens,
                waiting_tokens,
                waiting_programs,
            )
            < 0
        ):
            raise ValueError("capacity accounting inputs must be >= 0")
        acting = sorted(
            (
                record
                for record in self.resident.values()
                if record.action == "pressure"
                and (
                    reasoning_program_ids is None
                    or record.program_id not in reasoning_program_ids
                )
            ),
            key=lambda record: (
                record.resident_tokens,
                record.created_at,
                record.program_id,
            ),
        )
        acting_tokens = sum(record.resident_tokens for record in acting)
        required = max(
            0,
            reasoning_tokens
            + waiting_tokens
            + acting_tokens
            - shared_tokens
            + (reasoning_programs + len(acting) + waiting_programs)
            * buffer_per_program,
        )
        evicted: list[RetainedPrefix] = []
        for record in acting:
            if required <= capacity_tokens:
                break
            required -= record.resident_tokens + buffer_per_program
            evicted.append(record)
        return evicted


@dataclass
class PauseBook:
    """Scheduler-side pause state machine (memo Q3 hold-set).

    A request moves RUNNING -> ``pausing`` (save emitted, blocks NOT yet freed)
    -> ``paused`` (save confirmed, blocks freed, held in no queue) -> resumed.
    Both dicts map ``request_id -> num_computed_tokens`` captured at the pause
    instant; the value is what resume restores as the saved token count. The
    dicts double as the fail-fast bookkeeping (double-pause / resume-unknown
    raise).
    """

    pausing: dict[str, int] = field(default_factory=dict)
    paused: dict[str, int] = field(default_factory=dict)
    # RESUME arrived while still pausing (save not yet confirmed) -- honored
    # immediately after confirm_saved instead of being dropped (memo Q3: a
    # dropped RESUME here would hang the request paused forever).
    resume_deferred: set[str] = field(default_factory=set)

    def mark_pausing(self, request_id: str, num_computed_tokens: int) -> None:
        if request_id in self.pausing or request_id in self.paused:
            raise ValueError(f"request {request_id!r} already pausing/paused")
        if num_computed_tokens <= 0:
            raise ValueError(f"cannot pause with {num_computed_tokens} computed tokens")
        self.pausing[request_id] = num_computed_tokens

    def confirm_saved(self, request_id: str) -> int:
        """pausing -> paused; return the captured token count (blocks now free)."""
        if request_id not in self.pausing:
            raise ValueError(f"request {request_id!r} is not pausing")
        n = self.pausing.pop(request_id)
        self.paused[request_id] = n
        return n

    def resume(self, request_id: str) -> int:
        """paused -> resumed; return the captured token count to restore."""
        if request_id not in self.paused:
            raise ValueError(f"request {request_id!r} is not paused")
        return self.paused.pop(request_id)

    def is_pausing(self, request_id: str) -> bool:
        return request_id in self.pausing

    def is_paused(self, request_id: str) -> bool:
        return request_id in self.paused

    def defer_resume(self, request_id: str) -> None:
        """Record that a RESUME arrived while ``request_id`` is still pausing."""
        if request_id not in self.pausing:
            raise ValueError(f"request {request_id!r} is not pausing")
        self.resume_deferred.add(request_id)

    def pop_deferred_resume(self, request_id: str) -> bool:
        """True (and clears) if a RESUME was deferred AND the save is confirmed.

        Gated on ``is_paused`` rather than trusting the caller's ordering: if
        this were called while still ``pausing``, popping the flag now and
        returning True would make the caller's `resume_request` no-op (still
        pausing, not paused) and the deferred RESUME would be lost for good --
        the exact hang this mechanism exists to prevent. Left set until the
        save actually confirms, so a premature poll is a safe no-op, not a
        silent drop.
        """
        if request_id in self.resume_deferred and self.is_paused(request_id):
            self.resume_deferred.discard(request_id)
            return True
        return False

    def drop_finished(self, finished_ids: object) -> None:
        """Liveness: a request that finished between trigger and pause is gone.

        Same rule as the connector's ``_drop_finished`` -- silently forget any
        finished id in either state so a stale entry never drives a free/resume
        against reassigned blocks.
        """
        for req_id in finished_ids or ():
            self.pausing.pop(req_id, None)
            self.paused.pop(req_id, None)
            self.resume_deferred.discard(req_id)


def guard_pause_trigger(action: Callable[[], None]) -> Exception | None:
    """Run a pause-trigger action, catching any exception instead of propagating it.

    A PAUSE trigger can legitimately fail fast (e.g.
    ``assert_saved_covers_tokens`` rejecting a stale block count -- the exact
    GPU-confirmed bug this guard exists for). EngineCore has no top-level guard
    around ``schedule()``, so letting that exception escape kills the whole
    engine -- every co-tenant, not just the request being paused. This must
    fail the PAUSE, not the ENGINE. Returns the caught exception (for the
    caller to log) or ``None`` on success.
    """
    try:
        action()
    except Exception as exc:  # noqa: BLE001 - intentional: contain to this trigger
        return exc
    return None


@dataclass
class SavedKVRegistry:
    """Connector-side host registry of saved KV (memo item 3).

    Maps ``request_id -> (num_tokens_saved, block_count)``. ``register`` is
    called when the pause-save directive is emitted; ``matched_tokens`` answers
    ``get_num_new_matched_tokens`` delta-only; ``record_resume_blocks`` stashes
    the freshly allocated block ids the worker must load into on resume; ``drop``
    forgets the request once the resume-load has been emitted.
    """

    saved: dict[str, tuple[int, int]] = field(default_factory=dict)
    resume_blocks: dict[str, list[int]] = field(default_factory=dict)

    def register(
        self, request_id: str, num_tokens_saved: int, block_count: int
    ) -> None:
        if num_tokens_saved <= 0 or block_count <= 0:
            raise ValueError("saved token/block counts must be > 0")
        self.saved[request_id] = (num_tokens_saved, block_count)

    def is_saved(self, request_id: str) -> bool:
        return request_id in self.saved

    def matched_tokens(
        self, request_id: str, num_local_computed: int
    ) -> tuple[int, bool]:
        """Return ``(delta, load_async)`` for a resuming request, else ``(0, False)``.

        Only known (previously paused) request_ids match; everything else falls
        through to ``(0, False)`` exactly like the base connector. ``load_async``
        is False -- the spike uses the synchronous resume-load path.
        """
        if request_id not in self.saved:
            return 0, False
        saved_tokens, _ = self.saved[request_id]
        return matched_tokens_delta(saved_tokens, num_local_computed), False

    def record_resume_blocks(self, request_id: str, block_ids: list[int]) -> None:
        if request_id not in self.saved:
            raise ValueError(f"resume blocks for unknown saved request {request_id!r}")
        self.resume_blocks[request_id] = list(block_ids)

    def drop(self, request_id: str) -> None:
        self.saved.pop(request_id, None)
        self.resume_blocks.pop(request_id, None)


class SaveConfirmationReader:
    """Worker -> scheduler pause-save confirmation over the timing JSONL.

    The worker appends a ``{"phase": "pause_saved", "request_id": ...}`` row
    after a synchronous pause-save completes; the scheduler polls this reader
    each step for newly confirmed request ids. A line cursor makes ``poll``
    return only rows appended since the previous call. This is the simplest
    correct handshake given a synchronous save (memo: upgrade path is the async
    saved-set returned in KVConnectorOutput, consumed like finished_sending).

    ponytail: re-reads the whole file and slices past the cursor each poll --
    fine for a spike (steps ~10 ms, file is tiny); switch to the async saved-set
    if it ever shows up in scheduler-step time.
    """

    CONFIRM_PHASE = "pause_saved"

    def __init__(self, timing_path: str):
        self._path = timing_path
        self._cursor = 0

    def poll(self) -> set[str]:
        from pathlib import Path

        p = Path(self._path)
        if not p.exists():
            return set()
        lines = p.read_text().splitlines()
        new = lines[self._cursor :]
        self._cursor = len(lines)
        confirmed: set[str] = set()
        for line in new:
            if not line:
                continue
            row = json.loads(line)
            if row.get("phase") == self.CONFIRM_PHASE and row.get("request_id"):
                confirmed.add(row["request_id"])
        return confirmed


@dataclass
class PauseResult:
    """One pause/resume cycle in the ``--scenario pause`` run.

    ``identical`` is the GPU-only unknown the memo flags (risk #2): whether the
    greedy continuation after pause/resume is token-for-token identical to an
    uninterrupted run of the same prompt+seed -- i.e. loading the saved KV is
    bit-faithful, not merely close.
    """

    tool_duration_s: float
    blocks_freed: int
    pause_to_freed_ms: float
    resume_to_first_token_ms: float
    identical: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
