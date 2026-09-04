"""GPU / vLLM-dependent layer of the spike.

This is the ONLY module that imports ``torch`` or ``vllm``. Import it eagerly
only on the GPU box. On a CPU host the imports fail loudly (:data:`HAVE_VLLM`
is False and constructing the classes raises), while ``core.py`` /
``control.py`` -- which hold all the timing, control, and reporting logic --
import cleanly for unit tests.

Seam chosen (vLLM 0.11.0): a custom ``KVConnectorBase_V1`` subclass.
``build_connector_meta`` runs in the scheduler process every step; there we
read the out-of-band :class:`OffloadControl` and, for the target request whose
block ids we tracked in ``update_state_after_alloc``, emit metadata telling the
worker to copy those blocks GPU<->pinned-host. The worker executes the copy
inside ``start_load_kv`` / ``wait_for_save`` with CUDA-event timing. See
README for why this is the closest public seam and what it does NOT do
(it copies KV; it does not by itself evict blocks or pause the request).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .control import OffloadControl, OffloadPhase
from .core import (
    FinishedRetentionBook,
    continuum_priority,
    OffloadedBlocks,
    RetainedPrefix,
    RetentionMatch,
    SavedKVRegistry,
    TransferTiming,
    assert_saved_covers_tokens,
    chunk_ranges,
    resume_load_block_ids,
    thunderagent_reasoning_pauses,
    thunderagent_resume_admissions,
    validate_block_ids,
    validate_staging_capacity,
)

logger = logging.getLogger(__name__)

try:  # torch is pulled in by vllm; both only present on the GPU box.
    import torch

    HAVE_TORCH = True
except ImportError:  # pragma: no cover - exercised only off-GPU
    HAVE_TORCH = False

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )

    HAVE_VLLM = True
except ImportError:  # pragma: no cover - exercised only off-GPU
    HAVE_VLLM = False
    KVConnectorBase_V1 = object  # type: ignore[assignment,misc]
    KVConnectorMetadata = object  # type: ignore[assignment,misc]

if TYPE_CHECKING:  # pragma: no cover
    from vllm.config import VllmConfig


@dataclass
class SelectiveOffloadMeta(KVConnectorMetadata):  # type: ignore[misc,valid-type]
    """Per-step connector metadata: the offload/restore/pause/resume directives.

    Built in ``build_connector_meta`` (scheduler process), delivered to
    ``start_load_kv`` (worker process) via vLLM's own metadata-passing --
    verify on the box that a single step's metadata always makes it to the
    worker that executes it (README risk).

    Each directive is ``(phase, request_id, block_ids)``. ``phase`` is one of
    ``offload`` / ``restore`` (W1 copy scenario), ``pause`` / ``resume`` (P4
    evict scenario), or ``release`` (P4 cleanup: drop a retained buffer without
    loading it -- finish-while-paused or a free self-hit resume; ``block_ids``
    is unused/empty for ``release``). ``request_id`` keys the worker's
    per-request retained pinned buffer for pause/resume/release.
    """

    directives: list[tuple[str, str, list[int]]] = field(default_factory=list)
    retention_directives: list[dict[str, Any]] = field(default_factory=list)


# Internal directive phase for dropping a retained buffer without loading it
# (not an OffloadControl-facing phase -- never appears on the control file).
_RELEASE = "release"


def vllm_version() -> str | None:
    try:
        import vllm

        return vllm.__version__
    except ImportError:
        return None


class _CompletedTransfer:
    """A synchronous transfer whose timing is already known.

    Returned by the strided (baseline) path, which blocks to completion inside
    ``begin`` exactly like the original implementation, so ``wait`` is a no-op
    that hands back the timing measured then.
    """

    def __init__(self, timing: TransferTiming):
        self._timing = timing

    def wait(self) -> TransferTiming:
        return self._timing


class _StreamTransfer:
    """A transfer enqueued on a dedicated CUDA stream, timed by CUDA events.

    ``begin`` records ``start``/``end`` events around the copy and returns
    without synchronizing, so the worker step does not stall behind it. The
    host-side wait (needed to read ``elapsed_time``) is deferred to ``wait``,
    which the connector calls in ``wait_for_save`` -- off the forward-pass
    critical path.
    """

    def __init__(
        self,
        start: "torch.cuda.Event",
        end: "torch.cuda.Event",
        bytes_moved: int,
        num_blocks: int,
    ):
        self._start = start
        self._end = end
        self._bytes_moved = bytes_moved
        self._num_blocks = num_blocks

    def wait(self) -> TransferTiming:
        self._end.synchronize()
        ms = self._start.elapsed_time(self._end)
        return TransferTiming(
            bytes_moved=self._bytes_moved, milliseconds=ms, num_blocks=self._num_blocks
        )


class CudaBlockTransfer:
    """Copy paged KV blocks between the GPU cache and pinned host memory.

    ``kv_caches`` maps layer name -> the layer's paged KV tensor as handed to
    the connector by ``register_kv_caches``. Block ``b`` of a layer is
    ``tensor.movedim(block_dim, 0)[b]``.

    Two transfer modes (selectable for A/B measurement via ``mode``):

    - ``"strided"`` -- the original baseline: a Python loop of per-block copies
      bracketed by full ``torch.cuda.synchronize()``. Kept so the next GPU
      session can reproduce the W1 numbers (72 ms / 2.8 GB/s, 189 ms co-tenant
      ITL tail) against the improved path.
    - ``"staged"`` (default) -- pre-allocate one pinned host buffer and one
      contiguous device gather buffer per layer at construction (sized for
      ``max_blocks``). Each transfer gathers the target blocks into the
      contiguous device buffer with a single ``index_select`` per chunk, then
      does one large ``copy_`` per chunk to/from pinned host. All work runs on
      a dedicated CUDA stream with event timing, so it overlaps the forward
      pass instead of serializing the whole device. ``chunk_bytes`` (0 =
      disabled) caps the bytes per copy as a rate limiter.

    ``block_dim`` is honored, not decorative: every shape/index computation
    operates on ``tensor.movedim(block_dim, 0)`` (a view, so writes through it
    mutate the original storage), so a non-default layout is a single
    constructor arg away -- verify the real dim on the box (README).
    """

    def __init__(
        self,
        kv_caches: dict[str, "torch.Tensor"],
        block_dim: int = 0,
        *,
        mode: str = "staged",
        max_blocks: int = 128,
        chunk_bytes: int = 0,
    ):
        if not HAVE_TORCH:  # pragma: no cover - off-GPU guard
            raise RuntimeError("CudaBlockTransfer requires torch (GPU box only)")
        if not kv_caches:
            raise ValueError("kv_caches is empty; register_kv_caches not called?")
        if mode not in ("strided", "staged"):
            raise ValueError(f"mode must be 'strided' or 'staged', got {mode!r}")
        self.kv_caches = kv_caches
        self.block_dim = block_dim
        self.mode = mode
        self.max_blocks = max_blocks
        self.chunk_bytes = chunk_bytes
        # strided: restore reads back from here; staged: pre-allocated pinned
        # host staging (reused every transfer).
        self._host: dict[str, torch.Tensor] = {}
        self._dev_stage: dict[str, torch.Tensor] = {}
        # Per-request retained pinned buffers for pause/resume (NOT recycled
        # into the staging pool): request_id -> {layer_name: pinned host tensor}.
        self._retained: dict[str, dict[str, torch.Tensor]] = {}
        self._stream: torch.cuda.Stream | None = None
        if mode == "staged":
            self._stream = torch.cuda.Stream()
            self._alloc_staging()

    def _fronted(self, t: "torch.Tensor") -> "torch.Tensor":
        """View of ``t`` with the block dim moved to the front (dim 0)."""
        return t.movedim(self.block_dim, 0)

    def _bytes_per_block(self) -> int:
        total = 0
        for t in self.kv_caches.values():
            fronted = self._fronted(t)
            total += fronted[0].numel() * t.element_size()
        return total

    def _alloc_staging(self) -> None:
        for name, dev in self.kv_caches.items():
            dev_f = self._fronted(dev)
            rest = dev_f.shape[1:]
            self._host[name] = torch.empty(
                (self.max_blocks, *rest),
                dtype=dev_f.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._dev_stage[name] = torch.empty(
                (self.max_blocks, *rest), dtype=dev_f.dtype, device=dev.device
            )

    def begin(self, block_ids: list[int], *, to_host: bool):
        """Start a transfer; return a handle whose ``wait()`` yields timing.

        Staged mode returns without host-side synchronization (copy runs on the
        dedicated stream); strided mode blocks to completion here.
        """
        if not block_ids:
            raise ValueError("refusing to transfer zero blocks")
        if self.mode == "staged":
            return self._begin_staged(block_ids, to_host=to_host)
        return self._begin_strided(block_ids, to_host=to_host)

    def _begin_strided(
        self, block_ids: list[int], *, to_host: bool
    ) -> _CompletedTransfer:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for name, dev in self.kv_caches.items():
            dev_f = self._fronted(dev)
            validate_block_ids(dev_f.shape[0], block_ids)
            if to_host:
                host = torch.empty(
                    (len(block_ids), *dev_f.shape[1:]),
                    dtype=dev_f.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                for i, b in enumerate(block_ids):
                    host[i].copy_(dev_f[b], non_blocking=True)
                self._host[name] = host
            else:
                host = self._host[name]
                for i, b in enumerate(block_ids):
                    dev_f[b].copy_(host[i], non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        nbytes = len(block_ids) * self._bytes_per_block()
        return _CompletedTransfer(
            TransferTiming(
                bytes_moved=nbytes, milliseconds=ms, num_blocks=len(block_ids)
            )
        )

    def _begin_staged(self, block_ids: list[int], *, to_host: bool) -> _StreamTransfer:
        nb = len(block_ids)
        validate_staging_capacity(self.max_blocks, nb)
        bpb = self._bytes_per_block()
        default = torch.cuda.current_stream()
        assert self._stream is not None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self._stream):
            # See writes the forward pass already queued on the default stream
            # before we read/overwrite the same KV blocks.
            self._stream.wait_stream(default)
            start.record()
            for name, dev in self.kv_caches.items():
                dev_f = self._fronted(dev)
                validate_block_ids(dev_f.shape[0], block_ids)
                host = self._host[name]
                stage = self._dev_stage[name]
                for c0, c1 in chunk_ranges(nb, bpb, self.chunk_bytes):
                    idx = torch.tensor(
                        block_ids[c0:c1], device=dev.device, dtype=torch.long
                    )
                    if to_host:
                        torch.index_select(dev_f, 0, idx, out=stage[c0:c1])
                        host[c0:c1].copy_(stage[c0:c1], non_blocking=True)
                    else:
                        stage[c0:c1].copy_(host[c0:c1], non_blocking=True)
                        dev_f.index_copy_(0, idx, stage[c0:c1])
            end.record()
        if not to_host:
            # Restore writes KV the forward pass will read this step; make the
            # default stream wait for the copy (GPU-side ordering, no host stall).
            # NOTE: this ordering is correct ONLY under enforce_eager=True (the
            # forward runs on current_stream()); a CUDA-graph replay stream
            # would NOT be gated by this wait — revisit before dropping eager.
            default.wait_event(end)
        return _StreamTransfer(start, end, nb * bpb, nb)

    def offload(self, block_ids: list[int]) -> TransferTiming:
        return self.begin(block_ids, to_host=True).wait()

    def restore(self, block_ids: list[int]) -> TransferTiming:
        return self.begin(block_ids, to_host=False).wait()

    # --- pause/resume: dedicated retained buffers ------------------------
    # A pause-save must SURVIVE eviction until the matching resume-load, so it
    # cannot reuse the recycled staging buffer (a co-tenant's next offload would
    # overwrite it). Each paused request gets its own pinned host buffer, freed
    # only after its resume-load. The copy is SYNCHRONOUS (blocking) regardless
    # of ``mode`` -- a pause-save is a rare, one-off event allowed to cost the
    # full ~10-72 ms once (memo), so correctness over overlap here.

    def save_retained(self, request_id: str, block_ids: list[int]) -> TransferTiming:
        if request_id in self._retained:
            raise ValueError(f"request {request_id!r} already has a retained save")
        if not block_ids:
            raise ValueError("refusing to save zero blocks")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        buf: dict[str, "torch.Tensor"] = {}
        for name, dev in self.kv_caches.items():
            dev_f = self._fronted(dev)
            validate_block_ids(dev_f.shape[0], block_ids)
            host = torch.empty(
                (len(block_ids), *dev_f.shape[1:]),
                dtype=dev_f.dtype,
                device="cpu",
                pin_memory=True,
            )
            idx = torch.tensor(block_ids, device=dev.device, dtype=torch.long)
            host.copy_(torch.index_select(dev_f, 0, idx), non_blocking=True)
            buf[name] = host
        end.record()
        torch.cuda.synchronize()
        self._retained[request_id] = buf
        ms = start.elapsed_time(end)
        return TransferTiming(
            bytes_moved=len(block_ids) * self._bytes_per_block(),
            milliseconds=ms,
            num_blocks=len(block_ids),
        )

    def load_retained(
        self, request_id: str, block_ids: list[int], *, source_start: int = 0
    ) -> TransferTiming:
        if request_id not in self._retained:
            raise ValueError(f"no retained save for request {request_id!r}")
        if source_start < 0:
            raise ValueError("source_start must be >= 0")
        buf = self._retained[request_id]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for name, dev in self.kv_caches.items():
            dev_f = self._fronted(dev)
            validate_block_ids(dev_f.shape[0], block_ids)
            host = buf[name]
            if source_start + len(block_ids) > host.shape[0]:
                raise ValueError(
                    f"load slice [{source_start}, {source_start + len(block_ids)}) "
                    f"exceeds {host.shape[0]} saved blocks for {request_id!r}"
                )
            idx = torch.tensor(block_ids, device=dev.device, dtype=torch.long)
            dev_f.index_copy_(
                0,
                idx,
                host[source_start : source_start + len(block_ids)].to(
                    dev.device, non_blocking=True
                ),
            )
        end.record()
        torch.cuda.synchronize()
        del self._retained[request_id]  # free the pinned buffer post-resume
        ms = start.elapsed_time(end)
        return TransferTiming(
            bytes_moved=len(block_ids) * self._bytes_per_block(),
            milliseconds=ms,
            num_blocks=len(block_ids),
        )

    def free_retained(self, request_id: str) -> None:
        """Drop a retained pause-save without loading it.

        Used when the paused request finishes/aborts, or resumes for free via
        the local prefix cache (no load needed) -- either way the pinned
        buffer must not leak. No-op if nothing is retained for the id.
        """
        self._retained.pop(request_id, None)


class SelectiveOffloadConnector(KVConnectorBase_V1):  # type: ignore[misc,valid-type]
    """Externally-triggered selective KV offload for one target request.

    vLLM loads this by import path via ``KVTransferConfig`` /
    ``KVConnectorFactory.register_connector`` (see README) -- it must live at
    module level, not behind a factory. Off-GPU the base is ``object`` and the
    class is never instantiated, so importing this module stays torch/vllm-free
    for the CPU test suite (which never touches this class).

    Config is threaded through ``kv_connector_extra_config``:
    ``control_path`` (the :class:`OffloadControl` sentinel), ``timing_path``
    (a JSONL the worker appends timed transfers to, so the driver reads
    latencies back without shared memory), and ``block_dim`` (optional, default
    0 -- forwarded to :class:`CudaBlockTransfer`).
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: object | None = None,
    ):
        # vllm 0.11.2 added a third positional arg (kv_cache_config) to
        # KVConnectorBase_V1.__init__; 0.11.0 has only (vllm_config, role).
        # Accept both so the pin can float within the tested patch series.
        if not HAVE_VLLM:  # pragma: no cover - off-GPU guard
            raise RuntimeError("SelectiveOffloadConnector requires vllm (GPU box only)")
        try:
            super().__init__(vllm_config, role, kv_cache_config)
        except TypeError:  # 0.11.0 two-arg base
            super().__init__(vllm_config, role)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        retention_path = extra.get("retention_path")
        self._retention_path = Path(retention_path) if retention_path else None
        self._retention_control_version: tuple[int, int, int] | None = None
        self._retention_control_payload: dict[str, Any] = {}
        self._release_programs_seen: set[str] = set()
        self._control = OffloadControl(extra["control_path"])
        self._timing_path = extra["timing_path"]
        self._block_dim = int(extra.get("block_dim", 0))
        self._mode = str(extra.get("transfer_mode", "staged"))
        self._max_blocks = int(extra.get("max_blocks", 128))
        self._chunk_bytes = int(extra.get("chunk_bytes", 0))
        self._block_size = int(vllm_config.cache_config.block_size)
        self._thunder_buffer_tokens = int(extra.get("thunderagent_buffer_tokens", 100))
        self._role = role
        self._last_epoch = -1
        self._block_ids: dict[str, list[int]] = {}
        self._transfer: CudaBlockTransfer | None = None
        # Transfers kicked in start_load_kv, drained (timed + recorded) in
        # wait_for_save / get_finished so start_load_kv never blocks the worker.
        self._pending: list[tuple[str, Any]] = []
        # P4 pause/resume (scheduler side): saved-KV registry + a queue of
        # directives the scheduler asked for (pause-save / resume-load) that
        # build_connector_meta drains into the next step's metadata.
        self._saved = SavedKVRegistry()
        self._queued: list[tuple[str, str, list[int]]] = []
        # W5 finished-turn retention state, plus worker/scheduler directive
        # handoff. The two connector roles live in separate processes.
        self._retention = FinishedRetentionBook()
        self._retention_matches: dict[str, RetentionMatch] = {}
        self._retention_queued: list[dict[str, Any]] = []
        self._retention_claims: dict[str, str] = {}
        self._retained_block_counts: dict[str, int] = {}
        self._prefix_invalidations: list[tuple[int, ...]] = []
        self._retention_finished_seen: set[str] = set()
        self._retention_release_ready: set[str] = set()
        self._thunder_paused_programs: set[str] = set()
        self._thunder_admitted_programs: set[str] = set()
        self._thunder_marked_programs: set[str] = set()
        self._thunder_resuming_programs: dict[str, str] = {}
        self._continuum_priority_state: dict[str, tuple[int, bool, bool]] = {}
        # W1 copy scenario: exact block ids staged host-side per target.
        self._offloaded = OffloadedBlocks()

    # --- worker side -----------------------------------------------------
    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        self._transfer = CudaBlockTransfer(
            kv_caches,
            block_dim=self._block_dim,
            mode=self._mode,
            max_blocks=self._max_blocks,
            chunk_bytes=self._chunk_bytes,
        )

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Execute scheduler directives in the worker process."""
        meta = self._get_connector_metadata()
        for directive in meta.retention_directives:  # type: ignore[attr-defined]
            assert self._transfer is not None
            kind = directive["kind"]
            program_id = directive["program_id"]
            retained_id = f"retention:{program_id}"
            if kind == "save":
                started = time.monotonic()
                timing = self._transfer.save_retained(
                    retained_id, directive["block_ids"]
                )
                self._record(
                    "retention_offload",
                    timing,
                    request_id=directive["old_request_id"],
                    program_id=program_id,
                    started_monotonic_s=started,
                    completed_monotonic_s=time.monotonic(),
                )
                self._retention_release_ready.add(directive["old_request_id"])
            elif kind == "load":
                started = time.monotonic()
                timing = self._transfer.load_retained(
                    retained_id,
                    directive["block_ids"],
                    source_start=directive["source_start"],
                )
                self._record(
                    "retention_restore",
                    timing,
                    request_id=directive["request_id"],
                    program_id=program_id,
                    started_monotonic_s=started,
                    completed_monotonic_s=time.monotonic(),
                )
            elif kind == "drop_host":
                self._transfer.free_retained(retained_id)
            elif kind == "release_finished":
                self._retention_release_ready.add(directive["old_request_id"])
            else:
                raise ValueError(f"unknown retention directive {kind!r}")

        for phase, request_id, block_ids in meta.directives:  # type: ignore[attr-defined]
            assert self._transfer is not None
            if phase == OffloadPhase.PAUSE.value:
                timing = self._transfer.save_retained(request_id, block_ids)
                self._record(phase, timing, request_id=request_id)
                self._record("pause_saved", timing, request_id=request_id)
            elif phase == OffloadPhase.RESUME.value:
                timing = self._transfer.load_retained(request_id, block_ids)
                self._record(phase, timing, request_id=request_id)
            elif phase == _RELEASE:
                self._transfer.free_retained(request_id)
            else:
                to_host = phase == OffloadPhase.OFFLOAD.value
                self._pending.append(
                    (phase, self._transfer.begin(block_ids, to_host=to_host))
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Restore correctness is enforced GPU-side in the staged path (the
        # forward stream waits on the copy's completion event), so no per-layer
        # host wait is needed here.
        return None

    def save_kv_layer(self, *args: Any, **kwargs: Any) -> None:
        return None

    def wait_for_save(self) -> None:
        # End-of-step drain: block on each kicked transfer's completion event
        # (off the forward-pass critical path), then record its timing.
        self._drain_pending()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        # Called on empty-batch steps too, so asynchronous W1 copies still drain.
        self._drain_pending()
        self._retention_finished_seen.update(finished_req_ids)
        released = self._retention_finished_seen & self._retention_release_ready
        self._retention_finished_seen.difference_update(released)
        self._retention_release_ready.difference_update(released)
        return released or None, None

    def _drain_pending(self) -> None:
        for phase, pending in self._pending:
            self._record(phase, pending.wait())
        self._pending.clear()

    def _record(
        self,
        phase: str,
        timing: TransferTiming,
        request_id: str | None = None,
        **fields: Any,
    ) -> None:
        import json

        with open(self._timing_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "phase": phase,
                        "request_id": request_id,
                        "milliseconds": timing.milliseconds,
                        "bytes_moved": timing.bytes_moved,
                        "num_blocks": timing.num_blocks,
                        "effective_gbps": timing.effective_gbps,
                        **fields,
                    }
                )
                + "\n"
            )

    def _record_control_event(self, phase: str, **fields: Any) -> None:
        with open(self._timing_path, "a") as f:
            f.write(
                json.dumps({"phase": phase, "monotonic_s": time.monotonic(), **fields})
                + "\n"
            )

    # --- scheduler side --------------------------------------------------
    def _retention_payload(self) -> dict[str, Any]:
        if self._retention_path is None:
            return {}
        stat = self._retention_path.stat()
        version = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if version != self._retention_control_version:
            payload = json.loads(self._retention_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("retention control must be a JSON object")
            self._retention_control_payload = payload
            self._retention_control_version = version
        return self._retention_control_payload

    def _retention_spec(self, request_id: str) -> dict[str, Any] | None:
        spec = self._retention_payload().get(request_id)
        if spec is None:
            return None
        if not isinstance(spec, dict):
            raise ValueError(f"retention spec for {request_id!r} must be an object")
        return spec

    def refresh_retention_expiries(self) -> None:
        """Apply causally revealed timer updates to still-resident prefixes."""
        for record in tuple(self._retention.resident.values()):
            spec = self._retention_spec(record.request_id)
            if spec is None or spec.get("expire_at_monotonic_s") is None:
                continue
            self._retention.arm_expiry_earlier(
                record.program_id,
                record.request_id,
                spec["expire_at_monotonic_s"],
            )

    def matching_resident_program_ids(self, requests: list[Any]) -> set[str]:
        program_ids: set[str] = set()
        for request in requests:
            spec = self._retention_spec(request.request_id)
            program_id = None if spec is None else spec.get("program_id")
            record = (
                None
                if not isinstance(program_id, str)
                else self._retention.resident.get(program_id)
            )
            if record is not None and (
                tuple(request.block_hashes[: record.block_count]) == record.block_hashes
            ):
                program_ids.add(program_id)
        return program_ids

    def thunder_resume_capacity_tokens(
        self, running: list[Any], *, free_tokens: int
    ) -> int:
        if free_tokens < 0:
            raise ValueError("free_tokens must be >= 0")
        active_program_ids = {
            record.program_id
            for record in self._retention.resident.values()
            if record.policy == "thunderagent" and record.action == "pressure"
        }
        for request in running:
            spec = self._retention_spec(request.request_id)
            program_id = None if spec is None else spec.get("program_id")
            if (
                spec is not None
                and spec.get("policy") == "thunderagent"
                and isinstance(program_id, str)
                and program_id not in self._thunder_paused_programs
            ):
                active_program_ids.add(program_id)
        return max(
            0,
            free_tokens - len(active_program_ids) * self._thunder_buffer_tokens,
        )

    def order_thunder_waiting(
        self, requests: list[Any], *, capacity_tokens: int
    ) -> list[Any]:
        paused: list[tuple[str, int]] = []
        request_by_program: dict[str, Any] = {}
        for request in requests:
            spec = self._retention_spec(request.request_id)
            program_id = None if spec is None else spec.get("program_id")
            if (
                isinstance(program_id, str)
                and program_id in self._thunder_paused_programs
            ):
                tokens = (
                    (int(request.num_tokens) + self._block_size - 1)
                    // self._block_size
                    * self._block_size
                )
                paused.append((program_id, tokens))
                request_by_program[program_id] = request
        if not paused:
            self._thunder_admitted_programs.clear()
            return requests
        admitted = thunderagent_resume_admissions(
            paused,
            capacity_tokens=capacity_tokens,
            buffer_per_program=self._thunder_buffer_tokens,
        )
        self._thunder_admitted_programs = set(admitted)
        admitted_set = set(admitted)
        paused_request_ids = {
            request.request_id for request in request_by_program.values()
        }
        return (
            [request_by_program[program_id] for program_id in admitted]
            + [
                request
                for request in requests
                if request.request_id not in paused_request_ids
            ]
            + [
                request_by_program[program_id]
                for program_id, _ in sorted(paused, key=lambda item: (item[1], item[0]))
                if program_id not in admitted_set
            ]
        )

    def refresh_continuum_priorities(self, requests: list[Any]) -> bool:
        changed = False
        for request in requests:
            spec = self._retention_spec(request.request_id)
            if spec is None or spec.get("policy") != "continuum":
                continue
            program_id = spec.get("program_id")
            program_index = spec.get("program_index")
            stride = spec.get("priority_stride")
            if (
                not isinstance(program_id, str)
                or not isinstance(program_index, int)
                or isinstance(program_index, bool)
                or not isinstance(stride, int)
                or isinstance(stride, bool)
            ):
                raise ValueError(
                    f"request {request.request_id!r} lacks Continuum priority metadata"
                )
            record = self._retention.resident.get(program_id)
            ttl_hit = record is not None and (
                tuple(request.block_hashes[: record.block_count]) == record.block_hashes
            )
            preempted = getattr(request, "num_preemptions", 0) > 0
            priority = continuum_priority(
                program_index,
                stride,
                ttl_hit=ttl_hit,
                preempted=preempted,
            )
            state = (priority, ttl_hit, preempted)
            if self._continuum_priority_state.get(request.request_id) != state:
                self._continuum_priority_state[request.request_id] = state
                self._record_control_event(
                    "continuum_priority",
                    request_id=request.request_id,
                    priority=priority,
                    ttl_hit=ttl_hit,
                    preempted=preempted,
                )
            if request.priority != priority:
                request.priority = priority
                changed = True
        return changed

    def register_pause_save(
        self, request_id: str, num_computed_tokens: int, block_ids: list[int]
    ) -> int:
        """Scheduler poke (in-process): queue a pause-save for a running request.

        Called by :class:`PausableScheduler` at the pause instant, BEFORE it
        frees anything. ``block_ids`` MUST be the request's CURRENT block ids
        (e.g. ``kv_cache_manager.get_block_ids(request_id)[0]``) fetched by the
        caller at the pause instant -- NOT ``self._block_ids``, which this
        connector only refreshes in ``update_state_after_alloc`` (admission /
        resume). v0.11.2 does not call that hook as a running request's decode
        grows past its originally allocated blocks, so ``self._block_ids`` goes
        stale mid-generation (GPU-confirmed: 97 admission blocks vs 99 actual
        at 1581 computed tokens, block_size 16). Records the saved-KV registry
        entry (so a later resume's ``get_num_new_matched_tokens`` can report
        the count) and queues the SAVE directive the worker executes. Fail
        fast unless ``block_ids`` exactly covers ``num_computed_tokens``.
        Returns the block count (== blocks that will be freed).
        """
        if not block_ids:
            raise ValueError(f"cannot pause-save {request_id!r} with no block ids")
        assert_saved_covers_tokens(
            num_computed_tokens, len(block_ids), self._block_size
        )
        self._saved.register(request_id, num_computed_tokens, len(block_ids))
        self._queued.append((OffloadPhase.PAUSE.value, request_id, list(block_ids)))
        return len(block_ids)

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        paused_tokens, paused_async = self._saved.matched_tokens(
            request.request_id, num_computed_tokens
        )
        if paused_tokens:
            return paused_tokens, paused_async
        spec = self._retention_spec(request.request_id)
        if spec is None:
            return 0, False
        program_id = spec.get("program_id")
        if not isinstance(program_id, str) or not program_id:
            raise ValueError(f"request {request.request_id!r} lacks program_id")
        if (
            program_id in self._thunder_paused_programs
            and program_id not in self._thunder_admitted_programs
        ):
            return None, False
        if program_id in self._thunder_paused_programs:
            self._thunder_resuming_programs[request.request_id] = program_id
        owner = self._retention_claims.get(program_id)
        if owner is not None and owner != request.request_id:
            return 0, False
        match = self._retention.match(
            program_id,
            request.block_hashes,
            num_local_tokens=num_computed_tokens,
            block_size=self._block_size,
        )
        if match.record is not None:
            self._retention_claims[program_id] = request.request_id
        self._retention_matches[request.request_id] = match
        return match.external_blocks * self._block_size, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        new_block_ids = list(blocks.get_block_ids()[0])
        self._block_ids[request.request_id] = new_block_ids
        resumed_program = self._thunder_resuming_programs.pop(request.request_id, None)
        if resumed_program is not None:
            self._thunder_paused_programs.remove(resumed_program)
            self._thunder_admitted_programs.discard(resumed_program)
            self._record_control_event(
                "thunderagent_reasoning_resumed",
                program_id=resumed_program,
                request_id=request.request_id,
            )
        if self._saved.is_saved(request.request_id):
            if num_external_tokens > 0:
                saved_block_count = self._saved.saved[request.request_id][1]
                resume_block_ids = resume_load_block_ids(
                    new_block_ids, saved_block_count
                )
                self._saved.record_resume_blocks(request.request_id, resume_block_ids)
                self._queued.append(
                    (OffloadPhase.RESUME.value, request.request_id, resume_block_ids)
                )
            else:
                self._queued.append((_RELEASE, request.request_id, []))
            self._saved.drop(request.request_id)
            return

        match = self._retention_matches.pop(request.request_id, None)
        if match is None or match.record is None:
            return
        self._retention_claims.pop(match.record.program_id, None)
        record = match.record
        if match.source == "mismatch":
            source = (
                "resident" if record.program_id in self._retention.resident else "host"
            )
            if source == "resident":
                self.invalidate_retained_prefix(record)
            self._retention.drop(record.program_id)
            if source == "host":
                self._retention_queued.append(
                    {"kind": "drop_host", "program_id": record.program_id}
                )
            else:
                self._retention_queued.append(
                    {
                        "kind": "release_finished",
                        "program_id": record.program_id,
                        "old_request_id": record.request_id,
                    }
                )
            return

        expected_external = match.external_blocks * self._block_size
        if num_external_tokens != expected_external:
            raise ValueError(
                f"connector allocated {num_external_tokens} external tokens; "
                f"expected {expected_external} for {request.request_id!r}"
            )
        if len(new_block_ids) < record.block_count:
            raise ValueError(
                f"request {request.request_id!r} has {len(new_block_ids)} blocks, "
                f"fewer than retained prefix {record.block_count}"
            )
        load_ids = new_block_ids[match.local_blocks : record.block_count]
        if match.source == "resident":
            self._retention.drop(record.program_id)
            if load_ids:
                self._retention_queued.append(
                    {
                        "kind": "save",
                        "program_id": record.program_id,
                        "old_request_id": record.request_id,
                        "block_ids": list(record.block_ids),
                    }
                )
            else:
                self._retention_queued.append(
                    {
                        "kind": "release_finished",
                        "program_id": record.program_id,
                        "old_request_id": record.request_id,
                    }
                )
        else:
            self._retention.drop(record.program_id)
        if load_ids:
            self._retention_queued.append(
                {
                    "kind": "load",
                    "program_id": record.program_id,
                    "request_id": request.request_id,
                    "source_start": match.local_blocks,
                    "block_ids": load_ids,
                }
            )
        elif match.source == "host":
            self._retention_queued.append(
                {"kind": "drop_host", "program_id": record.program_id}
            )

    def take_retained_block_count(self, request_id: str) -> int | None:
        return self._retained_block_counts.pop(request_id, None)

    def invalidate_retained_prefix(self, record: RetainedPrefix) -> None:
        """Queue the old GPU blocks for prefix-cache invalidation."""
        self._prefix_invalidations.append(record.block_ids)

    def take_prefix_invalidations(self) -> list[tuple[int, ...]]:
        invalidations = self._prefix_invalidations
        self._prefix_invalidations = []
        return invalidations

    def request_finished(
        self, request: Any, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        spec = self._retention_spec(request.request_id)
        if spec is None or spec.get("final") or spec.get("action") is None:
            return False, None
        program_id = spec.get("program_id")
        action = spec.get("action")
        policy = spec.get("policy")
        expire_ms = spec.get("expire_ms")
        program_arrival_s = spec.get("program_arrival_s", 0.0)
        if (
            not isinstance(program_id, str)
            or not isinstance(action, str)
            or not isinstance(policy, str)
            or (
                expire_ms is not None
                and (
                    not isinstance(expire_ms, (int, float))
                    or isinstance(expire_ms, bool)
                    or expire_ms < 0
                )
            )
            or not isinstance(program_arrival_s, (int, float))
            or isinstance(program_arrival_s, bool)
            or program_arrival_s < 0
        ):
            raise ValueError(f"malformed retention spec for {request.request_id!r}")
        thunder_pause = (
            policy == "thunderagent" and program_id in self._thunder_marked_programs
        )
        if thunder_pause:
            self._thunder_marked_programs.remove(program_id)
            action = "release"
            expire_ms = 0.0
        hashes = request.block_hashes
        full_blocks = min(
            len(block_ids),
            len(hashes),
            max(0, (request.num_prompt_tokens - 1) // self._block_size),
        )
        if full_blocks == 0:
            return False, None
        now = time.monotonic()
        self._retention.register(
            RetainedPrefix(
                program_id=program_id,
                request_id=request.request_id,
                num_tokens=full_blocks * self._block_size,
                resident_tokens=full_blocks * self._block_size,
                block_ids=tuple(block_ids[:full_blocks]),
                block_hashes=tuple(hashes[:full_blocks]),
                created_at=now,
                expires_at=(
                    None if expire_ms is None else now + float(expire_ms) / 1000.0
                ),
                action=action,
                policy=policy,
                program_arrival_s=float(program_arrival_s),
            )
        )
        if thunder_pause:
            self._thunder_paused_programs.add(program_id)
            self._thunder_admitted_programs.discard(program_id)
            self._record_control_event(
                "thunderagent_reasoning_paused",
                program_id=program_id,
                request_id=request.request_id,
            )
        self._retained_block_counts[request.request_id] = full_blocks
        return True, None

    def has_due_retention(self) -> bool:
        return bool(self._pending_release_programs()) or bool(
            self._retention.due(time.monotonic())
        )

    def _pending_release_programs(self) -> list[str]:
        value = self._retention_payload().get("__release_programs__", [])
        if not isinstance(value, list) or not all(
            isinstance(program_id, str) and program_id for program_id in value
        ):
            raise ValueError("__release_programs__ must be a list of non-empty strings")
        return [
            program_id
            for program_id in value
            if program_id not in self._release_programs_seen
        ]

    def release_requested_programs(self) -> None:
        for program_id in self._pending_release_programs():
            source = "resident" if program_id in self._retention.resident else "host"
            record = self._retention.drop(program_id)
            self._release_programs_seen.add(program_id)
            self._thunder_marked_programs.discard(program_id)
            self._thunder_paused_programs.discard(program_id)
            self._thunder_admitted_programs.discard(program_id)
            if record is None:
                continue
            if source == "resident":
                self.invalidate_retained_prefix(record)
                self._retention_queued.append(
                    {
                        "kind": "release_finished",
                        "program_id": record.program_id,
                        "old_request_id": record.request_id,
                    }
                )
            else:
                self._retention_queued.append(
                    {"kind": "drop_host", "program_id": record.program_id}
                )

    def release_retention_due(self, waiting_program_ids: set[str]) -> None:
        for record in self._retention.due(
            time.monotonic(), waiting_program_ids=waiting_program_ids
        ):
            self.invalidate_retained_prefix(record)
            if record.action == "offload":
                self._retention.move_to_host(record.program_id)
                self._retention_queued.append(
                    {
                        "kind": "save",
                        "program_id": record.program_id,
                        "old_request_id": record.request_id,
                        "block_ids": list(record.block_ids),
                    }
                )
            else:
                self._retention.drop(record.program_id)
                self._retention_queued.append(
                    {
                        "kind": "release_finished",
                        "program_id": record.program_id,
                        "old_request_id": record.request_id,
                    }
                )

    def release_retention_for_pressure(
        self,
        requests: list[Any],
        *,
        capacity_tokens: int,
        active_block_ids: dict[str, tuple[int, ...]],
        waiting_request: Any | None,
    ) -> None:
        held_ids = {
            record.request_id
            for record in (
                *self._retention.resident.values(),
                *self._retention.host.values(),
            )
        }
        active = [
            request
            for request in requests
            if request.request_id not in held_ids
            and not request.is_finished()
            and active_block_ids.get(request.request_id)
        ]
        program_ids: set[str] = set()
        for request in active:
            spec = self._retention_spec(request.request_id)
            if spec is not None and isinstance(spec.get("program_id"), str):
                program_ids.add(spec["program_id"])
        waiting_program_id: str | None = None
        waiting_tokens = 0
        waiting_policy: str | None = None
        waiting_counts_for_pressure = waiting_request is not None
        if waiting_request is not None:
            waiting_spec = self._retention_spec(waiting_request.request_id)
            waiting_policy = (
                waiting_spec.get("policy") if waiting_spec is not None else None
            )
            candidate_program = (
                None if waiting_spec is None else waiting_spec.get("program_id")
            )
            if (
                waiting_policy == "thunderagent"
                and isinstance(candidate_program, str)
                and candidate_program in self._thunder_paused_programs
                and candidate_program not in self._thunder_admitted_programs
            ):
                waiting_counts_for_pressure = False
            retained = (
                None
                if not isinstance(candidate_program, str)
                else self._retention.resident.get(candidate_program)
            )
            reusable_tokens = 0
            if retained is not None and (
                tuple(waiting_request.block_hashes[: retained.block_count])
                == retained.block_hashes
            ):
                waiting_program_id = candidate_program
                reusable_tokens = retained.num_tokens
            missing_tokens = max(1, int(waiting_request.num_tokens) - reusable_tokens)
            waiting_tokens = (
                (missing_tokens + self._block_size - 1) // self._block_size
            ) * self._block_size
        block_references = [
            active_block_ids.get(request.request_id, ())
            for request in active
            if active_block_ids.get(request.request_id)
        ]
        block_references.extend(
            record.block_ids
            for record in self._retention.resident.values()
            if record.program_id not in program_ids
        )
        shared_blocks = sum(map(len, block_references)) - len(
            {block_id for blocks in block_references for block_id in blocks}
        )
        thunder_protected_program_ids = set(program_ids)
        if waiting_program_id is not None:
            thunder_protected_program_ids.add(waiting_program_id)
        capacity_args = {
            "reasoning_tokens": sum(
                len(active_block_ids[request.request_id]) * self._block_size
                for request in active
            ),
            "reasoning_programs": len(program_ids),
            "reasoning_program_ids": thunder_protected_program_ids,
            "shared_tokens": shared_blocks * self._block_size,
            "capacity_tokens": capacity_tokens,
            "buffer_per_program": self._thunder_buffer_tokens,
            "waiting_tokens": (
                waiting_tokens
                if waiting_policy == "thunderagent" and waiting_counts_for_pressure
                else 0
            ),
            "waiting_programs": int(
                waiting_policy == "thunderagent" and waiting_counts_for_pressure
            ),
        }
        evictions = self._retention.pressure_evictions(**capacity_args)
        evictions.extend(
            self._retention.continuum_pressure_evictions(
                active_tokens=capacity_args["reasoning_tokens"],
                waiting_tokens=waiting_tokens,
                capacity_tokens=capacity_tokens,
                shared_tokens=capacity_args["shared_tokens"],
                protected_program_ids=thunder_protected_program_ids,
            )
        )
        acting_records = [
            record
            for record in self._retention.resident.values()
            if record.action == "pressure" and record.program_id not in program_ids
        ]
        remaining_required = max(
            0,
            capacity_args["reasoning_tokens"]
            + capacity_args["waiting_tokens"]
            + sum(record.resident_tokens for record in acting_records)
            - capacity_args["shared_tokens"]
            + (
                capacity_args["reasoning_programs"]
                + len(acting_records)
                + capacity_args["waiting_programs"]
            )
            * self._thunder_buffer_tokens,
        )
        remaining_required -= sum(
            record.resident_tokens + self._thunder_buffer_tokens
            for record in evictions
            if record.action == "pressure"
        )
        if remaining_required > capacity_tokens:
            candidate_requests: dict[str, tuple[int, str]] = {}
            for request in active:
                spec = self._retention_spec(request.request_id)
                candidate_program = None if spec is None else spec.get("program_id")
                if (
                    spec is not None
                    and spec.get("policy") == "thunderagent"
                    and isinstance(candidate_program, str)
                    and candidate_program not in self._thunder_marked_programs
                ):
                    candidate_requests[candidate_program] = (
                        int(request.num_tokens),
                        request.request_id,
                    )
            marked = thunderagent_reasoning_pauses(
                [
                    (program_id, tokens)
                    for program_id, (tokens, _) in candidate_requests.items()
                ],
                required_tokens=remaining_required,
                capacity_tokens=capacity_tokens,
                buffer_per_program=self._thunder_buffer_tokens,
            )
            for candidate_program in marked:
                num_tokens, request_id = candidate_requests[candidate_program]
                self._thunder_marked_programs.add(candidate_program)
                self._record_control_event(
                    "thunderagent_reasoning_marked",
                    program_id=candidate_program,
                    request_id=request_id,
                    tokens=num_tokens,
                )
        for record in evictions:
            self._retention.drop(record.program_id)
            self.invalidate_retained_prefix(record)
            if record.policy == "continuum":
                self._record_control_event(
                    "continuum_pressure_unpinned",
                    program_id=record.program_id,
                    request_id=record.request_id,
                )
            self._retention_queued.append(
                {
                    "kind": "release_finished",
                    "program_id": record.program_id,
                    "old_request_id": record.request_id,
                }
            )

    def _drop_finished(self, scheduler_output: Any) -> None:
        # A request can finish (or be preempted out) between the driver's
        # offload trigger and the step where it would fire; its block ids are
        # reassigned to other requests once freed, so a directive built against
        # a stale entry would offload/restore the WRONG request's KV bytes.
        # SchedulerOutput carries the per-step finished-request set -- prune
        # our tracking table before reading it below (README risk #2).
        finished = getattr(scheduler_output, "finished_req_ids", None)
        if finished:
            for req_id in finished:
                self._block_ids.pop(req_id, None)
                self._continuum_priority_state.pop(req_id, None)
                claimed_programs = [
                    program_id
                    for program_id, owner in self._retention_claims.items()
                    if owner == req_id
                ]
                for program_id in claimed_programs:
                    self._retention_claims.pop(program_id, None)
                if self._offloaded.pop(req_id) is not None:
                    # The request finished between OFFLOAD and RESTORE; its
                    # block ids may already belong to another request.
                    self._write_restore_skipped(
                        req_id, "target request finished before restore fired"
                    )
                if self._saved.is_saved(req_id):
                    self._queued.append((_RELEASE, req_id, []))
                    self._saved.drop(req_id)

    def build_connector_meta(self, scheduler_output: Any) -> "KVConnectorMetadata":
        self._drop_finished(scheduler_output)
        meta = SelectiveOffloadMeta()
        # 1. Drain any pause-save / resume-load directives the scheduler queued
        #    in-process this step (P4 evict scenario).
        meta.directives.extend(self._queued)
        self._queued.clear()
        meta.retention_directives.extend(self._retention_queued)
        self._retention_queued.clear()
        # 2. W1 copy scenario: read the control file directly for OFFLOAD /
        #    RESTORE. PAUSE / RESUME are NOT handled here -- PausableScheduler
        #    owns those and pokes us via register_pause_save / the alloc path,
        #    so reacting to them here too would double-emit.
        state = self._control.read()
        if state.epoch != self._last_epoch and state.phase in (
            OffloadPhase.OFFLOAD,
            OffloadPhase.RESTORE,
        ):
            self._last_epoch = state.epoch
            target = state.target_request_id or ""
            if state.phase is OffloadPhase.RESTORE:
                # GPU-validation bugfix: RESTORE must reuse EXACTLY the block
                # ids OFFLOAD staged host-side, not self._block_ids (which is
                # keyed by allocation events, not decode -- see
                # update_state_after_alloc -- and reads stale/absent once the
                # target has already finished). A miss here means "never
                # offloaded" or "target finished before restore fired", not
                # "seam broken" -- refuse and record why (see _drop_finished).
                block_ids = self._offloaded.pop(target)
                if not block_ids:
                    self._write_restore_skipped(
                        target, "no matching offload record for this target"
                    )
                    logger.warning(
                        "restore directive for request %r skipped: no matching "
                        "offload record (never offloaded, already restored, or "
                        "finished)",
                        target,
                    )
                    return meta
            else:
                block_ids = self._block_ids.get(target, [])
                if not block_ids:
                    logger.warning(
                        "offload directive for request %r skipped: "
                        "not tracked or already finished",
                        target,
                    )
                    return meta
                self._offloaded.record(target, block_ids)
            meta.directives.append((state.phase.value, target, block_ids))
        return meta

    def _write_restore_skipped(self, request_id: str, reason: str) -> None:
        import json

        with open(self._timing_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "phase": "restore_skipped",
                        "request_id": request_id,
                        "reason": reason,
                    }
                )
                + "\n"
            )
