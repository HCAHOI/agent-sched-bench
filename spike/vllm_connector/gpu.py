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

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .control import OffloadControl, OffloadPhase
from .core import (
    SavedKVRegistry,
    TransferTiming,
    assert_saved_covers_tokens,
    chunk_ranges,
    resume_load_block_ids,
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

    def _begin_strided(self, block_ids: list[int], *, to_host: bool) -> _CompletedTransfer:
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
            TransferTiming(bytes_moved=nbytes, milliseconds=ms, num_blocks=len(block_ids))
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
                    idx = torch.tensor(block_ids[c0:c1], device=dev.device, dtype=torch.long)
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

    def load_retained(self, request_id: str, block_ids: list[int]) -> TransferTiming:
        if request_id not in self._retained:
            raise ValueError(f"no retained save for request {request_id!r}")
        buf = self._retained[request_id]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for name, dev in self.kv_caches.items():
            dev_f = self._fronted(dev)
            validate_block_ids(dev_f.shape[0], block_ids)
            host = buf[name]
            if host.shape[0] != len(block_ids):
                raise ValueError(
                    f"resume-load {len(block_ids)} blocks != saved {host.shape[0]} "
                    f"for request {request_id!r} (self-hit partial-resume is an "
                    "untested low-pressure edge; target regime is full-delta)"
                )
            idx = torch.tensor(block_ids, device=dev.device, dtype=torch.long)
            dev_f.index_copy_(0, idx, host.to(dev.device, non_blocking=True))
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
        self._control = OffloadControl(extra["control_path"])
        self._timing_path = extra["timing_path"]
        self._block_dim = int(extra.get("block_dim", 0))
        self._mode = str(extra.get("transfer_mode", "staged"))
        self._max_blocks = int(extra.get("max_blocks", 128))
        self._chunk_bytes = int(extra.get("chunk_bytes", 0))
        self._block_size = int(vllm_config.cache_config.block_size)
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
        # Execute the directives the scheduler queued for this step. Runs in the
        # worker process where the KV tensors live.
        #   offload/restore -- kicked async onto the transfer stream, drained in
        #     wait_for_save / get_finished (do NOT block the worker step).
        #   pause  -- SYNCHRONOUS save into a dedicated retained buffer, then
        #     append a pause_saved confirmation the scheduler polls (memo: the
        #     one-off save is allowed to cost the full transfer once).
        #   resume -- SYNCHRONOUS load from the retained buffer into the freshly
        #     allocated blocks, so the KV is present before this step's forward.
        #   release -- drop a retained buffer with no transfer at all (P4
        #     cleanup: finish-while-paused or a free self-hit resume).
        meta = self._get_connector_metadata()
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
        # Called every step, INCLUDING kv_connector_no_forward / empty-batch
        # steps where wait_for_save is skipped -- exactly the steps a pause
        # creates. Draining here guarantees kicked offload/restore transfers are
        # always timed + recorded (lane-A carry-over fix). We do not use the
        # finished_sending/recving sets, so return empty.
        self._drain_pending()
        return None, None

    def _drain_pending(self) -> None:
        for phase, pending in self._pending:
            self._record(phase, pending.wait())
        self._pending.clear()

    def _record(
        self, phase: str, timing: TransferTiming, request_id: str | None = None
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
                    }
                )
                + "\n"
            )

    # --- scheduler side --------------------------------------------------
    def register_pause_save(self, request_id: str, num_computed_tokens: int) -> int:
        """Scheduler poke (in-process): queue a pause-save for a running request.

        Called by :class:`PausableScheduler` at the pause instant, BEFORE it
        frees anything. Records the saved-KV registry entry (so a later resume's
        get_num_new_matched_tokens can report the count) and queues the SAVE
        directive the worker executes. Fail fast unless the tracked blocks
        exactly cover ``num_computed_tokens``. Returns the block count (== blocks
        that will be freed).
        """
        block_ids = self._block_ids.get(request_id)
        if not block_ids:
            raise ValueError(f"cannot pause-save untracked request {request_id!r}")
        assert_saved_covers_tokens(num_computed_tokens, len(block_ids), self._block_size)
        self._saved.register(request_id, num_computed_tokens, len(block_ids))
        self._queued.append((OffloadPhase.PAUSE.value, request_id, list(block_ids)))
        return len(block_ids)

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        # For a resuming (previously paused) request, report the saved token
        # count MINUS the local prefix-cache hit (delta-only, memo Q2 self-hit);
        # every other request falls through to (0, False) exactly like the base.
        return self._saved.matched_tokens(request.request_id, num_computed_tokens)

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        # Track the request's block ids (used by pause-save and offload).
        new_block_ids = list(blocks.get_block_ids()[0])
        self._block_ids[request.request_id] = new_block_ids
        if not self._saved.is_saved(request.request_id):
            return
        if num_external_tokens > 0:
            # A resuming saved request just got fresh blocks allocated for its
            # external tokens. v0.11.2 hands us the FULL prefix block set here
            # (computed + new), which can be WIDER than the save if the
            # save-confirm lag advanced the request past a block boundary
            # (P4_EVICTION_DESIGN.md appendix) -- slice to what the retained
            # buffer actually covers; the scheduler recomputes the rest.
            saved_block_count = self._saved.saved[request.request_id][1]
            resume_block_ids = resume_load_block_ids(new_block_ids, saved_block_count)
            self._saved.record_resume_blocks(request.request_id, resume_block_ids)
            self._queued.append(
                (OffloadPhase.RESUME.value, request.request_id, resume_block_ids)
            )
        else:
            # Low-pressure self-hit (memo Q2): the local prefix cache already
            # covers the saved tokens, so there is nothing to load -- but the
            # retained pinned buffer would otherwise leak forever.
            self._queued.append((_RELEASE, request.request_id, []))
        self._saved.drop(request.request_id)  # resume (loaded or not) is once

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
                if self._saved.is_saved(req_id):
                    # Request finished/aborted while pausing-or-paused: the
                    # retained buffer has no resume coming, so release it
                    # explicitly rather than leaking it (memo Q4 liveness).
                    self._queued.append((_RELEASE, req_id, []))
                    self._saved.drop(req_id)

    def build_connector_meta(self, scheduler_output: Any) -> "KVConnectorMetadata":
        self._drop_finished(scheduler_output)
        meta = SelectiveOffloadMeta()
        # 1. Drain any pause-save / resume-load directives the scheduler queued
        #    in-process this step (P4 evict scenario).
        meta.directives.extend(self._queued)
        self._queued.clear()
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
            block_ids = self._block_ids.get(target, [])
            if block_ids:
                meta.directives.append((state.phase.value, target, block_ids))
            else:
                # Untracked or already-finished target: refuse the directive
                # rather than offload/restore nothing silently.
                logger.warning(
                    "offload directive for request %r (phase=%s) skipped: "
                    "not tracked or already finished",
                    target,
                    state.phase.value,
                )
        return meta
