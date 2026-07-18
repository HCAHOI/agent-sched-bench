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
    TransferTiming,
    chunk_ranges,
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
    """Per-step connector metadata: the offload/restore directive, if any.

    Built in ``build_connector_meta`` (scheduler process), delivered to
    ``start_load_kv`` (worker process) via vLLM's own metadata-passing --
    verify on the box that a single step's metadata always makes it to the
    worker that executes it (README risk).
    """

    directives: list[tuple[str, list[int]]] = field(default_factory=list)


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
        self._role = role
        self._last_epoch = -1
        self._block_ids: dict[str, list[int]] = {}
        self._transfer: CudaBlockTransfer | None = None
        # Transfers kicked in start_load_kv, drained (timed + recorded) in
        # wait_for_save so start_load_kv never blocks the worker step.
        self._pending: list[tuple[str, Any]] = []

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
        # Kick any offload/restore directives the scheduler queued for this
        # step onto the transfer stream and return immediately -- do NOT block
        # the worker step behind the copy. Timing is read later in
        # wait_for_save. Runs in the worker process where the KV tensors live.
        meta = self._get_connector_metadata()
        for phase, block_ids in meta.directives:  # type: ignore[attr-defined]
            assert self._transfer is not None
            to_host = phase == OffloadPhase.OFFLOAD.value
            self._pending.append((phase, self._transfer.begin(block_ids, to_host=to_host)))

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
        for phase, pending in self._pending:
            self._record(phase, pending.wait())
        self._pending.clear()

    def _record(self, phase: str, timing: TransferTiming) -> None:
        import json

        with open(self._timing_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "phase": phase,
                        "milliseconds": timing.milliseconds,
                        "bytes_moved": timing.bytes_moved,
                        "num_blocks": timing.num_blocks,
                        "effective_gbps": timing.effective_gbps,
                    }
                )
                + "\n"
            )

    # --- scheduler side --------------------------------------------------
    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        # Track the target request's block ids so we can offload them later.
        self._block_ids[request.request_id] = list(blocks.get_block_ids()[0])

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

    def build_connector_meta(self, scheduler_output: Any) -> "KVConnectorMetadata":
        self._drop_finished(scheduler_output)
        meta = SelectiveOffloadMeta()
        state = self._control.read()
        if state.epoch != self._last_epoch and state.phase != OffloadPhase.RESIDENT:
            self._last_epoch = state.epoch
            target = state.target_request_id or ""
            block_ids = self._block_ids.get(target, [])
            if block_ids:
                meta.directives.append((state.phase.value, block_ids))
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
