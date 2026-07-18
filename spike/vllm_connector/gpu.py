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
from .core import TransferTiming, validate_block_ids

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


class CudaBlockTransfer:
    """Copy paged KV blocks between the GPU cache and pinned host memory.

    ``kv_caches`` maps layer name -> the layer's paged KV tensor as handed to
    the connector by ``register_kv_caches``. The standard vLLM v1 layout puts
    the block index on dim 0, so block ``b`` of a layer is ``tensor[b]``; we
    gather the target blocks per layer into a pinned host tensor (offload) and
    scatter them back (restore), timing the whole gather/scatter with CUDA
    events. This is exactly the operation vLLM's own preemption-swap performs.

    ``block_dim`` is honored, not decorative: every shape/index computation
    below operates on ``tensor.movedim(block_dim, 0)`` (a view, so writes
    through it mutate the original storage), so a non-default layout is a
    single constructor arg away -- verify the real dim on the box (README).
    """

    def __init__(self, kv_caches: dict[str, "torch.Tensor"], block_dim: int = 0):
        if not HAVE_TORCH:  # pragma: no cover - off-GPU guard
            raise RuntimeError("CudaBlockTransfer requires torch (GPU box only)")
        if not kv_caches:
            raise ValueError("kv_caches is empty; register_kv_caches not called?")
        self.kv_caches = kv_caches
        self.block_dim = block_dim
        self._host: dict[str, torch.Tensor] = {}

    def _fronted(self, t: "torch.Tensor") -> "torch.Tensor":
        """View of ``t`` with the block dim moved to the front (dim 0)."""
        return t.movedim(self.block_dim, 0)

    def _bytes_per_block(self) -> int:
        total = 0
        for t in self.kv_caches.values():
            fronted = self._fronted(t)
            total += fronted[0].numel() * t.element_size()
        return total

    def _copy(self, block_ids: list[int], to_host: bool) -> TransferTiming:
        if not block_ids:
            raise ValueError("refusing to transfer zero blocks")
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
        return TransferTiming(bytes_moved=nbytes, milliseconds=ms, num_blocks=len(block_ids))

    def offload(self, block_ids: list[int]) -> TransferTiming:
        return self._copy(block_ids, to_host=True)

    def restore(self, block_ids: list[int]) -> TransferTiming:
        return self._copy(block_ids, to_host=False)


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
        self._role = role
        self._last_epoch = -1
        self._block_ids: dict[str, list[int]] = {}
        self._transfer: CudaBlockTransfer | None = None

    # --- worker side -----------------------------------------------------
    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        self._transfer = CudaBlockTransfer(kv_caches, block_dim=self._block_dim)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        # Execute any offload/restore directives the scheduler queued for this
        # step. Runs in the worker process where the KV tensors live.
        meta = self._get_connector_metadata()
        for phase, block_ids in meta.directives:  # type: ignore[attr-defined]
            assert self._transfer is not None
            timing = (
                self._transfer.offload(block_ids)
                if phase == OffloadPhase.OFFLOAD.value
                else self._transfer.restore(block_ids)
            )
            self._record(phase, timing)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(self, *args: Any, **kwargs: Any) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

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
