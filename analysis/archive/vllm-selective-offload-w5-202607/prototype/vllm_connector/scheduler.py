"""P4 pause/resume scheduler: force-preempt a named request + hold it.

INJECTION MECHANISM (the architecture decision, with evidence)
--------------------------------------------------------------
vLLM 0.11.2 lets a custom scheduler be injected BY CONFIG -- no forked file:

- ``SchedulerConfig.scheduler_cls: str | type[object] | None`` -- "Can be a
  class directly or the path to a class of form 'mod.custom_class'."
  (vllm/config/scheduler.py, v0.11.2). Resolved by ``get_scheduler_cls()`` via
  ``resolve_obj_by_qualname`` for a string.
- ``EngineCore`` instantiates it with
  ``Scheduler = vllm_config.scheduler_config.get_scheduler_cls()`` then
  ``Scheduler(vllm_config=..., kv_cache_config=..., structured_output_manager
  =..., include_finished_set=..., log_stats=..., block_size=...)``
  (vllm/v1/engine/core.py, v0.11.2).
- ``EngineArgs.scheduler_cls`` surfaces it to ``AsyncEngineArgs``
  (vllm/engine/arg_utils.py, v0.11.2).

So this ``PausableScheduler(Scheduler)`` ships INSIDE our repo, versioned with
the ``vllm>=0.11,<0.12`` pin, and is selected with
``scheduler_cls="spike.vllm_connector.scheduler.PausableScheduler"``. ZERO
forked/vendored vLLM files. The P4 design memo assumed a fork patch of
``scheduler.py``; the config seam makes that unnecessary -- the only deviation
from the memo, and a strictly cleaner one (documented in the memo appendix).

WHAT IT ADDS (memo item 1 + the save-before-free handshake, item 2)
-------------------------------------------------------------------
- ``force_preempt(req_id)`` mirrors the upstream preemption free primitive
  (scheduler.py:321-324: kv_cache_manager.free / encoder_cache_manager.free /
  status=PREEMPTED / num_computed_tokens=0 / num_preemptions+=1) but does NOT
  ``waiting.prepend_request`` -- the request goes to a hold set instead, so the
  scheduler never reschedules it until an explicit resume (memo Q3).
- ``resume_request(req_id)`` moves a held request back to ``waiting`` via
  ``prepend_request``; the stock ``num_computed_tokens==0`` resume gate
  (scheduler.py:447) then consults our connector's
  ``get_num_new_matched_tokens`` and loads the saved KV (memo Q2).
- Save-before-free: on PAUSE we fetch the request's CURRENT block ids from
  ``kv_cache_manager.get_block_ids`` (the scheduler's own source of truth --
  NOT the connector's admission-time cache, which goes stale as a running
  request's decode grows past its originally allocated blocks), register the
  save with the connector, and mark the request "pausing" but DO NOT free --
  the blocks stay resident so the worker's synchronous save reads valid KV.
  Only once the worker confirms the save (a ``pause_saved`` row it appends to
  the timing file, polled by :class:`SaveConfirmationReader`) do we
  ``force_preempt`` and free (memo's synchronous-save v1; upgrade path is the
  async saved-set in KVConnectorOutput). A PAUSE trigger failure (e.g. a
  block-count mismatch) is contained by :func:`guard_pause_trigger` -- it
  fails the PAUSE, not the whole EngineCore process.

All correctness-bearing bookkeeping lives in CPU-tested pure logic
(:class:`PauseBook`, :class:`SaveConfirmationReader`, the connector's
``SavedKVRegistry``); this class is the thin vLLM-facing wrapper, verified on
the GPU box.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .control import OffloadControl, OffloadPhase
from .core import (
    PauseBook,
    SaveConfirmationReader,
    continuum_priority,
    guard_pause_trigger,
)

logger = logging.getLogger(__name__)

try:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.sched.request_queue import PriorityRequestQueue
    from vllm.v1.request import RequestStatus

    HAVE_VLLM = True
except ImportError:  # pragma: no cover - exercised only off-GPU
    HAVE_VLLM = False
    Scheduler = object  # type: ignore[assignment,misc]
    RequestStatus = None  # type: ignore[assignment]
    PriorityRequestQueue = object  # type: ignore[assignment,misc]


class PausableScheduler(Scheduler):  # type: ignore[misc,valid-type]
    """Scheduler with external pause/resume of a NAMED request (see module doc)."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        extra = self.vllm_config.kv_transfer_config.kv_connector_extra_config
        self._control = OffloadControl(extra["control_path"])
        self._confirm = SaveConfirmationReader(extra["timing_path"])
        self._events_path = extra.get(
            "pause_events_path",
            str(Path(extra["control_path"]).parent / "pause_events.jsonl"),
        )
        self._pause = PauseBook()
        self._last_epoch = -1
        # req_id -> block count freed at pause, stashed for the freed event.
        self._pause_blocks: dict[str, int] = {}

    # --- schedule() hook -------------------------------------------------
    def schedule(self) -> Any:
        self._drop_finished_paused()  # liveness: forget requests that finished
        self._read_control()  # PAUSE -> mark pausing+register; RESUME -> resume
        self._confirm_saves()  # save confirmed -> force_preempt + free
        return super().schedule()

    def _drop_finished_paused(self) -> None:
        # self.requests is ground truth: a finished request is gone from it, so
        # anything in a pause state but absent from requests finished between
        # trigger and pause -- forget it (memo Q4 trigger-liveness).
        tracked = list(self._pause.pausing) + list(self._pause.paused)
        gone = [rid for rid in tracked if rid not in self.requests]
        self._pause.drop_finished(gone)
        for rid in gone:
            self._pause_blocks.pop(rid, None)

    def _read_control(self) -> None:
        state = self._control.read()
        if state.epoch == self._last_epoch:
            return
        target = state.target_request_id
        if state.phase == OffloadPhase.PAUSE:
            # Advance the epoch BEFORE attempting the pause so a failure is
            # never retried on the next step (would spin on the same bad
            # trigger). guard_pause_trigger contains any exception here --
            # e.g. a stale block count failing assert_saved_covers_tokens --
            # to this PAUSE only; it must not kill EngineCore for every
            # co-tenant (GPU-confirmed crash this replaces).
            self._last_epoch = state.epoch
            exc = guard_pause_trigger(lambda: self._begin_pause(target))
            if exc is not None:
                logger.error(
                    "pause trigger failed for %r; refusing (engine keeps serving): %r",
                    target,
                    exc,
                )
        elif state.phase == OffloadPhase.RESUME:
            self._last_epoch = state.epoch
            self._request_resume(target)
        # RESIDENT / OFFLOAD / RESTORE: not ours (the connector owns those).

    def _request_resume(self, req_id: str | None) -> None:
        if req_id is None:
            logger.warning("resume skipped: no target request id")
            return
        if self._pause.is_pausing(req_id):
            # Save not confirmed yet -- a RESUME here would be silently
            # dropped by resume_request's is_paused guard, hanging the
            # request paused forever. Defer it; _confirm_saves honors it
            # immediately after force_preempt confirms the save.
            self._pause.defer_resume(req_id)
            logger.info("resume for %r deferred: pause-save still in flight", req_id)
            return
        self.resume_request(req_id)

    def _begin_pause(self, req_id: str | None) -> None:
        req = self.requests.get(req_id) if req_id else None
        if (
            req is None
            or req.status != RequestStatus.RUNNING
            or self._pause.is_pausing(req_id)
            or self._pause.is_paused(req_id)
        ):
            logger.warning(
                "pause skipped for %r: not a running, unpaused request", req_id
            )
            return
        n_c_t = req.num_computed_tokens
        # Fetch the request's CURRENT block ids from the scheduler's own
        # source of truth at the pause instant -- NOT the connector's
        # self._block_ids, which only refreshes on alloc (admission/resume)
        # and goes stale as a running request's decode grows past its
        # originally allocated blocks (GPU-confirmed: 97 admission blocks vs
        # 99 actual at 1581 computed tokens). kv_cache_manager.get_block_ids
        # mirrors update_state_after_alloc's own blocks.get_block_ids() call
        # (group 0), just read fresh instead of cached.
        block_ids = list(self.kv_cache_manager.get_block_ids(req_id)[0])
        # Register the save with our scheduler-side connector (in-process) and
        # mark pausing. Blocks are NOT freed yet -- the save reads them first.
        block_count = self.connector.register_pause_save(  # type: ignore[attr-defined]
            req_id, n_c_t, block_ids
        )
        self._pause.mark_pausing(req_id, n_c_t)
        self._pause_blocks[req_id] = block_count
        self._record_event(
            "pausing", req_id, num_computed_tokens=n_c_t, block_count=block_count
        )

    def _confirm_saves(self) -> None:
        for req_id in self._confirm.poll():
            if self._pause.is_pausing(req_id):
                self.force_preempt(req_id)
                if self._pause.pop_deferred_resume(req_id):
                    self.resume_request(req_id)

    # --- memo item 1: force-preempt (no reschedule) + resume -------------
    def force_preempt(self, req_id: str) -> None:
        """Free a named request's blocks and hold it (mirror scheduler.py:321-324)."""
        req = self.requests.get(req_id)
        if req is None:  # finished between confirm and free -- liveness no-op
            self._pause.pausing.pop(req_id, None)
            self._pause.resume_deferred.discard(req_id)
            self._pause_blocks.pop(req_id, None)
            logger.warning("force_preempt no-op: request %r is gone", req_id)
            return
        self.kv_cache_manager.free(req)
        self.encoder_cache_manager.free(req)
        req.status = RequestStatus.PREEMPTED
        req.num_computed_tokens = 0
        req.num_preemptions += 1
        if req in self.running:
            self.running.remove(req)
        self._pause.confirm_saved(req_id)  # pausing -> paused (held, no queue)
        blocks_freed = self._pause_blocks.pop(req_id, 0)
        self._record_event("freed", req_id, blocks_freed=blocks_freed)

    def resume_request(self, req_id: str | None) -> None:
        """Move a held request back to ``waiting`` so the stock resume path runs."""
        req = self.requests.get(req_id) if req_id else None
        if req_id is None or not self._pause.is_paused(req_id):
            logger.warning("resume skipped for %r: not paused", req_id)
            return
        if req is None:  # finished while paused -- liveness no-op
            self._pause.paused.pop(req_id, None)
            return
        self._pause.resume(req_id)
        self.waiting.prepend_request(req)
        self._record_event("resumed", req_id)

    def _record_event(self, phase: str, req_id: str, **fields: Any) -> None:
        row = {"phase": phase, "request_id": req_id, "t": time.perf_counter(), **fields}
        with open(self._events_path, "a") as f:
            f.write(json.dumps(row) + "\n")


class _ContinuumPriorityQueue(PriorityRequestQueue):  # type: ignore[misc,valid-type]
    def __init__(self, stride: int):
        super().__init__()
        self._stride = stride

    def prepend_request(self, request: Any) -> None:
        if request.status == RequestStatus.PREEMPTED:
            request.priority = continuum_priority(
                request.priority % self._stride,
                self._stride,
                ttl_hit=False,
                preempted=True,
            )
        super().prepend_request(request)


class RetentionScheduler(Scheduler):  # type: ignore[misc,valid-type]
    """Scheduler hook for timer expiry and ThunderAgent capacity eviction."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        extra = self.vllm_config.kv_transfer_config.kv_connector_extra_config
        self._retention_events_path = extra["retention_events_path"]
        stride = int(extra.get("continuum_priority_stride", 0))
        if stride:
            if not isinstance(self.waiting, PriorityRequestQueue):
                raise ValueError("Continuum requires vLLM priority scheduling")
            self.waiting = _ContinuumPriorityQueue(stride)

    def _free_request(self, request: Any) -> dict[str, Any] | None:
        params = super()._free_request(request)
        keep_blocks = self.connector.take_retained_block_count(request.request_id)
        if keep_blocks is not None:
            for manager in self.kv_cache_manager.coordinator.single_type_managers:
                blocks = manager.req_to_blocks[request.request_id]
                if len(blocks) < keep_blocks:
                    raise ValueError(
                        f"request {request.request_id!r} has {len(blocks)} blocks; "
                        f"cannot retain {keep_blocks}"
                    )
                suffix = blocks[keep_blocks:]
                for block in suffix:
                    manager.block_pool._maybe_evict_cached_block(block)
                manager.req_to_blocks[request.request_id] = blocks[:keep_blocks]
                manager.num_cached_block[request.request_id] = min(
                    manager.num_cached_block.get(request.request_id, 0),
                    keep_blocks,
                )
                manager.block_pool.free_blocks(reversed(suffix))
        return params

    def _free_blocks(self, request: Any) -> None:
        block_pool = self.kv_cache_manager.block_pool
        free_blocks_before = block_pool.get_num_free_blocks()
        block_ids = list(self.kv_cache_manager.get_block_ids(request.request_id)[0])
        ref_counts_before = {
            block_id: block_pool.blocks[block_id].ref_cnt for block_id in block_ids
        }
        super()._free_blocks(request)
        freed_block_ids = [
            block_id
            for block_id in block_ids
            if ref_counts_before[block_id] > 0
            and block_pool.blocks[block_id].ref_cnt == 0
            and not block_pool.blocks[block_id].is_null
        ]
        free_blocks_after = block_pool.get_num_free_blocks()
        if free_blocks_after - free_blocks_before != len(freed_block_ids):
            raise RuntimeError("freed block IDs do not match the free-pool delta")
        self._record_retention_event(
            "retention_blocks_freed",
            request_id=request.request_id,
            block_ids=freed_block_ids,
            free_blocks_before=free_blocks_before,
            free_blocks_after=free_blocks_after,
        )

    def has_requests(self) -> bool:
        connector = self.connector
        return super().has_requests() or (
            connector is not None and connector.has_due_retention()
        )

    def schedule(self) -> Any:
        connector = self.connector
        running_before = {request.request_id for request in self.running}
        block_pool = self.kv_cache_manager.block_pool
        free_blocks_before = block_pool.get_num_free_blocks()
        if connector is not None:
            connector.release_requested_programs()
            connector.refresh_retention_expiries()
            connector.release_retention_due(
                connector.matching_resident_program_ids(list(self.waiting))
            )
            num_gpu_blocks = self.cache_config.num_gpu_blocks
            if num_gpu_blocks is None or num_gpu_blocks <= 0:
                raise ValueError("vLLM did not configure a positive GPU block count")
            block_pool = self.kv_cache_manager.block_pool
            running = list(self.running)
            waiting = list(self.waiting)
            ordered_waiting = connector.order_thunder_waiting(
                waiting,
                capacity_tokens=connector.thunder_resume_capacity_tokens(
                    running,
                    free_tokens=block_pool.get_num_free_blocks() * self.block_size,
                ),
            )
            if ordered_waiting != waiting:
                self.waiting.remove_requests(waiting)
                for request in ordered_waiting:
                    self.waiting.add_request(request)
            requests = list(self.requests.values())
            active_block_ids = {
                request.request_id: tuple(
                    self.kv_cache_manager.get_block_ids(request.request_id)[0]
                )
                for request in requests
                if not request.is_finished()
            }
            connector.release_retention_for_pressure(
                requests,
                capacity_tokens=num_gpu_blocks * self.block_size,
                active_block_ids=active_block_ids,
                waiting_request=self.waiting.peek_request() if self.waiting else None,
            )
            for block_ids in connector.take_prefix_invalidations():
                for block_id in block_ids:
                    block_pool._maybe_evict_cached_block(block_pool.blocks[block_id])
            waiting = list(self.waiting)
            ordered_waiting = connector.order_thunder_waiting(
                waiting,
                capacity_tokens=connector.thunder_resume_capacity_tokens(
                    running,
                    free_tokens=block_pool.get_num_free_blocks() * self.block_size,
                ),
            )
            if ordered_waiting != waiting:
                self.waiting.remove_requests(waiting)
                for request in ordered_waiting:
                    self.waiting.add_request(request)
            waiting = list(self.waiting)
            if connector.refresh_continuum_priorities(waiting):
                self.waiting.remove_requests(waiting)
                for request in waiting:
                    self.waiting.add_request(request)
        output = super().schedule()
        free_blocks_after = block_pool.get_num_free_blocks()
        for request in self.running:
            if request.request_id not in running_before:
                self._record_retention_event(
                    "retention_request_admitted",
                    request_id=request.request_id,
                    block_ids=list(
                        self.kv_cache_manager.get_block_ids(request.request_id)[0]
                    ),
                    free_blocks_before=free_blocks_before,
                    free_blocks_after=free_blocks_after,
                )
        return output

    def _record_retention_event(self, phase: str, **fields: Any) -> None:
        row = {"phase": phase, "monotonic_s": time.monotonic(), **fields}
        with open(self._retention_events_path, "a") as file:
            file.write(json.dumps(row) + "\n")
