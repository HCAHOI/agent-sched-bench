"""Exclusive tiering for the LMCache DRAM tier (LMCache 0.3.7, vLLM 0.10.2, one engine, TP=1).

Loaded through PYTHONPATH by run_two_instance_fcfs.sh when EXCLUSIVE_TIER=1 (the same run writes
`extra_config: {exclusive_tier: true}` into lmcache.yaml; the two must agree or the engine refuses to start).

Stock LMCache is inclusive: every prefilled chunk is copied to DRAM and a loaded context is promoted to most
recently used, so the DRAM tier holds every active context twice over (in flight in HBM and in DRAM) and, when
full, evicts the contexts waiting in tool gaps, which are the ones about to return (M4 §3.3). This patch makes
the DRAM tier prefer evicting what HBM already holds:

  1. chunks of an in-flight request (just loaded or just stored) are moved to the evict-first end of the LRU;
  2. when a request finishes, its blocks are held for one more step (vLLM's delayed free) while the worker
     looks up how much of the context is still in DRAM, copies back only the missing tail, and promotes the whole
     context to most recently used;
  3. nothing else changes: lookups, loads, chunking, capacity and the LRU order of tool-gap contexts.

With no DRAM pressure the copy traffic is identical to inclusive (every token stored once). Under pressure the
finish copy-back of evicted in-flight chunks is the mechanism's cost; every such event is logged as
`[exclusive-tier] finish ...` for the post-run accounting. Preempted requests lose their evict-first chunks
first and recompute (1-2% of requests at c24).
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import time

_TARGET = "lmcache.integration.vllm.vllm_v1_adapter"


def _patch(M) -> None:
    import torch
    from lmcache.logging import init_logger
    from lmcache.v1.storage_backend.cache_policy.lru import LRUCachePolicy
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

    log = init_logger("exclusive-tier")

    def demote(self, keys) -> None:
        """Move chunks to the evict-first end of the LRU (their content is also in HBM)."""
        with self.cpu_lock:
            for key in keys:
                if key in self.hot_cache:
                    self.hot_cache.move_to_end(key, last=False)

    def promote(self, keys) -> None:
        with self.cpu_lock:
            for key in keys:
                if key in self.hot_cache:
                    self.hot_cache.move_to_end(key)

    LocalCPUBackend.demote = demote
    LocalCPUBackend.promote = promote

    Impl = M.LMCacheConnectorV1Impl
    orig_init = Impl.__init__
    orig_request_finished = Impl.request_finished
    orig_build_meta = Impl.build_connector_meta
    orig_wait_for_save = Impl.wait_for_save
    orig_get_finished = Impl.get_finished

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        extra = self.config.extra_config or {}
        if extra.get("exclusive_tier") is not True:
            raise RuntimeError("exclusive-tier patch loaded but lmcache.yaml extra_config.exclusive_tier is not true")
        if self.worker_count != 1 or self.kv_role != "kv_both":
            raise RuntimeError("exclusive tiering is implemented for one engine with TP=1 and kv_role kv_both")
        self._xt_finish_queue: list = []   # scheduler side: (req_id, token_ids, block_ids)
        self._xt_mismatch = 0
        if self.lmcache_engine is not None:  # worker side
            if self.use_layerwise:
                raise RuntimeError("exclusive tiering needs use_layerwise=False (chunk keys, not per-layer keys)")
            cpu = self.lmcache_engine.storage_manager.storage_backends["LocalCPUBackend"]
            if not isinstance(cpu.cache_policy, LRUCachePolicy):
                raise RuntimeError("exclusive tiering orders an LRU; cache_policy must be LRU")
            self._xt_cpu = cpu
        log.info("exclusive tiering active (role=%s)", "worker" if self.lmcache_engine is not None else "scheduler")

    # ---- scheduler side -------------------------------------------------------------------------------------
    def request_finished(self, request, block_ids):
        delay, params = orig_request_finished(self, request, block_ids)
        tracker = self._request_trackers.get(request.request_id)
        if tracker is None:
            return delay, params
        n = min(len(tracker.token_ids), request.num_computed_tokens)
        if n != len(tracker.token_ids):
            self._xt_mismatch += 1
            log.warning("finish %s: tracker has %d tokens, computed %d (mismatch #%d)", request.request_id,
                        len(tracker.token_ids), request.num_computed_tokens, self._xt_mismatch)
        n = n // self._lmcache_chunk_size * self._lmcache_chunk_size
        if n == 0:
            return delay, params
        self._xt_finish_queue.append((request.request_id, tracker.token_ids[:n], list(block_ids)))
        return True, params

    def build_connector_meta(self, scheduler_output):
        meta = orig_build_meta(self, scheduler_output)
        finish = []
        for req_id, token_ids, block_ids in self._xt_finish_queue:
            bs = self._block_size
            assert len(token_ids) <= len(block_ids) * bs, (req_id, len(token_ids), len(block_ids), bs)
            blocks = torch.tensor(block_ids, dtype=torch.long)
            slots = (torch.arange(0, bs, dtype=torch.long).reshape(1, bs) + blocks.reshape(-1, 1) * bs)
            finish.append(M.ReqMeta(req_id=req_id, token_ids=token_ids, slot_mapping=slots.flatten()[: len(token_ids)],
                                    is_last_prefill=True, save_spec=M.SaveSpec(0, True)))
        self._xt_finish_queue = []
        meta.xt_finish = finish
        return meta

    # ---- worker side ----------------------------------------------------------------------------------------
    def _keys(self, token_ids):
        return [key for _, _, key in self.lmcache_engine.token_database.process_tokens(token_ids)]

    def wait_for_save(self):
        orig_wait_for_save(self)
        # Every request in this step's metadata is in flight (loaded or stored this step): evict-first.
        meta = self._parent._get_connector_metadata()
        for request in meta.requests:
            self._xt_cpu.demote(_keys(self, request.token_ids))

    def get_finished(self, finished_req_ids):
        sending, recving = orig_get_finished(self, finished_req_ids)
        meta = self._parent._get_connector_metadata()
        finish = getattr(meta, "xt_finish", None) or []
        if not finish:
            return sending, recving
        engine = self.lmcache_engine
        kvcaches = list(self.kv_caches.values())
        done = set(sending or ())
        for rm in finish:
            t0 = time.perf_counter()
            tokens = rm.token_ids
            keys = _keys(self, tokens)
            chunks = 0  # contiguous prefix still in DRAM (a hole ends it: LMCache lookups stop at the first miss)
            while chunks < len(keys) and self._xt_cpu.contains(keys[chunks]):
                chunks += 1
            present = chunks * self._lmcache_chunk_size
            self._xt_cpu.promote(keys[:chunks])  # the store below must not evict this request's own prefix
            stored = len(tokens) - present
            if stored > 0:
                mask = torch.ones(len(tokens), dtype=torch.bool)
                mask[:present] = False
                engine.store(tokens, mask=mask, kvcaches=kvcaches, slot_mapping=rm.slot_mapping.cuda(), offset=present)
                self._xt_cpu.promote(keys)
            done.add(rm.req_id)
            log.info("[exclusive-tier] finish req=%s tokens=%d present=%d stored=%d ms=%.1f", rm.req_id, len(tokens),
                     present, stored, (time.perf_counter() - t0) * 1e3)
        return done, recving

    Impl.__init__ = __init__
    Impl.request_finished = request_finished
    Impl.build_connector_meta = build_connector_meta
    Impl.wait_for_save = wait_for_save
    Impl.get_finished = get_finished
    print("[exclusive-tier] LMCache adapter patched", flush=True)


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Patches the adapter module right after its normal import; every other import is untouched."""

    _busy = False

    def find_spec(self, name, path, target=None):
        if name != _TARGET or _Finder._busy:
            return None
        _Finder._busy = True
        try:
            spec = importlib.util.find_spec(name)
        finally:
            _Finder._busy = False
        if spec is None:
            return None
        self._loader = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _patch(module)


sys.meta_path.insert(0, _Finder())
