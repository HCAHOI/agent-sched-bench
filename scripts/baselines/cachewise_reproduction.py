#!/usr/bin/env python3
"""Reproduce CacheWise's published scheduling policy on its official vLLM fork.

Pinned sources:

* predictor: cachewise-project/cachewise-coding-traces@181c435a
* serving fork: cachewise-project/vllm@16cc7d43

The authors' vLLM branch contains the transport and heap skeleton, but its
released score is a Python hash, its waiting policy maximizes cached tokens,
and it never performs the paper's N_rebuild=3 rescore.  ``VLLM_PATCH`` replaces
those placeholders with the published policy equations while retaining the
authors' request-body transport.

The paper does not publish how the engine learns a just-generated tool call.
This reproduction adds a narrow engine update endpoint: after an LLM response
reveals a tool call, the client selects one official survival curve and updates
that session's resident KV metadata.  Attaching the eventual tool call to the
request that generates it remains hindsight and is invalid.
The paper specifies no online miss-feedback or learning rule; similarity misses
retain the official predictor's overall-per-tool curve fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

PREDICTOR_REPO = "https://github.com/cachewise-project/cachewise-coding-traces.git"
PREDICTOR_COMMIT = "181c435a090d328d00bbbee4c8eeb27d32f3abd2"
VLLM_REPO = "https://github.com/cachewise-project/vllm.git"
VLLM_COMMIT = "16cc7d43d0e1a84f68f046e6caecfef21012f3fc"
VLLM_UPSTREAM_BASE = "b1388b1fbf5aaef47937fabe98931211684666a6"
N_REBUILD = 3
_MODEL_ARTIFACT_SUFFIXES = (".json", ".pkl")
_ORACLE_LENGTH_EVENT_LOG_ENV = "CACHEWISE_ORACLE_LENGTH_EVENT_LOG"
_ORACLE_PREFILL_MS_PER_TOKEN_ENV = "CACHEWISE_ORACLE_PREFILL_MS_PER_TOKEN"
_ORACLE_DECODE_MS_PER_TOKEN_ENV = "CACHEWISE_ORACLE_DECODE_MS_PER_TOKEN"
_ORACLE_LENGTH_MARKER = "# CACHEWISE_ORACLE_LENGTH_V1"
_ORACLE_LENGTH_HEADER = '''        """Pick the request requiring the fewest additional KV blocks."""
        if not self.waiting:
            return None
        best: Request | None = None
        best_key: tuple[float, float, str] | None = None
'''
_ORACLE_LENGTH_HEADER_PATCH = '''        """Pick the request with the least cache-aware Oracle service."""
        # CACHEWISE_ORACLE_LENGTH_V1
        from scripts.baselines.cachewise_reproduction import (
            cachewise_oracle_length_key,
            record_cachewise_oracle_length_decision,
        )
        if not self.waiting:
            return None
        best: Request | None = None
        best_key: tuple[float, float, str] | None = None
        cachewise_best: Request | None = None
        cachewise_best_key: tuple[float, float, str] | None = None
        candidate_count = 0
'''
_ORACLE_LENGTH_KEY = '''            key = (additional_blocks, req.arrival_time, req.request_id)
            if best_key is None or key < best_key:
                best_key = key
                best = req
'''
_ORACLE_LENGTH_KEY_PATCH = '''            candidate_count += 1
            cachewise_key = (
                float(additional_blocks), req.arrival_time, req.request_id
            )
            if cachewise_best_key is None or cachewise_key < cachewise_best_key:
                cachewise_best_key = cachewise_key
                cachewise_best = req
            key = cachewise_oracle_length_key(
                req, missing_tokens, additional_blocks
            )
            if best_key is None or key < best_key:
                best_key = key
                best = req
'''
_ORACLE_LENGTH_RETURN = '''        return best


    def _handle_stopped_request'''
_ORACLE_LENGTH_RETURN_PATCH = '''        record_cachewise_oracle_length_decision(
            cachewise_best, best, candidate_count
        )
        return best


    def _handle_stopped_request'''


def cachewise_oracle_length_key(
    request: Any, missing_tokens: int, additional_blocks: int
) -> tuple[float, float, str]:
    """Estimate cache-aware request service using exact trace output length."""

    if os.environ.get(_ORACLE_LENGTH_EVENT_LOG_ENV) is None:
        return (float(additional_blocks), request.arrival_time, request.request_id)
    output_tokens = request.priority
    if not isinstance(output_tokens, int) or output_tokens < 1:
        raise RuntimeError("CacheWise Oracle request requires positive output tokens")
    try:
        prefill_ms_per_token = float(os.environ[_ORACLE_PREFILL_MS_PER_TOKEN_ENV])
        decode_ms_per_token = float(os.environ[_ORACLE_DECODE_MS_PER_TOKEN_ENV])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("CacheWise Oracle service coefficients are required") from exc
    if not all(
        math.isfinite(value) and value > 0
        for value in (prefill_ms_per_token, decode_ms_per_token)
    ):
        raise RuntimeError("CacheWise Oracle service coefficients must be positive")
    service_ms = (
        missing_tokens * prefill_ms_per_token
        + output_tokens * decode_ms_per_token
    )
    return (service_ms, request.arrival_time, request.request_id)


def record_cachewise_oracle_length_decision(
    cachewise_request: Any, selected_request: Any, candidate_count: int
) -> None:
    """Record whether adding exact output length changed CacheWise selection."""

    event_log = os.environ.get(_ORACLE_LENGTH_EVENT_LOG_ENV)
    if event_log is None or cachewise_request is None or selected_request is None:
        return
    event = {
        "schema": "cachewise-oracle-length-v1",
        "candidate_count": candidate_count,
        "cachewise_request_id": cachewise_request.request_id,
        "cachewise_output_tokens": cachewise_request.priority,
        "selected_request_id": selected_request.request_id,
        "selected_output_tokens": selected_request.priority,
        "selection_changed": cachewise_request.request_id
        != selected_request.request_id,
    }
    with Path(event_log).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, separators=(",", ":")) + "\n")


def patch_cachewise_oracle_length_scheduler(path: Path) -> bool:
    """Add the optional combined CacheWise and exact-length waiting score."""

    source = path.read_text(encoding="utf-8")
    if _ORACLE_LENGTH_MARKER in source:
        return False
    for anchor in (
        _ORACLE_LENGTH_HEADER,
        _ORACLE_LENGTH_KEY,
        _ORACLE_LENGTH_RETURN,
    ):
        if source.count(anchor) != 1:
            raise ValueError(f"unexpected CacheWise scheduler anchor: {anchor!r}")
    source = source.replace(
        _ORACLE_LENGTH_HEADER, _ORACLE_LENGTH_HEADER_PATCH, 1
    ).replace(_ORACLE_LENGTH_KEY, _ORACLE_LENGTH_KEY_PATCH, 1).replace(
        _ORACLE_LENGTH_RETURN, _ORACLE_LENGTH_RETURN_PATCH, 1
    )
    temporary = path.with_name(path.name + ".cachewise-oracle-length.tmp")
    temporary.write_text(source, encoding="utf-8")
    os.chmod(temporary, path.stat().st_mode)
    os.replace(temporary, path)
    return True

VLLM_PATCH = r'''
diff --git a/vllm/entrypoints/serve/cache/api_router.py b/vllm/entrypoints/serve/cache/api_router.py
index 10015f02c..7d99d1500 100644
--- a/vllm/entrypoints/serve/cache/api_router.py
+++ b/vllm/entrypoints/serve/cache/api_router.py
@@ -3,7 +3,7 @@
__BLANK_CONTEXT__
__BLANK_CONTEXT__
 from fastapi import APIRouter, FastAPI, Query, Request
-from fastapi.responses import Response
+from fastapi.responses import JSONResponse, Response
__BLANK_CONTEXT__
 import vllm.envs as envs
 from vllm.engine.protocol import EngineClient
@@ -44,6 +44,15 @@ async def reset_prefix_cache(
     return Response(status_code=200)
__BLANK_CONTEXT__
__BLANK_CONTEXT__
+@router.post("/cachewise/sessions/update")
+async def update_cachewise_session(raw_request: Request):
+    policy = await raw_request.json()
+    updated = await engine_client(raw_request).engine_core.call_utility_async(
+        "update_cachewise_session_policy", policy
+    )
+    return JSONResponse(content={"updated": updated})
+
+
 @router.post("/reset_mm_cache")
 async def reset_mm_cache(raw_request: Request):
     """
diff --git a/vllm/v1/core/block_pool.py b/vllm/v1/core/block_pool.py
index 96f3bbaa8..9eae13516 100644
--- a/vllm/v1/core/block_pool.py
+++ b/vllm/v1/core/block_pool.py
@@ -26,7 +26,10 @@ from vllm.v1.core.kv_cache_utils import (
     maybe_convert_block_hash,
 )
 from vllm.v1.request import Request
-from vllm.v1.core.cachewise_policy import cachewise_eviction_score
+from vllm.v1.core.cachewise_policy import (
+    CachewiseSessionPolicies,
+    cachewise_eviction_score,
+)
 import heapq
__BLANK_CONTEXT__
 logger = init_logger(__name__)
@@ -192,35 +195,55 @@ class BlockPool:
         self._free_heap: list[tuple[int, float, int, int]] = []
         self._free_ids: set[int] = set()
         self._free_heap_seq: int = 0
+        self._cachewise_session_policies = CachewiseSessionPolicies()
         self._free_heap_slack: float = 2.0  # rebuild if len(heap) > slack * len(_free_ids)
         if self.enable_cachewise_free_heap:
             self._init_free_heap_from_dll()
__INDENTED_BLANK_CONTEXT__
-    def _free_heap_key(self, block: KVCacheBlock) -> tuple[int, float, int, int]:
+    def _free_heap_key(
+        self,
+        block: KVCacheBlock,
+        now_s: float | None = None,
+        score_cache: dict[str | None, float] | None = None,
+    ) -> tuple[int, float, int, int]:
         """Min-heap key: smaller = allocate / reuse this free row first.
         Tuple:
         - 0 = no prefix hash yet (uncached free slot), 1 = still prefix-cached
           while free — prefer 0 so we reuse empty slots before dropping prefix.
-        - Negated cachewise score so *larger* policy score is *less* urgent to
-          pop first (tune: flip sign if you want highest-score-first).
+        - Negated expected time to reuse, so the session predicted to return
+          latest is evicted first, as specified by CacheWise.
         - free_heap_seq: LRU tie among equal (tier, score).
         - block_id: final tie-break.
         """
         tier = 0 if block.block_hash is None else 1
-        s = (
-            cachewise_eviction_score(block.cachewise_policy)
-            if self.enable_caching
-            else 0.0
-        )
+        s = 0.0
+        if self.enable_caching:
+            session_id = self._cachewise_session_policies.session_id(
+                block.cachewise_policy
+            )
+            if score_cache is not None and session_id in score_cache:
+                s = score_cache[session_id]
+            else:
+                s = cachewise_eviction_score(block.cachewise_policy, now_s)
+                if score_cache is not None:
+                    score_cache[session_id] = s
         return (tier, -s, block.free_heap_seq, block.block_id)
-    def _push_free_block_to_heap(self, block: KVCacheBlock) -> None:
+
+    def _push_free_block_to_heap(
+        self,
+        block: KVCacheBlock,
+        score_cache: dict[str | None, float] | None = None,
+    ) -> None:
         if not self.enable_cachewise_free_heap:
             return
         assert block.ref_cnt == 0 and not block.is_null
         self._free_heap_seq += 1
         block.free_heap_seq = self._free_heap_seq
         self._free_ids.add(block.block_id)
-        heapq.heappush(self._free_heap, self._free_heap_key(block))
+        heapq.heappush(
+            self._free_heap,
+            self._free_heap_key(block, score_cache=score_cache),
+        )
         self._maybe_rebuild_free_heap_if_stale()
     def _init_free_heap_from_dll(self) -> None:
         self._free_heap.clear()
@@ -237,27 +260,42 @@ class BlockPool:
         if len(self._free_heap) <= int(len(self._free_ids) * self._free_heap_slack):
             return
         self._rebuild_free_heap()
-    def _rebuild_free_heap(self) -> None:
-        # Resync from authoritative DLL membership.
+    def _rebuild_free_heap(self, now_s: float | None = None) -> None:
+        """Rescore all free blocks once per session and preserve LRU ties."""
         self._free_ids.clear()
         self._free_heap.clear()
         self._free_heap_seq = 0
+        score_cache: dict[str | None, float] = {}
         for block in self.free_block_queue.get_all_free_blocks():
             if block.is_null:
                 continue
             self._free_ids.add(block.block_id)
-        for bid in sorted(self._free_ids):
-            block = self.blocks[bid]
             self._free_heap_seq += 1
             block.free_heap_seq = self._free_heap_seq
-            self._free_heap.append(self._free_heap_key(block))
+            self._free_heap.append(
+                self._free_heap_key(block, now_s, score_cache)
+            )
         heapq.heapify(self._free_heap)
+
+    def rebuild_cachewise_heap(self, now_s: float) -> None:
+        if self.enable_cachewise_free_heap:
+            self._rebuild_free_heap(now_s)
+
+    def update_cachewise_session_policy(
+        self, policy: dict[str, Any] | None, now_s: float
+    ) -> dict[str, Any] | None:
+        shared = self._cachewise_session_policies.update(policy, self.blocks)
+        self.rebuild_cachewise_heap(now_s)
+        return shared
+
     def _pop_heap_free_block(self) -> KVCacheBlock:
         while self._free_heap:
             _tier, _ns, _seq, bid = heapq.heappop(self._free_heap)
             if bid not in self._free_ids:
                 continue
             block = self.blocks[bid]
+            if block.free_heap_seq != _seq:
+                continue
             if block.ref_cnt != 0 or block.is_null:
                 self._free_ids.discard(bid)
                 continue
@@ -355,7 +393,9 @@ class BlockPool:
             )
             blk.block_hash = block_hash_with_group_id
             self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
-            blk.cachewise_policy = request.cachewise_policy
+            blk.cachewise_policy = self._cachewise_session_policies.canonical(
+                request.cachewise_policy
+            )
             if new_hashes is not None:
                 new_hashes.append(maybe_convert_block_hash(block_hash))
__BLANK_CONTEXT__
@@ -577,8 +617,9 @@ class BlockPool:
         ]
         self.free_block_queue.append_n(to_append)
         if self.enable_cachewise_free_heap:
+            score_cache: dict[str | None, float] = {}
             for block in to_append:
-                self._push_free_block_to_heap(block)
+                self._push_free_block_to_heap(block, score_cache)
__BLANK_CONTEXT__
__BLANK_CONTEXT__
     def evict_blocks(self, block_ids: set[int]) -> None:
diff --git a/vllm/v1/core/cachewise_policy.py b/vllm/v1/core/cachewise_policy.py
index 5415ea957..f5db6acf6 100644
--- a/vllm/v1/core/cachewise_policy.py
+++ b/vllm/v1/core/cachewise_policy.py
@@ -1,41 +1,209 @@
-"""Parse cachewise_policy from the client JSON and compute priorities for KV cache."""
+"""Parse CacheWise metadata and score KV-cache eviction candidates."""
__BLANK_CONTEXT__
 from __future__ import annotations
-import json
+
+import bisect
+import math
+import time
+from collections.abc import Iterable
 from typing import Any
-_V = 1
__BLANK_CONTEXT__
-def parse_cachewise_policy_body(raw: Any) -> dict[str, Any] | None:
-    if raw is None or not isinstance(raw, dict):
+_VERSION = 1
+
+
+def _parse_curve(raw: Any) -> list[tuple[float, float]] | None:
+    if not isinstance(raw, list) or not raw:
         return None
-    if raw.get("version", _V) != _V:
+    curve: list[tuple[float, float]] = []
+    previous_t = -1.0
+    previous_p = 1.0
+    for point in raw:
+        if not isinstance(point, dict):
+            return None
+        try:
+            t_ms = float(point["t_ms"])
+            probability = float(point["prob_still_running"])
+        except (KeyError, TypeError, ValueError):
+            return None
+        if (
+            not math.isfinite(t_ms)
+            or not math.isfinite(probability)
+            or t_ms < 0
+            or t_ms <= previous_t
+            or not 0 <= probability <= previous_p
+        ):
+            return None
+        curve.append((t_ms, probability))
+        previous_t = t_ms
+        previous_p = probability
+    return curve
+
+
+def parse_cachewise_policy_body(raw: Any) -> dict[str, Any] | None:
+    if not isinstance(raw, dict) or raw.get("version", _VERSION) != _VERSION:
         return None
-    scope = raw.get("scope")
-    if scope is not None and not isinstance(scope, dict):
+    scope = raw.get("scope") or {}
+    hints = raw.get("hints") or {}
+    if not isinstance(scope, dict) or not isinstance(hints, dict):
         return None
-    hints = raw.get("hints")
-    if hints is not None and not isinstance(hints, dict):
+    session_id = scope.get("session_id")
+    if not isinstance(session_id, str) or not session_id:
         return None
-    return {"version": _V, "scope": dict(scope) if scope else {}, "hints": dict(hints) if hints else {}}
__BLANK_CONTEXT__
+    parsed_hints: dict[str, Any] = {}
+    if set(hints) - {"duration_curve", "oracle", "elapsed_ms"}:
+        return None
+    if "duration_curve" in hints and "oracle" in hints:
+        return None
+    curve = None
+    if "duration_curve" in hints:
+        curve = _parse_curve(hints["duration_curve"])
+        if curve is None:
+            return None
+        parsed_hints["duration_curve"] = curve
+    oracle = hints.get("oracle")
+    if "oracle" in hints:
+        if not isinstance(oracle, dict):
+            return None
+        if set(oracle) not in ({"total_duration_ms"}, {"ground_truth_idle_s"}):
+            return None
+        try:
+            total_ms = float(oracle["total_duration_ms"])
+        except (KeyError, TypeError, ValueError):
+            try:
+                remaining_ms = float(oracle["ground_truth_idle_s"]) * 1000
+            except (KeyError, TypeError, ValueError):
+                return None
+            else:
+                if not math.isfinite(remaining_ms) or remaining_ms < 0:
+                    return None
+                parsed_hints["oracle_remaining_ms"] = remaining_ms
+        else:
+            if not math.isfinite(total_ms) or total_ms < 0:
+                return None
+            parsed_hints["oracle_total_duration_ms"] = total_ms
+
+    try:
+        elapsed_ms = float(hints.get("elapsed_ms", 0.0))
+    except (TypeError, ValueError):
+        return None
+    if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
+        return None
+    parsed_hints["elapsed_ms"] = elapsed_ms
+    parsed_hints["attached_monotonic_s"] = time.monotonic()
+    return {
+        "version": _VERSION,
+        "scope": {
+            "session_id": session_id,
+            "idle_session": scope.get("idle_session") is True,
+        },
+        "hints": parsed_hints,
+    }
+
+
+class CachewiseSessionPolicies:
+    """Keep one mutable policy object for every agent session."""
+
+    def __init__(self) -> None:
+        self._by_session: dict[str, dict[str, Any]] = {}
+
+    @staticmethod
+    def session_id(policy: dict[str, Any] | None) -> str | None:
+        if not policy:
+            return None
+        session_id = (policy.get("scope") or {}).get("session_id")
+        return session_id if isinstance(session_id, str) else None
__BLANK_CONTEXT__
-def cachewise_eviction_score(policy: dict[str, Any] | None) -> float:
-    """Higher => keep longer when reordering eviction among cached+free blocks."""
+    def update(
+        self,
+        policy: dict[str, Any] | None,
+        resident_blocks: Iterable[Any],
+    ) -> dict[str, Any] | None:
+        session_id = self.session_id(policy)
+        if session_id is None or policy is None:
+            return policy
+        shared = self._by_session.get(session_id)
+        if shared is None:
+            shared = policy
+            self._by_session[session_id] = shared
+        elif shared is not policy:
+            shared.clear()
+            shared.update(policy)
+        for block in resident_blocks:
+            if self.session_id(block.cachewise_policy) == session_id:
+                block.cachewise_policy = shared
+        return shared
+
+    def canonical(
+        self, policy: dict[str, Any] | None
+    ) -> dict[str, Any] | None:
+        session_id = self.session_id(policy)
+        if session_id is None or policy is None:
+            return policy
+        return self._by_session.setdefault(session_id, policy)
+
+
+def _survival_at(curve: list[tuple[float, float]], elapsed_ms: float) -> float:
+    times = [point[0] for point in curve]
+    index = bisect.bisect_left(times, elapsed_ms)
+    if index == 0:
+        return curve[0][1]
+    if index >= len(curve):
+        return 0.0
+    t0, p0 = curve[index - 1]
+    t1, p1 = curve[index]
+    weight = (elapsed_ms - t0) / (t1 - t0)
+    return p0 + weight * (p1 - p0)
+
+
+def _expected_remaining_ms(
+    curve: list[tuple[float, float]], elapsed_ms: float
+) -> float:
+    survival = _survival_at(curve, elapsed_ms)
+    if survival <= 0:
+        return 0.0
+    index = bisect.bisect_left([point[0] for point in curve], elapsed_ms)
+    if index >= len(curve):
+        return 0.0
+    start_probability = _survival_at(curve, elapsed_ms)
+    integral = 0.0
+    if elapsed_ms < curve[index][0]:
+        integral += (
+            (curve[index][0] - elapsed_ms)
+            * (start_probability + curve[index][1])
+            / 2.0
+        )
+    for left, right in zip(curve[index:], curve[index + 1 :]):
+        integral += (right[0] - left[0]) * (left[1] + right[1]) / 2.0
+    return integral / survival
+
+
+def cachewise_eviction_score(
+    policy: dict[str, Any] | None, now_s: float | None = None
+) -> float:
+    """Return expected seconds to reuse; larger scores are evicted first."""
__BLANK_CONTEXT__
     if not policy:
         return 0.0
+    if (policy.get("scope") or {}).get("idle_session") is True:
+        return math.inf
     hints = policy.get("hints") or {}
-    oracle = hints.get("oracle")
-    if isinstance(oracle, dict):
-        try:
-            return float(oracle.get("ground_truth_idle_s", 0) or 0)
-        except (TypeError, ValueError):
-            return 0.0
-
-    next_tool = hints.get("next_tool")
-    if isinstance(next_tool, dict):
-        name = str(next_tool.get("name") or "")
-        args = next_tool.get("args")
-        tail = json.dumps(args, sort_keys=True) if args is not None else ""
-        return float(hash(name + "\0" + tail) % (2**31))
-    return 0.0
\ No newline at end of file
+    raw_oracle = hints.get("oracle")
+    if isinstance(raw_oracle, dict) and "ground_truth_idle_s" in raw_oracle:
+        return max(0.0, float(raw_oracle["ground_truth_idle_s"]))
+    attached_s = float(hints.get("attached_monotonic_s", 0.0))
+    elapsed_ms = float(hints.get("elapsed_ms", 0.0))
+    current_s = time.monotonic() if now_s is None else now_s
+    elapsed_ms += max(0.0, current_s - attached_s) * 1000
+
+    oracle_total_ms = hints.get("oracle_total_duration_ms")
+    if oracle_total_ms is not None:
+        return max(0.0, float(oracle_total_ms) - elapsed_ms) / 1000.0
+    oracle_remaining_ms = hints.get("oracle_remaining_ms")
+    if oracle_remaining_ms is not None:
+        elapsed_since_attach_ms = max(0.0, current_s - attached_s) * 1000
+        return max(0.0, float(oracle_remaining_ms) - elapsed_since_attach_ms) / 1000
+    curve = hints.get("duration_curve")
+    if not curve:
+        return 0.0
+    return _expected_remaining_ms(curve, elapsed_ms) / 1000.0
diff --git a/vllm/v1/core/sched/scheduler.py b/vllm/v1/core/sched/scheduler.py
index c0bca1169..07ecbbbde 100644
--- a/vllm/v1/core/sched/scheduler.py
+++ b/vllm/v1/core/sched/scheduler.py
@@ -63,6 +63,9 @@ from vllm.v1.utils import record_function_or_nullcontext
__BLANK_CONTEXT__
 logger = init_logger(__name__)
__BLANK_CONTEXT__
+# Fixed by the CacheWise evaluation (§5.3), not tuned in this reproduction.
+_CACHEWISE_REBUILD_INTERVAL = 3
+
__BLANK_CONTEXT__
 class Scheduler(SchedulerInterface):
     def __init__(
@@ -235,6 +238,7 @@ class Scheduler(SchedulerInterface):
             metrics_collector=self.kv_metrics_collector,
             enable_cachewise_free_heap=self.cache_config.enable_cachewise_free_heap,
         )
+        self._cachewise_schedule_iterations = 0
         # Bind GPU block pool to the KV connector. This must happen after
         # kv_cache_manager is constructed so block_pool is available.
         if self.connector is not None and hasattr(
@@ -379,6 +383,17 @@ class Scheduler(SchedulerInterface):
         # For logging.
         scheduled_timestamp = time.monotonic()
__BLANK_CONTEXT__
+        if self.cache_config.enable_cachewise_free_heap:
+            self._cachewise_schedule_iterations += 1
+            if (
+                self._cachewise_schedule_iterations
+                % _CACHEWISE_REBUILD_INTERVAL
+                == 0
+            ):
+                self.kv_cache_manager.block_pool.rebuild_cachewise_heap(
+                    scheduled_timestamp
+                )
+
         self.kv_cache_manager.new_step_starts()
__BLANK_CONTEXT__
         # First, schedule the RUNNING requests.
@@ -1685,7 +1700,7 @@ class Scheduler(SchedulerInterface):
     def _pick_waiting_request_by_prefix_cache(
         self, scheduled_loras: set[int]
     ) -> Request | None:
-        """Among eligible main-queue waiters, pick max local+connector prefix."""
+        """Pick the request requiring the fewest additional KV blocks."""
         if not self.waiting:
             return None
         best: Request | None = None
@@ -1703,7 +1718,7 @@ class Scheduler(SchedulerInterface):
             ):
                 continue
             if req.num_computed_tokens > 0:
-                score = float(req.num_computed_tokens)
+                matched_tokens = req.num_computed_tokens
             else:
                 _, num_local = self.kv_cache_manager.get_computed_blocks(req)
                 if self.connector is not None:
@@ -1712,10 +1727,14 @@ class Scheduler(SchedulerInterface):
                     )
                     if ext_tokens is None:
                         continue
-                    score = float(num_local + ext_tokens)
+                    matched_tokens = num_local + ext_tokens
                 else:
-                    score = float(num_local)
-            key = (-score, req.arrival_time, req.request_id)
+                    matched_tokens = num_local
+            missing_tokens = max(0, req.num_tokens - matched_tokens)
+            additional_blocks = (
+                missing_tokens + self.block_size - 1
+            ) // self.block_size
+            key = (additional_blocks, req.arrival_time, req.request_id)
             if best_key is None or key < best_key:
                 best_key = key
                 best = req
@@ -1871,9 +1890,29 @@ class Scheduler(SchedulerInterface):
         """Returns (num_running_reqs, num_waiting_reqs)."""
         return len(self.running), len(self.waiting) + len(self.skipped_waiting)
__BLANK_CONTEXT__
+    def update_cachewise_session_policy(self, raw_policy: dict[str, Any]) -> bool:
+        """Attach causally observed tool metadata to resident session KV."""
+        from vllm.v1.core.cachewise_policy import parse_cachewise_policy_body
+
+        policy = parse_cachewise_policy_body(raw_policy)
+        if policy is None:
+            raise ValueError("invalid cachewise_policy")
+        self.kv_cache_manager.block_pool.update_cachewise_session_policy(
+            policy, time.monotonic()
+        )
+        return True
+
     def add_request(self, request: Request) -> None:
+        if request.cachewise_policy is not None:
+            request.cachewise_policy = (
+                self.kv_cache_manager.block_pool.update_cachewise_session_policy(
+                    request.cachewise_policy, time.monotonic()
+                )
+            )
         existing = self.requests.get(request.request_id)
         if existing is not None:
+            if request.cachewise_policy is not None:
+                existing.cachewise_policy = request.cachewise_policy
             update = StreamingUpdate.from_request(request)
             if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                 assert existing.streaming_queue is not None, "duplicate request id"
diff --git a/vllm/v1/engine/core.py b/vllm/v1/engine/core.py
index 0fa59579e..ec53b72b5 100644
--- a/vllm/v1/engine/core.py
+++ b/vllm/v1/engine/core.py
@@ -578,6 +578,9 @@ class EngineCore:
             reset_running_requests, reset_connector
         )
__BLANK_CONTEXT__
+    def update_cachewise_session_policy(self, policy: dict[str, Any]) -> bool:
+        return self.scheduler.update_cachewise_session_policy(policy)
+
     def reset_encoder_cache(self) -> None:
         """Reset the encoder cache to invalidate all cached encoder outputs.
__BLANK_CONTEXT__
'''.replace("__BLANK_CONTEXT__", " ").replace("__INDENTED_BLANK_CONTEXT__", "     ")

_PATCHED_FILE_SHA256 = {
    "vllm/entrypoints/serve/cache/api_router.py": "dce01df1fc6b690f5e2132b4de18b97591e53f7e842033caadb2a8b8fbd1950f",
    "vllm/v1/core/block_pool.py": "1cee3a74c550016135ad57e8da88f109b2944f133aa9c47902d961d69c71a73c",
    "vllm/v1/core/cachewise_policy.py": "a66f80fdf750eb7ec2a52dd4a34161ea88b1140caffa918e831822fa4028e809",
    "vllm/v1/core/sched/scheduler.py": "cb2135b5d5e81e9006ed54e0ec6c4e3af8f2bfc895b2d8185e71e3f3f1b46a6a",
    "vllm/v1/engine/core.py": "a477d203291463efc8b0ec9f41d0d399e72771689acb448a695ccdee7fc7b834",
}


def _git(checkout: Path, *args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        input=input_text,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_identity(checkout: Path, commit: str, remote: str) -> None:
    if _git(checkout, "rev-parse", "HEAD") != commit:
        raise RuntimeError(f"{checkout} is not pinned at {commit}")
    if _git(checkout, "remote", "get-url", "origin") != remote:
        raise RuntimeError(f"{checkout} is not the official checkout {remote}")


def verify_checkout(
    checkout: Path,
    commit: str,
    remote: str,
    *,
    allowed_untracked: tuple[str, ...] = (),
) -> None:
    _verify_identity(checkout, commit, remote)
    tracked = _git(checkout, "status", "--porcelain=v1", "--untracked-files=no")
    if tracked:
        raise RuntimeError(f"checkout has tracked changes: {tracked}")
    untracked = _git(
        checkout, "ls-files", "--others", "--exclude-standard", "-z"
    ).split("\0")
    unexpected = sorted(
        path
        for path in untracked
        if path
        and not (
            path.endswith(_MODEL_ARTIFACT_SUFFIXES)
            and any(path.startswith(prefix) for prefix in allowed_untracked)
        )
    )
    if unexpected:
        raise RuntimeError(f"checkout has unexpected untracked files: {unexpected}")


def verify_clean_patch(checkout: Path) -> None:
    verify_checkout(checkout, VLLM_COMMIT, VLLM_REPO)
    _git(checkout, "apply", "--check", "-", input_text=VLLM_PATCH)


def apply_vllm_patch(checkout: Path) -> None:
    verify_clean_patch(checkout)
    _git(checkout, "apply", "-", input_text=VLLM_PATCH)


def verify_applied_patch(checkout: Path) -> None:
    _verify_identity(checkout, VLLM_COMMIT, VLLM_REPO)
    if _git(checkout, "diff", "--cached", "--name-only"):
        raise RuntimeError("patched vLLM checkout has staged changes")
    if _git(checkout, "ls-files", "--others", "--exclude-standard"):
        raise RuntimeError("patched vLLM checkout has untracked files")
    changed = set(_git(checkout, "diff", "--name-only", "HEAD", "--").splitlines())
    if changed != set(_PATCHED_FILE_SHA256):
        raise RuntimeError("patched vLLM checkout changes unexpected files")
    for relative, expected in _PATCHED_FILE_SHA256.items():
        actual = hashlib.sha256((checkout / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"patched vLLM file differs: {relative}")
    _git(checkout, "apply", "--reverse", "--check", "-", input_text=VLLM_PATCH)


def _load_predictor(checkout: Path) -> ModuleType:
    infer_path = checkout / "tool_duration_prediction" / "infer.py"
    if not infer_path.is_file():
        raise FileNotFoundError(f"official predictor not found: {infer_path}")
    sys.path.insert(0, str(infer_path.parent))
    try:
        spec = importlib.util.spec_from_file_location("cachewise_infer", infer_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {infer_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class PolicyBuilder:
    """Load the official predictor once, then score causally observed calls."""

    def __init__(self, predictor_checkout: Path, models_dir: Path) -> None:
        verify_checkout(
            predictor_checkout,
            PREDICTOR_COMMIT,
            PREDICTOR_REPO,
            allowed_untracked=("tool_duration_prediction/models/",),
        )
        self._predictor = _load_predictor(predictor_checkout)
        self._models = self._predictor.load_models(models_dir)

    def build(
        self,
        session_id: str,
        tool_name: str,
        arguments: str,
        elapsed_ms: float = 0.0,
    ) -> dict[str, Any]:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        model = (
            self._models.get(tool_name)
            or self._models.get(tool_name.lower())
            or self._models.get(tool_name.capitalize())
        )
        if model is None:
            raise ValueError(f"no official CacheWise model for tool {tool_name!r}")
        curve, similarity, cluster_id = self._predictor.select_curve(
            model, self._predictor.canonicalize_text(arguments)
        )
        return {
            "version": 1,
            "scope": {"session_id": session_id, "idle_session": False},
            "hints": {
                "duration_curve": curve,
                "elapsed_ms": float(elapsed_ms),
            },
            "predictor_provenance": {
                "commit": PREDICTOR_COMMIT,
                "similarity": float(similarity),
                "cluster_id": int(cluster_id),
            },
        }


def build_policy(
    predictor_checkout: Path,
    models_dir: Path,
    session_id: str,
    tool_name: str,
    arguments: str,
    elapsed_ms: float = 0.0,
) -> dict[str, Any]:
    return PolicyBuilder(predictor_checkout, models_dir).build(
        session_id, tool_name, arguments, elapsed_ms
    )


def idle_policy(session_id: str) -> dict[str, Any]:
    return {
        "version": 1,
        "scope": {"session_id": session_id, "idle_session": True},
        "hints": {"elapsed_ms": 0.0},
    }


def active_policy(session_id: str) -> dict[str, Any]:
    """Identify resident session KV before its next tool call is known."""

    return {
        "version": 1,
        "scope": {"session_id": session_id, "idle_session": False},
        "hints": {"elapsed_ms": 0.0},
    }


def oracle_policy(
    session_id: str, total_duration_ms: float, elapsed_ms: float = 0.0
) -> dict[str, Any]:
    """Build the paper's CacheWise* upper-bound payload, never the real method."""

    return {
        "version": 1,
        "scope": {"session_id": session_id, "idle_session": False},
        "hints": {
            "oracle": {"total_duration_ms": float(total_duration_ms)},
            "elapsed_ms": float(elapsed_ms),
        },
    }


def inference_manifest() -> dict[str, Any]:
    """Separate paper facts from choices required by omitted implementation."""

    return {
        "published": {
            "eviction_score": "conditional expected remaining tool time",
            "waiting_order": "fewest additional KV blocks",
            "rebuild_interval_engine_iterations": N_REBUILD,
            "idle_sessions": "evict before tool-delayed sessions",
            "miss_feedback": None,
        },
        "official_public_sources": {
            "predictor_commit": PREDICTOR_COMMIT,
            "vllm_fork_commit": VLLM_COMMIT,
            "vllm_upstream_base_commit": VLLM_UPSTREAM_BASE,
            "paper_vllm_version": None,
        },
        "inferred_not_tuned": {
            "policy_transport": "selected survival curve in request body",
            "post_response_update": (
                "client attaches source tool metadata after shadow LLM completion"
            ),
            "attachment_clock": "elapsed at attach plus engine monotonic time",
            "rebuild_phase": "first periodic rescore on schedule iteration 3",
            "score_ties": "author-fork LRU then block ID",
            "waiting_ties": "arrival time then request ID",
        },
        "unpublished": [
            "paper evaluation session IDs and exact 80/20 split",
            "paper C100 model artifact and fixed-C100 training command",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    patch = subparsers.add_parser("verify-patch")
    patch.add_argument("checkout", type=Path)
    predictor = subparsers.add_parser("verify-predictor")
    predictor.add_argument("checkout", type=Path)
    apply_patch_parser = subparsers.add_parser("apply-patch")
    apply_patch_parser.add_argument("checkout", type=Path)
    applied = subparsers.add_parser("verify-applied")
    applied.add_argument("checkout", type=Path)

    policy = subparsers.add_parser("policy")
    policy.add_argument("--predictor-checkout", type=Path, required=True)
    policy.add_argument("--models", type=Path, required=True)
    policy.add_argument("--session-id", required=True)
    policy.add_argument("--tool", required=True)
    policy.add_argument("--arguments", default="")
    policy.add_argument("--elapsed-ms", type=float, default=0.0)

    idle = subparsers.add_parser("idle-policy")
    idle.add_argument("--session-id", required=True)
    subparsers.add_parser("manifest")
    oracle = subparsers.add_parser("oracle-policy")
    oracle.add_argument("--session-id", required=True)
    oracle.add_argument("--total-duration-ms", type=float, required=True)
    oracle.add_argument("--elapsed-ms", type=float, default=0.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "verify-patch":
        verify_clean_patch(args.checkout)
        print(f"verified CacheWise patch against {VLLM_COMMIT}")
    elif args.command == "verify-predictor":
        verify_checkout(
            args.checkout,
            PREDICTOR_COMMIT,
            PREDICTOR_REPO,
            allowed_untracked=("tool_duration_prediction/models/",),
        )
        print(f"verified CacheWise predictor at {PREDICTOR_COMMIT}")
    elif args.command == "apply-patch":
        apply_vllm_patch(args.checkout)
        print(f"applied CacheWise patch to {VLLM_COMMIT}")
    elif args.command == "verify-applied":
        verify_applied_patch(args.checkout)
        print(f"verified applied CacheWise patch on {VLLM_COMMIT}")
    elif args.command == "policy":
        print(
            json.dumps(
                build_policy(
                    args.predictor_checkout,
                    args.models,
                    args.session_id,
                    args.tool,
                    args.arguments,
                    args.elapsed_ms,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "idle-policy":
        print(json.dumps(idle_policy(args.session_id), sort_keys=True))
    elif args.command == "manifest":
        print(json.dumps(inference_manifest(), sort_keys=True))
    elif args.command == "oracle-policy":
        print(
            json.dumps(
                oracle_policy(args.session_id, args.total_duration_ms, args.elapsed_ms),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
