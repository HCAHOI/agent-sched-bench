#!/usr/bin/env python3
"""Paper-derived, executable subset of SAGA (arXiv:2605.00528v2).

Published semantics reproduced here:

* AEG reuse probability and WA-LRU's eviction score (paper Eqs. 1--5).
* Tool-call TTL from the empirical percentile and memory pressure
  (Algorithm 1 and Eq. 6).
* Agent Fair Share (AFS), the sum of remaining profiled GPU work divided by
  deadline slack (Eqs. 8--9).

The authors describe an unpublished 8.5K-line Python / 1.2K-line CUDA vLLM
extension.  This reproduction therefore exposes two bounded subsets rather
than claiming the private system: an AFS arrival-priority proxy and a
single-GPU KV policy that applies the published WA-LRU score and adaptive TTL
to stock vLLM 0.11.2's resident prefix blocks.  Pattern/latency inputs must come
from a frozen causal profile.  There is no speculative prefetch, periodic AFS,
session routing, work stealing, migration, or multi-worker coordinator.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

PAPER_SOURCE = "https://arxiv.org/abs/2605.00528v2"
PAPER_VLLM_VERSION = "0.6.0"
PAPER_VLLM_COMMIT = "32e7db25365415841ebc7c4215851743fbb1bad1"
EXECUTABLE_VLLM_VERSION = "0.11.2"
EXECUTABLE_VLLM_COMMIT = "275de34170654274616082721348b7edd9741d32"
EXECUTABLE_VLLM_CORE_SHA256 = (
    "7a800832d7e0f0fdd0de27458687f746e19849dac4f852a4aa158f1bad030f0c"
)
EXECUTABLE_VLLM_SOURCE_SHA256 = {
    "v1/core/block_pool.py": (
        "851bbdf1911e7726fb162afbd081fbfff48dc58279449924c99305ae19a9ea79"
    ),
    "v1/core/kv_cache_manager.py": (
        "39ef4ed095251efab344c84d7b81227127ac29f2a684255dd08dc523319142cf"
    ),
    "v1/engine/core.py": EXECUTABLE_VLLM_CORE_SHA256,
    "v1/core/sched/request_queue.py": (
        "93e013bcd52490b72038202d718b65c55abb4dd38b94b5a6e4dda7fba2b03d8b"
    ),
    "v1/core/sched/scheduler.py": (
        "82d9cbbc71e147ba3b0e3623f72931f383e07ae72cb3aa9fc2eb4b4c427350e8"
    ),
    "v1/request.py": "f9c2de51229a988260260db53419163d2d01d36dd0fa10f35d3e3a85a97db4c7",
    "entrypoints/openai/protocol.py": (
        "df0b19ca2a725caecbf4247f9ac9c7ca6b19719370b60121c40aa2a4611d0425"
    ),
    "entrypoints/openai/serving_chat.py": (
        "4ea06e2c1b5b324df16184cad08356ab7b3da0173faf9c9331fa7f85a199cbbd"
    ),
    "entrypoints/openai/serving_completion.py": (
        "fe8b0182e4cbea7652638f34c362dc60c3b068aeac2e05a1f6f127d41a2a6d0a"
    ),
    "entrypoints/openai/serving_engine.py": (
        "283a26317b6da46da41e68c72c42813f2c7f21bf6c7b9e99b234777a022dc51d"
    ),
    "v1/engine/processor.py": (
        "681092d5ba78c872a3744ac8a263b9d0a46cdb344b8319f9c49ccef805210d85"
    ),
    "v1/engine/__init__.py": (
        "ffa86a22a538e3eda1c1118bc6219abb18261fefd8eb2997f8c8149c82b59ea7"
    ),
    "v1/engine/async_llm.py": (
        "f50e373fa58e6d5950c046bdc0f37b32237067b87b096a4cced5f8181bcec8d3"
    ),
    "v1/engine/core_client.py": (
        "48861ee3dbb142f91f41d4e46ab9e9d80e300e4abc2b6b98b6e3f67ac272bc5a"
    ),
    "entrypoints/openai/api_server.py": (
        "63964b55127c7eee24809288901d63b54f540f1733aff0c3d13e32d1e958867c"
    ),
}
PATCHED_VLLM_SOURCE_SHA256 = EXECUTABLE_VLLM_SOURCE_SHA256 | {
    "v1/core/block_pool.py": (
        "945f8502fc3480cfd38698b390b1542045a4948e72f6aeec7121f2e73efc9d02"
    ),
    "v1/core/kv_cache_manager.py": (
        "43a17c1a9b18bc28f9d501182aa486ba92a2943489b4ad83468b2b033e0112e5"
    ),
    "v1/request.py": "0d8578450670a77d944f7cbada120c09202cc16ae8988c809119303802b7750c",
    "v1/core/sched/scheduler.py": (
        "ed2bc2a511a2c3c62842a89720a281009c8bd1043766a9b283f9a8d4eb767415"
    ),
    "v1/engine/core.py": (
        "47e84459ffbf0a08975c1b3e85acce0352cbcbcc5d79e8b7f681b1765cd327f4"
    ),
    "entrypoints/openai/api_server.py": (
        "45681c572060860b78d0b9116ff906cb91abeb834d44a6194be85f93654da389"
    ),
}
VLLM_PATCH = Path(__file__).with_name("saga_vllm.patch")

ALPHA = 0.3
BETA = 0.5
GAMMA = 0.2
TTL_MAX_S = 300.0
PRESSURE_LOW = 0.7
PRESSURE_HIGH = 0.9

EventSink = Callable[[dict[str, Any]], None]
Clock = Callable[[], float]

_HOP_BY_HOP = {
    "connection",
    "content-encoding",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def reuse_probability(successors: Iterable[Mapping[str, object]]) -> float:
    """Compute Eq. 4: sum(P(edge) * estimated prefix overlap)."""

    result = 0.0
    total_probability = 0.0
    for index, successor in enumerate(successors):
        probability = _number(
            successor.get("probability"), f"successors[{index}].probability"
        )
        overlap = _number(successor.get("overlap"), f"successors[{index}].overlap")
        if probability > 1 or overlap > 1:
            raise ValueError("successor probabilities and overlaps must be <= 1")
        total_probability += probability
        result += probability * overlap
    if total_probability > 1 + 1e-12:
        raise ValueError("AEG successor probabilities exceed 1")
    if result > 1 + 1e-12:
        raise ValueError("AEG reuse probability exceeds 1")
    return min(result, 1.0)


def verify_vllm_source_tree(
    package_root: Path,
    *,
    expected: Mapping[str, str] = EXECUTABLE_VLLM_SOURCE_SHA256,
) -> None:
    """Verify the exact official Python path that carries request priority."""

    for relative, wanted in expected.items():
        path = package_root / relative
        if not path.is_file():
            raise ValueError(f"missing stock vLLM source file: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != wanted:
            raise ValueError(f"stock vLLM source differs at {relative}: {actual}")


def apply_vllm_patch(package_root: Path) -> None:
    """Apply the exact SAGA subset patch to the pinned wheel sources."""

    verify_vllm_source_tree(package_root)
    if shutil.which("git") is None:
        raise RuntimeError("git is required to apply the SAGA vLLM patch")
    subprocess.run(
        ["git", "apply", "--check", str(VLLM_PATCH)],
        cwd=package_root.parent,
        check=True,
    )
    subprocess.run(
        ["git", "apply", str(VLLM_PATCH)], cwd=package_root.parent, check=True
    )
    verify_vllm_source_tree(package_root, expected=PATCHED_VLLM_SOURCE_SHA256)


def eviction_score(
    *,
    idle_s: float,
    max_idle_s: float,
    size: float,
    max_size: float,
    reuse: float,
) -> float:
    """Compute WA-LRU Eq. 1; a larger score is evicted first."""

    idle_s = _number(idle_s, "idle_s")
    max_idle_s = _number(max_idle_s, "max_idle_s", minimum=1e-300)
    size = _number(size, "size")
    max_size = _number(max_size, "max_size", minimum=1e-300)
    reuse = _number(reuse, "reuse")
    if idle_s > max_idle_s or size > max_size or reuse > 1:
        raise ValueError("WA-LRU normalized inputs must remain in [0, 1]")
    return ALPHA * idle_s / max_idle_s + BETA * (1 - reuse) + GAMMA * size / max_size


def memory_pressure(
    used_fraction: float,
    *,
    low: float = PRESSURE_LOW,
    high: float = PRESSURE_HIGH,
) -> float:
    """Compute Eq. 6, clamped to its declared m in [0, 1] domain."""

    used_fraction = _number(used_fraction, "used_fraction")
    low = _number(low, "low")
    high = _number(high, "high")
    if used_fraction > 1 or high > 1 or not low < high:
        raise ValueError("memory fractions require 0 <= low < high <= 1")
    return min(1.0, max(0.0, (used_fraction - low) / (high - low)))


def empirical_percentile(values: Iterable[float], percentile: float) -> float:
    """Nearest-rank empirical percentile; the paper omits interpolation."""

    ordered = sorted(_number(value, "latency history value") for value in values)
    percentile = _number(percentile, "percentile", minimum=1e-300)
    if not ordered or percentile > 1:
        raise ValueError("percentile requires non-empty history and 0 < p <= 1")
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def adaptive_ttl_s(
    history_s: Iterable[float],
    used_fraction: float,
    *,
    percentile: float = 0.95,
) -> float:
    """Compute Algorithm 1's empirical-percentile TTL in seconds."""

    base = empirical_percentile(history_s, percentile)
    return min(base * (1 - 0.5 * memory_pressure(used_fraction)), TTL_MAX_S)


class SagaKVIndex:
    """Session ownership and WA-LRU/TTL ordering for vLLM free blocks."""

    def __init__(
        self,
        free_block_ids: Iterable[int],
        *,
        total_blocks: int,
        clock: Clock = time.monotonic,
    ) -> None:
        self._clock = clock
        self._total_blocks = total_blocks
        self._enabled = False
        self._owners: dict[int, set[str]] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._empty_free = dict.fromkeys(free_block_ids)
        self._free_groups: dict[frozenset[str], dict[int, None]] = {}
        self._updates = 0
        self._evicted_blocks = 0
        self._hard_fallback_blocks = 0

    def _session(self, session_id: str) -> dict[str, Any]:
        return self._sessions.setdefault(
            session_id,
            {
                "resident": set(),
                "reuse_probability": 0.0,
                "last_access_s": self._clock(),
                "ttl_until_s": 0.0,
                "finished": False,
            },
        )

    def attach(self, block_id: int, session_id: str | None) -> None:
        if session_id is None:
            return
        self._enabled = True
        owners = self._owners.setdefault(block_id, set())
        if session_id not in owners:
            owners.add(session_id)
            self._session(session_id)["resident"].add(block_id)

    def untrack_free(self, block_id: int) -> None:
        self._empty_free.pop(block_id, None)
        owners = frozenset(self._owners.get(block_id, ()))
        group = self._free_groups.get(owners)
        if group is not None:
            group.pop(block_id, None)
            if not group:
                del self._free_groups[owners]

    def track_free(self, block_id: int, *, cached: bool) -> None:
        self.untrack_free(block_id)
        if not cached:
            self._empty_free[block_id] = None
            return
        owners = frozenset(self._owners.get(block_id, ()))
        self._free_groups.setdefault(owners, {})[block_id] = None

    def touch(self, block_id: int, session_id: str | None, *, was_free: bool) -> None:
        if was_free:
            self.untrack_free(block_id)
        self.attach(block_id, session_id)
        if session_id is not None:
            self._session(session_id)["last_access_s"] = self._clock()

    def evict(self, block_id: int) -> None:
        self.untrack_free(block_id)
        for session_id in self._owners.pop(block_id, ()):
            state = self._sessions.get(session_id)
            if state is not None:
                state["resident"].discard(block_id)
        self._evicted_blocks += 1

    def _group_key(
        self, owners: frozenset[str], now_s: float
    ) -> tuple[bool, float, tuple[str, ...]]:
        if not owners:
            return (True, math.inf, ())
        states = [self._session(session_id) for session_id in owners]
        all_states = [state for state in self._sessions.values() if state["resident"]]
        max_idle_s = max(
            (max(0.0, now_s - state["last_access_s"]) for state in all_states),
            default=1.0,
        )
        max_size = max((len(state["resident"]) for state in all_states), default=1)
        scores = [
            ALPHA * max(0.0, now_s - state["last_access_s"]) / max(max_idle_s, 1e-9)
            + BETA * (1.0 - state["reuse_probability"])
            + GAMMA * len(state["resident"]) / max_size
            for state in states
        ]
        protected = any(
            not state["finished"] and now_s < state["ttl_until_s"] for state in states
        )
        return (not protected, min(scores), tuple(sorted(owners)))

    def take(self, count: int) -> list[int]:
        selected: list[int] = []
        while len(selected) < count:
            hard_fallback = False
            if self._empty_free:
                block_id = next(iter(self._empty_free))
            else:
                now_s = self._clock()
                keys = {
                    owners: self._group_key(owners, now_s)
                    for owners in self._free_groups
                }
                unprotected = [owners for owners, key in keys.items() if key[0]]
                candidates = unprotected or list(keys)
                if not candidates:
                    raise RuntimeError("SAGA free index is out of sync")
                hard_fallback = not unprotected
                owners = max(candidates, key=lambda item: keys[item])
                block_id = next(iter(self._free_groups[owners]))
            self.untrack_free(block_id)
            if hard_fallback:
                self._hard_fallback_blocks += 1
            selected.append(block_id)
        return selected

    def update(self, raw: Mapping[str, object]) -> dict[str, object]:
        if raw.get("version") != 1:
            raise ValueError("invalid SAGA policy version")
        if set(raw) - {
            "version",
            "session_id",
            "reuse_probability",
            "base_ttl_s",
            "finished",
        }:
            raise ValueError("unknown SAGA policy field")
        session_id = raw.get("session_id")
        reuse = raw.get("reuse_probability")
        base_ttl_s = raw.get("base_ttl_s")
        finished = raw.get("finished", False)
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be non-empty")
        reuse = _number(reuse, "reuse_probability")
        base_ttl_s = _number(base_ttl_s, "base_ttl_s")
        if reuse > 1:
            raise ValueError("reuse_probability must be in [0, 1]")
        if not isinstance(finished, bool):
            raise ValueError("finished must be boolean")
        used_fraction = 1.0 - len(self._empty_free) / max(1, self._total_blocks)
        ttl_s = 0.0 if finished else adaptive_ttl_s([base_ttl_s], used_fraction)
        now_s = self._clock()
        self._session(session_id).update(
            reuse_probability=reuse,
            last_access_s=now_s,
            ttl_until_s=now_s + ttl_s,
            finished=finished,
        )
        self._updates += 1
        return {
            "updated": True,
            "session_id": session_id,
            "ttl_s": ttl_s,
            "used_fraction": used_fraction,
        }

    def reset(self, free_block_ids: Iterable[int]) -> None:
        self._enabled = False
        self._owners.clear()
        self._sessions.clear()
        self._empty_free = dict.fromkeys(free_block_ids)
        self._free_groups.clear()
        self._updates = 0
        self._evicted_blocks = 0
        self._hard_fallback_blocks = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def state(self) -> dict[str, object]:
        return {
            "enabled": self._enabled,
            "sessions": len(self._sessions),
            "managed_resident_blocks": sum(
                len(state["resident"]) for state in self._sessions.values()
            ),
            "updates": self._updates,
            "evicted_blocks": self._evicted_blocks,
            "hard_fallback_blocks": self._hard_fallback_blocks,
        }


def kv_policy(payload: Mapping[str, object]) -> dict[str, object]:
    """Return paper KV decisions without pretending stock vLLM applies them."""

    successors = payload.get("successors")
    history = payload.get("latency_history_s")
    if not isinstance(successors, list) or not all(
        isinstance(item, Mapping) for item in successors
    ):
        raise ValueError("successors must be a list of objects")
    if not isinstance(history, list):
        raise ValueError("latency_history_s must be a list")
    used_fraction = _number(payload.get("used_kv_fraction"), "used_kv_fraction")
    reuse = reuse_probability(successors)
    return {
        "schema": "saga-kv-decision-v1",
        "applied_to_vllm": False,
        "reuse_probability": reuse,
        "eviction_score": eviction_score(
            idle_s=_number(payload.get("idle_s"), "idle_s"),
            max_idle_s=_number(payload.get("max_idle_s"), "max_idle_s"),
            size=_number(payload.get("size"), "size"),
            max_size=_number(payload.get("max_size"), "max_size"),
            reuse=reuse,
        ),
        "memory_pressure": memory_pressure(used_fraction),
        "ttl_s": adaptive_ttl_s(
            history,
            used_fraction,
            percentile=_number(payload.get("percentile", 0.95), "percentile"),
        ),
        "boundary": "decision-only; stock vLLM exposes no WA-LRU or TTL API",
    }


class SagaProfile:
    """Frozen causal tool history used by the SAGA KV subset."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        if payload.get("schema") != "saga-causal-profile-v1":
            raise ValueError("invalid SAGA profile schema")
        required_strings = (
            "training_manifest",
            "excluded_evaluation_manifest",
            "tokenizer",
        )
        if any(
            not isinstance(payload.get(field_name), str) or not payload[field_name]
            for field_name in required_strings
        ):
            raise ValueError("SAGA profile lacks frozen provenance")
        if not all(
            Path(str(payload[field_name])).is_absolute()
            for field_name in ("training_manifest", "excluded_evaluation_manifest")
        ):
            raise ValueError("SAGA profile manifest paths must be absolute")
        task_sets: dict[str, set[str]] = {}
        for field_name in ("training_task_ids", "evaluation_task_ids"):
            values = payload.get(field_name)
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) and value for value in values)
            ):
                raise ValueError(f"SAGA profile {field_name} must be non-empty strings")
            task_sets[field_name] = set(values)
        if task_sets["training_task_ids"] & task_sets["evaluation_task_ids"]:
            raise ValueError("SAGA profile training/evaluation tasks overlap")
        trace_sets: dict[str, set[str]] = {}
        for field_name in ("training_trace_paths", "evaluation_trace_paths"):
            values = payload.get(field_name)
            if (
                not isinstance(values, list)
                or not values
                or not all(
                    isinstance(value, str) and Path(value).is_absolute()
                    for value in values
                )
            ):
                raise ValueError(f"SAGA profile {field_name} must be absolute paths")
            trace_sets[field_name] = set(values)
        if trace_sets["training_trace_paths"] & trace_sets["evaluation_trace_paths"]:
            raise ValueError("SAGA profile training/evaluation traces overlap")
        self._evaluation_task_ids = task_sets["evaluation_task_ids"]
        self._evaluation_trace_paths = {
            Path(value).resolve() for value in trace_sets["evaluation_trace_paths"]
        }
        self._evaluation_manifest = Path(
            str(payload["excluded_evaluation_manifest"])
        ).resolve()
        tools = payload.get("tools")
        if not isinstance(tools, Mapping):
            raise ValueError("SAGA profile tools must be an object")
        self._tools: dict[str, dict[str, object]] = {}
        for tool_name, raw in tools.items():
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError("SAGA profile tool names must be non-empty")
            if not isinstance(raw, Mapping):
                raise ValueError(f"SAGA profile for {tool_name!r} must be an object")
            p95_latency_s = raw.get("p95_latency_s")
            successors = raw.get("successors")
            p95_latency_s = _number(p95_latency_s, f"{tool_name}.p95_latency_s")
            if not isinstance(successors, list):
                raise ValueError(f"{tool_name!r} successors must be a list")
            parsed_successors: list[dict[str, float]] = []
            probability_sum = 0.0
            for index, successor in enumerate(successors):
                if not isinstance(successor, Mapping):
                    raise ValueError(f"{tool_name!r} successor {index} is invalid")
                probability = _number(
                    successor.get("probability"),
                    f"{tool_name}.successors[{index}].probability",
                )
                observation_tokens = _number(
                    successor.get("expected_observation_tokens"),
                    f"{tool_name}.successors[{index}].expected_observation_tokens",
                )
                probability_sum += probability
                parsed_successors.append(
                    {
                        "probability": probability,
                        "expected_observation_tokens": observation_tokens,
                    }
                )
            if probability_sum > 1 + 1e-12:
                raise ValueError(f"{tool_name!r} successor probabilities exceed 1")
            self._tools[tool_name] = {
                "p95_latency_s": p95_latency_s,
                "successors": parsed_successors,
            }

    @classmethod
    def load(cls, path: Path) -> "SagaProfile":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("SAGA profile must be a JSON object")
        return cls(payload)

    def validate_evaluation_manifest(self, path: Path) -> None:
        if path.resolve() != self._evaluation_manifest:
            raise ValueError(
                "SAGA profile was frozen for a different evaluation manifest"
            )
        entries = _manifest_traces(path)
        if {task_id for task_id, _ in entries} != self._evaluation_task_ids or {
            trace for _, trace in entries
        } != self._evaluation_trace_paths:
            raise ValueError("SAGA evaluation manifest differs from the frozen profile")

    def build(
        self, session_id: str, tool_name: str, current_context_tokens: int
    ) -> tuple[dict[str, object], bool]:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if (
            not isinstance(current_context_tokens, int)
            or isinstance(current_context_tokens, bool)
            or current_context_tokens <= 0
        ):
            raise ValueError("current_context_tokens must be a positive integer")
        profile = self._tools.get(tool_name)
        if profile is None:
            return finished_saga_policy(session_id, finished=False), False
        successors = profile["successors"]
        assert isinstance(successors, list)
        reuse = sum(
            successor["probability"]
            * current_context_tokens
            / (current_context_tokens + successor["expected_observation_tokens"])
            for successor in successors
        )
        return (
            {
                "version": 1,
                "session_id": session_id,
                "reuse_probability": reuse,
                "base_ttl_s": profile["p95_latency_s"],
                "finished": False,
            },
            True,
        )


def finished_saga_policy(
    session_id: str, *, finished: bool = True
) -> dict[str, object]:
    return {
        "version": 1,
        "session_id": session_id,
        "reuse_probability": 0.0,
        "base_ttl_s": 0.0,
        "finished": finished,
    }


def _trace_task_id(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "trace_metadata":
                task_id = row.get("task_instance_id", row.get("instance_id"))
                if isinstance(task_id, str) and task_id:
                    return task_id
            if row.get("type") == "action":
                task_id = row.get("task_instance_id", row.get("instance_id"))
                if isinstance(task_id, str) and task_id:
                    return task_id
    raise ValueError(f"trace lacks a canonical task ID: {path}")


def _manifest_traces(path: Path) -> list[tuple[str, Path]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("traces"), list):
        raise ValueError(f"invalid trace manifest: {path}")
    entries: list[tuple[str, Path]] = []
    for index, entry in enumerate(payload["traces"]):
        if not isinstance(entry, Mapping):
            raise ValueError(f"manifest trace {index} must be an object")
        trace = entry.get("trace")
        if not isinstance(trace, str) or not trace:
            raise ValueError(f"manifest trace {index} lacks trace")
        trace_path = Path(trace)
        if not trace_path.is_absolute():
            trace_path = path.parent / trace_path
        trace_path = trace_path.resolve()
        if not trace_path.is_file():
            raise FileNotFoundError(f"missing manifest trace: {trace_path}")
        entries.append((_trace_task_id(trace_path), trace_path))
    return entries


def build_causal_profile(
    manifest: Path,
    *,
    tokenizer: Any,
    tokenizer_name: str,
    exclude_manifest: Path,
) -> dict[str, object]:
    """Build the paper's per-tool history from explicitly allowed traces."""

    entries = _manifest_traces(manifest)
    if len({label for label, _ in entries}) != len(entries):
        raise ValueError("training manifest task IDs must be unique")
    evaluation_entries = _manifest_traces(exclude_manifest)
    excluded = {task_id for task_id, _ in evaluation_entries}
    overlap = sorted(task_id for task_id, _ in entries if task_id in excluded)
    if overlap:
        raise ValueError(f"training/evaluation task overlap: {overlap}")
    training_paths = {trace for _, trace in entries}
    evaluation_paths = {trace for _, trace in evaluation_entries}
    if training_paths & evaluation_paths:
        raise ValueError("training/evaluation trace paths overlap")
    durations: dict[str, list[float]] = defaultdict(list)
    transitions: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    occurrences: dict[str, int] = defaultdict(int)
    for label, trace in entries:
        if not trace.is_file():
            raise FileNotFoundError(f"missing training trace for {label}: {trace}")
        actions: list[tuple[str, float, int]] = []
        with trace.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("type") != "action" or row.get("action_type") != "tool_exec":
                    continue
                data = row.get("data")
                if not isinstance(data, Mapping):
                    raise ValueError(f"malformed tool action in {trace}")
                tool_name = data.get("tool_name")
                if not isinstance(tool_name, str) or not tool_name:
                    raise ValueError(f"tool action lacks a name in {trace}")
                duration_s = _number(data.get("duration_ms"), "duration_ms") / 1000.0
                result = data.get("tool_result")
                if not isinstance(result, str):
                    result = json.dumps(result, ensure_ascii=False, sort_keys=True)
                result_tokens = len(tokenizer.encode(result, add_special_tokens=False))
                actions.append((tool_name, duration_s, result_tokens))
                durations[tool_name].append(duration_s)
                occurrences[tool_name] += 1
        for current, following in zip(actions, actions[1:]):
            transitions[current[0]][following[0]].append(current[2])
    tools: dict[str, object] = {}
    for tool_name in sorted(durations):
        successor_rows = []
        for successor, observations in sorted(transitions[tool_name].items()):
            successor_rows.append(
                {
                    "tool": successor,
                    "probability": len(observations) / occurrences[tool_name],
                    "expected_observation_tokens": sum(observations)
                    / len(observations),
                    "samples": len(observations),
                }
            )
        tools[tool_name] = {
            "p95_latency_s": empirical_percentile(durations[tool_name], 0.95),
            "latency_samples": len(durations[tool_name]),
            "successors": successor_rows,
        }
    return {
        "schema": "saga-causal-profile-v1",
        "training_manifest": str(manifest.resolve()),
        "excluded_evaluation_manifest": str(exclude_manifest.resolve()),
        "training_task_ids": [task_id for task_id, _ in entries],
        "evaluation_task_ids": sorted(excluded),
        "training_trace_paths": [str(trace) for _, trace in entries],
        "evaluation_trace_paths": sorted(str(trace) for trace in evaluation_paths),
        "tokenizer": tokenizer_name,
        "tools": tools,
    }


@dataclass(frozen=True)
class Assignment:
    session_id: str
    tenant_id: str
    node_id: str
    afs: float
    priority: int


@dataclass
class _Task:
    tenant_id: str
    deadline: float
    pending_work_s: dict[str, float]
    active: set[str] = field(default_factory=set)


class AFSState:
    """Causal task registry implementing SAGA Eqs. 8--9 at arrival time."""

    def __init__(self, *, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        self._tasks: dict[str, _Task] = {}
        self._lock = threading.Lock()

    def register(
        self,
        session_id: str,
        tenant_id: str,
        deadline_after_s: float,
        nodes: Iterable[Mapping[str, object]],
    ) -> None:
        if (
            not isinstance(session_id, str)
            or not session_id
            or not isinstance(tenant_id, str)
            or not tenant_id
        ):
            raise ValueError("session_id and tenant_id must be non-empty")
        deadline_after_s = _number(deadline_after_s, "deadline_after_s", minimum=1e-300)
        pending: dict[str, float] = {}
        for index, node in enumerate(nodes):
            node_id = node.get("node_id")
            if not isinstance(node_id, str) or not node_id:
                raise ValueError(f"nodes[{index}].node_id must be non-empty")
            if node_id in pending:
                raise ValueError(f"duplicate node_id {node_id!r}")
            work_s = _number(
                node.get("prefill_s"), f"nodes[{index}].prefill_s"
            ) + _number(node.get("decode_s"), f"nodes[{index}].decode_s")
            if work_s <= 0:
                raise ValueError(f"nodes[{index}] must have positive profiled work")
            pending[node_id] = work_s
        if not pending:
            raise ValueError("nodes must be non-empty")
        with self._lock:
            if session_id in self._tasks:
                raise RuntimeError(f"session {session_id!r} is already registered")
            self._tasks[session_id] = _Task(
                tenant_id=tenant_id,
                deadline=self._clock() + deadline_after_s,
                pending_work_s=pending,
            )

    def _scores(self, now: float) -> dict[str, float]:
        scores: dict[str, float] = {}
        for task in self._tasks.values():
            remaining = sum(task.pending_work_s.values())
            if remaining == 0:
                continue
            slack = task.deadline - now
            if slack <= 0:
                continue
            scores[task.tenant_id] = scores.get(task.tenant_id, 0.0) + remaining / slack
        return scores

    def assign(self, session_id: str, node_id: str) -> Assignment:
        if not session_id or not node_id:
            raise ValueError("saga_session_id and saga_node_id must be non-empty")
        with self._lock:
            task = self._tasks.get(session_id)
            if task is None:
                raise RuntimeError(f"unknown session {session_id!r}")
            if node_id not in task.pending_work_s:
                raise RuntimeError(f"node {node_id!r} is not pending")
            if node_id in task.active:
                raise RuntimeError(f"node {node_id!r} is already active")
            now = self._clock()
            if task.deadline <= now:
                raise RuntimeError(
                    f"session {session_id!r} has no positive deadline slack"
                )
            scores = self._scores(now)
            ordered = sorted(scores, key=lambda tenant: (-scores[tenant], tenant))
            task.active.add(node_id)
            return Assignment(
                session_id=session_id,
                tenant_id=task.tenant_id,
                node_id=node_id,
                afs=scores[task.tenant_id],
                priority=ordered.index(task.tenant_id),
            )

    def complete(self, assignment: Assignment) -> None:
        with self._lock:
            task = self._tasks[assignment.session_id]
            if assignment.node_id not in task.active:
                raise RuntimeError("completion does not match an active node")
            task.active.remove(assignment.node_id)
            del task.pending_work_s[assignment.node_id]

    def abort(self, assignment: Assignment) -> None:
        with self._lock:
            task = self._tasks.get(assignment.session_id)
            if task is not None:
                task.active.discard(assignment.node_id)

    def release(self, session_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(session_id)
            if task is None:
                return False
            if task.active:
                raise RuntimeError(f"session {session_id!r} still has active requests")
            del self._tasks[session_id]
            return True

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                session_id: {
                    "tenant_id": task.tenant_id,
                    "pending_nodes": sorted(task.pending_work_s),
                    "active_nodes": sorted(task.active),
                }
                for session_id, task in sorted(self._tasks.items())
            }


def _headers(
    headers: Mapping[str, str],
    assignment: Assignment,
    *,
    raw_stream: bool = False,
) -> dict[str, str]:
    result = {
        name: value
        for name, value in headers.items()
        if name.lower() not in _HOP_BY_HOP
    }
    result.update(
        {
            "x-saga-subset": "afs-arrival-priority",
            "x-saga-session-id": assignment.session_id,
            "x-saga-tenant-id": assignment.tenant_id,
            "x-saga-afs": f"{assignment.afs:.9f}",
            "x-saga-priority": str(assignment.priority),
        }
    )
    if raw_stream and "content-encoding" in headers:
        result["content-encoding"] = headers["content-encoding"]
    return result


def create_app(
    *,
    backend: str,
    state: AFSState | None = None,
    event_sink: EventSink | None = None,
    backend_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Create the OpenAI-compatible AFS arrival-priority subset proxy."""

    afs = state or AFSState()
    emit = event_sink or (
        lambda event: print(
            json.dumps(event, sort_keys=True), file=sys.stderr, flush=True
        )
    )
    client = httpx.AsyncClient(
        base_url=backend.rstrip("/"), timeout=None, transport=backend_transport
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="SAGA AFS arrival-priority subset", lifespan=lifespan)
    app.state.afs = afs

    async def proxy(path: str, request: Request) -> Response:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request JSON must be an object")
            payload = dict(payload)
            session_id = payload.pop("saga_session_id", None)
            node_id = payload.pop("saga_node_id", None)
            if not isinstance(session_id, str) or not isinstance(node_id, str):
                raise ValueError(
                    "saga_session_id and saga_node_id are required strings"
                )
            if "priority" in payload:
                raise ValueError("client priority would invalidate the SAGA subset")
            assignment = afs.assign(session_id, node_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

        payload["priority"] = assignment.priority
        emit({"event": "assigned", **asdict(assignment), "path": path})
        forward_headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _HOP_BY_HOP
        }

        if payload.get("stream") is True:
            try:
                upstream = await client.send(
                    client.build_request(
                        "POST", path, json=payload, headers=forward_headers
                    ),
                    stream=True,
                )
            except Exception as error:
                afs.abort(assignment)
                raise HTTPException(status_code=502, detail=str(error)) from error
            if upstream.status_code >= 400:
                body = await upstream.aread()
                await upstream.aclose()
                afs.abort(assignment)
                return Response(
                    body,
                    status_code=upstream.status_code,
                    headers=_headers(upstream.headers, assignment),
                    media_type=upstream.headers.get("content-type"),
                )

            async def stream() -> Any:
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                    afs.complete(assignment)
                    emit({"event": "completed", **asdict(assignment)})
                except (asyncio.CancelledError, GeneratorExit):
                    afs.abort(assignment)
                    raise
                except Exception:
                    afs.abort(assignment)
                    raise
                finally:
                    await upstream.aclose()

            return StreamingResponse(
                stream(),
                status_code=upstream.status_code,
                headers=_headers(upstream.headers, assignment, raw_stream=True),
                media_type=upstream.headers.get("content-type", "text/event-stream"),
            )

        try:
            upstream = await client.post(path, json=payload, headers=forward_headers)
        except Exception as error:
            afs.abort(assignment)
            raise HTTPException(status_code=502, detail=str(error)) from error
        if upstream.status_code >= 400:
            afs.abort(assignment)
            return Response(
                upstream.content,
                status_code=upstream.status_code,
                headers=_headers(upstream.headers, assignment),
                media_type=upstream.headers.get("content-type"),
            )
        try:
            afs.complete(assignment)
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        emit({"event": "completed", **asdict(assignment)})
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            headers=_headers(upstream.headers, assignment),
            media_type=upstream.headers.get("content-type"),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await proxy("/v1/chat/completions", request)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await proxy("/v1/completions", request)

    @app.post("/tasks/register")
    async def register(request: Request) -> dict[str, object]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400, detail="request JSON must be an object"
            )
        nodes = payload.get("nodes")
        try:
            if not isinstance(nodes, list) or not all(
                isinstance(node, Mapping) for node in nodes
            ):
                raise ValueError("nodes must be a list of objects")
            afs.register(
                payload.get("session_id"),
                payload.get("tenant_id"),
                payload.get("deadline_after_s"),
                nodes,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"registered": True, "subset": "saga-afs-arrival-subset"}

    @app.post("/tasks/release")
    async def release(request: Request) -> dict[str, object]:
        payload = await request.json()
        session_id = payload.get("session_id") if isinstance(payload, dict) else None
        if not isinstance(session_id, str) or not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        try:
            released = afs.release(session_id)
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"released": released}

    @app.get("/tasks/state")
    async def task_state() -> dict[str, object]:
        return {"tasks": afs.snapshot(), "subset": "saga-afs-arrival-subset"}

    @app.post("/policy/kv")
    async def policy(request: Request) -> dict[str, object]:
        payload = await request.json()
        try:
            if not isinstance(payload, Mapping):
                raise ValueError("request JSON must be an object")
            return kv_policy(payload)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def inference_manifest() -> dict[str, object]:
    return {
        "schema": "saga-reproduction-manifest-v1",
        "classification": "paper-derived executable subset; not full SAGA",
        "official_source": {
            "paper": PAPER_SOURCE,
            "version": "arXiv:2605.00528v2 (2026-06-19)",
            "artifact_status": (
                "no implementation repository or artifact URL is identified by the "
                "official arXiv source or manuscript"
            ),
            "paper_vllm": {
                "version": PAPER_VLLM_VERSION,
                "official_tag_commit": PAPER_VLLM_COMMIT,
                "priority_request_api": False,
                "compatibility_observation": (
                    "the manuscript calls v0.6.0 a V1 engine, while the official "
                    "tag has neither the vllm/v1 package nor an OpenAI priority field"
                ),
            },
        },
        "published_semantics": {
            "wa_lru_weights": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA},
            "ttl_percentile": 0.95,
            "ttl_max_s": TTL_MAX_S,
            "memory_pressure": {"low": PRESSURE_LOW, "high": PRESSURE_HIGH},
            "afs": "sum(remaining profiled GPU-seconds / deadline slack)",
        },
        "inferred_not_tuned": {
            "ttl_percentile_interpolation": "nearest-rank empirical percentile",
            "tool_history_update": (
                "caller supplies settled causal samples; the paper gives no EMA factor"
            ),
            "memory_pressure_above_high": "clamp to the paper-declared [0,1] domain",
            "kv_hard_pressure": (
                "when every free cached block is TTL-protected and allocation must "
                "progress, evict the highest WA-LRU candidate"
            ),
            "shared_prefix_blocks": (
                "retain for the owner with the lowest eviction score; evict only "
                "when all owners are unprotected or hard pressure requires it"
            ),
            "profile_transport": (
                "frozen per-tool history supplies latency and successor observation "
                "lengths; missing tools receive no TTL protection"
            ),
            "afs_to_vllm": "descending AFS rank at request arrival",
            "equal_afs_tie": "tenant ID lexical order",
            "expired_deadline": (
                "reject that session and exclude it from other tenants' rankings"
            ),
            "executable_vllm": {
                "version": EXECUTABLE_VLLM_VERSION,
                "official_tag_commit": EXECUTABLE_VLLM_COMMIT,
                "priority_path_sha256": EXECUTABLE_VLLM_SOURCE_SHA256,
                "patched_path_sha256": PATCHED_VLLM_SOURCE_SHA256,
                "reason": "first shared baseline runtime already pinned with stock priority API",
            },
        },
        "decision_only": ["AEG/profile construction before a frozen profile exists"],
        "unavailable_private_system": [
            "tool-aware speculative CUDA prefetch",
            "authors' exact pattern-based AEG inference and EMA history model",
            "100ms proportional-capacity AFS epochs and 500ms preemption",
            "session-affinity multi-worker routing",
            "randomized work stealing and Llumnix KV migration",
            "Ray/gRPC coordinator and multi-node 64-GPU execution",
        ],
        "executable_action": [
            "AFS arrival-priority on one vLLM worker",
            "single-GPU WA-LRU eviction and tool-call TTL over resident prefix blocks",
        ],
        "full_saga": False,
    }


def _load_json(path: str) -> Mapping[str, object]:
    raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("input JSON must be an object")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("manifest")
    verify = commands.add_parser("verify-vllm-tree")
    verify.add_argument("package_root")
    apply_patch_parser = commands.add_parser("apply-vllm-patch")
    apply_patch_parser.add_argument("package_root")
    verify_patched = commands.add_parser("verify-patched-vllm")
    verify_patched.add_argument("package_root")
    policy = commands.add_parser("kv-policy")
    policy.add_argument("input", help="JSON file or - for stdin")
    profile = commands.add_parser("build-profile")
    profile.add_argument("--manifest", type=Path, required=True)
    profile.add_argument("--exclude-manifest", type=Path, required=True)
    profile.add_argument("--tokenizer", required=True)
    profile.add_argument("--output", type=Path, required=True)
    serve = commands.add_parser("serve-afs-subset")
    serve.add_argument("--backend", required=True, help="stock vLLM API root")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=9001)
    args = parser.parse_args(argv)

    if args.command == "manifest":
        print(json.dumps(inference_manifest(), indent=2, sort_keys=True))
    elif args.command == "verify-vllm-tree":
        verify_vllm_source_tree(Path(args.package_root))
        print(f"verified stock vLLM priority path at {args.package_root}")
    elif args.command == "apply-vllm-patch":
        apply_vllm_patch(Path(args.package_root))
        print(f"applied SAGA KV subset patch at {args.package_root}")
    elif args.command == "verify-patched-vllm":
        verify_vllm_source_tree(
            Path(args.package_root), expected=PATCHED_VLLM_SOURCE_SHA256
        )
        print(f"verified SAGA KV subset patch at {args.package_root}")
    elif args.command == "kv-policy":
        print(json.dumps(kv_policy(_load_json(args.input)), indent=2, sort_keys=True))
    elif args.command == "build-profile":
        from transformers import AutoTokenizer

        payload = build_causal_profile(
            args.manifest,
            exclude_manifest=args.exclude_manifest,
            tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),
            tokenizer_name=args.tokenizer,
        )
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        import uvicorn

        uvicorn.run(
            create_app(backend=args.backend),
            host=args.host,
            port=args.port,
            log_level="info",
        )


if __name__ == "__main__":
    main()
