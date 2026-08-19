#!/usr/bin/env python3
"""Paper-derived, executable subset of SAGA (arXiv:2605.00528v2).

Published semantics reproduced here:

* AEG reuse probability and WA-LRU's eviction score (paper Eqs. 1--5).
* Tool-call TTL from the empirical percentile and memory pressure
  (Algorithm 1 and Eq. 6).
* Agent Fair Share (AFS), the sum of remaining profiled GPU work divided by
  deadline slack (Eqs. 8--9).

The authors describe an unpublished 8.5K-line Python / 1.2K-line CUDA vLLM
extension.  Stock vLLM exposes neither per-session KV eviction priority nor a
TTL/prefetch/migration API, so the KV functions below are decision-only: they
never claim to retain or evict real blocks.  The only real scheduling action
is an explicitly narrower AFS arrival-priority proxy.  It maps higher AFS to a
smaller integer rank accepted by stock vLLM 0.11.2's priority scheduler.  This
rank mapping, tie break, and vLLM version are paper-omitted compatibility
choices.  Already-admitted requests are not reprioritized, capacity is not
allocated proportionally, and there is no 100ms epoch, 500ms preemption,
session routing, work stealing, KV migration, or CUDA prefetch.  Consequently
this executable must be reported as ``saga-afs-arrival-subset``, never SAGA.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
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
}

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

    import hashlib

    for relative, wanted in expected.items():
        path = package_root / relative
        if not path.is_file():
            raise ValueError(f"missing stock vLLM source file: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != wanted:
            raise ValueError(f"stock vLLM source differs at {relative}: {actual}")


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
        deadline_after_s = _number(
            deadline_after_s, "deadline_after_s", minimum=1e-300
        )
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
        lambda event: print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True)
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
                raise ValueError("saga_session_id and saga_node_id are required strings")
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
            raise HTTPException(status_code=400, detail="request JSON must be an object")
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
            "afs_to_vllm": "descending AFS rank at request arrival",
            "equal_afs_tie": "tenant ID lexical order",
            "expired_deadline": (
                "reject that session and exclude it from other tenants' rankings"
            ),
            "executable_vllm": {
                "version": EXECUTABLE_VLLM_VERSION,
                "official_tag_commit": EXECUTABLE_VLLM_COMMIT,
                "priority_path_sha256": EXECUTABLE_VLLM_SOURCE_SHA256,
                "reason": "first shared baseline runtime already pinned with stock priority API",
            },
        },
        "decision_only": [
            "AEG reuse probability",
            "WA-LRU eviction score",
            "tool-call TTL",
        ],
        "unavailable_private_system": [
            "WA-LRU/TTL enforcement in vLLM block management",
            "tool-aware speculative CUDA prefetch",
            "pattern-based AEG inference and its history/model",
            "100ms proportional-capacity AFS epochs and 500ms preemption",
            "session-affinity multi-worker routing",
            "randomized work stealing and Llumnix KV migration",
            "Ray/gRPC coordinator and multi-node 64-GPU execution",
        ],
        "executable_action": "AFS arrival-priority on one stock vLLM worker",
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
    policy = commands.add_parser("kv-policy")
    policy.add_argument("input", help="JSON file or - for stdin")
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
    elif args.command == "kv-policy":
        print(json.dumps(kv_policy(_load_json(args.input)), indent=2, sort_keys=True))
    else:
        import uvicorn

        uvicorn.run(
            create_app(backend=args.backend), host=args.host, port=args.port, log_level="info"
        )


if __name__ == "__main__":
    main()
