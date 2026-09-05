#!/usr/bin/env python3
"""Executable, paper-derived subset of Agentix/Autellix.

Faithful subset (Agentix, NSDI'26, Algorithm 1 and Sec. 4.2):

* PLAS assigns call ``c_j`` the completed service of its program,
  ``p(c_j) = sum(t_k)`` for prior completed calls with the same program ID.
  Smaller values have higher priority.
* The continuous service value is mapped to K queues whose ranges are
  ``[Q_i.low, Q_i.high)``.  The paper does not publish K or the boundaries,
  so ``--queue-upper-bounds`` is mandatory rather than guessed here.
* Calls arriving in parallel inherit the same completed-service snapshot.
  Algorithm 1 then updates the program scalar with
  ``max(old_service, inherited_service + call_model_time)``.  This is equal
  to a sum for sequential PLAS calls and is the paper's ATLAS-style critical
  path update when calls overlap.

This is not the complete Agentix engine.  Stock vLLM can consume a static
integer request priority, but cannot change it after admission.  Therefore
the proxy cannot reproduce Algorithm 1's per-quantum demotion, multi-step
scheduling, or contiguous KV-swap kernel.  Anti-starvation is also omitted:
the paper gives ``(W_p + W_c) / (T_p + T_c) >= beta`` and resets ``W_c,T_c``,
but publishes neither beta nor queue/quanta constants, and stock vLLM cannot
change an in-flight request's priority through its API.
ATLAS and multi-engine routing are outside this single-engine PLAS subset.

Stock vLLM 0.11.2 is installed and minimally patched by the companion shell
script.  For every successful engine step, the patch writes the scheduled
request IDs and the elapsed ``execute_model`` future to ``AGENTIX_SERVICE_LOG``.
The proxy charges each participating request the full shared step duration;
requests merely waiting in the scheduler are absent and receive no service.
This is an inferred engine-step attained-service counter, not the authors'
private exact hook, but unlike wall latency or TTFT it excludes scheduler
queueing and is causal.  There is deliberately no wall/TTFT fallback.

The proxy accepts ``run_instance_id`` (preferred) or ``program_id`` in the
OpenAI request body, removes only that routing metadata, adds the PLAS
``priority``, and otherwise forwards messages and generation parameters
unchanged.  Start stock vLLM with ``--scheduling-policy priority``.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import hashlib
import json
import math
import os
import sys
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

EventSink = Callable[[dict[str, Any]], None]

VLLM_VERSION = "0.11.2"
VLLM_CORE_ORIGINAL_SHA256 = (
    "7a800832d7e0f0fdd0de27458687f746e19849dac4f852a4aa158f1bad030f0c"
)
VLLM_CORE_PATCHED_SHA256 = (
    "0e7a8efc15fd81c2d82b5cbc853e2eaf09838be77fda65ddbe3922b48b6fb69d"
)
_PATCH_MARKER = "# AGENTIX_ENGINE_STEP_SERVICE_V1"

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

_CORE_IMPORT_ANCHOR = "import os\n"
_CORE_CONSTANT_ANCHOR = "HANDSHAKE_TIMEOUT_MINS = 5\n"
_CORE_STEP_CALL = (
    "future = self.model_executor.execute_model(scheduler_output, non_block=True)"
)
_CORE_BATCH_CALL = """exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )"""
_CORE_HELPER = r'''

# AGENTIX_ENGINE_STEP_SERVICE_V1
def _agentix_execute_model(model_executor, scheduler_output):
    """Record causal attained service for requests in one real engine step."""
    service_log = os.environ.get("AGENTIX_SERVICE_LOG")
    if not service_log or not scheduler_output.num_scheduled_tokens:
        return model_executor.execute_model(scheduler_output, non_block=True)

    started = time.perf_counter()
    raw_future = model_executor.execute_model(scheduler_output, non_block=True)
    logged_future = Future()

    def record(completed_future):
        try:
            result = completed_future.result()
            record = {
                "schema": "agentix-engine-step-service-v1",
                "elapsed_s": time.perf_counter() - started,
                "request_ids": list(scheduler_output.num_scheduled_tokens),
                "scheduled_tokens": dict(scheduler_output.num_scheduled_tokens),
            }
            line = (json.dumps(record, separators=(",", ":")) + "\n").encode()
            fd = os.open(service_log, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                if os.write(fd, line) != len(line):
                    raise RuntimeError("short AGENTIX_SERVICE_LOG write")
            finally:
                os.close(fd)
        except BaseException as error:
            logged_future.set_exception(error)
        else:
            logged_future.set_result(result)

    raw_future.add_done_callback(record)
    return logged_future
'''


def patch_vllm_core(
    path: Path, *, expected_sha256: str = VLLM_CORE_ORIGINAL_SHA256
) -> bool:
    """Patch an exact official vLLM 0.11.2 core.py; return False if patched."""

    source = path.read_text(encoding="utf-8")
    if _PATCH_MARKER in source:
        return False
    actual_sha256 = hashlib.sha256(source.encode()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"refusing to patch unexpected core.py sha256 {actual_sha256}")
    for anchor, expected_count in (
        (_CORE_IMPORT_ANCHOR, 1),
        (_CORE_CONSTANT_ANCHOR, 1),
        (_CORE_STEP_CALL, 1),
        (_CORE_BATCH_CALL, 1),
    ):
        if source.count(anchor) != expected_count:
            raise ValueError(f"unexpected vLLM core.py anchor: {anchor!r}")
    source = source.replace(_CORE_IMPORT_ANCHOR, "import json\nimport os\n", 1)
    source = source.replace(
        _CORE_CONSTANT_ANCHOR, _CORE_CONSTANT_ANCHOR + _CORE_HELPER, 1
    )
    source = source.replace(
        _CORE_STEP_CALL,
        "future = _agentix_execute_model(self.model_executor, scheduler_output)",
        1,
    )
    source = source.replace(
        _CORE_BATCH_CALL,
        "exec_future = _agentix_execute_model(self.model_executor, scheduler_output)",
        1,
    )
    temporary = path.with_name(path.name + ".agentix.tmp")
    temporary.write_text(source, encoding="utf-8")
    os.chmod(temporary, path.stat().st_mode)
    os.replace(temporary, path)
    return True


class EngineServiceLog:
    """Incrementally sum real engine-step time per scheduled request."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._offset = 0
        self._totals: dict[str, float] = {}
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def service_s(self, request_id: str) -> float | None:
        with self._lock:
            with self.path.open("r", encoding="utf-8") as handle:
                handle.seek(self._offset)
                while True:
                    line_offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        handle.seek(line_offset)
                        break
                    row = json.loads(line)
                    self._add(row)
                self._offset = handle.tell()
            return self._totals.get(request_id) if request_id in self._seen else None

    def _add(self, row: Any) -> None:
        if not isinstance(row, dict) or row.get("schema") != (
            "agentix-engine-step-service-v1"
        ):
            raise ValueError("invalid AGENTIX_SERVICE_LOG schema")
        elapsed = row.get("elapsed_s")
        request_ids = row.get("request_ids")
        scheduled_tokens = row.get("scheduled_tokens")
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or elapsed < 0
            or not isinstance(request_ids, list)
            or not request_ids
            or any(not isinstance(value, str) or not value for value in request_ids)
            or not isinstance(scheduled_tokens, dict)
            or set(request_ids) != set(scheduled_tokens)
        ):
            raise ValueError("invalid AGENTIX_SERVICE_LOG row")
        for request_id in request_ids:
            self._seen.add(request_id)
            self._totals[request_id] = self._totals.get(request_id, 0.0) + float(
                elapsed
            )


@dataclass(frozen=True)
class Assignment:
    program_id: str
    request_id: str
    completed_service_s: float
    queue_index: int
    priority: int


@dataclass
class _Program:
    completed_service_s: float = 0.0
    completed_calls: int = 0
    active: dict[str, Assignment] = field(default_factory=dict)
    seen_request_ids: set[str] = field(default_factory=set)
    accounting_error: str | None = None


class PLASState:
    """Thread-safe implementation of PLAS's completed-service equation."""

    def __init__(self, queue_upper_bounds_s: Iterable[float]) -> None:
        bounds = tuple(float(value) for value in queue_upper_bounds_s)
        if not bounds or any(
            not math.isfinite(value) or value <= 0 for value in bounds
        ):
            raise ValueError("queue upper bounds must be positive finite seconds")
        if any(left >= right for left, right in zip(bounds, bounds[1:])):
            raise ValueError("queue upper bounds must be strictly increasing")
        self.queue_upper_bounds_s = bounds
        self._programs: dict[str, _Program] = {}
        self._lock = threading.Lock()

    def assign(self, program_id: str, request_id: str) -> Assignment:
        if not program_id or not request_id:
            raise ValueError("program_id and request_id must be non-empty")
        with self._lock:
            program = self._programs.setdefault(program_id, _Program())
            if program.accounting_error:
                raise RuntimeError(
                    f"program {program_id!r} has invalid service accounting: "
                    f"{program.accounting_error}"
                )
            if request_id in program.seen_request_ids:
                raise ValueError(f"duplicate request_id {request_id!r}")
            queue_index = bisect.bisect_right(
                self.queue_upper_bounds_s, program.completed_service_s
            )
            assignment = Assignment(
                program_id=program_id,
                request_id=request_id,
                completed_service_s=program.completed_service_s,
                queue_index=queue_index,
                priority=queue_index,
            )
            program.seen_request_ids.add(request_id)
            program.active[request_id] = assignment
            return assignment

    def complete(self, assignment: Assignment, model_time_s: float) -> float:
        if not math.isfinite(model_time_s) or model_time_s < 0:
            raise ValueError("model_time_s must be a non-negative finite value")
        with self._lock:
            program = self._programs[assignment.program_id]
            if program.active.pop(assignment.request_id, None) != assignment:
                raise ValueError(f"request {assignment.request_id!r} is not active")
            # Algorithm 1, line 4. Sequential PLAS reduces to cumulative sum;
            # overlapping calls retain only the longest completed branch.
            program.completed_service_s = max(
                program.completed_service_s,
                assignment.completed_service_s + model_time_s,
            )
            program.completed_calls += 1
            return program.completed_service_s

    def abort(self, assignment: Assignment, error: str | None = None) -> None:
        with self._lock:
            program = self._programs[assignment.program_id]
            program.active.pop(assignment.request_id, None)
            if error:
                program.accounting_error = error

    def release(self, program_id: str) -> bool:
        with self._lock:
            program = self._programs.get(program_id)
            if program is None:
                return False
            if program.active:
                raise RuntimeError(f"program {program_id!r} still has active calls")
            del self._programs[program_id]
            return True

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                program_id: {
                    "completed_service_s": program.completed_service_s,
                    "completed_calls": program.completed_calls,
                    "active_request_ids": sorted(program.active),
                    "accounting_error": program.accounting_error,
                }
                for program_id, program in sorted(self._programs.items())
            }


def simulate_plas_events(
    events: Iterable[tuple[str, str, str, float | None]],
    queue_upper_bounds_s: Iterable[float],
) -> list[Assignment]:
    """Deterministically replay arrivals/completions with known model time."""

    state = PLASState(queue_upper_bounds_s)
    assignments: dict[tuple[str, str], Assignment] = {}
    arrivals: list[Assignment] = []
    for kind, program_id, request_id, model_time_s in events:
        key = (program_id, request_id)
        if kind == "arrive":
            if model_time_s is not None:
                raise ValueError("arrival events cannot carry model time")
            assignment = state.assign(program_id, request_id)
            assignments[key] = assignment
            arrivals.append(assignment)
        elif kind == "complete":
            if model_time_s is None:
                raise ValueError("completion events require model time")
            state.complete(assignments[key], model_time_s)
        else:
            raise ValueError(f"unknown event kind {kind!r}")
    return arrivals


def _program_id(payload: dict[str, Any]) -> str:
    run_id = payload.pop("run_instance_id", None)
    program_id = payload.pop("program_id", None)
    if run_id is not None and program_id is not None and run_id != program_id:
        raise ValueError("run_instance_id and program_id disagree")
    value = run_id if run_id is not None else program_id
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a stable run_instance_id or program_id is required")
    return value


def _request_id(payload: Mapping[str, Any], headers: Mapping[str, str]) -> str:
    body_value = payload.get("request_id")
    header_value = headers.get("x-request-id")
    if body_value is not None and not isinstance(body_value, str):
        raise ValueError("request_id must be a string")
    if body_value and header_value and body_value != header_value:
        raise ValueError("request_id and x-request-id disagree")
    return body_value or header_value or uuid.uuid4().hex


def _forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in _HOP_BY_HOP
    }


def _response_headers(
    headers: Mapping[str, str], assignment: Assignment, engine_request_id: str
) -> dict[str, str]:
    result = _forward_headers(headers)
    result.update(
        {
            "x-agentix-program-id": assignment.program_id,
            "x-agentix-request-id": assignment.request_id,
            "x-agentix-engine-request-id": engine_request_id,
            "x-agentix-priority": str(assignment.priority),
            "x-agentix-queue-index": str(assignment.queue_index),
            "x-agentix-completed-service-s": f"{assignment.completed_service_s:.9f}",
            "x-agentix-subset": "plas-arrival-priority",
        }
    )
    return result


def create_app(
    *,
    backend: str,
    queue_upper_bounds_s: Iterable[float],
    service_log: Path,
    event_sink: EventSink | None = None,
    backend_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Create the OpenAI-compatible PLAS arrival-priority proxy."""

    state = PLASState(queue_upper_bounds_s)
    service = EngineServiceLog(service_log)
    emit = event_sink or (lambda event: print(json.dumps(event), flush=True))
    client = httpx.AsyncClient(
        base_url=backend.rstrip("/"), timeout=None, transport=backend_transport
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="Agentix PLAS arrival-priority reproduction", lifespan=lifespan)
    app.state.plas = state

    async def proxy(path: str, request: Request) -> Response:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request JSON must be an object")
            payload = dict(payload)
            program_id = _program_id(payload)
            request_id = _request_id(payload, request.headers)
            if "priority" in payload:
                raise ValueError("client priority would invalidate the PLAS baseline")
            if payload.get("n", 1) != 1:
                raise ValueError("PLAS service attribution requires n=1")
            if path == "/v1/completions" and isinstance(payload.get("prompt"), list):
                raise ValueError("PLAS service attribution requires one prompt")
            assignment = state.assign(program_id, request_id)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (json.JSONDecodeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        payload["priority"] = assignment.priority
        headers = _forward_headers(request.headers)
        backend_header_id = f"agentix-{uuid.uuid4().hex}"
        engine_request_id = (
            "chatcmpl-" if path == "/v1/chat/completions" else "cmpl-"
        ) + backend_header_id
        headers["x-request-id"] = backend_header_id
        emit(
            {
                "event": "assigned",
                **asdict(assignment),
                "path": path,
                "engine_request_id": engine_request_id,
            }
        )

        def charge() -> float:
            model_time_s = service.service_s(engine_request_id)
            if model_time_s is None:
                raise ValueError(
                    f"no engine-step service for request {engine_request_id!r}"
                )
            total = state.complete(assignment, model_time_s)
            emit(
                {
                    "event": "completed",
                    **asdict(assignment),
                    "engine_request_id": engine_request_id,
                    "charged_service_s": model_time_s,
                    "program_completed_service_s": total,
                    "service_accounting": "inferred-engine-step",
                }
            )
            return model_time_s

        if payload.get("stream") is True:
            try:
                upstream = await client.send(
                    client.build_request("POST", path, json=payload, headers=headers),
                    stream=True,
                )
            except Exception as error:
                state.abort(assignment)
                raise HTTPException(status_code=502, detail=str(error)) from error
            if upstream.status_code >= 400:
                body = await upstream.aread()
                await upstream.aclose()
                state.abort(assignment)
                return Response(
                    body,
                    status_code=upstream.status_code,
                    headers=_response_headers(
                        upstream.headers, assignment, engine_request_id
                    ),
                    media_type=upstream.headers.get("content-type"),
                )

            async def stream() -> Any:
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                    charge()
                except (asyncio.CancelledError, GeneratorExit):
                    state.abort(assignment)
                    raise
                except Exception as error:
                    state.abort(assignment, str(error))
                    emit(
                        {
                            "event": "accounting_error",
                            **asdict(assignment),
                            "error": str(error),
                        }
                    )
                    raise
                finally:
                    await upstream.aclose()

            return StreamingResponse(
                stream(),
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "text/event-stream"),
                headers=_response_headers(
                    upstream.headers, assignment, engine_request_id
                ),
            )

        try:
            upstream = await client.post(path, json=payload, headers=headers)
            if upstream.status_code >= 400:
                state.abort(assignment)
                return Response(
                    upstream.content,
                    status_code=upstream.status_code,
                    headers=_response_headers(
                        upstream.headers, assignment, engine_request_id
                    ),
                    media_type=upstream.headers.get("content-type"),
                )
            model_time_s = charge()
            response_headers = _response_headers(
                upstream.headers, assignment, engine_request_id
            )
            response_headers["x-agentix-charged-service-s"] = f"{model_time_s:.9f}"
            return Response(
                upstream.content,
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=upstream.headers.get("content-type"),
            )
        except Exception as error:
            state.abort(assignment, str(error))
            emit(
                {"event": "accounting_error", **asdict(assignment), "error": str(error)}
            )
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await proxy("/v1/chat/completions", request)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await proxy("/v1/completions", request)

    @app.post("/programs/release")
    async def release(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400, detail="request JSON must be an object"
            )
        try:
            program_id = _program_id(dict(payload))
            released = state.release(program_id)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        emit({"event": "released", "program_id": program_id, "released": released})
        return {"released": released}

    @app.get("/programs/state")
    async def programs_state() -> dict[str, Any]:
        return {"programs": state.snapshot()}

    return app


def _queue_bounds(raw: str) -> tuple[float, ...]:
    try:
        return tuple(float(value) for value in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated seconds") from error


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, help="stock vLLM API root")
    parser.add_argument(
        "--queue-upper-bounds",
        required=True,
        type=_queue_bounds,
        help=(
            "comma-separated Q_i upper bounds in seconds; required because "
            "the paper does not publish K or its boundaries"
        ),
    )
    parser.add_argument(
        "--service-log",
        required=True,
        type=Path,
        help="read-only AGENTIX_SERVICE_LOG written by the patched vLLM engine",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--event-log", type=Path)
    args = parser.parse_args(argv)

    sink: EventSink | None = None
    if args.event_log:
        args.event_log.parent.mkdir(parents=True, exist_ok=True)

        def sink(event: dict[str, Any]) -> None:
            with args.event_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

    app = create_app(
        backend=args.backend,
        queue_upper_bounds_s=args.queue_upper_bounds,
        service_log=args.service_log,
        event_sink=sink,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main(sys.argv[1:])
