"""Request-level least-outstanding routing for independent FCFS engines.

Optional FIFO working-set admission (--admission-tokens N): a request is dispatched only when it is the oldest waiting
request and the estimated prompt tokens of the requests in flight plus its own fit the budget (a request always
dispatches when nothing is in flight). No priority, no residency term: arrival order only, so nothing starves. Prompt
tokens are estimated from the message characters (--chars-per-token, 3.3 measured on the replay steps)."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from collections import deque
from pathlib import Path
import time
from typing import Any, Callable
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import Response, StreamingResponse
import uvicorn


def create_app(
    backends: list[str],
    emit: Callable[[dict[str, Any]], None],
    transport: httpx.AsyncBaseTransport | None = None,
    *,
    task_sticky: bool = False,
    admission_tokens: int = 0,
    chars_per_token: float = 3.3,
) -> FastAPI:
    """Count requests from dispatch until the upstream stream closes.

    A dropped engine connection is retried once on the same engine while no byte has
    reached the client; once streaming has started the request is not idempotent.
    """
    if not backends or len(set(backends)) != len(backends):
        raise ValueError("at least one backend, all distinct, is required")  # one backend: single-engine runs
    clients = [httpx.AsyncClient(base_url=b, timeout=None, transport=transport) for b in backends]
    outstanding = [0] * len(backends)
    next_tie = 0
    previous: dict[str, int] = {}
    visited: dict[str, set[int]] = {}
    waiting: deque[str] = deque()          # FIFO of route ids waiting for admission
    in_flight = {"tokens": 0}
    admission = asyncio.Condition()

    def estimate_tokens(messages: Any) -> int:
        chars = 0
        for m in messages if isinstance(messages, list) else []:
            content = m.get("content") if isinstance(m, dict) else None
            chars += len(content) if isinstance(content, str) else len(json.dumps(content or ""))
            for tc in (m.get("tool_calls") or []) if isinstance(m, dict) else []:
                chars += len(json.dumps(tc))
        return int(chars / chars_per_token)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for client in clients:
            await client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.outstanding = outstanding

    @app.get("/health")
    async def health() -> dict[str, Any]:
        for client in clients:
            response = await client.get("/health", timeout=5)
            response.raise_for_status()
        return {"healthy": True, "outstanding": list(outstanding)}

    @app.get("/v1/models")
    async def models() -> Response:
        response = await clients[0].get("/v1/models", timeout=5)
        return Response(response.content, response.status_code, media_type="application/json")

    @app.post("/continuum/programs/release")
    async def release(request: Request) -> dict[str, bool]:
        payload = await request.json()
        # A job may have visited both engines, so release on both.
        for index in sorted(visited.get(payload["job_id"], set())):
            client = clients[index]
            response = await client.post("/continuum/programs/release", json=payload)
            response.raise_for_status()
            if response.json().get("released") is not True:
                raise HTTPException(502, "backend did not release job")
        previous.pop(payload["job_id"], None)
        visited.pop(payload["job_id"], None)
        return {"released": True}

    @app.post("/v1/chat/completions")
    async def completion(request: Request) -> Response:
        nonlocal next_tie
        payload = await request.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("job_id"), str):
            raise HTTPException(400, "job_id is required")
        if "priority" in payload or payload.get("n", 1) != 1:
            raise HTTPException(400, "FCFS routing requires n=1 and no priority")
        job_id = payload["job_id"]
        route_id = uuid.uuid4().hex
        arrival = time.time()
        estimated = estimate_tokens(payload.get("messages")) if admission_tokens else 0
        emit({"event": "arrival", "timestamp_s": arrival, "route_id": route_id, "job_id": job_id,
              "engine_request_id": "chatcmpl-" + route_id, "estimated_prompt_tokens": estimated})
        if admission_tokens:
            waiting.append(route_id)
            try:
                async with admission:
                    await admission.wait_for(
                        lambda: waiting[0] == route_id
                        and (in_flight["tokens"] == 0 or in_flight["tokens"] + estimated <= admission_tokens))
                    waiting.popleft()
                    in_flight["tokens"] += estimated
                    admission.notify_all()
            except BaseException:
                if route_id in waiting:
                    waiting.remove(route_id)
                async with admission:
                    admission.notify_all()
                raise
        # One event loop and no await between choosing and reserving an engine.
        last = previous.get(job_id)
        if task_sticky and last is not None:
            index = last
        else:
            index = min(range(len(backends)), key=lambda i: (outstanding[i], (i - next_tie) % len(backends)))
            next_tie = (index + 1) % len(backends)
        before = list(outstanding)
        outstanding[index] += 1
        previous[job_id] = index
        visited.setdefault(job_id, set()).add(index)
        common = {"route_id": route_id, "job_id": job_id, "instance": index,
                  "engine_request_id": "chatcmpl-" + route_id}
        emit({"event": "dispatch", "timestamp_s": time.time(), **common,
              "outstanding_before": before, "previous_instance": last,
              "switched_instance": last is not None and last != index,
              "admission_wait_s": round(time.time() - arrival, 4), "estimated_prompt_tokens": estimated,
              "in_flight_tokens_after": in_flight["tokens"]})
        upstream: httpx.Response | None = None
        finished = False
        retries = 0

        async def finish(outcome: str) -> None:
            nonlocal finished
            if finished:
                return
            finished = True
            try:
                if upstream is not None:
                    await upstream.aclose()
            finally:
                outstanding[index] -= 1
                if admission_tokens:
                    async with admission:
                        in_flight["tokens"] -= estimated
                        admission.notify_all()
                emit({"event": "finish", "timestamp_s": time.time(), **common,
                      "outcome": outcome, "outstanding_after": list(outstanding),
                      "retries": retries})

        async def attempt() -> bytes | None:
            """Open upstream; read the body here when it is not streamed onward."""
            nonlocal upstream
            upstream = await clients[index].send(
                clients[index].build_request(
                    "POST", "/v1/chat/completions", json=payload,
                    headers={"x-request-id": route_id, "accept-encoding": "identity"},
                ), stream=True,
            )
            if not payload.get("stream") or upstream.status_code >= 400:
                return await upstream.aread()
            return None

        try:
            try:
                body = await attempt()
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as error:
                # The engine dropped the connection before the client saw a byte, so
                # resending is safe; the same request id keeps telemetry joinable.
                if upstream is not None:
                    await upstream.aclose()
                    upstream = None
                retries = 1
                emit({"event": "retry", "timestamp_s": time.time(), **common,
                      "reason": type(error).__name__})
                body = await attempt()
            headers = {"x-route-id": route_id, "x-serving-instance": str(index)}
            if body is not None:
                await finish("completed" if upstream.status_code < 400 else "http_error")
                return Response(body, upstream.status_code, headers=headers,
                                media_type=upstream.headers.get("content-type"))

            class RoutedStream(StreamingResponse):
                async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
                    outcome = "disconnected"

                    async def chunks():
                        nonlocal outcome
                        async for chunk in upstream.aiter_raw():
                            yield chunk
                        outcome = "completed"

                    self.body_iterator = chunks()
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        # Cleanup is outside StreamingResponse's disconnect cancel scope.
                        await finish(outcome)

            return RoutedStream(iter(()), status_code=upstream.status_code,
                                headers=headers, media_type="text/event-stream")
        except BaseException:
            await finish("error")
            raise

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", required=True)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--task-sticky", action="store_true")
    parser.add_argument("--admission-tokens", type=int, default=0, help="FIFO working-set budget in estimated prompt tokens (0 = off)")
    parser.add_argument("--chars-per-token", type=float, default=3.3)
    args = parser.parse_args()
    with args.events.open("x", buffering=1) as output:
        app = create_app(args.backends, lambda row: output.write(json.dumps(row) + "\n"),
                         task_sticky=args.task_sticky, admission_tokens=args.admission_tokens,
                         chars_per_token=args.chars_per_token)
        # Counters are process-local: exactly one worker is part of this baseline.
        uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
