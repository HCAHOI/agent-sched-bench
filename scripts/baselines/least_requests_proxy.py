"""Request-level least-outstanding routing for independent FCFS engines."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
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
) -> FastAPI:
    """Count requests from dispatch until the upstream stream closes; no retries."""
    if len(backends) != 2 or len(set(backends)) != 2:
        raise ValueError("two distinct backends are required")
    clients = [httpx.AsyncClient(base_url=b, timeout=None, transport=transport) for b in backends]
    outstanding = [0, 0]
    next_tie = 0
    previous: dict[str, int] = {}
    visited: dict[str, set[int]] = {}

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
        # One event loop and no await between choosing and reserving an engine.
        job_id = payload["job_id"]
        last = previous.get(job_id)
        if task_sticky and last is not None:
            index = last
        else:
            index = min(range(2), key=lambda i: (outstanding[i], (i - next_tie) % 2))
            next_tie = 1 - index
        before = list(outstanding)
        outstanding[index] += 1
        previous[job_id] = index
        visited.setdefault(job_id, set()).add(index)
        route_id = uuid.uuid4().hex
        common = {"route_id": route_id, "job_id": job_id, "instance": index,
                  "engine_request_id": "chatcmpl-" + route_id}
        emit({"event": "dispatch", "timestamp_s": time.time(), **common,
              "outstanding_before": before, "previous_instance": last,
              "switched_instance": last is not None and last != index})
        upstream: httpx.Response | None = None
        finished = False

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
                emit({"event": "finish", "timestamp_s": time.time(), **common,
                      "outcome": outcome, "outstanding_after": list(outstanding)})

        try:
            upstream = await clients[index].send(
                clients[index].build_request(
                    "POST", "/v1/chat/completions", json=payload,
                    headers={"x-request-id": route_id, "accept-encoding": "identity"},
                ), stream=True,
            )
            headers = {"x-route-id": route_id, "x-serving-instance": str(index)}
            if not payload.get("stream") or upstream.status_code >= 400:
                body = await upstream.aread()
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
    parser.add_argument("--backends", nargs=2, required=True)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--task-sticky", action="store_true")
    args = parser.parse_args()
    with args.events.open("x", buffering=1) as output:
        app = create_app(args.backends, lambda row: output.write(json.dumps(row) + "\n"),
                         task_sticky=args.task_sticky)
        # Counters are process-local: exactly one worker is part of this baseline.
        uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
