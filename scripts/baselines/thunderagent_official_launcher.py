from __future__ import annotations

import os
import asyncio
import contextvars
import json
import math
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI


_request_record: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("request_record")


def set_resume_wait_limit(router: Any, seconds: float) -> None:
    """Configure the upstream forced-resume deadline; 1800 s is its default."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("THUNDERAGENT_MAX_WAIT_S must be finite and positive")
    router._wait_for_resume = partial(router._wait_for_resume, timeout=seconds)


class RequestAudit:
    """Record complete ASGI request lifetimes and enforce an elapsed-time deadline."""

    def __init__(self, app: Any, events: Any, timeout_s: float) -> None:
        self.app, self.events, self.timeout_s = app, events, timeout_s

    def record(self, event: str, row: dict[str, Any], **fields: Any) -> None:
        self.events.write(json.dumps(dict(row, event=event, timestamp_s=time.time(), **fields)) + "\n")
        self.events.flush()

    async def dispatch(self, request: httpx.Request) -> None:
        if request.url.path != "/v1/chat/completions":
            return
        row = _request_record.get()
        row["backend"] = str(request.url.copy_with(path=""))
        request.headers["x-request-id"] = row["route_id"]
        self.record("dispatch", row)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["path"] != "/v1/chat/completions":
            await self.app(scope, receive, send)
            return
        route_id = uuid.uuid4().hex
        row = dict(route_id=route_id, engine_request_id="chatcmpl-" + route_id)
        token = _request_record.set(row)
        body = bytearray()
        started = False
        complete = False
        status = None
        outcome = "error"

        async def read() -> Any:
            message = await receive()
            if message["type"] == "http.request":
                body.extend(message.get("body", b""))
                if not message.get("more_body", False):
                    payload = json.loads(body)
                    row["job_id"] = payload.get("program_id")
                    self.record("arrival", row)
            return message

        async def write(message: Any) -> None:
            nonlocal started, complete, status
            if message["type"] == "http.response.start":
                started = True
                status = message["status"]
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                complete = True
            await send(message)

        try:
            async with asyncio.timeout(self.timeout_s):
                await self.app(scope, read, write)
            outcome = "complete" if complete and status == 200 else "incomplete"
        except TimeoutError:
            outcome = "timeout"
            if started:
                raise  # Terminate the stream; never turn partial output into success.
            await send(dict(type="http.response.start", status=504, headers=[]))
            await send(dict(type="http.response.body", body=b"Request deadline exceeded"))
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            self.record("finish", row, outcome=outcome, status=status)
            _request_record.reset(token)


async def _disable_backend_keepalive(router: Any, audit: RequestAudit | None = None) -> None:
    """Avoid reusing proxy-to-vLLM connections under high concurrency."""
    old_client = router.client
    router.client = httpx.AsyncClient(
        timeout=float(os.environ.get("SHADOW_LLM_TIMEOUT_S", "900")),
        limits=httpx.Limits(
            max_connections=None,
            max_keepalive_connections=0,
        ),
        event_hooks={"request": [audit.dispatch]} if audit else None,
    )
    await old_client.aclose()


def main() -> None:
    from ThunderAgent.app import register_routes
    from ThunderAgent.config import Config, set_config
    from ThunderAgent.scheduler import MultiBackendRouter

    backends = [
        value.strip()
        for value in os.environ.get(
            "THUNDERAGENT_BACKENDS", "http://127.0.0.1:8000"
        ).split(",")
        if value.strip()
    ]
    profile_dir = os.environ.get("THUNDERAGENT_PROFILE_DIR", "./thunderagent_profiles")
    config = Config(
        backends=backends,
        router_mode="tr",
        backend_type="vllm",
        profile_enabled=True,
        profile_dir=profile_dir,
        metrics_enabled=True,
        metrics_interval=5.0,
        scheduler_interval=5.0,
        acting_token_weight=1.0,
        use_acting_token_decay=True,
    )
    set_config(config)

    class ContinuumFCFSRouter(MultiBackendRouter):
        """Supply the fork's task metadata without changing ThunderAgent scheduling."""

        async def proxy_request(self, backend: Any, payload: dict[str, Any], **callbacks: Any) -> Any:
            payload = dict(payload, job_id=payload["program_id"], this_func_call="", is_last_step=False)
            return await super().proxy_request(backend, payload, **callbacks)

        async def release_program(self, program_id: str) -> bool:
            released = await super().release_program(program_id)
            for backend_url in self.backends:
                response = await self.client.post(backend_url + "/continuum/programs/release",
                                                  json={"job_id": program_id})
                response.raise_for_status()
                assert isinstance(response.json()["released"], bool)
            return released

    router_class = ContinuumFCFSRouter if os.environ.get("THUNDERAGENT_CONTINUUM_FCFS") == "1" else MultiBackendRouter
    router = router_class(
        backends,
        profile_enabled=True,
        scheduling_enabled=True,
        scheduler_interval=5.0,
        backend_type="vllm",
        acting_token_weight=1.0,
        use_acting_token_decay=True,
    )
    if "THUNDERAGENT_MAX_WAIT_S" in os.environ:
        set_resume_wait_limit(router, float(os.environ["THUNDERAGENT_MAX_WAIT_S"]))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await _disable_backend_keepalive(router, audit)
        await router.start()
        try:
            for backend in router.backends.values():
                assert backend.metrics_client.cache_config.total_tokens_capacity > 0
            yield
        finally:
            await router.stop()

    app = FastAPI(title="ThunderAgent official baseline", lifespan=lifespan)
    register_routes(app, router, config)
    events_path = os.environ.get("THUNDERAGENT_ROUTING_EVENTS")
    audit = RequestAudit(app, open(events_path, "a", buffering=1),
                         float(os.environ["SHADOW_LLM_TIMEOUT_S"])) if events_path else None
    uvicorn.run(
        audit or app,
        host=os.environ.get("THUNDERAGENT_HOST", "127.0.0.1"),
        port=int(os.environ.get("THUNDERAGENT_PORT", "9000")),
        timeout_graceful_shutdown=2,
    )


if __name__ == "__main__":
    main()
