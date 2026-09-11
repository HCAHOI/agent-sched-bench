import asyncio
import json

import httpx
import pytest

from scripts.baselines.least_requests_proxy import create_app


@pytest.mark.asyncio
async def test_routing_stream_lifetime_errors_and_release():
    gates = [asyncio.Event(), asyncio.Event()]
    started = [asyncio.Event(), asyncio.Event()]
    received = []
    released = []
    events = []

    class Stream(httpx.AsyncByteStream):
        def __init__(self, index):
            self.index = index

        async def __aiter__(self):
            started[self.index].set()
            yield b'data: {"choices":[]}\n\n'
            await gates[self.index].wait()
            yield b'data: [DONE]\n\n'

    async def backend(request):
        index = int(request.url.host[-1])
        payload = json.loads(request.content)
        if request.url.path.endswith("/release"):
            released.append((index, payload["job_id"]))
            return httpx.Response(200, json={"released": True})
        received.append((index, payload))
        if payload.get("test_error"):
            return httpx.Response(500, json={"error": "test"})
        if payload.get("test_raise"):
            raise httpx.ConnectError("test", request=request)
        return httpx.Response(200, stream=Stream(index), headers={"content-type": "text/event-stream"})

    app = create_app(["http://engine0", "http://engine1"], events.append, httpx.MockTransport(backend))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy") as client:
            payload = {"job_id": "same-job", "stream": True, "messages": [{"role": "user", "content": "original"}]}
            first = asyncio.create_task(client.post("/v1/chat/completions", json=payload))
            await asyncio.wait_for(started[0].wait(), 2)
            second = asyncio.create_task(client.post("/v1/chat/completions", json=payload))
            await asyncio.wait_for(started[1].wait(), 2)
            assert app.state.outstanding == [1, 1]  # Headers/first chunks do not release capacity.
            gates[1].set()
            assert (await second).content.endswith(b"data: [DONE]\n\n")
            assert app.state.outstanding == [1, 0]
            response = await client.post("/v1/chat/completions", json=payload)
            assert response.headers["x-serving-instance"] == "1"
            assert received[-1] == (1, payload)  # Payload is not rewritten.
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert app.state.outstanding == [0, 0]
            for flag in ("test_error", "test_raise"):
                try:
                    response = await client.post("/v1/chat/completions", json={**payload, flag: True})
                    assert response.status_code == 500
                except httpx.ConnectError:
                    assert flag == "test_raise"
                assert app.state.outstanding == [0, 0]
            response = await client.post("/continuum/programs/release", json={"job_id": "same-job"})
            assert response.json() == {"released": True}
            assert released == [(0, "same-job"), (1, "same-job")]
            assert [e["switched_instance"] for e in events if e["event"] == "dispatch"][:3] == [False, True, False]
            assert len([e for e in events if e["event"] == "dispatch"]) == len([e for e in events if e["event"] == "finish"])


@pytest.mark.asyncio
async def test_dropped_engine_connection_retries_once_before_the_first_byte():
    attempts = []
    events = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[]}\n\n'
            raise httpx.ReadError("dropped mid-stream")

    async def backend(request):
        payload = json.loads(request.content)
        attempts.append((payload["job_id"], request.headers["x-request-id"]))
        job = payload["job_id"]
        seen = sum(1 for item in attempts if item[0] == job)
        if job == "recovers" and seen == 1:
            raise httpx.ReadError("engine dropped the connection")
        if job == "broken":
            raise httpx.RemoteProtocolError("engine dropped the connection")
        if job == "streaming":
            return httpx.Response(200, stream=Stream(),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"ok": True})

    app = create_app(["http://engine0", "http://engine1"], events.append,
                     httpx.MockTransport(backend), task_sticky=True)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy") as client:
            response = await client.post("/v1/chat/completions", json={"job_id": "recovers"})
            assert response.status_code == 200
            # Same engine, same request id: one logical request, two dispatches upstream.
            assert attempts == [("recovers", attempts[0][1])] * 2
            assert len([e for e in events if e["event"] == "dispatch"]) == 1
            retry = next(e for e in events if e["event"] == "retry")
            finish = next(e for e in events if e["event"] == "finish")
            assert retry["engine_request_id"] == finish["engine_request_id"]
            assert retry["reason"] == "ReadError"
            assert finish["retries"] == 1

            with pytest.raises(httpx.RemoteProtocolError):
                await client.post("/v1/chat/completions", json={"job_id": "broken"})
            assert [item[0] for item in attempts].count("broken") == 2
            assert app.state.outstanding == [0, 0]

            with pytest.raises(httpx.ReadError):
                await client.post("/v1/chat/completions",
                                  json={"job_id": "streaming", "stream": True})
            # A drop after the first forwarded byte is not retried.
            assert [item[0] for item in attempts].count("streaming") == 1
            assert not [e for e in events if e["event"] == "retry" and e["job_id"] == "streaming"]
            assert app.state.outstanding == [0, 0]


@pytest.mark.asyncio
async def test_task_sticky_survives_idle_and_imbalance_until_release():
    gate = asyncio.Event()
    started = asyncio.Event()
    events = []
    released = []

    async def backend(request):
        payload = json.loads(request.content)
        if request.url.path.endswith("/release"):
            released.append(request.url.host)
            return httpx.Response(200, json={"released": True})
        if payload.get("hold"):
            started.set()
            await gate.wait()
        return httpx.Response(200, json={"ok": True})

    app = create_app(["http://engine0", "http://engine1"], events.append,
                     httpx.MockTransport(backend), task_sticky=True)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy") as client:
            async def send(job, **extra):
                return await client.post("/v1/chat/completions", json={"job_id": job, **extra})

            first = asyncio.create_task(send("A", hold=True))
            await asyncio.wait_for(started.wait(), 2)
            assert (await send("B")).headers["x-serving-instance"] == "1"
            assert app.state.outstanding == [1, 0]
            # Continuation stays with its busy engine despite the idle alternative.
            assert (await send("A")).headers["x-serving-instance"] == "0"
            gate.set()
            await first
            assert app.state.outstanding == [0, 0]
            assert (await send("B")).headers["x-serving-instance"] == "1"
            assert (await send("A")).headers["x-serving-instance"] == "0"
            assert not any(e["switched_instance"] for e in events if e["event"] == "dispatch")
            await client.post("/continuum/programs/release", json={"job_id": "A"})
            assert released == ["engine0"]
            gate.clear()
            started.clear()
            other = asyncio.create_task(send("C", hold=True))
            await asyncio.wait_for(started.wait(), 2)
            assert app.state.outstanding == [1, 0]
            # Released IDs can be assigned anew; replacement IDs are independent.
            assert (await send("A")).headers["x-serving-instance"] == "1"
            gate.set()
            await other
