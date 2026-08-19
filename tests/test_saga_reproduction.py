from __future__ import annotations

import json
import gzip
import hashlib
import subprocess
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from scripts.baselines.saga_reproduction import (
    AFSState,
    EXECUTABLE_VLLM_SOURCE_SHA256,
    adaptive_ttl_s,
    create_app,
    eviction_score,
    inference_manifest,
    kv_policy,
    memory_pressure,
    reuse_probability,
    verify_vllm_source_tree,
)

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/baselines/saga_reproduction.sh"


def test_published_kv_equations_and_fail_closed_boundary() -> None:
    successors = [
        {"probability": 0.75, "overlap": 0.8},
        {"probability": 0.25, "overlap": 0.4},
    ]
    reuse = reuse_probability(successors)
    assert reuse == pytest.approx(0.7)
    assert eviction_score(
        idle_s=50, max_idle_s=100, size=5, max_size=10, reuse=reuse
    ) == pytest.approx(0.4)
    assert memory_pressure(0.8) == pytest.approx(0.5)
    assert adaptive_ttl_s([1, 2, 3, 4], 0.8) == pytest.approx(3.0)

    decision = kv_policy(
        {
            "successors": successors,
            "latency_history_s": [1, 2, 3, 4],
            "used_kv_fraction": 0.8,
            "idle_s": 50,
            "max_idle_s": 100,
            "size": 5,
            "max_size": 10,
        }
    )
    assert decision["applied_to_vllm"] is False
    assert "decision-only" in decision["boundary"]
    with pytest.raises(ValueError, match="probabilities exceed 1"):
        reuse_probability([{"probability": 0.8, "overlap": 0.1}] * 2)


def test_afs_equation_arrival_rank_and_causal_progress() -> None:
    now = 100.0
    state = AFSState(clock=lambda: now)
    state.register(
        "urgent-session",
        "tenant-a",
        100,
        [{"node_id": "a0", "prefill_s": 30, "decode_s": 10}],
    )
    state.register(
        "light-session",
        "tenant-b",
        100,
        [{"node_id": "b0", "prefill_s": 5, "decode_s": 5}],
    )

    light = state.assign("light-session", "b0")
    urgent = state.assign("urgent-session", "a0")
    assert (urgent.afs, urgent.priority) == pytest.approx((0.4, 0))
    assert (light.afs, light.priority) == pytest.approx((0.1, 1))
    state.complete(urgent)
    state.abort(light)
    assert state.assign("light-session", "b0").priority == 0
    with pytest.raises(ValueError, match="positive profiled work"):
        AFSState().register(
            "bad", "tenant", 10, [{"node_id": "n", "prefill_s": 0, "decode_s": 0}]
        )


def test_expired_task_rejects_itself_without_blocking_valid_tenant() -> None:
    now = 0.0
    state = AFSState(clock=lambda: now)
    node = [{"node_id": "n", "prefill_s": 1, "decode_s": 1}]
    state.register("late", "tenant-late", 1, node)
    state.register("valid", "tenant-valid", 10, node)
    now = 2.0

    with pytest.raises(RuntimeError, match="no positive deadline slack"):
        state.assign("late", "n")
    assert state.assign("valid", "n").priority == 0


def test_priority_source_provenance_rejects_tampering(tmp_path: Path) -> None:
    assert {
        "entrypoints/openai/serving_engine.py",
        "v1/engine/processor.py",
        "v1/engine/__init__.py",
        "v1/engine/async_llm.py",
        "v1/engine/core_client.py",
        "v1/core/sched/request_queue.py",
        "v1/core/sched/scheduler.py",
    } <= EXECUTABLE_VLLM_SOURCE_SHA256.keys()
    source = tmp_path / "request_queue.py"
    source.write_text("smaller_priority_first = True\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    verify_vllm_source_tree(tmp_path, expected={"request_queue.py": digest})

    source.write_text("smaller_priority_first = False\n")
    with pytest.raises(ValueError, match="source differs at request_queue.py"):
        verify_vllm_source_tree(tmp_path, expected={"request_queue.py": digest})


def _backend(
    captured: list[dict[str, object]], *, stream: bool = False, gzip_stream: bool = False
) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        captured.append(await request.json())
        if stream:

            async def chunks():
                body = (
                    b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                yield gzip.compress(body) if gzip_stream else body

            headers = {"content-encoding": "gzip"} if gzip_stream else None
            return StreamingResponse(
                chunks(), media_type="text/event-stream", headers=headers
            )
        return JSONResponse({"choices": [{"message": {"content": "ok"}}]})

    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_real_proxy_interface_preserves_request_and_applies_priority(
    stream: bool,
) -> None:
    captured: list[dict[str, object]] = []
    state = AFSState(clock=lambda: 10.0)
    state.register(
        "session-a",
        "tenant-a",
        100,
        [{"node_id": "n0", "prefill_s": 1, "decode_s": 1}],
    )
    proxy = create_app(
        backend="http://backend",
        state=state,
        backend_transport=httpx.ASGITransport(app=_backend(captured, stream=stream)),
        event_sink=lambda _event: None,
    )
    body = {
        "saga_session_id": "session-a",
        "saga_node_id": "n0",
        "model": "model",
        "messages": [{"role": "user", "content": "unchanged"}],
        "temperature": 0,
        "stream": stream,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy), base_url="http://proxy"
    ) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert response.headers["x-saga-subset"] == "afs-arrival-priority"
    assert response.headers["x-saga-priority"] == "0"
    assert captured == [
        {
            "model": "model",
            "messages": [{"role": "user", "content": "unchanged"}],
            "temperature": 0,
            "stream": stream,
            "priority": 0,
        }
    ]
    assert state.snapshot()["session-a"]["pending_nodes"] == []


@pytest.mark.asyncio
async def test_streaming_preserves_content_encoding() -> None:
    captured: list[dict[str, object]] = []
    state = AFSState(clock=lambda: 10.0)
    state.register(
        "session-a",
        "tenant-a",
        100,
        [{"node_id": "n0", "prefill_s": 1, "decode_s": 1}],
    )
    proxy = create_app(
        backend="http://backend",
        state=state,
        backend_transport=httpx.ASGITransport(
            app=_backend(captured, stream=True, gzip_stream=True)
        ),
        event_sink=lambda _event: None,
    )
    body = {
        "saga_session_id": "session-a",
        "saga_node_id": "n0",
        "model": "model",
        "messages": [{"role": "user", "content": "same"}],
        "stream": True,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy), base_url="http://proxy"
    ) as client:
        async with client.stream("POST", "/v1/chat/completions", json=body) as response:
            raw = b"".join([chunk async for chunk in response.aiter_raw()])

    assert response.headers["content-encoding"] == "gzip"
    assert gzip.decompress(raw).endswith(b"data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_streaming_error_drops_encoding_after_decoding() -> None:
    backend = FastAPI()

    @backend.post("/v1/chat/completions")
    async def error():
        return StreamingResponse(
            iter([gzip.compress(b"bad request")]),
            status_code=400,
            headers={"content-encoding": "gzip"},
        )

    state = AFSState(clock=lambda: 10.0)
    state.register(
        "session-a",
        "tenant-a",
        100,
        [{"node_id": "n0", "prefill_s": 1, "decode_s": 1}],
    )
    proxy = create_app(
        backend="http://backend",
        state=state,
        backend_transport=httpx.ASGITransport(app=backend),
        event_sink=lambda _event: None,
    )
    body = {
        "saga_session_id": "session-a",
        "saga_node_id": "n0",
        "model": "model",
        "messages": [{"role": "user", "content": "same"}],
        "stream": True,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy), base_url="http://proxy"
    ) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert "content-encoding" not in response.headers
    assert response.content == b"bad request"


def test_manifest_and_launcher_cannot_be_mistaken_for_full_saga() -> None:
    manifest = inference_manifest()
    assert manifest["full_saga"] is False
    assert manifest["official_source"]["paper_vllm"]["version"] == "0.6.0"
    assert manifest["inferred_not_tuned"]["executable_vllm"]["version"] == "0.11.2"
    assert "no EMA factor" in manifest["inferred_not_tuned"]["tool_history_update"]
    assert "WA-LRU/TTL enforcement" in manifest["unavailable_private_system"][0]

    subprocess.run(["bash", "-n", SCRIPT], check=True)
    help_text = subprocess.run(
        ["bash", SCRIPT, "--help"], check=True, text=True, capture_output=True
    ).stdout
    assert "This is not full SAGA" in help_text
    assert "serve-afs-subset" in help_text
    assert "serve MODEL" not in help_text
    assert json.dumps(manifest)
