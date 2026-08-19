from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from scripts.baselines.agentix_reproduction import (
    EngineServiceLog,
    PLASState,
    VLLM_CORE_PATCHED_SHA256,
    create_app,
    patch_vllm_core,
    simulate_plas_events,
)

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts/baselines/agentix_reproduction.sh"


def test_plas_equation_buckets_and_parallel_inheritance() -> None:
    state = PLASState([1.0, 3.0])
    first = state.assign("program-a", "call-1")
    parallel = state.assign("program-a", "call-2")

    assert first.completed_service_s == parallel.completed_service_s == 0.0
    state.complete(first, 1.0)
    after_one_completion = state.assign("program-a", "call-3")
    assert (
        after_one_completion.completed_service_s,
        after_one_completion.priority,
    ) == (
        1.0,
        1,
    )

    # Algorithm 1 uses max(old, inherited + call time), not a sum of branches.
    state.complete(parallel, 3.0)
    state.complete(after_one_completion, 0.5)
    assert state.assign("program-a", "call-4").priority == 2


def test_deterministic_event_simulation_and_validation() -> None:
    events = [
        ("arrive", "a", "1", None),
        ("arrive", "a", "2", None),
        ("complete", "a", "1", 2.0),
        ("arrive", "a", "3", None),
        ("complete", "a", "2", 1.0),
        ("complete", "a", "3", 1.5),
        ("arrive", "a", "4", None),
    ]
    first = simulate_plas_events(events, [1.0, 3.0])
    second = simulate_plas_events(events, [1.0, 3.0])

    assert first == second
    assert [item.priority for item in first] == [0, 0, 1, 2]
    with pytest.raises(ValueError, match="strictly increasing"):
        PLASState([2.0, 1.0])


def test_exact_core_patch_is_checked_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = """from concurrent.futures import Future
import os
import time
HANDSHAKE_TIMEOUT_MINS = 5

class Core:
    def step(self, scheduler_output):
        future = self.model_executor.execute_model(scheduler_output, non_block=True)

    def batch(self, scheduler_output):
        exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
"""
    core = tmp_path / "core.py"
    core.write_text(source)
    expected = hashlib.sha256(source.encode()).hexdigest()

    assert patch_vllm_core(core, expected_sha256=expected)
    assert not patch_vllm_core(core, expected_sha256=expected)
    patched = core.read_text()
    assert "# AGENTIX_ENGINE_STEP_SERVICE_V1" in patched
    assert patched.count("_agentix_execute_model(self.model_executor") == 2
    compile(patched, str(core), "exec")

    namespace: dict[str, object] = {}
    exec(patched, namespace)
    service_log = tmp_path / "patched-service.jsonl"
    monkeypatch.setenv("AGENTIX_SERVICE_LOG", str(service_log))

    class Executor:
        @staticmethod
        def execute_model(_scheduler_output, *, non_block: bool):
            assert non_block
            future: Future[object] = Future()
            future.set_result(object())
            return future

    namespace["_agentix_execute_model"](
        Executor(), SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1})
    ).result()
    row = json.loads(service_log.read_text())
    assert row["request_ids"] == ["a", "b"]
    assert row["scheduled_tokens"] == {"a": 1, "b": 1}
    assert row["elapsed_s"] >= 0

    untrusted = tmp_path / "untrusted.py"
    untrusted.write_text(source + "# changed\n")
    with pytest.raises(ValueError, match="unexpected core.py sha256"):
        patch_vllm_core(untrusted, expected_sha256=expected)


def _append_step(path: Path, elapsed_s: float, *request_ids: str) -> None:
    row = {
        "schema": "agentix-engine-step-service-v1",
        "elapsed_s": elapsed_s,
        "request_ids": list(request_ids),
        "scheduled_tokens": {request_id: 1 for request_id in request_ids},
    }
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def test_engine_log_charges_shared_batch_and_excludes_waiters(tmp_path: Path) -> None:
    path = tmp_path / "service.jsonl"
    path.touch()
    log = EngineServiceLog(path)

    _append_step(path, 2.0, "running-a", "running-b")
    _append_step(path, 0.5, "running-a")

    assert log.service_s("running-a") == 2.5
    assert log.service_s("running-b") == 2.0
    assert log.service_s("queued-c") is None


def _fake_backend(
    service_log: Path, captured: list[dict[str, object]], *, streaming: bool = False
) -> FastAPI:
    backend = FastAPI()

    @backend.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        raw_request_id = request.headers["x-request-id"]
        captured.append({"payload": payload, "raw_request_id": raw_request_id})
        _append_step(service_log, 1.0, f"chatcmpl-{raw_request_id}")
        if streaming:

            async def chunks():
                yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                yield b"data: [DONE]\n\n"

            return StreamingResponse(chunks(), media_type="text/event-stream")
        return JSONResponse(
            {
                "id": f"fake-{len(captured)}",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
        )

    return backend


@pytest.mark.asyncio
async def test_fake_backend_end_to_end_preserves_workload_and_exposes_actions(
    tmp_path: Path,
) -> None:
    service_log = tmp_path / "service.jsonl"
    service_log.touch()
    captured: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    proxy = create_app(
        backend="http://backend",
        queue_upper_bounds_s=[0.5, 2.0],
        service_log=service_log,
        event_sink=events.append,
        backend_transport=httpx.ASGITransport(app=_fake_backend(service_log, captured)),
    )
    request = {
        "run_instance_id": "stable-task-attempt-1",
        "model": "model",
        "messages": [{"role": "user", "content": "do not alter"}],
        "temperature": 0,
        "max_tokens": 7,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy), base_url="http://proxy"
    ) as client:
        first = await client.post("/v1/chat/completions", json=request)
        second = await client.post("/v1/chat/completions", json=request)

    assert first.status_code == second.status_code == 200
    assert first.headers["x-agentix-priority"] == "0"
    assert second.headers["x-agentix-priority"] == "1"
    assert [item["payload"]["priority"] for item in captured] == [0, 1]
    assert first.headers["x-agentix-engine-request-id"].startswith("chatcmpl-agentix-")
    for item in captured:
        payload = item["payload"]
        assert "run_instance_id" not in payload
        assert payload["messages"] == request["messages"]
        assert payload["max_tokens"] == 7
    assert [event["event"] for event in events] == [
        "assigned",
        "completed",
        "assigned",
        "completed",
    ]
    assert events[1]["service_accounting"] == "inferred-engine-step"


@pytest.mark.asyncio
async def test_streaming_uses_engine_log_without_response_metrics(
    tmp_path: Path,
) -> None:
    service_log = tmp_path / "service.jsonl"
    service_log.touch()
    captured: list[dict[str, object]] = []
    proxy = create_app(
        backend="http://backend",
        queue_upper_bounds_s=[0.5, 2.0],
        service_log=service_log,
        backend_transport=httpx.ASGITransport(
            app=_fake_backend(service_log, captured, streaming=True)
        ),
        event_sink=lambda _event: None,
    )
    body = {
        "program_id": "task-a",
        "model": "model",
        "messages": [{"role": "user", "content": "same"}],
        "stream": True,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy), base_url="http://proxy"
    ) as client:
        first = await client.post("/v1/chat/completions", json=body)
        second = await client.post("/v1/chat/completions", json=body)

    assert first.content == second.content
    assert first.content.endswith(b"data: [DONE]\n\n")
    assert [item["payload"]["priority"] for item in captured] == [0, 1]


@pytest.mark.asyncio
async def test_missing_engine_service_fails_without_wall_fallback(
    tmp_path: Path,
) -> None:
    service_log = tmp_path / "service.jsonl"
    service_log.touch()
    backend = FastAPI()

    @backend.post("/v1/chat/completions")
    async def chat():
        return {"choices": []}

    proxy = create_app(
        backend="http://backend",
        queue_upper_bounds_s=[1.0],
        service_log=service_log,
        backend_transport=httpx.ASGITransport(app=backend),
        event_sink=lambda _event: None,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy), base_url="http://proxy"
    ) as client:
        response = await client.post(
            "/v1/chat/completions", json={"program_id": "p", "messages": []}
        )

    assert response.status_code == 502
    assert "no engine-step service" in response.json()["detail"]


def test_isolated_launcher_contract() -> None:
    subprocess.run(["bash", "-n", LAUNCHER], check=True)
    source = LAUNCHER.read_text()
    help_text = subprocess.run(
        ["bash", LAUNCHER, "--help"], check=True, capture_output=True, text=True
    ).stdout

    assert 'uv pip install --python "$venv/bin/python" "vllm==$VLLM_VERSION"' in source
    assert 'test ! -e "$service_log"' in source
    assert "--scheduling-policy priority --enable-request-id-headers" in source
    assert VLLM_CORE_PATCHED_SHA256 in source
    assert "vLLM==0.11.2" in help_text
    assert "new AGENTIX_SERVICE_LOG" in help_text
    assert "not full Agentix" in help_text
    assert "anti-starvation" in help_text
