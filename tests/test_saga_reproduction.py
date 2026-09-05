from __future__ import annotations

import asyncio
import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from scripts.baselines.saga_reproduction import (
    AFSState,
    EXECUTABLE_VLLM_SOURCE_SHA256,
    PATCHED_VLLM_SOURCE_SHA256,
    SagaKVIndex,
    SagaProfile,
    apply_vllm_patch,
    adaptive_ttl_s,
    build_causal_profile,
    create_app,
    eviction_score,
    finished_saga_policy,
    inference_manifest,
    kv_policy,
    memory_pressure,
    reuse_probability,
    verify_vllm_source_tree,
)
from trace_collect.openclaw_host_runtime import (
    OpenClawReplayProvider,
    ShadowGenerationConfig,
)

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/baselines/saga_reproduction.sh"


def _profile_payload(tools: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "saga-causal-profile-v1",
        "training_manifest": "/train.yaml",
        "excluded_evaluation_manifest": "/evaluation.yaml",
        "training_task_ids": ["train-task"],
        "evaluation_task_ids": ["eval-task"],
        "training_trace_paths": ["/train/trace.jsonl"],
        "evaluation_trace_paths": ["/eval/trace.jsonl"],
        "tokenizer": "tokenizer",
        "tools": tools,
    }


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


def test_real_kv_index_applies_ttl_wa_lru_and_shared_ownership() -> None:
    now = 100.0
    index = SagaKVIndex(range(1, 7), total_blocks=6, clock=lambda: now)
    assert index.take(6) == [1, 2, 3, 4, 5, 6]
    index.attach(1, "finished")
    index.attach(2, "protected")
    index.attach(3, "protected")
    index.attach(3, "shared")
    for block_id in (1, 2, 3):
        index.track_free(block_id, cached=True)
    index.update(
        {
            "version": 1,
            "session_id": "finished",
            "reuse_probability": 0.0,
            "base_ttl_s": 0.0,
            "finished": True,
        }
    )
    index.update(
        {
            "version": 1,
            "session_id": "protected",
            "reuse_probability": 0.9,
            "base_ttl_s": 30.0,
            "finished": False,
        }
    )
    index.update(
        {
            "version": 1,
            "session_id": "shared",
            "reuse_probability": 0.0,
            "base_ttl_s": 0.0,
            "finished": True,
        }
    )

    assert index.take(1) == [1]
    index.evict(1)
    for block_id in (4, 5, 6):
        index.attach(block_id, "protected")
        index.track_free(block_id, cached=True)
    assert index.state()["hard_fallback_blocks"] == 0
    assert index.take(1)[0] in {2, 3, 4, 5, 6}
    assert index.state()["hard_fallback_blocks"] == 1

    clock = [0.0]
    recency = SagaKVIndex(range(1, 4), total_blocks=3, clock=lambda: clock[0])
    assert recency.take(3) == [1, 2, 3]
    recency.attach(1, "a-old")
    recency.attach(2, "z-new")
    recency.attach(3, "a-old")
    recency.attach(3, "z-new")
    recency.update(finished_saga_policy("a-old", finished=False))
    clock[0] = 90
    recency.update(finished_saga_policy("z-new", finished=False))
    for block_id in (1, 2, 3):
        recency.track_free(block_id, cached=True)
    clock[0] = 100
    recency.touch(3, "z-new", was_free=True)
    assert recency.take(1) == [1]


def test_frozen_saga_profile_builds_causal_policy_and_miss_fallback() -> None:
    profile = SagaProfile(
        _profile_payload(
            {
                "exec": {
                    "p95_latency_s": 4,
                    "successors": [
                        {
                            "probability": 0.75,
                            "expected_observation_tokens": 100,
                        },
                        {
                            "probability": 0.25,
                            "expected_observation_tokens": 300,
                        },
                    ],
                }
            }
        )
    )
    policy, hit = profile.build("session", "exec", 900)
    assert hit is True
    assert policy["reuse_probability"] == pytest.approx(0.75 * 0.9 + 0.25 * 0.75)
    assert policy["base_ttl_s"] == 4
    miss, hit = profile.build("session", "unknown", 900)
    assert hit is False
    assert miss["base_ttl_s"] == 0
    with pytest.raises(ValueError, match="frozen provenance"):
        SagaProfile({"schema": "saga-causal-profile-v1", "tools": {}})


def test_build_causal_profile_uses_only_declared_disjoint_tasks(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "task-a" / "attempt_1" / "trace.jsonl"
    trace.parent.mkdir(parents=True)
    rows = [
        {
            "type": "trace_metadata",
            "task_instance_id": "task-a",
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "data": {
                "tool_name": "exec",
                "duration_ms": 2000,
                "tool_result": "two tokens",
            },
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "data": {
                "tool_name": "read_file",
                "duration_ms": 100,
                "tool_result": "done",
            },
        },
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "train.yaml"
    manifest.write_text(f"version: 1\ntraces:\n  - label: task-a\n    trace: {trace}\n")
    evaluation = tmp_path / "evaluation.yaml"
    evaluation_trace = tmp_path / "task-b" / "attempt_1" / "trace.jsonl"
    evaluation_trace.parent.mkdir(parents=True)
    evaluation_trace.write_text(
        json.dumps({"type": "trace_metadata", "task_instance_id": "task-b"}) + "\n"
    )
    evaluation.write_text(
        f"version: 1\ntraces:\n  - label: task-a\n    trace: {evaluation_trace}\n"
    )

    class Tokenizer:
        def encode(self, value: str, *, add_special_tokens: bool) -> list[str]:
            assert add_special_tokens is False
            return value.split()

    profile = build_causal_profile(
        manifest,
        exclude_manifest=evaluation,
        tokenizer=Tokenizer(),
        tokenizer_name="tokenizer",
    )
    assert profile["training_task_ids"] == ["task-a"]
    assert profile["evaluation_task_ids"] == ["task-b"]
    assert profile["tools"]["exec"] == {
        "p95_latency_s": 2.0,
        "latency_samples": 1,
        "successors": [
            {
                "tool": "read_file",
                "probability": 1.0,
                "expected_observation_tokens": 2.0,
                "samples": 1,
            }
        ],
    }
    frozen = SagaProfile(profile)
    frozen.validate_evaluation_manifest(evaluation)
    evaluation.write_text(
        f"version: 1\ntraces:\n  - label: task-b\n    trace: {trace}\n"
    )
    with pytest.raises(ValueError, match="differs from the frozen profile"):
        frozen.validate_evaluation_manifest(evaluation)
    with pytest.raises(ValueError, match="training/evaluation task overlap"):
        build_causal_profile(
            manifest,
            exclude_manifest=manifest,
            tokenizer=Tokenizer(),
            tokenizer_name="tokenizer",
        )


def test_saga_patch_applies_to_exact_stock_vllm(tmp_path: Path) -> None:
    spec = importlib.util.find_spec("vllm")
    assert spec is not None and spec.origin is not None
    source_root = Path(spec.origin).parent
    package_root = tmp_path / "site-packages" / "vllm"
    for relative in EXECUTABLE_VLLM_SOURCE_SHA256:
        destination = package_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)

    apply_vllm_patch(package_root)
    verify_vllm_source_tree(package_root, expected=PATCHED_VLLM_SOURCE_SHA256)
    with pytest.raises(ValueError, match="source differs"):
        apply_vllm_patch(package_root)
    protocol = package_root / "entrypoints/openai/protocol.py"
    protocol.write_text(protocol.read_text() + "\n# polluted\n")
    with pytest.raises(ValueError, match="source differs"):
        verify_vllm_source_tree(package_root, expected=PATCHED_VLLM_SOURCE_SHA256)


class _SagaStreamResponse:
    def raise_for_status(self) -> None:
        pass

    async def aiter_lines(self):
        yield 'data: {"id":"shadow","prompt_token_ids":[1],"choices":[{"delta":{"token_ids":[2]},"finish_reason":"length"}]}'
        yield 'data: {"id":"shadow","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
        yield "data: [DONE]"


class _SagaClient:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def stream(self, _method: str, url: str, *, json: dict[str, object]):
        self.events.append((url, json))

        class _Context:
            async def __aenter__(self) -> _SagaStreamResponse:
                return _SagaStreamResponse()

            async def __aexit__(self, *_args: object) -> bool:
                return False

        return _Context()

    async def post(self, url: str, *, json: dict[str, object]):
        self.events.append((url, json))

        class _Response:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, bool]:
                return {"updated": True}

        return _Response()

    async def aclose(self) -> None:
        pass


def test_replay_tags_request_then_updates_saga_from_actual_shadow_tokens(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            _profile_payload(
                {
                    "exec": {
                        "p95_latency_s": 2,
                        "successors": [
                            {
                                "probability": 1,
                                "expected_observation_tokens": 1,
                            }
                        ],
                    }
                }
            )
        )
    )
    action = {
        "action_id": "llm-0",
        "_source_action_index": 0,
        "data": {
            "messages_in": [{"role": "user", "content": "work"}],
            "completion_tokens": 1,
            "raw_response": {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call-0",
                                    "function": {
                                        "name": "exec",
                                        "arguments": '{"command":"pytest -q"}',
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
        },
    }
    config = ShadowGenerationConfig(
        api_base="http://127.0.0.1:8000/v1",
        model="model",
        timeout_s=12,
        seed=0,
        mode="saga",
        saga_profile=str(profile),
    )
    provider = OpenClawReplayProvider(
        llm_actions=[action],
        replay_speed=1,
        timing_mode="source_scaled",
        shadow_generation=config,
        program_id="task-a",
    )
    assert provider._shadow_client is not None
    asyncio.run(provider._shadow_client.aclose())
    client = _SagaClient()
    provider._shadow_client = client

    response = asyncio.run(provider.chat([]))
    asyncio.run(provider.aclose())

    assert response.extra["shadow_generation"]["saga_profile_hit"] is True
    assert client.events[0][1]["vllm_xargs"] == {"saga_session_id": "task-a"}
    assert client.events[1][1]["reuse_probability"] == pytest.approx(2 / 3)
    assert client.events[1][1]["base_ttl_s"] == 2
    assert client.events[2][1]["finished"] is True


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
    captured: list[dict[str, object]],
    *,
    stream: bool = False,
    gzip_stream: bool = False,
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
    assert "single-GPU WA-LRU" in manifest["executable_action"][1]

    subprocess.run(["bash", "-n", SCRIPT], check=True)
    help_text = subprocess.run(
        ["bash", SCRIPT, "--help"], check=True, text=True, capture_output=True
    ).stdout
    assert "This is not full SAGA" in help_text
    assert "serve-afs-subset" in help_text
    assert "single-GPU" in help_text
    assert "serve MODEL" not in help_text
    assert json.dumps(manifest)
