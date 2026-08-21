from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest

from trace_collect.openclaw_host_runtime import (
    OpenClawReplayProvider,
    ShadowGenerationConfig,
    shadow_generation_payload,
)


class _StreamResponse:
    def raise_for_status(self) -> None:
        pass

    async def aiter_lines(self):
        yield 'data: {"id":"shadow-1","prompt_token_ids":[11,12],"choices":[{"delta":{"token_ids":[101]},"finish_reason":"length"}]}'
        yield 'data: {"id":"shadow-1","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}'
        yield "data: [DONE]"


class _Stream:
    async def __aenter__(self) -> _StreamResponse:
        return _StreamResponse()

    async def __aexit__(self, *_args: object) -> bool:
        return False


class _ReleaseResponse:
    def __init__(self, released: bool) -> None:
        self._released = released

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, bool]:
        return {"released": self._released}


class _Client:
    def __init__(self, *, released: bool = True, **_kwargs: object) -> None:
        self.released = released
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.release_calls: list[tuple[str, dict[str, Any]]] = []

    def stream(self, method: str, url: str, *, json: dict[str, Any]) -> _Stream:
        self.requests.append((method, url, json))
        return _Stream()

    async def post(self, url: str, *, json: dict[str, Any]) -> _ReleaseResponse:
        self.release_calls.append((url, json))
        return _ReleaseResponse(self.released)

    async def aclose(self) -> None:
        pass


def _source_action() -> dict[str, Any]:
    messages = [{"role": "user", "content": "unchanged workload"}]
    return {
        "action_type": "llm_call",
        "action_id": "llm-1",
        "_source_action_index": 0,
        "data": {
            "messages_in": messages,
            "completion_tokens": 1,
            "raw_response": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "source response"},
                    }
                ]
            },
        },
    }


def _provider(client: _Client) -> OpenClawReplayProvider:
    provider = OpenClawReplayProvider(
        llm_actions=[_source_action()],
        replay_speed=1.0,
        timing_mode="source_scaled",
        shadow_generation=ShadowGenerationConfig(
            api_base="http://127.0.0.1:9000/v1",
            model="test-model",
            timeout_s=12.0,
            seed=0,
            mode="thunderagent",
        ),
        program_id="task-a",
    )
    assert provider._shadow_client is not None
    asyncio.run(provider._shadow_client.aclose())
    provider._shadow_client = client
    return provider


def test_thunderagent_adapter_tags_request_and_releases_once() -> None:
    client = _Client()
    provider = _provider(client)

    asyncio.run(provider.chat([{"role": "user", "content": "ignored replay input"}]))
    asyncio.run(provider.aclose())
    asyncio.run(provider.aclose())

    method, url, payload = client.requests[0]
    assert (method, url) == (
        "POST",
        "http://127.0.0.1:9000/v1/chat/completions",
    )
    assert payload["program_id"] == "task-a"
    assert payload["messages"] == _source_action()["data"]["messages_in"]
    assert client.release_calls == [
        (
            "http://127.0.0.1:9000/programs/release",
            {"program_id": "task-a"},
        )
    ]


def test_thunderagent_release_failure_is_fatal() -> None:
    provider = _provider(_Client(released=False))

    asyncio.run(provider.chat([{"role": "user", "content": "ignored"}]))
    with pytest.raises(RuntimeError, match="did not release program"):
        asyncio.run(provider.aclose())


def test_thunderagent_mode_is_recorded_without_changing_plain_metadata() -> None:
    kwargs = {
        "api_base": "http://127.0.0.1:9000/v1",
        "model": "test-model",
        "timeout_s": 12.0,
        "seed": 0,
    }

    assert "thunderagent" not in shadow_generation_payload(
        ShadowGenerationConfig(**kwargs)
    )
    assert shadow_generation_payload(
        ShadowGenerationConfig(**kwargs, mode="thunderagent")
    )["thunderagent"] is True


def test_paper_baseline_suite_smokes_matching_methods_before_full_run() -> None:
    root = Path(__file__).parents[1]
    runner = root / "scripts/evaluation/run_pennylane_thunderagent_baseline.sh"
    suite = root / "scripts/evaluation/run_pennylane_paper_baseline_suite.sh"
    subprocess.run(["bash", "-n", runner, suite], check=True)
    suite_text = suite.read_text()
    assert 'export PATH="$HOME/.local/bin:$PATH"' in suite_text
    assert "--kv-cache-dtype auto --kv-layout-dtype bfloat16" in suite_text
    assert 'RUN_OUTPUT_DIR="$suite_root/profile-continuum"' in suite_text
    assert '"$repo/.venv/bin/python" -c \'import loguru, trace_collect\'' in suite_text
    assert "--resume-after-profile) resume_after_profile" in suite_text
    assert '[[ -s "$profile" ]]' in suite_text
    assert 'smoke_succeeded "$method" || failed=1' in suite_text
    assert 'smoke_succeeded "$method" && continue' in suite_text
    assert "methods=(agentix continuum-public continuum-reproduction cachewise)" in suite_text
    assert "smoke-$method" in suite_text
    assert suite_text.index('for method in "${methods[@]}"') < suite_text.index(
        'RUN_ROOT="$suite_root/full"'
    )
    assert "saga" not in suite_text.lower()
    assert "murakkab" not in suite_text.lower()
    runner_text = runner.read_text()
    assert 'continuum_reproduction.sh" serve "$model" --dtype bfloat16 --kv-cache-dtype auto' in runner_text
    assert "--shadow-llm-cachewise-predictor-checkout" in runner_text
    assert "--queue-upper-bounds 0.25,1,4,16" in runner_text
