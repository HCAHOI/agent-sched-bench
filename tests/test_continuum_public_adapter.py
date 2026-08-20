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


SCRIPT = Path(__file__).parents[1] / "scripts/baselines/continuum_public.sh"


def test_continuum_public_adapter_contract() -> None:
    subprocess.run(["bash", "-n", SCRIPT], check=True)
    help_text = subprocess.run(
        ["bash", SCRIPT, "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "316a587" in help_text
    assert "fixed 2-second KV pinning" in help_text
    assert "not the" in help_text and "TTL estimator" in help_text
    assert "Do not label it full Continuum" in help_text
    assert SCRIPT.read_text().count("'transformers>=4.55.2,<5'") == 2
    assert SCRIPT.read_text().count("cp --remove-destination") == 2

    from trace_collect.cli import parse_simulate_args

    args = parse_simulate_args(
        [
            "--manifest",
            "manifest.yaml",
            "--shadow-llm-mode",
            "continuum-public",
        ]
    )
    assert args.shadow_llm_mode == "continuum-public"


class _StreamResponse:
    def raise_for_status(self) -> None:
        pass

    async def aiter_lines(self):
        yield 'data: {"id":"shadow-1","prompt_token_ids":[11],"choices":[{"delta":{"token_ids":[101]},"finish_reason":"length"}]}'
        yield 'data: {"id":"shadow-1","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
        yield "data: [DONE]"


class _Stream:
    async def __aenter__(self) -> _StreamResponse:
        return _StreamResponse()

    async def __aexit__(self, *_args: object) -> bool:
        return False


class _Client:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def stream(self, _method: str, _url: str, *, json: dict[str, Any]) -> _Stream:
        self.requests.append(json)
        return _Stream()

    async def aclose(self) -> None:
        pass


def _source_action(index: int) -> dict[str, Any]:
    return {
        "action_type": "llm_call",
        "action_id": f"llm-{index}",
        "_source_action_index": index,
        "data": {
            "messages_in": [{"role": "user", "content": f"request {index}"}],
            "completion_tokens": 1,
            "raw_response": {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "source"}}
                ]
            },
        },
    }


def test_continuum_public_sends_only_causal_official_metadata() -> None:
    client = _Client()
    provider = OpenClawReplayProvider(
        llm_actions=[_source_action(index) for index in range(3)],
        replay_speed=1.0,
        timing_mode="source_scaled",
        shadow_generation=ShadowGenerationConfig(
            api_base="http://127.0.0.1:8000/v1",
            model="test-model",
            timeout_s=12.0,
            seed=0,
            mode="continuum_public",
        ),
        program_id="task-a",
        continuum_step_limit=2,
    )
    assert provider._shadow_client is not None
    asyncio.run(provider._shadow_client.aclose())
    provider._shadow_client = client

    asyncio.run(provider.chat([]))
    asyncio.run(provider.chat([]))
    asyncio.run(provider.aclose())

    assert [request["job_id"] for request in client.requests] == ["task-a", "task-a"]
    assert [request["is_last_step"] for request in client.requests] == [False, True]
    assert all("last_func_call" not in request for request in client.requests)
    assert all("this_func_call" not in request for request in client.requests)
    assert shadow_generation_payload(provider._shadow_generation)[
        "continuum_public"
    ] is True


def test_continuum_public_requires_declared_step_limit() -> None:
    with pytest.raises(ValueError, match="positive step limit"):
        OpenClawReplayProvider(
            llm_actions=[_source_action(0)],
            replay_speed=1.0,
            timing_mode="source_scaled",
            shadow_generation=ShadowGenerationConfig(
                api_base="http://127.0.0.1:8000/v1",
                model="test-model",
                timeout_s=12.0,
                seed=0,
                mode="continuum_public",
            ),
            program_id="task-a",
        )
