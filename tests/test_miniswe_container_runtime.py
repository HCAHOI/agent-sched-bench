"""Tests for explicit MiniSWE container runtime selection."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("minisweagent", reason="requires mini-swe-agent")

from agents.miniswe.agent import MiniSWECodeAgent


class _FakeDockerEnvironment:
    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs

    def cleanup(self) -> None:
        return None


class _FakeContextManagedAgent:
    def __init__(self, lm, env, **kwargs) -> None:
        self.env = env
        self._full_messages: list[dict] = []

    def run(self, problem_statement: str) -> dict[str, str]:
        return {
            "exit_status": "Submitted",
            "submission": "diff --git a/x b/x",
        }


def _task() -> dict[str, str]:
    return {
        "instance_id": "encode__httpx-2701",
        "problem_statement": "Fix bug",
        "image_name": "docker.io/swerebench/example:latest",
    }


def test_miniswe_docker_runtime_requires_explicit_container_executable() -> None:
    agent = MiniSWECodeAgent(
        agent_id="encode__httpx-2701",
        api_base="https://example.com",
        model="qwen-plus-latest",
        api_key="test-key",
        runtime_mode="docker_container",
    )

    with pytest.raises(RuntimeError, match="container executable"):
        asyncio.run(agent.run(_task()))


@pytest.mark.parametrize(
    ("container_executable", "expected_user_args", "forbidden_arg"),
    [
        ("docker", ["--user", f"{os.getuid()}:{os.getgid()}"], "--userns=keep-id"),
        ("podman", ["--userns=keep-id"], "--user"),
    ],
)
def test_miniswe_uses_runtime_specific_container_args_not_env(
    monkeypatch,
    container_executable: str,
    expected_user_args: list[str],
    forbidden_arg: str,
) -> None:
    seen: dict[str, object] = {}

    def forbid_getenv(key: str, default: str | None = None) -> str | None:
        raise AssertionError(f"os.getenv should not be consulted: {key}")

    monkeypatch.setenv("MSWEA_DOCKER_EXECUTABLE", "podman")
    monkeypatch.setattr("agents.miniswe.agent.os.getenv", forbid_getenv)
    monkeypatch.setattr(
        "minisweagent.environments.docker.DockerEnvironment",
        lambda *args, **kwargs: (
            seen.update(
                {
                    "container_executable": kwargs["executable"],
                    "run_args": list(kwargs["run_args"]),
                }
            )
            or _FakeDockerEnvironment(*args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        "minisweagent.models.litellm_model.LitellmModel",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "agents.miniswe.agent.ContextManagedAgent",
        _FakeContextManagedAgent,
    )
    monkeypatch.setattr(
        "trace_collect.prompt_loader.load_prompt_template",
        lambda template_name: "Problem: {{task}}",
    )

    agent = MiniSWECodeAgent(
        agent_id="encode__httpx-2701",
        api_base="https://example.com",
        model="qwen-plus-latest",
        api_key="test-key",
        runtime_mode="docker_container",
        container_executable=container_executable,
    )

    success = asyncio.run(agent.run(_task()))

    assert success is True
    assert seen["container_executable"] == container_executable
    for arg in expected_user_args:
        assert arg in seen["run_args"]
    assert forbidden_arg not in seen["run_args"]
