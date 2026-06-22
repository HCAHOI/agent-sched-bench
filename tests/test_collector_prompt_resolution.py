"""Tests for benchmark-driven prompt template defaults."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import os
import sys

import pytest

from trace_collect.cli import parse_collect_args
from trace_collect.cli import _run_collect
from trace_collect.cli import main

from trace_collect.collector import (
    _INTERNAL_HF_API_KEY,
    _prepare_collect_model_backend,
    _recording_server_public_host,
    _resolve_prompt_template,
)


def test_parse_collect_args_prompt_template_defaults_to_none() -> None:
    args = parse_collect_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--container",
            "docker",
        ]
    )
    assert args.prompt_template is None


def test_parse_collect_args_max_iterations_defaults_to_100() -> None:
    args = parse_collect_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--container",
            "docker",
        ]
    )
    assert args.max_iterations == 100


def test_parse_collect_args_record_internals_defaults_to_false() -> None:
    args = parse_collect_args(["--provider", "openrouter", "--model", "z-ai/glm-5.1"])
    assert args.record_internals is False


def test_parse_collect_args_local_hf_defaults_to_false() -> None:
    args = parse_collect_args(["--provider", "openrouter", "--model", "z-ai/glm-5.1"])
    assert args.local_hf is False


def test_parse_collect_args_accepts_record_internals() -> None:
    args = parse_collect_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--record-internals",
        ]
    )
    assert args.record_internals is True


def test_parse_collect_args_accepts_local_hf() -> None:
    args = parse_collect_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--local-hf",
        ]
    )
    assert args.local_hf is True


def test_parse_collect_args_allows_omitted_container_for_host_mode() -> None:
    args = parse_collect_args(["--provider", "openrouter", "--model", "z-ai/glm-5.1"])
    assert args.container is None


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_parse_collect_args_accepts_explicit_container(
    container_executable: str,
) -> None:
    args = parse_collect_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--mcp-config",
            "none",
            "--container",
            container_executable,
        ]
    )

    assert args.container == container_executable


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_run_collect_passes_container_to_collect_traces(
    container_executable: str,
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    async def fake_collect_traces(**kwargs):
        seen.update(kwargs)
        return Path("/tmp/fake-run")

    monkeypatch.setattr(
        "trace_collect.cli.resolve_llm_config",
        lambda **kwargs: SimpleNamespace(
            name="openrouter",
            api_base="https://example.com",
            api_key="test-key",
            model="z-ai/glm-5.1",
            env_key="OPENROUTER_API_KEY",
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.collect_traces",
        fake_collect_traces,
    )

    args = parse_collect_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--mcp-config",
            "none",
            "--container",
            container_executable,
        ]
    )

    _run_collect(args)

    assert seen["container_executable"] == container_executable


def test_run_collect_passes_record_internals(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def fake_collect_traces(**kwargs):
        seen.update(kwargs)
        return Path("/tmp/fake-run")

    monkeypatch.setattr(
        "trace_collect.cli.resolve_llm_config",
        lambda **kwargs: SimpleNamespace(
            name="openrouter",
            api_base="https://example.com",
            api_key="test-key",
            model="z-ai/glm-5.1",
            env_key="OPENROUTER_API_KEY",
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.collect_traces",
        fake_collect_traces,
    )

    args = parse_collect_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--mcp-config",
            "none",
            "--record-internals",
        ]
    )

    _run_collect(args)

    assert seen["record_internals"] is True


def test_run_collect_passes_local_hf_without_api_key(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def fake_collect_traces(**kwargs):
        seen.update(kwargs)
        return Path("/tmp/fake-run")

    monkeypatch.delenv("NANOBOT_MAX_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.setattr(
        "trace_collect.cli.resolve_llm_config",
        lambda **kwargs: SimpleNamespace(
            name="openrouter",
            api_base="https://example.com",
            api_key="",
            model="z-ai/glm-5.1",
            env_key="OPENROUTER_API_KEY",
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.collect_traces",
        fake_collect_traces,
    )

    args = parse_collect_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--mcp-config",
            "none",
            "--container",
            "docker",
            "--local-hf",
        ]
    )

    _run_collect(args)

    assert seen["local_hf"] is True
    assert seen["record_internals"] is False
    assert seen["api_key"] == ""
    assert seen["eviction_config"] is None
    assert os.environ["NANOBOT_MAX_CONCURRENT_REQUESTS"] == "1"


def test_run_collect_allows_metadata_kv_without_record_internals(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def fake_collect_traces(**kwargs):
        seen.update(kwargs)
        return Path("/tmp/fake-run")

    monkeypatch.delenv("NANOBOT_MAX_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.setattr(
        "trace_collect.cli.resolve_llm_config",
        lambda **kwargs: SimpleNamespace(
            name="openrouter",
            api_base="https://example.com",
            api_key="test-key",
            model="z-ai/glm-5.1",
            env_key="OPENROUTER_API_KEY",
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.collect_traces",
        fake_collect_traces,
    )

    args = parse_collect_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--mcp-config",
            "none",
            "--container",
            "docker",
            "--kv-policy",
            "metadata",
            "--kv-budget",
            "4096",
            "--kv-record",
            "off",
        ]
    )

    _run_collect(args)

    eviction_config = seen["eviction_config"]
    assert seen["record_internals"] is False
    assert eviction_config.name == "metadata"
    assert eviction_config.record is False
    assert seen["sparse_attention_config"] is None
    assert os.environ["NANOBOT_MAX_CONCURRENT_REQUESTS"] == "1"


def test_run_collect_rejects_attention_kv_without_record_internals(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "trace_collect.cli.resolve_llm_config",
        lambda **kwargs: SimpleNamespace(
            name="openrouter",
            api_base="https://example.com",
            api_key="test-key",
            model="z-ai/glm-5.1",
            env_key="OPENROUTER_API_KEY",
        ),
    )

    args = parse_collect_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--mcp-config",
            "none",
            "--container",
            "docker",
            "--kv-policy",
            "h2o",
            "--kv-budget",
            "4096",
        ]
    )

    with pytest.raises(SystemExit) as excinfo:
        _run_collect(args)

    assert excinfo.value.code == 2
    assert "requires attention" in capsys.readouterr().err


def test_prepare_collect_model_backend_preserves_external_endpoint() -> None:
    with ExitStack() as stack:
        backend = _prepare_collect_model_backend(
            use_hf_backend=False,
            record_internals=False,
            local_hf=False,
            model="Qwen/Qwen3-32B",
            api_base="http://localhost:8000/v1",
            api_key="dummy",
            provider_name="openai",
            execution_environment="host",
            runtime_mode="host_controller",
            container_executable=None,
            cleanup_stack=stack,
            eviction_config=None,
            sparse_attention_config=None,
            per_head_stats_layers=(),
            per_head_block_stats=False,
            record_per_head_topk=False,
            per_head_topk_rank=64,
            generation_seed=0,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            generation_config={},
        )

    assert backend.provider_name == "openai"
    assert backend.api_base == "http://localhost:8000/v1"
    assert backend.api_key == "dummy"
    assert backend.recording_provider is None
    assert backend.trace_run_config == {}


def test_prepare_collect_model_backend_materializes_internal_hf_endpoint(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    class FakeRecordingConfig:
        def __init__(self, **kwargs) -> None:
            seen["recording_config"] = dict(kwargs)

    class FakeProvider:
        def __init__(self, **kwargs) -> None:
            seen["provider_kwargs"] = dict(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            pass

    class FakeServer:
        api_base = "http://127.0.0.1:4321/v1"

        def __init__(self, provider, **kwargs) -> None:
            seen["server_provider"] = provider
            seen["server_kwargs"] = dict(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            pass

    monkeypatch.setattr("serving.recording.RecordingConfig", FakeRecordingConfig)
    monkeypatch.setattr("serving.recording.HFRecordingProvider", FakeProvider)
    monkeypatch.setattr("serving.recording.HFRecordingServer", FakeServer)

    with ExitStack() as stack:
        backend = _prepare_collect_model_backend(
            use_hf_backend=True,
            record_internals=False,
            local_hf=True,
            model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            api_base="https://unused.example/v1",
            api_key="",
            provider_name="openrouter",
            execution_environment="container",
            runtime_mode="task_container_agent",
            container_executable="docker",
            cleanup_stack=stack,
            eviction_config=None,
            sparse_attention_config=None,
            per_head_stats_layers=(),
            per_head_block_stats=False,
            record_per_head_topk=False,
            per_head_topk_rank=64,
            generation_seed=0,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            generation_config={},
        )

    assert backend.provider_name == "openai"
    assert backend.api_base == "http://127.0.0.1:4321/v1"
    assert backend.api_key == _INTERNAL_HF_API_KEY
    assert backend.recording_provider is seen["server_provider"]
    assert backend.trace_run_config == {"local_hf": True, "hf_backend": "local_hf"}
    assert seen["recording_config"]["record_artifacts"] is False
    assert seen["provider_kwargs"]["eviction_config"] is None


def test_recording_server_public_host_defaults_for_docker_container(monkeypatch) -> None:
    monkeypatch.delenv("HF_RECORDING_PUBLIC_HOST", raising=False)

    assert (
        _recording_server_public_host(
            execution_environment="container",
            runtime_mode="task_container_agent",
            container_executable="docker",
        )
        == "172.17.0.1"
    )


def test_recording_server_public_host_prefers_env_override(monkeypatch) -> None:
    monkeypatch.setenv("HF_RECORDING_PUBLIC_HOST", "10.0.0.5")

    assert (
        _recording_server_public_host(
            execution_environment="container",
            runtime_mode="task_container_agent",
            container_executable="docker",
        )
        == "10.0.0.5"
    )


def test_main_dispatches_simulate_without_collect_container_flag(monkeypatch) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "trace_collect.cli._run_simulate",
        lambda args: seen.setdefault("source_trace", args.source_trace),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trace_collect.cli",
            "simulate",
            "--source-trace",
            "trace.jsonl",
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
        ],
    )

    main()

    assert seen["source_trace"] == "trace.jsonl"


def test_resolve_prompt_template_uses_benchmark_default_when_unset() -> None:
    benchmark = SimpleNamespace(
        config=SimpleNamespace(default_prompt_template="cc_aligned")
    )
    assert (
        _resolve_prompt_template(benchmark=benchmark, prompt_template=None)
        == "cc_aligned"
    )


def test_resolve_prompt_template_respects_explicit_override() -> None:
    benchmark = SimpleNamespace(
        config=SimpleNamespace(default_prompt_template="cc_aligned")
    )
    assert (
        _resolve_prompt_template(benchmark=benchmark, prompt_template="default")
        == "default"
    )
