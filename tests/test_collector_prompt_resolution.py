"""Tests for benchmark-driven prompt template defaults."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from trace_collect.cli import parse_collect_args
from trace_collect.cli import _run_collect
from trace_collect.cli import main
from trace_collect.collector import _resolve_prompt_template


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


def test_parse_collect_args_requires_container_for_collect() -> None:
    with pytest.raises(SystemExit, match="2"):
        parse_collect_args(["--provider", "openrouter", "--model", "z-ai/glm-5.1"])


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
            "--container",
            container_executable,
        ]
    )

    _run_collect(args)

    assert seen["container_executable"] == container_executable


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


def test_main_dispatches_import_without_collect_container_flag(monkeypatch) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "trace_collect.cli._run_import_claude_code",
        lambda args: seen.setdefault("session", args.session),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trace_collect.cli",
            "import-claude-code",
            "--session",
            "session.jsonl",
        ],
    )

    main()

    assert seen["session"] == "session.jsonl"


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
