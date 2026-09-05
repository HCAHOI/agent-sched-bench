"""Tests for trace_collect CLI argument parsing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trace_collect.cli import (
    _resource_collection_invalid,
    _resource_run_manifests_invalid,
    _run_collect,
    parse_collect_args,
    parse_simulate_args,
)


def test_parse_collect_args_accepts_skip_and_concurrency() -> None:
    args = parse_collect_args([
        "--provider",
        "openrouter",
        "--model",
        "z-ai/glm-5.1",
        "--skip",
        "7",
        "--selection-seed",
        "43",
        "--concurrency",
        "3",
        "--tool-resource-profile",
        "resource.yaml",
        "--service-tier",
        "fast",
    ])

    assert args.skip == 7
    assert args.selection_seed == 43
    assert args.concurrency == 3
    assert args.tool_resource_profile == "resource.yaml"
    assert args.service_tier == "fast"
    assert args.tool_resource_telemetry == "off"


def test_parse_collect_args_accepts_managed_clause_telemetry() -> None:
    args = parse_collect_args([
        "--provider",
        "openrouter",
        "--model",
        "z-ai/glm-5.1",
        "--tool-resource-telemetry",
        "clause",
    ])

    assert args.tool_resource_telemetry == "clause"
    assert args.tool_resource_profile is None


def test_parse_collect_args_rejects_profile_with_managed_telemetry() -> None:
    with pytest.raises(SystemExit):
        parse_collect_args([
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--tool-resource-profile",
            "resource.yaml",
            "--tool-resource-telemetry",
            "clause",
        ])


def test_managed_clause_telemetry_cleans_up_on_sigterm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trace_collect.cli as cli

    processes = [SimpleNamespace(pid=101), SimpleNamespace(pid=102)]
    stopped: list[tuple[int, bool]] = []
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *_args, **_kwargs: processes.pop(0))
    monkeypatch.setattr(cli, "_wait_for_socket", lambda *_args: None)
    monkeypatch.setattr(
        cli,
        "_stop_service",
        lambda process, privileged=False: stopped.append((process.pid, privileged)),
    )
    previous_sigterm = cli.signal.getsignal(cli.signal.SIGTERM)

    with pytest.raises(SystemExit) as exc:
        with cli._managed_clause_telemetry(
            tmp_path / ("long-run-name-" * 10),
            container_runtime="docker",
            verbose=False,
        ) as profile_path:
            profile = profile_path.read_text(encoding="utf-8")
            endpoint = next(
                line.split("unix://", 1)[1]
                for line in profile.splitlines()
                if "endpoint: unix://" in line
            )
            assert len(endpoint.encode()) < 108
            socket_dir = Path(endpoint).parent
            handler = cli.signal.getsignal(cli.signal.SIGTERM)
            assert callable(handler)
            handler(cli.signal.SIGTERM, None)

    assert exc.value.code == 128 + cli.signal.SIGTERM
    assert stopped == [(102, False), (101, True)]
    assert cli.signal.getsignal(cli.signal.SIGTERM) == previous_sigterm
    assert not socket_dir.exists()


def test_parse_collect_args_rejects_negative_skip() -> None:
    with pytest.raises(SystemExit):
        parse_collect_args([
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--skip",
            "-1",
        ])


def test_parse_collect_args_rejects_negative_sample() -> None:
    with pytest.raises(SystemExit):
        parse_collect_args([
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--sample",
            "-1",
        ])


def test_parse_collect_args_rejects_zero_concurrency() -> None:
    with pytest.raises(SystemExit):
        parse_collect_args([
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--concurrency",
            "0",
        ])


def test_parse_simulate_args_does_not_default_task_source() -> None:
    args = parse_simulate_args([
        "--manifest",
        "/abs/path/to/manifest.yaml",
    ])

    assert args.task_source is None


def test_parse_simulate_args_accepts_explicit_task_source() -> None:
    args = parse_simulate_args([
        "--manifest",
        "/abs/path/to/manifest.yaml",
        "--task-source",
        "/abs/path/to/tasks.json",
    ])

    assert args.task_source == "/abs/path/to/tasks.json"


@pytest.mark.parametrize("evidence_valid", [True, False])
def test_resource_collection_gate_uses_run_evidence(
    tmp_path: Path,
    evidence_valid: bool,
) -> None:
    trace_path = tmp_path / "simulate.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "type": "summary",
                "collection_validity": "not_requested",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_dir = tmp_path / "tool_resource_runs" / trace_path.stem
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "scope.json").write_text(
        json.dumps({"evidence_valid": evidence_valid}),
        encoding="utf-8",
    )
    assert _resource_collection_invalid(trace_path) is (not evidence_valid)


@pytest.mark.parametrize("evidence_valid", [True, False])
def test_collect_resource_gate_uses_persisted_run_evidence(
    tmp_path: Path,
    evidence_valid: bool,
) -> None:
    manifest_dir = tmp_path / "tool_resource_runs"
    manifest_dir.mkdir()
    (manifest_dir / "scope.json").write_text(
        json.dumps({"evidence_valid": evidence_valid}),
        encoding="utf-8",
    )
    assert _resource_run_manifests_invalid(manifest_dir) is (not evidence_valid)


def test_collect_resource_gate_rejects_missing_manifest(tmp_path: Path) -> None:
    assert _resource_run_manifests_invalid(tmp_path / "missing") is True


def test_collect_cli_fails_after_persisting_invalid_resource_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import agents.benchmarks
    from agents.benchmarks.base import BenchmarkConfig
    import trace_collect.cli as cli
    import trace_collect.collector

    run_dir = tmp_path / "run"
    manifest_dir = run_dir / "tool_resource_runs"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "scope.json").write_text(
        json.dumps({"evidence_valid": False}),
        encoding="utf-8",
    )
    config_path = tmp_path / "configs" / "benchmarks" / "fixture.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("slug: fixture\n", encoding="utf-8")

    async def fake_collect_traces(**_kwargs):
        return run_dir

    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "resolve_llm_config",
        lambda **_kwargs: SimpleNamespace(
            name="openai",
            env_key="OPENAI_API_KEY",
            api_base="https://example.invalid/v1",
            api_key="key",
            model="model",
        ),
    )
    monkeypatch.setattr(
        BenchmarkConfig,
        "from_yaml",
        lambda _path: SimpleNamespace(slug="fixture"),
    )
    monkeypatch.setattr(
        agents.benchmarks,
        "get_benchmark_class",
        lambda _slug: lambda _config: object(),
    )
    monkeypatch.setattr(
        trace_collect.collector,
        "collect_traces",
        fake_collect_traces,
    )
    args = parse_collect_args(
        [
            "--benchmark",
            "fixture",
            "--model",
            "model",
            "--api-key",
            "key",
            "--mcp-config",
            "none",
            "--tool-resource-profile",
            str(tmp_path / "resource.yaml"),
        ]
    )

    with pytest.raises(SystemExit) as exc:
        _run_collect(args)
    assert exc.value.code == 1
