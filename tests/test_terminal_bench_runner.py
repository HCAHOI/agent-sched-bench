from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.terminal_bench.runner import TerminalBenchRunner
from trace_collect.attempt_pipeline import AttemptContext


def _make_runner() -> TerminalBenchRunner:
    return TerminalBenchRunner(
        provider_name="openrouter",
        env_key="OPENROUTER_API_KEY",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        model="z-ai/glm-5.1",
        workspace_base=Path("workspace"),
        max_iterations=50,
        context_window_tokens=256_000,
        benchmark_slug="terminal-bench",
        benchmark_extras={"dataset_name": "terminal-bench-core", "dataset_version": "head"},
    )


def _make_ctx(tmp_path: Path) -> AttemptContext:
    return AttemptContext(
        run_dir=tmp_path / "run",
        instance_id="hello-world",
        attempt=1,
        task={"instance_id": "hello-world"},
        model="z-ai/glm-5.1",
        scaffold="openclaw",
        source_image=None,
    )


def test_preflight_requires_tb_and_docker(monkeypatch) -> None:
    runner = _make_runner()
    monkeypatch.setattr("agents.terminal_bench.runner.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("agents.terminal_bench.runner.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="tb CLI"):
        runner._preflight()


def test_build_tb_command_uses_agent_import_path() -> None:
    runner = _make_runner()
    cmd = runner._build_tb_command(
        task={"dataset_root": "/tmp/dataset", "task_id": "hello-world"},
        run_root=Path("/tmp/out"),
        run_id="hello-world",
    )
    joined = " ".join(cmd)
    assert "tb run" in joined
    assert TerminalBenchRunner.AGENT_IMPORT_PATH in joined
    assert "--dataset-path /tmp/dataset" in joined
    assert "--task-id hello-world" in joined
    assert "max_iterations=50" in joined


def test_extract_success_reads_terminal_bench_results(tmp_path: Path) -> None:
    runner = _make_runner()
    run_path = tmp_path / "tb-run"
    run_path.mkdir()
    (run_path / "results.json").write_text(
        json.dumps({"results": [{"is_resolved": True}]}),
        encoding="utf-8",
    )
    assert runner._extract_success(run_path) is True


def test_find_trace_path_prefers_agent_logs(tmp_path: Path) -> None:
    runner = _make_runner()
    trace = tmp_path / "task" / "trial" / "agent-logs" / "openclaw-trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("{}\n", encoding="utf-8")
    assert runner._find_trace_path(tmp_path) == trace


def test_augment_trace_metadata_stamps_terminal_bench_fields(tmp_path: Path) -> None:
    runner = _make_runner()
    src = tmp_path / "src.jsonl"
    src.write_text(
        json.dumps({"type": "trace_metadata", "model": "old", "instance_id": "x"}) + "\n"
        + json.dumps({"type": "summary", "success": True}) + "\n",
        encoding="utf-8",
    )
    dst = tmp_path / "dst.jsonl"
    runner._augment_trace_metadata(
        src=src,
        dst=dst,
        task={
            "instance_id": "hello-world",
            "task_source_kind": "terminal_bench_registry",
            "task_source_id": "hello-world",
            "task_source_path": "/tmp/dataset/hello-world",
            "tb_dataset": "terminal-bench-core",
            "tb_registry_source": "registry.json",
        },
        prompt_template="default",
        tb_version="0.2.18",
    )
    metadata = json.loads(dst.read_text(encoding="utf-8").splitlines()[0])
    assert metadata["benchmark"] == "terminal-bench"
    assert metadata["agent_runtime_mode"] == "host_controller"
    assert metadata["tb_version"] == "0.2.18"
    assert metadata["task_source_kind"] == "terminal_bench_registry"
