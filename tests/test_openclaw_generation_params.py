from __future__ import annotations

from pathlib import Path

import pytest

from agents.terminal_bench.runner import TerminalBenchRunner
from llm_call.openclaw import UnifiedProvider


def test_unified_provider_includes_optional_generation_params() -> None:
    provider = UnifiedProvider(
        api_key="test",
        api_base="http://127.0.0.1:1/v1",
        default_model="test-model",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
    )

    kwargs = provider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model=None,
        max_tokens=16,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    )

    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.8
    assert kwargs["extra_body"] == {
        "top_k": 20,
        "repetition_penalty": 1.05,
    }


def test_terminal_bench_runner_passes_generation_agent_kwargs(tmp_path: Path) -> None:
    runner = TerminalBenchRunner(
        provider_name="openai",
        env_key="OPENAI_API_KEY",
        api_base="http://127.0.0.1:1/v1",
        api_key="test",
        model="test-model",
        workspace_base=tmp_path / "workspace",
        max_iterations=100,
        context_window_tokens=256_000,
        benchmark_slug="terminal-bench",
        benchmark_extras={},
        generation_config={
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
        },
    )

    command = runner._build_tb_command(
        task={
            "dataset_root": str(tmp_path / "dataset"),
            "task_id": "sample-task",
        },
        run_root=tmp_path / "run",
        run_id="sample-task",
        prompt_template="default",
    )

    assert "--agent-kwarg" in command
    assert "temperature=0.7" in command
    assert "top_p=0.8" in command
    assert "top_k=20" in command
    assert "repetition_penalty=1.05" in command


def test_terminal_bench_runner_disables_asciinema_in_runtime_copy(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_task = source_root / "sample-task"
    source_task.mkdir(parents=True)
    (source_task / "task.yaml").write_text(
        "instruction: fix it\n",
        encoding="utf-8",
    )
    (source_task / "solution.sh").write_text("true\n", encoding="utf-8")

    runtime_task = TerminalBenchRunner._materialize_runtime_task(
        task={
            "dataset_root": str(source_root),
            "task_id": "sample-task",
            "task_source_path": str(source_task),
        },
        run_root=tmp_path / "run",
    )

    runtime_yaml = (
        Path(runtime_task["task_source_path"]) / "task.yaml"
    ).read_text(encoding="utf-8")
    source_yaml = (source_task / "task.yaml").read_text(encoding="utf-8")

    assert runtime_task["dataset_root"] == str(tmp_path / "run" / "_dataset_no_asciinema")
    assert runtime_task["task_source_path"] == str(
        tmp_path / "run" / "_dataset_no_asciinema" / "sample-task"
    )
    assert "disable_asciinema: true" in runtime_yaml
    assert "disable_asciinema" not in source_yaml
    assert (Path(runtime_task["task_source_path"]) / "solution.sh").exists()


def test_terminal_bench_runner_replaces_task_yaml_symlink_in_runtime_copy(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_task = source_root / "sample-task"
    source_task.mkdir(parents=True)
    shared_yaml = tmp_path / "shared-task.yaml"
    shared_yaml.write_text("instruction: fix it\n", encoding="utf-8")
    (source_task / "task.yaml").symlink_to(shared_yaml)

    runtime_task = TerminalBenchRunner._materialize_runtime_task(
        task={
            "dataset_root": str(source_root),
            "task_id": "sample-task",
            "task_source_path": str(source_task),
        },
        run_root=tmp_path / "run",
    )

    runtime_yaml_path = Path(runtime_task["task_source_path"]) / "task.yaml"
    assert not runtime_yaml_path.is_symlink()
    assert "disable_asciinema: true" in runtime_yaml_path.read_text(
        encoding="utf-8"
    )
    assert "disable_asciinema" not in shared_yaml.read_text(encoding="utf-8")


def test_terminal_bench_runner_rejects_pathlike_task_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="task_id must be a simple name"):
        TerminalBenchRunner._materialize_runtime_task(
            task={
                "dataset_root": str(tmp_path / "source"),
                "task_id": "../escape",
            },
            run_root=tmp_path / "run",
        )
