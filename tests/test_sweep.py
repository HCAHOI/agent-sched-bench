from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from harness.sweep import execute_sweep, expand_sweep_matrix, extract_agent_kwargs

minisweagent = pytest.importorskip("minisweagent", reason="requires mini-swe-agent")


def write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_expand_sweep_matrix_and_manifest_paths(tmp_path: Path) -> None:
    write_yaml(
        tmp_path / "configs/sweeps/default.yaml",
        """
matrix:
  systems: [vllm-baseline]
  workloads: [code_agent]
  concurrency: [1, 2]
""".strip(),
    )
    write_yaml(
        tmp_path / "configs/workloads/code_agent.yaml",
        "task_source: " + str(tmp_path / "tasks/code.json"),
    )
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks/code.json").write_text("[]\n", encoding="utf-8")

    runs = expand_sweep_matrix(
        sweep_config_path=tmp_path / "configs/sweeps/default.yaml",
        configs_root=tmp_path / "configs",
        output_root=tmp_path / "runs",
    )
    assert len(runs) == 2
    assert runs[0].tasks_file.endswith("code.json")
    assert runs[-1].output_file.endswith(".json")


def test_execute_sweep_writes_result_and_manifest_shape(tmp_path: Path) -> None:
    write_yaml(
        tmp_path / "configs/sweeps/default.yaml",
        """
matrix:
  systems: [vllm-baseline]
  workloads: [code_agent]
  concurrency: [1]
""".strip(),
    )
    write_yaml(
        tmp_path / "configs/workloads/code_agent.yaml",
        """
max_iterations: 40
command_timeout_s: 30
task_timeout_s: 300
task_source: """
        + str(tmp_path / "tasks/code.json"),
    )
    write_yaml(
        tmp_path / "configs/systems/vllm_baseline.yaml",
        'api_base: "http://localhost:8000/v1"',
    )
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks/code.json").write_text("[]\n", encoding="utf-8")

    runs = expand_sweep_matrix(
        sweep_config_path=tmp_path / "configs/sweeps/default.yaml",
        configs_root=tmp_path / "configs",
        output_root=tmp_path / "runs",
    )
    completed = asyncio.run(
        execute_sweep(
            runs=runs,
            configs_root=tmp_path / "configs",
            model="mock",
            arrival_mode="closed_loop",
            arrival_rate_per_s=None,
            arrival_seed=None,
            task_source_overrides={},
            sweep_config_path=str(tmp_path / "configs/sweeps/default.yaml"),
        )
    )
    assert len(completed) == 1
    result_path = Path(completed[0]["result_path"])
    assert result_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["execution_config"]["model"] == "mock"
    assert completed[0]["execution_config"]["sweep_config_path"].endswith(
        "default.yaml"
    )


def test_extract_agent_kwargs_max_iterations_defaults_to_100() -> None:
    kwargs = extract_agent_kwargs("code_agent", {"repos_root": "repos"})
    assert kwargs["max_iterations"] == 100
