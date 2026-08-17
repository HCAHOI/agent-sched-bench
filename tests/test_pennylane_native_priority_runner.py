from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/evaluation/run_pennylane_native_priority.sh"
INPUT = ROOT / "analysis/development/pennylane-native-priority-v1"


def test_frozen_inputs_and_runner_contract() -> None:
    manifest = yaml.safe_load((INPUT / "manifest.yaml").read_text())
    ids = [Path(item["trace"]).parents[1].name for item in manifest["traces"]]
    assert ids == [
        "PennyLaneAI__pennylane-3182",
        "PennyLaneAI__pennylane-5835",
        "PennyLaneAI__pennylane-4366",
        "PennyLaneAI__pennylane-4251",
        "PennyLaneAI__pennylane-1405",
        "PennyLaneAI__pennylane-6062",
        "PennyLaneAI__pennylane-5623",
        "PennyLaneAI__pennylane-6939",
    ]
    assert [item["label"] for item in manifest["traces"]] == ids
    assert manifest["defaults"]["task_source"] == "../../../data/swe-rebench/tasks.json"
    assert yaml.safe_load((INPUT / "resource-profile.yaml").read_text())[
        "tool_resource"
    ] == {
        "endpoint": "unix:///run/agent-sched/resource/resource.sock",
        "behavior": "observe",
        "update_policy": "frozen",
        "snapshot": "latest_at_run_start",
        "telemetry_requirement": "required_for_valid_evidence",
        "latency_bucket_edges_ms": [500, 2000, 8000, 30000],
    }

    source = RUNNER.read_text()
    assert "model=NousResearch/Meta-Llama-3.1-8B-Instruct" in source
    assert (
        "run_root=/home/Ubuntu/pennylane-native-priority-physical-v1-20260818" in source
    )
    assert "--scheduling-policy priority" in source
    assert "max_model_len=131072" in source
    assert '--max-model-len "$max_model_len"' in source
    assert "max_context_tokens == 111_057" in source
    assert "--shadow-llm-max-concurrency" not in source
    assert "warmup-request" not in source
    assert "allowed-uid 1002" not in source
    assert "uid=$(id -u)" in source and "gid=$(id -g)" in source
    assert 'setsid "${vllm_args[@]}"' in source
    assert 'kill -TERM -- "-$vpid"' in source


def test_cell_order_and_mapping() -> None:
    script = f'''source "{RUNNER}"
printf '%s\\n' "${{cells[@]}}"
for cell in "${{cells[@]}}"; do
  cell_dir "$cell"
  cell_config "$cell"
done
'''
    result = subprocess.run(
        ["bash", "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.splitlines() == [
        "fixed-r1",
        "feedback-r1",
        "priority-feedback-r1",
        "priority-feedback-r2",
        "feedback-r2",
        "fixed-r2",
        "cell_01_fixed-r1",
        "fixed|1|fixed|",
        "cell_02_feedback-r1",
        "feedback|1|feedback|",
        "cell_03_priority-feedback-r1",
        "priority-feedback|1|feedback|1",
        "cell_04_priority-feedback-r2",
        "priority-feedback|2|feedback|1",
        "cell_05_feedback-r2",
        "feedback|2|feedback|",
        "cell_06_fixed-r2",
        "fixed|2|fixed|",
    ]


def test_preflight_fails_closed_outside_git(tmp_path: Path) -> None:
    script = f'''source "{RUNNER}"
repo="{tmp_path}"
preflight
'''
    result = subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "not a Git worktree" in result.stderr
