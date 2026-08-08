import asyncio
import json
from pathlib import Path

import pytest

from scripts.evaluation import evaluate_paired_sqlglot_cpu_borrowing as experiment


def test_frozen_cohorts_partition_validation_tasks() -> None:
    initial = experiment._protocol("initial24")
    remaining = experiment._protocol("remaining26")
    preflight = experiment._protocol("pair09_contract_v2")
    corrected = experiment._protocol("remaining26_contract_v2")
    compatible = experiment._protocol("remaining26_compatible_v2")
    quartet = experiment._protocol("quartet48_contract_v2")
    split = json.loads(experiment.SPLIT.read_text(encoding="utf-8"))

    initial_ids = set(initial["selected_task_ids"])
    remaining_ids = set(remaining["selected_task_ids"])
    assert initial["selection_range"] == [0, 24]
    assert remaining["selection_range"] == [24, 50]
    assert len(initial_ids) == 24
    assert len(remaining_ids) == 26
    assert initial_ids.isdisjoint(remaining_ids)
    assert initial_ids | remaining_ids == set(split["validation"])
    assert len(initial["pairs"]) == 12
    assert len(remaining["pairs"]) == 13
    assert preflight["pairs"] == [remaining["pairs"][8]]
    assert corrected["pairs"] == remaining["pairs"]
    assert compatible["pairs"] == remaining["pairs"]
    assert compatible["compatible_prefix_pairs"] == 9
    assert preflight["paired_workload_contract"] is True
    assert corrected["paired_workload_contract"] is True
    assert compatible["paired_workload_contract"] is True
    assert quartet["group_size"] == 4
    assert quartet["decision_unit"] == "group"
    assert quartet["arm_order_seed"] == 20_260_812
    assert quartet["bootstrap_seed"] == 20_260_813
    assert len(quartet["pairs"]) == 12
    assert all(len(group["task_ids"]) == 4 for group in quartet["pairs"])
    full_order = initial["selected_task_ids"] + remaining["selected_task_ids"]
    assert [
        task_id for group in quartet["pairs"] for task_id in group["task_ids"]
    ] == full_order[:48]
    assert set(quartet["selected_task_ids"]).isdisjoint(full_order[48:])
    frozen = json.loads(
        (
            experiment.ROOT
            / "analysis/results/tool-resource-5-3-3-3-20260804"
            / "sqlglot24-paired-cpu-borrowing-timeout-floor-v2/protocol.json"
        ).read_text(encoding="utf-8")
    )
    assert initial["selected_task_ids"] == frozen["selected_task_ids"]
    assert initial["pairs"] == frozen["pairs"]
    assert remaining["arm_order_seed"] == 20_260_810
    assert remaining["bootstrap_seed"] == 20_260_811
    assert [(pair["task_ids"], pair["arm_order"]) for pair in remaining["pairs"]] == [
        (
            ["tobymao__sqlglot-3620", "tobymao__sqlglot-3333"],
            ["hard_two", "burstable_two"],
        ),
        (
            ["tobymao__sqlglot-3238", "tobymao__sqlglot-2439"],
            ["hard_two", "burstable_two"],
        ),
        (
            ["tobymao__sqlglot-2857", "tobymao__sqlglot-2739"],
            ["hard_two", "burstable_two"],
        ),
        (
            ["tobymao__sqlglot-2754", "tobymao__sqlglot-3284"],
            ["hard_two", "burstable_two"],
        ),
        (
            ["tobymao__sqlglot-3182", "tobymao__sqlglot-3394"],
            ["hard_two", "burstable_two"],
        ),
        (
            ["tobymao__sqlglot-3734", "tobymao__sqlglot-3440"],
            ["burstable_two", "hard_two"],
        ),
        (
            ["tobymao__sqlglot-2794", "tobymao__sqlglot-2619"],
            ["burstable_two", "hard_two"],
        ),
        (
            ["tobymao__sqlglot-3417", "tobymao__sqlglot-3515"],
            ["hard_two", "burstable_two"],
        ),
        (
            ["tobymao__sqlglot-2443", "tobymao__sqlglot-3230"],
            ["burstable_two", "hard_two"],
        ),
        (
            ["tobymao__sqlglot-3179", "tobymao__sqlglot-2337"],
            ["hard_two", "burstable_two"],
        ),
        (
            ["tobymao__sqlglot-3198", "tobymao__sqlglot-3566"],
            ["burstable_two", "hard_two"],
        ),
        (
            ["tobymao__sqlglot-3637", "tobymao__sqlglot-2822"],
            ["burstable_two", "hard_two"],
        ),
        (
            ["tobymao__sqlglot-3630", "tobymao__sqlglot-3204"],
            ["burstable_two", "hard_two"],
        ),
    ]


def _runs(protocol: dict, improvements: list[float]) -> list[dict]:
    runs = []
    for pair, improvement in zip(protocol["pairs"], improvements, strict=True):
        pair_number = pair["pair"]
        arms = [
            {
                "arm": "hard_two",
                "pair_makespan_s": 1.0,
                "action_sequences": {"task": [["agent", "tool_exec", "tool"]]},
                "tasks": [
                    {"task_id": task_id, "replay_action_contract": {}}
                    for task_id in pair["task_ids"]
                ],
                "task_stats": [
                    {"agent_id": task_id, "elapsed_s": 1.0}
                    for task_id in pair["task_ids"]
                ],
            },
            {
                "arm": "burstable_two",
                "pair_makespan_s": 1.0 - improvement,
                "action_sequences": {"task": [["agent", "tool_exec", "tool"]]},
                "tasks": [
                    {"task_id": task_id, "replay_action_contract": {}}
                    for task_id in pair["task_ids"]
                ],
                "task_stats": [
                    {"agent_id": task_id, "elapsed_s": 1.0 - improvement}
                    for task_id in pair["task_ids"]
                ],
            },
        ]
        runs.append({"pair": pair_number, "arms": arms})
    return runs


def test_remaining_cohort_requires_ten_improving_pairs() -> None:
    protocol = experiment._protocol("remaining26")
    runs = _runs(protocol, [0.1] * 10 + [0.0] * 3)

    result = experiment._aggregate(protocol, runs)

    assert result["status"] == "go"
    assert result["comparison"]["improving_pairs"] == 10
    assert result["comparison"]["gate"]["at_least_10_of_13_pairs_improve"]

    runs = _runs(protocol, [0.1] * 9 + [0.0] * 4)
    result = experiment._aggregate(protocol, runs)
    assert result["status"] == "no_go"
    assert result["comparison"]["improving_pairs"] == 9
    assert not result["comparison"]["gate"]["at_least_10_of_13_pairs_improve"]


def test_remaining_cohort_freezes_bootstrap_and_effect_gates() -> None:
    protocol = experiment._protocol("remaining26")
    varied = [
        0.31,
        0.23,
        0.18,
        0.14,
        0.11,
        0.09,
        0.07,
        0.052,
        0.031,
        0.013,
        -0.017,
        -0.083,
        -0.271,
    ]
    result = experiment._aggregate(protocol, _runs(protocol, varied))
    comparison = result["comparison"]
    assert comparison["bootstrap_draws"] == 10_000
    assert comparison["ci95_paired_pair_bootstrap"] == pytest.approx(
        [-0.01361923076923077, 0.13892307692307695]
    )
    assert comparison["gate"]["mean_improvement_at_least_5_percent"]
    assert not comparison["gate"]["bootstrap_lower_above_zero"]
    assert result["status"] == "no_go"

    result = experiment._aggregate(protocol, _runs(protocol, [0.05] * 13))
    assert result["status"] == "go"
    result = experiment._aggregate(protocol, _runs(protocol, [0.049999] * 13))
    assert not result["comparison"]["gate"]["mean_improvement_at_least_5_percent"]
    assert result["comparison"]["gate"]["bootstrap_lower_above_zero"]
    assert result["status"] == "no_go"


def test_quartet_cohort_uses_group_gate_and_reports_task_completion() -> None:
    protocol = experiment._protocol("quartet48_contract_v2")
    result = experiment._aggregate(protocol, _runs(protocol, [0.1] * 9 + [0.0] * 3))

    assert result["status"] == "go"
    assert result["comparison"]["improving_groups"] == 9
    assert result["comparison"]["group_count"] == 12
    assert result["comparison"]["gate"]["at_least_9_of_12_groups_improve"]
    assert len(result["groups"]) == 12
    assert len(result["task_completion"]["paired_task_deltas"]) == 48
    assert result["task_completion"]["slowed_more_than_10_percent"] == 0

    result = experiment._aggregate(protocol, _runs(protocol, [0.1] * 8 + [0.0] * 4))
    assert result["status"] == "no_go"
    assert not result["comparison"]["gate"]["at_least_9_of_12_groups_improve"]


def test_rolling_queue_freezes_concurrency_order_and_effect_gate() -> None:
    protocol = experiment._protocol("rolling48_contract_v1")
    quartet = experiment._protocol("quartet48_contract_v2")

    assert protocol["selected_task_ids"] == quartet["selected_task_ids"]
    assert protocol["group_size"] == 48
    assert protocol["queue_concurrency"] == 4
    assert protocol["queue_workers"] == 1
    assert protocol["decision_unit"] == "queue"
    assert protocol["immediate_refill"] is True
    assert protocol["makespan_source"] == "throughput_summary.wall_time_s"
    assert len(protocol["pairs"]) == 1
    assert protocol["pairs"][0]["arm_order"] == ["burstable_two", "hard_two"]

    result = experiment._aggregate(protocol, _runs(protocol, [0.05]))
    assert result["status"] == "go"
    assert result["comparison"]["queue_count"] == 1
    assert result["comparison"]["makespan_improvement_fraction"] == pytest.approx(
        0.05
    )
    assert result["comparison"]["gate"] == {
        "makespan_improvement_at_least_5_percent": True,
        "all_validity_checks_passed": True,
    }
    assert "bootstrap_draws" not in result["comparison"]
    assert len(result["task_runtime"]["paired_task_deltas"]) == 48

    result = experiment._aggregate(protocol, _runs(protocol, [0.049999]))
    assert result["status"] == "no_go"
    assert not result["comparison"]["gate"][
        "makespan_improvement_at_least_5_percent"
    ]


def test_rolling_arm_limits_workers_without_limiting_task_count(
    tmp_path, monkeypatch
) -> None:
    task_ids = [f"task-{index}" for index in range(5)]
    captured = {}

    async def fake_simulate(**kwargs):
        captured.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        trace_file = output_dir / "trace.jsonl"
        trace_file.write_text("", encoding="utf-8")
        (output_dir / "throughput_summary.json").write_text(
            json.dumps(
                {
                    "attempted_traces": len(task_ids),
                    "wall_time_s": 9.0,
                    "concurrency": 4,
                    "effective_concurrency": 4,
                    "workers": 1,
                    "scheduler_mode": "bounded_queue",
                    "tasks": [
                        {
                            "agent_id": task_id,
                            "failed_action_count": 0,
                            "elapsed_s": float(index + 1),
                        }
                        for index, task_id in enumerate(task_ids)
                    ],
                }
            ),
            encoding="utf-8",
        )
        return trace_file

    monkeypatch.setattr(experiment, "simulate", fake_simulate)
    monkeypatch.setattr(
        experiment,
        "_task_artifacts",
        lambda _arm_dir, task_id, _arm, **_kwargs: {
            "task_id": task_id,
            "replay_action_contract": {},
        },
    )
    monkeypatch.setattr(experiment, "_action_sequences", lambda _path: {})

    result = asyncio.run(
        experiment._run_arm(
            pair=1,
            task_ids=task_ids,
            arm="burstable_two",
            manifest=tmp_path / "manifest.json",
            replay_dir=tmp_path,
            paired_workload_contract=True,
            concurrency=4,
            workers=1,
            makespan_source="throughput_summary.wall_time_s",
        )
    )

    assert captured["concurrency"] == 4
    assert captured["workers"] == 1
    assert captured["prep_concurrency"] == 4
    assert len(result["task_stats"]) == 5
    assert result["pair_makespan_s"] == 9.0


def test_quartet_resume_preserves_invalid_result(tmp_path) -> None:
    protocol = experiment._protocol("quartet48_contract_v2")
    result_path = tmp_path / "result.json"
    payload = {
        "status": "invalid",
        "protocol": protocol,
        "error": {"type": "RuntimeError", "message": "interrupted"},
        "traceback": "original traceback",
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    preserved = experiment._preserve_interruption_result(result_path, protocol)

    assert not result_path.exists()
    assert preserved == [tmp_path / "interruption-01.json"]
    assert json.loads(preserved[0].read_text(encoding="utf-8")) == payload


def test_timeout_gate_requires_matching_action_and_wrapper_failure(tmp_path) -> None:
    action = {
        "type": "action",
        "action_type": "tool_exec",
        "action_id": "tool_1_call-a",
        "data": {
            "tool_name": "exec",
            "tool_call_id": "call-a",
            "success": True,
        },
    }
    request = {"source_actions": [action]}
    trace = tmp_path / "trace.jsonl"

    def write_replay(*, success: bool, result: str, call_id: str = "call-a") -> None:
        replay = json.loads(json.dumps(action))
        replay["data"].update(
            {"success": success, "tool_result": result, "tool_call_id": call_id}
        )
        trace.write_text(json.dumps(replay) + "\n", encoding="utf-8")

    write_replay(success=True, result="[timeout]\nExit code: 124")
    assert experiment._source_success_replay_timeout_count(request, trace) == 0
    write_replay(success=False, result="Error: [timeout]\nExit code: 124")
    assert experiment._source_success_replay_timeout_count(request, trace) == 1
    write_replay(success=False, result="Error: [timeout]", call_id="call-b")
    with pytest.raises(AssertionError, match="action identities differ"):
        experiment._source_success_replay_timeout_count(request, trace)


def test_legacy_reuse_accepts_only_preserved_preexecution_failures() -> None:
    result = (
        "Error: Command blocked by safety guard (dangerous pattern detected)\n\n"
        "[Analyze the error above and try a different approach.]"
    )
    source = [
        {
            "action_type": "tool_exec",
            "data": {
                "tool_name": "exec",
                "tool_call_id": "call-a",
                "success": False,
                "tool_result": result,
            },
        }
    ]
    replay = {
        arm: [
            {
                "action_type": "tool_exec",
                "data": {
                    "tool_name": "exec",
                    "tool_call_id": "call-a",
                    "success": False,
                    "tool_result": result,
                },
            }
        ]
        for arm in ("hard_two", "burstable_two")
    }

    assert experiment._legacy_contract_compatibility(source, replay) == {
        "pytest_seed_count": 0,
        "preserved_preexecution_failure_count": 1,
    }

    replay["hard_two"][0]["data"]["tool_result"] = "Error: [timeout]"
    with pytest.raises(AssertionError, match="was not preserved"):
        experiment._legacy_contract_compatibility(source, replay)

    source[0]["data"]["tool_result"] = result + "\ncommand output"
    with pytest.raises(AssertionError, match="not a pre-execution safety rejection"):
        experiment._legacy_contract_compatibility(source, replay)


def test_saved_run_reconstruction_pins_replay_root(tmp_path) -> None:
    declared = experiment._protocol("remaining26_compatible_v2")["pairs"][0]
    stored = {
        "pair": 1,
        "arms": [
            {
                "arm": arm,
                "output_dir": str(tmp_path / "substitute" / arm),
                "trace_file": str(tmp_path / "substitute" / arm / "trace.jsonl"),
                "summary_path": str(
                    tmp_path / "substitute" / arm / "throughput_summary.json"
                ),
            }
            for arm in declared["arm_order"]
        ],
    }

    with pytest.raises(AssertionError, match="paths are invalid"):
        experiment._reconstruct_saved_run(
            declared,
            stored,
            paired_workload_contract=None,
            expected_replay_root=tmp_path / "expected",
        )


def test_resume_accepts_only_complete_frozen_pair_prefix(tmp_path, monkeypatch) -> None:
    protocol = experiment._protocol("remaining26")
    result_dir = tmp_path / "results"
    replay_dir = tmp_path / "replay"
    result_dir.mkdir()
    (result_dir / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    runs = []
    artifact_checks = []
    monkeypatch.setattr(
        experiment,
        "_task_artifacts",
        lambda arm_dir, task_id, arm, **_kwargs: artifact_checks.append(
            (str(arm_dir), task_id, arm)
        ),
    )
    for declared in protocol["pairs"][:3]:
        pair = declared["pair"]
        arms = []
        for arm in declared["arm_order"]:
            arm_dir = replay_dir / f"pair_{pair:02d}" / arm
            arm_dir.mkdir(parents=True)
            trace_file = arm_dir / "trace.jsonl"
            trace_file.write_text(
                "".join(
                    json.dumps(
                        {
                            "type": "action",
                            "action_type": "tool_exec",
                            "action_id": "tool",
                            "agent_id": task_id,
                            "instance_id": task_id,
                            "data": {"task_instance_id": task_id},
                        }
                    )
                    + "\n"
                    for task_id in declared["task_ids"]
                ),
                encoding="utf-8",
            )
            summary_path = arm_dir / "throughput_summary.json"
            task_stats = [
                {
                    "agent_id": task_id,
                    "failed_action_count": 0,
                    "elapsed_s": float(index + 1),
                }
                for index, task_id in enumerate(declared["task_ids"])
            ]
            summary_path.write_text(
                json.dumps({"attempted_traces": 2, "tasks": task_stats}) + "\n",
                encoding="utf-8",
            )
            arms.append(
                {
                    "arm": arm,
                    "output_dir": str(arm_dir),
                    "trace_file": str(trace_file),
                    "summary_path": str(summary_path),
                    "pair_makespan_s": 2.0,
                    "task_stats": task_stats,
                    "action_sequences": experiment._action_sequences(trace_file),
                }
            )
        runs.append({"pair": pair, "arms": arms})
    partial = result_dir / "partial.json"
    partial.write_text(
        json.dumps({"protocol": protocol, "runs": runs}), encoding="utf-8"
    )

    loaded = experiment._load_resume_runs(protocol, result_dir, replay_dir)

    assert len(loaded) == 3
    assert protocol["pairs"][len(loaded)]["pair"] == 4
    assert len(artifact_checks) == 12

    (result_dir / "protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError, match="stored protocol differs"):
        experiment._load_resume_runs(protocol, result_dir, replay_dir)
    (result_dir / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")

    original_summary_path = runs[0]["arms"][0]["summary_path"]
    runs[0]["arms"][0]["summary_path"] = runs[0]["arms"][0]["trace_file"]
    partial.write_text(
        json.dumps({"protocol": protocol, "runs": runs}), encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="output path is invalid"):
        experiment._load_resume_runs(protocol, result_dir, replay_dir)
    runs[0]["arms"][0]["summary_path"] = original_summary_path

    runs[0]["arms"][0]["pair_makespan_s"] = 3.0
    partial.write_text(
        json.dumps({"protocol": protocol, "runs": runs}), encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="makespan differs"):
        experiment._load_resume_runs(protocol, result_dir, replay_dir)
    runs[0]["arms"][0]["pair_makespan_s"] = 2.0

    summary_path = runs[0]["arms"][0]["summary_path"]
    failed_stats = list(runs[0]["arms"][0]["task_stats"])
    failed_stats[0] = {**failed_stats[0], "failed_action_count": 1}
    Path(summary_path).write_text(
        json.dumps({"attempted_traces": 2, "tasks": failed_stats}) + "\n",
        encoding="utf-8",
    )
    runs[0]["arms"][0]["task_stats"] = failed_stats
    partial.write_text(
        json.dumps({"protocol": protocol, "runs": runs}), encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="summary tasks differ"):
        experiment._load_resume_runs(protocol, result_dir, replay_dir)
    runs[0]["arms"][0]["task_stats"] = [
        {**failed_stats[0], "failed_action_count": 0},
        failed_stats[1],
    ]
    Path(summary_path).write_text(
        json.dumps({"attempted_traces": 2, "tasks": runs[0]["arms"][0]["task_stats"]})
        + "\n",
        encoding="utf-8",
    )

    runs[0]["arms"].reverse()
    partial.write_text(
        json.dumps({"protocol": protocol, "runs": runs}), encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="arm order differs"):
        experiment._load_resume_runs(protocol, result_dir, replay_dir)


def test_rolling_resume_accepts_completed_first_arm(tmp_path, monkeypatch) -> None:
    protocol = experiment._protocol("rolling48_contract_v1")
    result_dir = tmp_path / "results"
    replay_dir = tmp_path / "replay"
    result_dir.mkdir()
    (result_dir / "protocol.json").write_text(
        json.dumps(protocol), encoding="utf-8"
    )
    declared = protocol["pairs"][0]
    arm = declared["arm_order"][0]
    arm_dir = replay_dir / "pair_01" / arm
    arm_dir.mkdir(parents=True)
    trace_file = arm_dir / "trace.jsonl"
    trace_file.write_text("", encoding="utf-8")
    task_stats = [
        {"agent_id": task_id, "failed_action_count": 0, "elapsed_s": 1.0}
        for task_id in declared["task_ids"]
    ]
    summary_path = arm_dir / "throughput_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "attempted_traces": 48,
                "wall_time_s": 12.0,
                "concurrency": 4,
                "effective_concurrency": 4,
                "workers": 1,
                "scheduler_mode": "bounded_queue",
                "tasks": list(reversed(task_stats)),
            }
        ),
        encoding="utf-8",
    )
    run = {
        "pair": 1,
        "arms": [
            {
                "arm": arm,
                "output_dir": str(arm_dir),
                "trace_file": str(trace_file),
                "summary_path": str(summary_path),
                "pair_makespan_s": 12.0,
                "task_stats": task_stats,
                "action_sequences": {},
            }
        ],
    }
    (result_dir / "partial.json").write_text(
        json.dumps({"protocol": protocol, "runs": [run]}), encoding="utf-8"
    )
    monkeypatch.setattr(experiment, "_task_artifacts", lambda *_args, **_kwargs: {})

    loaded = experiment._load_resume_runs(protocol, result_dir, replay_dir)

    assert loaded == [run]
    assert declared["arm_order"][len(loaded[0]["arms"]) :] == ["hard_two"]
