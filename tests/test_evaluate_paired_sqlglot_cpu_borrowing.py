import asyncio
import json
from pathlib import Path

import pytest

from scripts.evaluation import evaluate_paired_sqlglot_cpu_borrowing as experiment


def test_fresh_counterbalanced_protocol_matches_preregistration() -> None:
    frozen_path = (
        experiment.ROOT
        / "analysis/results/tool-resource-5-3-3-3-20260804"
        / "sqlglot-final48-counterbalanced-rolling-exact-v1/protocol.json"
    )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    protocol = experiment._protocol("fresh_final48_counterbalanced_v1")

    assert protocol == frozen
    assert len(protocol["selected_task_ids"]) == 48
    assert len(protocol["pairs"]) == 4
    assert all(len(queue["task_ids"]) == 12 for queue in protocol["pairs"])
    assert [queue["arm_order"] for queue in protocol["pairs"]] == [
        ["hard_two", "burstable_two"],
        ["burstable_two", "hard_two"],
        ["burstable_two", "hard_two"],
        ["hard_two", "burstable_two"],
    ]
    assert set(protocol["selected_task_ids"]).isdisjoint(
        protocol["excluded_task_ids"]
    )
    assert protocol["framework_gate"][
        "source_images_absent_before_and_after_each_arm"
    ]
    assert protocol["framework_gate"][
        "maximum_resource_sample_boundary_gap_s"
    ] == 2.0
    assert protocol["quality_gate"][
        "timing_gate_requires_zero_source_terminal_class_differences"
    ]


def test_preregistered_result_directory_is_reused_without_overwrite(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "result"
    replay_dir = tmp_path / "replay"
    result_dir.mkdir()
    protocol = {
        "cohort": "fresh",
        "selected_task_ids": ["task"],
        "arm_level_resume": True,
    }
    protocol_path = result_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    config = {
        "result_dir": result_dir,
        "replay_dir": replay_dir,
        "preregistered_protocol": protocol_path,
    }

    experiment._prepare_new_run_directories(config, protocol)

    assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol
    assert json.loads((result_dir / "partial.json").read_text(encoding="utf-8")) == {
        "protocol": protocol,
        "runs": [],
    }
    assert replay_dir.is_dir()
    (result_dir / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="unexpected preregistration artifact"):
        experiment._prepare_new_run_directories(
            {**config, "replay_dir": tmp_path / "other-replay"}, protocol
        )


def test_exact_replay_environment_disables_timeout_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(experiment.OPENCLAW_EXEC_TIMEOUT_FLOOR_ENV, "old-floor")
    monkeypatch.setenv(experiment.OPENCLAW_PAIRED_WORKLOAD_CONTRACT_ENV, "old-contract")
    protocol = {
        "paired_workload_contract": True,
        "paired_workload_contract_version": 2,
        "exec_timeout_floor_s": None,
    }

    with experiment._replay_environment(protocol):
        assert experiment.OPENCLAW_EXEC_TIMEOUT_FLOOR_ENV not in experiment.os.environ
        assert (
            experiment.os.environ[experiment.OPENCLAW_PAIRED_WORKLOAD_CONTRACT_ENV]
            == "2"
        )

    assert experiment.os.environ[experiment.OPENCLAW_EXEC_TIMEOUT_FLOOR_ENV] == (
        "old-floor"
    )
    assert experiment.os.environ[experiment.OPENCLAW_PAIRED_WORKLOAD_CONTRACT_ENV] == (
        "old-contract"
    )


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


def _fresh_quality_runs(
    protocol: dict,
    improvements: list[float],
    *,
    burstable_failure_task_id: str | None = None,
    both_arms_source_mismatch_task_id: str | None = None,
) -> list[dict]:
    runs = _runs(protocol, improvements)
    for run in runs:
        for arm in run["arms"]:
            for task in arm["tasks"]:
                replay_class = (
                    "exit_nonzero"
                    if (
                        arm["arm"] == "burstable_two"
                        and task["task_id"] == burstable_failure_task_id
                    )
                    or task["task_id"] == both_arms_source_mismatch_task_id
                    else "exit_zero"
                )
                task["tool_outcomes"] = [
                    {
                        "action_id": f"tool-{task['task_id']}",
                        "tool_name": "exec",
                        "tool_call_id": f"call-{task['task_id']}",
                        "source": {"class": "exit_zero", "exit_code": 0},
                        "replay": {
                            "class": replay_class,
                            "exit_code": 0 if replay_class == "exit_zero" else 1,
                        },
                    }
                ]
    return runs


def test_fresh_counterbalanced_timing_gate_requires_three_queues() -> None:
    protocol = experiment._protocol("fresh_final48_counterbalanced_v1")
    runs = _fresh_quality_runs(protocol, [0.1, 0.1, 0.1, -0.01])

    result = experiment._aggregate(protocol, runs)

    assert result["status"] == "go"
    assert result["comparison"]["mean_improvement_fraction"] == pytest.approx(
        0.0725
    )
    assert result["comparison"]["improving_queues"] == 3
    assert result["comparison"]["gate"] == {
        "mean_improvement_at_least_5_percent": True,
        "at_least_3_of_4_queues_improve": True,
    }
    assert result["arm_outcome_quality"]["arm_terminal_class_difference_count"] == 0
    assert result["arm_outcome_quality"]["timing_gate_eligible"] is True


def test_fresh_counterbalanced_quality_difference_is_inconclusive() -> None:
    protocol = experiment._protocol("fresh_final48_counterbalanced_v1")
    changed_task = protocol["selected_task_ids"][0]
    runs = _fresh_quality_runs(
        protocol,
        [0.1, 0.1, 0.1, 0.1],
        burstable_failure_task_id=changed_task,
    )

    result = experiment._aggregate(protocol, runs)

    assert result["status"] == "inconclusive_quality_difference"
    assert result["arm_outcome_quality"]["relation_counts"] == {
        "same_terminal_class": 47,
        "burstable_success_hard_failure": 0,
        "burstable_failure_hard_success": 1,
        "different_failure_class": 0,
    }
    assert result["arm_outcome_quality"]["quality_regression_task_ids"] == [
        changed_task
    ]
    assert result["arm_outcome_quality"]["timing_gate_eligible"] is False
    assert result["comparison"]["gate"] is None


def test_fresh_counterbalanced_source_drift_is_inconclusive() -> None:
    protocol = experiment._protocol("fresh_final48_counterbalanced_v1")
    changed_task = protocol["selected_task_ids"][0]
    runs = _fresh_quality_runs(
        protocol,
        [0.1, 0.1, 0.1, 0.1],
        both_arms_source_mismatch_task_id=changed_task,
    )

    result = experiment._aggregate(protocol, runs)

    quality = result["arm_outcome_quality"]
    assert result["status"] == "inconclusive_quality_difference"
    assert quality["arm_terminal_class_difference_count"] == 0
    assert quality["source_terminal_class_difference_count"] == 2
    assert quality["source_terminal_class_difference_task_ids"] == [changed_task]
    assert quality["timing_gate_eligible"] is False
    assert result["comparison"]["gate"] is None


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
    assert protocol["cleanup_images"] is True
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
    cache_probes = []
    monkeypatch.setattr(
        experiment,
        "_cached_source_images",
        lambda task_ids, *, container_executable: cache_probes.append(
            (list(task_ids), container_executable)
        )
        or [],
    )

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
            cleanup_images=True,
            require_cold_source_images=True,
        )
    )

    assert captured["concurrency"] == 4
    assert captured["workers"] == 1
    assert captured["prep_concurrency"] == 4
    assert captured["cleanup_images"] is True
    assert cache_probes == [(task_ids, "docker"), (task_ids, "docker")]
    assert len(result["task_stats"]) == 5
    assert result["pair_makespan_s"] == 9.0


def test_rolling_arm_rejects_source_image_left_cached_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_ids = ["task"]

    async def fake_simulate(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        trace_file = output_dir / "trace.jsonl"
        trace_file.write_text("", encoding="utf-8")
        (output_dir / "throughput_summary.json").write_text(
            json.dumps(
                {
                    "attempted_traces": 1,
                    "wall_time_s": 1.0,
                    "concurrency": 1,
                    "effective_concurrency": 1,
                    "workers": 1,
                    "scheduler_mode": "bounded_queue",
                    "tasks": [
                        {
                            "agent_id": "task",
                            "failed_action_count": 0,
                            "elapsed_s": 1.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return trace_file

    probes = iter([[], ["docker.io/source-image"]])
    monkeypatch.setattr(experiment, "simulate", fake_simulate)
    monkeypatch.setattr(
        experiment,
        "_cached_source_images",
        lambda *_args, **_kwargs: next(probes),
    )

    with pytest.raises(AssertionError, match="source images cached after arm"):
        asyncio.run(
            experiment._run_arm(
                pair=1,
                task_ids=task_ids,
                arm="hard_two",
                manifest=tmp_path / "manifest.json",
                replay_dir=tmp_path,
                paired_workload_contract=True,
                concurrency=1,
                workers=1,
                makespan_source="throughput_summary.wall_time_s",
                cleanup_images=True,
                require_cold_source_images=True,
            )
        )


def test_source_image_probe_failure_is_not_treated_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment, "_task_source_images_by_id", lambda: {"task": "docker.io/image"}
    )
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda *_args, **_kwargs: experiment.subprocess.CompletedProcess(
            args=["docker"],
            returncode=1,
            stdout="",
            stderr="Cannot connect to the Docker daemon",
        ),
    )

    with pytest.raises(RuntimeError, match="source image probe failed"):
        experiment._cached_source_images(["task"], container_executable="docker")


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


def test_source_replay_tool_outcomes_preserve_quality_difference(tmp_path: Path) -> None:
    source = {
        "type": "action",
        "action_type": "tool_exec",
        "action_id": "tool-1",
        "data": {
            "tool_name": "exec",
            "tool_call_id": "call-1",
            "tool_args": json.dumps({"command": "pytest", "timeout": 60}),
            "tool_result": "ok\nExit code: 0",
            "success": True,
        },
    }
    replay = json.loads(json.dumps(source))
    replay["data"].update(
        {"tool_result": "failed\nExit code: 1", "success": True}
    )
    trace = tmp_path / "replay.jsonl"
    trace.write_text(json.dumps(replay) + "\n", encoding="utf-8")

    outcomes = experiment._source_replay_tool_outcomes(
        {"source_actions": [source]}, trace
    )

    assert outcomes == [
        {
            "action_id": "tool-1",
            "tool_name": "exec",
            "tool_call_id": "call-1",
            "source": {"class": "exit_zero", "exit_code": 0},
            "replay": {"class": "exit_nonzero", "exit_code": 1},
        }
    ]


def test_paired_task_artifact_retains_timeout_as_quality_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id = "task"
    arm = "hard_two"
    attempt = tmp_path / task_id / "attempt_1"
    attempt.mkdir(parents=True)
    (attempt / "container_startup.json").write_text(
        json.dumps(
            {
                "status": "success",
                "phases": [
                    {
                        "name": "start_task_container",
                        "status": "success",
                        "start_extra_args": list(experiment.ARM_ARGS[arm]),
                        "cpu_controls": {
                            "nano_cpus": 2_000_000_000,
                            "cpu_shares": 1024,
                            "cpuset_cpus": "0-7",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (attempt / "openclaw_host_replay_status.json").write_text(
        json.dumps(
            {
                "success": True,
                "missing_source_action_count": 0,
                "action_sequence_matches": True,
                "error": None,
                "emitted_actions": 1,
                "expected_actions": 1,
                "unexpected_replay_failed_actions": 0,
                "telemetry_quality": "ok",
                "telemetry_integrity_failed": False,
                "telemetry_errors": [],
            }
        ),
        encoding="utf-8",
    )
    (attempt / "resources.json").write_text(
        json.dumps(
            {
                "samples": [
                    {"timestamp": "start", "epoch": 9.0},
                    {"timestamp": "end", "epoch": 21.0},
                ],
                "summary": {
                    "sample_count": 2,
                    "duration_seconds": 12.0,
                    "monitoring_disabled": False,
                    "monitoring": {
                        "status": "collected",
                        "resource_enabled": True,
                        "per_task_resource_enabled": True,
                    },
                    "container_final_state": {
                        "status": "running",
                        "running": True,
                        "oom_killed": False,
                        "exit_code": 0,
                        "memory_events": {"oom": 0, "oom_kill": 0},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    request = {
        "exec_timeout_floor_s": experiment.EXEC_TIMEOUT_FLOOR_S,
        "paired_workload_contract": True,
        "replay_action_contract": {"require_source_outcome_match": False},
    }
    (attempt / "openclaw_host_replay_request.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    (attempt / "openclaw_host_replay.jsonl").write_text(
        json.dumps(
            {
                "type": "action",
                "action_type": "llm_call",
                "action_id": "llm-1",
                "ts_start": 10.0,
                "ts_end": 20.0,
                "data": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        experiment, "_source_success_replay_timeout_count", lambda *_args: 1
    )

    artifact = experiment._task_artifacts(
        tmp_path,
        task_id,
        arm,
        paired_workload_contract=True,
        require_telemetry_integrity=True,
    )

    assert artifact["source_success_replay_timeout_count"] == 1
    assert artifact["resource_sample_count"] == 2

    resources = json.loads((attempt / "resources.json").read_text(encoding="utf-8"))
    resources["samples"] = []
    resources["summary"]["sample_count"] = 0
    resources["summary"]["duration_seconds"] = 0.0
    (attempt / "resources.json").write_text(
        json.dumps(resources), encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="invalid resource telemetry"):
        experiment._task_artifacts(
            tmp_path,
            task_id,
            arm,
            paired_workload_contract=True,
            require_telemetry_integrity=True,
        )

    resources["samples"] = [
        {"timestamp": "start", "epoch": 9.0},
        {"timestamp": "stopped-early", "epoch": 11.0},
    ]
    resources["summary"]["sample_count"] = 2
    resources["summary"]["duration_seconds"] = 2.0
    (attempt / "resources.json").write_text(
        json.dumps(resources), encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="do not cover replay actions"):
        experiment._task_artifacts(
            tmp_path,
            task_id,
            arm,
            paired_workload_contract=True,
            require_telemetry_integrity=True,
        )

    resources["samples"] = [
        {"timestamp": "start", "epoch": 9.0},
        {"timestamp": "end", "epoch": 21.0},
    ]
    resources["summary"]["duration_seconds"] = 12.0
    (attempt / "resources.json").write_text(
        json.dumps(resources), encoding="utf-8"
    )
    request["replay_action_contract"]["require_source_outcome_match"] = True
    request["paired_workload_contract"] = False
    (attempt / "openclaw_host_replay_request.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="source-success replay timeout"):
        experiment._task_artifacts(
            tmp_path, task_id, arm, paired_workload_contract=False
        )


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
    protocol = {
        **experiment._protocol("remaining26"),
        "paired_workload_contract": True,
        "exec_timeout_floor_s": None,
        "paired_workload_contract_version": 2,
    }
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
            (str(arm_dir), task_id, arm, _kwargs)
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
    assert all(
        check[3]
        == {
            "paired_workload_contract": True,
            "exec_timeout_floor_s": None,
            "paired_workload_contract_version": 2,
            "require_telemetry_integrity": False,
            "telemetry_boundary_tolerance_s": 0.0,
            "use_latest_attempt": False,
        }
        for check in artifact_checks
    )

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


def test_arm_level_resume_accepts_empty_checkpoint(tmp_path: Path) -> None:
    protocol = experiment._protocol("rolling48_contract_v1")
    result_dir = tmp_path / "results"
    replay_dir = tmp_path / "replay"
    result_dir.mkdir()
    (result_dir / "protocol.json").write_text(
        json.dumps(protocol), encoding="utf-8"
    )
    (result_dir / "partial.json").write_text(
        json.dumps({"protocol": protocol, "runs": []}), encoding="utf-8"
    )

    assert experiment._load_resume_runs(protocol, result_dir, replay_dir) == []


def test_retry_uses_latest_attempt_directory(tmp_path: Path) -> None:
    instance_dir = tmp_path / "task"
    (instance_dir / "attempt_1").mkdir(parents=True)
    (instance_dir / "attempt_2").mkdir()

    assert experiment._task_attempt_dir(
        tmp_path, "task", use_latest_attempt=True
    ) == instance_dir / "attempt_2"
    assert experiment._task_attempt_dir(
        tmp_path, "task", use_latest_attempt=False
    ) == instance_dir / "attempt_1"


def test_quality_resume_rebuilds_task_artifacts(tmp_path, monkeypatch) -> None:
    protocol = {
        **experiment._protocol("rolling48_contract_v1"),
        "quality_gate": {"unit": "tool_call_terminal_class"},
        "framework_gate": {"telemetry_integrity_required": True},
    }
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
                "tasks": task_stats,
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
                "tasks": [{"task_id": "stale"}],
            }
        ],
    }
    (result_dir / "partial.json").write_text(
        json.dumps({"protocol": protocol, "runs": [run]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        experiment,
        "_task_artifacts",
        lambda _arm_dir, task_id, _arm, **_kwargs: {
            "task_id": task_id,
            "tool_outcomes": [{"replay": {"class": "exit_zero"}}],
        },
    )

    loaded = experiment._load_resume_runs(protocol, result_dir, replay_dir)

    assert [
        task["task_id"] for task in loaded[0]["arms"][0]["tasks"]
    ] == declared["task_ids"]


def test_fresh_resume_preserves_matching_invalid_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = {"cohort": "fresh"}
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps({"status": "invalid", "protocol": protocol}), encoding="utf-8"
    )
    expected_runs = [{"pair": 1, "arms": [{"arm": "hard_two"}]}]
    monkeypatch.setattr(
        experiment, "_load_resume_runs", lambda *_args: expected_runs
    )

    runs, preserved = experiment._resume_requested_run(
        "fresh_final48_counterbalanced_v1", protocol, tmp_path, tmp_path / "replay"
    )

    assert runs == expected_runs
    assert preserved == [tmp_path / "interruption-01.json"]
    assert not result_path.exists()


@pytest.mark.parametrize(
    ("success", "tool_result", "expected"),
    [
        (True, "done\n\nExit code: 0", {"class": "exit_zero", "exit_code": 0}),
        (
            True,
            "88 failed, 332 passed\n\nExit code: 1",
            {"class": "exit_nonzero", "exit_code": 1},
        ),
        (
            False,
            "Error: [timeout]\n\nExit code: 124",
            {"class": "timeout", "exit_code": 124},
        ),
        (
            False,
            "output\n[resource_stall_timeout]\n\nExit code: 124",
            {"class": "timeout", "exit_code": 124},
        ),
        (
            False,
            experiment.SAFETY_GUARD_REJECTION,
            {"class": "safety_rejection", "exit_code": None},
        ),
        (
            True,
            "Killed\n\nExit code: 137",
            {"class": "exit_nonzero", "exit_code": 137},
        ),
        (
            True,
            "Out of memory\n\nExit code: 137",
            {"class": "explicit_oom", "exit_code": 137},
        ),
    ],
)
def test_exec_terminal_outcome_uses_wrapper_status(
    success: bool, tool_result: str, expected: dict
) -> None:
    action = {
        "action_type": "tool_exec",
        "data": {
            "tool_name": "exec",
            "tool_call_id": "call-a",
            "success": success,
            "tool_result": tool_result,
        },
    }

    assert experiment._tool_terminal_outcome(action) == expected


def test_exec_terminal_outcome_fails_closed_without_wrapper_status() -> None:
    action = {
        "action_type": "tool_exec",
        "data": {
            "tool_name": "exec",
            "tool_call_id": "call-a",
            "success": True,
            "tool_result": "command output without wrapper status",
        },
    }

    with pytest.raises(ValueError, match="has no terminal status"):
        experiment._tool_terminal_outcome(action)


def test_existing_outcome_audit_separates_source_clean_diagnostic(tmp_path) -> None:
    def action(
        task_id: str,
        call_id: str,
        *,
        success: bool,
        tool_result: str,
        duration_s: float,
    ) -> dict:
        return {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": f"tool_1_{call_id}",
            "agent_id": task_id,
            "instance_id": task_id,
            "data": {
                "tool_name": "exec",
                "tool_call_id": call_id,
                "tool_args": json.dumps({"command": f"run {task_id}"}),
                "success": success,
                "tool_result": tool_result,
                "duration_ms": duration_s * 1000,
            },
        }

    clean_source = action(
        "clean", "call-clean", success=True, tool_result="ok\nExit code: 0", duration_s=5
    )
    drift_source = action(
        "drift", "call-drift", success=True, tool_result="ok\nExit code: 0", duration_s=5
    )
    failed_source = action(
        "failed",
        "call-failed",
        success=False,
        tool_result="Error: [timeout]\nExit code: 124",
        duration_s=10,
    )
    result = {
        "schema": "rolling",
        "status": "no_go",
        "protocol": {
            "queue_concurrency": 2,
            "pairs": [{"pair": 1, "task_ids": ["clean", "drift", "failed"]}],
        },
        "queues": [
            {
                "queue": 1,
                "task_ids": ["clean", "drift", "failed"],
                "arms": [],
            }
        ],
    }
    arm_specs = {
        "burstable_two": {
            "clean": (clean_source, 4.0, 1.0),
            "drift": (drift_source, 4.0, 1.0),
            "failed": (failed_source, 10.0, 1.0),
        },
        "hard_two": {
            "clean": (clean_source, 6.0, 1.0),
            "drift": (
                action(
                    "drift",
                    "call-drift",
                    success=True,
                    tool_result="tests failed\nExit code: 1",
                    duration_s=2,
                ),
                2.0,
                1.0,
            ),
            "failed": (
                action(
                    "failed",
                    "call-failed",
                    success=True,
                    tool_result="tests failed\nExit code: 1",
                    duration_s=2,
                ),
                2.0,
                1.0,
            ),
        },
    }
    for arm, tasks in arm_specs.items():
        arm_tasks = []
        task_stats = []
        for task_id, (replay_action, elapsed_s, prep_s) in tasks.items():
            attempt = tmp_path / arm / task_id / "attempt_1"
            attempt.mkdir(parents=True)
            source_action = {
                "clean": clean_source,
                "drift": drift_source,
                "failed": failed_source,
            }[task_id]
            (attempt / "openclaw_host_replay_request.json").write_text(
                json.dumps({"source_actions": [source_action]}), encoding="utf-8"
            )
            (attempt / "openclaw_host_replay.jsonl").write_text(
                json.dumps(replay_action) + "\n", encoding="utf-8"
            )
            (attempt / "container_startup.json").write_text(
                json.dumps({"elapsed_s": prep_s}), encoding="utf-8"
            )
            arm_tasks.append(
                {
                    "task_id": task_id,
                    "attempt_dir": str(attempt),
                    "replay_status": {
                        "source_failed_actions": int(task_id == "failed")
                    },
                }
            )
            task_stats.append({"agent_id": task_id, "elapsed_s": elapsed_s})
        result["queues"][0]["arms"].append(
            {"arm": arm, "tasks": arm_tasks, "task_stats": task_stats}
        )
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    audit = experiment._audit_existing_rolling_outcomes(result_path)

    assert audit["status"] == "diagnostic_only_no_verdict"
    assert audit["frozen_result_status"] == "no_go"
    assert audit["counts"] == {
        "task_count": 3,
        "source_clean_task_count": 2,
        "source_failed_task_count": 1,
        "source_clean_outcome_stable_task_count": 1,
        "source_clean_outcome_mismatch_task_count": 1,
        "tool_call_count": 3,
        "outcome_stable_tool_call_count": 1,
        "outcome_mismatch_tool_call_count": 2,
        "outcome_mismatch_task_count": 2,
        "arm_pair_outcome_mismatch_tool_call_count": 2,
        "arm_pair_outcome_mismatch_task_count": 2,
    }
    assert audit["mismatches"] == [
        {
            "task_id": "drift",
            "tool_call_id": "call-drift",
            "tool_name": "exec",
            "source": {"class": "exit_zero", "exit_code": 0},
            "burstable_two": {"class": "exit_zero", "exit_code": 0},
            "hard_two": {"class": "exit_nonzero", "exit_code": 1},
            "duration_s": {"burstable_two": 5.0, "hard_two": 2.0},
            "tool_args": json.dumps({"command": "run drift"}),
        },
        {
            "task_id": "failed",
            "tool_call_id": "call-failed",
            "tool_name": "exec",
            "source": {"class": "timeout", "exit_code": 124},
            "burstable_two": {"class": "timeout", "exit_code": 124},
            "hard_two": {"class": "exit_nonzero", "exit_code": 1},
            "duration_s": {"burstable_two": 10.0, "hard_two": 2.0},
            "tool_args": json.dumps({"command": "run failed"}),
        }
    ]
    assert audit["arm_outcome_quality"] == {
        "unit": "tool_call",
        "success_classes": ["exit_zero", "tool_success"],
        "relation_counts": {
            "same_terminal_class": 1,
            "burstable_success_hard_failure": 1,
            "burstable_failure_hard_success": 0,
            "different_failure_class": 1,
        },
        "burstable_quality_regression_count": 0,
        "selection_timing": "post-hoc descriptive audit",
    }
    assert audit["source_clean_outcome_stable_fixed_duration_diagnostic"] == {
        "task_count": 1,
        "queue_concurrency": 2,
        "execution_only": {
            "burstable_two_makespan_s": 4.0,
            "hard_two_makespan_s": 6.0,
            "improvement_fraction": pytest.approx(1 / 3),
        },
        "recorded_container_startup_plus_replay": {
            "burstable_two_makespan_s": 5.0,
            "hard_two_makespan_s": 7.0,
            "improvement_fraction": pytest.approx(2 / 7),
        },
    }
