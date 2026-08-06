from scripts.evaluation.measure_resource_underreservation import (
    EXPECTED_AST_IDENTITY,
    _burst_order,
    _burst_summary,
    _completion_order,
    _completion_summary,
    _cpu_quota_cores,
    _delta,
    _run_order,
    _summarize,
)


def _row(block: int, arm: str, elapsed: float, throttled: int, high: int = 0):
    return {
        "block": block,
        "arm": arm,
        "timed_out": False,
        "workload_exit": 0,
        "container_exit": 0,
        "workload": {
            "elapsed_s": elapsed,
            "source_files": 10,
            "bursts": [{"files": 10, "ast_nodes": 100}],
        },
        "cpu_delta": {"throttled_usec": throttled},
        "memory_events_delta": {"high": high, "oom_kill": 0},
    }


def test_frozen_order_and_gates() -> None:
    assert _run_order() == [
        (1, "cpu2"),
        (1, "cpu4"),
        (1, "memory_high_2g"),
        (1, "baseline"),
        (2, "memory_high_2g"),
        (2, "cpu2"),
        (2, "baseline"),
        (2, "cpu4"),
        (3, "cpu4"),
        (3, "memory_high_2g"),
        (3, "cpu2"),
        (3, "baseline"),
    ]
    rows = []
    for block in range(1, 4):
        rows.extend(
            [
                _row(block, "baseline", 20.0, 10),
                _row(block, "cpu4", 25.0, 20),
                _row(block, "cpu2", 35.0, 30),
                _row(block, "memory_high_2g", 24.0, 10, high=5),
            ]
        )
    result = _summarize(rows)
    assert result["cpu_gate"]
    assert result["memory_gate"]
    assert result["identical_successful_output"]
    assert _delta({"a": 7, "b": 2}, {"a": 3}) == {"a": 4, "b": 2}
    assert _cpu_quota_cores("200000 100000") == 2.0


def test_failed_or_inconsistent_runs_cannot_pass() -> None:
    rows = []
    for block in range(1, 4):
        rows.extend(
            [
                _row(block, "baseline", 20.0, 10),
                _row(block, "cpu4", 25.0, 20),
                _row(block, "cpu2", 35.0, 30),
                _row(block, "memory_high_2g", 24.0, 10, high=5),
            ]
        )
    rows[2]["workload"]["bursts"][0]["ast_nodes"] = 99
    rows[3]["timed_out"] = True
    rows[3]["workload"]["elapsed_s"] = 10_000.0

    result = _summarize(rows)

    assert result["elapsed_median_s"]["memory_high_2g"] == 24.0
    assert not result["identical_successful_output"]
    assert not result["cpu_gate"]
    assert not result["memory_gate"]


def test_completion_protocol_reports_finished_and_censored_results() -> None:
    assert _completion_order() == [
        (1, "memory_high_2g"),
        (1, "baseline"),
        (2, "memory_high_2g"),
        (2, "baseline"),
        (3, "baseline"),
        (3, "memory_high_2g"),
    ]
    rows = []
    for block in range(1, 4):
        rows.extend(
            [
                _row(block, "baseline", 20.0, 10),
                _row(block, "memory_high_2g", 400.0, 10, high=5),
            ]
        )
    completed = _completion_summary(rows)
    assert completed["valid"]
    assert completed["characterization"] == "completed"
    assert completed["memory_successful_ratio_to_baseline"] == 20.0

    for row in rows:
        if row["arm"] == "memory_high_2g" and row["block"] != 3:
            row["timed_out"] = True
    censored = _completion_summary(rows)
    assert censored["valid"]
    assert censored["characterization"] == "greater_than_3600s"
    assert censored["memory_successful_median_s"] is None
    assert censored["memory_successful_ratio_to_baseline"] is None


def test_burst_contention_gate_requires_speed_and_throttle_effect() -> None:
    assert _burst_order() == [
        (1, "burstable_two"),
        (1, "hard_two"),
        (1, "hard_four"),
        (2, "hard_four"),
        (2, "burstable_two"),
        (2, "hard_two"),
        (3, "burstable_two"),
        (3, "hard_four"),
        (3, "hard_two"),
    ]

    def batch(block: int, arm: str, wall_s: float, throttled: int):
        jobs = []
        for role, slot in (("ast", 1), ("ast", 2), ("idle", 1), ("idle", 2)):
            jobs.append(
                {
                    "role": role,
                    "slot": slot,
                    "workload_exit": 0,
                    "container_exit": 0,
                    "workload": {
                        "elapsed_s": wall_s,
                        "source_files": EXPECTED_AST_IDENTITY[0],
                        "bursts": [
                            {
                                "files": EXPECTED_AST_IDENTITY[1],
                                "ast_nodes": EXPECTED_AST_IDENTITY[2],
                            }
                        ],
                    },
                    "observed": {"cpuset_cpus_effective": "0-7"},
                    "telemetry_lost": False,
                    "docker_oom_killed": False,
                    "cpu_delta": {
                        "throttled_usec": throttled if role == "ast" else 0
                    },
                    "memory_events_delta": {"oom": 0, "oom_kill": 0},
                }
            )
        return {
            "block": block,
            "arm": arm,
            "timed_out": False,
            "abrupt_exit": False,
            "batch_wall_s": wall_s,
            "jobs": jobs,
        }

    rows = []
    for block in range(1, 4):
        rows.extend(
            [
                batch(block, "hard_two", 80.0, 100),
                batch(block, "burstable_two", 40.0, 0),
                batch(block, "hard_four", 38.0, 5),
            ]
        )
    result = _burst_summary(rows)
    assert result["valid"]
    assert result["gate"]
    assert result["status"] == "development_go_to_fresh_sqlglot_burst_protocol"

    rows[1]["batch_wall_s"] = 90.0
    assert not _burst_summary(rows)["gate"]

    rows[1]["batch_wall_s"] = 40.0
    rows[1]["jobs"][0]["memory_events_delta"]["oom"] = 1
    assert not _burst_summary(rows)["valid"]
