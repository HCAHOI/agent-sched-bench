import numpy as np

from scripts.evaluation.evaluate_kv_prediction_actionability import (
    _dev100_static_rows,
    _pmf_predictor,
    _remaining_from_pmf,
)
from tool_resource_eval.cachewise_kv_factorial import Program, Session, Turn
from tool_resource_eval.cachewise_reproduction import Gap


def test_decoder_conditions_on_survival_and_static_rows_ignore_dynamic() -> None:
    remaining = _remaining_from_pmf(
        (0.25, 0.75),
        2.0,
        (np.asarray([1.0]), np.asarray([3.0])),
    )
    row = _dev100_static_rows(
        [
            {
                "sample_id": "task:0",
                "task_id": "task",
                "command": "pytest",
                "labels": {
                    "latency": 4,
                    "peak_cpu_cores": 2,
                    "sampled_peak_rss_mb": 2,
                    "disk_read_write_bytes_total": 0,
                },
                "current_dynamic": {"latency": 4},
                "frozen_at_80": {
                    "latency": 0,
                    "sampled_peak_rss_mb": "low",
                    "disk_read_write_bytes_total": "low",
                    "probability_by_bucket": {
                        "latency": [1.0, 0.0, 0.0, 0.0, 0.0],
                        "peak_cpu_cores": [1.0, 0.0, 0.0],
                        "sampled_peak_rss_mb": [1.0, 0.0, 0.0],
                        "disk_read_write_bytes_total": [1.0, 0.0, 0.0],
                    },
                },
            }
        ]
    )[0]

    assert remaining == 1.0
    assert _remaining_from_pmf(
        (0.25, 0.75), 4.0, (np.asarray([1.0]), np.asarray([3.0]))
    ) is None
    assert row.current["latency"] == 0
    assert row.current["peak_cpu_cores"] is None


def test_pmf_lookup_uses_task_and_turn_not_future_gap_end() -> None:
    def session(end: float) -> Session:
        gap = Gap("task", 0.0, end, "exec", "pytest")
        return Session(
            Program(
                "task",
                (Turn(16, 16, 1.0, gap), Turn(32, 16, 1.0, None)),
            ),
            seed_rank=0,
            turn_index=1,
            arrival_s=end,
            gap_started_s=0.0,
        )

    predictor = _pmf_predictor(
        {("task", 0): (1.0,)},
        (np.asarray([10.0]),),
        lambda _session, _now: -1.0,
    )

    assert predictor(session(5.0), 1.0) == 9.0
    assert predictor(session(20.0), 1.0) == 9.0
