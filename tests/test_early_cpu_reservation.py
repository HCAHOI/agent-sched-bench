import pytest

from tool_resource_eval.early_cpu_reservation import feedback_action_row


def _action(*, duration: float | None = None, sample_dt: float = 0.5) -> dict:
    if duration is None:
        duration = 4 * sample_dt + 0.02
    samples = [
        (sample_dt, 0.5 * sample_dt),
        (2 * sample_dt, 1.5 * sample_dt),
        (3 * sample_dt, 3.0 * sample_dt),
        (4 * sample_dt, 0.4 * sample_dt),
    ]
    return {
        "type": "action",
        "action_id": "action-1",
        "ts_start": 10.0,
        "ts_end": 10.0 + duration,
        "data": {
            "tool_name": "exec",
            "tool_call_id": "call-1",
            "resource_timeline": {
                "version": 1,
                "sample_interval_s": 0.5,
                "samples": [
                    {
                        "offset_s": offset,
                        "dt_s": sample_dt,
                        "cpu_core_s": cpu,
                        "cpu_quota_cores": 8,
                    }
                    for offset, cpu in samples
                    if offset <= duration
                ],
            },
        },
    }


def test_feedback_selects_a_page_then_returns_to_eight_after_throttling() -> None:
    action = _action()
    samples = action["data"]["resource_timeline"]["samples"]
    samples[0]["cpu_core_s"] = 0.5  # 1 core: select 2 after the delayed update.
    samples[1]["cpu_core_s"] = 1.5  # 3 cores: throttle after selecting 2.
    samples[2]["cpu_core_s"] = 1.5  # The delayed next update returns to 8.

    row, reason = feedback_action_row(action)

    assert reason == "eligible"
    assert row["eligible"] is True
    feedback = row["arms"]["feedback"]
    probe = row["arms"]["probe_then_two"]
    assert set(feedback["request_counts"]) == {"2", "8"}
    assert feedback["throttled_samples"] == 2
    assert feedback["added_service_s"] < probe["added_service_s"]
    assert row["arms"]["fixed8"]["added_service_s"] == pytest.approx(0.0)


def test_feedback_keeps_short_actions_at_eight() -> None:
    row, reason = feedback_action_row(_action(sample_dt=0.45))

    assert reason == "no_full_decision_sample"
    assert row["eligible"] is False
    assert row["arms"]["feedback"] == row["arms"]["fixed8"]


def test_feedback_rejects_a_censored_source_quota() -> None:
    action = _action()
    action["data"]["resource_timeline"]["samples"][0]["cpu_quota_cores"] = 4

    with pytest.raises(ValueError, match="not collected at eight cores"):
        feedback_action_row(action)


def test_feedback_caps_counter_quantization_above_cpu_opportunity() -> None:
    action = _action()
    action["data"]["resource_timeline"]["samples"][0]["cpu_core_s"] = 4.01

    row, reason = feedback_action_row(action)

    assert reason == "eligible"
    assert row["clipped_samples"] == 1
    assert row["clipped_cpu_core_s"] == pytest.approx(0.01)
    assert row["raw_timeline_cpu_core_s"] - row["timeline_cpu_core_s"] == pytest.approx(0.01)
