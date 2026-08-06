import pytest

from scripts.evaluation.evaluate_continuous_latency import (
    CallTiming,
    ClauseTiming,
    DynamicModel,
    _align_timing,
    _clause_prediction,
    _command_prediction,
    _condition_values,
    _status,
    _stratified_empirical_draws,
    _update_times,
)


def _draws(command: str, values: tuple[tuple[float, ...], ...]):
    return tuple(
        _stratified_empirical_draws(
            row,
            f"{command}\0latency_ms\0{index}\0repo\0exact_clause",
        )
        for index, row in enumerate(values)
    )


def test_empirical_survival_updates_inside_a_bucket_and_is_strict_at_edges() -> None:
    pmf, fallback = _condition_values((400.0, 800.0, 1800.0, 5000.0), 1000.0)

    assert fallback is False
    assert pmf == pytest.approx((0.0, 0.5, 0.5, 0.0, 0.0))
    edge, fallback = _condition_values((500.0,), 500.0)
    assert fallback is True
    assert edge is None


def test_command_survival_can_skip_buckets_and_exhaust_to_unavailable() -> None:
    model = DynamicModel(
        "slow",
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (9000.0, 10_000.0, 20_000.0),
        (),
        (),
        None,
    )

    hard, pmf, exhausted = _command_prediction(model, 600.0)
    assert hard == 3
    assert pmf == (0.0, 0.0, 0.0, 1.0, 0.0)
    assert exhausted is False
    assert _command_prediction(model, 20_000.0) == (None, None, True)


def test_clause_alignment_preserves_event_clock_when_identity_is_ambiguous() -> None:
    timing = _align_timing(
        "echo x; echo x",
        10.0,
        1000.0,
        [
            {
                "bin": "echo",
                "argv": ["echo", "x"],
                "in_pipe": False,
                "pipeline_position": -1,
                "ts_start": 10.1,
                "ts_end": 10.2,
            }
        ],
    )

    assert timing.clause_usable is True
    assert timing.fallback_reason is None
    assert timing.clauses[0].static_index is None
    assert timing.clauses[0].fallback_reason == "causal_alignment_ambiguous"
    assert timing.clauses[0].start_ms == pytest.approx(100.0)
    assert timing.clauses[0].end_ms == pytest.approx(200.0)
    assert _update_times(timing) == pytest.approx([0.0, 100.0, 200.0, 500.0])


def test_future_ambiguous_clause_does_not_trigger_early_fallback() -> None:
    command = "true; echo x; echo x"
    values = ((50.0,), (100.0,), (100.0,))
    model = DynamicModel(
        command,
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (250.0,),
        values,
        _draws(command, values),
        ((0,), (1,), (2,)),
    )
    timing = _align_timing(
        command,
        10.0,
        500.0,
        [
            {
                "bin": "true",
                "argv": ["true"],
                "in_pipe": False,
                "pipeline_position": -1,
                "ts_start": 10.01,
                "ts_end": 10.05,
            },
            {
                "bin": "echo",
                "argv": ["echo", "x"],
                "in_pipe": False,
                "pipeline_position": -1,
                "ts_start": 10.1,
                "ts_end": 10.2,
            },
        ],
    )

    _hard, _pmf, before = _clause_prediction(model, timing, 75.0)
    _hard, _pmf, after = _clause_prediction(model, timing, 125.0)

    assert before["fallback_to_command"] is False
    assert before["completed"] == 1
    assert after["fallback_to_command"] is True
    assert after["fallback_reason"] == "causal_alignment_ambiguous"


def test_sequential_clause_update_uses_completed_active_and_future_work() -> None:
    command = "first; second"
    values = ((700.0,), (500.0, 1500.0, 5000.0))
    model = DynamicModel(
        command,
        (0.0, 1.0, 0.0, 0.0, 0.0),
        (1200.0, 2200.0, 5700.0),
        values,
        _draws(command, values),
        ((0,), (1,)),
    )
    timing = CallTiming(
        6000.0,
        (
            ClauseTiming(0, 0.0, 700.0),
            ClauseTiming(1, 700.0, 5700.0),
        ),
        True,
    )

    hard, pmf, state = _clause_prediction(model, timing, 1700.0)

    assert hard == 2
    assert pmf == pytest.approx((0.0, 0.0, 1.0, 0.0, 0.0))
    assert state["completed"] == 1
    assert state["active"] == 1
    assert state["fallback_to_command"] is False


def test_active_clause_prediction_cannot_read_its_future_end() -> None:
    command = "first; second"
    values = ((700.0,), (500.0, 1500.0, 5000.0))
    model = DynamicModel(
        command,
        (0.0, 1.0, 0.0, 0.0, 0.0),
        (1200.0, 2200.0, 5700.0),
        values,
        _draws(command, values),
        ((0,), (1,)),
    )
    early_end = CallTiming(
        6000.0,
        (ClauseTiming(0, 0.0, 700.0), ClauseTiming(1, 700.0, 3000.0)),
        True,
    )
    late_end = CallTiming(
        9000.0,
        (ClauseTiming(0, 0.0, 700.0), ClauseTiming(1, 700.0, 8000.0)),
        True,
    )

    assert _clause_prediction(model, early_end, 1700.0) == _clause_prediction(
        model, late_end, 1700.0
    )


def test_pipeline_clause_update_takes_max_not_sum() -> None:
    command = "left | right"
    values = ((1000.0,), (1000.0, 3000.0, 5000.0))
    model = DynamicModel(
        command,
        (0.0, 0.0, 1.0, 0.0, 0.0),
        (3000.0, 5000.0),
        values,
        _draws(command, values),
        ((0, 1),),
    )
    timing = CallTiming(
        5500.0,
        (ClauseTiming(0, 0.0, 1000.0), ClauseTiming(1, 0.0, 5000.0)),
        True,
    )

    hard, pmf, _state = _clause_prediction(model, timing, 2000.0)

    assert hard == 2
    assert pmf == pytest.approx((0.0, 0.0, 1.0, 0.0, 0.0))


def test_top_level_status_depends_only_on_empirical_command_survival() -> None:
    assert _status(True) == "development_empirical_command_survival_go"
    assert _status(False) == "development_empirical_command_survival_no_go"
