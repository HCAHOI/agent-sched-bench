from __future__ import annotations

import pytest

from trace_collect.command_features import segment_prefix_keys, shell_command_segments
from trace_collect.segment_cost_model import fit_segment_cost_model


@pytest.mark.parametrize(
    ("command", "segments"),
    [
        ("cd /x && python a.py", [["cd", "/x"], ["python", "a.py"]]),
        # Pipeline parts run concurrently: one unit, pipe token kept.
        ("find . | head -5", [["find", ".", "|", "head", "-5"]]),
        ("(cd /a && make) | tee log", [["cd", "/a"], ["make", "|", "tee", "log"]]),
        ("a ; b || c", [["a"], ["b"], ["c"]]),
        ("echo 'unbalanced", []),
    ],
)
def test_shell_command_segments(command: str, segments: list[list[str]]) -> None:
    assert shell_command_segments(command) == segments


def test_segment_prefix_keys_nest_and_cap() -> None:
    assert segment_prefix_keys("exec", ["make", "-j2", "all"], max_depth=2) == (
        "exec:make",
        "exec:make -j2",
    )
    assert segment_prefix_keys("exec", [], max_depth=2) == ()
    with pytest.raises(ValueError, match="max_depth must be >= 1"):
        segment_prefix_keys("exec", ["make"], max_depth=0)


def test_fit_recovers_additive_costs_exactly() -> None:
    rows = (
        [_command_row(f"a{i}", "alpha", 100.0, i) for i in range(2)]
        + [_command_row(f"b{i}", "beta", 300.0, 2 + i) for i in range(2)]
        + [_command_row(f"ab{i}", "alpha && beta", 400.0, 4 + i) for i in range(2)]
    )

    model = fit_segment_cost_model(rows, command_field="command")

    assert model.fitted_row_count == 6
    assert model.costs_by_head["exec:alpha"] == pytest.approx(100.0, abs=1e-6)
    assert model.costs_by_head["exec:beta"] == pytest.approx(300.0, abs=1e-6)
    assert model.counts_by_head == {"exec:alpha": 4, "exec:beta": 4}
    assert model.residual_rms_ms == pytest.approx(0.0, abs=1e-6)


def test_fit_learns_zero_cost_preamble() -> None:
    rows = [_command_row(f"pw{i}", "prep && work", 500.0, i) for i in range(2)] + [
        _command_row(f"w{i}", "work", 500.0, 2 + i) for i in range(2)
    ]

    model = fit_segment_cost_model(rows, command_field="command")

    assert model.costs_by_head["exec:prep"] == pytest.approx(0.0, abs=1e-6)
    assert model.costs_by_head["exec:work"] == pytest.approx(500.0, abs=1e-6)


def test_dominant_segment_and_deduction() -> None:
    rows = [_command_row(f"p{i}", "prep", 200.0, i) for i in range(2)] + [
        _command_row(f"pw{i}", "prep && work", 500.0, 2 + i) for i in range(2)
    ]
    model = fit_segment_cost_model(rows, command_field="command")
    assert model.costs_by_head["exec:prep"] == pytest.approx(200.0, abs=1e-6)
    assert model.costs_by_head["exec:work"] == pytest.approx(300.0, abs=1e-6)

    dominant, deduction = model.dominant_segment_and_deduction(
        "exec", "prep && work", min_head_count=1
    )
    assert dominant == ["work"]
    assert deduction == pytest.approx(200.0, abs=1e-6)

    # Single segment: itself, no deduction.
    dominant, deduction = model.dominant_segment_and_deduction(
        "exec", "prep", min_head_count=1
    )
    assert dominant == ["prep"]
    assert deduction == 0.0

    # Unknown heads only: no basis for ranking.
    dominant, deduction = model.dominant_segment_and_deduction(
        "exec", "mystery && enigma", min_head_count=1
    )
    assert dominant is None
    assert deduction == 0.0

    # Evidence gate: work appears only twice; min_head_count=3 excludes it.
    dominant, deduction = model.dominant_segment_and_deduction(
        "exec", "prep && work", min_head_count=3
    )
    assert dominant == ["prep"]
    assert deduction == 0.0


def test_repeated_head_within_one_command_is_additive_but_counted_once() -> None:
    rows = [_command_row(f"h{i}", "hop && hop", 400.0, i) for i in range(3)]

    model = fit_segment_cost_model(rows, command_field="command")

    # Design matrix is additive (2 * c(hop) = 400) ...
    assert model.costs_by_head["exec:hop"] == pytest.approx(200.0, abs=1e-6)
    # ... but the evidence gate counts observing commands, not occurrences.
    assert model.counts_by_head["exec:hop"] == 3


def test_lad_recovers_costs_on_consistent_data() -> None:
    rows = [_command_row(f"p{i}", "prep", 200.0, i) for i in range(2)] + [
        _command_row(f"pw{i}", "prep && work", 500.0, 2 + i) for i in range(2)
    ]

    model = fit_segment_cost_model(rows, command_field="command", fit_method="lad")

    assert model.costs_by_head["exec:prep"] == pytest.approx(200.0, abs=1e-6)
    assert model.costs_by_head["exec:work"] == pytest.approx(300.0, abs=1e-6)
    assert model.residual_rms_ms == pytest.approx(0.0, abs=1e-6)


def test_lad_resists_the_outlier_that_breaks_nnls() -> None:
    rows = (
        [_command_row(f"p{i}", "prep", 200.0, i) for i in range(5)]
        + [_command_row(f"pw{i}", "prep && work", 500.0, 5 + i) for i in range(5)]
        + [_command_row("outlier", "prep", 100_000.0, 10)]
    )

    lad = fit_segment_cost_model(rows, command_field="command", fit_method="lad")
    nnls_fit = fit_segment_cost_model(rows, command_field="command", fit_method="nnls")

    # Median regression pins prep at its typical cost despite the outlier...
    assert lad.costs_by_head["exec:prep"] == pytest.approx(200.0, abs=1e-4)
    assert lad.costs_by_head["exec:work"] == pytest.approx(300.0, abs=1e-4)
    # ...while squared loss lets the single 100s row inflate it massively.
    assert nnls_fit.costs_by_head["exec:prep"] > 1_000.0


def test_fit_rejects_unknown_method() -> None:
    rows = [_command_row("r", "alpha", 100.0, 0)]

    with pytest.raises(ValueError, match="unknown fit_method"):
        fit_segment_cost_model(rows, command_field="command", fit_method="huber")


def test_fit_rejects_rows_without_commands() -> None:
    with pytest.raises(ValueError, match="no command rows"):
        fit_segment_cost_model(
            [{"tool_name": "read_file", "latency_ms": 10.0, "tool_args": {"path": "/x"}}],
            command_field="command",
        )


def _command_row(
    sample_id: str,
    command: str,
    latency_ms: float,
    ts: float,
    *,
    source_trace: str = "trace-p",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source_trace": source_trace,
        "tool_name": "exec",
        "latency_ms": latency_ms,
        "tool_ts_start": float(ts),
        "tool_ts_end": float(ts) + latency_ms / 1000.0,
        "tool_args": {"command": command},
    }
