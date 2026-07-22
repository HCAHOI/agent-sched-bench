from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.exploration.evaluate_tool_latency_qerror import main as evaluate_qerror_main
from scripts.trace.extract_tool_latencies import main as extract_latencies_main
from trace_collect.tool_latency_dataset import (
    MISSING_TOOL_NAME,
    extract_tool_latency_samples,
    read_tool_latency_jsonl,
)
from trace_collect.tool_latency_qerror import evaluate_latency_qerror


def test_extract_tool_latencies_cli_preserves_identity_outcome_and_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = _write_trace_jsonl(
        tmp_path / "trace.jsonl",
        [
            _metadata_record(instance_id="task-123"),
            _tool_exec(
                action_id="tool-bash-1",
                agent_id="agent-a",
                iteration=3,
                ts_start=10.000,
                ts_end=10.125,
                tool_name="bash",
                tool_call_id="call-bash-1",
                success=True,
                duration_ms=777.0,
                instance_id="container-session-a",
            ),
            _tool_exec(
                action_id="tool-read-1",
                agent_id="agent-a",
                iteration=4,
                ts_start=11.250,
                ts_end=11.500,
                tool_name="read",
                tool_call_id="call-read-1",
                success=False,
            ),
        ],
    )
    output_path = tmp_path / "latencies.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        ["extract_tool_latencies.py", str(trace_path), "--output", str(output_path)],
    )

    extract_latencies_main()

    assert "Wrote 2 tool latency samples" in capsys.readouterr().out
    rows = _read_jsonl(output_path)
    assert rows == [
        {
            "action_id": "tool-bash-1",
            "agent_id": "agent-a",
            "instance_id": "container-session-a",
            "iteration": 3,
            "latency_ms": pytest.approx(125.0),
            "reported_duration_ms": 777.0,
            "sample_id": f"{trace_path}:agent-a:3:tool-bash-1",
            "source_trace": str(trace_path),
            "success": True,
            "task_id": "task-123",
            "tool_call_id": "call-bash-1",
            "tool_name": "bash",
            "tool_ts_end": 10.125,
            "tool_ts_start": 10.0,
        },
        {
            "action_id": "tool-read-1",
            "agent_id": "agent-a",
            "instance_id": "task-123",
            "iteration": 4,
            "latency_ms": pytest.approx(250.0),
            "sample_id": f"{trace_path}:agent-a:4:tool-read-1",
            "source_trace": str(trace_path),
            "success": False,
            "task_id": "task-123",
            "tool_call_id": "call-read-1",
            "tool_name": "read",
            "tool_ts_end": 11.5,
            "tool_ts_start": 11.25,
        },
    ]


def test_extract_tool_latencies_preserves_missing_tool_name_as_explicit_class(
    tmp_path: Path,
) -> None:
    trace_path = _write_trace_jsonl(
        tmp_path / "trace.jsonl",
        [
            _metadata_record(instance_id="task-malformed"),
            _tool_exec(
                action_id="tool-empty-1",
                agent_id="agent-a",
                iteration=8,
                ts_start=12.0,
                ts_end=12.0,
                tool_name="",
                tool_call_id="call-empty-1",
                success=False,
                duration_ms=0.0,
            ),
        ],
    )

    samples = extract_tool_latency_samples(trace_path)

    assert len(samples) == 1
    row = samples[0].to_json_obj()
    assert row["tool_name"] == MISSING_TOOL_NAME
    assert row["tool_name_missing"] is True
    assert row["success"] is False
    assert row["latency_ms"] == 0.0
    assert row["reported_duration_ms"] == 0.0


def test_read_tool_latencies_rejects_non_bool_missing_tool_marker(
    tmp_path: Path,
) -> None:
    path = tmp_path / "latencies.jsonl"
    row = _latency_row("missing", MISSING_TOOL_NAME, 0.0, tool_ts_start=1.0)
    row["tool_name_missing"] = "true"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tool_name_missing.*must be a bool"):
        read_tool_latency_jsonl(path)


def test_evaluate_latency_qerror_is_causal_by_tool_timestamp_not_input_order() -> None:
    rows = [
        _latency_row("read-future", "read", 10_000.0, tool_ts_start=40.0),
        _latency_row("read-first", "read", 100.0, tool_ts_start=10.0),
        _latency_row("bash-first", "bash", 50.0, tool_ts_start=30.0),
        _latency_row("read-second", "read", 300.0, tool_ts_start=20.0),
    ]

    summary = evaluate_latency_qerror(rows, quantile=0.5, epsilon_ms=1.0)

    predictions = {row["sample_id"]: row for row in summary["predictions"]}
    assert summary["row_count"] == 4
    assert summary["cold_start_count"] == 1
    assert summary["evaluated_count"] == 3

    assert predictions["read-first"]["tool_name"] == "read"
    assert predictions["read-first"]["latency_ms"] == 100.0
    assert predictions["read-first"]["predicted_latency_ms"] is None
    assert predictions["read-first"]["qerror"] is None
    assert predictions["read-first"]["prediction_source"] == "cold_start"
    assert predictions["read-first"]["history_count"] == 0
    assert predictions["read-second"]["prediction_source"] == "tool_history"
    assert predictions["read-second"]["history_count"] == 1
    assert predictions["read-second"]["predicted_latency_ms"] == 100.0
    assert predictions["read-second"]["qerror"] == pytest.approx(3.0)

    assert predictions["bash-first"]["prediction_source"] == "global_history"
    assert predictions["bash-first"]["history_count"] == 2
    assert predictions["bash-first"]["predicted_latency_ms"] == 200.0
    assert predictions["bash-first"]["qerror"] == pytest.approx(4.0)

    assert predictions["read-future"]["prediction_source"] == "tool_history"
    assert predictions["read-future"]["history_count"] == 2
    assert predictions["read-future"]["predicted_latency_ms"] == 200.0
    assert predictions["read-future"]["qerror"] == pytest.approx(50.0)


def test_evaluate_latency_qerror_scores_same_timestamp_bucket_before_history_update() -> (
    None
):
    rows = [
        _latency_row("read-history", "read", 100.0, tool_ts_start=1.0),
        _latency_row("read-slow-parallel", "read", 10_000.0, tool_ts_start=2.0),
        _latency_row("read-fast-parallel", "read", 10.0, tool_ts_start=2.0),
        _latency_row("write-slow-parallel", "write", 7_000.0, tool_ts_start=2.0),
        _latency_row("write-fast-parallel", "write", 5.0, tool_ts_start=2.0),
    ]

    summary = evaluate_latency_qerror(rows, quantile=1.0, epsilon_ms=1.0)

    predictions = {row["sample_id"]: row for row in summary["predictions"]}
    assert summary["row_count"] == 5
    assert summary["cold_start_count"] == 1
    assert summary["evaluated_count"] == 4

    assert predictions["read-history"]["prediction_source"] == "cold_start"
    assert predictions["read-history"]["history_count"] == 0
    assert predictions["read-history"]["predicted_latency_ms"] is None

    for sample_id, qerror in [
        ("read-slow-parallel", 100.0),
        ("read-fast-parallel", 10.0),
    ]:
        assert predictions[sample_id]["prediction_source"] == "tool_history"
        assert predictions[sample_id]["history_count"] == 1
        assert predictions[sample_id]["predicted_latency_ms"] == 100.0
        assert predictions[sample_id]["qerror"] == pytest.approx(qerror)

    for sample_id, qerror in [
        ("write-slow-parallel", 70.0),
        ("write-fast-parallel", 20.0),
    ]:
        assert predictions[sample_id]["prediction_source"] == "global_history"
        assert predictions[sample_id]["history_count"] == 1
        assert predictions[sample_id]["predicted_latency_ms"] == 100.0
        assert predictions[sample_id]["qerror"] == pytest.approx(qerror)


def test_evaluate_latency_qerror_waits_for_overlapping_tool_completion() -> None:
    rows = [
        _latency_row("global-baseline", "write", 200.0, tool_ts_start=0.0),
        _latency_row("bash-long", "bash", 10_000.0, tool_ts_start=1.0),
        _latency_row("bash-overlap", "bash", 1_000.0, tool_ts_start=2.0),
        _latency_row("read-overlap", "read", 500.0, tool_ts_start=2.0),
        _latency_row("bash-after-long", "bash", 100.0, tool_ts_start=12.0),
        _latency_row("curl-after-long", "curl", 100.0, tool_ts_start=12.0),
    ]

    summary = evaluate_latency_qerror(rows, quantile=1.0, epsilon_ms=1.0)

    predictions = {row["sample_id"]: row for row in summary["predictions"]}
    assert summary["row_count"] == 6
    assert summary["cold_start_count"] == 1
    assert summary["evaluated_count"] == 5

    assert predictions["global-baseline"]["prediction_source"] == "cold_start"
    assert predictions["global-baseline"]["history_count"] == 0

    assert predictions["bash-long"]["prediction_source"] == "global_history"
    assert predictions["bash-long"]["history_count"] == 1
    assert predictions["bash-long"]["predicted_latency_ms"] == 200.0

    for sample_id, qerror in [
        ("bash-overlap", 5.0),
        ("read-overlap", 2.5),
    ]:
        assert predictions[sample_id]["prediction_source"] == "global_history"
        assert predictions[sample_id]["history_count"] == 1
        assert predictions[sample_id]["predicted_latency_ms"] == 200.0
        assert predictions[sample_id]["qerror"] == pytest.approx(qerror)

    for sample_id in ["bash-after-long", "curl-after-long"]:
        assert predictions[sample_id]["predicted_latency_ms"] == 10_000.0
        assert predictions[sample_id]["qerror"] == pytest.approx(100.0)

    assert predictions["bash-after-long"]["prediction_source"] == "tool_history"
    assert predictions["bash-after-long"]["history_count"] == 2
    assert predictions["curl-after-long"]["prediction_source"] == "global_history"
    assert predictions["curl-after-long"]["history_count"] == 4


def test_evaluate_latency_qerror_flushes_source_trace_history_despite_timestamp_reset() -> (
    None
):
    rows = [
        _latency_row(
            "trace-a-long-tool",
            "boundary-probe",
            10_000.0,
            tool_ts_start=100.0,
            source_trace="trace-a",
        ),
        _latency_row(
            "trace-b-first-tool",
            "boundary-probe",
            100.0,
            tool_ts_start=0.0,
            source_trace="trace-b",
        ),
    ]

    summary = evaluate_latency_qerror(rows, quantile=0.5, epsilon_ms=1.0)

    predictions = {row["sample_id"]: row for row in summary["predictions"]}
    assert summary["row_count"] == 2
    assert summary["cold_start_count"] == 1
    assert summary["evaluated_count"] == 1

    assert predictions["trace-a-long-tool"]["prediction_source"] == "cold_start"
    assert predictions["trace-a-long-tool"]["history_count"] == 0
    assert predictions["trace-a-long-tool"]["predicted_latency_ms"] is None
    assert predictions["trace-a-long-tool"]["qerror"] is None

    assert predictions["trace-b-first-tool"]["prediction_source"] == "tool_history"
    assert predictions["trace-b-first-tool"]["history_count"] == 1
    assert predictions["trace-b-first-tool"]["predicted_latency_ms"] == 10_000.0
    assert predictions["trace-b-first-tool"]["qerror"] == pytest.approx(100.0)


def test_evaluate_latency_qerror_floors_near_zero_actual_and_prediction() -> None:
    rows = [
        _latency_row("zero-first", "bash", 0.0, tool_ts_start=1.0),
        _latency_row("positive-second", "bash", 4.0, tool_ts_start=2.0),
        _latency_row("near-zero-third", "bash", 0.5, tool_ts_start=3.0),
    ]

    summary = evaluate_latency_qerror(rows, quantile=1.0, epsilon_ms=2.0)

    predictions = {row["sample_id"]: row for row in summary["predictions"]}
    assert summary["clamped_count"] == 2
    assert predictions["positive-second"]["predicted_latency_ms"] == 0.0
    assert predictions["positive-second"]["qerror"] == pytest.approx(2.0)
    assert predictions["near-zero-third"]["predicted_latency_ms"] == 4.0
    assert predictions["near-zero-third"]["qerror"] == pytest.approx(2.0)


def test_malformed_negative_latency_and_bad_latency_rows_fail_closed(
    tmp_path: Path,
) -> None:
    negative_trace = _write_trace_jsonl(
        tmp_path / "negative-trace.jsonl",
        [
            _metadata_record(instance_id="task-negative"),
            _tool_exec(
                action_id="bad-tool",
                agent_id="agent-a",
                iteration=0,
                ts_start=20.0,
                ts_end=19.999,
                tool_name="bash",
                tool_call_id="call-bad",
                success=False,
            ),
        ],
    )

    with pytest.raises(ValueError, match="ts_end < ts_start"):
        extract_tool_latency_samples(negative_trace)

    with pytest.raises(ValueError, match="latency_ms"):
        evaluate_latency_qerror(
            [
                _latency_row("negative", "bash", -1.0, tool_ts_start=1.0),
            ]
        )

    with pytest.raises(ValueError, match="tool_ts_start"):
        evaluate_latency_qerror(
            [
                {
                    "sample_id": "missing-timestamp",
                    "source_trace": "trace-a",
                    "tool_name": "bash",
                    "latency_ms": 1.0,
                },
            ]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ts_start", float("nan")),
        ("ts_start", float("inf")),
        ("ts_end", float("nan")),
        ("ts_end", float("inf")),
    ],
)
def test_extract_tool_latency_samples_rejects_non_finite_trace_timestamps(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    trace_action = _tool_exec(
        action_id=f"bad-{field}",
        agent_id="agent-a",
        iteration=0,
        ts_start=20.0,
        ts_end=20.125,
        tool_name="bash",
        tool_call_id=f"call-bad-{field}",
        success=False,
    )
    trace_action[field] = value
    trace_path = _write_trace_jsonl(
        tmp_path / f"non-finite-{field}.jsonl",
        [
            _metadata_record(instance_id="task-non-finite"),
            trace_action,
        ],
    )

    with pytest.raises(ValueError, match="non-finite timestamp"):
        extract_tool_latency_samples(trace_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epsilon_ms": float("inf")}, "epsilon_ms must be finite and positive"),
        ({"epsilon_ms": float("nan")}, "epsilon_ms must be finite and positive"),
        ({"quantile": float("nan")}, "quantile must be finite"),
    ],
)
def test_evaluate_latency_qerror_rejects_non_finite_parameters(
    kwargs: dict[str, float],
    message: str,
) -> None:
    rows = [
        _latency_row("first", "bash", 100.0, tool_ts_start=1.0),
        _latency_row("second", "bash", 200.0, tool_ts_start=2.0),
    ]

    with pytest.raises(ValueError, match=message):
        evaluate_latency_qerror(rows, **kwargs)


def test_malformed_rows_missing_source_trace_fail_closed(tmp_path: Path) -> None:
    row = _latency_row("missing-source", "bash", 1.0, tool_ts_start=1.0)
    del row["source_trace"]

    with pytest.raises(ValueError, match="source_trace"):
        evaluate_latency_qerror([row])

    latencies_path = tmp_path / "missing-source.jsonl"
    _write_jsonl(latencies_path, [row])
    with pytest.raises(ValueError, match="source_trace"):
        read_tool_latency_jsonl(latencies_path)


@pytest.mark.parametrize("field", ["source_trace", "sample_id", "tool_name"])
@pytest.mark.parametrize("value", [1.25, float("nan"), float("inf")])
def test_evaluate_latency_qerror_rejects_non_string_identity_fields(
    field: str,
    value: float,
) -> None:
    row = _latency_row("bad-identity", "bash", 1.0, tool_ts_start=1.0)
    row[field] = value

    with pytest.raises(ValueError, match=rf"{field!r} must be a string"):
        evaluate_latency_qerror([row])


@pytest.mark.parametrize("field", ["source_trace", "sample_id", "tool_name", "task_id"])
@pytest.mark.parametrize("value", [1.25, float("nan"), float("inf")])
def test_tool_latency_jsonl_loader_rejects_non_string_identity_fields(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    row = _latency_row("loader-bad-identity", "bash", 1.0, tool_ts_start=1.0)
    row[field] = value
    latencies_path = tmp_path / f"non-string-{field}.jsonl"
    _write_jsonl(latencies_path, [row])

    with pytest.raises(ValueError, match=rf"{field!r} must be a string"):
        read_tool_latency_jsonl(latencies_path)


def test_tool_latency_jsonl_loader_backfills_legacy_task_id(tmp_path: Path) -> None:
    row = _latency_row("legacy", "bash", 1.0, tool_ts_start=1.0)
    latencies_path = tmp_path / "legacy.jsonl"
    _write_jsonl(latencies_path, [row])

    (loaded,) = read_tool_latency_jsonl(latencies_path)

    assert loaded["task_id"] == row["source_trace"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latency_ms", float("nan")),
        ("latency_ms", float("inf")),
        ("tool_ts_start", float("nan")),
        ("tool_ts_start", float("-inf")),
    ],
)
def test_evaluate_latency_qerror_rejects_non_finite_numeric_rows(
    field: str,
    value: float,
) -> None:
    row = _latency_row("non-finite", "bash", 1.0, tool_ts_start=1.0)
    row[field] = value

    with pytest.raises(ValueError, match=rf"{field!r} must be finite"):
        evaluate_latency_qerror([row])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latency_ms", float("nan")),
        ("latency_ms", float("inf")),
        ("tool_ts_start", float("nan")),
        ("tool_ts_start", float("-inf")),
        ("tool_ts_end", float("inf")),
    ],
)
def test_tool_latency_jsonl_loader_rejects_non_finite_numeric_rows(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    row = _latency_row("loader-non-finite", "bash", 1.0, tool_ts_start=1.0)
    row[field] = value
    latencies_path = tmp_path / "non-finite.jsonl"
    _write_jsonl(latencies_path, [row])

    with pytest.raises(ValueError, match=rf"{field!r} must be finite"):
        read_tool_latency_jsonl(latencies_path)


def test_evaluate_tool_latency_qerror_cli_writes_summary_and_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    latencies_path = tmp_path / "latencies.jsonl"
    summary_path = tmp_path / "summary.json"
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        latencies_path,
        [
            _latency_row("read-first", "read", 100.0, tool_ts_start=1.0),
            _latency_row("read-second", "read", 300.0, tool_ts_start=2.0),
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_tool_latency_qerror.py",
            "--latencies",
            str(latencies_path),
            "--quantile",
            "0.5",
            "--epsilon-ms",
            "1.0",
            "--output",
            str(summary_path),
            "--predictions-output",
            str(predictions_path),
        ],
    )

    evaluate_qerror_main()

    assert f"Evaluated 1 / 2 rows -> {summary_path}" in capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = _read_jsonl(predictions_path)
    assert "predictions" not in summary
    assert summary["row_count"] == 2
    assert summary["evaluated_count"] == 1
    assert summary["cold_start_count"] == 1
    assert predictions[0]["sample_id"] == "read-first"
    assert predictions[0]["prediction_source"] == "cold_start"
    assert predictions[1]["sample_id"] == "read-second"
    assert predictions[1]["predicted_latency_ms"] == 100.0
    assert predictions[1]["qerror"] == pytest.approx(3.0)


def _metadata_record(*, instance_id: str) -> dict[str, object]:
    return {
        "type": "trace_metadata",
        "trace_format_version": 5,
        "scaffold": "openclaw",
        "instance_id": instance_id,
        "model": "fixture-model",
    }


def _tool_exec(
    *,
    action_id: str,
    agent_id: str,
    iteration: int,
    ts_start: float,
    ts_end: float,
    tool_name: str,
    tool_call_id: str,
    success: bool,
    duration_ms: float | None = None,
    instance_id: str | None = None,
) -> dict[str, object]:
    data: dict[str, object] = {
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "success": success,
    }
    if duration_ms is not None:
        data["duration_ms"] = duration_ms
    action: dict[str, object] = {
        "type": "action",
        "action_type": "tool_exec",
        "action_id": action_id,
        "agent_id": agent_id,
        "iteration": iteration,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "data": data,
    }
    if instance_id is not None:
        action["instance_id"] = instance_id
    return action


def _latency_row(
    sample_id: str,
    tool_name: str,
    latency_ms: float,
    *,
    tool_ts_start: float,
    source_trace: str = "trace-a",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source_trace": source_trace,
        "tool_name": tool_name,
        "latency_ms": latency_ms,
        "tool_ts_start": tool_ts_start,
        "tool_ts_end": tool_ts_start + latency_ms / 1000.0,
    }


def _write_trace_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(path, records)
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
