from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts.evaluation.evaluate_clause_latency_buckets import (
    CANONICAL_RESOURCE_BUCKET_EDGES,
    CommandRow,
    PipExecEvent,
    RESOURCE_BUCKET_LABELS,
    Row,
)
from scripts.evaluation.evaluate_full_test_phase_fresh import (
    EVALUATION_SCHEMA,
    _result_gate,
    _validation_authorizes_final,
    apply_phase_candidate,
    evaluate,
    fit_phase_pmfs,
    phase_coverage,
)


def _task(
    task_id: str, manifest_index: int
) -> tuple[list[CommandRow], list[PipExecEvent]]:
    commands = []
    events = []
    for call_index in range(3):
        call_id = f"{task_id}:{call_index}"
        clause = Row(
            task_id=task_id,
            repo="tobymao__sqlglot",
            manifest_index=manifest_index,
            bin="python3",
            argv=("python3", "-m", "pytest"),
            latency_ms=40_000.0,
            peak_cpu_cores=6.0,
            sampled_peak_rss_mb=3_000.0,
            disk_read_write_bytes_total=0.0,
        )
        commands.append(
            CommandRow(
                task_id,
                clause.repo,
                manifest_index,
                call_index,
                call_id,
                "python3 -m pytest",
                clause.latency_ms,
                (clause,),
            )
        )
        events.append(PipExecEvent(call_id, "python3 -m pytest", ""))
    return commands, events


def test_fresh_phase_fit_coverage_and_monotone_application() -> None:
    fit_ids = [f"fit-{index}" for index in range(5)]
    fit_commands = []
    fit_events = {}
    for index, task_id in enumerate(fit_ids):
        commands, events = _task(task_id, index)
        fit_commands.extend(commands)
        fit_events[task_id] = events

    pmfs, support = fit_phase_pmfs(fit_ids, fit_commands, fit_events)
    assert support["phase_commands"] == support["phase_tasks"] == 5
    assert all(pmf[-1] == 1.0 for pmf in pmfs.values())

    role_ids = [f"role-{index}" for index in range(3)]
    role_commands = []
    role_events = {}
    for index, task_id in enumerate(role_ids):
        commands, events = _task(task_id, index)
        role_commands.extend(commands)
        role_events[task_id] = events
    assert phase_coverage(role_ids, role_events)["passed"]
    assert not phase_coverage(role_ids[:2], dict(list(role_events.items())[:2]))[
        "passed"
    ]

    baseline = []
    for command in role_commands:
        probability_by_bucket = {
            "latency": [1.0, 0.0, 0.0, 0.0, 0.0],
            **{target: [1.0, 0.0, 0.0] for target in CANONICAL_RESOURCE_BUCKET_EDGES},
        }
        baseline.append(
            {
                "sample_id": f"{command.task_id}:{command.call_index}",
                "task_id": command.task_id,
                "latency_label": 4,
                "resource_labels": {
                    target: 0 for target in CANONICAL_RESOURCE_BUCKET_EDGES
                },
                "current_dynamic": {
                    "latency": 0,
                    **{
                        target: RESOURCE_BUCKET_LABELS[0]
                        for target in CANONICAL_RESOURCE_BUCKET_EDGES
                    },
                    "probability_by_bucket": probability_by_bucket,
                },
            }
        )

    already_high = baseline[-1]["current_dynamic"]
    already_high["latency"] = 4
    already_high["peak_cpu_cores"] = RESOURCE_BUCKET_LABELS[2]
    already_high["sampled_peak_rss_mb"] = RESOURCE_BUCKET_LABELS[2]
    already_high["probability_by_bucket"]["latency"] = [0.0, 0.0, 0.0, 0.0, 1.0]
    already_high["probability_by_bucket"]["peak_cpu_cores"] = [0.0, 0.0, 1.0]
    already_high["probability_by_bucket"]["sampled_peak_rss_mb"] = [0.0, 0.0, 1.0]
    baseline[0]["current_dynamic"].pop("peak_cpu_cores")
    baseline[0]["current_dynamic"]["probability_by_bucket"].pop("peak_cpu_cores")

    rows = apply_phase_candidate(baseline, role_commands, role_events, pmfs)
    assert rows[0]["candidate"]["peak_cpu_cores"] is None
    assert rows[0]["candidate_probability_by_bucket"]["peak_cpu_cores"] is None
    unchanged = [row for row in rows if row["full_test_phase"] != 2]
    changed = [row for row in rows if row["full_test_phase"] == 2]
    assert all(row["candidate"] == row["current_dynamic"] for row in unchanged)
    assert all(
        row["phase_applied_targets"]
        == ["latency", "peak_cpu_cores", "sampled_peak_rss_mb"]
        for row in changed[:-1]
    )
    assert all(
        row["candidate"]["disk_read_write_bytes_total"]
        == row["current_dynamic"]["disk_read_write_bytes_total"]
        and row["candidate_probability_by_bucket"]["disk_read_write_bytes_total"]
        == row["current_probability_by_bucket"]["disk_read_write_bytes_total"]
        for row in rows
    )
    assert rows[-1]["phase_applied_targets"] == []

    mismatched = {task_id: list(events) for task_id, events in role_events.items()}
    mismatched[role_ids[0]][0] = replace(
        mismatched[role_ids[0]][0], command="make test"
    )
    with pytest.raises(ValueError, match="matching raw event"):
        apply_phase_candidate(baseline, role_commands, mismatched, pmfs)

    overlapping = {task_id: list(events) for task_id, events in role_events.items()}
    overlapping[role_ids[0]][0] = replace(
        overlapping[role_ids[0]][0], ts_start=0.0, ts_end=2.0
    )
    overlapping[role_ids[0]][1] = replace(
        overlapping[role_ids[0]][1], ts_start=1.0, ts_end=3.0
    )
    with pytest.raises(ValueError, match="overlap"):
        apply_phase_candidate(baseline, role_commands, overlapping, pmfs)


def test_final_authorization_and_disk_gate_fail_closed() -> None:
    digest = "a" * 64
    authorization = {
        "schema": EVALUATION_SCHEMA,
        "role": "validation",
        "status": "validation_go",
        "claim_bearing": False,
        "artifact_sha256": digest,
        "split_manifest_sha256": digest,
        "labels_scored": True,
        "coverage": {"passed": True},
        "gate": {"go": True},
        "row_identity": {"identical_rows": True},
        "rows_sha256": digest,
    }
    assert _validation_authorizes_final(authorization, digest, digest)
    assert not _validation_authorizes_final(
        {**authorization, "gate": {"go": False}}, digest, digest
    )

    row = {
        "candidate": {"disk_read_write_bytes_total": RESOURCE_BUCKET_LABELS[0]},
        "current_dynamic": {"disk_read_write_bytes_total": RESOURCE_BUCKET_LABELS[0]},
        "candidate_probability_by_bucket": {
            "disk_read_write_bytes_total": [1.0, 0.0, 0.0]
        },
        "current_probability_by_bucket": {
            "disk_read_write_bytes_total": [1.0, 0.0, 0.0]
        },
    }
    assert _result_gate({"gate": {"go": True}}, [row])["go"]
    row["candidate_probability_by_bucket"]["disk_read_write_bytes_total"] = [
        0.0,
        1.0,
        0.0,
    ]
    assert not _result_gate({"gate": {"go": True}}, [row])["go"]


def test_validation_coverage_no_go_does_not_load_role_labels(
    tmp_path, monkeypatch
) -> None:
    import scripts.evaluation.evaluate_full_test_phase_fresh as module

    reserved = tmp_path / "reserved"
    development = tmp_path / "development"
    out_dir = tmp_path / "out"
    role_ids = ["tobymao__sqlglot-a", "tobymao__sqlglot-b"]
    split = {
        "development": ["tobymao__sqlglot-dev"],
        "validation": role_ids,
        "final_test": [],
        "reserved_run": str(reserved),
    }
    development_fit = module._fit_fingerprint(split["development"], [], [])
    load_calls = []

    def load_development_only(*_args, **_kwargs):
        load_calls.append(True)
        if len(load_calls) > 1:
            raise AssertionError("role label loader ran after coverage NO-GO")
        return split["development"], [], []

    monkeypatch.setattr(module, "_committed_file", lambda _path: (b"{}", "c" * 40))
    monkeypatch.setattr(module, "_load_split_manifest", lambda _path: (split, "s" * 64))
    monkeypatch.setattr(
        module,
        "_load_artifact",
        lambda *_args: (
            {
                "development_run": str(development),
                "development_fit_sha256": development_fit,
                "public_inputs": [],
            },
            {},
            "a" * 64,
            "c" * 40,
        ),
    )
    monkeypatch.setattr(module, "load_run_rows", load_development_only)
    monkeypatch.setattr(module, "_frozen_public_inputs", lambda *_args: [])
    monkeypatch.setattr(module, "_reserved_role_complete", lambda *_args: True)
    monkeypatch.setattr(module, "_attempt_records", lambda *_args: ([], {}))
    monkeypatch.setattr(
        module,
        "_telemetry_valid_records",
        lambda *_args: [{"instance_id": task_id} for task_id in role_ids],
    )
    monkeypatch.setattr(module, "_write_result_view", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "_load_exec_events",
        lambda *_args, **_kwargs: {
            task_id: _task(task_id, index)[1] for index, task_id in enumerate(role_ids)
        },
    )

    evaluate(
        SimpleNamespace(
            out_dir=out_dir,
            split_manifest=module.SPLIT_MANIFEST,
            run_dir=reserved,
            artifact_dir=tmp_path / "artifact",
            role="validation",
            validation_result=None,
            development_run=development,
            public_telemetry=[],
        )
    )
    result = json.loads((out_dir / "result.json").read_text())
    assert result["status"] == "validation_coverage_no_go"
    assert result["labels_scored"] is False
    assert len(load_calls) == 1
