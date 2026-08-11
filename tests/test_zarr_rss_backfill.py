from scripts.evaluation.evaluate_zarr_rss_backfill import (
    Dataset,
    RSS_CAPACITY_MB,
    _composed_rss_source,
    _phase_raises,
    _upper_index,
    _validated_rss_source,
)
from scripts.evaluation.evaluate_clause_resource_classes import CommandRow, Row
from scripts.evaluation.evaluate_zarr_rss_evidence_gate import (
    _candidate_reservations,
    _guarded_reservation,
)


def _clause(rss, latency, *, pipeline_position=-1):
    return {
        "sampled_peak_rss_mb": rss,
        "latency_ms": latency,
        "in_pipe": pipeline_position >= 0,
        "in_subst": False,
        "pipeline_position": pipeline_position,
    }


def test_zarr_rss_source_composition_and_upper_reservation() -> None:
    assert _composed_rss_source(
        {
            "eligible_for_kb": True,
            "clauses": [_clause(None, 10), _clause(700, 1_000)],
        }
    ) == (700.0, "clause_short_null_upper")
    assert _composed_rss_source(
        {
            "eligible_for_kb": True,
            "clauses": [
                _clause(300, 1_000, pipeline_position=0),
                _clause(400, 1_000, pipeline_position=1),
            ],
        }
    ) == (700.0, "observed_clause_composition")
    assert _composed_rss_source(
        {"eligible_for_kb": True, "clauses": [_clause(None, 500)]}
    ) == (
        RSS_CAPACITY_MB,
        "full_fallback",
    )
    assert _upper_index((0.75, 0.25, 0.0)) == 1
    assert _phase_raises(1, 0, 0)
    assert not _phase_raises(1, 0, None)
    assert not _phase_raises(1, 0, 2)
    assert _validated_rss_source(
        {"eligible_for_kb": False, "clauses": [_clause(10, 1_000)]}
    ) == (RSS_CAPACITY_MB, "full_fallback")


def test_exact_support_gate_requires_two_tasks_and_covers_history() -> None:
    assert _guarded_reservation(500.0, {"task-a": 0}) == RSS_CAPACITY_MB
    assert _guarded_reservation(500.0, {"task-a": 0, "task-b": 0}) == 500.0
    assert (
        _guarded_reservation(500.0, {"task-a": 0, "task-b": 1})
        == RSS_CAPACITY_MB
    )
    assert _guarded_reservation(2_000.0, {"task-a": 0, "task-b": 1}) == 2_000.0
    assert _validated_rss_source(
        {
            "eligible_for_kb": True,
            "invalid_reasons": ["lossy_mapping"],
            "clauses": [_clause(10, 1_000)],
        }
    ) == (RSS_CAPACITY_MB, "full_fallback")


def test_exact_supported_demotion_uses_loto_and_independent_tasks() -> None:
    def command(task_id: str, call_id: str, text: str, rss: float) -> CommandRow:
        clause = Row(
            task_id,
            "zarr-developers/zarr-python",
            0,
            "tool",
            ("tool",),
            1_000.0,
            1.0,
            rss,
            0.0,
        )
        return CommandRow(
            task_id, clause.repo, 0, 0, call_id, text, 1_000.0, (clause,)
        )

    fit_rows = {
        "fit-a": (
            command("fit-a", "a1", "same", 100.0),
            command("fit-a", "a2", "same", 100.0),
            command("fit-a", "a3", "one-task", 100.0),
            command("fit-a", "a4", "one-task", 100.0),
            command("fit-a", "a5", "no-raise", 100.0),
        ),
        "fit-b": (
            command("fit-b", "b1", "same", 100.0),
            command("fit-b", "b2", "no-raise", 100.0),
        ),
        "target": (
            command("target", "past1", "one-task", 100.0),
            command("target", "past2", "same", 3_000.0),
        ),
    }
    target_rows = (
        command("target", "same", "same", 100.0),
        command("target", "one", "one-task", 100.0),
        command("target", "raise", "no-raise", 100.0),
    )
    empty = Dataset((), {}, {}, {}, {}, {}, {}, frozenset())
    fit = Dataset(
        tuple(fit_rows),
        empty.programs,
        empty.profiles,
        empty.clauses_by_task,
        fit_rows,
        empty.events_by_task,
        empty.rss_source_by_command,
        empty.unverified_command_ids,
    )
    target = Dataset(
        ("target",),
        empty.programs,
        {
            **{f"target:{row.call_id}": () for row in target_rows},
            "target:unmatched": (),
        },
        empty.clauses_by_task,
        {"target": target_rows},
        empty.events_by_task,
        empty.rss_source_by_command,
        empty.unverified_command_ids,
    )
    candidate, diagnostics = _candidate_reservations(
        fit,
        target,
        {
            "clause_kb": {
                "target:same": 2_000.0,
                "target:one": 500.0,
                "target:raise": 500.0,
            },
            "task_aware_upper": {
                "target:same": 500.0,
                "target:one": 500.0,
                "target:raise": RSS_CAPACITY_MB,
            },
        },
        leave_one_out=True,
    )

    assert candidate == {
        "target:same": 500.0,
        "target:one": RSS_CAPACITY_MB,
        "target:raise": 500.0,
        "target:unmatched": RSS_CAPACITY_MB,
    }
    assert diagnostics["authorized_demotions"] == 1
    assert diagnostics["authorized_demotion_command_counts"] == {"same": 1}
    assert diagnostics["blocked_low"] == 1
