from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.evaluate_clause_latency_buckets import (
    ScoredRow,
    _argmax_bucket,
    _exact_bucket_metrics,
    _parser,
    _telemetry_metrics,
    _telemetry_scored_rows,
    _validate_partition,
    evaluate_clause_telemetry,
)
from scripts.evaluation.evaluate_clause_resource_classes import load_rows


def _scored(label_bucket: int, probability_by_bucket: tuple[float, ...]) -> ScoredRow:
    return ScoredRow(
        sample_id="s",
        task_id="owner__repo-1",
        repo="owner__repo",
        command="a",
        label_bucket=label_bucket,
        probability_by_bucket=probability_by_bucket,
        layer="repo",
        key_kind="bin",
        evidence_count=1,
        fallback_path=("repo:bin",),
        unavailable_reason=None,
        mapping_evidence="canonical_clause_telemetry",
    )


def test_partition_overlap_fails_closed() -> None:
    with pytest.raises(ValueError, match="fit/eval task overlap"):
        _validate_partition(["owner__repo-1"], ["owner__repo-1"])


def test_exact_bucket_metrics_use_lowest_argmax_on_ties() -> None:
    # A tie between buckets 0 and 1 must resolve to the lower bucket id, so the
    # row labelled 1 is scored wrong and the row labelled 2 is scored right.
    tie = _scored(1, (0.5, 0.5, 0.0))
    hit = _scored(2, (0.0, 0.0, 1.0))

    assert _argmax_bucket(tie) == 0
    assert _exact_bucket_metrics([tie, hit]) == {
        "three_class_accuracy": 0.5,
        "eligible_examples": 2,
    }


def test_cli_has_no_bucket_override() -> None:
    # Bucket edges come from the canonical objective, never from the CLI.
    with pytest.raises(SystemExit):
        _parser().parse_args(["--bucket-edges-ms", "100,1000"])


def test_cli_requires_both_telemetry_corpora() -> None:
    # Clause telemetry is the only evidence source; neither side is optional.
    with pytest.raises(SystemExit):
        _parser().parse_args([])
    with pytest.raises(SystemExit):
        _parser().parse_args(["--telemetry-fit", "fit.jsonl"])
    args = _parser().parse_args(
        ["--telemetry-fit", "fit.jsonl", "--telemetry-eval", "eval.jsonl"]
    )
    assert (args.telemetry_fit.name, args.telemetry_eval.name) == (
        "fit.jsonl",
        "eval.jsonl",
    )


def _telemetry_record(
    task_id: str,
    manifest_index: int,
    clauses: tuple[tuple[str, tuple[str, ...], float], ...],
) -> str:
    return json.dumps(
        {
            "data": {
                "task_instance_id": task_id,
                "manifest_index": manifest_index,
                "clause_telemetry": {
                    "eligible_for_kb": True,
                    "clauses": [
                        {
                            "eligible_for_kb": True,
                            "bin": bin_,
                            "argv": list(argv),
                            "latency_ms": latency_ms,
                        }
                        for bin_, argv, latency_ms in clauses
                    ],
                },
            }
        }
    )


def _write_telemetry(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_clause_telemetry_path_is_causal_and_reports_bucket_semantics(
    tmp_path: Path,
) -> None:
    # Public prior says `sleep` is fast; the evaluated repo is actually slow.
    fit = _write_telemetry(
        tmp_path / "fit.jsonl",
        [
            _telemetry_record(
                "owner__fitrepo-1", 0, (("sleep", ("sleep", "0"), 100.0),) * 3
            )
        ],
    )
    evaluation = _write_telemetry(
        tmp_path / "eval.jsonl",
        [
            # Two identical clauses inside ONE task: neither may learn from the other.
            _telemetry_record(
                "owner__evalrepo-1", 0, (("sleep", ("sleep", "9"), 3000.0),) * 2
            ),
            # A later task may use what the earlier task settled.
            _telemetry_record(
                "owner__evalrepo-2",
                1,
                (("sleep", ("sleep", "9"), 3000.0), ("sleep", ("sleep", "1"), 500.0)),
            ),
        ],
    )
    rows = _telemetry_scored_rows(load_rows(fit), load_rows(evaluation))

    assert len(rows) == 4
    first_task = [row for row in rows if row.task_id == "owner__evalrepo-1"]
    # No intra-task leakage: both clauses still fall back to the public prior.
    assert [row.layer for row in first_task] == ["public", "public"]
    assert [_argmax_bucket(row) for row in first_task] == [0, 0]
    assert [row.label_bucket for row in first_task] == [1, 1]

    second_task = {row.command: row for row in rows if row.task_id.endswith("-2")}
    learned = second_task["sleep 9"]
    # The earlier task settled before this one queried, so repo evidence wins.
    assert learned.layer == "repo"
    assert _argmax_bucket(learned) == 1
    assert learned.evidence_count == 2
    # 500 ms is below the first edge and belongs to the short bucket.
    assert second_task["sleep 1"].label_bucket == 0

    metrics = _telemetry_metrics(rows)
    assert metrics["eligible_examples"] == 4
    assert metrics["three_class_accuracy"] == 0.25
    assert metrics["majority_class"] == "middle"
    assert metrics["majority_class_id"] == 1
    assert metrics["majority_class_accuracy"] == 0.75
    assert metrics["accuracy_minus_majority_percentage_points"] == -50.0
    assert metrics["accuracy_minus_current_percentage_points"] == 0.0
    assert metrics["prediction_unavailable"] == 0
    assert sum(sum(row) for row in metrics["confusion_label_by_prediction"]) == 4
    assert metrics["confusion_label_by_prediction"][1][0] == 2
    assert metrics["per_class"][1] == {
        "class": "middle",
        "class_id": 1,
        "label_count": 3,
        "label_share": 0.75,
        "predicted_count": 2,
        "predicted_share": 0.5,
    }
    assert set(metrics["scope_counts"]) == {"public", "repo"}
    assert sum(metrics["support_band_counts"].values()) == 4
    assert sum(metrics["evidence_count_counts"].values()) == 4
    assert metrics["fallback_path_counts"]

    result, result_rows = evaluate_clause_telemetry(
        load_rows(fit),
        load_rows(evaluation),
        {"fixture": True},
    )
    assert result_rows == rows
    assert result["row_identity"] == {
        "identical_row_ids_and_labels": True,
        "eligible_row_count": 4,
    }
    assert set(result["baselines"]) == {
        "majority",
        "current",
        "public_only",
        "local_only_diagnostic",
    }
    assert set(result["oracles"]) == {"current_public", "current_nodes"}
    current = result["baselines"]["current"]["three_class_accuracy"]
    public = result["baselines"]["public_only"]["three_class_accuracy"]
    assert current is not None and public is not None
    assert result["oracles"]["current_public"]["three_class_accuracy"] >= max(
        current,
        public,
    )
    assert result["oracles"]["current_nodes"]["three_class_accuracy"] >= result[
        "oracles"
    ]["current_public"]["three_class_accuracy"]
    assert result["oracles"]["current_nodes"]["oracle"] is True
    local = result["baselines"]["local_only_diagnostic"]
    assert local["selection_forbidden"] is True
    assert local["prediction_coverage"] == 0.5
    assert local["three_class_accuracy"] is None


def test_clause_telemetry_path_backs_off_to_global_and_reports_the_path(
    tmp_path: Path,
) -> None:
    fit = _write_telemetry(
        tmp_path / "fit.jsonl",
        [_telemetry_record("owner__fitrepo-1", 0, (("sleep", ("sleep", "0"), 10.0),))],
    )
    evaluation = _write_telemetry(
        tmp_path / "eval.jsonl",
        [_telemetry_record("owner__evalrepo-1", 0, (("pytest", ("pytest",), 10.0),))],
    )
    # An unseen bin backs off to the pooled global node - real fit evidence, not a
    # synthesized PMF - and the path it took stays visible in the row.
    (row,) = _telemetry_scored_rows(load_rows(fit), load_rows(evaluation))

    assert (row.layer, row.key_kind) == ("public", "global")
    assert row.fallback_path == ("repo:exact_clause", "repo:bin", "public:bin", "public:global")
    assert row.evidence_count == 1


def test_clause_telemetry_path_fails_when_the_fit_corpus_has_no_latency(
    tmp_path: Path,
) -> None:
    fit = tmp_path / "fit.jsonl"
    fit.write_text(
        json.dumps(
            {
                "data": {
                    "task_instance_id": "owner__fitrepo-1",
                    "manifest_index": 0,
                    "clause_telemetry": {
                        "eligible_for_kb": True,
                        "clauses": [
                            {
                                "eligible_for_kb": True,
                                "bin": "sleep",
                                "argv": ["sleep"],
                                "latency_ms": None,
                            }
                        ],
                    },
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation = _write_telemetry(
        tmp_path / "eval.jsonl",
        [_telemetry_record("owner__evalrepo-1", 0, (("sleep", ("sleep",), 10.0),))],
    )
    # No usable fit evidence must stop the run, never produce a default PMF.
    with pytest.raises(ValueError, match="no eligible clause telemetry rows"):
        _telemetry_scored_rows(load_rows(fit), load_rows(evaluation))


def test_clause_telemetry_path_rejects_a_corpus_with_no_eligible_clauses(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text(
        json.dumps({"data": {"clause_telemetry": {"eligible_for_kb": False}}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no eligible clause telemetry rows"):
        load_rows(empty)
