from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.evaluate_clause_latency_buckets import (
    ScoredRow,
    _argmax_bucket,
    _bounded_node_oracle_candidates,
    _exact_bucket_metrics,
    _parser,
    _select_shrinkage_alpha,
    _telemetry_metrics,
    _telemetry_scored_rows,
    _validate_partition,
    evaluate_clause_telemetry,
)
from scripts.evaluation.evaluate_clause_resource_classes import (
    CandidateSSelection,
    Row,
    evaluate as evaluate_resources,
    load_candidate_s_selection,
    load_rows,
)
from tool_resource.runtime_kb import ClauseLatencyBucketPrediction


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
        canonicalizer_version="raw-argv-prefix-v1",
        arbitration="hard-first-nonempty-v1",
        local_key_kind=None,
        local_evidence_count=0,
        public_key_kind=None,
        public_evidence_count=0,
        shrinkage_alpha=None,
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


def test_node_oracle_excludes_raw_prefix_candidates() -> None:
    prefix = ClauseLatencyBucketPrediction(
        probability_by_bucket=(1.0, 0.0, 0.0),
        scope="repo",
        key_kind="argv_prefix_depth_3",
        evidence_count=1,
        fallback_path=("repo:exact_clause", "repo:argv_prefix_depth_3"),
        canonicalizer_version="raw-argv-prefix-v1",
    )
    exact = ClauseLatencyBucketPrediction(
        probability_by_bucket=(0.0, 1.0, 0.0),
        scope="repo",
        key_kind="exact_clause",
        evidence_count=1,
        fallback_path=("repo:exact_clause",),
        canonicalizer_version="raw-argv-prefix-v1",
    )

    assert _bounded_node_oracle_candidates((prefix, exact)) == (exact,)


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
    assert args.candidate_s is False
    candidate_args = _parser().parse_args(
        [
            "--telemetry-fit",
            "fit.jsonl",
            "--telemetry-eval",
            "eval.jsonl",
            "--candidate-s",
        ]
    )
    assert candidate_args.candidate_s is True


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
    assert set(result["candidates"]) == {
        "candidate_r_public_only",
        "candidate_r",
    }
    assert set(result["oracles"]) == {
        "current_public",
        "current_and_candidate_nodes",
    }
    current = result["baselines"]["current"]["three_class_accuracy"]
    public = result["baselines"]["public_only"]["three_class_accuracy"]
    assert current is not None and public is not None
    assert result["oracles"]["current_public"]["three_class_accuracy"] >= max(
        current,
        public,
    )
    assert result["oracles"]["current_and_candidate_nodes"][
        "three_class_accuracy"
    ] >= result["oracles"]["current_public"]["three_class_accuracy"]
    assert result["oracles"]["current_and_candidate_nodes"]["oracle"] is True
    candidate = result["candidates"]["candidate_r"]
    assert candidate["prediction_unavailable"] == 0
    assert candidate["canonicalizer_version_counts"] == {
        "generic-argv-v3-role": 4
    }
    bootstrap = result["uncertainty"]["candidate_r_vs_best_baseline"]
    assert bootstrap["cluster_unit"] == "repository"
    assert bootstrap["draws"] == 2000
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


def test_candidate_s_alpha_selection_groups_repos_and_breaks_ties_larger() -> None:
    fit = [
        Row(
            task_id=f"owner__fit-{index}",
            repo=f"owner__fit-{index}",
            manifest_index=index,
            bin="runner",
            argv=("runner", "deploy", f"--path=/tmp/{index}"),
            latency_ms=100.0,
            peak_cpu_cores=None,
            sampled_peak_rss_mb=None,
            disk_read_write_bytes_total=None,
        )
        for index in range(5)
    ]

    selection = _select_shrinkage_alpha(fit)

    assert selection["fold_count"] == 5
    assert selection["fit_repo_count"] == 5
    assert selection["selected_alpha"] == 64.0
    assert len(set(selection["accuracy_by_alpha"].values())) == 1
    assert all(fold["validation_repo_count"] == 1 for fold in selection["folds"])
    assert selection["outer_labels_used"] is False


def test_candidate_s_result_uses_fit_selected_alpha_on_identical_outer_rows() -> None:
    fit = [
        Row(
            task_id=f"owner__fit-{index}",
            repo=f"owner__fit-{index}",
            manifest_index=index,
            bin="runner",
            argv=("runner", "deploy", f"--path=/tmp/{index}"),
            latency_ms=100.0,
            peak_cpu_cores=None,
            sampled_peak_rss_mb=None,
            disk_read_write_bytes_total=None,
        )
        for index in range(5)
    ]
    evaluation = [
        Row(
            task_id=f"owner__eval-{index}",
            repo="owner__eval",
            manifest_index=index,
            bin="runner",
            argv=("runner", "deploy", "--path=/tmp/eval"),
            latency_ms=3000.0,
            peak_cpu_cores=None,
            sampled_peak_rss_mb=None,
            disk_read_write_bytes_total=None,
        )
        for index in range(2)
    ]

    result, rows = evaluate_clause_telemetry(
        fit,
        evaluation,
        {"fixture": True},
        include_candidate_s=True,
    )

    assert len(rows) == 2
    assert result["row_identity"]["identical_row_ids_and_labels"] is True
    assert result["selection"]["candidate_s_alpha"]["selected_alpha"] == 64.0
    candidate = result["candidates"]["candidate_s"]
    assert candidate["eligible_examples"] == 2
    assert candidate["prediction_unavailable"] == 0
    assert candidate["shrinkage_alpha_counts"] == {"64.0": 2}
    assert candidate["arbitration_counts"] == {"public-local-posterior-v1": 2}
    assert result["uncertainty"]["candidate_s_vs_best_baseline"][
        "statistic"
    ].startswith("candidate_s_accuracy")


def test_resource_candidate_s_selection_verifies_latency_result_inputs(
    tmp_path: Path,
) -> None:
    fit = tmp_path / "fit.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    other = tmp_path / "other.jsonl"
    for path in (fit, evaluation, other):
        path.write_text("", encoding="utf-8")
    result_path = tmp_path / "latency.json"
    result = {
        "bucket_edges_ms": [2000.0, 8000.0],
        "fit_clause_observation_count": 5,
        "eval_clause_observation_count": 2,
        "row_identity": {"identical_row_ids_and_labels": True},
        "selection": {
            "candidate_s_alpha": {
                "alpha_grid": [1.0, 4.0, 16.0, 64.0],
                "selected_alpha": 16.0,
                "selection_target": "three_class_latency_accuracy",
                "tie_break": "larger_alpha",
                "fit_row_count": 5,
                "outer_labels_used": False,
            }
        },
        "provenance": {
            "fit_telemetry": str(fit.resolve()),
            "eval_telemetry": str(evaluation.resolve()),
            "candidate_s": {"enabled": True},
        },
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")

    selection = load_candidate_s_selection(
        result_path,
        fit_path=fit,
        eval_path=evaluation,
        fit_row_count=5,
        eval_row_count=2,
    )
    assert selection.alpha == 16.0
    assert selection.latency_result_path == str(result_path.resolve())

    with pytest.raises(ValueError, match="eval input differs"):
        load_candidate_s_selection(
            result_path,
            fit_path=fit,
            eval_path=other,
            fit_row_count=5,
            eval_row_count=2,
        )


def test_resource_evaluator_compares_candidate_on_identical_label_rows() -> None:
    mib = 1024 * 1024
    fit = [
        Row(
            task_id=f"owner__fit-{index}",
            repo=f"owner__fit-{index}",
            manifest_index=index,
            bin="runner",
            argv=("runner", "deploy", "--mode=fast", f"/tmp/{index}"),
            latency_ms=1000.0,
            peak_cpu_cores=3.0,
            sampled_peak_rss_mb=100.0,
            disk_read_write_bytes_total=200 * mib,
        )
        for index in range(3)
    ]
    evaluation = [
        Row(
            task_id="owner__eval-1",
            repo="owner__eval",
            manifest_index=0,
            bin="runner",
            argv=("runner", "deploy", "--mode=slow", "/tmp/eval"),
            latency_ms=1000.0,
            peak_cpu_cores=3.0,
            sampled_peak_rss_mb=100.0,
            disk_read_write_bytes_total=200 * mib,
        ),
        Row(
            task_id="owner__eval-2",
            repo="owner__eval",
            manifest_index=1,
            bin="runner",
            argv=("runner", "deploy", "--mode=slow", "/tmp/eval"),
            latency_ms=1000.0,
            peak_cpu_cores=1.0,
            sampled_peak_rss_mb=600.0,
            disk_read_write_bytes_total=10 * mib,
        ),
        Row(
            task_id="owner__eval-3",
            repo="owner__eval",
            manifest_index=2,
            bin="runner",
            argv=("runner", "deploy", "--mode=slow", "/tmp/null"),
            latency_ms=600.0,
            peak_cpu_cores=None,
            sampled_peak_rss_mb=None,
            disk_read_write_bytes_total=None,
        ),
    ]

    result = evaluate_resources(
        fit,
        evaluation,
        candidate_s_selection=CandidateSSelection(
            alpha=64.0,
            latency_result_path="fixture-latency.json",
            fit_path="fixture-fit.jsonl",
            eval_path="fixture-eval.jsonl",
            fit_row_count=len(fit),
            eval_row_count=len(evaluation),
        ),
    )

    assert result["row_identity"] == {
        "identical_label_rows_across_arms": True,
        "fit_eval_task_overlap_count": 0,
    }
    assert result["candidate_r"]["representation"] == "generic-argv-v3-role"
    assert result["candidate_s"]["shrinkage_alpha"] == 64.0
    for resource, current in result["metrics"].items():
        candidate = result["candidates"]["candidate_r"]["metrics"][resource]
        shrinkage = result["candidates"]["candidate_s"]["metrics"][resource]
        assert current["eligible_n"] == candidate["eligible_n"] == 2
        assert current["null_unavailable"] == candidate["null_unavailable"] == 1
        assert current["prediction_unavailable"] == 0
        assert candidate["prediction_unavailable"] == 0
        assert candidate["current_accuracy"] == current["accuracy"]
        assert shrinkage["eligible_n"] == current["eligible_n"]
        assert shrinkage["prediction_unavailable"] == 0
