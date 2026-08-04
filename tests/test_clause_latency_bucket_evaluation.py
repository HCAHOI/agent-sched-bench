from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import pytest

from scripts.evaluation.evaluate_clause_latency_buckets import (
    _EpisodicSubsetKB,
    _InteractionPosetKB,
    ScoredRow,
    _argmax_bucket,
    _bounded_node_oracle_candidates,
    _exact_bucket_metrics,
    _empty_resource_bucket_confusion,
    _finalize_resource_bucket_metric,
    _interaction_feature_set,
    _maximal_intersections,
    _parser,
    _select_shrinkage_alpha,
    _subset_count,
    _subset_kernel,
    _telemetry_metrics,
    _telemetry_scored_rows,
    _validate_partition,
    evaluate_clause_telemetry,
    evaluate_interaction_commands,
    evaluate_poset_resources,
    evaluate_prequential_commands,
)
from scripts.evaluation.evaluate_clause_resource_classes import (
    CandidateSSelection,
    CommandRow,
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
        "exact_class_accuracy": 0.5,
        "within_one_bucket_accuracy": 1.0,
        "severe_underprediction_rate": 0.0,
        "eligible_examples": 2,
    }


def test_resource_bucket_metrics_count_severe_underprediction() -> None:
    raw = _empty_resource_bucket_confusion()
    raw["confusion_label_by_prediction"] = [
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
    ]

    metric = _finalize_resource_bucket_metric(
        raw,
        Counter({0: 1, 1: 1, 2: 1}),
        Counter({"observed": 3}),
    )

    assert metric["accuracy"] == pytest.approx(2 / 3)
    assert metric["within_one_bucket_accuracy"] == pytest.approx(2 / 3)
    assert metric["severe_underprediction_rate"] == pytest.approx(1 / 3)
    assert metric["majority_class_id"] == 0


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


def test_interaction_poset_features_and_maximal_frontier() -> None:
    stable = frozenset({("runner", "deploy")})
    reordered_left = _interaction_feature_set(
        "runner",
        ("runner", "deploy", "--path=/tmp/a", "--count=10", "/tmp/x", "2"),
        stable,
    )
    reordered_right = _interaction_feature_set(
        "runner",
        ("runner", "deploy", "--count=10", "--path=/tmp/a", "/tmp/x", "2"),
        stable,
    )
    positional_reordered = _interaction_feature_set(
        "runner",
        ("runner", "deploy", "--count=10", "--path=/tmp/a", "2", "/tmp/x"),
        stable,
    )
    repeated = _interaction_feature_set(
        "runner",
        ("runner", "deploy", "--tag=x", "--tag=y"),
        stable,
    )
    flag_a = _interaction_feature_set("x", ("x", "--a"), frozenset())
    flag_b = _interaction_feature_set("x", ("x", "--b"), frozenset())

    assert reordered_left == reordered_right
    assert reordered_left != positional_reordered
    assert not (flag_a.features & flag_b.features)
    assert {
        feature for feature in repeated.features if feature.startswith("option:--tag")
    } == {
        "option:--tag=<ARG>:occurrence:1",
        "option:--tag=<ARG>:occurrence:2",
    }
    assert _maximal_intersections(
        frozenset({"a", "b", "c"}),
        (
            frozenset({"a"}),
            frozenset({"a", "c"}),
            frozenset({"a", "b"}),
            frozenset({"d"}),
        ),
    ) == frozenset({frozenset({"a", "c"}), frozenset({"a", "b"})})

    def row(task: int, *options: str) -> Row:
        return Row(
            task_id=f"owner__repo-{task}",
            repo="owner__repo",
            manifest_index=task - 1,
            bin="runner",
            argv=("runner", "deploy", *options),
            latency_ms=100.0,
            peak_cpu_cores=None,
            sampled_peak_rss_mb=None,
            disk_read_write_bytes_total=None,
        )

    poset = _InteractionPosetKB(stable)
    query = row(3, "--a=x", "--b=x", "--c=x")
    assert poset.query(query).observations == ()
    history = [
        row(1, "--a=x"),
        row(1, "--a=x", "--c=x"),
        row(2, "--a=x", "--b=x"),
    ]
    poset.observe(history)

    match = poset.query(query)
    assert match.exact is False
    assert [item.row for item in match.observations] == history[1:]
    assert len({item.observation_id for item in match.observations}) == 2
    assert poset.query(history[1]).observations[0].row == history[1]


def test_episodic_subset_kernel_matches_explicit_subsets() -> None:
    query = frozenset({"a", "b", "c"})
    history = frozenset({"b", "c", "d", "e"})

    def explicit_subsets(values: frozenset[str], order: int | None) -> set[tuple[str, ...]]:
        limit = len(values) if order is None else min(order, len(values))
        return {
            subset
            for size in range(1, limit + 1)
            for subset in combinations(sorted(values), size)
        }

    for order in (None, 1, 2, 3):
        query_subsets = explicit_subsets(query, order)
        history_subsets = explicit_subsets(history, order)
        expected = len(query_subsets & history_subsets) / math.sqrt(
            len(query_subsets) * len(history_subsets)
        )
        assert _subset_kernel(query, history, order) == pytest.approx(expected)
        assert _subset_count(len(query), order) == len(query_subsets)
    assert _subset_kernel(frozenset(), history) == 0.0
    with pytest.raises(ValueError, match="positive"):
        _subset_count(2, 0)

    def row(task: int, *options: str) -> Row:
        return Row(
            task_id=f"owner__repo-{task}",
            repo="owner__repo",
            manifest_index=task - 1,
            bin="runner",
            argv=("runner", "deploy", *options),
            latency_ms=100.0,
            peak_cpu_cores=None,
            sampled_peak_rss_mb=None,
            disk_read_write_bytes_total=None,
        )

    memory = _EpisodicSubsetKB(frozenset({("runner", "deploy")}))
    histories = [row(1, "--a=x", "--b=x"), row(2, "--a=x", "--c=x")]
    memory.observe(histories)
    match = memory.query(row(3, "--a=x", "--b=x", "--c=x"))
    assert match.exact is False
    assert [item.observation.row for item in match.contributions] == histories
    assert all(item.weight > 0.0 for item in match.contributions)
    assert len({item.observation.observation_id for item in match.contributions}) == 2
    exact = memory.query(histories[0])
    assert exact.exact is True
    assert [(item.observation.row, item.weight) for item in exact.contributions] == [
        (histories[0], 1.0)
    ]


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


def test_command_prequential_baseline_updates_only_between_tasks() -> None:
    mib = 1024 * 1024

    def row(task: int, latency: float, heavy: bool) -> Row:
        return Row(
            task_id=f"target__repo-{task}",
            repo="target__repo",
            manifest_index=task - 1,
            bin="x",
            argv=("x",),
            latency_ms=latency,
            peak_cpu_cores=3.0 if heavy else 1.0,
            sampled_peak_rss_mb=600.0 if heavy else 100.0,
            disk_read_write_bytes_total=(200 if heavy else 1) * mib,
        )

    warm = row(1, 3000.0, False)
    test_rows = [row(2, 9000.0, True), row(2, 9000.0, True), row(3, 9000.0, True)]
    clauses = [warm, *test_rows]
    commands = [
        CommandRow(
            task_id=clause.task_id,
            repo=clause.repo,
            manifest_index=clause.manifest_index,
            call_index=call_index,
            call_id=f"call-{index}",
            command="x",
            duration_ms=clause.latency_ms,
            clauses=(clause,),
        )
        for index, (call_index, clause) in enumerate(
            zip((0, 0, 1, 0), clauses, strict=True)
        )
    ]
    public = [
        Row(
            task_id="public__repo-1",
            repo="public__repo",
            manifest_index=0,
            bin="x",
            argv=("x",),
            latency_ms=100.0,
            peak_cpu_cores=1.0,
            sampled_peak_rss_mb=100.0,
            disk_read_write_bytes_total=mib,
        )
    ]

    result, sidecar = evaluate_prequential_commands(
        public,
        ["target__repo-1", "target__repo-2", "target__repo-3"],
        clauses,
        commands,
        {"fixture": True},
        warmup_task_count=1,
    )

    assert len(sidecar) == 3
    assert [item["current_dynamic"]["latency"] for item in sidecar] == [2, 2, 3]
    assert [item["frozen_at_80"]["latency"] for item in sidecar] == [2, 2, 2]
    assert result["latency"]["majority"] == {
        "class": "long",
        "class_id": 3,
        "accuracy": 1.0,
    }
    assert result["latency"]["current_dynamic"]["exact_class_accuracy"] == 1 / 3
    assert result["latency"]["frozen_at_80"]["exact_class_accuracy"] == 0.0
    for metric in result["resources"].values():
        assert metric["majority"]["accuracy"] == 1.0
        assert metric["constant_low_accuracy"] == 0.0
        assert metric["current_dynamic"]["accuracy"] == 1 / 3
        assert metric["frozen_at_80"]["accuracy"] == 0.0


def test_interaction_evaluator_updates_only_after_task_settlement() -> None:
    def row(task: int) -> Row:
        return Row(
            task_id=f"target__repo-{task}",
            repo="target__repo",
            manifest_index=task - 1,
            bin="x",
            argv=("x", "--mode=slow"),
            latency_ms=9000.0,
            peak_cpu_cores=None,
            sampled_peak_rss_mb=None,
            disk_read_write_bytes_total=None,
        )

    clauses = [row(1), row(1), row(2)]
    commands = [
        CommandRow(
            task_id=clause.task_id,
            repo=clause.repo,
            manifest_index=clause.manifest_index,
            call_index=call_index,
            call_id=f"call-{index}",
            command="x --mode=slow",
            duration_ms=clause.latency_ms,
            clauses=(clause,),
        )
        for index, (call_index, clause) in enumerate(
            zip((0, 1, 0), clauses, strict=True)
        )
    ]
    public = [
        Row(
            task_id="public__repo-1",
            repo="public__repo",
            manifest_index=0,
            bin="x",
            argv=("x", "--mode=fast"),
            latency_ms=100.0,
            peak_cpu_cores=None,
            sampled_peak_rss_mb=None,
            disk_read_write_bytes_total=None,
        )
    ]

    result, sidecar = evaluate_interaction_commands(
        public,
        ["target__repo-1", "target__repo-2"],
        clauses,
        commands,
        {"fixture": True},
    )

    for arm in ("current", "interaction_poset", "subset_kernel"):
        assert [item["arms"][arm]["prediction"] for item in sidecar] == [0, 0, 3]
        assert result["latency"]["arms"][arm]["exact_class_accuracy"] == 1 / 3
    assert result["row_identity"][
        "identical_command_ids_labels_and_availability"
    ] is True
    assert result["gates"]["stage0"]["pass"] is False
    assert result["counts"]["stored_target_observations_by_candidate"] == {
        "interaction_poset": 3,
        "subset_kernel": 3,
        "subset_k1": 3,
        "subset_k2": 3,
        "subset_k3": 3,
    }


def test_interaction_queries_use_static_not_observed_argv() -> None:
    target = [
        Row(
            task_id=f"target__repo-{task}",
            repo="target__repo",
            manifest_index=task - 1,
            bin="x",
            argv=("x", "slow"),
            latency_ms=9000.0,
            peak_cpu_cores=None,
            sampled_peak_rss_mb=None,
            disk_read_write_bytes_total=None,
        )
        for task in (1, 2)
    ]
    commands = [
        CommandRow(
            task_id=clause.task_id,
            repo=clause.repo,
            manifest_index=clause.manifest_index,
            call_index=0,
            call_id=f"call-{index}",
            command='x "$MODE"',
            duration_ms=clause.latency_ms,
            clauses=(clause,),
        )
        for index, clause in enumerate(target)
    ]
    public = [
        replace(
            target[0],
            task_id="public__repo-1",
            repo="public__repo",
            argv=("x", "fast"),
            latency_ms=100.0,
        )
    ]

    _, sidecar = evaluate_interaction_commands(
        public,
        ["target__repo-1", "target__repo-2"],
        target,
        commands,
        {"fixture": True},
    )

    for arm in ("interaction_poset", "subset_kernel"):
        assert sidecar[1]["arms"][arm]["all_clauses_exact"] is False


def test_interaction_exact_compound_uses_current_composition() -> None:
    mib = 1024 * 1024
    values = ((100.0, 9000.0), (9000.0, 100.0), (100.0, 100.0))
    clauses: list[Row] = []
    commands: list[CommandRow] = []
    for task, (a_latency, b_latency) in enumerate(values, start=1):
        pair = tuple(
            Row(
                task_id=f"target__repo-{task}",
                repo="target__repo",
                manifest_index=task - 1,
                bin=bin_,
                argv=(bin_,),
                latency_ms=latency,
                peak_cpu_cores=3.0 if latency > 8000 else 1.0,
                sampled_peak_rss_mb=600.0 if latency > 8000 else 100.0,
                disk_read_write_bytes_total=(200 if latency > 8000 else 1) * mib,
            )
            for bin_, latency in (("a", a_latency), ("b", b_latency))
        )
        clauses.extend(pair)
        commands.append(
            CommandRow(
                task_id=pair[0].task_id,
                repo=pair[0].repo,
                manifest_index=pair[0].manifest_index,
                call_index=0,
                call_id=f"call-{task}",
                command="a; b",
                duration_ms=a_latency + b_latency,
                clauses=pair,
            )
        )
    public = [
        replace(
            clauses[0],
            task_id="public__repo-1",
            repo="public__repo",
        )
    ]

    _, sidecar = evaluate_interaction_commands(
        public,
        [f"target__repo-{task}" for task in range(1, 4)],
        clauses,
        commands,
        {"fixture": True},
    )

    current = sidecar[2]["arms"]["current"]["probability_by_bucket"]
    for arm in ("interaction_poset", "subset_kernel", "subset_k1", "subset_k2", "subset_k3"):
        assert sidecar[2]["arms"][arm]["all_clauses_exact"] is True
        assert sidecar[2]["arms"][arm]["probability_by_bucket"] == current

    resource_provenance = {
        "target_run_dir": "fixture-run",
        "public_telemetry": ["fixture-public"],
        "public_excluded_repositories": ["target__repo"],
        "public_clause_observations_before_repo_filter": 1,
        "public_clause_observations_after_repo_filter": 1,
        "public_online_eligible_clause_observations": 1,
    }
    latency_gate = {
        "status": "development_exposed_latency_go",
        "objective": "command_latency_interaction_kb_comparison",
        "inputs": resource_provenance,
        "counts": {
            "tasks": 3,
            "commands": 3,
            "target_online_clause_observations": 6,
            "public_online_clause_observations": 1,
        },
        "protocol": {
            "evaluation_unit": "eligible_exec_command",
            "pooling_alpha": 16.0,
        },
        "row_identity": {
            "identical_command_ids_labels_and_availability": True,
        },
        "gates": {
            "stage0": {"pass": True},
            "stage1_resource_evaluation": {
                "interaction_poset": {"go": True}
            },
        },
    }
    resource_result, resource_sidecar = evaluate_poset_resources(
        public,
        [f"target__repo-{task}" for task in range(1, 4)],
        clauses,
        commands,
        resource_provenance,
        latency_gate,
    )
    assert resource_result["row_identity"][
        "identical_command_ids_labels_and_availability"
    ] is True
    for resource in (
        "peak_cpu_cores",
        "sampled_peak_rss_mb",
        "disk_read_write_bytes_total",
    ):
        assert resource_sidecar[2]["arms"]["interaction_poset"][resource][
            "all_clauses_exact"
        ] is True
        assert resource_sidecar[2]["arms"]["interaction_poset"][resource][
            "probability_heavy"
        ] == resource_sidecar[2]["arms"]["current"][resource][
            "probability_heavy"
        ]

    wrong_gate = json.loads(json.dumps(latency_gate))
    wrong_gate["inputs"]["target_run_dir"] = "different-run"
    with pytest.raises(ValueError, match="does not match"):
        evaluate_poset_resources(
            public,
            [f"target__repo-{task}" for task in range(1, 4)],
            clauses,
            commands,
            resource_provenance,
            wrong_gate,
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


def test_public_loader_recovers_pipeline_positions(tmp_path: Path) -> None:
    record = json.loads(
        _telemetry_record(
            "owner__repo-1",
            0,
            (("a", ("a",), 1.0), ("b", ("b",), 1.0)),
        )
    )
    telemetry = record["data"]["clause_telemetry"]
    telemetry["command"] = "a | b"
    telemetry["static_word_intent"] = [
        {"bin": "a", "argv": ["a"]},
        {"bin": "b", "argv": ["b"]},
    ]
    path = tmp_path / "public.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert [row.pipeline_position for row in load_rows(path)] == [0, 1]

    telemetry["command"] = "a | a"
    telemetry["static_word_intent"] = [
        {"bin": "a", "argv": ["a"]},
        {"bin": "a", "argv": ["a"]},
    ]
    telemetry["clauses"] = telemetry["clauses"][:1]
    path.write_text(json.dumps(record) + "\n")
    assert load_rows(path)[0].structure_known is False


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
    assert [row.label_bucket for row in first_task] == [2, 2]

    second_task = {row.command: row for row in rows if row.task_id.endswith("-2")}
    learned = second_task["sleep 9"]
    # The earlier task settled before this one queried, so repo evidence wins.
    assert learned.layer == "repo"
    assert _argmax_bucket(learned) == 2
    assert learned.evidence_count == 2
    # 500 ms is exactly the first edge and belongs to the instant bucket.
    assert second_task["sleep 1"].label_bucket == 0

    metrics = _telemetry_metrics(rows)
    assert metrics["eligible_examples"] == 4
    assert metrics["exact_class_accuracy"] == 0.25
    assert metrics["majority_class"] == "medium"
    assert metrics["majority_class_id"] == 2
    assert metrics["majority_class_accuracy"] == 0.75
    assert metrics["accuracy_minus_majority_percentage_points"] == -50.0
    assert metrics["accuracy_minus_current_percentage_points"] == 0.0
    assert metrics["prediction_unavailable"] == 0
    assert sum(sum(row) for row in metrics["confusion_label_by_prediction"]) == 4
    assert metrics["confusion_label_by_prediction"][2][0] == 2
    assert metrics["per_class"][2] == {
        "class": "medium",
        "class_id": 2,
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
    current = result["baselines"]["current"]["exact_class_accuracy"]
    public = result["baselines"]["public_only"]["exact_class_accuracy"]
    assert current is not None and public is not None
    assert result["oracles"]["current_public"]["exact_class_accuracy"] >= max(
        current,
        public,
    )
    assert result["oracles"]["current_and_candidate_nodes"][
        "exact_class_accuracy"
    ] >= result["oracles"]["current_public"]["exact_class_accuracy"]
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
    assert local["exact_class_accuracy"] is None


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
        "bucket_edges_ms": [500.0, 2000.0, 8000.0, 30000.0],
        "fit_clause_observation_count": 5,
        "eval_clause_observation_count": 2,
        "row_identity": {"identical_row_ids_and_labels": True},
        "selection": {
            "candidate_s_alpha": {
                "alpha_grid": [1.0, 4.0, 16.0, 64.0],
                "selected_alpha": 16.0,
                "selection_target": "exact_latency_class_accuracy",
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
