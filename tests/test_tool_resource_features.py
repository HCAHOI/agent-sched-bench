from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tool_resource.features import (
    assign_repo_folds,
    build_tabular_dataset,
    parse_command_clauses,
    repo_of,
)
from tool_time.command import make_row_command_prefix_keys
from tool_time.prior import build_latency_prior


_COMMAND = (
    "cd /testbed && for f in $(ls src); do python check.py $f | head -20 | "
    "wc -l; done && pytest tests/ -x -q 2>&1 | tail -5"
)


def test_parse_command_clauses_preserves_context_and_spans() -> None:
    parsed = parse_command_clauses(_COMMAND)
    clauses = parsed["clauses"]
    by_bin = {clause["bin"]: clause for clause in clauses}

    assert not parsed["parse_failed"]
    assert [clause["bin"] for clause in clauses] == [
        "cd",
        "ls",
        "python",
        "head",
        "wc",
        "pytest",
        "tail",
    ]
    assert not by_bin["cd"]["in_loop"]
    assert by_bin["ls"]["in_subst"] and not by_bin["ls"]["in_loop"]
    for position, binary in enumerate(("python", "head", "wc")):
        assert by_bin[binary]["in_loop"]
        assert by_bin[binary]["in_pipe"]
        assert by_bin[binary]["pipeline_position"] == position
    for position, binary in enumerate(("pytest", "tail")):
        assert not by_bin[binary]["in_loop"]
        assert by_bin[binary]["in_pipe"]
        assert by_bin[binary]["pipeline_position"] == position
    assert all(
        _COMMAND[clause["span"][0] : clause["span"][1]] == clause["original"]
        for clause in clauses
    )


def test_quoted_heredoc_uses_mvdan_and_excludes_body_from_span() -> None:
    parsed = parse_command_clauses("cat <<'EOF'\nx\nEOF\n")

    assert not parsed["parse_failed"]
    assert parsed["clauses"][0]["bin"] == "cat"
    assert parsed["clauses"][0]["original"] == "cat <<'EOF'"


def test_tabular_history_is_strictly_causal() -> None:
    baseline_calls = [
        _sample("a", 0.0, 1.0, peak_memory=10.0, peak_cpu=1.0),
        _sample("b", 2.0, 4.0, peak_memory=20.0, peak_cpu=2.0),
        _sample("c", 5.0, 8.0, peak_memory=30.0, peak_cpu=3.0),
    ]
    changed_calls = [
        _sample("a", 0.0, 1.0, peak_memory=10.0, peak_cpu=1.0),
        _sample("b", 2.0, 40.0, peak_memory=999.0, peak_cpu=99.0),
        _sample("c", 5.0, 80.0, peak_memory=9999.0, peak_cpu=999.0),
    ]
    prior = _prior()
    baseline = build_tabular_dataset(
        {"owner__repo-1": baseline_calls}, prior, None, [150.0]
    )
    changed = build_tabular_dataset(
        {"owner__repo-1": changed_calls}, prior, None, [150.0]
    )

    assert baseline.features["exact_command_recurrence_count"].tolist() == [
        0.0,
        1.0,
        2.0,
    ]
    assert baseline.features["same_command_last_latency_ms"][1] == 1000.0
    assert baseline.features["same_command_last_peak_memory_mb"][1] == 10.0
    assert baseline.features["same_command_last_peak_cpu_cores"][1] == 1.0
    assert baseline.features["prior_duration_ewma_ms"][1] == 1000.0
    assert baseline.features["prior_latency_p50_ms"][0] == 100.0
    assert baseline.features["prior_latency_p90_ms"][0] == 200.0
    assert baseline.features["prior_latency_survival_gt_150_ms"][0] == 0.5
    assert baseline.features["same_command_last_latency_ms"][2] == 2000.0
    assert baseline.features["prior_duration_ewma_ms"][2] == 1300.0
    assert changed.features["exact_command_recurrence_count"][2] == 2.0
    assert changed.features["same_command_last_latency_ms"][2] == 1000.0
    assert changed.features["same_command_last_peak_memory_mb"][2] == 10.0
    assert changed.features["same_command_last_peak_cpu_cores"][2] == 1.0
    assert changed.features["prior_duration_ewma_ms"][2] == 1000.0
    for feature_name in baseline.feature_names:
        np.testing.assert_array_equal(
            baseline.features[feature_name][:2],
            changed.features[feature_name][:2],
        )


def test_missing_cost_table_zero_fills_cost_features() -> None:
    sample = _sample(
        "a",
        0.0,
        1.0,
        command="python work.py | wc -l",
        peak_memory=10.0,
        peak_cpu=1.0,
    )
    dataset = build_tabular_dataset({"owner__repo-1": [sample]}, _prior(), None, [])
    later = _sample(
        "b",
        2.0,
        3.0,
        command="future-only work",
        peak_memory=20.0,
        peak_cpu=2.0,
    )
    with_later = build_tabular_dataset(
        {"owner__repo-1": [sample, later]}, _prior(), None, []
    )

    assert dataset.features["heavy_bin_count"][0] == 0.0
    assert dataset.features["heavy_bin_mean_cpu_s_sum"][0] == 0.0
    assert dataset.features["heavy_bin_mean_cpu_s_max"][0] == 0.0
    assert dataset.features["heavy_bin_mean_mem_kb_sum"][0] == 0.0
    assert dataset.features["heavy_bin_mean_mem_kb_max"][0] == 0.0
    assert dataset.features["light_bin_count"][0] == 0.0
    assert dataset.metadata == {
        "command_clause_parser": {
            "name": "mvdan.cc/sh/v3",
            "version": "v3.13.1",
        }
    }
    assert not any(name.startswith("bin_count:") for name in dataset.feature_names)
    assert with_later.feature_names == dataset.feature_names

    cost_table = {"python": {"mean_cpu_s": 2.0, "mean_mem_kb": 100.0}}
    costed = build_tabular_dataset(
        {"owner__repo-1": [sample, later]}, _prior(), cost_table, []
    )
    assert costed.features["bin_count:python"].tolist() == [1.0, 0.0]
    assert "bin_count:wc" not in costed.feature_names
    assert "bin_count:future-only" not in costed.feature_names
    assert costed.features["heavy_bin_count"][0] == 1.0
    assert costed.features["light_bin_count"][0] == 1.0


def test_repo_of_strips_instance_suffix() -> None:
    assert repo_of("google__flax-4681") == "google__flax"
    assert repo_of("ArkEcosystem__python-crypto-116") == (
        "ArkEcosystem__python-crypto"
    )


def test_repo_folds_keep_repositories_disjoint() -> None:
    repos = [repo_of(f"owner__proj{project}-{task}") for project in range(6) for task in range(3)]
    folds = assign_repo_folds(repos, n_folds=3, seed=7)

    for fold in range(3):
        validation = {repo for repo, assigned in folds.items() if assigned == fold}
        training = {repo for repo, assigned in folds.items() if assigned != fold}
        assert validation.isdisjoint(training)
    assert set(folds.values()) == {0, 1, 2}


def test_repo_fold_assignment_deduplicates_across_corpora() -> None:
    task_ids = [
        "shared__repo-1",
        "onlya__repo-9",
        "shared__repo-2",
        "onlyb__repo-4",
    ]
    folds = assign_repo_folds(
        (repo_of(task_id) for task_id in task_ids),
        n_folds=3,
        seed=1,
    )

    assert "shared__repo" in folds
    assert len(folds) == 3


def _sample(
    sample_id: str,
    start: float,
    end: float,
    *,
    command: str = "pytest tests/ -q",
    peak_memory: float,
    peak_cpu: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        sample_id=sample_id,
        source_trace=f"trace-{sample_id}",
        tool_name="exec",
        tool_args={"command": command},
        tool_ts_start=start,
        tool_ts_end=end,
        censored=False,
        peak_memory_mb=peak_memory,
        peak_memory_mb_eligible=True,
        peak_cpu_cores=peak_cpu,
        peak_cpu_cores_eligible=True,
    )


def _prior():
    rows = [
        {
            "sample_id": "profile-a",
            "source_trace": "profile-a",
            "task_id": "profile-a",
            "tool_name": "exec",
            "tool_args": {"command": "pytest tests/ -q"},
            "latency_ms": 100.0,
        },
        {
            "sample_id": "profile-b",
            "source_trace": "profile-b",
            "task_id": "profile-b",
            "tool_name": "exec",
            "tool_args": {"command": "pytest tests/ -q"},
            "latency_ms": 200.0,
        },
    ]
    keyer = make_row_command_prefix_keys("command", max_depth=4)
    return build_latency_prior(rows, row_group_keys=keyer)
