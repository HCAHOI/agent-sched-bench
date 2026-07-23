from __future__ import annotations

from scripts.evaluation.evaluate_prequential_binary import (
    CpuRow,
    Decision,
    prequential_decisions,
    repo_clustered_uncertainty,
    repo_key,
)


def test_overlapping_same_command_is_not_observed_before_completion() -> None:
    rows = [
        _row(
            "running",
            "owner__repo-1",
            "pytest tests -q",
            observed=4.0,
            static=False,
            start=0.0,
            end=10.0,
        ),
        _row(
            "current",
            "owner__repo-1",
            "pytest tests -q",
            observed=1.0,
            static=False,
            start=5.0,
            end=6.0,
        ),
    ]

    current = _by_id(prequential_decisions(rows), "current")

    assert current.tier == "cold"
    assert not current.blended_label


def test_match_tiers_use_declared_priority() -> None:
    exact = "pytest tests -q"
    rows = [
        _row(
            "task-exact",
            "owner__repo-1",
            exact,
            observed=1.0,
            static=True,
            start=0.0,
            end=1.0,
        ),
        _row(
            "repo-exact",
            "owner__repo-2",
            exact,
            observed=4.0,
            static=False,
            start=0.1,
            end=2.0,
        ),
        _row(
            "repo-head",
            "owner__repo-2",
            "cd /testbed && pytest tests -x",
            observed=1.0,
            static=True,
            start=0.2,
            end=3.0,
        ),
        _row(
            "use-a",
            "owner__repo-1",
            exact,
            observed=4.0,
            static=True,
            start=4.0,
            end=20.0,
        ),
        _row(
            "use-b",
            "owner__repo-3",
            exact,
            observed=1.0,
            static=False,
            start=5.0,
            end=20.0,
        ),
        _row(
            "use-c",
            "owner__repo-4",
            "pytest tests -k focused",
            observed=4.0,
            static=True,
            start=6.0,
            end=20.0,
        ),
    ]

    decisions = prequential_decisions(rows)

    assert (
        _by_id(decisions, "use-a").tier,
        _by_id(decisions, "use-a").blended_label,
    ) == (
        "a",
        False,
    )
    assert (
        _by_id(decisions, "use-b").tier,
        _by_id(decisions, "use-b").blended_label,
    ) == (
        "b",
        True,
    )
    assert (
        _by_id(decisions, "use-c").tier,
        _by_id(decisions, "use-c").blended_label,
    ) == (
        "c",
        False,
    )


def test_repo_key_strips_only_trailing_numeric_suffix() -> None:
    assert repo_key("owner__repo-name-123") == "owner__repo-name"
    assert repo_key("owner__repo-name") == "owner__repo-name"
    assert repo_key("owner__repo-name-12x") == "owner__repo-name-12x"


def test_repo_bootstrap_is_seed_deterministic() -> None:
    decisions = []
    for index, repo in enumerate(("owner__a", "owner__b", "owner__c")):
        positive = _row(
            f"{repo}-positive",
            f"{repo}-{index}",
            "pytest tests -q",
            observed=4.0,
            static=index == 0,
            start=float(index),
            end=float(index + 1),
        )
        negative = _row(
            f"{repo}-negative",
            f"{repo}-{index}",
            "pytest tests -q",
            observed=1.0,
            static=index == 1,
            start=float(index),
            end=float(index + 1),
        )
        decisions.extend(
            [
                Decision(positive, "a", True),
                Decision(negative, "a", False),
            ]
        )

    first = repo_clustered_uncertainty(decisions, replicates=200, seed=7)
    second = repo_clustered_uncertainty(decisions, replicates=200, seed=7)

    assert first == second
    assert first["seed"] == 7
    assert first["blended"]["point"] == 1.0


def _row(
    sample_id: str,
    task_id: str,
    command: str,
    *,
    observed: float,
    static: bool,
    start: float,
    end: float,
) -> CpuRow:
    return CpuRow(
        sample_id=sample_id,
        task_id=task_id,
        repo=repo_key(task_id),
        command=command,
        observed=observed,
        label=observed > 2.0,
        static_label=static,
        tool_ts_start=start,
        tool_ts_end=end,
    )


def _by_id(decisions: list[Decision], sample_id: str) -> Decision:
    return next(
        decision for decision in decisions if decision.row.sample_id == sample_id
    )
