from __future__ import annotations

from types import SimpleNamespace

from scripts.evaluation.evaluate_prequential_binary import CpuRow
from scripts.evaluation.evaluate_two_layer_resource import (
    BackoffLattice,
    paired_ba_uncertainty,
    predict_heavy,
    prequential_forecasts,
)


def test_repo_node_is_preferred_to_public_at_equal_depth() -> None:
    fit = _sample("fit", "public__repo-1", 0.0, 1.0, cpu=1.0)
    repo_history = _sample("history", "owner__repo-1", 0.0, 1.0, cpu=4.0)
    current = _sample("current", "owner__repo-2", 2.0, 3.0, cpu=1.0)
    lattice = BackoffLattice.from_fit_samples([fit])
    lattice.add_repo_observation(repo_history)

    selected = lattice.select(current, "peak_cpu_cores")

    assert selected.scope == "repo"
    assert selected.granularity == "command_prefix_depth_3"
    assert selected.values == [4.0]


def test_public_node_is_used_when_repo_node_is_absent() -> None:
    fit = _sample("fit", "public__repo-1", 0.0, 1.0, cpu=3.0)
    current = _sample("current", "owner__other-1", 2.0, 3.0, cpu=1.0)
    lattice = BackoffLattice.from_fit_samples([fit])

    selected = lattice.select(current, "peak_cpu_cores")

    assert selected.scope == "public"
    assert selected.granularity == "command_prefix_depth_3"
    assert selected.values == [3.0]


def test_exceedance_rule_uses_strict_threshold_and_half_inclusive_vote() -> None:
    assert predict_heavy([1.0, 3.0])
    assert not predict_heavy([1.0, 2.0, 3.0])


def test_overlapping_candidate_is_excluded_until_strictly_completed() -> None:
    fit = _sample("fit", "public__repo-1", 0.0, 1.0, cpu=1.0)
    running = _sample("running", "owner__repo-1", 0.0, 10.0, cpu=4.0)
    current = _sample("current", "owner__repo-2", 5.0, 6.0, cpu=1.0)
    lattice = BackoffLattice.from_fit_samples([fit])

    forecasts = prequential_forecasts(lattice, [running, current])
    current_cpu = next(
        row
        for row in forecasts
        if row.sample_id == "current" and row.target == "peak_cpu_cores"
    )

    assert current_cpu.selected_scope == "public"
    assert not current_cpu.predicted_heavy


def test_order_and_bootstrap_are_deterministic() -> None:
    fit = [_sample("fit", "public__repo-1", 0.0, 1.0, cpu=1.0)]
    eval_rows = [
        _sample("first", "owner__a-1", 0.0, 1.0, cpu=4.0),
        _sample("second", "owner__a-2", 2.0, 3.0, cpu=1.0),
    ]
    first = prequential_forecasts(
        BackoffLattice.from_fit_samples(fit),
        eval_rows,
    )
    second = prequential_forecasts(
        BackoffLattice.from_fit_samples(fit),
        list(reversed(eval_rows)),
    )
    assert first == second

    rows = [
        _cpu_row("a-pos", "owner__a-1", True),
        _cpu_row("a-neg", "owner__a-2", False),
        _cpu_row("b-pos", "owner__b-1", True),
        _cpu_row("b-neg", "owner__b-2", False),
    ]
    candidate = [True, False, True, False]
    baseline = [False, False, False, True]
    uncertainty_a = paired_ba_uncertainty(
        rows,
        candidate,
        baseline,
        replicates=200,
        seed=7,
    )
    uncertainty_b = paired_ba_uncertainty(
        rows,
        candidate,
        baseline,
        replicates=200,
        seed=7,
    )
    assert uncertainty_a == uncertainty_b


def _sample(
    sample_id: str,
    task_id: str,
    start: float,
    end: float,
    *,
    cpu: float,
    command: str = "pytest tests -q",
) -> SimpleNamespace:
    return SimpleNamespace(
        sample_id=sample_id,
        task_id=task_id,
        tool_name="exec",
        tool_args={"command": command},
        tool_ts_start=start,
        tool_ts_end=end,
        censored=False,
        peak_cpu_cores=cpu,
        peak_cpu_cores_eligible=True,
        peak_memory_mb=cpu * 10.0,
        peak_memory_mb_eligible=True,
    )


def _cpu_row(sample_id: str, task_id: str, label: bool) -> CpuRow:
    return CpuRow(
        sample_id=sample_id,
        task_id=task_id,
        repo=task_id.rsplit("-", 1)[0],
        command="pytest tests -q",
        observed=4.0 if label else 1.0,
        label=label,
        static_label=False,
        tool_ts_start=0.0,
        tool_ts_end=1.0,
    )
