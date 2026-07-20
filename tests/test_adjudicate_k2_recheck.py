from __future__ import annotations

import numpy as np
import pytest

from scripts.adjudicate_k2_recheck import (
    AdjudicationConfig,
    _hazard_candidates,
    adjudicate_node_cell,
    enumerate_selected_nodes,
    k2_stage_value,
    run_adjudication,
    single_trigger_utilities,
)
from trace_collect.tool_latency_dataset import ToolLatencySample
from trace_collect.tool_latency_profiled import hazard_recheck_ms


def _cell(values, *, kv_cost_ms, restore_cost_fraction=0.0, overhead_ms=0.0):
    return adjudicate_node_cell(
        values,
        node_key="test",
        fold="f1",
        prior_source="prior_global",
        prior_group_key=None,
        kv_cost_ms=kv_cost_ms,
        guard_ms=0.0,
        restore_cost_fraction=restore_cost_fraction,
        overhead_ms=overhead_ms,
    )


def test_stage_value_reduces_to_single_trigger_utility() -> None:
    # The honest two-stage simulation must equal the single-trigger utility at
    # whichever checkpoint actually fires -- the reduction the optimizer relies
    # on. Checked at both branches against the shared functional.
    values = np.asarray([80.0, 80.0, 150.0], dtype=float)
    threshold_ms, kv_cost_ms, restore_cost_ms = 100.0, 100.0, 35.0
    candidates = _hazard_candidates(
        values, threshold_ms=threshold_ms, kv_cost_ms=kv_cost_ms
    )
    benefit = single_trigger_utilities(
        values,
        candidates,
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )
    idx0 = int(np.where(candidates == 0.0)[0][0])
    idx80 = int(np.where(candidates == 80.0)[0][0])

    swap = k2_stage_value(
        values,
        t1_ms=0.0,
        t2_ms=80.0,
        branch="swap",
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )
    defer = k2_stage_value(
        values,
        t1_ms=0.0,
        t2_ms=80.0,
        branch="defer",
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )
    assert swap == pytest.approx(float(benefit[idx0]))
    assert defer == pytest.approx(float(benefit[idx80]))
    # Analytic: fire@80 hides 70 minus 30 exposed on the one long call -> 40/3.
    assert defer == pytest.approx(40.0 / 3.0)


def test_value_equivalence_known_k1_optimum() -> None:
    # restore=0: hazard_recheck_ms fires at 0; k=2 optimum must tie it exactly.
    values = [80.0, 80.0, 150.0]
    cell = _cell(values, kv_cost_ms=100.0, restore_cost_fraction=0.0)
    assert hazard_recheck_ms(values, threshold_ms=100.0, kv_cost_ms=100.0) == 0.0
    assert cell.k1_trigger_ms == 0.0
    assert cell.value_gap == pytest.approx(0.0, abs=1e-9)
    assert cell.paired_delta == pytest.approx(0.0, abs=1e-9)
    assert cell.k2_branch == "swap"  # earliest candidate, no earlier defer point


def test_defer_branch_is_exercised_at_optimum() -> None:
    # restore shifts the k=1 optimum to 80; the k=2 DP must DEFER past the
    # earlier candidates {0, 50} to reach it, with an exact value tie.
    values = [80.0, 80.0, 150.0]
    cell = _cell(values, kv_cost_ms=100.0, restore_cost_fraction=0.35)
    assert cell.k1_trigger_ms == pytest.approx(80.0)
    assert cell.k2_branch == "defer"
    assert cell.k2_trigger_ms == pytest.approx(80.0)
    assert cell.value_gap == pytest.approx(0.0, abs=1e-9)
    assert cell.paired_delta == pytest.approx(0.0, abs=1e-9)


def test_value_equivalence_across_varied_sets_and_costs() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        values = (rng.gamma(2.0, 800.0, size=rng.integers(3, 40)) + 1.0).tolist()
        for kv in (500.0, 2500.0, 5000.0):
            cell = _cell(values, kv_cost_ms=kv, restore_cost_fraction=0.94)
            assert abs(cell.value_gap) <= 1e-6
            assert abs(cell.paired_delta) <= 1e-6
            assert cell.k2_value >= cell.k1_value - 1e-9


def test_overhead_makes_k2_never_strictly_better() -> None:
    # With a per-check overhead, k=2 is weakly dominated everywhere, and a
    # genuine two-check (defer) policy is strictly below the k=1 optimum.
    values = [80.0, 80.0, 150.0]
    cell = _cell(
        values, kv_cost_ms=100.0, restore_cost_fraction=0.35, overhead_ms=5.0
    )
    assert cell.priced_domination_gap <= 1e-9
    assert cell.priced_best_defer_value is not None
    assert cell.priced_best_defer_value < cell.priced_k1_value - 1e-9


def test_overhead_domination_holds_on_random_sets() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        values = (rng.gamma(2.0, 800.0, size=rng.integers(3, 40)) + 1.0).tolist()
        for kv in (500.0, 3500.0, 5000.0):
            cell = _cell(
                values, kv_cost_ms=kv, restore_cost_fraction=0.94, overhead_ms=2.0
            )
            assert cell.priced_domination_gap <= 1e-9


def test_determinism() -> None:
    values = [10.0, 200.0, 200.0, 3000.0, 50.0]
    first = _cell(values, kv_cost_ms=1500.0, restore_cost_fraction=0.94, overhead_ms=1.0)
    second = _cell(
        values, kv_cost_ms=1500.0, restore_cost_fraction=0.94, overhead_ms=1.0
    )
    assert first.to_json_obj() == second.to_json_obj()


def _sample(task_id, action_id, tool_name, latency_ms, command):
    return ToolLatencySample(
        sample_id=f"{task_id}:{action_id}",
        source_trace=f"trace-{task_id}",
        task_id=task_id,
        agent_id="a",
        instance_id=task_id,
        iteration=0,
        action_id=action_id,
        tool_name=tool_name,
        tool_call_id=action_id,
        tool_ts_start=0.0,
        tool_ts_end=latency_ms / 1000.0,
        latency_ms=latency_ms,
        success=True,
        reported_duration_ms=None,
        tool_args={"command": command},
    )


def _synthetic_corpus():
    samples_by_task = {}
    task_ids = []
    for t in range(6):
        task_id = f"task{t}"
        task_ids.append(task_id)
        samples_by_task[task_id] = [
            _sample(task_id, f"a{i}", "bash", 100.0 + 50.0 * t + 10.0 * i, "pytest -q")
            for i in range(3)
        ]
    return samples_by_task, task_ids


def test_enumerate_and_run_end_to_end() -> None:
    samples_by_task, task_ids = _synthetic_corpus()
    cfg = AdjudicationConfig(
        fold_count=3,
        command_field="command",
        max_prefix_depth=4,
        skip_leading_cd=False,
        min_tool_history=1,
        min_profile_tasks=1,
        costs_ms=(200.0, 500.0),
        guard_ms=0.0,
        restore_cost_fraction=0.94,
        overhead_ms=1.0,
        tolerance_ms=1e-6,
    )
    nodes = enumerate_selected_nodes(
        samples_by_task,
        task_ids,
        fold_count=cfg.fold_count,
        command_field=cfg.command_field,
        max_prefix_depth=cfg.max_prefix_depth,
        skip_leading_cd=cfg.skip_leading_cd,
        min_tool_history=cfg.min_tool_history,
        min_profile_tasks=cfg.min_profile_tasks,
    )
    assert nodes  # at least one selected node per fold
    results = run_adjudication(samples_by_task, task_ids, cfg)
    assert run_adjudication(samples_by_task, task_ids, cfg, workers=2) == results
    assert results["collapse_confirmed"] is True
    assert results["verdict"] == "COLLAPSE CONFIRMED"
    assert results["max_value_gap"] <= 1e-6
    assert results["domination"]["holds"] is True
    assert not results["offending_cells"]
