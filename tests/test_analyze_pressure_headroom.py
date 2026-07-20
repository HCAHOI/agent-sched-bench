from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from scripts.analyze_pressure_headroom import (
    PressureHeadroomConfig,
    believed_trigger_ms,
    fixed_lambda_candidates,
    footprint_growth_stats,
    load_kv_footprint_tokens,
    pressure_price_ms,
    main,
    realized_utilities_ms,
    sample_footprint_tokens,
    score_decisions,
    select_fixed_lambda_ms,
    summarize,
)
from trace_collect.tool_latency_dataset import ToolLatencySample
from trace_collect.tool_latency_profiled import LatencyPriorNode, hazard_recheck_ms
from trace_collect.tool_latency_utility_clock import (
    trigger_policy_utility_ms,
    utility_matrix,
)

_RHO = 0.94


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #
def _sample(
    task_id: str, sample_id: str, latency_ms: float, *, iteration: int = 0
) -> ToolLatencySample:
    return ToolLatencySample(
        sample_id=sample_id,
        source_trace=f"/traces/{task_id}.jsonl",
        task_id=task_id,
        agent_id="agent",
        instance_id=task_id,
        iteration=iteration,
        action_id=sample_id,
        tool_name="bash",
        tool_call_id=sample_id,
        tool_ts_start=0.0,
        tool_ts_end=latency_ms,
        latency_ms=latency_ms,
        success=True,
        reported_duration_ms=None,
        tool_args={"command": "run"},
    )


def _node(values_by_task: dict[str, list[float]]) -> LatencyPriorNode:
    return LatencyPriorNode(
        source="tool",
        group_key=None,
        values=[v for vs in values_by_task.values() for v in vs],
        values_by_task=values_by_task,
    )


def _footprints(spec: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    return {
        (str(Path(f"/traces/{task}.jsonl").resolve()), it): tokens
        for (task, it), tokens in spec.items()
    }


def _cfg(costs_ms: tuple[float, ...], **kw) -> PressureHeadroomConfig:
    base = dict(
        fold_count=2,
        command_field="command",
        max_prefix_depth=4,
        skip_leading_cd=False,
        min_tool_history=1,
        min_profile_tasks=1,
        costs_ms=costs_ms,
        guard_ms=0.0,
        restore_cost_fraction=_RHO,
        replicates=200,
        confidence_level=0.95,
        seed=0,
        fixed_lambda_grid=7,
    )
    base.update(kw)
    return PressureHeadroomConfig(**base)


# --------------------------------------------------------------------------- #
# The lambda map: monotone and outcome-independent.
# --------------------------------------------------------------------------- #
def test_price_map_is_strictly_monotone_in_footprint() -> None:
    prices = [
        pressure_price_ms(tok, kv_cost_ms=3500.0, reference_tokens=10000.0)
        for tok in (1000.0, 5000.0, 10000.0, 50000.0)
    ]
    assert prices == sorted(prices)
    assert all(b > a for a, b in zip(prices, prices[1:]))


def test_price_map_is_linear_and_anchored_at_reference() -> None:
    # Physical contract: KV bytes are linear in resident tokens, so the price is
    # linear in tokens and equals the panel cell exactly at the reference.
    assert pressure_price_ms(
        10000.0, kv_cost_ms=3500.0, reference_tokens=10000.0
    ) == pytest.approx(3500.0)
    assert pressure_price_ms(
        20000.0, kv_cost_ms=3500.0, reference_tokens=10000.0
    ) == pytest.approx(7000.0)


def test_price_map_ignores_outcomes() -> None:
    # The map's only inputs are the footprint and config -- there is no latency
    # or success argument it *could* consume. Two calls with wildly different
    # outcomes but equal footprints get an identical price.
    kw = dict(kv_cost_ms=3500.0, reference_tokens=10000.0)
    assert pressure_price_ms(12345.0, **kw) == pressure_price_ms(12345.0, **kw)
    with pytest.raises(TypeError):
        pressure_price_ms(12345.0, latency_ms=999.0, **kw)  # type: ignore[call-arg]


def test_price_map_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError):
        pressure_price_ms(100.0, kv_cost_ms=3500.0, reference_tokens=0.0)
    with pytest.raises(ValueError):
        pressure_price_ms(0.0, kv_cost_ms=3500.0, reference_tokens=100.0)


# --------------------------------------------------------------------------- #
# Footprint loading / joining.
# --------------------------------------------------------------------------- #
def test_load_footprints_keeps_first_llm_call_per_iteration(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"action_type": "llm_call", "iteration": 0, "data": {"prompt_tokens": 1000}},
                # A malformed-retry re-issue must NOT overwrite the emitting call.
                {"action_type": "llm_call", "iteration": 0, "data": {"prompt_tokens": 9999}},
                {"action_type": "tool_exec", "iteration": 0, "data": {}},
                {"action_type": "llm_call", "iteration": 1, "data": {"prompt_tokens": 2500}},
            ]
        ),
        encoding="utf-8",
    )
    footprints = load_kv_footprint_tokens([trace])
    assert footprints[(str(trace.resolve()), 0)] == 1000.0
    assert footprints[(str(trace.resolve()), 1)] == 2500.0


def test_unjoinable_sample_fails_fast() -> None:
    with pytest.raises(ValueError, match="no prompt_tokens footprint"):
        sample_footprint_tokens(_sample("t0", "t0:0", 100.0), {})


# --------------------------------------------------------------------------- #
# Headroom non-negativity: footprint_priced >= best-fixed by construction.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kv", [1000.0, 3500.0, 5000.0])
def test_footprint_priced_dominates_any_constant_lambda_per_call(kv: float) -> None:
    # Per call, the footprint_priced trigger is the exact optimum of the SAME
    # functional at the call's true price, so no believed constant price can
    # score higher on the node it was optimized for.
    node = _node({"a": [200.0, 4800.0], "b": [900.0, 6000.0]})
    for tokens in (2000.0, 12000.0, 60000.0):
        true_price = pressure_price_ms(
            tokens, kv_cost_ms=kv, reference_tokens=10000.0
        )
        footprint_priced = believed_trigger_ms(
            node,
            believed_price_ms=true_price,
            domain_max_ms=kv,
            guard_ms=0.0,
            restore_cost_fraction=_RHO,
        )
        best = max(
            np.mean(
                [
                    trigger_policy_utility_ms(
                        latency,
                        footprint_priced,
                        threshold_ms=true_price,
                        kv_cost_ms=true_price,
                        restore_cost_ms=_RHO * true_price,
                    )
                    for latency in node.values
                ]
            )
            for _ in (0,)
        )
        for believed in (0.3 * kv, kv, 3.0 * kv):
            fixed = believed_trigger_ms(
                node,
                believed_price_ms=believed,
                domain_max_ms=kv,
                guard_ms=0.0,
                restore_cost_fraction=_RHO,
            )
            fixed_value = np.mean(
                [
                    trigger_policy_utility_ms(
                        latency,
                        fixed,
                        threshold_ms=true_price,
                        kv_cost_ms=true_price,
                        restore_cost_ms=_RHO * true_price,
                    )
                    for latency in node.values
                ]
            )
            assert fixed_value <= best + 1e-9


def _node_expected_total(
    node: LatencyPriorNode, prices: Sequence[float], *, believed: float | None
) -> float:
    """Total node-expected utility; ``believed=None`` is the footprint_priced arm."""

    total = 0.0
    for price in prices:
        trigger = believed_trigger_ms(
            node,
            believed_price_ms=price if believed is None else believed,
            domain_max_ms=3500.0,
            guard_ms=0.0,
            restore_cost_fraction=_RHO,
        )
        total += float(
            np.mean(
                [
                    realized_utilities_ms(
                        latency,
                        np.asarray([trigger]),
                        true_price_ms=price,
                        guard_ms=0.0,
                        restore_cost_fraction=_RHO,
                    )[0]
                    for latency in node.values
                ]
            )
        )
    return total


def test_headroom_is_non_negative_in_node_expectation() -> None:
    # THE construction that makes the headroom a ceiling: in node-expectation the
    # footprint_priced trigger dominates the trigger induced by ANY constant believed
    # price. (This is an expectation guarantee, not a pointwise one on realized
    # held-out latencies -- see the module docstring.)
    node = _node({"a": [150.0, 4200.0], "b": [800.0, 7000.0], "c": [50.0, 3000.0]})
    kv = 3500.0
    prices = [
        pressure_price_ms(t, kv_cost_ms=kv, reference_tokens=20000.0)
        for t in (2000.0, 9000.0, 15000.0, 40000.0, 70000.0, 5000.0)
    ]
    footprint_priced = _node_expected_total(node, prices, believed=None)
    for believed in (0.25 * kv, kv, 2.0 * kv, 8.0 * kv, *prices):
        assert footprint_priced >= _node_expected_total(node, prices, believed=believed) - 1e-6


def test_best_fixed_is_never_worse_than_the_shipped_constant_price() -> None:
    # The conservative half of the asymmetry: lambda_bar is selected on the
    # REALIZED fit-fold total and the panel cell is always a candidate, so the
    # no-pressure arm is at least as strong as today's shipped constant price.
    node = _node({"a": [150.0, 4200.0], "b": [800.0, 7000.0], "c": [50.0, 3000.0]})
    kv = 3500.0
    tokens = [2000.0, 9000.0, 15000.0, 40000.0, 70000.0, 5000.0]
    latencies = [120.0, 4300.0, 900.0, 6800.0, 60.0, 3100.0]
    prices = np.asarray(
        [pressure_price_ms(t, kv_cost_ms=kv, reference_tokens=20000.0) for t in tokens]
    )
    fit_calls = list(zip([node] * len(tokens), latencies, prices))
    _, fixed_total = select_fixed_lambda_ms(
        fit_calls,
        candidates=fixed_lambda_candidates(prices, kv_cost_ms=kv, grid=9),
        kv_cost_ms=kv,
        guard_ms=0.0,
        restore_cost_fraction=_RHO,
    )
    _, status_quo_total = select_fixed_lambda_ms(
        fit_calls,
        candidates=np.asarray([kv]),
        kv_cost_ms=kv,
        guard_ms=0.0,
        restore_cost_fraction=_RHO,
    )
    assert fixed_total >= status_quo_total - 1e-9


# --------------------------------------------------------------------------- #
# Cross-fit isolation: the fixed lambda is chosen on FIT folds only.
# --------------------------------------------------------------------------- #
def test_fixed_lambda_ignores_eval_fold_entirely() -> None:
    # Two runs share every profile task but differ wildly in the eval task's
    # latency and footprint. The selected constant lambda and the fixed trigger
    # must be identical -- if eval data leaked into selection they would move.
    kv = 3500.0
    common = {
        "t1": [_sample("t1", "t1:0", 5000.0), _sample("t1", "t1:1", 200.0)],
        "t2": [_sample("t2", "t2:0", 100.0)],
        "t3": [_sample("t3", "t3:0", 4800.0), _sample("t3", "t3:1", 300.0)],
    }
    common_fp = {
        ("t1", 0): 8000.0,
        ("t2", 0): 30000.0,
        ("t3", 0): 15000.0,
    }
    selected = []
    for eval_latency, eval_tokens in ((4500.0, 5000.0), (30.0, 80000.0)):
        samples = dict(common)
        samples["t0"] = [_sample("t0", "t0:0", eval_latency)]
        footprints = _footprints({**common_fp, ("t0", 0): eval_tokens})
        decisions = score_decisions(
            samples, ["t0", "t1", "t2", "t3"], footprints, _cfg((kv,))
        )
        row = next(r for r in decisions if r["task_id"] == "t0")
        selected.append((row["fixed_lambda_ms"], row["fixed_trigger_ms"]))
    assert selected[0] == selected[1]


def test_reference_tokens_come_from_fit_fold_only() -> None:
    # The anchor is the fit-fold mean footprint; the eval call's own footprint
    # must not shift it.
    kv = 3500.0
    samples = {
        "t0": [_sample("t0", "t0:0", 4000.0)],
        "t1": [_sample("t1", "t1:0", 5000.0)],
        "t2": [_sample("t2", "t2:0", 100.0)],
        "t3": [_sample("t3", "t3:0", 4800.0)],
    }
    footprints = _footprints(
        {("t0", 0): 99000.0, ("t1", 0): 10000.0, ("t2", 0): 20000.0, ("t3", 0): 30000.0}
    )
    decisions = score_decisions(samples, ["t0", "t1", "t2", "t3"], footprints, _cfg((kv,)))
    row = next(r for r in decisions if r["task_id"] == "t0")
    # fold 1 evaluates {t0, t2}; its profile fold is {t1, t3} -> mean(10k, 30k).
    assert row["reference_tokens"] == pytest.approx(20000.0)
    assert row["true_price_ms"] == pytest.approx(kv * 99000.0 / 20000.0)


def test_candidate_grid_always_contains_the_panel_cell() -> None:
    # Guarantees best-fixed is never worse than the SHIPPED constant-price
    # policy, so the headroom is a conservative lower bound.
    prices = np.asarray([100.0, 200.0, 400.0])
    candidates = fixed_lambda_candidates(prices, kv_cost_ms=3500.0, grid=5)
    assert 3500.0 in set(candidates)


def test_fixed_lambda_ties_resolve_to_the_status_quo() -> None:
    # A node with no samples yields the deadline trigger for every believed
    # price, so all candidates tie -- the panel cell must win.
    empty = _node({"a": []})
    kv = 3500.0
    candidates = np.asarray([500.0, 3500.0, 9000.0])
    chosen, _ = select_fixed_lambda_ms(
        [(empty, 1000.0, 3500.0)],
        candidates=candidates,
        kv_cost_ms=kv,
        guard_ms=0.0,
        restore_cost_fraction=_RHO,
    )
    assert chosen == pytest.approx(kv)


def test_trigger_cache_distinguishes_tool_prior_nodes() -> None:
    # Tool-level prior nodes share metadata but have different sample lists.
    # Aliasing them makes the first tool's trigger leak into later tools.
    fast = _node({"fast": [100.0, 200.0, 300.0, 400.0]})
    slow = _node({"slow": [100.0, 1000.0, 5000.0, 9000.0]})
    chosen, _ = select_fixed_lambda_ms(
        [(fast, 200.0, 500.0), (slow, 1000.0, 500.0)],
        candidates=np.asarray([500.0, 1500.0, 3500.0, 7000.0]),
        kv_cost_ms=3500.0,
        guard_ms=0.0,
        restore_cost_fraction=_RHO,
    )
    assert chosen == 500.0


# --------------------------------------------------------------------------- #
# Trigger reuses the certified optimizer under the certified price semantics.
# --------------------------------------------------------------------------- #
def test_believed_trigger_matches_hazard_recheck_on_its_own_domain() -> None:
    # When the common domain coincides with the believed price's own threshold,
    # the local optimizer must reproduce the certified one exactly.
    node = _node({"a": [200.0, 4800.0], "b": [900.0, 6000.0]})
    price = 2750.0
    assert believed_trigger_ms(
        node,
        believed_price_ms=price,
        domain_max_ms=price,
        guard_ms=0.0,
        restore_cost_fraction=_RHO,
    ) == pytest.approx(
        hazard_recheck_ms(
            node.values,
            threshold_ms=price,
            kv_cost_ms=price,
            restore_cost_ms=_RHO * price,
        )
    )


def test_empty_node_inherits_conservative_deadline() -> None:
    empty = _node({"a": []})
    assert believed_trigger_ms(
        empty,
        believed_price_ms=1000.0,
        domain_max_ms=1025.0,
        guard_ms=25.0,
        restore_cost_fraction=_RHO,
    ) == pytest.approx(1025.0)


def test_trigger_domain_does_not_widen_with_the_believed_price() -> None:
    # The defect: a higher believed price used to buy a larger search domain.
    # Both arms must stay inside the common domain regardless of belief.
    node = _node({"a": [200.0, 4800.0], "b": [900.0, 6000.0]})
    domain = 3500.0
    for believed in (350.0, 3500.0, 35000.0):
        trigger = believed_trigger_ms(
            node,
            believed_price_ms=believed,
            domain_max_ms=domain,
            guard_ms=0.0,
            restore_cost_fraction=_RHO,
        )
        assert 0.0 <= trigger <= domain


def test_footprint_arm_is_never_beaten_on_the_common_domain() -> None:
    # Ports the reviewer's random-node probe. Under the OLD code this failed on
    # ~9.2% of nodes (worst 261 ms/call) because lambda_bar > lambda_i bought a
    # wider domain. On a common domain the footprint arm optimizes the true-price
    # objective over exactly the set the fixed arm draws from, so it cannot lose.
    rng = np.random.default_rng(0)
    for _ in range(600):
        size = int(rng.integers(2, 12))
        values = (rng.lognormal(mean=6.0, sigma=2.0, size=size) + 1.0).tolist()
        node = _node({"a": values[: size // 2 or 1], "b": values[size // 2 or 1 :]})
        kv = float(rng.choice([500.0, 1500.0, 3500.0, 5000.0]))
        domain = kv  # guard 0 at the certified operating point
        true_price = kv * float(rng.uniform(0.1, 5.0))
        fixed_lambda = kv * float(rng.uniform(0.1, 5.0))
        triggers = [
            believed_trigger_ms(
                node,
                believed_price_ms=price,
                domain_max_ms=domain,
                guard_ms=0.0,
                restore_cost_fraction=_RHO,
            )
            for price in (true_price, fixed_lambda)
        ]
        # Node-expected utility at the TRUE price, the quantity the bound is over.
        footprint_value, fixed_value = (
            float(
                np.mean(
                    utility_matrix(
                        np.asarray(node.values, dtype=float),
                        np.asarray([trigger], dtype=float),
                        threshold_ms=true_price,
                        kv_cost_ms=true_price,
                        restore_cost_ms=_RHO * true_price,
                    )
                )
            )
            for trigger in triggers
        )
        assert footprint_value >= fixed_value - 1e-9, (
            f"ceiling violated: footprint {footprint_value} < fixed {fixed_value} "
            f"(kv={kv}, true={true_price}, fixed={fixed_lambda})"
        )


# --------------------------------------------------------------------------- #
# Frozen kill arithmetic, both directions.
# --------------------------------------------------------------------------- #
def _decisions_with_headroom(per_task_ms: dict[str, float], kv: float) -> list[dict]:
    return [
        {
            "sample_id": f"{task}:0",
            "task_id": task,
            "outer_fold": "f1",
            "kv_cost_ms": kv,
            "headroom_ms": ms,
            "footprint_priced_utility_ms": ms,
            "fixed_utility_ms": 0.0,
            "fixed_lambda_ms": kv,
            "true_price_ms": kv,
            "footprint_priced_fired": True,
            "fixed_fired": False,
        }
        for task, ms in per_task_ms.items()
    ]


def test_kill_drops_when_headroom_below_banked_effect() -> None:
    kv = 3500.0
    # 100 s/277 total, well below the banked 156 s/277, though clearly positive.
    decisions = _decisions_with_headroom({f"t{i}": 5000.0 for i in range(20)}, kv)
    results = summarize(decisions, _cfg((kv,)), task_count=20)
    assert results["headline_headroom_seconds_per_277"] == pytest.approx(100.0)
    assert results["headline_exceeds_banked"] is False
    assert results["verdict"] == "DROP"


def test_kill_proceeds_when_headroom_materially_exceeds_banked_effect() -> None:
    kv = 3500.0
    # 400 s/277, comfortably above the banked 156 s/277, and unanimous in sign
    # so the permutation label is positive.
    decisions = _decisions_with_headroom({f"t{i}": 20000.0 for i in range(20)}, kv)
    results = summarize(decisions, _cfg((kv,)), task_count=20)
    assert results["headline_headroom_seconds_per_277"] == pytest.approx(400.0)
    assert results["headline_exceeds_banked"] is True
    assert results["headline_ci_excludes_zero"] is True
    assert results["verdict"] == "PROCEED"


def test_large_headroom_is_underpowered_without_a_ci_excluding_zero() -> None:
    kv = 3500.0
    # Mean far above the banked effect but the sign is not stable across tasks.
    # The CI spans the bar, so under the power rule this is NOT a DROP -- the
    # direction stays open.
    per_task = {f"t{i}": (400000.0 if i % 2 else -300000.0) for i in range(20)}
    results = summarize(_decisions_with_headroom(per_task, kv), _cfg((kv,)), task_count=20)
    assert results["headline_exceeds_banked"] is True
    assert results["headline_ci_excludes_zero"] is False
    assert results["verdict"] == "UNDERPOWERED"
    assert results["power_rule"]["direction_closed"] is False


def test_underpowered_when_point_below_bar_but_ci_spans_it() -> None:
    kv = 3500.0
    # Point estimate 100 s/277 (below the 156 bar) but a wide spread across
    # tasks, so the CI cannot rule the bar out. Closing the direction here would
    # be claiming absence of an effect the screen could not have detected.
    per_task = {f"t{i}": (20000.0 if i % 2 else -10000.0) for i in range(20)}
    results = summarize(_decisions_with_headroom(per_task, kv), _cfg((kv,)), task_count=20)
    power = results["power_rule"]
    assert results["headline_headroom_seconds_per_277"] == pytest.approx(100.0)
    assert power["ci_high_seconds_per_277"] > power["bar_seconds_per_277"]
    assert power["ci_upper_below_bar"] is False
    assert results["verdict"] == "UNDERPOWERED"
    assert power["direction_closed"] is False


def test_drop_requires_the_ci_upper_bound_below_the_bar() -> None:
    kv = 3500.0
    # Identical per-task contributions -> degenerate CI at 100 s/277, entirely
    # below the 156 bar. This is the ONLY shape that closes the direction.
    results = summarize(
        _decisions_with_headroom({f"t{i}": 5000.0 for i in range(20)}, kv),
        _cfg((kv,)),
        task_count=20,
    )
    power = results["power_rule"]
    assert power["ci_high_seconds_per_277"] < power["bar_seconds_per_277"]
    assert results["verdict"] == "DROP"
    assert power["direction_closed"] is True


def test_panel_coherence_is_descriptive_and_never_binding() -> None:
    kv, other = 3500.0, 5000.0
    # A sign-flipping panel alongside a headline whose CI sits below the bar:
    # the coherence flag fires but the verdict still follows the CI rule.
    decisions = _decisions_with_headroom({f"t{i}": 5000.0 for i in range(20)}, kv)
    decisions += _decisions_with_headroom({f"t{i}": -5000.0 for i in range(20)}, other)
    results = summarize(decisions, _cfg((kv, other)), task_count=20)
    assert results["panel_coherence"]["binding"] is False
    assert results["panel_coherence"]["negative_cells"] == 1
    assert results["verdict"] == "DROP"


def test_banked_threshold_is_configurable_and_moves_the_verdict() -> None:
    kv = 3500.0
    decisions = _decisions_with_headroom({f"t{i}": 20000.0 for i in range(20)}, kv)
    strict = summarize(
        decisions, _cfg((kv,), banked_seconds_per_277=1000.0), task_count=20
    )
    assert strict["verdict"] == "DROP"


def test_secondary_kv5000_row_is_reported_but_never_binding() -> None:
    kv, secondary = 3500.0, 5000.0
    # Headline stays below its banked 156; secondary clears its banked 317.9.
    decisions = _decisions_with_headroom({f"t{i}": 5000.0 for i in range(20)}, kv)
    decisions += _decisions_with_headroom(
        {f"t{i}": 20000.0 for i in range(20)}, secondary
    )
    results = summarize(decisions, _cfg((kv, secondary)), task_count=20)
    row = results["secondary_readout"]
    assert row["binding"] is False
    assert row["kv_cost_ms"] == secondary
    assert row["headroom_seconds_per_277"] == pytest.approx(400.0)
    assert row["banked_seconds_per_277"] == pytest.approx(317.9)
    assert row["exceeds_banked"] is True
    # ...and the verdict still follows the 3500-to-3500 comparison alone.
    assert results["verdict"] == "DROP"


def test_secondary_row_absent_when_panel_lacks_kv5000() -> None:
    kv = 3500.0
    decisions = _decisions_with_headroom({f"t{i}": 5000.0 for i in range(5)}, kv)
    assert summarize(decisions, _cfg((kv,)), task_count=5)["secondary_readout"] is None


def test_summarize_refuses_a_panel_without_the_headline_cell() -> None:
    decisions = _decisions_with_headroom({"t0": 1.0}, 1000.0)
    with pytest.raises(ValueError, match="missing the headline kv cell"):
        summarize(decisions, _cfg((1000.0,)), task_count=1)


def test_headroom_identity_holds_per_cell() -> None:
    kv = 3500.0
    decisions = _decisions_with_headroom({f"t{i}": 5000.0 for i in range(10)}, kv)
    cell = summarize(decisions, _cfg((kv,)), task_count=10)["cells"][0]
    assert cell["headroom_seconds_per_277"] == pytest.approx(
        cell["footprint_priced_seconds_per_277"] - cell["best_fixed_seconds_per_277"]
    )


def test_early_fire_excludes_deadline_fires() -> None:
    # The mandatory-reported rate must count only swaps STRICTLY before the
    # call's own deadline; a thin node falls back to the deadline trigger, which
    # fires but adds nothing over deadline_only.
    kv = 3500.0
    samples = {
        "t0": [_sample("t0", "t0:0", 400000.0)],  # long enough to fire on anything
        "t1": [_sample("t1", "t1:0", 5000.0)],
        "t2": [_sample("t2", "t2:0", 100.0)],
        "t3": [_sample("t3", "t3:0", 4800.0)],
    }
    footprints = _footprints(
        {("t0", 0): 20000.0, ("t1", 0): 20000.0, ("t2", 0): 20000.0, ("t3", 0): 20000.0}
    )
    decisions = score_decisions(samples, ["t0", "t1", "t2", "t3"], footprints, _cfg((kv,)))
    row = next(r for r in decisions if r["task_id"] == "t0")
    threshold = row["true_price_ms"]
    assert row["footprint_priced_fired"] is True
    assert row["footprint_priced_fired_early"] == (row["footprint_priced_trigger_ms"] < threshold)
    for r in decisions:
        # early fire implies fire, never the converse.
        assert not r["footprint_priced_fired_early"] or r["footprint_priced_fired"]
        assert not r["fixed_fired_early"] or r["fixed_fired"]


def test_footprint_growth_stats_are_computed_not_constants() -> None:
    # The mechanism figure must come from the corpus every run. Two different
    # corpora must yield different numbers, and the arithmetic must be right.
    samples = {
        "t0": [
            _sample("t0", "t0:0", 100.0, iteration=0),
            _sample("t0", "t0:1", 200.0, iteration=1),
        ],
        "t1": [_sample("t1", "t1:0", 300.0, iteration=0)],  # single call, no ratio
    }
    footprints = _footprints({("t0", 0): 1000.0, ("t0", 1): 4000.0, ("t1", 0): 2000.0})
    stats = footprint_growth_stats(samples, ["t0", "t1"], footprints)
    assert stats["corpus_min_tokens"] == 1000.0
    assert stats["corpus_max_tokens"] == 4000.0
    assert stats["corpus_spread_ratio"] == pytest.approx(4.0)
    # Only t0 has >=2 calls, so exactly one within-task ratio (4000/1000).
    assert stats["within_task_growth_tasks"] == 1
    assert stats["within_task_growth_median"] == pytest.approx(4.0)

    doubled = _footprints({("t0", 0): 1000.0, ("t0", 1): 8000.0, ("t1", 0): 2000.0})
    assert footprint_growth_stats(samples, ["t0", "t1"], doubled)[
        "within_task_growth_median"
    ] == pytest.approx(8.0)


def test_footprint_growth_handles_corpus_with_no_multi_call_task() -> None:
    samples = {"t0": [_sample("t0", "t0:0", 100.0)]}
    stats = footprint_growth_stats(samples, ["t0"], _footprints({("t0", 0): 5000.0}))
    assert stats["within_task_growth_tasks"] == 0
    assert stats["within_task_growth_median"] is None


# --------------------------------------------------------------------------- #
# Determinism.
# --------------------------------------------------------------------------- #
def test_scoring_and_certificate_are_deterministic() -> None:
    kv = 3500.0
    samples = {
        "t0": [_sample("t0", "t0:0", 4000.0)],
        "t1": [_sample("t1", "t1:0", 5000.0), _sample("t1", "t1:1", 220.0)],
        "t2": [_sample("t2", "t2:0", 100.0)],
        "t3": [_sample("t3", "t3:0", 4800.0), _sample("t3", "t3:1", 330.0)],
    }
    footprints = _footprints(
        {("t0", 0): 12000.0, ("t1", 0): 8000.0, ("t2", 0): 25000.0, ("t3", 0): 40000.0}
    )
    cfg = _cfg((kv,))
    first = score_decisions(samples, ["t0", "t1", "t2", "t3"], footprints, cfg)
    second = score_decisions(samples, ["t0", "t1", "t2", "t3"], footprints, cfg)
    assert first == second
    assert summarize(first, cfg, task_count=4) == summarize(second, cfg, task_count=4)


def test_final_rejects_a_different_manifest_even_if_it_claims_277_tasks(
    tmp_path: Path,
) -> None:
    other_manifest = tmp_path / "manifest.json"
    other_manifest.write_text(
        json.dumps({"expected_task_count": 277}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="--final requires the frozen manifest"):
        main(["--manifest", str(other_manifest), "--final"])
