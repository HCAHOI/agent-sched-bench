"""Unit tests for the WTN Stage-2 certified-decision-replay runner.

Synthetic rows and cert fixtures so cross-fit isolation, the P_0 empty-hook
byte-identity, the kill arithmetic, and permutation determinism are checkable
without any replay corpus on disk.
"""

from __future__ import annotations

from typing import Any

from scripts.analyze_wrapper_transparency import ScreenConfig, screen_transparency
from scripts.run_wtn_stage2 import (
    _certificate,
    _chains_from_samples,
    _divergence_by_cost,
    compute_verdict,
)
from trace_collect.tool_latency_dataset import ToolLatencySample
from trace_collect.tool_latency_offline_probe import evaluate_offline_probe_clock


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #
def _sample(task: str, command: str, latency_ms: float, index: int) -> ToolLatencySample:
    return ToolLatencySample(
        sample_id=f"{task}:{index}",
        source_trace=f"trace-{task}",
        task_id=task,
        agent_id="a",
        instance_id=task,
        iteration=index,
        action_id=f"{task}-{index}",
        tool_name="exec",
        tool_call_id=f"{task}-{index}",
        tool_ts_start=float(index),
        tool_ts_end=float(index) + latency_ms / 1000.0,
        latency_ms=latency_ms,
        success=True,
        reported_duration_ms=None,
        tool_args={"command": command},
    )


def _row(task: str, command: str, latency_ms: float, index: int) -> dict[str, Any]:
    return _sample(task, command, latency_ms, index).to_json_obj()


# --------------------------------------------------------------------------- #
# Cross-fit isolation: the transparent set is learned from FIT rows only.
# --------------------------------------------------------------------------- #
def test_screen_learns_from_fit_rows_only() -> None:
    # Profile (fit) rows use `cd` as the only wrapper; an eval-only verb never
    # appears in the fit-fold screen's candidate universe, proving the learned
    # set cannot leak eval structure.
    profile_samples = [
        _sample(f"task{t}", f"cd /home/task{t} && python mod.py", 1000.0 + t, k)
        for t in range(6)
        for k in range(4)
    ]
    eval_only = [_sample("evaltask", "frobnicate x && python mod.py", 999.0, 0)]
    screen = screen_transparency(
        _chains_from_samples(profile_samples, command_field="command"),
        ScreenConfig(tolerance=0.05, min_tasks=2, min_chains=4, inner_folds=2),
    )
    assert "cd" in screen.universe
    assert "frobnicate" not in screen.universe
    # Sanity: the eval-only sample really does carry the leaked verb as a
    # candidate, so its absence above is isolation, not a parsing artifact.
    leaked = _chains_from_samples(eval_only, command_field="command")
    assert leaked[0].parent_command.startswith("frobnicate")


def test_chains_carry_latency_as_total_and_command_text() -> None:
    samples = [_sample("t", "cd /a && python x", 1234.0, 0)]
    chains = _chains_from_samples(samples, command_field="command")
    assert chains[0].parent_total_ms == 1234.0
    assert chains[0].parent_command == "cd /a && python x"
    assert chains[0].segments == ()
    # A row without a usable command groups at the tool level (empty command).
    no_cmd = ToolLatencySample(
        sample_id="s", source_trace="tr", task_id="t", agent_id="a",
        instance_id="t", iteration=0, action_id="x", tool_name="exec",
        tool_call_id="x", tool_ts_start=0.0, tool_ts_end=1.0, latency_ms=1.0,
        success=True, reported_duration_ms=None, tool_args=None,
    )
    assert _chains_from_samples([no_cmd], command_field="command")[0].parent_command == ""


# --------------------------------------------------------------------------- #
# P_0 byte-identity: threading the empty hook changes nothing.
# --------------------------------------------------------------------------- #
def _disjoint_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    profile = [
        _row(f"p{t}", f"cd /home/p{t} && python mod_{t % 2}.py", 1000.0 + 100 * t, k)
        for t in range(4)
        for k in range(3)
    ]
    evaluation = [
        _row(f"e{t}", f"cd /home/e{t} && python mod_{t % 2}.py", 1200.0 + 90 * t, k)
        for t in range(2)
        for k in range(3)
    ]
    return evaluation, profile


def test_p0_arm_empty_hook_is_byte_identical() -> None:
    evaluation, profile = _disjoint_corpus()
    common = dict(
        kv_costs_ms=[1000.0, 2000.0],
        guard_ms=0.0,
        inner_folds=2,
        min_tool_history=1,
        min_profile_tasks=1,
        command_field="command",
        max_prefix_depth=4,
        skip_leading_cd=False,
        restore_cost_fraction=0.94,
    )
    baseline = evaluate_offline_probe_clock(evaluation, profile_rows=profile, **common)
    hooked = evaluate_offline_probe_clock(
        evaluation,
        profile_rows=profile,
        transparent_wrappers=frozenset(),
        **common,
    )
    assert hooked["decisions"] == baseline["decisions"]


def test_transparent_set_changes_decisions_when_nonempty() -> None:
    # A non-empty transparent set must actually alter the keying (else the whole
    # comparison is degenerate). We only require the decisions to differ.
    evaluation, profile = _disjoint_corpus()
    common = dict(
        kv_costs_ms=[1000.0, 2000.0],
        guard_ms=0.0,
        inner_folds=2,
        min_tool_history=1,
        min_profile_tasks=1,
        command_field="command",
        max_prefix_depth=4,
        skip_leading_cd=False,
        restore_cost_fraction=0.94,
    )
    base = evaluate_offline_probe_clock(evaluation, profile_rows=profile, **common)
    normed = evaluate_offline_probe_clock(
        evaluation,
        profile_rows=profile,
        transparent_wrappers=frozenset({"cd"}),
        **common,
    )
    assert normed["decisions"] != base["decisions"]


# --------------------------------------------------------------------------- #
# Kill arithmetic.
# --------------------------------------------------------------------------- #
def _cert(labels: dict[float, str], delta: float = -5.0) -> dict[str, Any]:
    return {
        "points": {
            str(cost): {
                "permutation_label": label,
                "permutation_p_positive": 0.001 if label == "positive" else 0.5,
                "permutation_p_harmful": 0.001 if label == "harmful" else 0.5,
                "paired_delta_ms": delta,
            }
            for cost, label in labels.items()
        }
    }


def _screen_ok() -> dict[str, Any]:
    return {"candidate_present": True, "cd_transparent_every_fold": True, "per_fold": []}


def test_verdict_certified_when_diverging_cell_positive() -> None:
    cert = _cert({3500.0: "inconclusive", 5000.0: "positive"})
    verdict = compute_verdict(
        cert,
        {5000.0: 12, 3500.0: 0},
        _screen_ok(),
        headline_costs_ms=(3500.0, 5000.0),
    )
    assert verdict["verdict"] == "CERTIFIED"
    assert verdict["diverging_positive_costs_ms"] == [5000.0]


def test_verdict_kills_when_positive_cell_did_not_diverge() -> None:
    # Positive label but zero diverged samples => not a certification "where they
    # diverge".
    cert = _cert({3500.0: "inconclusive", 5000.0: "positive"})
    verdict = compute_verdict(
        cert, {5000.0: 0, 3500.0: 0}, _screen_ok(), headline_costs_ms=(3500.0, 5000.0)
    )
    assert verdict["verdict"] == "KILL"


def test_verdict_kills_when_cd_not_transparent_every_fold() -> None:
    cert = _cert({3500.0: "positive", 5000.0: "positive"})
    screen = {"candidate_present": True, "cd_transparent_every_fold": False, "per_fold": []}
    verdict = compute_verdict(
        cert, {3500.0: 5, 5000.0: 5}, screen, headline_costs_ms=(3500.0, 5000.0)
    )
    assert verdict["verdict"] == "KILL"
    assert any("Stage-1" in reason for reason in verdict["killed_reasons"])


def test_verdict_kills_on_harmful_headline_cell() -> None:
    cert = _cert({3500.0: "positive", 5000.0: "harmful"})
    verdict = compute_verdict(
        cert, {3500.0: 5, 5000.0: 5}, _screen_ok(), headline_costs_ms=(3500.0, 5000.0)
    )
    assert verdict["verdict"] == "KILL"
    assert verdict["any_headline_harmful"] is True


# --------------------------------------------------------------------------- #
# Permutation determinism.
# --------------------------------------------------------------------------- #
def _merged_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in range(8):
        latency = 1500.0 + 400.0 * t
        for cost in (3500.0, 5000.0):
            rows.append(
                {
                    "sample_id": f"s{t}",
                    "task_id": f"task{t}",
                    "tool_name": "exec",
                    "kv_cost_ms": cost,
                    "threshold_ms": cost,
                    "latency_ms": latency,
                    "outer_fold": f"f{t % 3 + 1}",
                    "p0_trigger_ms": cost,  # deadline
                    "pnew_trigger_ms": cost * 0.5,  # earlier trigger
                    "oracle_trigger_ms": cost * 0.6,
                }
            )
    return rows


def test_certificate_is_deterministic_with_fixed_seed() -> None:
    rows = _merged_rows()
    kwargs = dict(
        treatment_field="pnew_trigger_ms",
        baseline_field="p0_trigger_ms",
        costs_ms=[3500.0, 5000.0],
        replicates=2000,
        confidence_level=0.95,
        seed=0,
        restore_cost_fraction=0.94,
    )
    first = _certificate(rows, **kwargs)
    second = _certificate(rows, **kwargs)
    for cost in ("3500.0", "5000.0"):
        assert (
            first["points"][cost]["permutation_p_positive"]
            == second["points"][cost]["permutation_p_positive"]
        )
        assert (
            first["points"][cost]["permutation_label"]
            == second["points"][cost]["permutation_label"]
        )
        assert (
            first["points"][cost]["paired_delta_ms"]
            == second["points"][cost]["paired_delta_ms"]
        )


def test_divergence_counts_only_differing_triggers() -> None:
    rows = _merged_rows()
    diverged = _divergence_by_cost(
        rows, treatment_field="pnew_trigger_ms", baseline_field="p0_trigger_ms"
    )
    # Every sample's pnew trigger differs from the p0 deadline at both costs.
    assert diverged == {3500.0: 8, 5000.0: 8}
