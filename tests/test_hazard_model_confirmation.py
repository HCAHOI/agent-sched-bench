from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trace_collect.restore_cost_analysis import (
    _merge_hazard_rows,
    run_hazard_model_confirmation,
)
from trace_collect.tool_latency_hazard_eval import evaluate_hazard_model_clock
from trace_collect.tool_latency_offline_probe import balanced_task_folds
from trace_collect.tool_latency_survival_features import (
    SurvivalFeatureSpec,
    fit_feature_encoder,
    iter_causal_row_features,
)
from trace_collect.tool_latency_utility_clock import trigger_policy_utility_ms


# The synthetic corpus separates on a single binary feature: a fast tool always
# returns 10 ms (well below the 100 ms threshold, so an early swap only ever
# loses) and a slow tool always returns 150 ms (inside the (threshold,
# threshold+kv) band, so firing early hides the full kv=100 cost). The model
# learns tool -> latency, so the slow tool's optimal trigger is 50 ms (any k<=50
# hides the whole cost on a 150 ms call; ties resolve to the latest k) with a
# normalized margin of 1.0, and the fast tool never fires on its own 10 ms call.
FAST_MS = 10.0
SLOW_MS = 150.0
COST_MS = 100.0
THRESHOLD_MS = 100.0  # guard_ms = 0
TOOL_SPEC = SurvivalFeatureSpec(
    command_field=None,
    use_tool_identity=True,
    use_command_prefix=False,
    use_within_task_history=False,
    use_task_aggregates=False,
)


def _row(
    sample_id: str,
    *,
    task_id: str,
    tool_name: str,
    latency_ms: float,
    ts_start: float,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source_trace": f"trace-{task_id}",
        "task_id": task_id,
        "tool_name": tool_name,
        "latency_ms": latency_ms,
        "tool_ts_start": ts_start,
        "tool_ts_end": ts_start + latency_ms / 1000.0,
    }


def _corpus(task_ids: list[str], *, fast: int, slow: int) -> list[dict[str, Any]]:
    """Each task issues ``fast`` fast calls then ``slow`` slow calls."""

    rows: list[dict[str, Any]] = []
    for offset, task_id in enumerate(task_ids):
        base = offset * 100_000.0
        ts = base
        for call in range(fast):
            rows.append(
                _row(
                    f"{task_id}-fast-{call}",
                    task_id=task_id,
                    tool_name="fast",
                    latency_ms=FAST_MS,
                    ts_start=ts,
                )
            )
            ts += 1000.0
        for call in range(slow):
            rows.append(
                _row(
                    f"{task_id}-slow-{call}",
                    task_id=task_id,
                    tool_name="slow",
                    latency_ms=SLOW_MS,
                    ts_start=ts,
                )
            )
            ts += 1000.0
    return rows


def _evaluate(
    eval_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    *,
    fractions: tuple[float, ...] = (0.0,),
    **kwargs: Any,
) -> dict[str, Any]:
    """Score the separable corpus with a pinned tiny L2 for determinism."""

    return evaluate_hazard_model_clock(
        eval_rows,
        profile_rows=profile_rows,
        kv_costs_ms=[COST_MS],
        guard_ms=0.0,
        inner_folds=2,
        spec=TOOL_SPEC,
        num_intervals=8,
        restore_cost_fractions=list(fractions),
        l2_penalty=1e-4,
        **kwargs,
    )


# --- Evaluator: separation, trigger, gate, hand-computed delta ---------------


def test_evaluate_hazard_model_clock_fires_early_only_on_slow_tool() -> None:
    profile_rows = _corpus(
        [f"p{index}" for index in range(4)], fast=3, slow=3
    )
    eval_rows = _corpus(["e0", "e1"], fast=2, slow=2)

    result = _evaluate(eval_rows, profile_rows)

    fraction_result = result["by_restore_cost_fraction"]["0.0"]
    decisions = fraction_result["decisions"]
    slow = [row for row in decisions if row["tool_name"] == "slow"]
    fast = [row for row in decisions if row["tool_name"] == "fast"]
    assert len(slow) == 4 and len(fast) == 4

    # Slow tool: the candidate fires no later than 50 ms (any k<=50 hides the
    # full cost on a 150 ms call) with a near-maximal margin, and clearing the
    # guard the deployed gated trigger keeps firing early on the call itself.
    for row in slow:
        assert 0.0 < row["hazard_trigger_ms"] <= 50.0 + 1e-9
        assert row["hazard_margin_normalized"] > 0.5
        assert row["offline_gated_hazard_trigger_ms"] == pytest.approx(
            row["hazard_trigger_ms"]
        )
        assert row["latency_ms"] > row["offline_gated_hazard_trigger_ms"]

    # Fast tool: its 10 ms call never exceeds any candidate trigger (>= 10), so
    # it never fires early whether or not the guard admits it.
    for row in fast:
        assert row["latency_ms"] <= row["hazard_trigger_ms"] + 1e-9
        fired = (
            row["latency_ms"] > row["offline_gated_hazard_trigger_ms"]
            and row["offline_gated_hazard_trigger_ms"] < row["threshold_ms"]
        )
        assert not fired

    # hazard_vs_deadline hand math at rho=0: each of the four slow calls hides
    # the full cost (+100 vs the deadline's 0); every fast call contributes 0.
    hazard_delta = sum(
        trigger_policy_utility_ms(
            row["latency_ms"],
            row["hazard_trigger_ms"],
            threshold_ms=row["threshold_ms"],
            kv_cost_ms=row["kv_cost_ms"],
        )
        - trigger_policy_utility_ms(
            row["latency_ms"],
            row["deadline_trigger_ms"],
            threshold_ms=row["threshold_ms"],
            kv_cost_ms=row["kv_cost_ms"],
        )
        for row in decisions
    )
    assert hazard_delta == pytest.approx(4 * 100.0)

    guard = fraction_result["calibration_guard"]["selected_guard_normalized"]
    assert guard is not None and 0.0 <= guard < 1.0


def test_evaluate_hazard_model_clock_fits_are_fraction_independent() -> None:
    # The model never sees rho, so scoring [0.0, 0.5] in one amortized call must
    # reproduce the 0.0 slice of a call that only requests [0.0] bit for bit
    # (same fits -> same masses -> same triggers/guard at a given fraction).
    profile_rows = _corpus([f"p{index}" for index in range(4)], fast=3, slow=3)
    eval_rows = _corpus(["e0", "e1"], fast=2, slow=2)
    multi = _evaluate(eval_rows, profile_rows, fractions=(0.0, 0.5))
    single = _evaluate(eval_rows, profile_rows, fractions=(0.0,))
    assert (
        multi["by_restore_cost_fraction"]["0.0"]
        == single["by_restore_cost_fraction"]["0.0"]
    )
    # The two fractions genuinely differ somewhere (0.5 charges short-call fires).
    assert "0.5" in multi["by_restore_cost_fraction"]


def test_evaluate_hazard_model_clock_use_candidate_predicate() -> None:
    # Reuse the separable corpus: the guard admits the high-margin slow candidate
    # (margin 1.0 clears the small guard) and its gated trigger equals the
    # candidate, while every fast decision falls back to the deadline trigger.
    result = _evaluate(
        _corpus(["e0", "e1"], fast=2, slow=2),
        _corpus([f"p{index}" for index in range(4)], fast=3, slow=3),
    )
    fraction_result = result["by_restore_cost_fraction"]["0.0"]
    guard = fraction_result["calibration_guard"]["selected_guard_normalized"]
    for row in fraction_result["decisions"]:
        candidate = row["hazard_trigger_ms"]
        threshold = row["threshold_ms"]
        margin = row["hazard_margin_normalized"]
        use_candidate = (
            guard is not None and candidate < threshold and margin > guard
        )
        expected = candidate if use_candidate else threshold
        assert row["offline_gated_hazard_trigger_ms"] == pytest.approx(expected)


def test_evaluate_hazard_model_clock_gbm_fires_early_only_on_slow_tool() -> None:
    # The GBM family drives the identical trigger/gate seam. On the separable
    # corpus it must recover the same qualitative decision as the logistic arm:
    # the slow tool swaps early (within the deadline) and the fast tool waits.
    profile_rows = _corpus([f"p{index}" for index in range(8)], fast=4, slow=4)
    eval_rows = _corpus(["e0", "e1"], fast=2, slow=2)

    result = evaluate_hazard_model_clock(
        eval_rows,
        profile_rows=profile_rows,
        kv_costs_ms=[COST_MS],
        guard_ms=0.0,
        inner_folds=2,
        spec=TOOL_SPEC,
        num_intervals=8,
        restore_cost_fractions=[0.0],
        model_family="gbm",
        seed=0,
    )
    assert result["model_family"] == "gbm"
    # The GBM has no L2 penalty; the logistic-only bookkeeping is None/False.
    assert result["l2_penalty"] is None
    assert result["l2_selected_on_full_profile"] is False

    decisions = result["by_restore_cost_fraction"]["0.0"]["decisions"]
    slow = [row for row in decisions if row["tool_name"] == "slow"]
    fast = [row for row in decisions if row["tool_name"] == "fast"]
    assert len(slow) == 4 and len(fast) == 4
    for row in slow:
        # Firing anywhere in (0, threshold) on a 150 ms call hides swap cost.
        assert 0.0 < row["hazard_trigger_ms"] < row["threshold_ms"]
        assert row["hazard_margin_normalized"] > 0.0
        assert row["latency_ms"] > row["hazard_trigger_ms"]
    for row in fast:
        # The 10 ms fast call never exceeds its own candidate trigger.
        assert row["latency_ms"] <= row["hazard_trigger_ms"] + 1e-9


def test_evaluate_hazard_model_clock_gbm_is_deterministic() -> None:
    profile_rows = _corpus([f"p{index}" for index in range(8)], fast=4, slow=4)
    eval_rows = _corpus(["e0", "e1"], fast=2, slow=2)
    kwargs = dict(
        profile_rows=profile_rows,
        kv_costs_ms=[COST_MS],
        guard_ms=0.0,
        inner_folds=2,
        spec=TOOL_SPEC,
        num_intervals=8,
        restore_cost_fractions=[0.0],
        model_family="gbm",
        seed=3,
    )
    first = evaluate_hazard_model_clock(eval_rows, **kwargs)
    second = evaluate_hazard_model_clock(eval_rows, **kwargs)
    assert (
        first["by_restore_cost_fraction"]["0.0"]["decisions"]
        == second["by_restore_cost_fraction"]["0.0"]["decisions"]
    )


def test_evaluate_hazard_model_clock_rejects_task_overlap() -> None:
    rows = _corpus(["shared"], fast=2, slow=2)
    with pytest.raises(AssertionError, match="profile and eval tasks overlap"):
        _evaluate(rows, rows)


def test_hazard_calibration_block_shape_is_json_serializable() -> None:
    result = _evaluate(
        _corpus(["e0", "e1"], fast=2, slow=2),
        _corpus([f"p{index}" for index in range(4)], fast=3, slow=3),
        calibration_edge_stride=2,
        reliability_bin_count=4,
    )
    calibration = result["hazard_calibration"]
    # A stride of 2 over the 8 finite edges yields 4 checkpoints; each carries a
    # predicted survival, an observed exceedance fraction, and a count.
    assert len(calibration["checkpoints"]) == 4
    for checkpoint in calibration["checkpoints"]:
        assert set(checkpoint) == {
            "edge_ms",
            "predicted_survival_mean",
            "observed_exceed_fraction",
            "count",
        }
        assert 0.0 <= checkpoint["predicted_survival_mean"] <= 1.0
        assert 0.0 <= checkpoint["observed_exceed_fraction"] <= 1.0
        assert checkpoint["count"] == 8
    assert len(calibration["reliability_bins"]) == 4
    assert sum(bin_["count"] for bin_ in calibration["reliability_bins"]) == 8
    # The whole payload round-trips through JSON (no numpy scalars leak).
    json.dumps(result["hazard_calibration"])


# --- OOF cleanliness: inner-train vocab excludes held-out-only tools ----------


def test_inner_fold_vocab_excludes_held_out_task_tools() -> None:
    # One task uses a tool that appears nowhere else. Whatever inner fold holds
    # that task out, the encoder fit on the remaining tasks must not learn the
    # tool, so it encodes to an all-zero tool block (never leaking the held-out
    # task into its own scored masses).
    rows = _corpus(["p0", "p1", "p2"], fast=2, slow=2)
    rows += [
        _row("unique-0", task_id="p3", tool_name="onlyheld", latency_ms=42.0, ts_start=0.0),
        _row("unique-1", task_id="p3", tool_name="onlyheld", latency_ms=44.0, ts_start=1.0),
    ]
    folds = balanced_task_folds(rows, fold_count=2)
    saw_held_out_task = False
    for held_out in folds:
        if "p3" not in held_out:
            continue
        saw_held_out_task = True
        inner_train = [row for row in rows if str(row["task_id"]) not in held_out]
        encoder = fit_feature_encoder(inner_train, spec=TOOL_SPEC)
        assert "onlyheld" not in encoder.tool_vocab
        held_feats = [
            feat
            for feat in iter_causal_row_features(rows, spec=TOOL_SPEC)
            if feat.tool_name == "onlyheld"
        ]
        assert held_feats
        for feat in held_feats:
            assert not encoder.transform(feat).any()
    assert saw_held_out_task


# --- Merge failure modes -----------------------------------------------------


def _mode_b_decision(
    sample_id: str,
    *,
    task_id: str,
    latency_ms: float,
    tool_name: str = "exec",
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "tool_name": tool_name,
        "latency_ms": latency_ms,
        "kv_cost_ms": COST_MS,
        "threshold_ms": THRESHOLD_MS,
        "deadline_trigger_ms": THRESHOLD_MS,
        "offline_gated_robust_trigger_ms": THRESHOLD_MS,
    }


def _hazard_decision(sample_id: str, *, task_id: str, latency_ms: float) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "tool_name": "exec",
        "latency_ms": latency_ms,
        "kv_cost_ms": COST_MS,
        "threshold_ms": THRESHOLD_MS,
        "deadline_trigger_ms": THRESHOLD_MS,
        "hazard_trigger_ms": 50.0,
        "hazard_margin_normalized": 1.0,
        "offline_gated_hazard_trigger_ms": 50.0,
        "offline_gated_hazard_guard_normalized": 0.0,
    }


def _gated_b1_row(sample_id: str, *, latency_ms: float) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "kv_cost_ms": COST_MS,
        "latency_ms": latency_ms,
        "threshold_ms": THRESHOLD_MS,
        "gated_within_task_trigger_ms": THRESHOLD_MS,
    }


def test_merge_hazard_rows_joins_all_three_sources(tmp_path: Path) -> None:
    decisions_path = tmp_path / "f1_decisions.jsonl"
    decisions_path.write_text(
        json.dumps(_mode_b_decision("s0", task_id="t0", latency_ms=SLOW_MS)) + "\n",
        encoding="utf-8",
    )
    gated_lookup = {(("s0"), COST_MS): _gated_b1_row("s0", latency_ms=SLOW_MS)}
    merged = _merge_hazard_rows(
        decisions_path,
        [_hazard_decision("s0", task_id="t0", latency_ms=SLOW_MS)],
        gated_within_task_by_key=gated_lookup,
        fold_name="f1",
    )
    assert len(merged) == 1
    row = merged[0]
    assert row["offline_gated_hazard_trigger_ms"] == 50.0
    assert row["gated_within_task_trigger_ms"] == THRESHOLD_MS
    assert row["offline_gated_robust_trigger_ms"] == THRESHOLD_MS
    assert row["outer_fold"] == "f1"
    assert not gated_lookup  # both directions consumed exactly


def test_merge_hazard_rows_raises_on_missing_hazard_row(tmp_path: Path) -> None:
    decisions_path = tmp_path / "f1_decisions.jsonl"
    decisions_path.write_text(
        json.dumps(_mode_b_decision("s0", task_id="t0", latency_ms=SLOW_MS)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no hazard row"):
        _merge_hazard_rows(
            decisions_path,
            [],
            gated_within_task_by_key={("s0", COST_MS): _gated_b1_row("s0", latency_ms=SLOW_MS)},
            fold_name="f1",
        )


def test_merge_hazard_rows_raises_on_missing_gated_b1_row(tmp_path: Path) -> None:
    decisions_path = tmp_path / "f1_decisions.jsonl"
    decisions_path.write_text(
        json.dumps(_mode_b_decision("s0", task_id="t0", latency_ms=SLOW_MS)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no gated-B1 row"):
        _merge_hazard_rows(
            decisions_path,
            [_hazard_decision("s0", task_id="t0", latency_ms=SLOW_MS)],
            gated_within_task_by_key={},
            fold_name="f1",
        )


def test_merge_hazard_rows_raises_on_unmatched_hazard_rows(tmp_path: Path) -> None:
    decisions_path = tmp_path / "f1_decisions.jsonl"
    decisions_path.write_text(
        json.dumps(_mode_b_decision("s0", task_id="t0", latency_ms=SLOW_MS)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hazard rows unmatched"):
        _merge_hazard_rows(
            decisions_path,
            [
                _hazard_decision("s0", task_id="t0", latency_ms=SLOW_MS),
                _hazard_decision("s1", task_id="t0", latency_ms=SLOW_MS),
            ],
            gated_within_task_by_key={
                ("s0", COST_MS): _gated_b1_row("s0", latency_ms=SLOW_MS),
                ("s1", COST_MS): _gated_b1_row("s1", latency_ms=SLOW_MS),
            },
            fold_name="f1",
        )


def test_merge_hazard_rows_raises_on_latency_mismatch(tmp_path: Path) -> None:
    decisions_path = tmp_path / "f1_decisions.jsonl"
    decisions_path.write_text(
        json.dumps(_mode_b_decision("s0", task_id="t0", latency_ms=SLOW_MS)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="latency_ms mismatch"):
        _merge_hazard_rows(
            decisions_path,
            [_hazard_decision("s0", task_id="t0", latency_ms=SLOW_MS + 1.0)],
            gated_within_task_by_key={("s0", COST_MS): _gated_b1_row("s0", latency_ms=SLOW_MS)},
            fold_name="f1",
        )


# --- End-to-end confirmation run ---------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_run_hazard_model_confirmation_merges_and_scores(tmp_path: Path) -> None:
    confirmation = tmp_path / "confirmation"
    (confirmation / "provenance").mkdir(parents=True)
    (confirmation / "provenance" / "manifest.json").write_text(
        json.dumps(
            {
                "fold_count": 1,
                "inner_folds": 2,
                "costs_ms": [COST_MS],
                "guard_ms": 0.0,
                "min_tool_history": 1,
                "min_profile_tasks": 1,
                "command_field": None,
                "max_prefix_depth": 4,
                "skip_leading_cd": False,
            }
        ),
        encoding="utf-8",
    )
    (confirmation / "data").mkdir()
    eval_rows = _corpus(["e0", "e1"], fast=2, slow=2)
    profile_rows = _corpus([f"p{index}" for index in range(4)], fast=3, slow=3)
    _write_jsonl(confirmation / "data" / "f1_eval.jsonl", eval_rows)
    _write_jsonl(confirmation / "data" / "f1_profile.jsonl", profile_rows)

    # Mode B refit rows: the robust trie never fires early here, isolating the
    # hazard contrasts. gated-B1 likewise waits for the deadline everywhere.
    mode_b = tmp_path / "mode-b"
    (mode_b / "rho_0.0").mkdir(parents=True)
    _write_jsonl(
        mode_b / "rho_0.0" / "f1_decisions.jsonl",
        [
            _mode_b_decision(
                row["sample_id"],
                task_id=row["task_id"],
                latency_ms=row["latency_ms"],
                tool_name=row["tool_name"],
            )
            for row in eval_rows
        ],
    )
    gated_b1 = tmp_path / "b1"
    gated_b1.mkdir()
    _write_jsonl(
        gated_b1 / "rho_0.0_decisions.jsonl",
        [_gated_b1_row(row["sample_id"], latency_ms=row["latency_ms"]) for row in eval_rows],
    )

    result = run_hazard_model_confirmation(
        confirmation,
        mode_b_root=mode_b,
        gated_b1_root=gated_b1,
        output_root=tmp_path / "hazard",
        restore_cost_fractions=[0.0],
        num_intervals=8,
        replicates=200,
        confidence_level=0.95,
        seed=0,
        l2_penalty=1e-4,
    )

    assert result["decision_row_count"] == len(eval_rows)
    # Each of the four slow eval calls hides the full cost early (+100); the
    # fast calls never fire, so every contrast nets +400 against a baseline that
    # only ever waits for the deadline.
    for name in (
        "hazard_vs_deadline",
        "gated_hazard_vs_deadline",
        "gated_hazard_vs_gated_robust",
        "gated_hazard_vs_gated_within_task",
    ):
        delta = result["comparisons"][name]["by_restore_cost_fraction"]["0.0"][
            "points"
        ][str(COST_MS)]["paired_delta_ms"]
        assert delta == pytest.approx(4 * 100.0)

    # Only slow-tool calls fire early under the deployed gated hazard trigger.
    merged = [
        json.loads(line)
        for line in (tmp_path / "hazard" / "rho_0.0_decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for row in merged:
        fired = (
            row["latency_ms"] > row["offline_gated_hazard_trigger_ms"]
            and row["offline_gated_hazard_trigger_ms"] < row["threshold_ms"]
        )
        assert fired == (row["tool_name"] == "slow")

    # Per-fold summary + calibration + guard are recorded.
    summary = json.loads(
        (tmp_path / "hazard" / "rho_0.0" / "f1_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["num_intervals"] == 8
    assert summary["restore_cost_fraction"] == 0.0
    assert "hazard_calibration" in summary
    assert result["hazard_guards"]["0.0"][0]["fold"] == "f1"
    assert result["grid_l2_choices"]["0.0"][0]["l2_penalty"] == pytest.approx(1e-4)
    assert (tmp_path / "hazard" / "summary.md").read_text(encoding="utf-8")


def _build_confirmation_inputs(
    tmp_path: Path, *, profile_tasks: int, eval_tasks: int
) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    """Materialize the confirmation/mode-b/gated-b1 inputs for a rho=0 run."""

    confirmation = tmp_path / "confirmation"
    (confirmation / "provenance").mkdir(parents=True)
    (confirmation / "provenance" / "manifest.json").write_text(
        json.dumps(
            {
                "fold_count": 1,
                "inner_folds": 2,
                "costs_ms": [COST_MS],
                "guard_ms": 0.0,
                "min_tool_history": 1,
                "min_profile_tasks": 1,
                "command_field": None,
                "max_prefix_depth": 4,
                "skip_leading_cd": False,
            }
        ),
        encoding="utf-8",
    )
    (confirmation / "data").mkdir()
    eval_rows = _corpus([f"e{i}" for i in range(eval_tasks)], fast=2, slow=2)
    profile_rows = _corpus([f"p{i}" for i in range(profile_tasks)], fast=3, slow=3)
    _write_jsonl(confirmation / "data" / "f1_eval.jsonl", eval_rows)
    _write_jsonl(confirmation / "data" / "f1_profile.jsonl", profile_rows)

    mode_b = tmp_path / "mode-b"
    (mode_b / "rho_0.0").mkdir(parents=True)
    _write_jsonl(
        mode_b / "rho_0.0" / "f1_decisions.jsonl",
        [
            _mode_b_decision(
                row["sample_id"],
                task_id=row["task_id"],
                latency_ms=row["latency_ms"],
                tool_name=row["tool_name"],
            )
            for row in eval_rows
        ],
    )
    gated_b1 = tmp_path / "b1"
    gated_b1.mkdir()
    _write_jsonl(
        gated_b1 / "rho_0.0_decisions.jsonl",
        [_gated_b1_row(row["sample_id"], latency_ms=row["latency_ms"]) for row in eval_rows],
    )
    return confirmation, mode_b, gated_b1, eval_rows


def test_run_hazard_model_confirmation_gbm_end_to_end(tmp_path: Path) -> None:
    # Full plumbing of model_family='gbm' through the driver: --seed threads into
    # every fold fit (no l2), the payload records the family, and the deployed
    # gated hazard trigger still fires early only on slow-tool calls.
    confirmation, mode_b, gated_b1, eval_rows = _build_confirmation_inputs(
        tmp_path, profile_tasks=8, eval_tasks=2
    )

    result = run_hazard_model_confirmation(
        confirmation,
        mode_b_root=mode_b,
        gated_b1_root=gated_b1,
        output_root=tmp_path / "hazard",
        restore_cost_fractions=[0.0],
        num_intervals=8,
        replicates=200,
        confidence_level=0.95,
        seed=0,
        model_family="gbm",
    )

    assert result["model_family"] == "gbm"
    assert result["decision_row_count"] == len(eval_rows)
    assert result["grid_l2_choices"]["0.0"][0]["l2_penalty"] is None

    merged = [
        json.loads(line)
        for line in (tmp_path / "hazard" / "rho_0.0_decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for row in merged:
        fired = (
            row["latency_ms"] > row["offline_gated_hazard_trigger_ms"]
            and row["offline_gated_hazard_trigger_ms"] < row["threshold_ms"]
        )
        assert fired == (row["tool_name"] == "slow")


def test_run_hazard_model_confirmation_refuses_existing_output(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError, match="refusing to mix stale output"):
        run_hazard_model_confirmation(
            tmp_path / "confirmation",
            mode_b_root=tmp_path / "mode-b",
            gated_b1_root=tmp_path / "b1",
            output_root=tmp_path / "out",
            restore_cost_fractions=[0.0],
            num_intervals=8,
            replicates=10,
            confidence_level=0.95,
            seed=0,
        )
