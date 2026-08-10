from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from scripts.evaluation.evaluate_static_survival_gap_action import (
    _DEFAULT_IDS,
    _ordered_ids,
    command_latency_evidence,
    expected_early_action,
)
from tool_resource.runtime_kb import ClauseObservation, ClauseResourceKB


def test_cli_is_directly_executable() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/evaluation/evaluate_static_survival_gap_action.py"),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_replay_order_must_match_frozen_ids(tmp_path: Path) -> None:
    frozen = _DEFAULT_IDS.read_text(encoding="utf-8").splitlines()[:3]
    permuted = tmp_path / "ids.txt"
    permuted.write_text("\n".join(reversed(frozen)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen replay order"):
        _ordered_ids(permuted, set(frozen))


def test_expected_early_action_accepts_history_that_pareto_dominates_feedback() -> None:
    decision = expected_early_action(
        (5_500.0, 10_000.0),
        deadline_ms=5_000.0,
        size_gib=2.0,
        swap_out_ms=1_000.0,
        swap_in_ms=500.0,
    )

    assert decision["probability_survives_deadline"] == 1.0
    assert decision["expected_released_delta_gib_s"] == pytest.approx(9.5)
    assert decision["expected_stall_delta_ms"] == pytest.approx(-250.0)
    assert decision["act_early"] is True


def test_expected_early_action_rejects_history_with_positive_expected_stall() -> None:
    decision = expected_early_action(
        (100.0, 10_000.0),
        deadline_ms=5_000.0,
        size_gib=2.0,
        swap_out_ms=1_000.0,
        swap_in_ms=500.0,
    )

    assert decision["probability_survives_deadline"] == 0.5
    assert decision["expected_released_delta_gib_s"] == pytest.approx(5.0)
    assert decision["expected_stall_delta_ms"] == pytest.approx(700.0)
    assert decision["act_early"] is False


def test_expected_early_action_rejects_empty_or_invalid_history() -> None:
    with pytest.raises(ValueError, match="duration samples"):
        expected_early_action(
            (),
            deadline_ms=5_000.0,
            size_gib=2.0,
            swap_out_ms=1_000.0,
            swap_in_ms=500.0,
        )
    with pytest.raises(ValueError, match="duration samples"):
        expected_early_action(
            (-1.0,),
            deadline_ms=5_000.0,
            size_gib=2.0,
            swap_out_ms=1_000.0,
            swap_in_ms=500.0,
        )


def test_command_latency_evidence_uses_the_current_causal_kb_node() -> None:
    public = ClauseObservation(
        repo="public__repo",
        bin="pytest",
        argv=("pytest", "-q"),
        ts_start=0.0,
        ts_end=1.0,
        latency_ms=100.0,
    )
    kb = ClauseResourceKB.fit_public((public,))
    kb.observe_completed_clause(
        ClauseObservation(
            repo="target__repo",
            bin="pytest",
            argv=("pytest", "-q"),
            ts_start=1.0,
            ts_end=2.0,
            latency_ms=10_000.0,
        )
    )

    before = command_latency_evidence(
        kb, "target__repo", "pytest -q", query_ts=2.0
    )
    after = command_latency_evidence(
        kb, "target__repo", "pytest -q", query_ts=2.1
    )

    assert before is not None
    assert before["duration_samples_ms"] == (100.0,)
    assert before["scope"] == "public"
    assert after is not None
    assert after["duration_samples_ms"] == (10_000.0,)
    assert after["scope"] == "repo"
    assert after["key_kind"] == "exact_clause"
