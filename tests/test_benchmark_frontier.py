from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trace_collect.benchmark_frontier import (
    FRONTIER_COMPARISONS,
    FRONTIER_TOOL_NAME_COMPARISONS,
    _merge_tool_name_rows,
    _merge_trie_hazard_rows,
    run_benchmark_frontier,
)


# The synthetic corpus separates on tool identity: a fast tool always returns
# 10 ms (below the 100 ms threshold, so an early swap only ever loses) and a
# slow tool always returns 150 ms (inside the (threshold, threshold+kv) band, so
# firing early hides the whole kv=100 cost). Both the empirical trie and the
# learned hazard model therefore fire early on every slow call and wait on every
# fast call, so at rho=0 each slow eval call nets +100 over the deadline and each
# fast call nets 0 — deterministic hand math shared with the hazard tests.
FAST_MS = 10.0
SLOW_MS = 150.0
COST_MS = 100.0
THRESHOLD_MS = 100.0  # guard_ms = 0


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
        ts = offset * 100_000.0
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


def _write_corpus(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _config_manifest(tmp_path: Path, *, fold_count: int) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "fold_count": fold_count,
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
    return manifest


def _run(
    tmp_path: Path,
    *,
    fold_count: int = 2,
    num_tasks: int = 12,
    fractions: list[float] | None = None,
    exposure_note: str = "dev-exposed corpus (sensitivity replication only)",
    output_name: str = "frontier",
    **kwargs: Any,
) -> dict[str, Any]:
    rows = _corpus([f"t{i:02d}" for i in range(num_tasks)], fast=2, slow=2)
    latencies = tmp_path / "corpus.jsonl"
    _write_corpus(latencies, rows)
    manifest = json.loads(_config_manifest(tmp_path, fold_count=fold_count).read_text())
    return run_benchmark_frontier(
        None,
        output_root=tmp_path / output_name,
        fold_count=manifest["fold_count"],
        inner_folds=manifest["inner_folds"],
        kv_costs_ms=[float(c) for c in manifest["costs_ms"]],
        guard_ms=manifest["guard_ms"],
        min_tool_history=manifest["min_tool_history"],
        min_profile_tasks=manifest["min_profile_tasks"],
        command_field=manifest["command_field"],
        max_prefix_depth=manifest["max_prefix_depth"],
        skip_leading_cd=manifest["skip_leading_cd"],
        num_intervals=8,
        model_family="logistic",
        seed=0,
        restore_cost_fractions=fractions if fractions is not None else [0.0],
        replicates=200,
        confidence_level=0.95,
        exposure_note=exposure_note,
        eval_latencies=latencies,
        **kwargs,
    )


# --- End-to-end: folds, join, contrasts, hand math ---------------------------


def test_run_benchmark_frontier_end_to_end_hand_math(tmp_path: Path) -> None:
    num_tasks = 12
    fold_count = 2
    result = _run(tmp_path, fold_count=fold_count, num_tasks=num_tasks)

    assert result["mode"] == "benchmark_frontier"
    assert result["task_count"] == num_tasks
    # Every task's rows land in exactly one eval fold, so the merged decisions
    # per fraction cover the whole corpus once, times the single kv cost.
    assert result["decision_row_count"] == num_tasks * (2 + 2)

    # Fold partition: eval tasks are the sorted-id positions == (fold-1) mod
    # fold_count, and profile is the disjoint remainder.
    task_ids = [f"t{i:02d}" for i in range(num_tasks)]
    for fold in range(1, fold_count + 1):
        eval_txt = (
            tmp_path / "frontier" / "folds" / f"f{fold}_eval.txt"
        ).read_text(encoding="utf-8").split()
        expected_eval = sorted(
            task
            for index, task in enumerate(task_ids)
            if index % fold_count == fold - 1
        )
        assert eval_txt == expected_eval
        profile_txt = (
            tmp_path / "frontier" / "folds" / f"f{fold}_profile.txt"
        ).read_text(encoding="utf-8").split()
        assert sorted(profile_txt) == sorted(set(task_ids) - set(expected_eval))
        assert not set(eval_txt) & set(profile_txt)

    # Both policies' fields survive the join in every merged decision.
    merged = [
        json.loads(line)
        for line in (tmp_path / "frontier" / "rho_0.0_decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(merged) == num_tasks * 4
    for row in merged:
        for field in (
            "deadline_trigger_ms",
            "robust_trigger_ms",
            "offline_gated_robust_trigger_ms",
            "hazard_trigger_ms",
            "offline_gated_hazard_trigger_ms",
            "outer_fold",
        ):
            assert field in row

    # Hand math at rho=0: each of the 24 slow calls hides the full cost (+100)
    # against the deadline; the fast calls never fire, so both gated policies net
    # +2400 against the deadline and tie each other (delta 0).
    slow_calls = num_tasks * 2
    for name in ("gated_robust_vs_deadline", "gated_hazard_vs_deadline"):
        delta = result["comparisons"][name]["by_restore_cost_fraction"]["0.0"][
            "points"
        ][str(COST_MS)]["paired_delta_ms"]
        assert delta == pytest.approx(slow_calls * 100.0)
    head_to_head = result["comparisons"]["gated_hazard_vs_gated_robust"][
        "by_restore_cost_fraction"
    ]["0.0"]["points"][str(COST_MS)]["paired_delta_ms"]
    assert head_to_head == pytest.approx(0.0)

    # Only slow-tool calls fire early under either deployed gated trigger.
    for row in merged:
        for trigger in (
            "offline_gated_robust_trigger_ms",
            "offline_gated_hazard_trigger_ms",
        ):
            fired = row["latency_ms"] > row[trigger] and row[trigger] < row["threshold_ms"]
            assert fired == (row["tool_name"] == "slow")

    assert (tmp_path / "frontier" / "summary.md").read_text(encoding="utf-8")
    assert result["exposure_note"].startswith("dev-exposed")
    # The three within-benchmark contrasts are all present; no ensemble arm.
    assert set(result["comparisons"]) == {name for name, *_ in FRONTIER_COMPARISONS}
    assert result["ensemble_members"] == 0


def test_run_benchmark_frontier_multi_fraction(tmp_path: Path) -> None:
    # Two fractions: the trie is refit per fraction and the hazard model is
    # scored from cached masses; both fraction slices exist and cover the corpus.
    result = _run(tmp_path, fractions=[0.0, 0.5])
    for key in ("0.0", "0.5"):
        assert (tmp_path / "frontier" / f"rho_{key}_decisions.jsonl").is_file()
        for name, *_ in FRONTIER_COMPARISONS:
            assert key in result["comparisons"][name]["by_restore_cost_fraction"]


# --- Exposure note is required and recorded ----------------------------------


def test_run_benchmark_frontier_requires_exposure_note(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exposure_note must be a non-empty string"):
        _run(tmp_path, exposure_note="   ")


def test_run_benchmark_frontier_records_exposure_note(tmp_path: Path) -> None:
    note = "Terminal-Bench was dev-exposed; sensitivity only, not certification."
    result = _run(tmp_path, exposure_note=note)
    assert result["exposure_note"] == note
    assert note in (tmp_path / "frontier" / "summary.md").read_text(encoding="utf-8")


# --- Refuse existing output --------------------------------------------------


def test_run_benchmark_frontier_refuses_existing_output(tmp_path: Path) -> None:
    (tmp_path / "frontier").mkdir()
    with pytest.raises(FileExistsError, match="refusing to mix stale output"):
        _run(tmp_path)


def test_run_benchmark_frontier_rejects_too_few_tasks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fewer than fold_count"):
        _run(tmp_path, fold_count=3, num_tasks=2)


# --- Join integrity (mirrors _merge_hazard_rows failure modes) ---------------


def _trie_decision(sample_id: str, *, latency_ms: float) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_id": "t0",
        "tool_name": "slow",
        "latency_ms": latency_ms,
        "kv_cost_ms": COST_MS,
        "threshold_ms": THRESHOLD_MS,
        "deadline_trigger_ms": THRESHOLD_MS,
        "robust_trigger_ms": 50.0,
        "offline_gated_robust_trigger_ms": 50.0,
    }


def _hazard_row(sample_id: str, *, latency_ms: float) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_id": "t0",
        "tool_name": "slow",
        "latency_ms": latency_ms,
        "kv_cost_ms": COST_MS,
        "threshold_ms": THRESHOLD_MS,
        "deadline_trigger_ms": THRESHOLD_MS,
        "hazard_trigger_ms": 50.0,
        "hazard_margin_normalized": 1.0,
        "offline_gated_hazard_trigger_ms": 50.0,
        "offline_gated_hazard_guard_normalized": 0.0,
    }


def test_merge_trie_hazard_rows_joins_both_sources() -> None:
    merged = _merge_trie_hazard_rows(
        [_trie_decision("s0", latency_ms=SLOW_MS)],
        [_hazard_row("s0", latency_ms=SLOW_MS)],
        fold_name="f1",
        ensemble_on=False,
    )
    assert len(merged) == 1
    row = merged[0]
    assert row["offline_gated_robust_trigger_ms"] == 50.0
    assert row["offline_gated_hazard_trigger_ms"] == 50.0
    assert row["outer_fold"] == "f1"
    assert "ensemble_hazard_trigger_ms" not in row


def test_merge_trie_hazard_rows_raises_on_missing_hazard_row() -> None:
    with pytest.raises(ValueError, match="no hazard row"):
        _merge_trie_hazard_rows(
            [_trie_decision("s0", latency_ms=SLOW_MS)],
            [],
            fold_name="f1",
            ensemble_on=False,
        )


def test_merge_trie_hazard_rows_raises_on_unmatched_hazard_rows() -> None:
    with pytest.raises(ValueError, match="hazard rows unmatched"):
        _merge_trie_hazard_rows(
            [_trie_decision("s0", latency_ms=SLOW_MS)],
            [
                _hazard_row("s0", latency_ms=SLOW_MS),
                _hazard_row("s1", latency_ms=SLOW_MS),
            ],
            fold_name="f1",
            ensemble_on=False,
        )


def test_merge_trie_hazard_rows_raises_on_latency_mismatch() -> None:
    with pytest.raises(ValueError, match="latency_ms mismatch"):
        _merge_trie_hazard_rows(
            [_trie_decision("s0", latency_ms=SLOW_MS)],
            [_hazard_row("s0", latency_ms=SLOW_MS + 1.0)],
            fold_name="f1",
            ensemble_on=False,
        )


# --- Ensemble arm ------------------------------------------------------------


def test_run_benchmark_frontier_ensemble_adds_contrasts(tmp_path: Path) -> None:
    result = _run(tmp_path, ensemble_members=2)
    assert result["ensemble_members"] == 2
    for name in (
        "gated_ensemble_vs_deadline",
        "gated_ensemble_vs_gated_hazard",
        "gated_ensemble_vs_gated_robust",
    ):
        assert name in result["comparisons"]
    merged = (tmp_path / "frontier" / "rho_0.0_decisions.jsonl").read_text(
        encoding="utf-8"
    )
    assert "offline_gated_ensemble_hazard_trigger_ms" in merged
    assert "ensemble_guards" in result


# --- Tool-name-only trie arm (P1: Continuum's per-tool-name prior) -----------


def _command_row(
    sample_id: str,
    *,
    task_id: str,
    command: str,
    latency_ms: float,
    ts_start: float,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source_trace": f"trace-{task_id}",
        "task_id": task_id,
        "tool_name": "exec",
        "tool_args": {"command": command},
        "latency_ms": latency_ms,
        "tool_ts_start": ts_start,
        "tool_ts_end": ts_start + latency_ms / 1000.0,
    }


def _command_corpus(task_ids: list[str]) -> list[dict[str, Any]]:
    """One tool ("exec") whose latency is a function of the command, not the tool.

    Each task issues two ``make build`` calls (150 ms, inside the (100, 200)
    band, so an early swap hides the whole 100 ms cost) and two ``ls`` calls
    (10 ms, short). The command-prefix (full) trie separates the two commands
    into distinct group nodes; the tool-name-only trie (command_field=None) can
    only pool them under the single ``exec`` tool node, so its evidence and its
    prior_source differ from the full trie by construction.
    """

    rows: list[dict[str, Any]] = []
    for offset, task_id in enumerate(task_ids):
        ts = offset * 100_000.0
        for call in range(2):
            rows.append(
                _command_row(
                    f"{task_id}-make-{call}",
                    task_id=task_id,
                    command="make build",
                    latency_ms=SLOW_MS,
                    ts_start=ts,
                )
            )
            ts += 1000.0
        for call in range(2):
            rows.append(
                _command_row(
                    f"{task_id}-ls-{call}",
                    task_id=task_id,
                    command="ls",
                    latency_ms=FAST_MS,
                    ts_start=ts,
                )
            )
            ts += 1000.0
    return rows


def _run_command_corpus(
    tmp_path: Path,
    *,
    num_tasks: int = 12,
    fold_count: int = 2,
    **kwargs: Any,
) -> dict[str, Any]:
    rows = _command_corpus([f"t{i:02d}" for i in range(num_tasks)])
    latencies = tmp_path / "command_corpus.jsonl"
    _write_corpus(latencies, rows)
    return run_benchmark_frontier(
        None,
        output_root=tmp_path / "frontier",
        fold_count=fold_count,
        inner_folds=2,
        kv_costs_ms=[COST_MS],
        guard_ms=0.0,
        min_tool_history=1,
        min_profile_tasks=1,
        command_field="command",
        max_prefix_depth=4,
        skip_leading_cd=False,
        num_intervals=8,
        model_family="logistic",
        seed=0,
        restore_cost_fractions=[0.0],
        replicates=200,
        confidence_level=0.95,
        exposure_note="dev-exposed corpus (sensitivity replication only)",
        eval_latencies=latencies,
        **kwargs,
    )


def test_tool_name_trie_arm_adds_p1_contrasts_and_stays_tool_level(
    tmp_path: Path,
) -> None:
    result = _run_command_corpus(tmp_path, tool_name_trie=True)

    assert result["tool_name_trie"] is True
    # The three P1 contrasts are present on top of the base three.
    for name, *_ in FRONTIER_TOOL_NAME_COMPARISONS:
        assert name in result["comparisons"]
    assert {
        "gated_tool_name_vs_deadline",
        "gated_robust_vs_gated_tool_name",
        "gated_hazard_vs_gated_tool_name",
    } == {name for name, *_ in FRONTIER_TOOL_NAME_COMPARISONS}

    merged = [
        json.loads(line)
        for line in (tmp_path / "frontier" / "rho_0.0_decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    # The join is 1:1: adding the tool-name trigger does not change row count.
    assert len(merged) == 12 * 4
    for row in merged:
        # Tool-name arm never groups by command: its prior is the tool node (or
        # the global cold-start back-off), never a command-prefix group.
        assert row["tool_name_prior_source"] in ("prior_tool", "prior_global")
        assert row["tool_name_prior_group_key"] is None
        assert "offline_gated_tool_name_trigger_ms" in row
    # The full trie, on the same rows, DOES condition on the command prefix —
    # proving the two conditionings are genuinely different estimators here.
    assert any(row["robust_source"] == "prior_group" for row in merged)
    assert any(
        row["tool_name_prior_group_key"] is None for row in merged
    )
    # Per-fold provenance summary for the tool-name arm is written out.
    assert (
        tmp_path / "frontier" / "rho_0.0" / "f1_tool_name_trie_summary.json"
    ).is_file()


def test_tool_name_trie_arm_off_by_default(tmp_path: Path) -> None:
    result = _run_command_corpus(tmp_path)
    assert result["tool_name_trie"] is False
    for name, *_ in FRONTIER_TOOL_NAME_COMPARISONS:
        assert name not in result["comparisons"]
    merged = (tmp_path / "frontier" / "rho_0.0_decisions.jsonl").read_text(
        encoding="utf-8"
    )
    assert "offline_gated_tool_name_trigger_ms" not in merged


# --- Tool-name merge join integrity (mirrors _merge_trie_hazard_rows) --------


def _base_merged_row(sample_id: str, *, latency_ms: float) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_id": "t0",
        "tool_name": "exec",
        "latency_ms": latency_ms,
        "kv_cost_ms": COST_MS,
        "threshold_ms": THRESHOLD_MS,
        "deadline_trigger_ms": THRESHOLD_MS,
        "offline_gated_robust_trigger_ms": 50.0,
        "offline_gated_hazard_trigger_ms": 50.0,
        "outer_fold": "f1",
    }


def _tool_name_decision(
    sample_id: str,
    *,
    latency_ms: float,
    trigger_ms: float,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_id": "t0",
        "tool_name": "exec",
        "latency_ms": latency_ms,
        "kv_cost_ms": COST_MS,
        "threshold_ms": THRESHOLD_MS,
        "deadline_trigger_ms": THRESHOLD_MS,
        "offline_gated_robust_trigger_ms": trigger_ms,
        "offline_gated_robust_guard_normalized": 0.0,
        "probe_robust_source": "prior_tool",
        "probe_robust_group_key": None,
    }


def test_merge_tool_name_rows_joins_trigger() -> None:
    merged = _merge_tool_name_rows(
        [_base_merged_row("s0", latency_ms=SLOW_MS)],
        [_tool_name_decision("s0", latency_ms=SLOW_MS, trigger_ms=70.0)],
        fold_name="f1",
    )
    assert len(merged) == 1
    row = merged[0]
    # The full-trie and GBM triggers are preserved untouched; only the
    # tool-name fields are added.
    assert row["offline_gated_robust_trigger_ms"] == 50.0
    assert row["offline_gated_hazard_trigger_ms"] == 50.0
    assert row["offline_gated_tool_name_trigger_ms"] == 70.0
    assert row["tool_name_prior_source"] == "prior_tool"
    assert row["tool_name_prior_group_key"] is None


def test_merge_tool_name_rows_raises_on_missing_row() -> None:
    with pytest.raises(ValueError, match="no tool-name row"):
        _merge_tool_name_rows(
            [_base_merged_row("s0", latency_ms=SLOW_MS)],
            [],
            fold_name="f1",
        )


def test_merge_tool_name_rows_raises_on_unmatched_rows() -> None:
    with pytest.raises(ValueError, match="tool-name rows unmatched"):
        _merge_tool_name_rows(
            [_base_merged_row("s0", latency_ms=SLOW_MS)],
            [
                _tool_name_decision("s0", latency_ms=SLOW_MS, trigger_ms=70.0),
                _tool_name_decision("s1", latency_ms=SLOW_MS, trigger_ms=70.0),
            ],
            fold_name="f1",
        )


def test_merge_tool_name_rows_raises_on_deadline_mismatch() -> None:
    stale = _tool_name_decision("s0", latency_ms=SLOW_MS, trigger_ms=70.0)
    stale["deadline_trigger_ms"] = THRESHOLD_MS + 1.0
    with pytest.raises(ValueError, match="deadline_trigger_ms mismatch"):
        _merge_tool_name_rows(
            [_base_merged_row("s0", latency_ms=SLOW_MS)],
            [stale],
            fold_name="f1",
        )
