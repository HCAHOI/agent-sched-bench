"""CPU tests for the P4 certified-trigger integration (no vllm/GPU).

Covers the deploy trigger table (build + fold policy, load validation,
exact/backoff/deadline lookup mirroring the production trie), the real-trace
replay selection (filtering + seeded determinism), and the per-call certified
accounting vs the deadline-only and never-pause baselines.
"""

from __future__ import annotations

import json

import pytest

from spike.certified_replay import account_call, aggregate_certified
from spike.tool_replay import ReplayCall, select_replay_calls
from spike.trigger_table import (
    build_trigger_table,
    load_trigger_table,
    lookup_trigger,
)
from trace_collect.tool_latency_dataset import ToolLatencySample


def _decision(
    *,
    kv_cost_ms: float,
    trigger: float,
    fold: str,
    group_key: str | None,
    tool_name: str = "exec",
    prior_source: str = "prior_group",
    deadline: float | None = None,
) -> dict:
    deadline = kv_cost_ms if deadline is None else deadline
    return {
        "kv_cost_ms": kv_cost_ms,
        "threshold_ms": deadline,
        "deadline_trigger_ms": deadline,
        "certified_union_trigger_ms": trigger,
        "prior_source": prior_source,
        "prior_group_key": group_key,
        "outer_fold": fold,
        "tool_name": tool_name,
    }


# --- trigger table: build + fold policy ------------------------------------


def test_build_table_median_of_per_fold_median() -> None:
    # One group across two folds; per-fold medians are 100 and 300 -> median 200.
    decisions = [
        _decision(kv_cost_ms=5000, trigger=100, fold="f1", group_key="exec:apt-get"),
        _decision(kv_cost_ms=5000, trigger=100, fold="f1", group_key="exec:apt-get"),
        _decision(kv_cost_ms=5000, trigger=300, fold="f2", group_key="exec:apt-get"),
    ]
    table = build_trigger_table(decisions, kv_cost_ms=5000, deadline_ms=5000)
    assert table.group_triggers == {"exec:apt-get": 200.0}
    spread = table.metadata["per_group_fold_spread"]["exec:apt-get"]
    assert spread["fold_count"] == 2
    assert spread["fold_spread_ms"] == 200.0
    assert spread["folds_agree"] is False


def test_build_table_drops_groups_that_never_beat_deadline() -> None:
    decisions = [
        _decision(kv_cost_ms=5000, trigger=5000, fold="f1", group_key="exec:ls"),
        _decision(kv_cost_ms=5000, trigger=400, fold="f1", group_key="exec:apt-get"),
    ]
    table = build_trigger_table(decisions, kv_cost_ms=5000, deadline_ms=5000)
    assert "exec:ls" not in table.group_triggers  # >= deadline -> falls back
    assert table.group_triggers == {"exec:apt-get": 400.0}


def test_build_table_ignores_tool_and_global_nodes() -> None:
    decisions = [
        _decision(
            kv_cost_ms=5000, trigger=100, fold="f1", group_key=None,
            prior_source="prior_tool",
        ),
        _decision(
            kv_cost_ms=5000, trigger=100, fold="f1", group_key=None,
            prior_source="prior_global",
        ),
        _decision(kv_cost_ms=5000, trigger=400, fold="f1", group_key="exec:apt-get"),
    ]
    table = build_trigger_table(decisions, kv_cost_ms=5000, deadline_ms=5000)
    assert table.group_triggers == {"exec:apt-get": 400.0}


def test_build_table_selects_only_the_requested_cell() -> None:
    decisions = [
        _decision(kv_cost_ms=3500, trigger=100, fold="f1", group_key="exec:apt-get"),
        _decision(kv_cost_ms=5000, trigger=400, fold="f1", group_key="exec:apt-get"),
    ]
    table = build_trigger_table(decisions, kv_cost_ms=5000, deadline_ms=5000)
    assert table.group_triggers == {"exec:apt-get": 400.0}
    assert table.metadata["cell_row_count"] == 1


def test_build_table_rejects_deadline_mismatch() -> None:
    decisions = [
        _decision(kv_cost_ms=5000, trigger=400, fold="f1", group_key="exec:apt-get"),
    ]
    with pytest.raises(ValueError, match="deadline"):
        build_trigger_table(decisions, kv_cost_ms=5000, deadline_ms=4000)


def test_build_table_rejects_empty_cell() -> None:
    decisions = [
        _decision(kv_cost_ms=3500, trigger=400, fold="f1", group_key="exec:apt-get"),
    ]
    with pytest.raises(ValueError, match="no decisions"):
        build_trigger_table(decisions, kv_cost_ms=5000, deadline_ms=5000)


# --- trigger table: load validation ----------------------------------------


def test_load_table_roundtrip(tmp_path) -> None:
    decisions = [
        _decision(kv_cost_ms=5000, trigger=400, fold="f1", group_key="exec:apt-get"),
    ]
    table = build_trigger_table(decisions, kv_cost_ms=5000, deadline_ms=5000)
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table.to_json_obj()))
    loaded = load_trigger_table(path)
    assert loaded.group_triggers == {"exec:apt-get": 400.0}
    assert loaded.deadline_ms == 5000.0
    assert loaded.max_prefix_depth == 4
    assert loaded.skip_leading_cd is False


def test_load_table_rejects_missing_fields(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"group_triggers": {}}))
    with pytest.raises(ValueError, match="missing fields"):
        load_trigger_table(path)


def test_load_table_rejects_trigger_at_or_above_deadline(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "kv_cost_ms": 5000,
                "deadline_ms": 5000,
                "max_prefix_depth": 4,
                "skip_leading_cd": False,
                "group_triggers": {"exec:ls": 5000},
            }
        )
    )
    with pytest.raises(ValueError, match=">= deadline"):
        load_trigger_table(path)


# --- trigger table: lookup mirrors the production trie ----------------------


def _table(**group_triggers: float):
    decisions = [
        _decision(kv_cost_ms=5000, trigger=t, fold="f1", group_key=k)
        for k, t in group_triggers.items()
    ]
    return build_trigger_table(decisions, kv_cost_ms=5000, deadline_ms=5000)


def test_lookup_exact_deepest_node() -> None:
    table = _table(**{"exec:apt-get update -qq &&": 372.0})
    lk = lookup_trigger(table, "exec", "apt-get update -qq && apt-get install")
    assert lk.trigger_ms == 372.0
    assert lk.group_key == "exec:apt-get update -qq &&"
    assert lk.backoff_level == 0
    assert lk.source == "group"


def test_lookup_backs_off_to_shorter_prefix() -> None:
    table = _table(**{"exec:pip3 install": 4691.0})
    lk = lookup_trigger(table, "exec", "pip3 install requests")
    # deepest computed key 'exec:pip3 install requests' misses; backs off one.
    assert lk.group_key == "exec:pip3 install"
    assert lk.backoff_level == 1
    assert lk.trigger_ms == 4691.0


def test_lookup_falls_back_to_deadline_on_miss() -> None:
    table = _table(**{"exec:apt-get": 400.0})
    lk = lookup_trigger(table, "exec", "ls -la")
    assert lk.trigger_ms == 5000.0
    assert lk.group_key is None
    assert lk.source == "deadline"


def test_lookup_empty_command_is_deadline() -> None:
    table = _table(**{"exec:apt-get": 400.0})
    lk = lookup_trigger(table, "exec", "   ")
    assert lk.source == "deadline"
    assert lk.trigger_ms == 5000.0


def test_lookup_prefers_deepest_when_multiple_prefixes_present() -> None:
    table = _table(**{"exec:apt-get": 400.0, "exec:apt-get update": 100.0})
    lk = lookup_trigger(table, "exec", "apt-get update -y")
    # 'exec:apt-get update' (depth 3) beats the shallower 'exec:apt-get'.
    assert lk.group_key == "exec:apt-get update"
    assert lk.trigger_ms == 100.0
    assert lk.backoff_level == 1  # deepest computed is 'apt-get update -y'


# --- real-trace replay selection -------------------------------------------


def _sample(sid: str, task: str, command, ts: float, latency: float) -> ToolLatencySample:
    return ToolLatencySample(
        sample_id=sid,
        source_trace=f"trace/{task}",
        task_id=task,
        agent_id="a",
        instance_id=task,
        iteration=0,
        action_id=sid,
        tool_name="exec",
        tool_call_id=sid,
        tool_ts_start=ts,
        tool_ts_end=ts + latency / 1000.0,
        latency_ms=latency,
        success=True,
        reported_duration_ms=None,
        tool_args={"command": command} if command is not None else None,
    )


def test_select_replay_keeps_only_command_calls() -> None:
    samples = [
        _sample("s1", "t1", "apt-get update", 1.0, 500.0),
        _sample("s2", "t1", None, 2.0, 10.0),  # no tool_args
        _sample("s3", "t1", "", 3.0, 10.0),  # empty command
    ]
    calls = select_replay_calls(samples, limit=10)
    assert [c.sample_id for c in calls] == ["s1"]
    assert calls[0] == ReplayCall("s1", "t1", "exec", "apt-get update", 500.0)


def test_select_replay_filters_task_ids() -> None:
    samples = [
        _sample("s1", "t1", "ls", 1.0, 5.0),
        _sample("s2", "t2", "ls", 1.0, 5.0),
    ]
    calls = select_replay_calls(samples, task_ids=["t2"], limit=10)
    assert [c.task_id for c in calls] == ["t2"]


def test_select_replay_is_deterministic_under_seed() -> None:
    # ts start = index, so temporal order == sample index order.
    samples = [_sample(f"s{i}", "t1", f"cmd{i}", float(i), 5.0) for i in range(20)]
    starts = {f"s{i}": float(i) for i in range(20)}
    a = select_replay_calls(samples, limit=5, seed=7)
    b = select_replay_calls(samples, limit=5, seed=7)
    assert [c.sample_id for c in a] == [c.sample_id for c in b]
    # replay order is temporal (sorted by ts) after the seeded subset selection.
    chosen_starts = [starts[c.sample_id] for c in a]
    assert chosen_starts == sorted(chosen_starts)


def test_select_replay_fails_when_nothing_replayable() -> None:
    with pytest.raises(ValueError, match="no replayable"):
        select_replay_calls([_sample("s1", "t1", None, 1.0, 5.0)], limit=10)


# --- per-call certified accounting -----------------------------------------


def test_account_correct_fire_on_long_call() -> None:
    table = _table(**{"exec:apt-get": 400.0})
    lk = lookup_trigger(table, "exec", "apt-get update -y")
    r = account_call(
        lk,
        sample_id="s",
        task_id="t",
        tool_name="exec",
        command="apt-get update -y",
        duration_ms=8000.0,
        kv_cost_ms=5000.0,
        deadline_ms=5000.0,
    )
    assert r.fired and r.correct_fire and not r.misfire
    # early fire hides the full kv cost; the deadline policy fires late.
    assert r.kv_saved_ms == 5000.0
    assert r.kv_saved_deadline_ms == 1000.0
    assert r.delta_vs_deadline_ms == 4000.0
    assert r.delta_vs_never_ms == 5000.0
    assert r.free_window_ms == pytest.approx(7600.0)


def test_account_misfire_on_short_call() -> None:
    table = _table(**{"exec:apt-get": 400.0})
    lk = lookup_trigger(table, "exec", "apt-get update -y")
    r = account_call(
        lk,
        sample_id="s",
        task_id="t",
        tool_name="exec",
        command="apt-get update -y",
        duration_ms=600.0,  # > trigger 400 but <= deadline 5000
        kv_cost_ms=5000.0,
        deadline_ms=5000.0,
    )
    assert r.fired and r.misfire and not r.correct_fire
    # short-call fire pays exposure the deadline policy never does.
    assert r.kv_saved_ms < 0.0
    assert r.kv_saved_deadline_ms == 0.0
    assert r.delta_vs_deadline_ms < 0.0


def test_account_no_fire_when_call_ends_before_trigger() -> None:
    table = _table(**{"exec:apt-get": 400.0})
    lk = lookup_trigger(table, "exec", "apt-get update -y")
    r = account_call(
        lk,
        sample_id="s",
        task_id="t",
        tool_name="exec",
        command="apt-get update -y",
        duration_ms=200.0,  # < trigger 400
        kv_cost_ms=5000.0,
        deadline_ms=5000.0,
    )
    assert not r.fired and not r.correct_fire and not r.misfire
    assert r.kv_saved_ms == 0.0
    assert r.free_window_ms == 0.0


def test_account_deadline_fallback_matches_deadline_policy() -> None:
    table = _table(**{"exec:apt-get": 400.0})
    lk = lookup_trigger(table, "exec", "ls -la")  # miss -> deadline
    r = account_call(
        lk,
        sample_id="s",
        task_id="t",
        tool_name="exec",
        command="ls -la",
        duration_ms=8000.0,
        kv_cost_ms=5000.0,
        deadline_ms=5000.0,
    )
    assert r.trigger_source == "deadline"
    assert r.delta_vs_deadline_ms == 0.0  # identical to the deadline baseline


def test_aggregate_certified_totals() -> None:
    table = _table(**{"exec:apt-get": 400.0})
    long = account_call(
        lookup_trigger(table, "exec", "apt-get update -y"),
        sample_id="s1", task_id="t", tool_name="exec",
        command="apt-get update -y", duration_ms=8000.0,
        kv_cost_ms=5000.0, deadline_ms=5000.0,
    )
    miss = account_call(
        lookup_trigger(table, "exec", "ls"),
        sample_id="s2", task_id="t", tool_name="exec",
        command="ls", duration_ms=8000.0,
        kv_cost_ms=5000.0, deadline_ms=5000.0,
    )
    agg = aggregate_certified([long, miss])
    assert agg["call_count"] == 2
    assert agg["fired_count"] == 2
    assert agg["correct_fire_count"] == 2
    assert agg["group_hit_count"] == 1
    assert agg["deadline_fallback_count"] == 1
    assert agg["delta_vs_deadline_ms"] == 4000.0


def test_aggregate_certified_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no call results"):
        aggregate_certified([])


# --- CRITICAL regression: restore cost must be charged (campaign finding F1) -


def test_short_misfire_restore_cost_prevents_f1_style_manufactured_win() -> None:
    """A trigger=0 short misfire (a real certified group in the fresh-corpus
    table, 'exec:apt-get update && apt-get') must show a real, negative cost
    once restore is charged at the measured operating point rho=0.94 -- at the
    old (bugged) rho=0 default it looked non-negative/"free", exactly campaign
    finding F1 (rho=0 manufactures success). This is the regression for the
    CRITICAL reviewer finding: run_certical/export_trigger_table previously
    scored the certified arm restore-free while the deadline arm's baseline
    (which never fires early on a short call) was unaffected either way -- an
    asymmetric, certified-arm-only inflation.
    """
    table = _table(**{"exec:apt-get update && apt-get": 0.0})
    lk = lookup_trigger(table, "exec", "apt-get update && apt-get install -y x")
    kv_cost_ms = 5000.0
    deadline_ms = 5000.0
    # == deadline: NOT "is_long" (is_long requires strictly >), so this is the
    # misfire branch; trigger=0 means it fires (duration > 0) and pays the full
    # exposure/restore of a short call rather than hiding any KV time.
    duration_ms = 5000.0

    old_rho0 = account_call(
        lk,
        sample_id="s",
        task_id="t",
        tool_name="exec",
        command="apt-get update && apt-get install -y x",
        duration_ms=duration_ms,
        kv_cost_ms=kv_cost_ms,
        deadline_ms=deadline_ms,
        restore_cost_ms=0.0,  # the old, bugged default
    )
    rho94_restore_ms = 0.94 * kv_cost_ms
    new_rho94 = account_call(
        lk,
        sample_id="s",
        task_id="t",
        tool_name="exec",
        command="apt-get update && apt-get install -y x",
        duration_ms=duration_ms,
        kv_cost_ms=kv_cost_ms,
        deadline_ms=deadline_ms,
        restore_cost_ms=rho94_restore_ms,
    )
    assert old_rho0.fired and old_rho0.misfire and not old_rho0.correct_fire
    assert new_rho94.fired and new_rho94.misfire and not new_rho94.correct_fire

    # Old (bugged) accounting: no exposure (remaining == kv_cost) and no
    # restore charged -- looks like a costless/non-negative outcome.
    assert old_rho0.kv_saved_ms == 0.0
    # Fixed accounting: the ~4700ms restore charge is now visible and negative.
    assert new_rho94.kv_saved_ms == pytest.approx(-rho94_restore_ms)
    assert new_rho94.kv_saved_ms < 0.0
    # Assert the exact difference the restore charge accounts for.
    assert old_rho0.kv_saved_ms - new_rho94.kv_saved_ms == pytest.approx(4700.0)


def test_build_table_rejects_negative_restore_cost_fraction() -> None:
    decisions = [
        _decision(kv_cost_ms=5000, trigger=400, fold="f1", group_key="exec:apt-get"),
    ]
    with pytest.raises(ValueError, match="restore_cost_fraction"):
        build_trigger_table(
            decisions, kv_cost_ms=5000, deadline_ms=5000, restore_cost_fraction=-0.1
        )


# --- exporter: required + validated --restore-cost-fraction (fix 1a/1c/3) ---


def test_export_validation_rejects_zero_restore_cost_fraction(tmp_path) -> None:
    from scripts.export_trigger_table import validate_restore_cost_fraction_for_export

    path = tmp_path / "rho_0.0_decisions.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="not positive"):
        validate_restore_cost_fraction_for_export(path, 0.0)


def test_export_validation_rejects_filename_mismatch(tmp_path) -> None:
    from scripts.export_trigger_table import validate_restore_cost_fraction_for_export

    path = tmp_path / "rho_0.94_decisions.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="does not match"):
        validate_restore_cost_fraction_for_export(path, 0.5)


def test_export_validation_rejects_unrecognized_filename(tmp_path) -> None:
    from scripts.export_trigger_table import validate_restore_cost_fraction_for_export

    path = tmp_path / "decisions.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="naming convention"):
        validate_restore_cost_fraction_for_export(path, 0.94)


def test_export_validation_accepts_matching_fraction(tmp_path) -> None:
    from scripts.export_trigger_table import validate_restore_cost_fraction_for_export

    path = tmp_path / "rho_0.94_decisions.jsonl"
    path.write_text("")
    validate_restore_cost_fraction_for_export(path, 0.94)  # no raise


def test_exporter_cli_requires_restore_cost_fraction() -> None:
    from scripts.export_trigger_table import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--decisions",
                "rho_0.94_decisions.jsonl",
                "--kv-cost-ms",
                "5000",
                "--deadline-ms",
                "5000",
                "--output",
                "out.json",
            ]
        )


# --- run_certified refuses to score at a 0.0-fraction table (defense in depth)


def test_run_certified_refuses_zero_restore_fraction_table(tmp_path) -> None:
    import asyncio
    from argparse import Namespace

    from spike.run_spike import run_certified

    decisions = [
        _decision(kv_cost_ms=5000, trigger=400, fold="f1", group_key="exec:apt-get"),
    ]
    table = build_trigger_table(
        decisions, kv_cost_ms=5000, deadline_ms=5000, restore_cost_fraction=0.0
    )
    table_path = tmp_path / "table.json"
    table_path.write_text(json.dumps(table.to_json_obj()))
    args = Namespace(
        trigger_table=str(table_path),
        trace_root=str(tmp_path),  # unused: rejection happens before replay
    )
    with pytest.raises(ValueError, match="restore_cost_fraction"):
        asyncio.run(run_certified(args))
