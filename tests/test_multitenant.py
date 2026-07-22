from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.serving.measure_prefill_cost import build_parser as build_prefill_parser
from scripts.serving.run_w5_multitenant import main as run_w5_main
from scripts.serving.run_w5_multitenant import validate_matrix_inputs
from spike.multitenant import (
    ContinuumProfile,
    ToolSpan,
    TraceProgram,
    TraceTurn,
    build_continuum_profile,
    build_prerestore_profile,
    build_retention_plan,
    continuum_ttl_ms,
    load_trace_programs,
    load_prefill_cost_profile,
)
from spike.run_multitenant import (
    _load_config,
    _read_task_ids,
    _transfer_totals,
    run_cell,
)
from spike.trigger_table import TriggerTable
from spike.vllm_connector import FinishedRetentionBook, RetainedPrefix
from spike.vllm_connector.core import (
    continuum_priority,
    thunderagent_reasoning_pauses,
    thunderagent_resume_admissions,
)
from trace_collect.command_features import command_prefix_keys


def _turn(*, gap_ms: float = 9000.0) -> TraceTurn:
    return TraceTurn(
        messages=({"role": "user", "content": "fix it"},),
        recorded_prompt_tokens=10,
        completion_tokens=3,
        gap_ms=gap_ms,
        tools=(ToolSpan("exec", "pytest -q", 25.0, 8000.0),),
    )


def _record(
    program_id: str,
    request_id: str,
    *,
    tokens: int,
    created_at: float = 0.0,
    resident_tokens: int | None = None,
    action: str = "pressure",
    policy: str = "thunderagent",
    program_arrival_s: float = 0.0,
) -> RetainedPrefix:
    blocks = tokens // 10
    return RetainedPrefix(
        program_id=program_id,
        request_id=request_id,
        num_tokens=tokens,
        resident_tokens=tokens if resident_tokens is None else resident_tokens,
        block_ids=tuple(range(blocks)),
        block_hashes=tuple(f"h{i}" for i in range(blocks)),
        created_at=created_at,
        expires_at=None,
        action=action,
        policy=policy,
        program_arrival_s=program_arrival_s,
    )


def test_continuum_tool_signature_uses_shell_command_head() -> None:
    assert _turn().tool_signature == "pytest"
    native = replace(
        _turn(),
        tools=(ToolSpan("read_file", "", 0.0, 1.0),),
    )
    assert native.tool_signature == "read_file"


def test_continuum_priority_orders_preempted_then_ttl_then_program() -> None:
    stride = 10
    assert continuum_priority(9, stride, ttl_hit=False, preempted=True) < (
        continuum_priority(0, stride, ttl_hit=True)
    )
    assert continuum_priority(9, stride, ttl_hit=True) < continuum_priority(
        0, stride, ttl_hit=False
    )


def test_policies_never_read_realized_next_turn_gap() -> None:
    table = TriggerTable(
        group_triggers={},
        deadline_ms=5000.0,
        kv_cost_ms=3500.0,
        max_prefix_depth=4,
        skip_leading_cd=False,
        metadata={},
    )
    short = _turn(gap_ms=100.0)
    long = replace(short, gap_ms=100_000.0)
    for policy in ("deadline", "ours", "thunderagent"):
        assert build_retention_plan(
            policy, short, deadline_ms=5000.0, trigger_table=table
        ) == build_retention_plan(policy, long, deadline_ms=5000.0, trigger_table=table)


def test_ours_uses_earliest_command_trigger() -> None:
    key = command_prefix_keys("exec", "pytest -q", max_depth=4)[-1]
    table = TriggerTable(
        group_triggers={key: 1200.0},
        deadline_ms=5000.0,
        kv_cost_ms=3500.0,
        max_prefix_depth=4,
        skip_leading_cd=False,
        metadata={},
    )
    turn = replace(
        _turn(),
        tools=(
            ToolSpan("exec", "pytest -q", 25.0, 8000.0),
            ToolSpan("read_file", "", 100.0, 200.0),
        ),
    )
    plan = build_retention_plan("ours", turn, deadline_ms=5000.0, trigger_table=table)
    assert plan is not None
    assert plan.expire_ms == 1225.0


def test_tool_free_turn_keeps_deadline_retention() -> None:
    plan = build_retention_plan(
        "deadline", replace(_turn(), tools=()), deadline_ms=5000.0
    )
    assert plan is not None
    assert plan.expire_ms == 5000.0


def test_prerestore_plan_uses_profile_tasks_and_never_realized_gap() -> None:
    profile_programs = [
        TraceProgram(f"t{i}", f"trace-{i}", (replace(_turn(), gap_ms=100.0),))
        for i in range(2)
    ]
    profile = build_prerestore_profile(
        profile_programs,
        max_prefix_depth=4,
        skip_leading_cd=False,
        min_tool_history=1,
        min_profile_tasks=1,
    )
    plan = build_retention_plan(
        "deadline",
        _turn(gap_ms=1.0),
        deadline_ms=10.0,
        prerestore_profile=profile,
        restore_cost_ms=50.0,
    )
    assert plan is not None
    assert plan.expire_ms == 35.0
    assert plan.prerestore_ms == 7950.0


def test_continuum_uses_cold_start_then_empirical_global_cdf() -> None:
    cold = ContinuumProfile((1000.0,) * 100, {}, 1.0)
    ttl, source = continuum_ttl_ms(
        cold, "exec", queue_delay_ms=0.0, prefill_reload_ms=3000.0
    )
    assert ttl == pytest.approx(1000.0 * __import__("math").log(3.0))
    assert source == "exp1_cold_start"

    profile = ContinuumProfile(
        global_gap_ms=(100.0,) * 60 + (1000.0,) * 41,
        gap_ms_by_tool={},
        memoryfulness=1.0,
    )
    ttl, source = continuum_ttl_ms(
        profile, "unknown", queue_delay_ms=0.0, prefill_reload_ms=500.0
    )
    assert (ttl, source) == (100.0, "global_cdf")


def test_continuum_profile_uses_only_inter_turn_gaps() -> None:
    program = TraceProgram(
        task_id="t",
        trace_path="trace.jsonl",
        turns=(_turn(gap_ms=100.0), _turn(gap_ms=200.0), _turn(gap_ms=0.0)),
    )
    profile = build_continuum_profile([program])
    assert profile.global_gap_ms == (100.0, 200.0)
    assert profile.memoryfulness == 1.0


def test_trace_loader_preserves_prompts_lengths_and_real_tool_gap(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "task" / "attempt_1" / "trace.jsonl"
    trace_path.parent.mkdir(parents=True)
    rows = [
        {
            "type": "trace_metadata",
            "trace_format_version": 5,
            "instance_id": "task-1",
        },
        {
            "type": "action",
            "action_type": "llm_call",
            "action_id": "a0",
            "agent_id": "task-1",
            "iteration": 0,
            "ts_start": 1.0,
            "ts_end": 2.0,
            "data": {
                "messages_in": [{"role": "user", "content": "one"}],
                "prompt_tokens": 7,
                "completion_tokens": 2,
            },
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "x0",
            "agent_id": "task-1",
            "iteration": 0,
            "ts_start": 2.1,
            "ts_end": 3.1,
            "data": {
                "tool_name": "exec",
                "tool_args": {"command": "pytest -q"},
            },
        },
        {
            "type": "action",
            "action_type": "llm_call",
            "action_id": "a1",
            "agent_id": "task-1",
            "iteration": 1,
            "ts_start": 4.0,
            "ts_end": 5.0,
            "data": {
                "messages_in": [{"role": "user", "content": "two"}],
                "prompt_tokens": 9,
                "completion_tokens": 1,
            },
        },
    ]
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    [program] = load_trace_programs(tmp_path)
    assert program.task_id == "task-1"
    assert program.turns[0].recorded_prompt_tokens == 7
    assert program.turns[0].completion_tokens == 2
    assert program.turns[0].gap_ms == pytest.approx(2000.0)
    assert program.turns[0].tools[0].end_offset_ms == pytest.approx(1100.0)


def test_transfer_totals_exclude_control_events() -> None:
    assert _transfer_totals(
        [
            {"phase": "retention_offload", "bytes_moved": 128},
            {"phase": "thunderagent_reasoning_marked"},
        ]
    ) == (1, 128)


def test_retention_book_matches_prefix_without_mutating_and_consumes_expiry() -> None:
    book = FinishedRetentionBook()
    record = _record("p", "r0", tokens=30, action="offload")
    record = replace(record, expires_at=5.0)
    book.register(record)

    match = book.match(
        "p", ["h0", "h1", "h2", "new"], num_local_tokens=10, block_size=10
    )
    assert match.source == "resident"
    assert match.external_blocks == 2
    assert book.resident["p"] is record
    mismatch = book.match("p", ["different"], num_local_tokens=0, block_size=10)
    assert mismatch.source == "mismatch"
    assert mismatch.external_blocks == 0
    assert book.due(4.9) == []
    assert book.due(5.0) == [record]
    assert book.due(5.0, waiting_program_ids={"p"}) == []
    assert book.move_to_host("p") is record
    assert book.drop("p") is record


def test_connector_refreshes_continuum_priority_from_live_retention() -> None:
    from spike.vllm_connector.gpu import SelectiveOffloadConnector

    connector = object.__new__(SelectiveOffloadConnector)
    connector._retention = FinishedRetentionBook()
    connector._continuum_priority_state = {}
    events: list[tuple[str, dict[str, object]]] = []
    connector._record_control_event = lambda phase, **fields: events.append(
        (phase, fields)
    )
    record = _record("p", "old", tokens=20)
    connector._retention.register(record)
    connector._retention_spec = lambda _: {
        "policy": "continuum",
        "program_id": "p",
        "program_index": 2,
        "priority_stride": 10,
    }
    request = SimpleNamespace(
        request_id="next",
        block_hashes=["h0", "h1"],
        priority=99,
        num_preemptions=0,
    )

    assert connector.refresh_continuum_priorities([request])
    assert request.priority == continuum_priority(2, 10, ttl_hit=True)
    connector._retention.drop("p")
    assert connector.refresh_continuum_priorities([request])
    assert request.priority == continuum_priority(2, 10, ttl_hit=False)
    assert [fields["ttl_hit"] for _, fields in events] == [True, False]


def test_connector_orders_thunder_paused_resumes_by_single_backend_bfd() -> None:
    from spike.vllm_connector.gpu import SelectiveOffloadConnector
    from spike.vllm_connector.core import SavedKVRegistry

    connector = object.__new__(SelectiveOffloadConnector)
    connector._block_size = 10
    connector._thunder_buffer_tokens = 10
    connector._thunder_paused_programs = {"large", "medium"}
    connector._saved = SavedKVRegistry()
    specs = {
        "new": {"program_id": "new", "policy": "thunderagent"},
        "large": {"program_id": "large", "policy": "thunderagent"},
        "medium": {"program_id": "medium", "policy": "thunderagent"},
    }
    connector._retention_spec = lambda request_id: specs[request_id]
    requests = [
        SimpleNamespace(request_id="new", num_tokens=20, block_hashes=[]),
        SimpleNamespace(request_id="medium", num_tokens=60, block_hashes=[]),
        SimpleNamespace(request_id="large", num_tokens=100, block_hashes=[]),
    ]

    assert [
        request.request_id
        for request in connector.order_thunder_waiting(requests, capacity_tokens=120)
    ] == ["medium", "new", "large"]
    assert connector._thunder_admitted_programs == {"medium"}
    assert connector.get_num_new_matched_tokens(requests[2], 0) == (None, False)
    connector._retention = FinishedRetentionBook()
    connector._retention.register(_record("acting", "old", tokens=20))
    connector._thunder_buffer_tokens = 100
    assert (
        connector.thunder_resume_capacity_tokens([requests[0]], free_tokens=300) == 100
    )


def test_continuum_pressure_unpins_latest_program_first() -> None:
    book = FinishedRetentionBook()
    early = _record(
        "early",
        "r0",
        tokens=50,
        action="release",
        policy="continuum",
        program_arrival_s=1.0,
    )
    late = _record(
        "late",
        "r1",
        tokens=50,
        action="release",
        policy="continuum",
        program_arrival_s=2.0,
    )
    book.register(early)
    book.register(late)
    assert book.continuum_pressure_evictions(
        active_tokens=100,
        waiting_tokens=20,
        capacity_tokens=170,
    ) == [late]
    assert book.continuum_pressure_evictions(
        active_tokens=100,
        waiting_tokens=20,
        capacity_tokens=170,
        protected_program_ids={"late"},
    ) == [early]


def test_thunderagent_marks_all_reasoning_programs_on_overflow() -> None:
    assert thunderagent_reasoning_pauses(
        [("large", 70), ("small", 20), ("medium", 40)],
        required_tokens=250,
        capacity_tokens=180,
        buffer_per_program=10,
    ) == ["small", "medium", "large"]


def test_thunderagent_selects_maximal_resume_set_then_bfd_orders_it() -> None:
    candidates = [("small", 40), ("large", 100), ("medium", 60)]
    assert thunderagent_resume_admissions(
        candidates, capacity_tokens=130, buffer_per_program=10
    ) == ["medium", "small"]
    assert thunderagent_resume_admissions(
        candidates, capacity_tokens=100, buffer_per_program=10
    ) == ["small"]


def test_thunderagent_pressure_evicts_smallest_acting_context_first() -> None:
    book = FinishedRetentionBook()
    small = _record("small", "r0", tokens=20, created_at=2.0)
    large = _record("large", "r1", tokens=50, created_at=1.0)
    book.register(large)
    book.register(small)

    evicted = book.pressure_evictions(
        reasoning_tokens=40,
        reasoning_programs=1,
        capacity_tokens=220,
        buffer_per_program=50,
    )
    assert evicted == [small]
    assert book.pressure_evictions(
        reasoning_tokens=0,
        reasoning_programs=0,
        capacity_tokens=250,
        buffer_per_program=10,
        waiting_tokens=200,
        waiting_programs=1,
    ) == [small, large]


def test_thunderagent_uses_resident_size_and_shared_kv() -> None:
    book = FinishedRetentionBook()
    large_resident = _record("large", "r0", tokens=10, resident_tokens=100)
    small_resident = _record("small", "r1", tokens=20, resident_tokens=30)
    book.register(large_resident)
    book.register(small_resident)
    assert book.pressure_evictions(
        reasoning_tokens=0,
        reasoning_programs=0,
        capacity_tokens=100,
        buffer_per_program=0,
    ) == [small_resident]

    shared = FinishedRetentionBook()
    shared.register(_record("a", "ra", tokens=60))
    shared.register(_record("b", "rb", tokens=60))
    assert (
        shared.pressure_evictions(
            reasoning_tokens=0,
            reasoning_programs=0,
            shared_tokens=60,
            capacity_tokens=60,
            buffer_per_program=0,
        )
        == []
    )


def test_thunderagent_does_not_double_count_returning_program() -> None:
    book = FinishedRetentionBook()
    returning = _record("returning", "r0", tokens=100)
    other = _record("other", "r1", tokens=30)
    book.register(returning)
    book.register(other)
    assert book.pressure_evictions(
        reasoning_tokens=100,
        reasoning_programs=1,
        reasoning_program_ids={"returning"},
        capacity_tokens=120,
        buffer_per_program=10,
    ) == [other]


def test_w5_task_inputs_match_owned_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/serving/w5_multitenant.yaml"
    inputs_dir = root / "analysis/serving/w5-multitenant/inputs"
    manifest = json.loads((inputs_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    entries = {row["path"]: row for row in manifest["files"]}

    for workload_name in (
        "swe-rebench-100-development",
        "terminal-bench-83-development",
        "swe-rebench-277-development-exposed",
    ):
        _, workload = _load_config(config_path, workload_name)
        configured = workload["task_ids_file"]
        paths = configured if isinstance(configured, list) else [configured]
        owned_paths = [root / path for path in paths]
        task_ids = _read_task_ids([str(path) for path in owned_paths])
        assert task_ids is not None
        assert len(task_ids) == workload["expected_task_count"]
        for path in owned_paths:
            assert path.parent == inputs_dir
            entry = entries[path.name]
            assert len(path.read_text(encoding="utf-8").splitlines()) == entry[
                "task_count"
            ]


def test_fresh277_is_marked_development_exposed() -> None:
    _, workload = _load_config(
        Path("configs/serving/w5_multitenant.yaml"),
        "swe-rebench-277-development-exposed",
    )
    assert workload["corpus_role"] == "development_exposed"


def test_final_rejects_development_exposed_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "workloads": [
                    {"name": "spent", "corpus_role": "development_exposed"}
                ],
                "policies": ["deadline"],
                "load_levels": [2],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_w5_multitenant.py",
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path / "output"),
            "--workloads",
            "spent",
            "--final",
        ],
    )
    with pytest.raises(ValueError, match="requires a configured heldout_eval"):
        run_w5_main()


def _matrix_inputs(tmp_path: Path, *, table_fraction: float = 0.94):
    replay = tmp_path / "replay"
    profile = tmp_path / "profile"
    replay.mkdir()
    profile.mkdir()
    task_ids = tmp_path / "task_ids.txt"
    task_ids.write_text("task-1\n", encoding="utf-8")
    trigger_table = tmp_path / "trigger.json"
    trigger_table.write_text(
        json.dumps({"metadata": {"restore_cost_fraction": table_fraction}}),
        encoding="utf-8",
    )
    config = {
        "trigger_table": str(trigger_table),
        "continuum_prefill_profile": str(tmp_path / "missing-prefill.json"),
        "restore_cost_fraction": 0.94,
    }
    workload = {
        "name": "demo",
        "replay_trace_root": str(replay),
        "profile_trace_root": str(profile),
        "task_ids_file": str(task_ids),
    }
    return config, workload


def test_w5_preflight_accepts_complete_matching_inputs(tmp_path: Path) -> None:
    config, workload = _matrix_inputs(tmp_path)
    validate_matrix_inputs(config, [workload], ["deadline"])


def test_w5_preflight_rejects_rho_mismatch_before_gpu(tmp_path: Path) -> None:
    config, workload = _matrix_inputs(tmp_path, table_fraction=1.0)
    with pytest.raises(ValueError, match="does not match runtime"):
        validate_matrix_inputs(config, [workload], ["ours"])


def test_w5_preflight_rejects_missing_trace_root_before_gpu(tmp_path: Path) -> None:
    config, workload = _matrix_inputs(tmp_path)
    Path(workload["replay_trace_root"]).rmdir()
    with pytest.raises(ValueError, match="replay_trace_root is not a directory"):
        validate_matrix_inputs(config, [workload], ["deadline"])


def test_w5_preflight_rejects_missing_continuum_profile(tmp_path: Path) -> None:
    config, workload = _matrix_inputs(tmp_path)
    with pytest.raises(ValueError, match="Continuum requires"):
        validate_matrix_inputs(config, [workload], ["continuum"])


def test_prefill_profile_prices_current_context(tmp_path: Path) -> None:
    path = tmp_path / "prefill.json"
    path.write_text(
        json.dumps(
            {
                "measurement": "prefill_recompute_cost",
                "model": "model",
                "kv_cache_dtype": "auto",
                "quantization": None,
                "device": {"host": "host", "device": "GPU"},
                "points": [{"context_tokens": 1000}],
                "quadratic_fit": {"coefficients": [0.001, 0.1, 10.0]},
                "overhead_floor_ms": 5.0,
            }
        ),
        encoding="utf-8",
    )
    profile = load_prefill_cost_profile(
        path,
        expected_model="model",
        expected_kv_cache_dtype="auto",
        expected_quantization=None,
    )
    assert profile.estimate_ms(100) == pytest.approx(30.0)
    assert profile.host_name == "host"
    with pytest.raises(ValueError, match="context_tokens"):
        profile.estimate_ms(1001)


def test_prefill_profiler_accepts_unquantized_model(tmp_path: Path) -> None:
    args = build_prefill_parser().parse_args(
        ["--output", str(tmp_path / "out.json"), "--quantization", "none"]
    )
    assert args.quantization is None


def test_nonpositive_subset_limits_fail_before_gpu(tmp_path: Path) -> None:
    args = argparse.Namespace(
        config=Path("configs/serving/w5_multitenant.yaml"),
        workload="swe-rebench-100-development",
        policy="deadline",
        load=2,
        output=tmp_path / "result.json",
        limit_programs=0,
        max_turns=None,
    )
    with pytest.raises(ValueError, match="limit-programs must be > 0"):
        asyncio.run(run_cell(args))
