from __future__ import annotations

import json

import pytest

from tool_resource.runtime_kb import (
    CompletedCall,
    RuntimeToolResourceKB,
    ToolCallQuery,
)


def _call(
    repo: str,
    command: str | None,
    start: float,
    end: float,
    *,
    tool: str = "exec",
    censored: bool = False,
    cpu: float | None = 1.0,
    mem: float | None = 600.0,
    ambient: float | None = 500.0,
) -> CompletedCall:
    return CompletedCall(
        repo=repo,
        tool_name=tool,
        command=command,
        ts_start=start,
        ts_end=end,
        censored=censored,
        peak_cpu_cores=cpu,
        peak_cpu_cores_eligible=cpu is not None,
        peak_memory_mb=mem,
        peak_memory_mb_eligible=mem is not None,
        ambient_before_mb=ambient,
    )


def _query(
    repo: str,
    command: str | None,
    ts_start: float,
    *,
    tool: str = "exec",
    ambient: float | None = 500.0,
) -> ToolCallQuery:
    return ToolCallQuery(
        repo=repo,
        tool_name=tool,
        command=command,
        ts_start=ts_start,
        ambient_before_mb=ambient,
    )


def _fit(*calls: CompletedCall) -> RuntimeToolResourceKB:
    return RuntimeToolResourceKB.fit_public(calls)


def test_cold_single_head_query_selects_public_binary() -> None:
    kb = _fit(_call("pub", "pytest -q tests", 0.0, 2.0, cpu=3.0))
    prediction = kb.query(_query("r1", "pytest -k foo tests/x.py", 10.0))

    for target in ("latency_ms", "peak_cpu_cores", "peak_memory_mb"):
        assert prediction[target].scope == "public"
        assert prediction[target].key_kind == "binary_head"
    # repo levels were tried first and public has no prefix/exact levels
    path = prediction["latency_ms"].fallback_path
    assert path[0] == "repo:exact_command"
    assert path[-1] == "public:binary_head"
    assert not any(level.startswith("public:command_prefix") for level in path)
    assert not any(level == "public:exact_command" for level in path)


def test_public_has_no_prefix_or_exact_nodes_and_is_immutable() -> None:
    kb = _fit(
        _call("pub", "pytest -q tests", 0.0, 2.0),
        _call("pub", "make && pytest", 3.0, 4.0),
    )
    allowed = {"binary_head", "tool_name", "global"}
    for nodes in kb._public.values():
        assert {kind for kind, _ in nodes} <= allowed

    snapshot = {
        target: dict(nodes) for target, nodes in kb._public.items()
    }
    kb.observe_completed_call(_call("r1", "pytest -q tests", 10.0, 20.0, cpu=8.0))
    kb.query(_query("r1", "pytest -q tests", 30.0))
    assert kb._public == snapshot

    # a different repo still sees the untouched public node
    other = kb.query(_query("r2", "pytest -q tests", 40.0))
    assert other["peak_cpu_cores"].scope == "public"
    assert other["peak_cpu_cores"].conditional_p90 == 1.0


def test_completed_same_repo_exact_command_becomes_available() -> None:
    kb = _fit(_call("pub", "pytest -q tests", 0.0, 2.0, cpu=1.0))
    kb.observe_completed_call(_call("r1", "pytest -q tests", 10.0, 12.0, cpu=6.0))
    prediction = kb.query(_query("r1", "pytest -q tests", 13.0))

    cpu = prediction["peak_cpu_cores"]
    assert cpu.scope == "repo"
    assert cpu.key_kind == "exact_command"
    assert cpu.conditional_p90 == 6.0
    assert cpu.evidence_count == 1


def test_shared_ordered_prefix_backs_off_through_repo_prefix() -> None:
    kb = _fit(_call("pub", "make -j2 all", 0.0, 1.0, cpu=1.0))
    kb.observe_completed_call(_call("r1", "make -j2 all", 10.0, 12.0, cpu=6.0))
    prediction = kb.query(_query("r1", "make -j2 clean", 13.0))

    cpu = prediction["peak_cpu_cores"]
    assert cpu.scope == "repo"
    assert cpu.key_kind == "command_prefix_depth_2"
    assert cpu.conditional_p90 == 6.0


def test_argument_order_is_not_sorted_or_merged() -> None:
    kb = _fit(_call("pub", "tar -x -f a.tar", 0.0, 1.0, cpu=1.0))
    kb.observe_completed_call(_call("r1", "tar -x -f a.tar", 10.0, 12.0, cpu=6.0))
    prediction = kb.query(_query("r1", "tar -f a.tar -x", 13.0))

    cpu = prediction["peak_cpu_cores"]
    # reordered arguments must not hit the exact node; only the shared
    # ordered depth-1 prefix ("tar") matches
    assert cpu.key_kind == "command_prefix_depth_1"
    assert cpu.scope == "repo"


def test_unseen_tail_arguments_miss_deep_nodes_and_back_off() -> None:
    kb = _fit(_call("pub", "pytest tests/a.py", 0.0, 1.0, cpu=1.0))
    kb.observe_completed_call(_call("r1", "pytest tests/a.py", 10.0, 12.0, cpu=6.0))
    prediction = kb.query(_query("r1", "pytest tests/zz_unseen_87213.py", 13.0))

    cpu = prediction["peak_cpu_cores"]
    assert cpu.scope == "repo"
    assert cpu.key_kind == "command_prefix_depth_1"


def test_repo_isolation() -> None:
    kb = _fit(_call("pub", "pytest -q tests", 0.0, 2.0, cpu=1.0))
    kb.observe_completed_call(_call("r1", "pytest -q tests", 10.0, 12.0, cpu=6.0))
    prediction = kb.query(_query("r2", "pytest -q tests", 13.0))

    cpu = prediction["peak_cpu_cores"]
    assert cpu.scope == "public"
    assert cpu.conditional_p90 == 1.0


def test_running_overlapping_same_start_and_future_do_not_leak() -> None:
    kb = _fit(_call("pub", "pytest -q tests", 0.0, 2.0, cpu=1.0))
    kb.observe_completed_call(_call("r1", "pytest -q tests", 0.0, 10.0, cpu=6.0))
    kb.observe_completed_call(_call("r1", "pytest -q tests", 20.0, 21.0, cpu=9.0))

    # overlapping (query starts mid-flight)
    assert kb.query(_query("r1", "pytest -q tests", 5.0))["peak_cpu_cores"].scope == "public"
    # same-start boundary: ts_end == ts_start is not strictly completed
    assert kb.query(_query("r1", "pytest -q tests", 10.0))["peak_cpu_cores"].scope == "public"
    # strictly completed becomes visible; the future call (ts 20-21) does not
    warm = kb.query(_query("r1", "pytest -q tests", 10.5))["peak_cpu_cores"]
    assert warm.scope == "repo"
    assert warm.conditional_p90 == 6.0
    assert warm.evidence_count == 1


def test_compound_command_falls_back_honestly() -> None:
    kb = _fit(
        _call("pub", "pytest -q tests", 0.0, 2.0, cpu=1.0),
        _call("pub", "make && pytest", 3.0, 4.0, cpu=7.0),
    )
    # the compound fit call must not create/feed per-binary nodes
    cpu_nodes = kb._public["peak_cpu_cores"]
    assert cpu_nodes[("binary_head", "pytest")] == (1.0,)
    assert ("binary_head", "make") not in cpu_nodes

    prediction = kb.query(_query("r1", "make && pytest", 10.0))
    cpu = prediction["peak_cpu_cores"]
    assert cpu.scope == "public"
    assert cpu.key_kind == "tool_name"

    # non-shell tool (no command) also lands on the tool-name node
    read_kb = _fit(_call("pub", None, 0.0, 1.0, tool="read_file", cpu=0.5))
    read = read_kb.query(_query("r1", None, 5.0, tool="read_file"))
    assert read["peak_cpu_cores"].key_kind == "tool_name"


def test_memory_residual_adds_query_ambient_and_never_future_state() -> None:
    kb = _fit(_call("pub", "pytest -q tests", 0.0, 2.0, mem=600.0, ambient=500.0))

    low = kb.query(_query("r1", "pytest -q tests", 10.0, ambient=250.0))
    high = kb.query(_query("r1", "pytest -q tests", 11.0, ambient=300.0))
    assert low["peak_memory_mb"].conditional_p90 == pytest.approx(350.0)
    assert high["peak_memory_mb"].conditional_p90 == pytest.approx(400.0)

    missing = kb.query(_query("r1", "pytest -q tests", 12.0, ambient=None))
    assert missing["peak_memory_mb"].conditional_p90 is None
    assert missing["peak_memory_mb"].scope is None
    assert "ambient_before_mb" in missing["peak_memory_mb"].note


def test_serialization_round_trip_preserves_predictions_and_pending() -> None:
    kb = _fit(
        _call("pub", "pytest -q tests", 0.0, 2.0, cpu=1.0),
        _call("pub", "make && pytest", 3.0, 4.0, cpu=7.0),
    )
    kb.observe_completed_call(_call("r1", "pytest -q tests", 10.0, 12.0, cpu=6.0))
    kb.query(_query("r1", "pytest -q tests", 13.0))  # absorb into repo state
    kb.observe_completed_call(_call("r1", "pytest -q tests", 20.0, 30.0, cpu=9.0))

    restored = RuntimeToolResourceKB.from_json_obj(
        json.loads(json.dumps(kb.to_json_obj()))
    )
    for query in (
        _query("r1", "pytest -q tests", 14.0),
        _query("r2", "make && pytest", 14.0),
        _query("r1", None, 14.0, tool="read_file"),
    ):
        assert restored.query(query) == kb.query(query)

    # pending state survives the round trip and releases causally
    late = restored.query(_query("r1", "pytest -q tests", 31.0))["peak_cpu_cores"]
    assert late.evidence_count == 2
    assert late.conditional_p90 == 9.0


def test_target_ineligibility_is_isolated() -> None:
    kb = _fit(_call("pub", "pytest -q tests", 0.0, 2.0, cpu=1.0, mem=600.0))
    # censored call: latency skipped; CPU ineligible; memory eligible
    kb.observe_completed_call(
        _call(
            "r1",
            "pytest -q tests",
            10.0,
            12.0,
            censored=True,
            cpu=None,
            mem=900.0,
            ambient=100.0,
        )
    )
    prediction = kb.query(_query("r1", "pytest -q tests", 13.0, ambient=100.0))

    assert prediction["peak_memory_mb"].scope == "repo"
    assert prediction["peak_memory_mb"].conditional_p90 == pytest.approx(900.0)
    assert prediction["latency_ms"].scope == "public"
    assert prediction["peak_cpu_cores"].scope == "public"


def test_backdated_query_fails_loudly_instead_of_leaking() -> None:
    kb = _fit(_call("pub", "pytest -q tests", 0.0, 2.0, cpu=1.0))
    kb.observe_completed_call(_call("r1", "pytest -q tests", 10.0, 12.0, cpu=6.0))
    kb.query(_query("r1", "pytest -q tests", 20.0))  # absorbs the ts_end=12 call

    with pytest.raises(ValueError, match="backdated query"):
        kb.query(_query("r1", "pytest -q tests", 5.0))

    # equal-time re-query stays legal (absorption is strict-< and idempotent)
    assert kb.query(_query("r1", "pytest -q tests", 20.0))[
        "peak_cpu_cores"
    ].scope == "repo"

    # the guard survives serialization: a restored KB also rejects rewinds
    restored = RuntimeToolResourceKB.from_json_obj(
        json.loads(json.dumps(kb.to_json_obj()))
    )
    with pytest.raises(ValueError, match="backdated query"):
        restored.query(_query("r1", "pytest -q tests", 5.0))


def test_invalid_call_and_unfit_public_fail_fast() -> None:
    with pytest.raises(ValueError):
        _call("r1", "pytest", 10.0, 9.0)
    with pytest.raises(ValueError):
        RuntimeToolResourceKB.fit_public(
            [_call("pub", "pytest", 0.0, 2.0, censored=True, cpu=None, mem=None)]
        )
