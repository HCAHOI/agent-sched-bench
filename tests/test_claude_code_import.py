"""Unit tests for the Claude Code → v5 converter.

Phase US-002 of the claude-code-gantt-import plan. Verifies that:
- The metadata header is well-formed (v5 + scaffold + run_config backfill)
- Assistant records become llm_call actions with all backfill fields
- Thinking blocks are preserved under data.thinking
- tool_result pairs with the matching tool_use via tool_use_id
- toolUseResult sidecar fields flow into data.subagent_meta + subagent_tokens
- Unpaired tool_results still emit a tool_exec action with a descriptive note
- Discarded record types produce zero v5 records
- Sidechain files fold into separate agent_id lanes
- The converted output loads via TraceData and renders through the Gantt payload
- Timestamp conversion handles ISO 8601 with Z suffix
- Per-lane iteration counters are independent
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trace_collect.claude_code_import import (
    SCAFFOLD_LABEL,
    V5_FORMAT_VERSION,
    _iso_to_unix,
    import_claude_code_session,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "claude_code_minimal.jsonl"
SUBAGENT_DIR = REPO_ROOT / "tests" / "fixtures" / "claude_code_minimal" / "subagents"


# ─── Shared fixture: run the converter once per test session ────────────


@pytest.fixture(scope="module")
def converted_trace(tmp_path_factory) -> Path:
    """Convert the minimal CC fixture once and share the output path."""
    out_dir = tmp_path_factory.mktemp("cc-import-default")
    return import_claude_code_session(
        session_path=FIXTURE,
        output_dir=out_dir,
        include_sidechains=True,
    )


@pytest.fixture(scope="module")
def converted_trace_no_sidechains(tmp_path_factory) -> Path:
    """Same fixture but with include_sidechains=False."""
    out_dir = tmp_path_factory.mktemp("cc-import-main-only")
    return import_claude_code_session(
        session_path=FIXTURE,
        output_dir=out_dir,
        include_sidechains=False,
    )


def _load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ─── Timestamp conversion helpers ───────────────────────────────────────


def test_iso_to_unix_handles_z_suffix() -> None:
    ts = _iso_to_unix("2026-04-03T08:33:47.356Z")
    assert ts is not None
    # Sanity: 2026-04 is ~1775xxxxxxx
    assert 1770_000_000 < ts < 1790_000_000


def test_iso_to_unix_handles_none_and_malformed() -> None:
    assert _iso_to_unix(None) is None
    assert _iso_to_unix("") is None
    assert _iso_to_unix("not-a-timestamp") is None


def test_iso_to_unix_handles_offset_timezone() -> None:
    """Explicit offset (not just Z) should also work."""
    ts = _iso_to_unix("2026-04-03T08:33:47.356+00:00")
    assert ts is not None
    assert 1770_000_000 < ts < 1790_000_000


# ─── Fixture sanity ─────────────────────────────────────────────────────


def test_fixtures_exist() -> None:
    assert FIXTURE.exists(), f"missing fixture: {FIXTURE}"
    assert SUBAGENT_DIR.exists(), f"missing subagent dir: {SUBAGENT_DIR}"
    agent_files = list(SUBAGENT_DIR.glob("agent-*.jsonl"))
    assert len(agent_files) >= 1, "expected at least one agent-*.jsonl in fixture"


# ─── Metadata header ────────────────────────────────────────────────────


def test_metadata_has_required_fields(converted_trace: Path) -> None:
    records = _load_records(converted_trace)
    assert records[0]["type"] == "trace_metadata"
    meta = records[0]
    assert meta["trace_format_version"] == V5_FORMAT_VERSION
    assert meta["scaffold"] == SCAFFOLD_LABEL
    assert meta["mode"] == "import"
    assert meta["instance_id"] == "claude_code_minimal"
    assert meta["model"] == "claude-sonnet-4-6"
    assert "source_trace" in meta


def test_metadata_run_config_backfill(converted_trace: Path) -> None:
    """cwd / git_branch / cli_version backfilled under metadata.run_config."""
    records = _load_records(converted_trace)
    meta = records[0]
    run_config = meta.get("run_config", {})
    assert run_config.get("cwd") == "/Users/test/project"
    assert run_config.get("git_branch") == "main"
    assert run_config.get("cli_version") == "2.1.92"


def test_metadata_max_iterations_reflects_main_lane(converted_trace: Path) -> None:
    """max_iterations should be the main-lane assistant count (2 in fixture)."""
    records = _load_records(converted_trace)
    meta = records[0]
    # Fixture has 2 main-lane assistant records
    assert meta.get("max_iterations") == 2


# ─── Assistant → llm_call conversion ───────────────────────────────────


def test_assistant_becomes_llm_call_action(converted_trace: Path) -> None:
    records = _load_records(converted_trace)
    llm_main = [
        r
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "llm_call"
        and r.get("agent_id") == "claude_code_minimal"
    ]
    assert len(llm_main) == 2  # two main-lane assistant records in fixture

    # Check monotonic iteration numbering per lane
    iterations = [r["iteration"] for r in llm_main]
    assert iterations == [0, 1]

    # Check backfill fields on the first llm_call
    first = llm_main[0]
    data = first["data"]
    assert data["prompt_tokens"] == 42
    assert data["completion_tokens"] == 28
    assert data["cache_creation_tokens"] == 1500
    assert data["cache_read_tokens"] == 0
    assert data["cache_ephemeral_5m_tokens"] == 1500
    assert data["cache_ephemeral_1h_tokens"] == 0
    assert data["message_id"] == "msg_test_a0001"
    assert data["service_tier"] == "standard"
    assert data["llm_latency_ms"] > 0


def test_thinking_blocks_preserved(converted_trace: Path) -> None:
    """The fixture's thinking block should end up under data.thinking."""
    records = _load_records(converted_trace)
    llm_main = [
        r
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "llm_call"
        and r.get("agent_id") == "claude_code_minimal"
    ]
    first = llm_main[0]
    data = first["data"]
    assert "thinking" in data, (
        f"thinking block lost during conversion; data keys: {sorted(data.keys())}"
    )
    assert "user wants me to read /etc/hosts" in data["thinking"]


def test_llm_content_extracted_from_text_blocks(converted_trace: Path) -> None:
    records = _load_records(converted_trace)
    llm_main = [
        r
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "llm_call"
        and r.get("agent_id") == "claude_code_minimal"
    ]
    assert "I'll read /etc/hosts" in llm_main[0]["data"]["llm_content"]
    assert "loopback entries" in llm_main[1]["data"]["llm_content"]


def test_raw_response_openai_shape(converted_trace: Path) -> None:
    """raw_response must be in OpenAI shape so _extract_detail_from_action works."""
    records = _load_records(converted_trace)
    first_llm = next(
        r
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "llm_call"
        and r.get("agent_id") == "claude_code_minimal"
    )
    resp = first_llm["data"]["raw_response"]
    assert "choices" in resp
    assert len(resp["choices"]) >= 1
    msg = resp["choices"][0]["message"]
    assert msg.get("role") == "assistant"
    # First assistant has a tool_use block → tool_calls present
    assert "tool_calls" in msg
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "Read"
    # Arguments must be a JSON string (OpenAI convention)
    args = json.loads(tc["function"]["arguments"])
    assert args["file_path"] == "/etc/hosts"


# ─── user + tool_result → tool_exec conversion ──────────────────────────


def test_tool_result_pairs_with_tool_use(converted_trace: Path) -> None:
    records = _load_records(converted_trace)
    tool_actions = [
        r
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "tool_exec"
        and r.get("agent_id") == "claude_code_minimal"
    ]
    assert len(tool_actions) == 1, (
        f"expected exactly 1 tool_exec in main lane; got {len(tool_actions)}"
    )
    ta = tool_actions[0]
    data = ta["data"]
    assert data["tool_name"] == "Read"
    # tool_args is a JSON-string-encoded input dict
    args = json.loads(data["tool_args"])
    assert args["file_path"] == "/etc/hosts"
    assert "127.0.0.1 localhost" in data["tool_result"]
    # Duration came from toolUseResult.totalDurationMs (450.0)
    assert data["duration_ms"] == 450.0
    assert data["success"] is True
    # Iteration inherited from the paired tool_use (not the user record position)
    assert ta["iteration"] == 0


def test_tooluseresult_backfill_fields(converted_trace: Path) -> None:
    """The fixture's user record has toolUseResult with totalTokens + usage."""
    records = _load_records(converted_trace)
    tool_actions = [
        r
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "tool_exec"
        and r.get("agent_id") == "claude_code_minimal"
    ]
    data = tool_actions[0]["data"]
    # subagent_tokens backfill from toolUseResult.totalTokens + usage
    assert "subagent_tokens" in data
    assert data["subagent_tokens"]["total"] == 12
    assert data["subagent_tokens"]["usage"]["input_tokens"] == 6
    # subagent_tool_use_count from toolUseResult.totalToolUseCount
    assert data.get("subagent_tool_use_count") == 1


def test_tool_exec_success_inferred_from_is_error(converted_trace: Path) -> None:
    records = _load_records(converted_trace)
    tool_actions = [
        r
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "tool_exec"
        and r.get("agent_id") == "claude_code_minimal"
    ]
    # Fixture has is_error: false → success: true
    assert tool_actions[0]["data"]["success"] is True


# ─── Discarded types ───────────────────────────────────────────────────


def test_discarded_types_do_not_produce_records(converted_trace: Path) -> None:
    """file-history-snapshot + system records in the fixture should produce ZERO v5 records."""
    records = _load_records(converted_trace)
    # No v5 record should have type in the discardable set
    for r in records:
        assert r["type"] in {"trace_metadata", "action", "summary"}, (
            f"unexpected record type in output: {r['type']}"
        )


# ─── Sidechain folding ──────────────────────────────────────────────────


def test_sidechain_becomes_separate_lane(converted_trace: Path) -> None:
    records = _load_records(converted_trace)
    agents = sorted(
        {r.get("agent_id") for r in records if r.get("type") == "action"}
    )
    # Main lane + one subagent lane from the fixture
    assert len(agents) == 2, f"expected 2 agent lanes, got {agents}"
    assert "claude_code_minimal" in agents
    assert any(a.startswith("agent-test") for a in agents), (
        f"expected a sidechain lane starting with 'agent-test'; got {agents}"
    )


def test_no_sidechains_when_opted_out(
    converted_trace_no_sidechains: Path,
) -> None:
    records = _load_records(converted_trace_no_sidechains)
    agents = sorted(
        {r.get("agent_id") for r in records if r.get("type") == "action"}
    )
    assert len(agents) == 1, (
        f"expected only main lane with include_sidechains=False; got {agents}"
    )
    assert agents[0] == "claude_code_minimal"


def test_iteration_counter_per_agent_lane(converted_trace: Path) -> None:
    """Main lane and subagent lane each have their own iteration counter."""
    records = _load_records(converted_trace)
    main_iters = sorted(
        r["iteration"]
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "llm_call"
        and r.get("agent_id") == "claude_code_minimal"
    )
    sub_iters = sorted(
        r["iteration"]
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "llm_call"
        and r.get("agent_id", "").startswith("agent-test")
    )
    # Both start from 0 — counter is per-lane, not global
    assert main_iters == [0, 1]
    assert sub_iters == [0, 1]


# ─── End-to-end: TraceData + Gantt payload ─────────────────────────────


def test_converted_output_loads_via_trace_data(converted_trace: Path) -> None:
    """The converted trace must load through the strict v5 reader."""
    from trace_collect.trace_inspector import TraceData

    data = TraceData.load(converted_trace)
    assert data.metadata["trace_format_version"] == 5
    assert data.metadata["scaffold"] == SCAFFOLD_LABEL
    assert len(data.actions) >= 3  # 2 main llm_call + 1 main tool_exec minimum


def test_build_gantt_payload_succeeds(converted_trace: Path) -> None:
    """The converted trace renders through build_gantt_payload_multi."""
    from demo.gantt_viewer.backend.payload import build_gantt_payload_multi
    from trace_collect.trace_inspector import TraceData

    data = TraceData.load(converted_trace)
    payload = build_gantt_payload_multi([("cc-test", data)])

    assert "registries" in payload
    assert "spans" in payload["registries"]
    assert "traces" in payload
    assert len(payload["traces"]) == 1

    lanes = payload["traces"][0]["lanes"]
    assert len(lanes) >= 2  # main + subagent

    # Collect all span types across lanes
    all_span_types = set()
    for lane in lanes:
        for span in lane.get("spans", []):
            all_span_types.add(span["type"])
    assert "llm" in all_span_types
    assert "tool" in all_span_types


# ─── Summary record ─────────────────────────────────────────────────────


def test_summary_record_per_lane(converted_trace: Path) -> None:
    records = _load_records(converted_trace)
    summaries = [r for r in records if r.get("type") == "summary"]
    # One summary per agent lane (main + sidechain = 2)
    assert len(summaries) == 2

    main_summary = next(
        s for s in summaries if s["agent_id"] == "claude_code_minimal"
    )
    # 2 assistant records in main lane fixture
    assert main_summary["n_steps"] == 2
    assert main_summary["n_tool_actions"] == 1
    assert main_summary["elapsed_s"] > 0
    assert main_summary["total_tokens"] > 0


# ─── Follow-ups from the Gate-C reviewer (M1-M5) ────────────────────────


def _write_synthetic_session(path: Path, records: list[dict]) -> None:
    """Helper: dump a list of records as JSONL to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def test_tooluseresult_string_variant_does_not_crash(tmp_path: Path) -> None:
    """M1 (reviewer): some CC records carry toolUseResult as a plain string.

    The converter must type-check defensively and fall back to an empty
    sidecar rather than crashing with ``AttributeError: 'str' object has
    no attribute 'get'``. Caught on a real 1.6MB session during the
    US-001 smoke run.
    """
    session = tmp_path / "cc-str-sidecar.jsonl"
    _write_synthetic_session(
        session,
        [
            {
                "type": "assistant",
                "sessionId": "cc-str-sidecar",
                "uuid": "a1",
                "timestamp": "2026-04-08T10:00:00.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.92",
                "isSidechain": False,
                "message": {
                    "id": "msg_str_test",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {},
                    },
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_str_0001",
                            "name": "Read",
                            "input": {"file_path": "/tmp/x"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "sessionId": "cc-str-sidecar",
                "uuid": "u1",
                "timestamp": "2026-04-08T10:00:00.500Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.92",
                "isSidechain": False,
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_str_0001",
                            "content": [{"type": "text", "text": "file contents"}],
                            "is_error": False,
                        }
                    ],
                },
                # The quirk: toolUseResult is a plain string instead of a dict.
                "toolUseResult": "file contents",
            },
        ],
    )

    # Should not crash
    result = import_claude_code_session(
        session_path=session, output_dir=tmp_path / "out", include_sidechains=False
    )
    records = _load_records(result)
    tool_actions = [
        r for r in records if r.get("action_type") == "tool_exec"
    ]
    assert len(tool_actions) == 1
    # duration_ms was derived from timestamps, not from the string sidecar
    assert tool_actions[0]["data"]["duration_ms"] > 0
    # subagent_tokens should NOT be present because sidecar wasn't a dict
    assert "subagent_tokens" not in tool_actions[0]["data"]


def test_tool_result_with_is_error_true(tmp_path: Path) -> None:
    """M2 (reviewer): tool_result with is_error: true → data.success is False."""
    session = tmp_path / "cc-is-error.jsonl"
    _write_synthetic_session(
        session,
        [
            {
                "type": "assistant",
                "sessionId": "cc-is-error",
                "uuid": "a1",
                "timestamp": "2026-04-08T10:00:00.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.92",
                "isSidechain": False,
                "message": {
                    "id": "msg_err_test",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {},
                    },
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_err_0001",
                            "name": "Bash",
                            "input": {"command": "false"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "sessionId": "cc-is-error",
                "uuid": "u1",
                "timestamp": "2026-04-08T10:00:00.500Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.92",
                "isSidechain": False,
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_err_0001",
                            "content": [{"type": "text", "text": "exit 1"}],
                            "is_error": True,
                        }
                    ],
                },
                "toolUseResult": {
                    "status": "error",
                    "totalDurationMs": 50.0,
                    "totalTokens": 2,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "totalToolUseCount": 1,
                },
            },
        ],
    )

    result = import_claude_code_session(
        session_path=session, output_dir=tmp_path / "out", include_sidechains=False
    )
    records = _load_records(result)
    tool_actions = [r for r in records if r.get("action_type") == "tool_exec"]
    assert len(tool_actions) == 1
    assert tool_actions[0]["data"]["success"] is False


def test_orphan_tool_use_is_drained_with_note(tmp_path: Path) -> None:
    """M3 (reviewer): orphan tool_use (no matching tool_result) must emit a stub.

    CLAUDE.md §5 "preserve all intermediate outputs" — silently dropping
    an orphaned tool_use loses the fact that the invocation ever
    happened, which is not acceptable.
    """
    session = tmp_path / "cc-orphan.jsonl"
    _write_synthetic_session(
        session,
        [
            {
                "type": "assistant",
                "sessionId": "cc-orphan",
                "uuid": "a1",
                "timestamp": "2026-04-08T10:00:00.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.92",
                "isSidechain": False,
                "message": {
                    "id": "msg_orphan",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {},
                    },
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_orphan_0001",
                            "name": "Read",
                            "input": {"file_path": "/never/arrived"},
                        }
                    ],
                },
            },
            # No user record with a matching tool_result — session cut off.
        ],
    )

    result = import_claude_code_session(
        session_path=session, output_dir=tmp_path / "out", include_sidechains=False
    )
    records = _load_records(result)
    tool_actions = [r for r in records if r.get("action_type") == "tool_exec"]
    assert len(tool_actions) == 1, (
        f"orphan tool_use should still produce a stub tool_exec; got {len(tool_actions)}"
    )
    orphan = tool_actions[0]
    assert orphan["data"]["tool_name"] == "Read"
    assert orphan["data"]["success"] is False
    assert orphan["data"]["duration_ms"] == 0.0
    assert "orphan tool_use" in orphan["data"].get("note", "")


def test_multiple_parallel_tool_uses_in_one_assistant(tmp_path: Path) -> None:
    """M4 (reviewer): assistant emits multiple tool_use blocks in parallel.

    Claude Code supports calling multiple tools in one turn. Each
    tool_use must get its own tool_exec action paired via tool_use_id.
    """
    session = tmp_path / "cc-parallel-tools.jsonl"
    _write_synthetic_session(
        session,
        [
            {
                "type": "assistant",
                "sessionId": "cc-parallel-tools",
                "uuid": "a1",
                "timestamp": "2026-04-08T10:00:00.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.92",
                "isSidechain": False,
                "message": {
                    "id": "msg_parallel",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {},
                    },
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_par_0001",
                            "name": "Read",
                            "input": {"file_path": "/a"},
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_par_0002",
                            "name": "Read",
                            "input": {"file_path": "/b"},
                        },
                    ],
                },
            },
            {
                "type": "user",
                "sessionId": "cc-parallel-tools",
                "uuid": "u1",
                "timestamp": "2026-04-08T10:00:00.500Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.92",
                "isSidechain": False,
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_par_0001",
                            "content": [{"type": "text", "text": "contents of /a"}],
                            "is_error": False,
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_par_0002",
                            "content": [{"type": "text", "text": "contents of /b"}],
                            "is_error": False,
                        },
                    ],
                },
                "toolUseResult": {
                    "status": "completed",
                    "totalDurationMs": 120.0,
                    "totalToolUseCount": 2,
                },
            },
        ],
    )

    result = import_claude_code_session(
        session_path=session, output_dir=tmp_path / "out", include_sidechains=False
    )
    records = _load_records(result)
    tool_actions = [r for r in records if r.get("action_type") == "tool_exec"]
    assert len(tool_actions) == 2, (
        f"expected 2 tool_exec actions (one per parallel tool_use); got {len(tool_actions)}"
    )
    # Both must have iteration 0 (same paired assistant)
    assert all(t["iteration"] == 0 for t in tool_actions)
    # Different tool_use_ids → different action_ids
    action_ids = {t["action_id"] for t in tool_actions}
    assert len(action_ids) == 2
    # Both tool_name are Read
    assert all(t["data"]["tool_name"] == "Read" for t in tool_actions)
    # Different tool_args (one reads /a, the other /b)
    args_paths = sorted(
        json.loads(t["data"]["tool_args"])["file_path"] for t in tool_actions
    )
    assert args_paths == ["/a", "/b"]


def test_empty_session_with_only_discardable_records(tmp_path: Path) -> None:
    """M5 (reviewer): a session with zero assistant records still produces valid output.

    The converter must not crash on sessions containing only
    file-history-snapshot / system / permission-mode records. It should
    produce a valid v5 trace with zero actions and the metadata header
    still well-formed.
    """
    session = tmp_path / "cc-empty.jsonl"
    _write_synthetic_session(
        session,
        [
            {
                "type": "file-history-snapshot",
                "snapshot": {"timestamp": "2026-04-08T10:00:00.000Z"},
            },
            {
                "type": "system",
                "subtype": "local_command",
                "content": "/help",
                "timestamp": "2026-04-08T10:00:01.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.92",
            },
            {
                "type": "permission-mode",
                "timestamp": "2026-04-08T10:00:02.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.92",
            },
        ],
    )

    result = import_claude_code_session(
        session_path=session, output_dir=tmp_path / "out", include_sidechains=False
    )
    records = _load_records(result)

    # Metadata header present
    assert records[0]["type"] == "trace_metadata"
    assert records[0]["trace_format_version"] == 5
    assert records[0]["scaffold"] == "claude-code"
    # No actions
    actions = [r for r in records if r.get("type") == "action"]
    assert actions == []
    # Summary still emitted per lane
    summaries = [r for r in records if r.get("type") == "summary"]
    assert len(summaries) == 1
    assert summaries[0]["n_steps"] == 0
    assert summaries[0]["n_tool_actions"] == 0

    # Loads via TraceData without ValueError
    from trace_collect.trace_inspector import TraceData

    data = TraceData.load(result)
    assert data.metadata["trace_format_version"] == 5
    assert data.actions == []


# ─── FIX-1 + FIX-2: collector path convention + last-prompt discard ─────


def test_collector_style_stem_falls_back_to_sessionId(tmp_path: Path) -> None:
    """FIX-1: when the filename stem is not UUID-shaped, the converter must
    fall back to the harvested ``sessionId`` for the canonical session UUID.

    Real swe-rebench traces are copied out of containers as ``trace.jsonl``,
    so naive ``session_path.stem`` would yield ``"trace"`` for every task —
    causing every converted output to clobber every other task's output at
    ``claude-code-import/trace/trace.jsonl``. The fix harvests sessionId
    from the records and uses that instead.
    """
    session = tmp_path / "trace.jsonl"  # collector-style filename
    real_uuid = "2a49ce6f-616e-4072-9a35-6934dcce7383"
    _write_synthetic_session(
        session,
        [
            {
                "type": "assistant",
                "sessionId": real_uuid,
                "uuid": "a1",
                "timestamp": "2026-04-08T11:27:19.036Z",
                "cwd": "/testbed",
                "gitBranch": "master",
                "version": "2.1.96",
                "isSidechain": False,
                "message": {
                    "id": "msg_collector_test",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 8,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 13610,
                        "cache_creation": {},
                    },
                    "content": [{"type": "text", "text": "Reading the issue."}],
                },
            },
        ],
    )

    out_dir = tmp_path / "out"
    result = import_claude_code_session(
        session_path=session, output_dir=out_dir, include_sidechains=False
    )

    # The output landed under the REAL UUID, not under "trace"
    expected = out_dir / "claude-code-import" / real_uuid / f"{real_uuid}.jsonl"
    assert result == expected, (
        f"FIX-1 broken: expected {expected}, got {result}"
    )
    assert not (out_dir / "claude-code-import" / "trace" / "trace.jsonl").exists()

    records = _load_records(result)
    meta = records[0]
    assert meta["instance_id"] == real_uuid
    # Every action's agent_id is the real UUID, not "trace"
    actions = [r for r in records if r.get("type") == "action"]
    assert actions, "expected at least one action"
    for action in actions:
        assert action["agent_id"] == real_uuid, (
            f"FIX-1 broken: action {action.get('action_id')} has "
            f"agent_id={action['agent_id']!r} (should be the real UUID)"
        )


def test_last_prompt_record_is_discarded(tmp_path: Path) -> None:
    """FIX-2: ``last-prompt`` records (CC CLI >= 2.1.96) must be explicitly
    discarded, not silently consumed by fall-through.

    A ``last-prompt`` record carries no timestamp and no content blocks,
    just ``{type, lastPrompt, sessionId}``. It must produce zero v5 records
    and must not interfere with surrounding assistant/user records.
    """
    session = tmp_path / "cc-last-prompt.jsonl"
    _write_synthetic_session(
        session,
        [
            {
                "type": "assistant",
                "sessionId": "cc-last-prompt-test",
                "uuid": "a1",
                "timestamp": "2026-04-08T10:00:00.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.96",
                "isSidechain": False,
                "message": {
                    "id": "msg_lp_test",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {},
                    },
                    "content": [{"type": "text", "text": "done"}],
                },
            },
            # The new CC 2.1.96 record type — must be discarded silently.
            {
                "type": "last-prompt",
                "lastPrompt": "Fix this issue: ...",
                "sessionId": "cc-last-prompt-test",
            },
        ],
    )

    result = import_claude_code_session(
        session_path=session, output_dir=tmp_path / "out", include_sidechains=False
    )
    records = _load_records(result)

    actions = [r for r in records if r.get("type") == "action"]
    # Exactly one llm_call from the assistant, zero from last-prompt.
    assert len(actions) == 1
    assert actions[0]["action_type"] == "llm_call"
    # Verify last-prompt is in the explicit discard set, not just fall-through.
    from trace_collect.claude_code_import import _DISCARDABLE_TYPES
    assert "last-prompt" in _DISCARDABLE_TYPES


# ─── FIX-4: per-tool rich backfill (Bash/Edit/Read) ─────────────────────


def _make_synthetic_tool_session(
    tmp_path: Path,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_use_result: Any,
    is_error: bool | None = False,
    result_text: str = "result content",
) -> Path:
    """Helper: build a 2-record session (assistant→user) for a single tool."""
    session = tmp_path / f"cc-{tool_name.lower()}-test.jsonl"
    tu_id = f"toolu_{tool_name.lower()}_0001"
    user_msg: dict[str, Any] = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tu_id,
                "content": [{"type": "text", "text": result_text}],
            }
        ],
    }
    if is_error is not None:
        user_msg["content"][0]["is_error"] = is_error
    user_record: dict[str, Any] = {
        "type": "user",
        "sessionId": f"cc-{tool_name.lower()}-test",
        "uuid": "u1",
        "timestamp": "2026-04-08T10:00:00.500Z",
        "cwd": "/tmp",
        "gitBranch": "main",
        "version": "2.1.96",
        "isSidechain": False,
        "message": user_msg,
        "toolUseResult": tool_use_result,
    }
    _write_synthetic_session(
        session,
        [
            {
                "type": "assistant",
                "sessionId": f"cc-{tool_name.lower()}-test",
                "uuid": "a1",
                "timestamp": "2026-04-08T10:00:00.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.96",
                "isSidechain": False,
                "message": {
                    "id": f"msg_{tool_name.lower()}_test",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {},
                    },
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tu_id,
                            "name": tool_name,
                            "input": tool_input,
                        }
                    ],
                },
            },
            user_record,
        ],
    )
    return session


def _convert_and_get_tool(session: Path, tmp_path: Path) -> dict[str, Any]:
    """Helper: convert a single-tool session and return its tool_exec action."""
    out_dir = tmp_path / f"out-{session.stem}"
    result = import_claude_code_session(
        session_path=session, output_dir=out_dir, include_sidechains=False
    )
    records = _load_records(result)
    tool_actions = [r for r in records if r.get("action_type") == "tool_exec"]
    assert len(tool_actions) == 1, f"expected 1 tool_exec, got {len(tool_actions)}"
    return tool_actions[0]


def test_bash_interrupted_marks_success_false(tmp_path: Path) -> None:
    """FIX-4: Bash.interrupted=True is the only reliable failure signal
    for interrupted bash commands. tool_result.is_error is typically
    None/False even when the command was killed."""
    session = _make_synthetic_tool_session(
        tmp_path,
        tool_name="Bash",
        tool_input={"command": "sleep 10"},
        tool_use_result={
            "interrupted": True,
            "isImage": False,
            "noOutputExpected": False,
            "stderr": "",
            "stdout": "",
        },
        is_error=None,  # the real shape: is_error absent on interrupted bash
    )
    tool_action = _convert_and_get_tool(session, tmp_path)
    assert tool_action["data"]["success"] is False, (
        "FIX-4 broken: interrupted Bash should be marked success=False"
    )


def test_bash_stderr_preview_populated(tmp_path: Path) -> None:
    """FIX-4: Bash stderr backfilled into data.stderr_preview, capped at 2000 chars."""
    session = _make_synthetic_tool_session(
        tmp_path,
        tool_name="Bash",
        tool_input={"command": "false"},
        tool_use_result={
            "interrupted": False,
            "isImage": False,
            "noOutputExpected": False,
            "stderr": "E: segfault at 0x0",
            "stdout": "",
        },
    )
    tool_action = _convert_and_get_tool(session, tmp_path)
    assert tool_action["data"].get("stderr_preview") == "E: segfault at 0x0"


def test_edit_structured_patch_preserved(tmp_path: Path) -> None:
    """FIX-4: Edit.structuredPatch (a parsed diff) is preserved as
    data.structured_patch — far more useful than re-derivable from newString."""
    patch = [
        {
            "oldStart": 10,
            "newStart": 10,
            "lines": ["-old line", "+new line"],
        }
    ]
    session = _make_synthetic_tool_session(
        tmp_path,
        tool_name="Edit",
        tool_input={"file_path": "/etc/hosts", "old_string": "old", "new_string": "new"},
        tool_use_result={
            "filePath": "/etc/hosts",
            "oldString": "old",
            "newString": "new",
            "originalFile": "old line",
            "replaceAll": False,
            "structuredPatch": patch,
            "userModified": False,
        },
    )
    tool_action = _convert_and_get_tool(session, tmp_path)
    assert tool_action["data"].get("structured_patch") == patch


def test_edit_user_modified_flag_preserved(tmp_path: Path) -> None:
    """FIX-4: Edit.userModified=True flagged into backfill."""
    session = _make_synthetic_tool_session(
        tmp_path,
        tool_name="Edit",
        tool_input={"file_path": "/x", "old_string": "a", "new_string": "b"},
        tool_use_result={
            "filePath": "/x",
            "oldString": "a",
            "newString": "b",
            "originalFile": "a",
            "replaceAll": False,
            "structuredPatch": [],
            "userModified": True,
        },
    )
    tool_action = _convert_and_get_tool(session, tmp_path)
    assert tool_action["data"].get("user_modified") is True


def test_read_file_meta_preserved(tmp_path: Path) -> None:
    """FIX-4: Read.file dict is filtered down to relevant metadata keys."""
    session = _make_synthetic_tool_session(
        tmp_path,
        tool_name="Read",
        tool_input={"file_path": "/etc/hosts"},
        tool_use_result={
            "type": "text",
            "file": {
                "filePath": "/etc/hosts",
                "numLines": 5,
                "totalLines": 5,
                "extraIgnoredKey": "should be filtered out",
            },
        },
    )
    tool_action = _convert_and_get_tool(session, tmp_path)
    fm = tool_action["data"].get("file_meta")
    assert fm == {"filePath": "/etc/hosts", "numLines": 5, "totalLines": 5}


def test_unknown_tool_backfill_is_noop(tmp_path: Path) -> None:
    """FIX-4: an unknown tool_name produces no per-tool backfill keys."""
    session = _make_synthetic_tool_session(
        tmp_path,
        tool_name="MysteryFutureTool",
        tool_input={"q": 1},
        tool_use_result={"some_field": "some_value"},
    )
    tool_action = _convert_and_get_tool(session, tmp_path)
    data = tool_action["data"]
    # Per-tool backfill keys must NOT be present
    for key in ("stderr_preview", "structured_patch", "user_modified", "file_meta"):
        assert key not in data, (
            f"FIX-4 broken: unknown tool produced unexpected backfill key {key!r}"
        )


# ─── FIX-5: requestId provenance backfill ───────────────────────────────


def _make_assistant_only_session(
    tmp_path: Path,
    extra_top_level: dict[str, Any] | None = None,
) -> Path:
    """Build a 1-record session with one assistant message."""
    session = tmp_path / "cc-request-id-test.jsonl"
    record: dict[str, Any] = {
        "type": "assistant",
        "sessionId": "cc-request-id-test",
        "uuid": "a1",
        "timestamp": "2026-04-08T10:00:00.000Z",
        "cwd": "/tmp",
        "gitBranch": "main",
        "version": "2.1.96",
        "isSidechain": False,
        "message": {
            "id": "msg_req_test",
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "usage": {
                "input_tokens": 5,
                "output_tokens": 3,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation": {},
            },
            "content": [{"type": "text", "text": "ack"}],
        },
    }
    if extra_top_level:
        record.update(extra_top_level)
    _write_synthetic_session(session, [record])
    return session


def test_request_id_preserved(tmp_path: Path) -> None:
    """FIX-5: top-level requestId is backfilled into data.request_id."""
    session = _make_assistant_only_session(
        tmp_path,
        extra_top_level={"requestId": "req_011CZrDvajP6jsqd1oCpD8RG"},
    )
    result = import_claude_code_session(
        session_path=session, output_dir=tmp_path / "out", include_sidechains=False
    )
    records = _load_records(result)
    llm = next(r for r in records if r.get("action_type") == "llm_call")
    assert llm["data"]["request_id"] == "req_011CZrDvajP6jsqd1oCpD8RG"


def test_request_id_absent_is_empty_string(tmp_path: Path) -> None:
    """FIX-5: missing requestId yields empty string (stable schema, not missing key)."""
    session = _make_assistant_only_session(tmp_path)  # no requestId
    result = import_claude_code_session(
        session_path=session, output_dir=tmp_path / "out", include_sidechains=False
    )
    records = _load_records(result)
    llm = next(r for r in records if r.get("action_type") == "llm_call")
    assert "request_id" in llm["data"], (
        "FIX-5 broken: request_id key should be present even when absent at source"
    )
    assert llm["data"]["request_id"] == ""


def test_request_id_explicit_none_becomes_empty_string(tmp_path: Path) -> None:
    """FIX-5 guard: requestId=None must coerce to empty string, not None.

    The ``or ""`` guard in the converter handles the case where the
    field is present but explicitly null (rare but legal in JSON).
    Without it, ``data.request_id`` would carry the literal Python
    ``None``, polluting the v5 schema with mixed types.
    """
    session = _make_assistant_only_session(
        tmp_path, extra_top_level={"requestId": None}
    )
    result = import_claude_code_session(
        session_path=session, output_dir=tmp_path / "out", include_sidechains=False
    )
    records = _load_records(result)
    llm = next(r for r in records if r.get("action_type") == "llm_call")
    assert llm["data"]["request_id"] == "", (
        f"FIX-5 None-guard broken: got {llm['data']['request_id']!r}, expected ''"
    )


def test_is_error_takes_precedence_over_interrupted(tmp_path: Path) -> None:
    """FIX-4 precedence: tool_result.is_error wins over per-tool overrides.

    is_error=False + interrupted=True → success=True (is_error said
    'not an error', so we trust the most authoritative direct signal).
    """
    session = _make_synthetic_tool_session(
        tmp_path,
        tool_name="Bash",
        tool_input={"command": "echo ok"},
        tool_use_result={
            "interrupted": True,
            "isImage": False,
            "noOutputExpected": False,
            "stderr": "",
            "stdout": "ok",
        },
        is_error=False,  # explicit "not an error" — must win
    )
    tool_action = _convert_and_get_tool(session, tmp_path)
    assert tool_action["data"]["success"] is True, (
        "FIX-4 precedence broken: is_error=False should override interrupted=True"
    )


# ─── End-to-end integration: real swe-rebench Claude Code trace ────────


REAL_CC_TRACE = (
    REPO_ROOT
    / "traces"
    / "swe-rebench"
    / "claude-code-haiku"
    / "encode__httpx-2701"
    / "attempt_1"
    / "trace.jsonl"
)
REAL_CC_SESSION_UUID = "2a49ce6f-616e-4072-9a35-6934dcce7383"


def test_real_swe_rebench_claude_code_trace_converts_cleanly(
    tmp_path: Path,
) -> None:
    """End-to-end smoke against the real swe-rebench Claude Code trace.

    This test exercises every fix in this Ralph pass against a real
    collector-produced CC session: FIX-1 (UUID fallback because the
    file is named ``trace.jsonl``), FIX-2 (the trailing ``last-prompt``
    record is discarded), FIX-4 (Edit tool actions get
    ``structured_patch`` backfill), FIX-5 (every assistant has
    ``request_id``).

    Skipped gracefully when the artifact is not present so the suite
    remains runnable on machines without the trace.
    """
    if not REAL_CC_TRACE.exists():
        pytest.skip(f"real CC trace not present at {REAL_CC_TRACE}")

    # FIX-2 source-side check: confirm the real trace actually contains a
    # last-prompt record. Without this guard, a future schema drift that
    # removes last-prompt from CC would render the discard assertion
    # below vacuous (no record to discard → trivially passes).
    with open(REAL_CC_TRACE, encoding="utf-8") as f:
        has_last_prompt = any(
            json.loads(line).get("type") == "last-prompt" for line in f
        )
    assert has_last_prompt, (
        "FIX-2 verification gap: real trace no longer contains a "
        "last-prompt record; this test cannot validate the discard"
    )

    out_dir = tmp_path / "real-cc-out"
    result = import_claude_code_session(
        session_path=REAL_CC_TRACE,
        output_dir=out_dir,
        include_sidechains=False,
    )

    # FIX-1: output landed under the real session UUID, not "trace"
    expected = (
        out_dir
        / "claude-code-import"
        / REAL_CC_SESSION_UUID
        / f"{REAL_CC_SESSION_UUID}.jsonl"
    )
    assert result == expected, (
        f"FIX-1 broken: expected {expected}, got {result}"
    )
    assert not (out_dir / "claude-code-import" / "trace").exists()

    records = _load_records(result)

    # Metadata header reflects the real session
    meta = records[0]
    assert meta["type"] == "trace_metadata"
    assert meta["trace_format_version"] == V5_FORMAT_VERSION
    assert meta["instance_id"] == REAL_CC_SESSION_UUID
    assert meta["model"] == "claude-sonnet-4-6"
    rc = meta.get("run_config", {})
    assert rc.get("cwd") == "/testbed"
    assert rc.get("git_branch") == "master"
    assert rc.get("cli_version") == "2.1.96"

    # Action counts match the verified record-type inventory
    llm_calls = [
        r for r in records
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    ]
    tool_execs = [
        r for r in records
        if r.get("type") == "action" and r.get("action_type") == "tool_exec"
    ]
    assert len(llm_calls) == 32, f"expected 32 llm_calls, got {len(llm_calls)}"
    assert len(tool_execs) == 22, f"expected 22 tool_execs, got {len(tool_execs)}"

    # FIX-1: every action's agent_id is the real UUID
    for action in llm_calls + tool_execs:
        assert action["agent_id"] == REAL_CC_SESSION_UUID

    # No orphans, no unpaired
    for tool in tool_execs:
        note = tool["data"].get("note", "")
        assert not note.startswith("orphan"), f"unexpected orphan: {note}"
        assert not note.startswith("unpaired"), f"unexpected unpaired: {note}"

    # Real trace has 5 thinking-only assistant records
    thinking_count = sum(
        1 for r in llm_calls if r["data"].get("thinking", "").strip()
    )
    assert thinking_count >= 5, (
        f"expected >=5 llm_calls with thinking; got {thinking_count}"
    )

    # FIX-5: every llm_call has request_id (real CC 2.1.96 always emits it)
    missing_req = [
        r for r in llm_calls if not r["data"].get("request_id", "").startswith("req_")
    ]
    assert not missing_req, (
        f"FIX-5 broken: {len(missing_req)} llm_calls missing request_id"
    )

    # FIX-4: the Edit tool calls (3 of them) all have structured_patch
    edits = [t for t in tool_execs if t["data"]["tool_name"] == "Edit"]
    assert len(edits) == 3
    for edit in edits:
        assert edit["data"].get("structured_patch") is not None, (
            f"FIX-4 broken: Edit action {edit['action_id']} missing structured_patch"
        )

    # Trace loads through the strict v5 reader and renders via the Gantt
    from demo.gantt_viewer.backend.payload import build_gantt_payload_multi
    from trace_collect.trace_inspector import TraceData

    data = TraceData.load(result)
    assert data.metadata["trace_format_version"] == 5
    assert data.metadata["scaffold"] == SCAFFOLD_LABEL

    payload = build_gantt_payload_multi([("real-cc", data)])
    assert payload["traces"][0]["lanes"], "expected at least one lane"
    span_types: set[str] = set()
    for lane in payload["traces"][0]["lanes"]:
        for span in lane.get("spans", []):
            span_types.add(span["type"])
    assert "llm" in span_types
    assert "tool" in span_types
