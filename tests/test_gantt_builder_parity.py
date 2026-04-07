"""Parity tests for src/trace_collect/gantt_builder.js vs gantt_data.py.

The JS builder must produce the same payload shape as the Python builder
when fed the same JSONL trace — otherwise a trace loaded via the CLI
(Python path) and the same trace loaded via drag-drop on an open viewer
(JS path) would render differently, silently breaking the visualization.

These tests skip cleanly when ``node`` is not available on PATH, so a
fresh checkout without node installed still has a green suite.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from trace_collect.gantt_data import build_gantt_payload_multi
from trace_collect.trace_inspector import TraceData


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_JS = REPO_ROOT / "src" / "trace_collect" / "gantt_builder.js"

_NODE = shutil.which("node")
_SKIP_REASON = "node is not installed; JS parity tests skipped"


def _has_node() -> bool:
    return _NODE is not None


def _run_js_builder(trace_text: str, label: str) -> dict[str, Any]:
    """Invoke the JS builder through node, return the multi-payload dict."""
    script = f"""
const path = require('path');
const GanttBuilder = require({json.dumps(str(BUILDER_JS))});
let stdin = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => stdin += chunk);
process.stdin.on('end', () => {{
  const parsed = GanttBuilder.parseJsonl(stdin);
  const payload = GanttBuilder.buildPayloadMulti([{{label: {json.dumps(label)}, parsed}}]);
  process.stdout.write(JSON.stringify(payload));
}});
"""
    proc = subprocess.run(
        [_NODE, "-e", script],
        input=trace_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"node exited {proc.returncode}\nstderr:\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def _normalize(obj: Any) -> Any:
    """Round-trip through json to collapse tuples→lists and None differences
    so Python and JS outputs compare cleanly on deep equality."""
    return json.loads(json.dumps(obj))


def _write_trace(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    trace = tmp_path / "trace.jsonl"
    head = {
        "type": "trace_metadata",
        "scaffold": "synthetic",
        "model": "test-model",
        "trace_format_version": 4,
        "max_iterations": 80,
    }
    with trace.open("w") as fh:
        fh.write(json.dumps(head) + "\n")
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return trace


# ── Fixture builders ───────────────────────────────────────────────


def _llm_action(iteration: int, ts_start: float, ts_end: float, *,
                content: str | None = None,
                tool_call: tuple[str, dict] | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_call is not None:
        name, args = tool_call
        msg["tool_calls"] = [{
            "function": {"name": name, "arguments": json.dumps(args)}
        }]
    return {
        "type": "action",
        "action_type": "llm_call",
        "action_id": f"llm_{iteration}",
        "agent_id": "a1",
        "iteration": iteration,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "data": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "llm_latency_ms": (ts_end - ts_start) * 1000,
            "raw_response": {"choices": [{"message": msg}]},
        },
    }


def _tool_action(iteration: int, ts_start: float, ts_end: float,
                 tool_name: str = "bash") -> dict[str, Any]:
    return {
        "type": "action",
        "action_type": "tool_exec",
        "action_id": f"tool_{iteration}_{tool_name}",
        "agent_id": "a1",
        "iteration": iteration,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "data": {
            "tool_name": tool_name,
            "duration_ms": (ts_end - ts_start) * 1000,
            "success": True,
        },
    }


def _event(category: str, name: str, iteration: int, ts: float) -> dict[str, Any]:
    return {
        "type": "event",
        "agent_id": "a1",
        "category": category,
        "event": name,
        "iteration": iteration,
        "ts": ts,
        "data": {},
    }


# ── Parity tests ───────────────────────────────────────────────────


@pytest.mark.skipif(not _has_node(), reason=_SKIP_REASON)
def test_parity_minimal_trace(tmp_path: Path) -> None:
    """One llm_call + one tool_exec → JS and Python produce the same payload."""
    records = [
        _llm_action(0, 1000.0, 1001.0, content="Let me list files"),
        _tool_action(0, 1001.0, 1001.1, "bash"),
    ]
    trace_path = _write_trace(tmp_path, records)

    py_payload = _normalize(
        build_gantt_payload_multi([("t", TraceData.load(trace_path))])
    )
    js_payload = _run_js_builder(trace_path.read_text(), "t")

    assert js_payload == py_payload


@pytest.mark.skipif(not _has_node(), reason=_SKIP_REASON)
def test_parity_silent_tool_call(tmp_path: Path) -> None:
    """LLM with content=None + tool_calls → both produce tool_calls_requested."""
    records = [
        _llm_action(0, 1000.0, 1001.0, content=None,
                    tool_call=("write_file", {"path": "src/main.ts", "content": "x"})),
        _tool_action(0, 1001.0, 1001.1, "write_file"),
    ]
    trace_path = _write_trace(tmp_path, records)

    py_payload = _normalize(
        build_gantt_payload_multi([("t", TraceData.load(trace_path))])
    )
    js_payload = _run_js_builder(trace_path.read_text(), "t")

    assert js_payload == py_payload
    # Sanity: both paths extracted the path primary field
    llm_spans = [
        s for s in js_payload["traces"][0]["lanes"][0]["spans"]
        if s["type"] == "llm"
    ]
    assert llm_spans[0]["detail"]["tool_calls_requested"] == [
        'write_file(path="src/main.ts")'
    ]


@pytest.mark.skipif(not _has_node(), reason=_SKIP_REASON)
def test_parity_scheduling_spans_with_events(tmp_path: Path) -> None:
    """Event-gated scheduling span rendering must match exactly."""
    records = [
        _llm_action(0, 1000.0, 1001.0, content="thinking"),
        _tool_action(0, 1001.0, 1001.1, "bash"),
        # Gap contains a SCHEDULING event → scheduling span must appear
        _event("SCHEDULING", "message_dispatch", 1, 1001.15),
        _llm_action(1, 1001.3, 1002.0, content="next"),
        _tool_action(1, 1002.0, 1002.1, "bash"),
        # Gap with NO event → no scheduling span
        _llm_action(2, 1002.5, 1003.0, content="final"),
    ]
    trace_path = _write_trace(tmp_path, records)

    py_payload = _normalize(
        build_gantt_payload_multi([("t", TraceData.load(trace_path))])
    )
    js_payload = _run_js_builder(trace_path.read_text(), "t")

    assert js_payload == py_payload

    # Sanity: exactly one scheduling span, tied to message_dispatch
    sched = [
        s for s in js_payload["traces"][0]["lanes"][0]["spans"]
        if s["type"] == "scheduling"
    ]
    assert len(sched) == 1
    assert sched[0]["detail"]["events"] == ["message_dispatch"]


@pytest.mark.skipif(not _has_node(), reason=_SKIP_REASON)
def test_parity_markers_from_events(tmp_path: Path) -> None:
    """SCHEDULING/SESSION/CONTEXT events become markers in both builders."""
    records = [
        _event("SCHEDULING", "message_dispatch", 0, 999.9),
        _event("SESSION", "session_load", 0, 999.95),
        _llm_action(0, 1000.0, 1001.0, content="ok"),
        _tool_action(0, 1001.0, 1001.1, "bash"),
    ]
    trace_path = _write_trace(tmp_path, records)

    py_payload = _normalize(
        build_gantt_payload_multi([("t", TraceData.load(trace_path))])
    )
    js_payload = _run_js_builder(trace_path.read_text(), "t")

    assert js_payload == py_payload
    markers = js_payload["traces"][0]["lanes"][0]["markers"]
    assert {m["event"] for m in markers} == {"message_dispatch", "session_load"}


@pytest.mark.skipif(not _has_node(), reason=_SKIP_REASON)
def test_parity_malformed_line_skipped(tmp_path: Path) -> None:
    """Both builders skip malformed JSONL lines without crashing."""
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"type": "trace_metadata", "scaffold": "s",
                    "trace_format_version": 4}) + "\n"
        + "NOT VALID JSON {{\n"
        + json.dumps(_llm_action(0, 1.0, 2.0, content="hi")) + "\n"
        + json.dumps(_tool_action(0, 2.0, 2.1)) + "\n"
    )

    py_payload = _normalize(
        build_gantt_payload_multi([("t", TraceData.load(trace_path))])
    )
    js_payload = _run_js_builder(trace_path.read_text(), "t")
    assert js_payload == py_payload
