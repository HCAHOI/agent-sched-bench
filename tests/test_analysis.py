from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.inefficiency_detector import detect_inefficiencies
from analysis.parse_traces import load_trace_jsonl, summarize_trace_frame
from analysis.plots import plot_throughput_vs_concurrency


def write_demo_trace(path: Path) -> None:
    entries = [
        {
            "type": "step",
            "agent_id": "agent-0001",
            "step_idx": 0,
            "phase": "reasoning",
            "program_id": "agent-0001",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "llm_latency_ms": 5.0,
            "ts_start": 1.0,
            "ts_end": 2.0,
            "tool_duration_ms": None,
            "tool_success": None,
        },
        {
            "type": "step",
            "agent_id": "agent-0001",
            "step_idx": 1,
            "phase": "acting",
            "program_id": "agent-0001",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_latency_ms": 0.0,
            "ts_start": 2.0,
            "ts_end": 4.0,
            "tool_duration_ms": 1500.0,
            "tool_success": False,
        },
        {
            "type": "summary",
            "agent_id": "agent-0001",
            "task_id": "t-1",
            "total_llm_ms": 5.0,
            "total_tool_ms": 1500.0,
            "success": False,
        },
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


def test_parse_traces_and_detect_inefficiency(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    write_demo_trace(trace_path)
    frame = load_trace_jsonl(trace_path)
    summary = summarize_trace_frame(frame)
    ineff = detect_inefficiencies(frame)
    assert summary["n_steps"] == 2
    assert summary["avg_jct_s"] == 3.0
    assert ineff["heuristic_long_tool_wait_count"] == 1
    assert ineff["heuristic_failed_tool_count"] == 1


def test_plot_throughput_vs_concurrency(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"concurrency": 1, "throughput_steps_per_min": 10.0},
            {"concurrency": 2, "throughput_steps_per_min": 18.0},
        ]
    )
    output = tmp_path / "plot.png"
    plot_throughput_vs_concurrency(frame, output)
    assert output.exists()
