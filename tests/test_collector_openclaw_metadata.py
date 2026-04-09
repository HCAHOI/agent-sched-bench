"""Phase 4 regression: collector writes trace_format_version + benchmark fields."""
from __future__ import annotations

import json
from pathlib import Path

from agents.benchmarks import get_benchmark_class
from agents.benchmarks.base import BenchmarkConfig


def _make_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        slug="swe-bench-verified",
        display_name="SWE-Bench Verified",
        harness_dataset="princeton-nlp/SWE-bench_Verified",
        harness_split="test",
        data_root=Path("data/swebench_verified"),
        repos_root=Path("data/swebench_repos"),
        trace_root=Path("traces/swebench_verified"),
        default_max_iterations=50,
        selection_n=32,
        selection_seed=42,
    )


def test_normalize_openclaw_trace_writes_v5_with_benchmark(tmp_path: Path) -> None:
    """_normalize_openclaw_trace must stamp trace_format_version=5,
    benchmark, benchmark_split on the destination file."""
    from trace_collect.collector import _normalize_openclaw_trace

    # Craft a minimal source trace (post-session_runner output): v5 metadata
    # from the in-process openclaw writer, plus one action.
    src = tmp_path / "src.jsonl"
    src.write_text(
        json.dumps({
            "type": "trace_metadata", "scaffold": "openclaw",
            "trace_format_version": 5, "model": "test/model",
            "instance_id": "test-1",
        }) + "\n"
        + json.dumps({
            "type": "action", "action_type": "llm_call",
            "action_id": "llm_0", "agent_id": "test-1",
            "iteration": 0, "ts_start": 1.0, "ts_end": 2.0, "data": {},
        }) + "\n"
    )

    dst = tmp_path / "dst.jsonl"
    plugin = get_benchmark_class("swe-bench-verified")(_make_config())
    _normalize_openclaw_trace(
        src=src, dst=dst,
        benchmark=plugin,
        model="test/model", api_base="https://x.y",
        max_iterations=50, instance_id="test-1",
    )

    lines = dst.read_text(encoding="utf-8").strip().splitlines()
    metadata = json.loads(lines[0])
    assert metadata["type"] == "trace_metadata"
    assert metadata["trace_format_version"] == 5
    assert metadata["scaffold"] == "openclaw"
    assert metadata["benchmark"] == "swe-bench-verified"
    assert metadata["benchmark_split"] == "test"
    assert metadata["instance_id"] == "test-1"
    # The source metadata was merged (not dropped); the action survives.
    assert any(json.loads(line).get("type") == "action" for line in lines[1:])


def test_normalize_openclaw_trace_preserves_runner_scaffold_capabilities(
    tmp_path: Path,
) -> None:
    """_normalize_openclaw_trace must not overwrite the runner's scaffold_capabilities."""
    # reviewer M2: BFCL runner writes tools='benchmark_provided'/file_ops='none';
    # merging with default-openclaw capabilities would silently corrupt the trace
    from trace_collect.collector import _normalize_openclaw_trace

    src = tmp_path / "src.jsonl"
    src.write_text(
        json.dumps({
            "type": "trace_metadata", "scaffold": "openclaw",
            "trace_format_version": 5, "model": "test/model",
            "scaffold_capabilities": {
                "tools": "benchmark_provided",
                "memory": False,
                "skills": False,
                "file_ops": "none",
            },
        }) + "\n"
        + json.dumps({
            "type": "action", "action_type": "llm_call",
            "action_id": "llm_0", "agent_id": "test-1",
            "iteration": 0, "ts_start": 1.0, "ts_end": 2.0, "data": {},
        }) + "\n"
    )

    dst = tmp_path / "dst.jsonl"
    plugin = get_benchmark_class("swe-bench-verified")(_make_config())
    _normalize_openclaw_trace(
        src=src, dst=dst, benchmark=plugin,
        model="test/model", api_base="https://x.y",
        max_iterations=50, instance_id="test-1",
    )

    metadata = json.loads(dst.read_text(encoding="utf-8").splitlines()[0])
    caps = metadata["scaffold_capabilities"]
    # Runner's values survive (no overwrite by collector defaults).
    assert caps["tools"] == "benchmark_provided"
    assert caps["memory"] is False
    assert caps["skills"] is False
    assert caps["file_ops"] == "none"


def test_normalize_openclaw_trace_conservative_default_when_no_source_metadata(
    tmp_path: Path,
) -> None:
    """When the source trace has no trace_metadata, the collector stamps a
    conservative 'unknown' marker rather than inventing a capability list."""
    # reviewer M2: previously hardcoded the openclaw bash+file+web tool list,
    # silently corrupting BFCL crash-recovery traces
    from trace_collect.collector import _normalize_openclaw_trace

    # Partial trace: one action but no metadata header.
    src = tmp_path / "partial.jsonl"
    src.write_text(
        json.dumps({
            "type": "action", "action_type": "llm_call",
            "action_id": "llm_0", "agent_id": "test-1",
            "iteration": 0, "ts_start": 1.0, "ts_end": 2.0, "data": {},
        }) + "\n"
    )

    dst = tmp_path / "dst.jsonl"
    plugin = get_benchmark_class("swe-bench-verified")(_make_config())
    _normalize_openclaw_trace(
        src=src, dst=dst, benchmark=plugin,
        model="test/model", api_base="https://x.y",
        max_iterations=50, instance_id="test-1",
    )

    metadata = json.loads(dst.read_text(encoding="utf-8").splitlines()[0])
    caps = metadata["scaffold_capabilities"]
    # Conservative default — NOT the legacy openclaw bash+file+web list.
    assert caps.get("unknown") is True
    assert "reason" in caps
    # And critically: no "bash" in the default tool list lie.
    assert "bash" not in str(caps)
