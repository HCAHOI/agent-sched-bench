# Trace Collect Cleanup Plan

## Goal

Execute the approved review findings across `src/trace_collect`, related agent
packages and directly affected tests. Delete dead or duplicated
schema ballast, fix broken dispatch and trace-shape reads, and keep the trace
artifacts internally consistent.

## Change Groups

1. Scaffold naming and trace field normalization
   Status: completed
   Scope:
   - Rename the mini SWE scaffold key from `mini-swe-agent` to `miniswe`
     everywhere in trace collection, simulation, CLI, registry, and tests.
   - Make inspector/analysis read v5 action fields from `data`.
   - Standardize tool success on `data["success"]`.
   Verification:
   - `python3 -m pytest tests/test_scaffold_registry.py tests/test_trace_inspector.py tests/test_simulator.py tests/test_simulator_miniswe_regression.py tests/test_collect_traces_kwarg_passthrough.py tests/test_cli_mcp_flag_enforcement.py tests/test_benchmark_protocol.py`

2. OpenClaw replay tool execution cleanup
   Status: completed
   Scope:
   - Replace reimplemented filesystem operations in
     `src/trace_collect/openclaw_tools.py` with the existing OpenClaw tool
     layer or its shared backend primitives.
   - Make unknown/unsupported tool names fail loudly instead of returning fake
     success.
   Verification:
   - `python3 -m pytest tests/test_openclaw_tool_runtime.py tests/test_simulator.py tests/test_claude_code_import.py`

3. Attempt artifact/schema pruning
   Status: completed
   Scope:
   - Remove dead stderr/pull-time fields and redundant model/result/resource
     payload duplication.
   - Update layout/docstrings/comments to match the actual artifact set.
   - Keep `results.json`, `resources.json`, `tool_calls.json`, and
     `container_stdout.txt` as the canonical split outputs.
   Verification:
   - `python3 -m pytest tests/test_attempt_pipeline.py tests/test_attempt_layout.py tests/test_collector_openclaw_metadata.py`

4. Claude import cleanup
   Status: completed
   Scope:
   - Stop describing Anthropic import as OpenAI-schema compatibility glue.
   - Remove isolated legacy status fallback in tool success resolution.
   Verification:
   - `python3 -m pytest tests/test_claude_code_import.py tests/test_trace_inspector.py tests/test_openclaw_raw_response.py demo/gantt_viewer/tests/test_payload.py`

5. Full targeted regression
   Status: completed
   Verification:
   - `python3 -m pytest tests/test_attempt_pipeline.py tests/test_attempt_layout.py tests/test_collector_openclaw_metadata.py tests/test_scaffold_registry.py tests/test_trace_inspector.py tests/test_simulator.py tests/test_simulator_miniswe_regression.py tests/test_collect_traces_kwarg_passthrough.py tests/test_cli_mcp_flag_enforcement.py tests/test_benchmark_protocol.py tests/test_openclaw_tool_runtime.py tests/test_openclaw_raw_response.py tests/test_claude_code_import.py tests/test_openclaw_simulate_adapter.py demo/gantt_viewer/tests/test_payload.py`

## Notes

- Use aggressive deletion where the review found dead or redundant code.
- Fix callers after each group instead of reverting cleanup.
- No TODOs or compatibility aliases unless required to preserve current
  behavior during the same patch series.
