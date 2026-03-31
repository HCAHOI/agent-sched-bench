# REPLAY-1 Review Record

Date: 2026-04-01
Reviewer: fresh-context sub-agent `019d44b5-e4c3-7c81-9ccc-5385b5c7948f`
Scope: `src/harness/trace_replayer.py` and `tests/test_trace_replayer.py`

## Initial Findings

1. Original inter-program arrival offsets were not preserved.
2. Tool-gap waiting incorrectly applied after the terminal step.
3. Replay tests did not meaningfully assert ordering or waiting semantics.

## Fixes Applied

1. Program replay now delays first requests by their original `ts_start` offset
   from the global minimum.
2. Tool-gap sleeps are now applied only between replayed steps for the same
   program.
3. Replay tests now assert both intra-program tool-gap waiting and inter-program
   offset preservation with stable thresholds.

## Verification

- `python3 -m pytest tests/test_trace_replayer.py`
- `python3 -m pytest tests/test_bootstrap.py tests/test_env1.py tests/test_env2.py tests/test_env3a.py tests/test_env3b.py tests/test_env3c.py tests/test_env4.py tests/test_env5.py tests/test_agent_basic.py tests/test_code_agent.py tests/test_data_agent.py tests/test_research_agent.py tests/test_harness_n2.py tests/test_metrics_collection.py tests/test_trace_logger.py tests/test_analysis.py tests/test_trace_replayer.py -q`
- `python3 -m compileall src tests`

## Residual Caveat

Final acceptance still requires exercising replay against a live
OpenAI-compatible serving stack under real load.

## Final Verdict

Approved. No remaining material code issues.
