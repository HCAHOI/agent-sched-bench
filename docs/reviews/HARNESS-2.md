# HARNESS-2 Review Record

Date: 2026-03-31
Reviewer: fresh-context sub-agent `019d4418-e24b-7c41-b2d4-e9257a088339`
Scope: `src/harness/metrics.py` and `tests/test_metrics_collection.py`

## Initial Findings

1. Incomplete Prometheus payloads were accepted as valid samples.
2. The `nvidia-smi` parser/test contract was weaker than realistic output.
3. Collector state persisted across repeated `poll()` calls.

## Fixes Applied

1. `VLLMMetricsCollector` now validates complete snapshots before appending.
2. `parse_nvidia_smi_csv()` now handles header/unit-bearing output.
3. `poll()` resets `self.snapshots` at the start of each collection cycle.
4. Tests now cover incomplete-snapshot failure, realistic `nvidia-smi` output,
   and state-reset semantics.

## Verification

- `python3 -m pytest tests/test_metrics_collection.py`
- `python3 -m pytest tests/test_bootstrap.py tests/test_env1.py tests/test_env2.py tests/test_env3a.py tests/test_env3b.py tests/test_env3c.py tests/test_env4.py tests/test_env5.py tests/test_agent_basic.py tests/test_code_agent.py tests/test_data_agent.py tests/test_research_agent.py tests/test_harness_n2.py tests/test_metrics_collection.py -q`
- `python3 -m compileall src tests`

## Residual Caveat

Final acceptance still depends on a live vLLM metrics endpoint and real GPU
sampling during an actual benchmark run.

## Final Verdict

Approved. No remaining material code issues.
