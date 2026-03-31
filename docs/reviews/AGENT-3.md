# AGENT-3 Review Record

Date: 2026-03-31
Reviewer: fresh-context sub-agent `019d43fa-a2ab-7883-8c65-e5a1653e7b4e`
Scope: `src/agents/data_agent.py`, `tests/test_data_agent.py`, and the
`smoke-data` Make target

## Initial Findings

1. SQL execution was mutating the benchmark database.
2. Final-answer scoring compared raw JSON strings instead of denotation.
3. The smoke path did not cover the real `sql_execute` / retry loop.
4. Final SQL timing was missing from the trace.

## Fixes Applied

1. `DataAgent` now opens SQLite in read-only mode and rejects non-read SQL.
2. Final SQL evaluation now compares normalized row data instead of raw JSON,
   including unordered `NULL`-containing results.
3. The local smoke path now includes a failing `sql_execute`, a retry, and a
   final SQL step.
4. `final_sql` trace records now include `tool_duration_ms` and evaluation mode.

## Verification

- `make smoke-data`
- `python3 -m pytest tests/test_data_agent.py`
- `python3 -m pytest tests/test_bootstrap.py tests/test_env1.py tests/test_env2.py tests/test_env3a.py tests/test_env3b.py tests/test_env3c.py tests/test_env4.py tests/test_env5.py tests/test_agent_basic.py tests/test_code_agent.py tests/test_data_agent.py -q`
- `python3 -m compileall src tests`

## Residual Caveat

Full runtime fidelity still depends on real BIRD databases and a live
OpenAI-compatible backend.

## Final Verdict

Approved. No remaining material code issues.
