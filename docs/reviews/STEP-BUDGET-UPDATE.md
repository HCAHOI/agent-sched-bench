# Step Budget Update Review Record

Date: 2026-03-31
Reviewer: fresh-context sub-agent `019d4337-4444-71f3-9a2c-fb096f427dad`
Scope: primary workload step budgets, aligned spec snippets, and bootstrap tests

## Change Summary

- Doubled the primary workload `max_steps` values:
  - `code_agent`: `20 -> 40`
  - `data_agent`: `10 -> 20`
  - `research_agent`: `15 -> 30`
- Updated the mirrored examples in `docs/agent_benchmark_spec.md`.
- Added bootstrap assertions for the expected step budgets.

## Verification

- `python3 -m pytest tests/test_bootstrap.py`
- `uv run --directory /Users/chiyuh/Workspace/agent-sched-bench pytest -q tests/test_bootstrap.py`

## Final Verdict

Approved. No actionable findings.
