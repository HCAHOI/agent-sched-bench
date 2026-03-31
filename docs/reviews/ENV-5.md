# ENV-5 Review Record

Date: 2026-03-31
Reviewer: fresh-context sub-agent `019d43b2-d8b7-7d11-a45b-9de875afbddf`
Scope: preemption-oriented vLLM configs, raw vLLM launcher flags, scheduler
hook parsing/reporting, and `tests/test_env5.py`

## Initial Findings

1. The metrics collector did not fail closed or distinguish current-run deltas
   from cumulative counters.
2. The vLLM log was append-only, so old eviction events could pollute the
   current run.
3. The report schema did not clearly distinguish static support from runtime
   evidence.
4. The launcher did not actually invoke preemption-report generation.

## Fixes Applied

1. `scheduler_hooks.py` now calls `response.raise_for_status()` and records
   baseline/delta status fields.
2. `serve_vllm.sh` now truncates the log before launch so the parsed log is
   current-run scoped.
3. The report schema now includes `metrics_fetch_ok`,
   `scheduler_log_provided`, `scheduler_hook_runtime_confirmed`, and
   `evidence_scope`.
4. `serve_vllm.sh` now explicitly calls `python -m harness.scheduler_hooks`
   after the readiness check to write `VLLM_PREEMPTION_REPORT_PATH`.

## Verification

- `make verify-env5`
- `python3 -m pytest tests/test_bootstrap.py tests/test_env1.py tests/test_env2.py tests/test_env3a.py tests/test_env3b.py tests/test_env3c.py tests/test_env4.py tests/test_env5.py -q`
- `python3 -m compileall src scripts tests`

## Residual Caveat

The code checkpoint is approved, but true runtime confirmation still requires a
live vLLM run with scheduler instrumentation and real preemption pressure.

## Final Verdict

Approved. No remaining material code issues.
