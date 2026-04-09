# OpenClaw Runtime Consolidation

## Goal

- Remove the legacy OpenClaw host-controller container-backed path.
- Make all supported SWE OpenClaw runs use `task_container_agent`.
- Keep the earlier `/testbed` workspace-prompt fix intact while simplifying the
  runtime surface before paying for a new Haiku trace.

## Findings

1. The real Kinto divergence came from the prompt-visible workspace, not the
   task repo's `python3`/`pytest` environment.
2. That prompt bug is already fixed and pushed.
3. The remaining architectural confusion comes from two OpenClaw SWE paths:
   `task_container_agent` (real SWE-rebench path) and a legacy host-controller
   container-backed path.
4. The legacy path is no longer needed for current supported SWE OpenClaw
   benchmarks if `swe-bench-verified` is migrated to `task_container_agent`.

## Work Items

1. Done: move `swe-bench-verified` OpenClaw onto `task_container_agent`.
2. Done: remove the host-controller container-backed branch from
   `collect_openclaw_traces()`.
3. Done: delete the dead `container_workspace` / `container_backend`
   OpenClaw code path and simplify runner/session/loop tool registration.
4. Done: update tests so they cover the single remaining OpenClaw SWE runtime.
5. Done: run the focused validation suite.
6. Done: complete a strict independent review pass before finalizing.
7. In progress: commit and push the cleanup.
8. Pending: rerun the single-task Haiku collection for
   `Kinto__kinto-http.py-384`.

## Acceptance Checks

- Supported SWE OpenClaw benchmarks resolve to `task_container_agent`.
- `collect_openclaw_traces()` has no host-controller container fallback.
- OpenClaw no longer ships unused `container_backend` / `container_workspace`
  code for SWE runs.
- Tests cover the migrated runtime selection and the simplified runner/loop
  behavior.
- The cleanup is reviewed, committed, and pushed before any paid rerun.

## Validation

- `.venv/bin/pytest -q tests/test_openclaw_eval_runner.py
  tests/test_openclaw_loop_tools.py tests/test_openclaw_runtime_selection.py
  tests/test_collector_task_container_runtime.py
  tests/test_task_container_entrypoint.py tests/test_collector_openclaw_metadata.py
  tests/test_collector_runtime_mode.py tests/test_swe_rebench_plugin.py`
  -> `30 passed`
- `.venv/bin/pytest -q tests/test_task_container_runtime.py
  tests/test_attempt_pipeline.py` -> `9 passed`
