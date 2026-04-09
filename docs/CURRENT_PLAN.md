# Cloud OpenClaw Serial Prefetch Smoke

## Summary

- Implement bounded serial image lifecycle for `swe-rebench` + `openclaw`.
- Preserve canonical v5 trace output and the normal `attempt_1/` layout.
- Keep the CLI contract stable; behavior change is internal to collection.

## Work Items

1. Update the serial collector path to:
   - preserve the exact `--instance-ids` order
   - prefetch the next source image while the current task runs
   - wait for the prefetch before the next task starts
   - clean up the completed task's fixed image and old source image after artifacts are written
2. Add image lifecycle helpers in `src/harness/container_image_prep.py`.
3. Extend focused tests for ordering, prefetch, cleanup, and cache eviction.
4. Run targeted tests.
5. Run an independent review sub-agent and fix findings.
6. Stop at the checkpoint before the real 3-task smoke run unless the user explicitly requests autopilot.

## Acceptance Checks

- Tasks execute serially.
- At most the current task image/fixed image plus the next prefetched source image are retained.
- Canonical `attempt_1/trace.jsonl` files still start with `trace_metadata` and `trace_format_version=5`.
- No raw OpenClaw session trace is mistaken for the canonical trace output.
