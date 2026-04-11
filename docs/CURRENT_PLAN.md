# Explicit `--container` For Benchmark Collection

## Goal
Make benchmark collection choose the container executable explicitly at the
CLI boundary and propagate that choice through the full benchmark collection
runtime path. Eliminate implicit `podman` defaults from the main collection
chain and remove MiniSWE's env-var-based runtime selection.

## Non-goals
- Do not add `--container` to `agents.openclaw._cli.py`.
- Do not require `--container` for `simulate` or `import-claude-code`.
- Do not change success semantics, canonical artifact writing, or
  Terminal-Bench non-image flows.

## Execution Plan
1. Inspect and map the current propagation path across `trace_collect.cli`,
   `trace_collect.collector`, `trace_collect.attempt_pipeline`,
   `trace_collect.runtime.task_container`, `harness.container_image_prep`, and
   `agents.miniswe.agent`.
2. Lock behavior with focused tests where coverage is missing:
   `collect` must fail fast without `--container`, and runtime selection must
   be asserted as an explicit value rather than an implicit `podman` default.
3. Update the collection CLI so `collect` requires `--container docker|podman`
   and threads that value into `collect_traces(...)`.
4. Propagate the explicit container executable through collector helpers,
   attempt orchestration, task-container runtime helpers, and image-prep
   helpers. Remove `podman` defaults from the benchmark collection main path.
5. Change MiniSWE to consume only caller-provided container runtime settings.
   Remove `MSWEA_DOCKER_EXECUTABLE` reads and ensure container-backed MiniSWE
   paths require an explicit runtime.
6. Update and extend tests so both `docker` and `podman` paths are covered,
   while non-container entrypoints remain unaffected.
7. Run targeted verification, then a fresh independent review sub-agent before
   finalizing.

## Checkpoints
- After tests are updated, verify the required propagation contract is pinned.
- After implementation, verify both runtime values travel from CLI to the
  actual helper invocations.
- Before finalizing, run an independent code review focused on correctness and
  research-integrity regressions.
