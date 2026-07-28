# SWE-ReBench 222-trace restart contract (2026-07-28)
- Goal: preserve the original seed-42/skip-428 zero-overlap 223-task cohort and accept 222 real traces; Gradio is declared attrition and must never run or be replaced.
- Fixed semantics: Codex `gpt-5.6-sol`, OpenClaw, max-iterations 100, concurrency 2, eBPF/resource telemetry, run `traces/swe-rebench/gpt-5.6-sol/seed42-skip428-n223-c2-ebpf`, and production selector seed 42/skip 451/sample 200.
- Baseline: retry `778c479`, retention `46c9131`, restart lock `90d7995`; preserve interrupted attempts, including Titiler attempt 4 as invalid resource evidence, without rewriting manifests.
- Gate: canonical root `telemetryd` uses `PYTHONPATH=src:/usr/lib/python3/dist-packages` and must pass a real import-and-attach smoke before collection.
- Acceptance: terminal workload state is accepted only when any present resource evidence is telemetry `ok`, collection `valid`, and cleanup `ok`; resume and watchdog both use `load_completed_ids`.
- Authorized resume: direct selector only, no instance IDs, same profile/run, `TASK_CONTAINER_CLEANUP_IMAGES=1`; seed-42 skip-451 sample-200 must equal the 223-tail and remain zero-overlap.
- Stop on host `oom_kill` increase, daemon death (`ps -p` for root telemetryd), systemic auth/provider failure, or resource-integrity failure; record and continue past task-local container OOM.
