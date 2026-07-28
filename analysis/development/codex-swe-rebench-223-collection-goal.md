# SWE-ReBench 450-total scope contract (2026-07-28)
- Goal: 450 unique resource-valid tasks across the existing 100, existing 277, and this run's intended 73.
- Fixed semantics: Codex `gpt-5.6-sol`, OpenClaw, max-iterations 100, concurrency 2, canonical eBPF/resource telemetry, and the existing `seed42-skip428-n223-c2-ebpf` run/profile.
- Allowed selector: direct production seed 42/skip 451/sample 50 only, no instance IDs and no task beyond those exact 50; `TASK_CONTAINER_CLEANUP_IMAGES=1`.
- Evidence boundary: preserve every historical/interrupted attempt without rewriting manifests; acceptance and watchdog accounting both use the shared resource-valid `load_completed_ids` predicate.
- Verified correction: the prelaunch 23 accepted tasks include tail ID `VirtusLab__git-machete-353`, so their union with the exact 50-ID selector is 72, not 73.
- Shutdown/resume: safely finalize the two workers interrupted from sample 200, verify cleanup, then resume only the exact sample-50 selector at c2.
- Stop and clean collection, daemons, and task images at selector exhaustion or canonical accepted 73; never schedule a 51st ID without an explicit objective amendment.
