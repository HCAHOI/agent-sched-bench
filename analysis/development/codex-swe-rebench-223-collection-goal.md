# SWE-ReBench 450-total scope contract (2026-07-28)
- Goal: 450 unique resource-valid tasks across the existing 100, existing 277, and this run's required 73.
- Fixed semantics: Codex `gpt-5.6-sol`, OpenClaw, max-iterations 100, concurrency 2, canonical eBPF/resource telemetry, and the existing `seed42-skip428-n223-c2-ebpf` run/profile.
- Allowed selector: direct production seed 42/skip 451/sample 50 only, no instance IDs and no task beyond those exact 50; `TASK_CONTAINER_CLEANUP_IMAGES=1`.
- Evidence boundary: preserve every historical/interrupted attempt without rewriting manifests; acceptance and watchdog accounting both use the shared resource-valid `load_completed_ids` predicate.
- Verified correction: the prelaunch 23 accepted tasks include tail ID `VirtusLab__git-machete-353`, so their union with the exact 50-ID selector is 72, not 73.
- Stop evidence: host `oom_kill` increased 133→134 at 16:17:25 with canonical accepted 49 and cohort accepted 27/50; Pandas/Web3 attempt 3 remain invalid interrupted evidence.
- Current gate: collection, daemons, containers, and selector images are stopped/clean; never resume or schedule a 51st ID without an explicit objective amendment resolving the 72-versus-73 conflict.
