# SWE-ReBench 222-trace restart contract (2026-07-28)
- Goal: preserve the original seed-42/skip-428 zero-overlap 223-task cohort and accept 222 real traces; Gradio is declared attrition and must never run or be replaced.
- Fixed semantics: Codex `gpt-5.6-sol`, OpenClaw, max-iterations 100, concurrency 2, canonical eBPF/resource telemetry, and run `traces/swe-rebench/gpt-5.6-sol/seed42-skip428-n223-c2-ebpf`.
- Baseline: retry fix `778c479`, telemetry retention fix `46c9131`, restart lock `90d7995`; preserve all current artifacts and derive completion only from coherent terminal manifests.
- Blocking bug: after a collector becomes disabled, its live poller can still append raw events while later calls return before pruning, reopening session-long memory growth; fix this once in the shared lifecycle path with the smallest regression test.
- Gate: run relevant focused tests, full suite, Ruff, one bounded material review, and a real eBPF memory smoke proving artifact semantics and bounded post-call RSS; commit the exact fix only.
- Make Phase 3 launch fail closed unless per-task image cleanup is enabled, Gradio is excluded, the exact 222 target IDs are used, and the canonical resource profile/run path are present.
- Do not start or resume collection; stop with a concise evidence-backed GO/NO-GO report. Never commit traces/logs or fabricate terminal state.
