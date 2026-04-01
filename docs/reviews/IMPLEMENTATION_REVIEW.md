# Project Implementation Review

This review re-assesses the project against [agent_benchmark_spec.md](/Users/chiyuh/Workspace/agent-sched-bench/docs/agent_benchmark_spec.md) and [AGENTS.md](/Users/chiyuh/Workspace/agent-sched-bench/AGENTS.md) after the remediation pass.

## Summary

The original review correctly identified several real implementation gaps, but it is now partially stale. The confirmed gaps around sweep automation, Poisson arrivals, GPU metrics integration, automated scheduler-hook activation, and richer analysis have been implemented. Two items were not actual defects by the time of re-check (`DataAgent Isolation`, `CPU Offloading Missing`), and one remains an explicit accepted tradeoff (`Sandbox Shortcut`).

## Status Matrix

| Issue | Current Status | Rationale | Primary Location |
| :--- | :--- | :--- | :--- |
| Missing Sweep Automation | `resolved` | `run_sweep.sh` now calls a real sweep orchestrator that expands `configs/sweep.yaml`, emits a manifest, and runs one benchmark cell per matrix entry. | `src/harness/sweep.py`, `scripts/run_sweep.sh` |
| Inefficiency Detection Gaps | `resolved (heuristic scope)` | The detector now reports heuristic thrashing, bubble, and idle-memory signals and surfaces thresholds explicitly. | `src/analysis/inefficiency_detector.py` |
| Manual vLLM Scheduler Hook | `resolved` | A runtime wrapper now applies a version-checked scheduler monkeypatch and fails closed if the expected symbols are missing. | `src/harness/scheduler_hooks.py`, `src/harness/vllm_entrypoint_with_hooks.py` |
| Incomplete Analysis & Plots | `resolved` | Analysis now includes latency breakdown, prefix-cache-hit plotting, and cliff-point identification. | `src/analysis/plots.py` |
| GPU Metrics Integration | `resolved` | GPU utilization samples are now collected in the same polling loop as vLLM metrics. | `src/harness/metrics.py` |
| Missing Poisson Arrival | `resolved` | `BenchmarkRunner` now supports deterministic seeded Poisson arrivals in addition to `closed_loop`. | `src/harness/runner.py` |
| Sandbox Shortcut | `accepted tradeoff` | The simplified repo-copy sandbox remains intentional for this phase and is not being replaced with container isolation now. | `src/agents/code_agent.py` |
| DataAgent Isolation | `resolved earlier` | The data agent uses read-only SQLite access and rejects mutating SQL, so per-task DB snapshotting is no longer required for isolation. | `src/agents/data_agent.py` |
| CPU Offloading Missing | `resolved earlier` | The Continuum path already exposes CPU-offload controls and related config. | `src/serving/continuum_launcher.py`, `scripts/serve_continuum.sh` |

## Research Integrity Check

- Rule 1 `No Benchmark Gaming`: still passes. No dataset-specific priors were introduced by the remediation work.
- Rule 2 `No Hindsight Contamination`: now improved further because `ResearchAgent` no longer leaks `reference_answer` into the prompt.
- Rule 4 `Real Workloads`: still partial at runtime. The code paths are real, but full acceptance still depends on running against live servers, models, and datasets.
- Rule 5 `Completeness Over Shortcuts`: improved. The major shortcut that remained manual was scheduler instrumentation; that is now automated. The simplified code-agent sandbox remains an explicitly accepted tradeoff.
- Rule 6 `Use Established Tools`: still passes. The implementation continues to rely on `vLLM`, `httpx`, `pandas`, `matplotlib`, `trafilatura`, and SQLite rather than custom replacements.

## Remaining Real Caveats

1. Live experiment acceptance is still outstanding.
   This repo now closes the implementation gaps, but actual benchmark validity still requires live server runs for the environment, serving, metrics, analysis, and replay paths.
2. Analysis and inefficiency signals are still heuristic.
   The new diagnostics are intentionally scoped as heuristics and should not be interpreted as exhaustive causal attribution without corroborating live traces.
3. The code-agent sandbox is still lightweight by design.
   This is acceptable for the current phase, but should be revisited if later experimental scale or safety requirements demand stronger isolation.
