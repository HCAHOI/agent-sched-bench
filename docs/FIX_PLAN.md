# Fix Plan — Review Issues Resolution (v2 — Post-Consensus)

**Created**: 2026-04-01  
**Source**: `docs/REVIEW_2026-04-01.md` (43 issues: 7 Critical, 16 Major, 20 Minor)  
**Status**: APPROVED — Planner + Architect + Critic consensus reached  
**Revision**: v2 — incorporates Architect ITERATE feedback (5 changes) + Critic ITERATE feedback (6 changes)

---

## RALPLAN-DR Summary

### Principles (5)

1. **Integration over isolation** — The root cause is disconnected modules. Fixes must wire components together, not just patch individual files.
2. **Unblock experiments first** — Prioritize fixes that unblock an end-to-end sweep run (C-1 through C-5), defer polish.
3. **Preserve existing tests** — All fixes must pass the existing test suite. New integration code gets new tests.
4. **Minimal surface area** — Each fix touches only the files needed. No refactoring beyond what the issue requires.
5. **Research integrity** — Security (SQL injection), correctness (empty choices, timeout), and measurement fidelity (no ping contamination) fixes cannot be deferred.

### Decision Drivers (Top 3)

1. **Can we run `python -m src.harness.sweep` end-to-end?** — This is the single gate. Every critical fix targets a different failure point on this path.
2. **Will the collected data answer the research question?** — Metrics integration (C-1), system configs (C-3), and histogram parsing (M-2) determine whether results are publishable.
3. **How much code needs to change?** — Smaller diffs are safer. Wave 0 is config/data only. Wave 1 is minimal integration wiring.

### Viable Options

**Option A: Wave-based incremental (Chosen)**
- 4 waves, each independently testable, each unlocking the next
- Pros: Each wave can be committed + reviewed independently per CLAUDE.md review gate
- Cons: ~4 commit cycles; some issues deferred to Wave 3

**Option B: Single big-bang fix — REJECTED**
- Reason: CLAUDE.md mandates review gates at major milestones. A 40-file diff cannot be meaningfully reviewed.

**Option C: Fix only Critical — REJECTED**
- Reason: Running experiments without histogram metrics (M-2) and proper timeout (H-1) would produce incomplete/hung results, wasting GPU time.

### ADR (Architecture Decision Record)

| Aspect | Decision | Rationale |
|:---|:---|:---|
| Metrics lifecycle | **Sweep-owned** (not Runner) | Architect: Runner is a reusable execution primitive; metrics is an orchestration concern. Sweep has `system_config` context with `metrics_url`. TraceLogger stays Runner-owned (needs `_run_single_task` access). |
| Preserve mechanism | **Defer and document** (option c) | Critic: ping-based keep-alive contaminates measurements, violating research integrity. "Vanilla vLLM has no public API for KV cache pinning" is itself a publishable finding. Investigate vLLM session hints as stretch goal. |
| `_validate_snapshot` strictness | **Relax to warning in Wave 1** | Both reviewers: strict `raise ValueError` in metrics collector will crash the polling loop. Change to `logging.warning` first, tighten after histogram parser lands in Wave 2. |

---

## Execution Plan

### Wave 0: Config & Data Prerequisites
**Goal**: Fix all non-code issues so the test suite passes and sweep can find its inputs.  
**Estimated files changed**: 7-10 (including new data files)  
**Risk**: Low (config and data only)  
**Rollback**: `git revert` the wave commit

| Task | Issue | File(s) | Change |
|:---|:---|:---|:---|
| 0.1 | C-5 | `pyproject.toml` | Add `matplotlib>=3.8,<4.0` to `[project.dependencies]` |
| 0.2 | C-3 | `configs/sweep.yaml` | Add `vllm-preserve` and `vllm-no-preempt` to matrix systems. Add comment: "These are vLLM variants using the same binary, different launch configs." |
| 0.3 | C-4 | `data/*/tasks.json` | Create minimal smoke task files per schema below |
| 0.4 | N-4 | `configs/systems/vllm_preserve.yaml` | Add `health:` section matching `vllm_baseline.yaml` |

**Task 0.3 — Smoke Task Schemas** (per Architect + Critic feedback):

```
# data/swebench_lite/tasks.json — CodeAgent schema
[{"instance_id": "smoke__test-001",
  "problem_statement": "Fix the off-by-one error in counter.py",
  "repo_path": "NOT_AVAILABLE",
  "test_cmd": "python -m pytest tests/"}]

# data/bird_sql/tasks.json — DataAgent schema
# ALSO create data/bird_sql/smoke.db with a simple table
[{"task_id": "smoke_sql_001",
  "question": "How many rows are in the employees table?",
  "db_path": "data/bird_sql/smoke.db",
  "gold_sql": "SELECT COUNT(*) FROM employees",
  "evidence": ""}]

# data/research_queries/tasks.json — ResearchAgent schema
[{"task_id": "smoke_research_001",
  "question": "What is the capital of France?"}]
```

A real SQLite `smoke.db` must be created with at least one table (`employees`) so DataAgent can actually execute `schema_inspect` and `sql_execute`.

**Acceptance**:
- `make test` passes (matplotlib import resolved)
- `make sync` installs new matplotlib dependency
- `python -m src.harness.sweep --dry-run --output-root /tmp/test --model test` generates manifest without errors
- `data/bird_sql/smoke.db` is a valid SQLite database: `sqlite3 data/bird_sql/smoke.db ".tables"` returns `employees`

---

### Wave 1: Core Integration + Security
**Goal**: Wire Sweep to MetricsCollector, wire Runner to TraceLogger. Fix security and crash bugs. Relax metrics validation.  
**Estimated files changed**: 5-6  
**Risk**: Medium — changes to sweep's async flow and runner's task lifecycle  
**Rollback**: `git revert` the wave commit; Wave 0 continues to function independently

| Task | Issue | File(s) | Change |
|:---|:---|:---|:---|
| 1.0 | (Architect/Critic) | `src/harness/metrics.py` | Relax `_validate_snapshot()`: change `raise ValueError` to `logging.warning` + continue. This prevents the metrics polling loop from crashing if any metric is missing or renamed across vLLM versions. Will be tightened in Wave 2 after histogram parser lands. |
| 1.1 | C-1 | `src/harness/sweep.py` | In `execute_sweep()`, create `VLLMMetricsCollector` per run cell (using `system_config["metrics_url"]`). Start `collector.poll()` as concurrent `asyncio.Task` before `runner.run()`. Cancel and save metrics after run completes. Use `try/finally` for cleanup. |
| 1.2 | C-2 | `src/harness/runner.py` | Add optional `trace_logger: TraceLogger | None` param to `BenchmarkRunner`. In `_run_single_task()`, call `trace_logger.log_step(agent_id, record)` after each step via a callback. Call `log_summary()` on completion. |
| 1.3 | C-2 | `src/harness/sweep.py` | Instantiate `TraceLogger` per run cell, pass to `BenchmarkRunner`. |
| 1.4 | A-1 | `src/agents/data_agent.py` | Validate `table_name` with `re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', table_name)` before PRAGMA interpolation. Raise `ValueError` on invalid names. |
| 1.5 | A-2 | `src/agents/base.py` | Add `if not response.choices: raise RuntimeError(f"LLM returned empty choices: {response.model_dump()}")` before `choices[0]` access. |
| 1.6 | H-1 | `src/harness/sweep.py` | In `execute_sweep()`, extract `task_timeout_s` from `workload_config` and pass to `BenchmarkRunner(task_timeout_s=...)`. Note: `extract_agent_kwargs()` already extracts it for agent-internal use; this is the *harness-level* timeout wrapping `asyncio.wait_for`. |

**Design note — Metrics vs TraceLogger ownership** (per Architect consensus):
- **Metrics**: Sweep-owned. `execute_sweep()` creates, starts, cancels, and saves the collector. Runner is unaware.
- **TraceLogger**: Runner-owned. `_run_single_task()` has access to agent internals needed for `log_step()`. Sweep instantiates it but passes it to Runner.

**Acceptance** (testable without live infrastructure):
- **Unit test** (new `tests/test_sweep_integration.py`): Mock HTTP server for metrics endpoint + mock agent factory. Verify that after `execute_sweep()`, both JSONL trace file AND metrics JSON file exist in the output directory.
- **Unit test**: `data_agent._schema_inspect_sync("x); DROP TABLE y")` raises `ValueError`.
- **Unit test**: Mock `AsyncOpenAI` returning empty `choices=[]`. Verify `_call_llm()` raises `RuntimeError`.
- **Unit test**: Sweep with `task_timeout_s=0.001` and a slow mock agent → verify `summary["timed_out"] == True`.
- **Manual (cloud)**: N=2 smoke run against live vLLM → verify JSONL + metrics output files.

Note: `VLLMMetricsCollector` accepts `gpu_sample_provider` parameter. In tests, inject a mock returning `[]` to avoid `nvidia-smi` dependency. On cloud, the real `sample_nvidia_smi()` is used.

---

### Wave 2: Behavior Gaps + Metrics Parser
**Goal**: Complete metrics parsing. Fix replay timing. Document preserve findings. Address trace parsing safety.  
**Estimated files changed**: 6-8  
**Risk**: Medium — histogram parsing is non-trivial; `no-preempt` needs careful config

| Task | Issue | File(s) | Change |
|:---|:---|:---|:---|
| 2.1 | M-1 | `configs/systems/vllm_preserve.yaml`, docs | **Defer implementation, document finding.** Remove `lifecycle_mode` and `preserve_strategy` fields from config (they are dead code). Add `notes:` section documenting: "Vanilla vLLM (v0.8.x) has no public API for session-level KV cache pinning. This config runs identically to vllm-baseline. The difference between preserve and baseline is itself a research finding: any preserve behavior requires external mechanisms (Continuum TTL, ThunderAgent pause)." Investigate vLLM `extra_body` session hints as stretch goal — if found, implement and re-add config fields. |
| 2.2 | M-5 | `configs/systems/vllm_no_preempt.yaml` | Rename config to `vllm_low_preempt.yaml`. Change `max_num_seqs` from 1 to 32. Update `name:` to `vllm-low-preempt`. Update `sweep.yaml` reference. Add notes explaining this reduces but does not eliminate preemption; true disabling requires vLLM source patch. |
| 2.3 | M-2 | `src/harness/metrics.py` | Add `vllm:e2e_request_latency_seconds` and `vllm:time_to_first_token_seconds` to `METRICS_OF_INTEREST`. Implement `_parse_prometheus_histogram()` to collect `_sum`, `_count` for histogram metrics. Re-tighten `_validate_snapshot` to warn on missing gauge metrics and raise only if ALL metrics are missing. |
| 2.4 | H-2 | `src/harness/trace_replayer.py` | Fix timing drift: compute `delay_s = first_offset_s - (time.monotonic() - replay_zero)`. Also fix `program_id` placement: move from top-level JSON body to `extra_body={"program_id": ...}` matching `base.py:91`. |
| 2.5 | A-3 | `src/agents/research_agent.py` | Rename `page_read` → `read_page` to match spec. Add `synthesize` as explicit completion tool. Update tool set, system prompt, and tests. |
| 2.6 | N-2 | `src/harness/metrics.py` | Fix nvidia-smi to use `--format=csv,noheader,nounits`. Add `try/except` around `subprocess.check_output` returning empty list on failure. |
| 2.7 | S-1 (promoted) | `src/analysis/parse_traces.py` | Add guard: `if "type" not in frame.columns: raise ValueError("Trace DataFrame missing 'type' column; was it produced by TraceLogger?")`. |

**Acceptance**:
- **Unit test**: `_parse_prometheus()` correctly extracts `_sum` and `_count` from a sample histogram payload string containing `vllm:e2e_request_latency_seconds_sum`, `_count`, and `_bucket` lines.
- **Unit test**: TraceReplayer timing test with 3 groups at offsets [0s, 1s, 2s] → verify actual delays are within 50ms tolerance of target offsets.
- **Unit test**: `parse_traces.summarize_trace_frame()` raises `ValueError` on DataFrame without `type` column.
- `vllm_preserve.yaml` notes section documents the research finding.
- `vllm-low-preempt` config has `max_num_seqs: 32`.

---

### Wave 3: Analysis & Polish
**Goal**: Complete the analysis pipeline. Fix remaining minor issues.  
**Estimated files changed**: 6-8  
**Risk**: Low  
**Rollback**: `git revert`

| Task | Issue | File(s) | Change |
|:---|:---|:---|:---|
| 3.1 | M-3 | `src/analysis/plots.py` | Add `plot_throughput_comparison(frames: dict[str, DataFrame])` for multi-system overlay. Add `--plot-type` CLI argument. Update `main()` to dispatch to all plot functions. |
| 3.2 | M-7 | `src/analysis/parse_traces.py` | Add `merge_metrics()` function to join metrics snapshots with trace data. Add GPU util and KV cache stats to `summarize_trace_frame()` when metrics data is available. |
| 3.3 | A-4 | `src/agents/code_agent.py`, `data_agent.py`, `research_agent.py` | Add configurable `max_tool_output_chars` (default 8000). Truncate tool output before appending to messages, preserving head + tail with `[... truncated N chars ...]`. |
| 3.4 | N-1 | `pyproject.toml`, `src/harness/sweep.py` | Add `tqdm` dependency. Add progress bar to `execute_sweep()`. |
| 3.5 | N-5 | `Makefile` | Add `build` target aliasing `sync`. |
| 3.6 | M-4 | `configs/systems/continuum.yaml`, `thunderagent.yaml` | Document that refs must be pinned before experiments. Add `scripts/check_refs.sh` that fails if any config contains `REPLACE_WITH_COMMIT_OR_TAG`. |

**Acceptance**:
- `python -m src.analysis.plots --plot-type throughput_comparison sample.csv --output /tmp/test.png` generates multi-system figure.
- All tests pass including `test_analysis.py`.
- `make build` works.
- `scripts/check_refs.sh` exits 1 while placeholders remain.

---

## Deferred (Not in scope)

| Issue | Reason |
|:---|:---|
| M-6 (previous review misleading) | Informational — resolved by this review replacing it |
| S-2 (unified launcher dispatch) | Nice-to-have; `sweep.py` works via per-system config lookup |
| S-3 (unused cpu_offload_gib) | Continuum integration is Phase 2+ |
| Continuum launcher missing parity params | Phase 2+ (when Continuum is actually deployed) |
| Various LOW/Nit issues | Cost/benefit doesn't justify separate fixes |

---

## Verification Strategy

**After each wave:**
1. `make test` — full test suite must pass
2. `make lint` — no new ruff violations
3. Spawn independent reviewer sub-agent with checklist:
   - [ ] Code correctness: does the logic do what the task description says?
   - [ ] Research integrity: any measurement contamination? hindsight leakage?
   - [ ] Edge cases: what happens on empty input, timeout, crash?
   - [ ] Consistency: does it match existing conventions?

**After Wave 1 specifically:**
- Dry-run sweep with smoke data: `python -m src.harness.sweep --dry-run --output-root /tmp/sweep_test --model test`
- Verify manifest JSON lists all 5 systems × 3 workloads × 7 concurrency levels = 105 cells

**End-to-end verification after all waves:**
- Full `make test && make lint`
- `python -m src.harness.sweep --dry-run` with all 5 system configs
- Verify all output file formats: JSONL traces, metrics JSON, sweep manifest

---

## Consensus Trail

| Round | Role | Verdict | Key Feedback |
|:---|:---|:---|:---|
| 1 | Planner | DRAFT | Initial 4-wave plan |
| 1 | Architect | ITERATE | (1) Relax `_validate_snapshot`, (2) Sweep owns metrics, (3) Redesign preserve, (4) Specify schemas, (5) Promote S-1 |
| 1 | Critic | ITERATE | Agreed with Architect on all 5. Added: (6) Fix acceptance criteria testability, (7) Note `trace_replayer` program_id bug, (8) Default preserve to option (c) |
| 2 | Planner | REVISED | Incorporated all 8 changes into v2 |

---

*Plan status: APPROVED (consensus reached) — Ready for execution*
