# `agent-test-bench` Gap Analysis for `dev/cpu-only-integrate`

Date: 2026-07-01  
Current repo: `/Users/chiyuh/Workspace/agent-sched-bench-cpu-only`  
Comparison repo: `/Users/chiyuh/Workspace/agent-test-bench`  
Current branch: `dev/cpu-only-integrate`

## Scope

This document summarizes functionality and mechanism differences between the current CPU-only/cloud-provider branch and the sibling `agent-test-bench` repository.

Explicitly excluded from this analysis:

- Legacy textual trace inspection / `trace_collect inspect`.
- GPU support as a target capability.

This exclusion does **not** automatically exclude every file that historically lived near GPU/local-serving code. When a removed feature has non-GPU relevance, it is called out separately and marked with a migration recommendation.

## Current Branch Constraints

The current branch has several deliberate constraints that should be preserved unless explicitly changed:

1. Cloud-provider-only LLM execution.
   - Local/private OpenAI-compatible API bases should remain rejected.
2. Benchmark support should go through the benchmark plugin layer:
   - `src/agents/benchmarks/`
   - `configs/benchmarks/<slug>.yaml`
3. No per-benchmark CLI flags for dataset-specific behavior.
4. Prefer Gantt viewer workflows over textual trace inspection.
5. Preserve checkpoint/restore timing semantics:
   - checkpoint/restore overhead must be recorded and excluded from workload timing.
6. Preserve resource-integrated timeout semantics:
   - applies to a single `exec.command`, not arbitrary multi-command tool calls.

## Executive Summary

`agent-test-bench` has broader benchmark/scaffold coverage, richer collection controls, and several operator conveniences. The current repo has stronger CPU-only replay fidelity, checkpoint forced-sync support, stricter cloud-provider security boundaries, and more hermetic task-container bootstrap behavior.

Recommended migration priorities:

### P0: High-value, required for near-term scale

- [x] Collect CLI `--concurrency`. Completed in this integration pass; applies at benchmark task level for both SWE-style task-container runs and Terminal-Bench host-controller runs.
- [x] Collect CLI `--skip`. Completed in `7b77929`.
- [x] Resume semantics for `max_iterations` / exhausted attempts. Completed in `8866c39` and tightened in `953dc3f`; resume now keys only on `status=completed` and `status=exhausted`.
- [x] DeepSeek cloud provider support. Completed in `0072fe7`.
- [x] Operator documentation for collect/simulate/Gantt/benchmark plugins. Completed in `OPERATIONS.md`.
- [x] High-concurrency simulate workers: add `simulate --workers` so large replay runs are split across independent OS processes/event loops instead of one congested asyncio loop. Implemented in this pass.
- [x] Simulate warm-up control: add `--prep-concurrency` and a global all-ready replay barrier so container preparation does not pollute replay timing. Implemented in this pass.
- [x] Simulate sleep-drift instrumentation: worker-local event loops and shared replay start time are in place; expected-vs-actual sleep drift metrics are recorded on action `sim_metrics` and per-task summaries. Implemented in this pass.

### P1: Valuable, requires design review

- BFCL benchmark plugin.
- BrowseComp / DeepResearchBench plugins.
- Tongyi DeepResearch scaffold, only if it can satisfy current cloud-provider and tracing constraints.
- [x] Monitoring policy layer focused on high-concurrency simulate safety: PMU is explicit and rejected when requested for concurrent replay; memory-bandwidth is explicit and safely disabled in concurrent replay; neither replaces action-level resource timelines. Implemented in this pass for simulate.
- Attempt timing breakdown and monitoring-disabled markers.

### P2: Optional / situational

- Claude Code import.
- Standalone HTML visualization.
- Pure sweep orchestration, separated from local serving/vLLM dependencies.

### Do not migrate by default

- Local/private OpenAI-compatible endpoints.
- `local_model` simulation mode.
- local HF/vLLM/self-hosted serving stack.
- conda/bootstrap setup flow.
- Terminal-Bench asciinema recording.
- Terminal-Bench progress watchdog.
- OpenClaw long-term memory prompt injection.

---

## Feature Surface Differences

### 1. Benchmark Plugins and Configs

#### Present in `agent-test-bench`

The sibling repo registers additional benchmark families beyond SWE/Terminal-Bench:

- BFCL / multi-turn function calling.
- BrowseComp.
- DeepResearchBench.
- Related configs under `configs/benchmarks/`, including examples like:
  - `bfcl-multi-turn-base.yaml`
  - `browsecomp.yaml`
  - `deep-research-bench.yaml`

#### Current repo

Current benchmark registry is intentionally narrower:

- `swe-bench-verified`
- `swe-rebench`
- `terminal-bench`

Current relevant files:

- `src/agents/benchmarks/__init__.py`
- `src/agents/benchmarks/swe_bench_verified.py`
- `src/agents/benchmarks/swe_rebench.py`
- `src/agents/benchmarks/terminal_bench.py`

#### Mechanism differences

`agent-test-bench` has a broader benchmark base layer:

- Unknown YAML keys can be folded into `extras`.
- There is a host-runner protocol concept for non-container benchmarks.
- There is a visualization filename customization hook.

Current repo has newer CPU-only branch-specific behavior:

- Opt-in local JSON task cache via `AGENT_SCHED_BENCH_USE_LOCAL_TASK_CACHE=1`.
- Provenance stamping for locally cached benchmark rows.
- Randomized seeded sampling by default.

#### Recommendation

Migrate benchmark plugins only through the existing plugin/YAML architecture. Do not add new benchmark-specific CLI flags.

Suggested order:

1. BFCL plugin first, because it expands evaluation modality without necessarily requiring local serving.
2. BrowseComp / DeepResearchBench next, after confirming provider and tracing compatibility.
3. Any host-mode runner interface should be designed as a benchmark plugin capability, not as ad-hoc collector logic.

---

### 2. Tongyi DeepResearch Scaffold

#### Present in `agent-test-bench`

Sibling supports a second scaffold:

- CLI choice: `--scaffold tongyi-deepresearch`.
- Code under `src/agents/tongyi_deepresearch/`.
- Runner patches vendored tools and LLM tracing.

#### Current repo

Current collector supports only:

- `--scaffold openclaw`

There is no `src/agents/tongyi_deepresearch/` package.

#### Mechanism differences

Tongyi scaffold is not just a benchmark plugin; it is a separate agent/scaffold execution path with its own tracing and tool adapter assumptions.

#### Recommendation

Do not directly copy it. First answer:

1. Can Tongyi run with cloud-provider-only API bases?
2. Can it preserve full intermediate trace outputs?
3. Can its tool calls map cleanly into current trace schema and Gantt viewer payloads?
4. Can it run without local/private endpoint assumptions?

If yes, migrate as a separate scaffold behind explicit config and tests.

---

### 3. Collect CLI Controls

#### Present in `agent-test-bench`

Sibling collector has richer CLI controls:

- `--concurrency`
- `--skip`
- monitoring controls
- internal recording controls

#### Current repo

Current collect CLI is simpler and mostly serial/basic.

Relevant current file:

- `src/trace_collect/cli.py`

#### Mechanism differences

Sibling can:

- Run multiple attempts/tasks concurrently.
- Skip leading tasks for manual sharding/resume workflows.
- Toggle monitoring behavior.
- Route OpenClaw through internal recording backends.

Current branch emphasizes:

- CPU-only trace fidelity.
- forced checkpoint sync.
- cloud provider constraints.
- more controlled task-container runtime behavior.

#### Recommendation

Migrate separately:

1. `--skip`: low-risk operator convenience.
2. `--concurrency`: useful but needs careful interaction with:
   - provider rate limits,
   - task-container naming,
   - Docker/Podman resource contention,
   - output directory collisions,
   - checkpoint artifact paths.
3. Monitoring controls: fold into a current-compatible monitoring policy.
4. Internal recording controls: do not migrate unless the recording backend is cloud-compatible and does not reintroduce local HF serving.

---

### 4. Monitoring Policy, PMU, and Micro-Architecture Telemetry

#### Present in `agent-test-bench`

Sibling has a policy layer for collect-time telemetry:

- `src/trace_collect/monitoring.py`
- PMU / ksys / micro-architecture collectors
- tri-state policy style such as `auto | on | off`

#### Current repo

Current repo has strong action-level resource timeline support:

- `src/trace_collect/resource_timeline.py`
- replay consumption in `src/trace_collect/openclaw_tools.py`
- cgroup CPU reading
- proc-net byte reading
- resource-integrated timeout for single exec commands

Current sampler supports broad CPU/memory/disk/network style summaries, but not the same PMU policy path.

#### Mechanism differences

Sibling telemetry is broader for CPU systems analysis. Current telemetry is more directly tied to replay causality and action-level timeout behavior.

#### Recommendation

Do not replace current resource timeline design. Instead, migrate the policy layer as an additive control surface:

- Keep current `resource_timeline.py` as the replay-facing source of truth.
- Add policy config for enabling/disabling expensive telemetry.
- Add PMU/micro-arch fields only when available; absence should be explicit in metadata.
- Ensure no synthetic or fallback metrics are introduced.

---

### 5. Simulation and Replay Modes

#### Present in `agent-test-bench`

Sibling supports more modes:

- `local_model`
- `cloud_model`
- host-mode replay via a `HostAgent`
- tool profiling modes

#### Current repo

Current simulator supports cloud-model replay and has newer CPU-only replay fidelity features:

- checkpoint forced-sync
- resource-timeline-aware execution
- source artifact remapping
- manifest entries with source task metadata
- replay task stats and LLM timing config

Current relevant file:

- `src/trace_collect/simulator.py`

#### Mechanism differences

Sibling is broader for experimentation with local model endpoints and host-mode runs. Current is stronger for faithful replay of collected traces under cloud-provider and resource-timeline constraints.

#### Recommendation

- Do not migrate `local_model` mode under current branch constraints.
- Consider host-mode replay only if needed by BFCL/DeepResearch-style benchmarks.
- Consider tool profiling if it can be represented without changing replay semantics.
- Preserve forced-sync and resource-integrated timeout behavior.

---

### 6. Trace Data and Visualization

#### Present in `agent-test-bench`

Sibling has:

- Claude Code import path.
- Standalone HTML trace visualization.
- Gantt viewer backend coupled to broader trace-inspection layer.

#### Current repo

Current repo has:

- strict canonical trace loader:
  - `src/trace_collect/trace_data.py`
- Gantt viewer using canonical trace data:
  - `demo/gantt_viewer/`
- no textual inspect CLI.

#### Mechanism differences

Current visualization path is intentionally simpler and stricter. Sibling has broader import/inspection compatibility.

#### Recommendation

- Keep Gantt as the primary viewer.
- Do not reintroduce textual inspect.
- Claude Code import is optional if external trace ingestion becomes necessary.
- Standalone HTML visualization is optional, but should not duplicate Gantt maintenance burden unless there is a clear offline-sharing use case.

---

### 7. Task Container Runtime

#### Current repo

Current task-container runtime now uses a guarded shared bootstrap cache:

- immutable shared runtime dependency generations under `~/.cache/task-container-bootstrap/<platform>/<config-hash>/`
- host file lock around bootstrap writes
- cache marker validation for requirements, Python runtime, pip index, pip-resolution env fingerprint, architecture, image platform, Python ABI/OS/libc fingerprint, and installed package/version manifests
- non-destructive rebuilds so active cache readers are not invalidated
- explicit in-container Python probing
- support for `/opt/miniconda3` and `/opt/conda` Python candidates
- explicit pip mirror support via `TASK_CONTAINER_PIP_INDEX_URL`
- isolated pip env using `PIP_CONFIG_FILE=os.devnull`
- apt mirror setup for Debian/Ubuntu task containers
- run-mode task-container stdout is streamed live while still teeing to raw log artifacts

Relevant files:

- `src/trace_collect/runtime/task_container.py`
- `src/trace_collect/attempt_pipeline.py`
- `src/trace_collect/collector.py`

#### Present in `agent-test-bench`

Sibling uses a similar shared bootstrap cache:

- shared runtime directory
- file lock around bootstrap
- cache contamination detection
- live stdout streaming for run-mode
- older host conda fallback assumptions

#### Mechanism differences

Current now matches the sibling performance-oriented path while preserving this branch's stricter runtime assumptions. The cache is shared across attempts but rejects stale or contaminated contents before reuse.

#### Recommendation

Status: aligned for shared bootstrap cache and live run stdout streaming. Keep the current explicit in-container Python probing and do not reintroduce host conda assumptions.

---

### 8. OpenClaw Terminal-Bench Adapter

#### Current repo

Current adapter is stricter:

- rejects local/private API bases
- uses FIFO secret handoff instead of exporting API key in normal env
- writes prompt to a file and invokes `--prompt-file`
- cleans FIFO on timeout

Relevant file:

- `src/agents/terminal_bench/openclaw_agent.py`

#### Present in `agent-test-bench`

Sibling adapter supports:

- local API gateway rewriting inside container
- direct API key export in setup env
- inline prompt passing via CLI args

#### Recommendation

Keep current implementation. It is better aligned with cloud-provider-only runs and has a smaller accidental exposure surface.

---

### 9. Provider Config

#### Present in `agent-test-bench`

Sibling supports DeepSeek as a provider.

#### Current repo

Current provider config rejects local/private API bases and now includes DeepSeek as a cloud OpenAI-compatible provider.

Relevant files:

- `src/llm_call/config.py`
- `src/llm_call/providers.py`

#### Status

Done in `0072fe7`: DeepSeek was added with `api_base=https://api.deepseek.com` and `DEEPSEEK_API_KEY`, while preserving local/private API-base rejection.

---

### 10. Docs, Smoke, and Experiment Scripts

#### Present in `agent-test-bench`

Sibling has broader docs and scripts:

- benchmark docs
- trace-collect docs
- sweep scripts
- serving/profiling scripts

#### Current repo

Current repo has fewer operator docs, but has a new forced-sync smoke helper:

- `scripts/smoke_checkpoint_forced_sync.py`

#### Recommendation

- Add docs for current branch behavior.
- Consider pure orchestration sweep scripts only if they do not depend on local serving.
- Do not migrate serving/profiling scripts wholesale.

---

## Active Removals / Disabled Functionality in Current History

This section lists functionality that was actively removed or disabled in current repo history, excluding trace inspection and GPU support.

### 1. Local/self-hosted model backends, local HF/vLLM serving, KV and sparse-attention experiments

Commit:

- `9e25d60` — `[refactor] Remove GPU/local backends`

Removed or disabled paths included:

- `src/serving/`
- `configs/kv_policies/`
- `configs/sparse_attention/`
- KV eviction tests
- sparse attention tests
- HF session cache tests
- recording E2E tests

Sibling still has related paths such as:

- `src/serving/kv_policies/base.py`
- `src/serving/recording/backend_hf.py`
- `src/serving/sparse_attention/block_topk.py`

Recommendation:

- Do not migrate by default.
- If any non-local-serving analysis remains useful, extract it into standalone offline analysis scripts without reintroducing local serving runtime.

### 2. Local/private OpenAI-compatible API bases rejected

Commit:

- `9e25d60` — `[refactor] Remove GPU/local backends`

Relevant current files:

- `src/llm_call/config.py`
- `src/llm_call/openclaw.py`
- `README.md`

Recommendation:

- Keep this rejection. It is a current branch invariant.

### 3. Claude Code trace import

Commit:

- `009a0e2` — `[refactor] Remove Claude Code support`

Removed paths included:

- `src/trace_collect/claude_code_import.py`
- `tests/fixtures/claude_code_minimal.jsonl`
- `tests/test_claude_code_import.py`
- CLI wiring in `src/trace_collect/cli.py`

Sibling still has:

- `src/trace_collect/claude_code_import.py`

Recommendation:

- Optional. Migrate only if external Claude Code JSONL import is needed.

### 4. Deep-research benchmark family and Tongyi scaffold

Commit:

- `83ca7cc` — `[refactor] Remove deep-research benchmark family and tongyi scaffold`

Removed paths included:

- `configs/benchmarks/browsecomp.yaml`
- `configs/benchmarks/deep-research-bench.yaml`
- `src/agents/benchmarks/_research.py`
- `src/agents/benchmarks/browsecomp.py`
- `src/agents/benchmarks/deep_research_bench.py`
- `src/agents/tongyi_deepresearch/`
- related tests

Recommendation:

- Worth evaluating for migration if benchmark parity is desired.
- Must go through plugin/YAML architecture.
- Tongyi scaffold needs separate compatibility review.

### 5. Sweep harness and legacy serving launchers

Commit:

- `9ea7c35` — `[refactor] Trim dead code, retire sweep harness, dedup config overlay`

Removed paths included:

- `src/harness/sweep.py`
- `scripts/run_sweep.sh`
- `configs/sweeps/default.yaml`
- legacy system configs
- serving launchers/checkers

Recommendation:

- Pure sweep orchestration may be worth migrating.
- Serving launchers should not be migrated.

### 6. Modal workspace analysis scripts

Commit:

- `dbfa7dc` — `[refactor] Remove modal workspace scripts`

Removed paths included:

- `scripts/modal_workspace/agent_attention_followup.py`
- `scripts/modal_workspace/kv_evict_100_run_analysis.py`
- `scripts/modal_workspace/h2o_causal_failure_analysis.py`
- plotting scripts

Recommendation:

- Do not migrate wholesale.
- Cherry-pick only if a specific analysis is still scientifically relevant.

### 7. Conda/setup bootstrap flow

Commit:

- `c85b471` — `[refactor] Drop conda; task containers bootstrap their own Python`

Removed paths included:

- `scripts/setup/bootstrap.sh`
- `scripts/setup/download_model.sh`
- `scripts/setup/install_deps.sh`

Recommendation:

- Do not migrate. Current repo uses `uv` and in-container task bootstrap by design.

### 8. Terminal-Bench asciinema recording disabled

Commit:

- `2353a02` — `[fix] Disable TB asciinema recording`

Current behavior disables asciinema in Terminal-Bench runtime task YAML.

Recommendation:

- Do not migrate back unless terminal video artifacts become a strict requirement.

### 9. Terminal-Bench progress watchdog removed

Commit:

- `0c6361c` — `refactor: remove progress_watchdog (redundant with tb's global timeout)`

Recommendation:

- Do not migrate back unless there is evidence Terminal-Bench global timeout is insufficient.

### 10. OpenClaw memory injection into prompt removed

Commit:

- `760ed15` — `[refactor] Remove unused memory from OpenClaw system prompt`

Removed behavior:

- long-term memory text injected into OpenClaw system prompt
- agent state workspace line

Sibling still has memory context injection in `src/agents/openclaw/_context.py`.

Recommendation:

- Do not migrate unless persistent memory is part of the experimental design.
- Reintroducing it changes agent behavior and may complicate reproducibility.

---

## Proposed Migration Plan

### Phase 0: Documentation and invariants

Status: roadmap-level invariants/checklist complete in `1f4a692`; full operator docs remain tracked under Phase 1.

- [x] Document current branch invariants:
  - cloud-provider-only
  - benchmark plugin/YAML architecture
  - no local/private API base
  - Gantt-first trace visualization
  - checkpoint forced-sync semantics
- [x] Add a migration checklist for any feature copied from `agent-test-bench`.

Success criteria:

- [x] Docs clearly state what is intentionally not supported.
- [x] Future diffs can be reviewed against these invariants.

### Phase 1: Low-risk operator improvements

Candidates:

- [x] collect `--skip` — completed in `7b77929`.
- [x] collect `--concurrency` — completed in this integration pass with task-level semantics shared by SWE-style and Terminal-Bench collection.
- [x] exhausted/max-iterations resume semantics — completed in `8866c39`; legacy `status=error` compatibility intentionally removed in `953dc3f`.
- [x] DeepSeek provider — completed in `0072fe7`.
- [x] trace collect/simulate docs — completed in `OPERATIONS.md`.

Risks:

- collect concurrency can still introduce Docker/Podman resource pressure and provider rate-limit effects; default remains `--concurrency 1`.
- exhausted semantics may change retry behavior.

Required tests:

- [x] unit tests for CLI parsing and collector scheduling (`--skip`, non-negative sample/skip).
- [x] resume tests for `completed` and `exhausted`; `error` manifests remain non-terminal.
- [x] provider config tests preserving local/private API rejection.

### Phase 2: Benchmark expansion

Candidates:

- BFCL
- BrowseComp
- DeepResearchBench

Rules:

- all benchmark-specific behavior belongs in plugin/YAML
- no new per-benchmark collector flags
- preserve task metadata and provenance
- no oracle or hindsight leakage

Required tests:

- plugin normalization tests
- local task cache interaction tests if supported
- smoke-level dry load tests

### Phase 3: High-concurrency simulate replay

Candidates:

- `simulate --workers` multi-process replay for 100s-1000s of concurrent agents
- `--prep-concurrency` as a system-wide container-preparation throttle
- global all-ready barrier and shared replay time-zero across workers
- per-worker/per-agent trace output to avoid multi-process JSONL write contention
- sleep-drift metadata for LLM/source-gap replay sleeps — completed in this pass

Rules:

- preserve current cloud-provider-only replay and source-trace timing semantics
- do not reintroduce local model/vLLM simulation
- do not let container preparation time count as replay workload time
- keep resource-integrated timeout behavior unchanged for single `exec.command`
- default `--workers 1` must preserve the current single-process path

Required tests:

- CLI validation for `--workers` and `--prep-concurrency`
- partitioning/order tests for worker chunks
- replay compatibility tests for single-worker/default mode
- focused multi-worker smoke with mocked container/tool execution where possible

Review audit:

- 2026-07-02: Fresh strict reviewer checked the worker implementation. Initial critical finding: shared replay time-zero was recorded but not used for scheduling. Fixed by adding a small cross-process replay-start grace window and making worker sessions sleep until shared monotonic zero before replay. Initial major findings about append-mode trace collisions and missing worker-mode global resource status were fixed by unlinking trace files before writing, adding millisecond run IDs, and recording global container resources as disabled in worker mode. Re-review found no critical/major issues; minor effective prep-concurrency metadata was added.
- 2026-07-02: A follow-up strict review after sleep-drift instrumentation found major issues in worker-mode per-task trace metadata and wave scheduling documentation. Fixed by splitting per-task traces from the combined global trace, returning task output dirs from workers, testing per-task metadata, and documenting all-ready wave semantics. Final re-review found no critical/major issues.

### Phase 4: Telemetry policy expansion

Candidates:

- monitoring policy layer — completed for simulate high-concurrency safety in this pass
- PMU / micro-architecture availability reporting
- monitoring-disabled marker — completed for attempt/simulate `resources.json` summaries (`disabled`, `enabled_no_samples`, `collected`)
- richer timing breakdown

Rules:

- do not replace action-level resource timelines
- missing PMU must be explicitly recorded, not silently faked
- PMU must be rejected when explicitly requested for concurrent replay; host memory-bandwidth must be safely disabled in concurrent replay unless an isolation-safe design is documented
- replay semantics must remain stable

Required tests:

- policy resolution tests — completed for simulate policy resolver
- telemetry availability tests — completed for memory-bandwidth disable paths
- replay compatibility tests — completed for simulate focused suite

Review audit:

- 2026-07-02: `openai-codex/gpt-5.5` strict reviewer checked monitoring-policy and worker-fix changes. Initial critical finding: worker preparation failure could leak already prepared sessions. Fixed by gathering preparation results with `return_exceptions=True`, retaining successful `PreparedTraceSession`s, aborting the barrier, and finalizing successes in `finally`; regression test added. Major findings about combined trace ordering and worker metadata were fixed by globally sorting combined worker records and logging global `concurrency` plus `worker_chunk_size`. Minor findings about disabled reason spelling and malformed timestamp sorting were fixed. Final focused tests passed.

### Phase 5: VM runtime feasibility

Candidates:

- Firecracker/KVM capability probe — started on `dev/vm`
- runtime-driver boundary for simulator container lifecycle
- same-architecture aarch64 Firecracker smoke
- Docker image to rootfs conversion for same-architecture images
- snapshot/restore timing comparison against Docker prebuild/start

Rules:

- Firecracker is only a performance candidate for same-architecture guests;
  x86_64 SWE-rebench images on aarch64 hosts must report
  `guest_arch_mismatch` rather than pretending to use KVM acceleration.
- VM work must not alter Docker collect/simulate semantics until an explicit
  runtime driver is selected and validated.
- Snapshot/checkpoint metrics must record kernel/rootfs/snapshot provenance.

Remote probe notes:

- 2026-07-02 on `chiyuh@193.124.7.2`: host is aarch64 openEuler 24.03,
  Firecracker v1.15.1 is installed, `/opt/ear/firecracker/vmlinux` and
  `rootfs.ext4` exist, and direct `/dev/kvm` access works after adding the user
  to group `kvm`. A read-only aarch64 rootfs boots under Firecracker. Current
  SWE-rebench task images are `amd64`, so they remain qemu-binfmt workloads on
  this ARM host unless same-architecture images are introduced.

### Phase 6: Optional imports/visualization/orchestration

Candidates:

- Claude Code import
- standalone HTML visualization
- pure sweep orchestration

Rules:

- no textual trace inspect reintroduction
- no local serving dependency
- no duplicated source of truth for trace parsing

---

## Review Checklist for Future Migration PRs

Before merging any feature from `agent-test-bench`, verify:

- [ ] It does not re-enable local/private API bases.
- [ ] It does not introduce local HF/vLLM/self-hosted model dependencies.
- [ ] It does not add benchmark-specific collector CLI flags.
- [ ] It preserves full trace metadata and intermediate outputs.
- [ ] It preserves checkpoint/restore timing accounting.
- [ ] It does not weaken current resource timeline replay semantics.
- [ ] It records provenance for any non-canonical task source.
- [ ] It has unit tests for plugin normalization/config/CLI behavior.
- [ ] It has smoke validation if it touches collection, replay, or evaluation.
- [ ] It has a fresh review before experiments produce analysis results.

## Open Questions

1. Resolved: `max_iterations` is recorded as `status=exhausted` and skipped on resume. Only top-level `status=completed` and `status=exhausted` are resume-terminal; legacy `status=error` manifests are intentionally not treated as terminal.
2. Should BFCL/BrowseComp/DeepResearchBench be part of the CPU-only branch target scope?
3. Do we need host-mode replay for non-container benchmarks, or should every benchmark be adapted into current container/replay abstractions?
4. Is PMU/micro-architecture telemetry required for near-term experiments, or is action-level CPU/network resource timeline sufficient?
5. Is Claude Code trace import still useful, given the Gantt-first direction?
6. Should sweep orchestration be rebuilt around current cloud-provider runs instead of migrated from sibling?
