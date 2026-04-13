# Next Steps Plan: I/O Profiling + Concurrent Scheduling (Ralplan R3 Consensus)

> Consensus reached: Architect APPROVE + Critic APPROVE (2 iterations)
> Date: 2026-04-13

## Principles
1. Real data, real bottleneck — no synthetic workloads
2. Reproducibility first — eliminate uncontrolled variables
3. Data-driven resource pooling argument
4. Incremental reuse of existing infrastructure
5. Characterization → Intervention — profiling must point to verifiable optimization

## Decision Drivers
1. Advisor's I/O focus: "4万美金的H100都在等磁盘I/O"
2. Concurrent replay reliability is a blocker for downstream experiments
3. Need quantitative "resource variation across agent phases" evidence (FaaSBoard analogy)

---

## TODO 1 [P0] — Extend ContainerStatsSampler for I/O Metrics

**Hypothesis**: Tool execution disk I/O is the dominant system bottleneck, exceeding CPU usage.

**Implementation**:
- **CPU + Memory**: 已有，通过 `podman/docker stats` 采集（保持不变）
- **Disk I/O**: Host-side `/proc/<container-init-pid>/io` (read_bytes, write_bytes) + `/proc/<pid>/status` (context switches)
- **Network I/O**: 扩展现有 `podman/docker stats` format string 增加 `{{.NetIO}}`（rx_bytes, tx_bytes），与 CPU/Memory 采集方式一致，零额外开销
- Runtime-conditional PID resolution: `docker inspect` vs `podman inspect` via `self.executable`
- Readability check at startup: if PermissionError (rootless Podman), fallback to `<executable> exec <cid> cat /proc/1/io`
- `summarize_samples()` adds `disk_read_mb`, `disk_write_mb`, `net_rx_mb`, `net_tx_mb`, `context_switches` (min/max/avg/delta)

**Files**: `src/harness/container_stats_sampler.py`, `src/trace_collect/attempt_pipeline.py`

**Acceptance**: Re-run collect on any trace → resources.json contains `disk_read_mb`, `disk_write_mb`, `net_rx_mb`, `net_tx_mb`, `context_switches` with non-zero values. Works on both Docker and Podman.

**Effort**: 0.5-1 day

---

## TODO 2 [P0] — Fix Concurrent Replay Reliability

**Hypothesis**: Eliminating network variance → tool_time_s CV < 10%.

**Pre-step audit** (30 min):
- Grep 10 traces for pip/git/curl/wget/apt commands, quantify fraction of tool_time

**Decision gate**:
- <20% network-dependent: `--network=none`, filter incomparable actions
- 20-50%: `--network=none` + warmup step (pre-execute critical dependencies)
- >50%: Abandon `--network=none`; use `--network=host` + N≥5 repeats + statistical filtering

**Dual-config**: `--network=host` for characterization (TODO 1,3), `--network=none` for concurrency (TODO 4)

**Files**: `src/trace_collect/simulator.py`, `src/trace_collect/attempt_pipeline.py` (parameterize `--network=host` at L115)

**Acceptance**: 3x replay of 3 traces → tool_time_s CV < 10%

**Effort**: 0.5-1 day

---

## TODO 2.5 [P1] — Resource Utilization Timeline in Gantt Viewer

**Hypothesis**: Interactive visualization of system metrics aligned with agent actions enables faster identification of I/O bottleneck patterns than static figures.

**Motivation**: TODO 3 (phase-resource analysis) produces data that needs visual exploration. The Gantt viewer already renders LLM/tool spans on a shared time axis — adding a resource chart below agent lanes gives researchers an interactive way to correlate agent phases with CPU/memory/disk/network spikes.

**Implementation**:
- **Backend** (`demo/gantt_viewer/backend/`):
  - Add `ResourceSample` Pydantic model to `schema.py` (t, t_abs, cpu_percent, memory_mb, disk_read_mb, disk_write_mb, net_rx_mb, net_tx_mb, context_switches)
  - Add `resource_timeline: list[ResourceSample] | None` to `TracePayload`
  - In `payload.py`: load `resources.json` from the same attempt directory as `trace.jsonl`, align sample timestamps with trace t0, build timeline
- **Frontend** (`demo/gantt_viewer/frontend/src/`):
  - Add `showResourceTimeline` + `resourceMetric` signals to `state/signals.ts`
  - In `canvas/layout.ts`: add `RESOURCE_CHART_H = 80` constant, update total content height
  - In `canvas/CanvasRenderer.ts`: render resource area chart below agent lanes, using the same `timeToX` mapping for alignment
  - In `canvas/hit.ts`: add resource chart hit detection for tooltip
  - In `components/Header.tsx`: add metric selector dropdown (CPU / Memory / Disk IO / Net IO) + visibility toggle

**Files**: `demo/gantt_viewer/backend/schema.py`, `demo/gantt_viewer/backend/payload.py`, `demo/gantt_viewer/frontend/src/canvas/CanvasRenderer.ts`, `demo/gantt_viewer/frontend/src/canvas/layout.ts`, `demo/gantt_viewer/frontend/src/state/signals.ts`, `demo/gantt_viewer/frontend/src/components/Header.tsx`

**Depends on**: TODO 1 (resources.json must contain I/O fields)

**Acceptance**:
- Gantt viewer displays resource utilization chart below agent lanes
- Chart is time-aligned with spans (same zoom/scroll)
- User can switch between CPU%, memory, disk I/O, network I/O metrics
- Hovering shows resource values at the cursor time in tooltip
- Existing Gantt functionality unchanged (no regression)

**Effort**: 1.5-2 days

---

## TODO 3 [P1] — Agent Phase × Resource Time-Aligned Analysis

**Hypothesis**: Resource demand varies ≥3x (peak vs trough) across tool_exec vs llm_call phases.

**Implementation**:
- Two-level phase model:
  - Level 1: `tool_exec` vs `llm_call` by `action_type` (always reliable)
  - Level 2: optional fine-grained (needs command string parsing)
- Analysis path: direct from raw trace actions + resources.json (NOT through Gantt payload)
- Output: timeline chart (x=time, y=CPU%/disk_io_rate/mem, background=phase)
- **Interactive exploration**: use TODO 2.5's Gantt resource timeline to visually validate phase boundaries before committing to automated analysis

**Files**: New `scripts/figures/plot_resource_phase_alignment.py`

**Depends on**: TODO 1, TODO 2.5 (for visual validation)

**Acceptance**: 10-trace charts with reported peak/trough ratios for tool vs LLM phases

**Effort**: 1-2 days

---

## TODO 4 [P1] — Multi-Agent Concurrent Scheduling Experiment

**Hypothesis**: N-agent concurrent completion time shows super-linear slowdown, primarily from I/O contention.

**Scope**: Uses simulator `cloud_model` replay (`asyncio.gather`). LLM calls = timed sleep. **Intentionally isolates tool contention.** Must state this limitation in paper.

**Implementation**:
- Host-level sampling: `/proc/stat` + `/proc/meminfo` + `/proc/diskstats` in separate thread
- Resource limits: N=1 unlimited; N=2 `--cpus=1.5 --memory=3g`; N=3 `--cpus=1.0 --memory=2g`
- **Plumbing**: Add `extra_args` pass-through in `simulator.py:_prepare_container_session` → `start_task_container` (already supports `extra_args` at `attempt_pipeline.py:103`)
- Trace selection: stratified by total_time_s (short/medium/long)
- Statistics: ≥3 repeats per config, mean ± 95% CI

**Files**: `src/trace_collect/simulator.py`, `src/harness/container_stats_sampler.py`

**Depends on**: TODO 1, TODO 2

**Acceptance**: Completion time table (mean ± CI) + host resource timeline + I/O vs CPU attribution

**Effort**: 2-3 days

---

## TODO 5 [P2] — Container Lifecycle Overhead Quantification

**Hypothesis**: Persistent container lifecycle overhead < 5% (vs AgentCGroup ~26%).

**Implementation**:
- Add trace events: `container_pull_start/end`, `container_start/ready`, `container_stop`
- Distinguish cold start vs warm start
- Align methodology with AgentCGroup's actual definition (lookup their paper)

**Files**: `src/trace_collect/attempt_pipeline.py`, `src/harness/trace_logger.py`

**Acceptance**: Traces contain lifecycle events; comparison table with AgentCGroup noting methodology differences

**Effort**: 0.5 day

---

## TODO 6 [P1] — Resource Pooling Prototype (Intervention — Paper Core Contribution)

**Hypothesis**: Phase-aware dynamic resource adjustment reduces N=3 concurrent completion time significantly.

**Acceptance criteria (tiered)**:
- Primary: Statistically significant improvement (p<0.05, paired t-test, ≥3 traces × 3 repeats)
- Stretch: ≥15% improvement
- Minimum publishable: ≥5% with clear I/O attribution
- <5%: Valid negative result ("resource pooling limited benefit at small scale")

**Implementation**:
- Online phase detection in simulator action replay loop (action_type transitions)
- Resource adjustment via `docker/podman update --cpus=X --memory=Y <cid>`
- Safety: memory min 512MB, CPU min 0.25 core, watchdog for sustained saturation
- **Pre-step**: Smoke test `docker/podman update` on target host

**Files**: New `src/harness/dynamic_resource_allocator.py`

**Depends on**: TODO 1, 3, 4

**Effort**: 3-5 days

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| QEMU confound (ARM→x86 emulation, 3-10x CPU overhead) | Generalizability | State as limitation; x86 validation on VastAI |
| 4 vCPU / 8GB constraint | N=3 experiments resource-limited | Strict per-container quotas; N=4 saturation test separate |
| Serial dependency chain (~8-12d critical path) | Timeline risk | TODO 5 parallel with TODO 1; TODO 2 audit during TODO 1 dev |
| cgroup compatibility across runtimes | TODO 6 complexity | Use docker/podman update command, not direct cgroup writes |

## Execution Timeline

```
Week 1:
  Day 1-2: TODO 1 (I/O profiling) ✅ DONE + TODO 2 pre-step audit (parallel)
  Day 2-3: TODO 2 (network fix) + TODO 5 (container lifecycle, parallel)

Week 2:
  Day 4-5: TODO 2.5 (Gantt resource visualization)
  Day 5-6: TODO 3 (phase-resource alignment, use 2.5 for visual validation)
  Day 6-8: TODO 4 (concurrent scheduling experiment)

Week 3:
  Day 9-13: TODO 6 (resource pooling prototype)

Optional: x86 validation on VastAI (1-2 days)
```

## Follow-ups
- Confirm x86 validation environment (VastAI or other)
- Align with advisor on paper submission target (venue + deadline)
- Coordinate with weitian's LLM serving work for complementary profiling data
