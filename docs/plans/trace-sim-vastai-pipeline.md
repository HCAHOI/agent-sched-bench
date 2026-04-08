# RALPLAN-DR: Trace→Simulate→Visualize Pipeline on vast.ai — REVISION 3 (APPROVED)

**Branch target:** `dev/trace-sim-vastai-pipeline` (fork from current `dev/trace-sim-vastai`)
**Scope:** Track 1 (pipeline completeness) + Track 2 (vast.ai container runtime) + Track 3 (OpenClaw MCP/skills realism)
**Mode:** DELIBERATE
**Status:** REVISION 3 — Critic APPROVE 2026-04-08 (iteration 2). Consensus reached via ralplan workflow: Planner R0 → Architect (SOUND WITH AMENDMENTS) → Planner R1 → Critic (ITERATE) → Planner R2 → Architect re-review (SOUND WITH NOTES) → Planner R3 → Critic APPROVE.
**Author:** Planner agent (ralplan consensus mode)
**Date:** 2026-04-08

---

## 0. CHANGELOG

### R0 → R1 (historical)

Six Architect amendments applied; mode promoted SHORT → DELIBERATE.

- **A1** — MCP default policy hardened. `--mcp-config` mandatory for openclaw; opt-out is the affirmative literal `none`. `configs/mcp/empty.yaml` deleted from plan.
- **A2** — Phase 2 metric model corrected to absolute `PreemptionSnapshot` snapshots stored field-for-field under `data.sim_metrics.vllm_scheduler_snapshot`. Deltas moved to a new analysis-time `src/analysis/sim_metrics_delta.py`.
- **A3** — Phase 1 rescoped to mini-swe-only; **Phase 1.5** added for openclaw simulator. The R0 60-LOC shared-protocol estimate disproved by direct code reads.
- **A4** — BFCL v4 explicit refusal guard added to Phase 6 with new `tests/test_simulator_bfcl_refusal.py`.
- **A5** — Principle 5 narrowed: Gantt parity required only for rendering-logic changes, not pure registry-content additions. Phase 5 gains a preflight classification sub-task.
- **A6** — Single review gate split into three: Gate-A after P2, Gate-B after P4, Gate-C after P6.
- **MODE** — Promoted SHORT → DELIBERATE. Added Phase 1 pre-mortem (Pre-mortem B) and per-phase test-lane fill-in.

### R1 → R2 (historical)

Three required Critic amendments + three minors + one bonus.

- **R1-1** — Phase 1.5 split into **Phase 1.5.0** (design audit, no code) and **Phase 1.5.1** (implementation). Phase 1.5 firmly in-plan, not deferral-eligible. New **Gate-D** + new **Pre-mortem C**. Honors user intent for 2×2 matrix through simulate.
- **R1-2** — `mcp_config` nests under `metadata.run_config` per new extension convention (probed `log_metadata(**kwargs)` at `src/harness/trace_logger.py:35`). Principle 2 amended. Phase 4 acceptance uses exact jq path.
- **R1-3** — Phase 5 screenshot-diff replaced with BeautifulSoup DOM snapshot test extending `tests/test_gantt_template_bugs.py`. Golden fixture at `tests/fixtures/gantt_baseline_pre_p5.golden.html`.
- **m1** — Phase 1 wall-clock timing flake replaced with `'agents.openclaw' not in sys.modules` assertion.
- **m2** — Phase 4 collector kwarg passthrough test added.
- **m3** — Phase 5 e2e Playwright tooling specified.
- **bonus** — Phase 0 deliverable (e): `gantt_data.py::build_gantt_payload_multi` payload shape.

### R2 → R3 (this revision — surgical, 3 fixes)

| # | Tag | What changed | Why | Sections touched |
|---|---|---|---|---|
| **R3-1** | DAG dependency honesty | Chose **Option (a)**: redrew the DAG so P1.5.0 + P1.5.1 require BOTH Gate-A AND Gate-B clean before starting. P4 has explicit edge into P1.5.0. Gate-B becomes a second upstream gate. New R10 critical-path risk. Option (b) (synthetic fixtures) explicitly rejected because Pre-mortem C item 2 requires real recorded MCP `tool_exec` payloads. | R2 prose listed `"Depends on: ... Phase 4"` for 1.5.1 but the DAG diagram only showed P4→P6. Critic-level inconsistency. | §4 DAG, §4 P1.5.0 (Depends on), §4.9 Gate-D scope, §7 R10 new |
| **R3-2** | Phase 1.5.1 warmup default → 0, opt-in via flag | Default `warmup_skip_iterations` from `N=1` to `N=0`. Researcher opts in via `--warmup-skip-iterations` only when empirical probe in Phase 1.5.0 shows first-iter variance ≥20% vs steady-state. The 20% threshold lives in design doc, not hardcoded. Phase 1.5.0 deliverable gains question (4): empirical probe + recommendation. Pre-mortem C item 3 rewritten. R9 rewritten. | Architect B.2: `=1` was unjustified magic, violating CLAUDE.md "No Unjustified Complexity". | §5 Pre-mortem C item 3, §4 P1.5.0 deliverables, §4 P1.5.1 acceptance, §7 R9 |
| **R3-3** | Playwright CI spec (chromium-only + cache) | Added explicit install command (`playwright install chromium`, ~150MB, not full matrix ~450MB), GitHub Actions cache spec keyed on `pyproject.toml` hash for clean invalidation on version bumps, and pinned versions in test extras (`playwright>=1.40,<2.0`, `beautifulsoup4>=4.12`, `lxml>=5.0`). Added new "Phase 5 CI / dependency requirements" subsection. Local runbook hook in `docs/vastai_setup.md`. | Architect B.4: R2 named Playwright without specifying install surface or CI cache, which would silently bloat CI runtime by ~300MB and re-download on every run. | §4 Phase 5 (new CI subsection), §6 Phase 5 e2e lane, §4 Phase 3 (runbook cross-reference) |

R0/R1/R2 direction validated. R3 is three surgical inserts — no restructuring.

---

## 1. PRINCIPLES

1. **Scaffold polymorphism over branching.** The simulator must not contain `if scaffold == "miniswe"` inside its replay loop. Dispatch happens via a registry keyed on `trace.scaffold` from v5 metadata. Unknown or unsupported scaffolds raise `NotImplementedError` at load time with a descriptive message.
2. **Trace format v5 is frozen.** All new trace-action fields live inside `TraceAction.data` blobs. **Extension convention: CLI-provided run configuration nests under `metadata.run_config`. Trace-action-level extensions nest under `TraceAction.data`. Neither requires a format version bump.** (`log_metadata` already accepts `**kwargs`, verified at `src/harness/trace_logger.py:35`.)
3. **No synthetic shortcuts.** vLLM runs for real. Podman pulls real images. MCP servers actually handshake with context7. If slow, report and wait — never stub. Fixtures used for replay testing must be real recorded traces, not hand-built stand-ins.
4. **Config over code, with explicit acknowledgement.** All configurable values via YAML/CLI. Where an opt-out affects experiment semantics, it MUST be an affirmative researcher choice (`--mcp-config none`), never an implicit default.
5. **Python/JS Gantt parity is mandatory for rendering-logic changes only.** Pure registry-content additions flow through `build_gantt_payload_multi` data-driven. Rendering-logic changes require paired Python+JS commits. Phase 5 classifies each edit before coding.
6. **Absolute snapshots, not deltas.** Heterogeneous metric bags cannot be uniformly delta'd. Collection-time storage is always absolute; deltas are an analysis-time concern.

---

## 2. DECISION DRIVERS

1. **vast.ai DinD-free constraint.** Forces a podman socket shim over forking the upstream harness.
2. **Trace fidelity for the simulator's research claim.** Sim traces missing vLLM scheduler metrics collapse the contribution. Snapshots must be absolute and lossless.
3. **OpenClaw realism delta — affirmative-only.** MCP + skills are what distinguish OpenClaw. CLI refuses to start an openclaw run without an explicit `--mcp-config`.

---

## 3. VIABLE OPTIONS (chosen set)

| Axis | Chosen | Notes |
|---|---|---|
| **A** Simulator scaffold support | **A1** scaffold registry, mini-swe in P1, openclaw via P1.5.0+P1.5.1 | Honest scoping. Polymorphism via lookup, not branching. |
| **B** Container runtime | **B1** podman rootless + `DOCKER_HOST` shim | No fork. |
| **C** MCP wiring | **C1** `--mcp-config` mandatory; `none` is affirmative opt-out | No silent omission. |
| **D** Gantt extensibility | **D1** extend default registries + Phase 5 preflight classification | Payload-driven; minimal JS edits. |

Rejected options with rationale documented in §8 ADR.

---

## 4. PHASED EXECUTION PLAN

### Dependency DAG (R3 — Option (a))

```
P0 → P1 → P2 → Gate-A ──┐
                        ├→ P1.5.0 → P1.5.1 → Gate-D → P6 → Gate-C
     P4 → Gate-B ───────┘                              ▲
                                                       │
P0 → P3 ───────────────────────────────────────────────┤
                                                       │
P2 + P4 → P5 ──────────────────────────────────────────┘
```

**Reading the DAG:**
- P1.5.0 requires BOTH Gate-A (Phase 2 clean) AND Gate-B (Phase 4 clean) before starting. Phase 4 is an upstream dependency of openclaw simulate because Phase 1.5.1 tests need real recorded openclaw MCP traces as fixtures, and those fixtures only exist after Phase 4 lands.
- The openclaw simulate critical path is `P0 → P1 → P2 → Gate-A + P0 → P4 → Gate-B → P1.5.0 → P1.5.1 → Gate-D → P6`.
- P3 (podman) remains fully parallel — no dependency on P1/P2/P4/P5/P1.5.
- P5 remains dependent on both P2 (sim_metrics) and P4 (MCP events).
- P6 converges all tracks for the full matrix smoke.

### Phase 0 — Branch + schema audit (no code changes)

**Files read:** `src/agents/openclaw/tools/mcp.py:138-270`, `src/agents/openclaw/_loop.py:186-318`, `src/harness/scheduler_hooks.py`, `src/harness/vllm_entrypoint_with_hooks.py`, `src/agents/miniswe/agent.py:208`, `src/agents/openclaw/eval/prepare.py:20`, `src/agents/openclaw/eval/runner.py:120-170`, `src/trace_collect/gantt_data.py`, `src/trace_collect/gantt_builder.js`, `configs/benchmarks/*.yaml`.

**Deliverables — `.omc/plans/phase0-schemas.md` documenting verbatim:**
- (a) `connect_mcp_servers` expected dict shape.
- (b) `PreemptionSnapshot.__dict__` field names + types tagged as counter / gauge / ratio.
- (c) `MiniSWECodeAgent.prepare()` signature and side-effects on `self`.
- (d) `prepare_workspace()` free-function signature and how `openclaw.eval.runner.run_task` composes it through `SessionRunner`.
- (e) `gantt_data.py::build_gantt_payload_multi` shape of `span_registry` and `marker_registry` kwargs and payload flow — confirms Phase 5's payload-only classification before any code is written.

**Acceptance:** doc committed; zero code changes.
**Verification:** `git diff --stat main..HEAD` shows only `.omc/plans/phase0-schemas.md`.
**Rollback:** `git branch -D dev/trace-sim-vastai-pipeline`.

---

### Phase 1 — Mini-swe simulator scaffold registry (mini-swe only)

**Files touched:**
- `src/trace_collect/simulator.py:213-260` — remove hardcoded `MiniSWECodeAgent` import.
- New: `src/trace_collect/scaffold_registry.py` (~40 LOC, lazy imports inside adapter callable).
- `src/agents/miniswe/__init__.py` — register `"miniswe"` adapter.

**What changes:** Registry has exactly one entry; `get_prepare("openclaw")` raises `NotImplementedError("Openclaw trace replay is not yet supported; see Phase 1.5 of the trace-sim-vastai-pipeline plan. Mini-swe traces are supported.")`; replay loop reads `trace.scaffold` from v5 metadata and calls `get_prepare(...)(task)`.

**Acceptance:**
- `grep -rn "MiniSWECodeAgent\|if scaffold" src/trace_collect/simulator.py` → zero matches.
- Mini-swe fixture trace replays byte-identical (timestamps normalized) vs pre-Phase-1 baseline.
- Openclaw trace → `NotImplementedError` with `"Phase 1.5"` substring AND `'agents.openclaw' not in sys.modules` at the moment of error (asserted in the test, m1).
- Bogus scaffold → descriptive `NotImplementedError` listing known scaffolds.

**Verification:**
- Unit `tests/test_scaffold_registry.py`.
- Integration `tests/test_simulator_miniswe_regression.py` — golden file diff with timestamp normalization.
- Observability: simulator startup log `scaffold=miniswe prepare=<callable>` at INFO.

**Rollback:** revert two commits.

---

### Phase 2 — vLLM scheduler metrics merge (absolute snapshots)

**Files touched:**
- `src/trace_collect/simulator.py` (streaming loop ~line 300-400).
- `src/harness/scheduler_hooks.py` — expose `get_snapshot() -> PreemptionSnapshot`.
- New: `src/harness/metrics_client.py`.
- New: `src/analysis/sim_metrics_delta.py`.

**What changes:** Per replay iteration, store `data.sim_metrics.vllm_scheduler_snapshot = {**snapshot.__dict__}` (absolute, field-for-field). Per-action timing under `data.sim_metrics.timing = {ttft_ms, tpot_ms, total_ms}`. Delta computation lives only in `sim_metrics_delta.py`, only for counter-typed fields, raises `TypeError` on gauge/ratio delta attempts.

**Acceptance:**
- `jq '.data.sim_metrics.vllm_scheduler_snapshot | keys'` matches `PreemptionSnapshot.__dict__.keys()` exactly.
- First 10 llm_call records have non-null snapshot dicts.
- `compute_preemption_delta([0,2,2,5,5])` returns `[2,0,3,0]`; gauge delta raises.

**Verification:**
- Unit `tests/test_sim_metrics_delta.py`, `tests/test_sim_metrics_absolute.py`.
- Integration: real localhost vLLM run, jq schema assertions.
- Observability: per-action log `iter=N ttft=Xms tpot=Yms preempt_abs=Z gpu_cache_pct=W`.

**Depends on:** Phase 1.

**→ Gate-A triggers here.**

---

### Phase 1.5.0 — OpenClaw simulate design audit (no code) [depends Gate-A AND Gate-B]

**Files read:** `src/agents/openclaw/eval/runner.py:120-170`, `src/agents/openclaw/eval/_session_runner.py` (if present), `src/agents/openclaw/_loop.py:186-318` (esp. MCP session reuse 295-318), `src/agents/openclaw/eval/prepare.py`.

**Deliverable — `.omc/plans/phase1.5-design.md` answering all four questions with code references:**

1. **Does replay need to preserve SessionRunner's bus-based dispatch shape, or is a linear `execute_trace_tool` loop sufficient for the simulator's measurement surface?** Cite line numbers.
2. **Is there any openclaw-only state (skills cache, MCP session) that would be lost in a linear replay and distort TTFT/TPOT measurements?** Does `_loop.py:295-318` reuse MCP connections across iterations? Cite line numbers.
3. **Does Phase 1.5 require a v5 header addition** to record which SessionRunner shape the trace used at collect time? If yes, route through `metadata.run_config.session_runner_shape` per the Principle 2 extension convention — no version bump.
4. **[R3-2] Empirical warmup probe.** Measure first-iteration vs steady-state latency variance on a probe openclaw trace (from Phase 4 fixtures). If variance <20%, the `--warmup-skip-iterations` default stays at `0` and the flag is documented as available-but-unused. If variance ≥20%, recommend a non-zero default with numeric justification and attach the probe data to the design doc. Threshold (20%) is documented HERE (in the plan + design doc), not hardcoded in code.

**Acceptance:** design doc exists; all 4 questions answered with code references; chosen replay strategy named (linear vs bus-shim); warmup recommendation with empirical data attached.

**Gate (mini-review):** fresh reviewer sub-agent confirms the design is coherent and references real code BEFORE Phase 1.5.1 starts. If the audit reveals unworkable complexity (Risk R7), halt the plan and escalate to user.

**Depends on:** **Gate-A clean AND Gate-B clean.** Gate-B is upstream because the empirical warmup probe (question 4) needs a real openclaw+MCP fixture, which only exists after Phase 4 lands. Logged at `.omc/logs/phase1.5.0-design-review.md`.

---

### Phase 1.5.1 — OpenClaw simulate implementation

**Files touched:**
- `src/trace_collect/scaffold_registry.py` — add `"openclaw"` entry.
- New: `src/agents/openclaw/simulate_adapter.py` — wraps `prepare_workspace()` + builds replay-mode executor per 1.5.0 decisions.
- `src/agents/openclaw/__init__.py` — register adapter callable (lazy import).

**What changes:** The simulator can now load openclaw v5 traces and replay them through the registry. Replay-mode executor reuses recorded MCP `tool_exec` results from the source trace — does NOT re-dispatch to context7 — preventing measurement contamination. CLI adds `--warmup-skip-iterations N` flag, plain integer ≥ 0, **default 0**. When non-zero, the first N iterations are still measured but tagged with `data.sim_metrics.warmup = true` for analysis-time exclusion.

**Acceptance:**
- `grep -rn "MiniSWECodeAgent\|OpenClaw.*Runner" src/trace_collect/simulator.py` → zero matches (dispatch is registry-only).
- Recorded openclaw fixture trace replays successfully with `data.sim_metrics.vllm_scheduler_snapshot` populated field-for-field equivalent to mini-swe replay output schema.
- MCP events from the source trace propagate through to the simulate trace (not stripped).
- TOOL-category events in collect-time trace and simulate-time trace match 1:1 in sequence and `tool_name` (timing differs intentionally).
- Per Pre-mortem C item 2: zero network egress to context7 during simulate replay (asserted by network mock).
- **[R3-2]** `--warmup-skip-iterations` default is `0`; when invoked with a non-zero N, the first N iterations carry `data.sim_metrics.warmup = true`; when invoked with default, no warmup flagging occurs and every iteration is a full measurement.

**Verification:**
- Unit `tests/test_openclaw_simulate_adapter.py` — adapter constructs without network; replays 5-action fixture; asserts zero context7 network egress; asserts MCP results sourced from trace; asserts warmup flag default-off behavior.
- Integration: replay a recorded openclaw fixture (from Phase 4) against local vLLM; diff TOOL-category events 1:1.
- Observability: startup log `scaffold=openclaw prepare=<callable_repr> warmup_skip=<N>`.

**Pre-mortem (Pre-mortem C, see §5):** three scenarios for the openclaw replay path.

**Depends on:** Phase 1.5.0 with its mini-gate clean. (Phase 1.5.0 itself encodes the Gate-A + Gate-B upstream dependencies.)

**→ Gate-D triggers here.**

---

### Phase 3 — Podman bootstrap for vast.ai

**Files touched:**
- New: `scripts/setup/install_podman_vastai.sh`, `scripts/setup/start_podman_socket.sh`, `docs/vastai_setup.md`.
- `scripts/full_setup.sh` — guarded preflight insert.

**What changes:** Install `podman`, `fuse-overlayfs`, `slirp4netns`, `uidmap`. Preflight `/etc/subuid`. Start `podman system service --time=0`. Export `DOCKER_HOST=unix:///run/user/$UID/podman/podman.sock`. Preflight via `hello-world` image before SWE-Bench pull. **[R3-3]** `docs/vastai_setup.md` also documents the local developer Playwright runbook hook (see Phase 5 CI requirements): the same `playwright install chromium` step runs on vast.ai before Phase 6 smoke.

**Acceptance:** Fresh A100, `bash scripts/full_setup.sh` exits 0 without sudo for podman ops; `docker.from_env().ping()` returns True; 1-instance harness eval reaches a patch verdict; storage driver is `overlay` not `vfs`.

**Verification:**
- Integration: install script exit 0; `podman info | grep storage.driver` → `overlay`.
- e2e: 1-instance SWE-rebench harness run reaches a verdict.
- Observability: `podman info` saved to `.omc/logs/phase3-podman-info.log`.

**Pre-mortem (Pre-mortem A, see §5).**

**Depends on:** none. Parallel to P1/P2/P4.

---

### Phase 4 — OpenClaw MCP enablement (mandatory `--mcp-config`)

**Files touched:**
- `scripts/run_nanobot_eval.py:86-93`, `src/trace_collect/cli.py:102-133`, `src/trace_collect/collector.py:290-350`.
- New: `configs/mcp/context7.yaml`.
- `src/agents/openclaw/_skills.py` — emit `CONTEXT` category event with `data.skill_name` per skill load.
- **DELETED:** `configs/mcp/empty.yaml`.

**What changes:**
- `--mcp-config` accepts YAML path or literal `none`. Absent → exit code 2 with message `"MCP config is required for openclaw; pass --mcp-config configs/mcp/context7.yaml or --mcp-config none to acknowledge running without MCP"`.
- `none` → empty MCP dict + trace header records the affirmative choice.
- Path → YAML loaded, passed to `SWEBenchRunner(mcp_servers=...)`, header records identity.
- Header storage uses the R1-2 extension convention: `log_metadata(scaffold="openclaw", run_config={"mcp_config": "context7.yaml"})` (or `"none"`). `run_config` is the conventional extension blob for CLI-provided configuration.

**Acceptance:**
- `collect --scaffold openclaw --mcp-config configs/mcp/context7.yaml --instances 1` → trace contains ≥1 `"category":"MCP"` event AND ≥1 `tool_exec` with `tool_name.startswith("mcp_")`. And `jq '.run_config.mcp_config' trace.jsonl | head -1` returns `"context7.yaml"`.
- `collect --scaffold openclaw --mcp-config none --instances 1` → zero MCP events; `jq '.run_config.mcp_config' trace.jsonl | head -1` returns `"none"`.
- `collect --scaffold openclaw --instances 1` (no flag) → exit 2 with exact stderr substring above.
- Skills loaded → `CONTEXT` events with `data.skill_name` populated.

**Verification:**
- Unit `tests/test_cli_mcp_flag_enforcement.py` — three subtests with exact stderr substring match.
- Integration:
  - `jq` assertions above.
  - **[m2]** `tests/test_collect_traces_kwarg_passthrough.py` monkeypatches `collect_traces`, invokes CLI with `--mcp-config configs/mcp/context7.yaml`, asserts `collect_traces` received `mcp_config` as a kwarg (not re-read from `sys.argv`).
- Observability: CLI logs `mcp_config=<value>` at startup.

**Depends on:** Phase 0.

**→ Gate-B triggers here.**

---

### Phase 5 — Gantt extension (preflight classification + DOM snapshot)

**Sub-task 0 (preflight, no code):** Read `gantt_builder.js` and `gantt_template.html` end-to-end. Classify each proposed edit as **payload-only** or **rendering-logic**. Document in `.omc/plans/phase5-parity-classification.md` BEFORE writing code. Phase 0 deliverable (e) feeds this directly.

**Files touched (Python, payload-only edits):**
- `src/trace_collect/gantt_data.py:21` — extend `_MARKER_CATEGORIES` with `"MCP"`.
- `src/trace_collect/gantt_data.py:48-51` — extend `ACTION_TYPE_MAP` with `mcp_call`.

**Files touched (Python + JS pair, rendering-logic edits — only if preflight classifies them so):**
- `src/trace_collect/gantt_data.py::_extract_detail_from_action` — add tooltip rows for `sim_metrics.vllm_scheduler_snapshot.{num_preemptions_total, gpu_cache_usage_perc, gpu_prefix_cache_hit_rate}` and `sim_metrics.timing.{ttft_ms, tpot_ms}`.
- `src/trace_collect/gantt_builder.js` — mirror extraction in the same commit IF preflight requires it.

**Acceptance:**
- Simulate-mode trace: Gantt renders TTFT/TPOT on LLM span tooltips.
- MCP-enabled OpenClaw trace: MCP markers render as a distinct category.
- **[R1-3]** Pre-Phase-5 baseline trace renders with structurally-identical SVG. Asserted by extending `tests/test_gantt_template_bugs.py` with `test_gantt_dom_snapshot_pre_p5_unchanged()`: loads `tests/fixtures/gantt_baseline_pre_p5.golden.html` (regenerated once at the Phase-5 starting commit from a frozen mini-swe trace), re-renders the same trace post-P5, parses both with `BeautifulSoup(..., "lxml")`, extracts SVG `<g>` subtree (tag tree, class attrs, data-* attrs), asserts equality. No pixel comparisons.

#### Phase 5 CI / dependency requirements [R3-3]

Test extras additions (pinned versions for reproducibility):
```
playwright>=1.40,<2.0
beautifulsoup4>=4.12
lxml>=5.0
```

**Install command (required in CI and local dev):**
```
playwright install chromium
```
NOT `playwright install` — the unqualified form downloads the full browser matrix (~450MB, chromium + firefox + webkit). Chromium-only is ~150MB. The qualified form is documented at the top of `tests/test_gantt_browser_smoke.py` as a module docstring, and in `docs/vastai_setup.md` for the vast.ai local runbook.

**GitHub Actions cache spec (in `.github/workflows/test.yml` or equivalent):**
```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/ms-playwright
    key: ${{ runner.os }}-playwright-${{ hashFiles('pyproject.toml') }}
    restore-keys: |
      ${{ runner.os }}-playwright-
```
Key is scoped to `pyproject.toml` hash so Playwright version bumps (via the pin `>=1.40,<2.0`) invalidate the cache cleanly. The `restore-keys` fallback allows partial-hit reuse during dev iterations that bump other deps.

**Local developer runbook:** `make playwright-install` target (or equivalent shell recipe) added to the project Makefile if one exists, else a one-liner documented in `docs/vastai_setup.md` alongside the podman bootstrap. Vast.ai smoke runs must execute this before Phase 6 matrix.

**Verification:** see §6 Phase 5 lane.

**Depends on:** Phase 2 + Phase 4 merged.

---

### Phase 6 — Full matrix smoke + BFCL refusal

**Matrix cells (4 core + 1 refusal):**
1. `mini-swe × swe-bench-verified` — collect → simulate → gantt.
2. `mini-swe × swe-rebench` — collect → simulate → gantt.
3. `openclaw × swe-bench-verified --mcp-config configs/mcp/context7.yaml` — collect → simulate (via P1.5.1) → gantt.
4. `openclaw × swe-rebench --mcp-config configs/mcp/context7.yaml` — collect → simulate (via P1.5.1) → gantt.
5. **BFCL v4 refusal guard** — `simulate --source-trace <bfcl_v4_fixture>` exits with `NotImplementedError("BFCL v4 traces have task_shape='function_call' with needs_prepare=False, which the simulator does not support. Simulate mode requires a prepare-able scaffold.")`

**Files touched:**
- New: `scripts/smoke_full_matrix.sh`.
- New: `tests/test_simulator_bfcl_refusal.py`.
- `src/trace_collect/simulator.py` — task_shape check before scaffold lookup; mirrors `bfcl_v4.py:294-301`.

**Acceptance:**
- All 4 core cells produce v5 traces with `sim_metrics`, Gantt HTML renders without JS errors, openclaw cells contain MCP events, openclaw simulate traces preserve recorded MCP results without re-dispatch.
- BFCL refusal: unit test green; smoke script's BFCL invocation exits nonzero with the expected message.

**Verification:** see §6 Phase 6 lane.

**Depends on:** P1, P2, P3, P4, P5, P1.5.1 all merged.

**→ Gate-C triggers here.**

---

### §4.9 — Four Review Gates

Each gate spawns a FRESH `oh-my-claudecode:code-reviewer` sub-agent (opus, fresh context). No gate reuses a previous gate's reviewer. Iterate until clean per CLAUDE.md.

**Gate-A (after Phase 2) — Metrics integration review.** Scope: P1 + P2 diffs. Verify: (a) no scaffold-specific branching in replay loop, (b) absolute snapshots, (c) no delta in collection path, (d) registry has one entry with descriptive `NotImplementedError`, (e) `sim_metrics_delta.py` imported only by analysis code, (f) Phase 1 byte-identical regression passes, (g) lazy imports prevent `agents.openclaw` from loading on mini-swe replay.

**Gate-B (after Phase 4) — MCP plumbing review.** Scope: P4 diffs. Verify: (a) `--mcp-config` mandatory with exact error string, (b) `none` affirmative + recorded in `metadata.run_config`, (c) skill events emit from `_skills.py`, (d) no `configs/mcp/empty.yaml` in tree, (e) collector receives `mcp_config` as kwarg per m2 test.

**Gate-D (after Phase 1.5.1) — OpenClaw replay review.** Scope: P1.5.0 design doc + P1.5.1 diffs. Verify: (a) replay strategy matches design doc decision, (b) MCP results are reused from source trace not re-dispatched, (c) warmup handling matches Pre-mortem C item 3 with default `N=0`, (d) TOOL-category 1:1 match between collect and simulate traces, (e) registry has both entries with no scaffold-branching anywhere in `simulator.py`, (f) **[R3-1]** verify that the P4 → P1.5.0 edge is respected in practice: the openclaw fixture used by P1.5.1 tests originated from a Gate-B-clean Phase 4 collect run (traceable via the `run_config.mcp_config` field in the fixture's header), not hand-crafted.

**Gate-C (after Phase 6) — Full regression review.** Scope: all phases. Verify: (a) Python/JS Gantt parity diff passes, (b) BFCL refusal test green, (c) Phase 6 matrix exit codes all zero, (d) no new hardcoded values in `src/**`, (e) v5 trace format still frozen (`trace_format_version == 5`) and extensions go through `metadata.run_config` or `TraceAction.data`, (f) all three pre-mortems (A, B, C) have mitigations implemented in code, (g) **[R3-3]** Playwright CI cache hits on second run (verifiable from CI logs), and the browser install is chromium-only.

---

## 5. PRE-MORTEMS (DELIBERATE mode — three required)

### Pre-mortem A: Phase 3 (podman)

1. **vast.ai base lacks fuse-overlayfs.** Falls back to `vfs`, pulls take ~45min, 3x disk. Mitigation: preflight `modprobe fuse || echo missing`; fail loudly.
2. **`/etc/subuid` missing.** Rootless podman EPERM. Mitigation: preflight `grep $(id -un) /etc/subuid`; print exact `usermod` remediation and exit. Don't auto-run.
3. **`podman system service` crashes mid-run.** `DOCKER_HOST` socket disappears. Mitigation: `--time=0`, log PID to `.omc/state/podman.pid`, smoke script restarts dead service.

### Pre-mortem B: Phase 1 (scaffold registry)

1. **Mini-swe regression through hidden state.** `MiniSWECodeAgent.prepare` stashes state on `self`. Wrapping in adapter could lose state across iterations. Mitigation: byte-identical regression test with golden fixture (REQUIRED in Phase 1 acceptance).
2. **Import cycle between `trace_collect.simulator` and `agents.miniswe`.** Mitigation: lazy imports inside the adapter callable; unit test asserts `import trace_collect.scaffold_registry` does not transitively load `agents.*`.
3. **Future scaffold forgets to register.** Mitigation: `get_prepare` raises descriptive `NotImplementedError` listing known scaffolds.

### Pre-mortem C: Phase 1.5.1 (openclaw replay) — R3-2 amended

1. **Replay-mode executor diverges from collect-time SessionRunner subtly**, producing sim_metrics that look right but are measured on a different execution shape than production. **How we catch:** diff collect-time and simulate-time traces on TOOL-category events — sequence and `tool_name` must match 1:1 (timing differs intentionally). Test: `test_openclaw_replay_event_parity`. Any sequence divergence fails loudly.

2. **MCP server handshake in simulate mode duplicates collect-time MCP calls**, re-contacting context7 and inflating measured latency. **How we catch:** simulate mode MUST reuse the source trace's recorded MCP `tool_exec` results (same pattern as other tools) rather than re-dispatching. Unit test in `tests/test_openclaw_simulate_adapter.py` mocks network and asserts zero outbound calls to context7 during MCP-containing trace replay. This is ALSO why Option (b) of R3-1 was rejected: synthetic fixtures would not have faithful recorded MCP payloads, so the "zero re-dispatch" invariant could not be credibly tested.

3. **Skills cache cold-start may skew first-iteration measurements.** **Default behavior (R3-2):** measure every iteration including the first (`N=0`). The researcher opts in via the new `--warmup-skip-iterations N` CLI flag if and only if the Phase 1.5.0 empirical probe shows first-iteration latency variance >20% vs steady-state on a representative trace. The 20% threshold is documented in the design doc (`.omc/plans/phase1.5-design.md`), not hardcoded in code. When the flag IS set by the researcher, the first N iterations carry `data.sim_metrics.warmup = true` for analysis-time exclusion, but they are still fully measured — the flag only controls analysis-time treatment, not collection-time behavior. This replaces R2's unjustified `N=1` default with an empirically-justified opt-in path per CLAUDE.md "No Unjustified Complexity".

---

## 6. EXPANDED TEST PLAN (DELIBERATE mode — per phase, 4 lanes)

### Phase 0
- Unit / Integration / e2e: N/A (docs only).
- Observability: `.omc/plans/phase0-schemas.md` exists with all five sections (a-e) populated.

### Phase 1
- Unit `tests/test_scaffold_registry.py` — registry contents `{"miniswe"}`; `NotImplementedError` for `"openclaw"` with `"Phase 1.5"` substring; `'agents.openclaw' not in sys.modules` after the error (m1); lazy-import assertion.
- Integration `tests/test_simulator_miniswe_regression.py` — fixture/golden byte-diff with timestamp normalization.
- e2e: Phase 6 matrix cells 1, 2.
- Observability: startup log `scaffold=miniswe prepare=<callable>`.

### Phase 2
- Unit `tests/test_sim_metrics_delta.py` (counter `[0,2,2,5,5]→[2,0,3,0]`, gauge raises); `tests/test_sim_metrics_absolute.py` (introspects `dataclasses.fields(PreemptionSnapshot)`).
- Integration: real vLLM run; `jq` schema match.
- e2e: cells 1, 2 carry sim_metrics.
- Observability: per-action log line.

### Phase 1.5.0
- Unit / Integration / e2e: N/A (design audit).
- Observability: `.omc/plans/phase1.5-design.md` with answers to all 4 questions (including R3-2 empirical warmup probe with raw data), plus mini-gate review log at `.omc/logs/phase1.5.0-design-review.md`.

### Phase 1.5.1
- Unit `tests/test_openclaw_simulate_adapter.py` — adapter constructs without network; replays 5-action fixture; asserts zero context7 network egress (network mock); asserts MCP results sourced from trace; asserts `--warmup-skip-iterations` **default is 0**; asserts warmup flag behavior for non-zero N.
- Integration: replay a recorded openclaw fixture (produced by a Gate-B-clean Phase 4 run) against local vLLM; TOOL-category 1:1 diff vs source trace.
- e2e: cells 3, 4.
- Observability: startup log `scaffold=openclaw prepare=<callable> warmup_skip=<N>`; warmup flag in trace when N>0.

### Phase 3
- Unit: N/A (bash).
- Integration: install script exit 0; `podman info | grep storage.driver` → `overlay`.
- e2e: 1-instance SWE-rebench harness run reaches a verdict.
- Observability: `podman info` saved to log.

### Phase 4
- Unit `tests/test_cli_mcp_flag_enforcement.py` — three subtests with exact stderr substring match.
- Integration: `jq` MCP-event assertion; `jq '.run_config.mcp_config'` matches expected value; **[m2]** `tests/test_collect_traces_kwarg_passthrough.py` monkeypatch test.
- e2e: cells 3, 4.
- Observability: CLI startup log; trace header.

### Phase 5
- Unit `tests/test_gantt_sim_metrics_tooltip.py` — fake action with `sim_metrics`, assert `_extract_detail_from_action` rows; payload-only assertions on `_MARKER_CATEGORIES` and `ACTION_TYPE_MAP`.
- Integration: build payload from real simulate fixture, assert MCP category in `marker_registry`.
- DOM snapshot: `test_gantt_dom_snapshot_pre_p5_unchanged()` extending `tests/test_gantt_template_bugs.py` per R1-3.
- **[m3 / R3-3]** e2e `tests/test_gantt_browser_smoke.py` using Playwright Python sync API:
  ```python
  def test_gantt_no_console_errors(rendered_html_path):
      with sync_playwright() as p:
          browser = p.chromium.launch()
          page = browser.new_page()
          errors = []
          page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
          page.goto(f"file://{rendered_html_path}")
          page.wait_for_load_state("networkidle")
          assert len(errors) == 0, f"Console errors: {errors}"
          browser.close()
  ```
  Module docstring documents `playwright install chromium` prerequisite (chromium-only, not full matrix). CI runs this after the cache-restore step documented in the §Phase 5 CI requirements subsection. If logic-level edits happened: parity test diffs Python vs JS extraction on a shared fixture; zero-diff.

### Phase 6
- Unit `tests/test_simulator_bfcl_refusal.py` — BFCL v4 trace header with `task_shape='function_call'`, asserts exact `NotImplementedError` substring.
- Integration: `scripts/smoke_full_matrix.sh` exit codes per cell.
- e2e: same script on real A100; manual Gantt verification.
- Observability: matrix log at `.omc/logs/smoke-matrix-$(date +%s).log`.

---

## 7. RISK REGISTER

| # | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | Podman shim fails on vast.ai | Med | High | Pre-mortem A items 1-3; escalate before forking harness | Phase 3 |
| R2 | ~~Delta off-by-one~~ REMOVED (absolute storage) | — | — | — | — |
| R2' | `PreemptionSnapshot` schema changes upstream | Low | Med | Schema captured in Phase 0; consumers read keys defensively | Phase 2 |
| R3 | Context7 handshake adds 5-10s/step; smoke timeouts | Low | Med | Session caching in `_loop.py:295-318`; raise smoke timeout to 600s | Phase 4 |
| R4 | JS Gantt builder diverges for logic-level edits | Med | High | Phase 5 preflight; Gate-C parity diff; DOM snapshot test | Phase 5/6 |
| R5 | Phase 1 mini-swe regression via hidden state | Low | High | Byte-identical regression test in Phase 1 acceptance | Phase 1 |
| R6 | Import cycle from scaffold registry | Low | Med | Lazy imports; unit test asserts no side-effect imports | Phase 1 |
| R7 | Phase 1.5.0 design audit reveals openclaw replay requires SessionRunner shim, significantly expanding 1.5.1 scope | Med | High | 1.5.0 is explicitly design-audit-only with no code deliverable; if audit reveals unworkable complexity, halt and escalate to user before any 1.5.1 code is written | Phase 1.5.0 |
| R8 | Openclaw simulate re-dispatches MCP, contaminates measurements | Low | High | Pre-mortem C item 2; network mock unit test; Gate-D verification | Phase 1.5.1 |
| **R9** | **Skills cache cold-start may skew first-iter measurements.** | Med | Low | **Empirical probe in Phase 1.5.0 audit (question 4); `--warmup-skip-iterations` opt-in flag default `0`; researcher documents justification if non-zero; 20% variance threshold lives in design doc, not in code.** | Phase 1.5.0 + 1.5.1 |
| **R10** | **Combined critical path P0→P1→P2→Gate-A + P0→P4→Gate-B → P1.5.0 → P1.5.1 → Gate-D is long. If either Gate-A or Gate-B blocks, openclaw simulate cannot start, delaying full-matrix smoke.** | Med | Med | Acknowledged in DAG diagram so the schedule reflects reality. P3 (podman) and P5 (Gantt) proceed in parallel lanes; Gate-A/Gate-B blockage does not block the whole plan. R7 halt-and-escalate covers the scope-change case. Flagged in Gate-D scope item (f) to verify the dependency is honored at review time. | Phase 1.5.0 / Gate-D |

---

## 8. ADR

**Decision:** Build the trace→simulate→gantt pipeline on vast.ai via (1) a scaffold-registry simulator with mini-swe in Phase 1 and openclaw in Phase 1.5 (split into 1.5.0 design audit + 1.5.1 implementation, requiring both Gate-A and Gate-B clean upstream), (2) a podman rootless shim exposing `DOCKER_HOST` to the unmodified upstream SWE-Bench harness, (3) mandatory `--mcp-config` for OpenClaw with affirmative `none` opt-out, (4) additive Gantt registry extensions classified per-edit as payload-only vs rendering-logic, (5) absolute vLLM scheduler snapshots in `sim_metrics` with post-hoc delta computation, and (6) `--warmup-skip-iterations` as an opt-in CLI flag defaulting to `0`, gated on an empirical probe in Phase 1.5.0. CLI-provided run configuration uses `metadata.run_config` per the v5 extension convention.

**Drivers:** vast.ai DinD-free constraint; trace fidelity requires vLLM scheduler metrics; OpenClaw realism requires real MCP events, not omission-by-default; user's original request requires both scaffolds × both benchmarks through simulate.

**Alternatives considered:**
- (a) Fork SWE-Bench harness — rejected (maintenance).
- (b) Separate `openclaw_simulator.py` — rejected (duplication).
- (c) Default-on context7 — rejected (reproducibility).
- (d) New `build_sim_gantt_payload()` — rejected (parity cost).
- (e) R0's both-scaffolds-in-Phase-1 protocol — rejected in R1 (false premise: openclaw has no shared `prepare(task)` signature).
- (f) R0's `configs/mcp/empty.yaml` default — rejected in R1 (silent omission gaming).
- (g) R0's delta-merging — rejected in R1 (category error: gauges and ratios cannot be uniformly delta'd).
- (h) R1's "may defer Phase 1.5" — rejected in R2 (user intent requires 2×2 simulate matrix).
- (i) R1's top-level `mcp_config` header field — refined in R2 to nest under `metadata.run_config`.
- (j) R1's "screenshot diff" — rejected in R2 (unspecified tooling); BeautifulSoup DOM snapshot replaces it.
- (k) R2's DAG showing P4→P6 only (not P4→P1.5.1) — rejected in R3 Option (a) makes the edge explicit. Option (b) (synthetic openclaw fixtures in 1.5.0) was also rejected because Pre-mortem C item 2 requires real recorded MCP `tool_exec` payloads for the zero-re-dispatch invariant to be credibly tested.
- (l) R2's `warmup_skip_iterations=1` default — rejected in R3. `1` is unjustified magic per CLAUDE.md. Default is now `0`, with opt-in via CLI flag gated on an empirical probe.
- (m) R2's unqualified `playwright install` — rejected in R3 (downloads ~450MB full browser matrix). Chromium-only is ~150MB. R3 specifies `playwright install chromium` + GitHub Actions cache.

**Why chosen:** Jointly minimize scaffold-specific branching at the replay-loop level, preserve upstream library compatibility, keep trace format v5 frozen, enforce research integrity on MCP usage, preserve all intermediate outputs per CLAUDE.md, complete the 2×2 simulate matrix per user intent, and avoid unjustified hyperparameters.

**Consequences:**
- Phase 1.5 (split into 1.5.0 design audit + 1.5.1 implementation) completes openclaw simulate support inside this plan. The design audit is the scope-honesty checkpoint — if it reveals unworkable complexity, halt and escalate before any code is written.
- Openclaw simulate's critical path requires BOTH Gate-A and Gate-B clean upstream. The DAG reflects this. P3 and P5 remain parallel lanes, so Gate-A/Gate-B blockage does not block the full plan.
- Every openclaw run requires an explicit `--mcp-config` flag; existing scripts must be updated.
- Every new scaffold must register in `SCAFFOLD_PREPARE_REGISTRY`.
- Every new `sim_metrics` field must be an absolute value.
- Every new Gantt edit must classify as payload-only vs rendering-logic.
- Every new CLI-provided config field nests under `metadata.run_config`.
- Warmup handling defaults to measuring every iteration. Researchers opt into warmup-skip only with empirical justification documented in the design doc.
- Test extras grow by pinned Playwright (chromium-only) + BeautifulSoup + lxml; CI adds a cache step keyed on `pyproject.toml` hash; vast.ai runbook adds a `playwright install chromium` prerequisite step.

**Follow-ups:**
1. Context7 rate limiting under batch collect — may need local cache layer past 10 instances/hour.
2. `sim_metrics_delta.py` may grow an aggregation API (mean/max/min for gauges and ratios) as analysis matures.
3. vLLM scheduler hook PID-namespace compat with rootless podman — flagged, verified in Phase 3 e2e.
4. If Phase 1.5.0 reveals SessionRunner shim is required, that becomes its own follow-up ralplan post-halt.
5. If the Phase 1.5.0 empirical probe recommends a non-zero warmup default on a specific trace class, that recommendation becomes a documented preset in `configs/simulate/*.yaml` — not a code-level default.

---

## Consensus signatures

- **Planner R0** — initial draft (SHORT mode)
- **Architect R0** — SOUND WITH AMENDMENTS (6 amendments + promote DELIBERATE)
- **Planner R1** — applied amendments
- **Critic R1** — ITERATE (3 required + 3 minors + 1 bonus)
- **Planner R2** — applied amendments
- **Architect R2** — SOUND WITH NOTES (B.2 warmup, B.4 Playwright)
- **Critic R2** — REJECT on process (looked at wrong file on disk); flagged DAG inconsistency
- **Planner R3** — applied 3 final fixes
- **Critic R3** — **APPROVE** ✅

**Consensus reached:** 2026-04-08, iteration 2.
