# Predictive Tool-Gap Loan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start waiting agent tasks during predicted-long tool calls earlier than causal elapsed-time feedback can, while preserving exact replay trajectories and LLM tail latency.

**Architecture:** Reuse the staged replay queue and host OpenClaw workers. A small filesystem-backed runtime records foreground LLM response times and emits one atomic loan event per foreground task; the parent queue raises its active-task capacity from four when it observes those events. Predictor rows are a label-free ordered subsequence of source exec commands and unavailable rows fall back to the same feedback timer.

**Tech Stack:** Python 3.12, asyncio, stdlib JSON/filesystem IPC, existing OpenClaw replay, ResourceTrace/eBPF, pytest, uv.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-10-predictive-tool-gap-loan-design.md` and Section 5.7 of the objective lock.
- Use exactly the frozen validation task IDs and six-cell order; do not inspect labels or tool durations while building the prediction map.
- No new dependency, service, vLLM change, request semaphore, probability threshold, recursive loan, or CPU/RSS/Disk action.
- Fixed, Feedback, and Predictor use the same coordinator and provider instrumentation.
- The formal run is already authorized after a measured smoke estimate; stop on invalid plumbing, OOM, thermal slowdown, or material overrun.

---

### Task 1: Filesystem-backed loan runtime

**Files:**
- Create: `src/trace_collect/tool_gap_loan.py`
- Create: `tests/test_tool_gap_loan.py`

**Interfaces:**
- Produces `ToolGapPrediction`, `ToolGapLoanConfig`, and `ToolGapLoanRuntime`.
- `ToolGapLoanRuntime.record_llm_response(action_id, latency_ms, wall_end_s)` atomically records foreground response completion.
- `ToolGapLoanRuntime.start_tool(call_id, command, wall_start_s)` returns an active handle and either emits an early loan or schedules the feedback threshold.
- `await ToolGapLoanRuntime.finish_tool(handle, wall_end_s)` cancels an untriggered timer and records tool completion.

- [ ] **Step 1: Write failing runtime tests**

Cover these exact cases in one focused file:

```python
def test_budget_requires_every_foreground_task(tmp_path): ...
def test_predictor_emits_early_loan_when_bucket_lower_edge_exceeds_budget(tmp_path): ...
def test_feedback_emits_only_after_frozen_budget(tmp_path): ...
def test_one_lender_cannot_emit_two_loans(tmp_path): ...
def test_waiting_task_cannot_lend(tmp_path): ...
def test_unmatched_prediction_falls_back_to_feedback(tmp_path): ...
```

The fixtures use four foreground IDs, millisecond-scale latencies, and one
prediction with PMF `[0, 0, 0, 1, 0]`, hard bucket `3`, and lower edge `8.0`.

- [ ] **Step 2: Run the tests and confirm missing-interface failures**

Run: `uv run pytest tests/test_tool_gap_loan.py -q`

Expected: collection/import failure because `tool_gap_loan` does not exist.

- [ ] **Step 3: Implement the minimum runtime**

Use frozen dataclasses and canonical bucket lower edges:

```python
LATENCY_BUCKET_LOWER_EDGES_S = (0.0, 0.5, 2.0, 8.0, 30.0)

@dataclass(frozen=True)
class ToolGapPrediction:
    sample_id: str
    command: str
    probability_by_bucket: tuple[float, ...]
    hard_bucket: int
    provenance: Mapping[str, object]

@dataclass(frozen=True)
class ToolGapLoanConfig:
    arm: Literal["fixed", "feedback", "predictor"]
    state_dir: str
    task_id: str
    foreground_task_ids: tuple[str, ...]
    can_lend: bool
    predictions: tuple[ToolGapPrediction, ...] = ()
```

Write response and loan JSON through a temporary sibling plus `os.replace`.
One worker owns each lender, so a local `loan_claimed` flag is sufficient;
the parent only reads completed files. Prediction entries are consumed as an
ordered subsequence: an unmatched raw exec is unavailable and does not advance
the prediction cursor.

- [ ] **Step 4: Verify runtime tests**

Run: `uv run pytest tests/test_tool_gap_loan.py -q`

Expected: all tests pass without timing retries.

- [ ] **Step 5: Commit the isolated runtime**

```bash
git add -- src/trace_collect/tool_gap_loan.py tests/test_tool_gap_loan.py
git commit -m "[feat] Add predictive gap-loan runtime"
```

### Task 2: OpenClaw lifecycle and label-free predictions

**Files:**
- Modify: `src/trace_collect/openclaw_host_runtime.py`
- Modify: `src/trace_collect/simulate_openclaw.py`
- Modify: `src/agents/openclaw/tools/container.py`
- Modify: `tests/test_simulator_validation.py`

**Interfaces:**
- Consumes `ToolGapLoanConfig` from Task 1.
- `_run_openclaw_replay_session(..., tool_gap_loan: ToolGapLoanConfig | None)` serializes only the current task's label-free prediction entries.
- `OpenClawReplayProvider` records full shadow HTTP latency after response completion.
- `ContainerExecTool` brackets the unchanged `_request("exec", ...)` with `start_tool`/`finish_tool`.

- [ ] **Step 1: Write failing worker/provider/tool tests**

Add assertions that:

```python
request["tool_gap_loan"] == {
    "arm": "predictor",
    "state_dir": str(state_dir),
    "task_id": "task-0",
    "foreground_task_ids": ["task-0", "task-1", "task-2", "task-3"],
    "can_lend": True,
    "predictions": [...],
}
```

The provider records `shadow_generation["latency_ms"]`; the exec tool emits a
loan before executing for `L > B`, emits after the feedback timer otherwise,
and always calls `finish_tool` in `finally`. Assert the original command and
tool result are unchanged.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `uv run pytest tests/test_simulator_validation.py -q -k 'tool_gap or shadow_generation'`

- [ ] **Step 3: Add the narrow lifecycle wiring**

Construct one `ToolGapLoanRuntime` in the host worker and pass it to both the
provider and `ContainerExecTool`. Keep `ResourceTrace.begin_tool_call` and
`finish_tool_call` ordering unchanged. Add tool-gap events to worker status;
do not add them to source/replay action identity.

- [ ] **Step 4: Verify lifecycle tests and existing replay-provider tests**

Run: `uv run pytest tests/test_simulator_validation.py -q`

- [ ] **Step 5: Commit lifecycle wiring**

```bash
git add -- src/trace_collect/openclaw_host_runtime.py src/trace_collect/simulate_openclaw.py src/agents/openclaw/tools/container.py tests/test_simulator_validation.py
git commit -m "[feat] Observe tool gaps in replay"
```

### Task 3: Dynamic staged admission and CLI

**Files:**
- Modify: `src/trace_collect/simulator.py`
- Modify: `src/trace_collect/cli.py`
- Modify: `src/trace_collect/simulate_outputs.py`
- Modify: `tests/test_simulate_cloud_model.py`
- Modify: `tests/test_simulator_validation.py`

**Interfaces:**
- Adds `--tool-gap-loan-arm {fixed,feedback,predictor}` and `--tool-gap-predictions PATH`.
- The prediction file schema is `{task_id: [{sample_id, command, probability_by_bucket, hard_bucket, provenance}, ...]}` and contains no other keys.
- `_run_staged_cloud_model_queue` starts four tasks and sets allowed active tasks to `4 + observed_unique_loans`, capped at eight.

- [ ] **Step 1: Write failing queue and validation tests**

Use fake replay sessions to assert:

```python
assert admitted[:4] == ["task-0", "task-1", "task-2", "task-3"]
assert admitted[4] == "task-4"       # starts after atomic loan file appears
assert max_active == 5
```

Also require workers=1, staged replay, concurrency=4, exactly eight sessions,
shadow generation enabled, request cap absent, and a prediction file only for
the Predictor arm. Reject extra prediction fields, wrong task IDs, malformed
PMFs, and any command list that is not an exact ordered subsequence of source
exec commands.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `uv run pytest tests/test_simulate_cloud_model.py tests/test_simulator_validation.py -q -k 'tool_gap'`

- [ ] **Step 3: Implement one coordinator path for all three arms**

Replace only the staged-queue worker loop when a tool-gap arm is supplied. The
coordinator polls completed atomic loan files, keeps FIFO task order, admits up
to the current capacity, and refills after terminal tasks. Fixed emits no
loans; Feedback and Predictor differ only inside worker decision timing.

Record arm, foreground/waiting IDs, prediction-file path, admission events,
loan events, and effective maximum concurrency in metadata and throughput
summary.

- [ ] **Step 4: Run all affected tests and lint**

Run:

```bash
uv run pytest tests/test_tool_gap_loan.py tests/test_simulator_validation.py tests/test_simulate_cloud_model.py -q
uv run ruff check src/trace_collect/tool_gap_loan.py src/trace_collect/openclaw_host_runtime.py src/trace_collect/simulate_openclaw.py src/trace_collect/simulator.py src/trace_collect/simulate_outputs.py src/trace_collect/cli.py src/agents/openclaw/tools/container.py tests/test_tool_gap_loan.py tests/test_simulator_validation.py tests/test_simulate_cloud_model.py
```

- [ ] **Step 5: Independent bounded review**

Reviewer scope: the files listed in Tasks 1-3. Check causal timestamps,
label-free prediction validation, exact action preservation, timer cleanup,
FIFO admission, arm isolation, and output provenance. Fix critical/major
findings and rerun only affected tests.

- [ ] **Step 6: Commit the integrated feature**

```bash
git add -- <only the files listed in Tasks 1-3>
git commit -m "[feat] Schedule predictive tool-gap loans"
```

### Task 4: Smoke, formal ABCCBA, and frozen verdict

**Files:**
- Modify: `analysis/development/tool-resource-canonical-objective.md`
- Create on GO/NO-GO: `analysis/results/predictive-tool-gap-loan-20260810/result.json`

**Interfaces:**
- Consumes the reviewed implementation and frozen task/prediction inputs.
- Produces one machine-readable result and a rewritten Section 5.7 decision.

- [ ] **Step 1: Build the label-free prediction input mechanically**

Filter the frozen SOTA validation rows to the eight registered task IDs and
copy only `sample_id`, `command`, latency PMF, hard bucket, and latency
provenance. Validate 172 prediction rows, exact task order, and exact ordered
subsequence alignment against source exec actions. Do not copy or inspect
`labels`.

- [ ] **Step 2: Run one full-workload Predictor smoke**

Use the frozen eight tasks once so the four-foreground causal budget and one
waiting-task release can activate without a smoke-only rule. Validate IPC,
early/feedback events, timer cancellation, direct LLM requests, exact
actions/tokens, and eBPF. Smoke timing and outcomes are plumbing evidence only.

- [ ] **Step 3: Report measured cost and launch the authorized formal run**

Run six fresh-vLLM cells in exactly this order:

```text
Fixed, Feedback, Predictor, Predictor, Feedback, Fixed
```

Use the same A100-80GB, 250 W cap, model, warm-up, two-core containers, eBPF,
and exact source trajectories in every cell. Monitor progress, OOM, thermal
state, and actual argv without reading effect metrics between cells.

- [ ] **Step 4: Audit validity and apply the frozen gate once**

Verify exact action/tool/token identity, no host request wait, causal `B` and
prediction timestamps, one loan per foreground lender, non-recursive loans,
activation counts, telemetry, OOM, and thermal state. Then compute the paired
and pooled metrics exactly as registered. Do not try another setting.

- [ ] **Step 5: Record, verify, and commit the result**

Write the result JSON and rewrite Section 5.7 with observation, mechanism,
uncertainty, and GO/NO-GO. Validate JSON, run `git diff --check`, stage only the
two result paths, and commit:

```bash
git commit -m "[docs] Record predictive gap loans"
```
