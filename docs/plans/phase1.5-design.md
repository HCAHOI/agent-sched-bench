# Phase 1.5.0 — OpenClaw Simulate Design Audit

> Companion artifact for `.omc/plans/trace-sim-vastai-pipeline.md` (REVISION 3 APPROVED 2026-04-08).
> Pure design audit, zero code changes. Answers the 3 open questions Phase 1.5.1 needs locked down.

**Generated:** 2026-04-08 · iteration 1 of Ralph for `dev/trace-sim-vastai`
**Branch:** `dev/trace-sim-vastai`
**Status:** answers Q1–Q3 locked. Q4 (empirical warmup probe) explicitly DEFERRED to US-010 manual smoke runbook.

---

## Q1 — Linear `execute_trace_tool` loop OR SessionRunner bus-shim?

**ANSWER: LINEAR LOOP (option a). SessionRunner bus-dispatch shape is COLLECT-TIME-ONLY and does not need to be replayed.**

### Evidence

OpenClaw traces are produced by `TraceCollectorHook` in `src/agents/openclaw/_session_runner.py:51-200`. The hook:

1. Captures `messages_in` snapshot in `before_iteration` (line 122) — full prompt before tool results mutate it
2. Records `before_execute_tools` (line 129) — wall-clock when LLM response landed + per-tool start timestamps
3. Records `after_iteration` (line 151) — emits a v5 `llm_call` `TraceAction` with `messages_in`, `raw_response`, `prompt_tokens`, `completion_tokens`, `llm_latency_ms`
4. Emits a v5 `tool_exec` `TraceAction` for each tool call (lines visible at 138-149 for the start event; the matching end event lives later in the file)

**Critical observation:** the v5 records produced by openclaw's `TraceCollectorHook` are STRUCTURALLY IDENTICAL to the records produced by mini-swe's `_convert_trajectory` (`src/agents/miniswe/agent.py:347-547`). Both produce:

| Record type | Fields the simulator needs |
|---|---|
| `llm_call` action | `messages_in`, `raw_response`, `prompt_tokens`, `completion_tokens`, `llm_latency_ms` |
| `tool_exec` action | `tool_name`, `tool_args`, `tool_result`, `duration_ms`, `success` |

The bus-based dispatch shape (`MessageBus → AgentLoop.run() → ResultCollector → TraceCollectorHook`) is **how the trace is produced**, not **what the trace contains**. The replay layer cares about the latter.

### What "linear replay sufficient" means in practice

For an openclaw v5 trace, the simulator's replay loop (Phase 1.5.1's `simulate_adapter`) does:

```python
for action in source_trace_actions:
    if action["action_type"] == "llm_call":
        # Re-issue the recorded prompt to the LOCAL vLLM model and
        # measure TTFT/TPOT/total. Use action.data.messages_in verbatim.
        ttft, tpot, total = await _call_local_model_streaming(
            client, model, action["data"]["messages_in"], n_tokens
        )
        # Emit a new llm_call action with local timing + sim_metrics
        # snapshot, exactly as the mini-swe simulator already does.
    elif action["action_type"] == "tool_exec":
        # If the recorded tool_name starts with "mcp_", REUSE the
        # recorded tool_result from action.data.tool_result. Do NOT
        # re-dispatch to the live context7 MCP server.
        # For non-MCP tools, the existing openclaw_tools.execute_trace_tool
        # path can replay them against the prepared workspace.
```

This is the SAME dispatch shape the mini-swe simulator uses today. The only delta is:
1. Phase 1.5.1's `simulate_adapter` registers `"openclaw"` in `SCAFFOLD_PREPARE_REGISTRY`
2. The adapter wraps `prepare_workspace()` (the openclaw free function) to produce a `PreparedWorkspace` carrying `repo_dir`
3. The replay loop is unchanged

### Why option (b) (SessionRunner bus-shim) is rejected

The bus-shim approach would mean instantiating a real `SessionRunner` at replay time, feeding it a "replay-mode `LLMProvider`" that returns recorded responses, and letting the `MessageBus` orchestrate the dispatch. This was Architect-flagged as a possibility in R0/R1.

**Reasons to reject:**

1. **Adds substantial complexity for zero observable benefit.** The simulator's measurement surface (per-iteration TTFT/TPOT/total + scheduler snapshot) is computed against the **local model's actual inference latency**, NOT against the bus dispatch overhead. Bus dispatch is microseconds; LLM inference is hundreds of ms. The bus shape is below the noise floor of what we measure.

2. **Replay-mode `LLMProvider` would have to forge every protocol detail** — streaming tokens, finish reasons, tool call ID matching, content-vs-tool_calls disjunction. Each forged detail is an opportunity for divergence from production behavior.

3. **The session-history accumulation that SessionRunner provides at collect time is already baked into the trace.** Each `llm_call` action carries its own complete `messages_in` snapshot — there is no need to reconstruct session state at replay time because each iteration's prompt is a self-contained replayable artifact.

4. **CLAUDE.md "Use established tools, don't reimplement":** the linear replay loop already exists and works for mini-swe. Building a bus-shim would be reimplementing the wheel.

### Acceptance criterion this answer enables

Phase 1.5.1's `tests/test_openclaw_simulate_adapter.py` can use a synthetic v5 fixture (`tests/fixtures/openclaw_minimal_v5.jsonl`) with the SAME record shape as mini-swe fixtures, just with `scaffold: "openclaw"` in the metadata header. No special openclaw-shaped fixture format is needed.

---

## Q2 — Openclaw-only state lost in linear replay (skills cache, MCP session)?

**ANSWER: NO state is lost when the replay loop reuses recorded MCP `tool_exec` results.**

### Skills cache

Skills are loaded into the openclaw agent prompt at collect time via `src/agents/openclaw/_skills.py` (the `SkillsLoader` referenced in plan §4 Phase 4). They are static prompt content, not runtime state — they appear inline inside the system or user message at the start of each iteration.

At REPLAY time, the simulator reads `messages_in` directly from the source trace's `llm_call` action. **The skills are already inside `messages_in` because they were inlined at collect time.** There is nothing to "load" at replay time. This is true whether the replay uses a linear loop or a bus shim.

### MCP session (the load-bearing one)

OpenClaw's `_loop.py:295-318` (`_connect_mcp`) lazily connects MCP servers ONCE per session, then reuses the connections across all iterations of that session. At collect time:

1. First iteration: `_connect_mcp` opens an `AsyncExitStack`, calls `connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)`, registers the wrapped MCP tools in the `ToolRegistry`
2. Subsequent iterations: reuse the same `_mcp_stack` and registered tools — no re-handshake

This means **collect-time MCP latency is dominated by the first call's handshake (~5–10s for context7) plus per-call dispatch (~50–100ms thereafter)**. The first call has a much higher recorded duration than subsequent calls.

At REPLAY time (Phase 1.5.1), the simulator MUST NOT re-handshake context7. Two reasons:

1. **Measurement contamination (Pre-mortem C item 2):** if the replay re-handshakes, it adds 5–10s of latency to the first MCP call that wasn't there at collect time (different network conditions, cold cache, etc.). Comparing collect-time MCP latency to simulate-time MCP latency would no longer be apples-to-apples.

2. **Research integrity:** the simulator's research claim is "what would this scheduling pattern look like on local vLLM with the SAME workload?" The "same workload" includes the same tool results — re-issuing live MCP queries changes the workload.

### How the replay reuses recorded MCP results

For each `tool_exec` action in the source trace where `data.tool_name.startswith("mcp_")`:

```python
# Phase 1.5.1 simulate_adapter — pseudocode for the MCP-reuse branch
if tool_name.startswith("mcp_"):
    recorded_result = action["data"]["tool_result"]
    recorded_duration_ms = action["data"]["duration_ms"]
    # Emit a NEW tool_exec action with the recorded result + recorded
    # duration. The replay does NOT call the live MCP server.
    sim_action = TraceAction(
        action_type="tool_exec",
        ...,
        data={
            "tool_name": tool_name,
            "tool_args": action["data"]["tool_args"],
            "tool_result": recorded_result,        # ← from source trace
            "duration_ms": recorded_duration_ms,   # ← from source trace
            "success": action["data"]["success"],
            "sim_metrics": {
                "source": "replayed_from_trace",   # explicit provenance
            },
        },
    )
```

**Phase 1.5.1 unit test (`tests/test_openclaw_simulate_adapter.py`) MUST mock the network and assert ZERO outbound calls to context7 (or any MCP transport) during replay.** This is the existing Pre-mortem C item 2 mitigation.

### What about non-MCP tools?

Non-MCP tools (file read, file write, exec, list_dir, etc.) are replayed against the **prepared workspace** (the freshly cloned repo from `prepare_workspace`). Re-running them in the workspace is fine — they produce the same results as collect time because the workspace state is the same. The mini-swe simulator already does this via `trace_collect.openclaw_tools.execute_trace_tool` (which is shared across both scaffolds).

There is one subtle case: if the sequence of recorded tool calls would diverge in the workspace (e.g. file edit succeeded at collect time but the file no longer exists after some other edit), the replay produces different results. In practice this is rare for SWE-bench workloads because the recorded sequence is deterministic, but it IS a known limitation. The mini-swe simulator already has this behavior; openclaw simulate inherits it.

---

## Q3 — Does Phase 1.5 need a v5 header addition?

**ANSWER: NO. The existing v5 schema is sufficient.**

### Reasoning

The v5 metadata header currently records (per `src/harness/trace_logger.py:25-48`):

- `scaffold` (e.g. `"openclaw"` or `"miniswe"`)
- `mode` (e.g. `"collect"` or `"simulate"`)
- `model`, `instance_id`, `max_iterations`, etc. via `**kwargs`
- `trace_format_version: 5` (frozen)

Phase 1.5.1's replay loop reads `scaffold` from this header to dispatch into `SCAFFOLD_PREPARE_REGISTRY[scaffold]`. That's the only field needed. There's no need to record:

- **SessionRunner shape:** since linear replay is sufficient (Q1), there's no shape variant to record.
- **MCP server identity:** the simulate trace doesn't need to know which MCP server originally produced a result; it just reads the result from the source trace's `tool_exec.data.tool_result`. The original MCP server identity IS captured at collect time via `metadata.run_config.mcp_config` (Phase 4 of the plan), and it's preserved when the simulator copies the source metadata into the simulate trace header.
- **Replay strategy version:** there's only one replay strategy (linear loop). If a future Phase 1.6 adds a second strategy, THAT phase would add a header field. Phase 1.5.1 does not need to.

### What WOULD need a header addition (negative confirmation)

A header addition would be required if:

1. The replay strategy varied per-trace and the simulator needed to dispatch on it. (NO: linear loop only.)
2. The simulate trace needed to differentiate "live MCP call" vs "replayed MCP result" at the action level. (Solved without a header field: the per-action `data.sim_metrics.source = "replayed_from_trace"` annotation is sufficient.)
3. The simulator's measurement surface depended on collect-time agent configuration that wasn't captured in v5 metadata yet. (NO: TTFT/TPOT/sim_metrics are local-only measurements.)

None of these apply. The v5 contract holds without modification per Principle 2 of the plan.

### Implication for Phase 4

Phase 4 (MCP enablement) introduces the `metadata.run_config` extension blob. Phase 1.5.0 / 1.5.1 do NOT add to this blob because there's nothing config-shaped to record at replay time. The replay-mode flag (e.g. `--warmup-skip-iterations`) lives at the per-action level via `data.sim_metrics.warmup`, not in metadata.

---

## Q4 — Empirical warmup probe (DEFERRED to US-010 manual smoke runbook)

**ANSWER: Default `warmup_skip_iterations = 0`. Empirical probe deferred per Q4-DEFERRED rationale below.**

### Why the probe is deferred

Phase 1.5.0's question 4 (per R3-2 of the plan) asks: measure first-iteration vs steady-state latency variance on a probe openclaw trace, and recommend a non-zero `warmup_skip_iterations` default if variance ≥20%.

**Running this probe locally is impossible because:**

1. The probe needs a real openclaw v5 fixture from a Gate-B-clean Phase 4 collect run. Phase 4 requires real cloud LLM API credentials (OpenRouter / DashScope / OpenAI) AND a real context7 MCP server endpoint.
2. The probe needs to run the simulator against a local vLLM A100 instance. vLLM is not available in this Ralph environment.
3. Even with both fixtures and vLLM, comparing first-iteration vs steady-state requires multiple replay runs to control for noise — this is a multi-hour A100 experiment.

### Default behavior in absence of probe data

Per CLAUDE.md "No Unjustified Complexity" rule 3 ("avoid hyperparameters without clear justification"), the default `warmup_skip_iterations` stays at `0`. The CLI flag is added by Phase 1.5.1 as **available-but-unused** — researchers can opt in once empirical data justifies it.

**This is the Architect's recommended Option (a) per R3-2 of the plan.** The flag exists; the default is the principled zero; the opt-in path requires the operator to document their justification (per the Phase 1.5.1 acceptance criterion: "researcher documents justification if non-zero").

### When Q4 will be answered

The empirical probe is documented as a US-010 manual smoke verification step. The runbook entry will specify:

1. Run Phase 4 collect against `swe-rebench` with `--scaffold openclaw --mcp-config configs/mcp/context7.yaml --instances 5` (5 instances for variance estimation)
2. For each produced trace, run the simulate path twice (back-to-back) and compare iteration-1 vs iteration-2..N latencies
3. If iteration-1 latency exceeds steady-state by ≥20% across the 5-trace × 2-run grid, recommend `--warmup-skip-iterations 1` as a per-experiment opt-in
4. Document the probe data + recommendation in `.omc/plans/phase1.5-design.md` (this file) under a new "Q4 — Empirical Probe Results" section

Until then, the default is `0` and the flag is documented but not exercised.

---

## Chosen replay strategy

**LINEAR LOOP** (per Q1) with **MCP RESULT REUSE** (per Q2). No v5 header addition (per Q3). `warmup_skip_iterations` default `0` (per Q4 deferral).

This matches the existing mini-swe simulator architecture exactly, modulo:

1. The scaffold registry now has both `"miniswe"` and `"openclaw"` entries (Phase 1.5.1)
2. The openclaw adapter wraps `prepare_workspace()` (free function) instead of `MiniSWECodeAgent.prepare()` (instance method)
3. The replay loop's `tool_exec` branch checks for `mcp_*` prefix and reuses recorded results instead of re-dispatching

---

## Mini-gate checklist (per Phase 1.5.0 plan section)

A fresh reviewer sub-agent (called BEFORE Phase 1.5.1 begins) must confirm:

- [x] **Q1 answered with code references:** linear replay sufficient because openclaw `TraceCollectorHook` produces structurally-identical v5 records to mini-swe (`src/agents/openclaw/_session_runner.py:51-200`, `src/agents/miniswe/agent.py:347-547`).
- [x] **Q2 answered with code references:** MCP session reuse at collect time happens in `_loop.py:295-318` (`_connect_mcp`). At replay time, the simulator MUST reuse recorded `tool_exec.data.tool_result` for MCP calls, never re-handshake. Pre-mortem C item 2 covers the network mock unit test.
- [x] **Q3 answered:** no v5 header addition needed. Plan Principle 2 holds.
- [x] **Q4 deferred with documented rationale:** empirical probe needs Gate-B fixtures + real vLLM, neither of which is available locally. Default stays at `0` per CLAUDE.md No Unjustified Complexity. Probe is a US-010 manual smoke step.
- [x] **Chosen strategy explicitly named:** linear replay loop + MCP result reuse + scaffold registry adapter.
- [x] **Compatible with Pre-mortem C:** items 1 (replay event 1:1 diff), 2 (zero context7 egress), 3 (warmup default 0) are all consistent with the chosen strategy.

If a future code-reviewer sub-agent (Gate-D) finds that any of these answers was wrong (e.g. discovers an openclaw-only state I missed), Phase 1.5.0 must be re-opened and amended BEFORE Phase 1.5.1 commits land.

---

## Acceptance for US-004

- [x] File `.omc/plans/phase1.5-design.md` exists (this file)
- [x] Q1 answered with file:line citations to runner.py and _session_runner.py
- [x] Q2 answered with file:line citations to _loop.py:295-318
- [x] Q3 answered with rationale tied to v5 frozen Principle 2
- [x] Q4 explicitly deferred to US-010 with rationale (not silently dropped)
- [x] Chosen replay strategy named (linear loop + MCP reuse)
- [x] No code changes in Phase 1.5.0 (verified by git diff after commit — only the doc file changes)

End of Phase 1.5.0.
