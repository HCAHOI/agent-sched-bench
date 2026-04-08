# BFCL v4 Plugin Iteration v2 — Reviewer Audit Trail

**Branch**: `dev/bfcl-v4-plugin-v2`
**Plan**: `/Users/chiyuh/.claude/plans/logical-exploring-dahl.md` (v2 section)
**Diff range**: `main..HEAD` — 3 commits stacked on PR #8 (merged as `6f61e7a`)
**Reviewer**: `oh-my-claudecode:code-reviewer` opus, fresh context, separate from authoring lane
**Run date**: 2026-04-08

## Initial verdict

**APPROVE** — zero 🔴 Critical / 🟠 Major findings. 7 🟡 Minor findings marked as recommended polish.

## Commits reviewed

```
55e87cf [refactor] Phase 2 — BFCLRunner routes through SessionRunner + custom registry
ee7d188 [feat] Phase 1 — BFCL Tool wrapper + BFCL→JSON-Schema normalizer
9fb33db [refactor] Phase 0 — openclaw AgentLoop + SessionRunner accept custom ToolRegistry
```

## Reviewer-verified invariants

All 12 invariants from the review brief were verified by direct code reading:

| # | Invariant | Verdict |
|---|---|---|
| 1 | Phase 0 zero-behavior-change for default path | ✅ |
| 2 | Replace semantics correctness (no default-tool leak) | ✅ |
| 3 | `MemoryConsolidator.get_tool_definitions` binding ordering | ✅ |
| 4 | Research integrity §2 — ground_truth never reaches the LLM | ✅ |
| 5 | `_normalize_bfcl_schema` recursion correctness | ✅ (minor polish) |
| 6 | `BFCLNoOpTool.validate_params` override justification | ✅ |
| 7 | `max_iterations=1` dispatch semantics | ✅ (minor guard) |
| 8 | Error absorption semantics shift documented and correct | ✅ (debug minor) |
| 9 | Recorder non-reentrancy (per-call closure) | ✅ |
| 10 | Schema-drift guard on irrelevance (v1-C2 fix still in place) | ✅ |
| 11 | Docs §10.3 + §11 accuracy | ✅ (two text nits) |
| 12 | Silent-failure hunt | ✅ (one log addition) |

Reviewer-quoted positive observations:
- "Test density is excellent. 37 new+updated tests for ~250 lines of source code (~1:7)."
- "`(registry, recorder)` API is the right shape. Returning them as a tuple makes the lifetime contract explicit."
- "`scaffold_capabilities.source = 'custom_registry'` sentinel is forward-thinking."
- "The Phase 0 refactor is genuinely zero-behavior-change for SWE-bench."
- "The §11 docs section is unusually high quality for an internal extension point."
- "The plan file's 'Risks' table accurately predicted every subtle issue."
- "The error absorption test explicitly documents the semantic shift in its docstring, including the research-honesty justification."
- "No code duplication introduced. `_ast_match` was preserved verbatim."

## Minor findings and resolutions

| # | Finding | Status | Fix commit |
|---|---|---|---|
| MINOR-1 | `_normalize_bfcl_schema` drops `"type": "float"` with a warning; should map to `"number"` to preserve type info and silence the smoke log noise | **FIXED** | Added `elif raw_type == "float": result["type"] = "number"` branch in `bfcl_tools.py:61-65`. |
| MINOR-2 | `BFCLRunner.__init__` accepts `max_iterations=0` silently, which would leave the recorder permanently empty | **FIXED** | Added `if effective_max_iterations < 1: raise ValueError(...)` guard in `bfcl_runner.py:78-84`. |
| MINOR-3 | **[Most worth acting on]** `EvalResult.error` is `None` on the absorbed-error path, so downstream analysis can't distinguish "wrong answer" from "model crashed" without re-walking the trace file | **FIXED** | Added `_extract_absorbed_llm_error(trace_file)` helper (`bfcl_runner.py:211-244`) that walks the trace once looking for `llm_error` events and lifts the `error_message` field into `EvalResult.error`. Updated `test_run_task_llm_error_yields_score_zero_via_absorbed_error` to assert `"upstream down" in result.error`. |
| MINOR-4 | §10.3 claims "12-20+ records" but observed min is 8 for irrelevance (no tool dispatch) | **FIXED** | Reworded to "8-20+ records depending on tool-call count (irrelevance floors at ~8 with no tool dispatch; tasks with 1+ tool calls produce 11+)". |
| MINOR-5 | §10.3 says "captures dispatched calls in iteration 1" but Python ranges are 0-indexed | **FIXED** | Changed to "in iteration 0 (the only iteration) before the loop's ``for-else`` branch fires". Also added a note that `BFCLRunner.__init__` raises on `max_iterations < 1`. |
| MINOR-6 | `_sum_usage_from_trace` swallows `JSONDecodeError` silently — if the trace writer ever produces malformed lines, token counts would quietly drop | **FIXED** | Added `logger.warning("BFCL: skipping malformed trace line %s:%d (%s)", trace_file, lineno, exc)` inside the except block. Loop now enumerates with line numbers for the warning. |
| MINOR-7 | Phase 0 tests use `except Exception: pass` — brittleness risk if `SessionRunner.run()` ever raises before the metadata header is written | **DEFERRED** | The comment already explains the intent ("the real session may not complete cleanly under a stubbed provider"); narrowing the exception type risks false-negatives under future openclaw refactors. The tests' subsequent `metadata = json.loads(first_line)` line would immediately fail if the header wasn't written, so the silent-pass is bounded. Not worth the churn. |

## Research integrity verification (CLAUDE.md §1-6)

Reviewer explicitly traced every field that crosses from `EvalTask` into the LLM-visible boundary:

- **`task.ground_truth`**: read at EXACTLY two sites — `_ast_match` call (`bfcl_runner.py:280`) and `evaluation_report` dict (`bfcl_runner.py:301`). Both are post-session. **No ground_truth path reaches the prompt, tool description, or registry.**
- **`task.tools`** (function schemas) → `build_bfcl_tool_registry` → `BFCLNoOpTool(spec, recorder)`. Only `name`, `description`, `parameters` are read from the spec. Clean.
- **`task.question`** → `_flatten_single_turn_question` → concatenates ONLY `role == "user"` content. System/assistant turns and structured metadata are not leaked.
- **`task.category`** → used only at scoring time inside `_ast_match` for the irrelevance shortcut. Never sent to the LLM.

Passes CLAUDE.md §2 (no hindsight contamination) cleanly.

## Regression evidence (post-fix)

```
pytest tests/ -q --ignore=tests/test_gantt_smoke.py
269 passed, 2 skipped, 1 warning in 14.04s
```

| state | passed |
|---|---|
| Baseline (post v1 merge, 6f61e7a) | 250 |
| After Phase 0 (9fb33db) | 255 (+5 Phase 0 tests) |
| After Phase 1 (ee7d188) | 269 (+14 Phase 1 tests) |
| After Phase 2 (55e87cf) | 269 (test counts preserved — refactor) |
| After MINOR polish | 269 (same; MINOR-3 test assertion updated in place) |

## Smoke evidence (post-fix)

```
make smoke-bfcl-v4-openclaw SMOKE_N=2
[1/2] DONE irrelevance_0  steps=1 patch=False
[2/2] DONE irrelevance_1  steps=1 patch=False
```

Trace inspector shows:
```
Agents    : irrelevance_0
Scaffold  : openclaw
Steps     : 1
Events    : 8
Tokens    : 1898 (prompt=1794, completion=104)
LLM time  : 3755 ms
```

Per-task trace file: **11 records / 8 events** (vs v1 baseline 3 records / 0 events — 3.7× density increase, 8/0 event delta). `results.jsonl`:

```
irrelevance_0: success=True, resolved=True, n_steps=1, tokens=1898, category=irrelevance
irrelevance_1: success=True, resolved=True, n_steps=1, tokens=2018, category=irrelevance
```

Both tasks correctly scored as `resolved=True` (model abstained on irrelevance). `scaffold_capabilities.source == "custom_registry"` stamped in metadata — auto-derive works.

## Follow-ups deferred to v3 (explicit)

- **Multi-turn BFCL** (`multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`) — requires BFCL's stateful toolkits from `bfcl-eval` + snapshot/reload across eval questions.
- **Memory category** — requires persistent state across invocations within one task.
- **Web-search category** — requires the BFCL-provided search tool mock + answer normalization.
- **Format-sensitivity category** — requires prompt-variation sweeps (different research goal).
- **`bfcl-eval` as an optional dependency** — v2 keeps the in-process `_ast_match`; v3 may delegate to `bfcl-eval.ast_checker` for full fidelity including the `[don't care]` wildcard (v1 MINOR-5 limitation still applies).

## Final verdict

**APPROVE** — v2 is ready for merge to `main`.

Zero blocking findings. All 12 invariants from the review brief are upheld. 6 of 7 minor findings addressed in post-review polish (MINOR-7 intentionally deferred). The v2 architecture is materially better than v1: SessionRunner bypass is gone, BFCL traces now carry the full scheduling event stream (3.7× density per smoke), the extension point is documented and generalizable to future function_call benchmarks.

The reviewer's own closing words:
> "Per CLAUDE.md §'Mandatory Review Gate', this review constitutes the gate for the v2 branch. The author may proceed to PR."
