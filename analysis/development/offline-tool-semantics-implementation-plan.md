# Offline Tool Semantics Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Compile official tool documentation once into a bounded `ToolSpec`, then test whether its command representation improves causal 5/3/3/3 prediction over the same generic partial-order estimator.

**Architecture:** Keep `ClauseResourceKB` and Task-Aware unchanged. A stdlib-only validator/interpreter maps one clause to a typed feature set or abstains. The existing interaction-poset matcher receives either the existing generic feature builder or this documentation-derived builder. One evaluation script owns label-blind freezing, one-call generation, and causal evaluation so reserved outcomes cannot be opened through a second path.

**Tech Stack:** Python 3.12, stdlib JSON/dataclasses, existing mvdan clause parser, existing `ClauseResourceKB`/interaction-poset evaluator, pytest.

---

## Global constraints

- SQLGlot's 200 valid tasks are exposed development data. Freeze 100 warm-up and 100 scored tasks before generation; run the generated method once and never repair it from those outcomes.
- PennyLane's 32 never-traced tasks are 16 warm-up plus 16 validation. DVC's 65 never-traced tasks are 32 warm-up plus 33 untouched final. Any task directory that already exists is quarantined even if invalid.
- The generator sees only frozen official documentation and selected tool version. It never sees argv, repo/task IDs, traces, outputs, labels, errors, or generated specs.
- One Codex call per tool, `gpt-5.6-sol`, requested fast tier, medium reasoning, temperature recorded as unsupported, at most 64,000 input tokens and 64 KiB response. No repair call.
- Fixed tools are pip install, pytest, git, and make. Label-blind support requires 50 parsed invocations across 10 tasks; a miss means unsupported, not replacement.
- Generic and docs arms share exact matching, maximal-intersection matching, alpha, public backoff, causal task-final updates, and physical compound composition. Unsupported docs commands copy Clause-KB PMFs bit-for-bit.
- No collection, EAR change, runtime service, or scheduler is part of this plan. A collection estimated above 30 minutes stops after smoke and reports cost for approval.

### Task 1: Freeze task identity and label-blind coverage

**Files:**
- Create: `scripts/evaluation/evaluate_doc_tool_semantics.py`
- Create: `tests/test_doc_tool_semantics.py`
- Create: `analysis/development/offline-tool-semantics-splits.json`

1. Add a failing test with a tiny manifest and fake trace tree proving that an existing task directory is quarantined without reading its contents, ordering is `(created_at, instance_id)`, split IDs are disjoint, and raw commands/resource fields are absent from the output.
2. Run `uv run pytest tests/test_doc_tool_semantics.py -q`; verify the import or behavior fails.
3. Implement `freeze` using `tasks.json`, explicit valid SQLGlot roots, and directory names only for quarantine. Sort every cohort by `(created_at, instance_id)` before slicing. Project `tool_calls.json` to `exec` command strings only after validating the four top-level collection statuses; parse clauses and emit only task/tool/version/count summaries.
4. Freeze exact SQLGlot/PennyLane/DVC IDs and verify counts 100+100, 16+16, and 32+33, full disjointness, and no raw command/output/resource values.
5. Re-run the focused test and execute `freeze` into the committed split artifact.
6. Commit as `[docs] Freeze tool semantics cohorts`.

### Task 2: Validate and interpret the bounded ToolSpec

**Files:**
- Create: `src/tool_resource/tool_spec.py`
- Create: `tests/test_tool_spec.py`

1. Add failing table-driven tests for invocation aliases (`pytest` and `python -m pytest`), longest operation selector, option alias/value arity, unordered work items, fixed-value equivalence, literal-delimiter scope prefixes, requires/excludes, and fail-closed unknown option/version. Add adversarial cases for extra keys, duplicate aliases, dangling references, cycles/conflicts, over-limit strings/collections/64-KiB JSON, and forbidden regex/code/resource/action fields.
2. Run `uv run pytest tests/test_tool_spec.py -q`; verify RED.
3. Implement the smallest stdlib dataclasses and `validate_tool_spec(value)`, `tool_spec_schema()`, and `interpret_argv(spec, bin, argv, observed_version)` APIs. Bounds are fixed in code and schema; parsing is a single linear argv scan with no regex from the spec.
4. Return a canonical tool scope plus a `frozenset[str]` of operation, role, canonical option, work-item, and scope-prefix facts. Invalid/unknown input returns `None`; it never guesses option arity.
5. Re-run the focused test; commit as `[feat] Interpret bounded tool specs` after the required independent module review.

### Task 3: Share the existing partial-order estimator

**Files:**
- Modify: `scripts/evaluation/evaluate_clause_latency_buckets.py`
- Modify: `tests/test_clause_latency_bucket_evaluation.py`

1. Add a failing regression test proving the default feature builder preserves existing generic matches, while an injected builder can canonicalize `pytest` and `python -m pytest` into the same non-exact scope.
2. Run only that test and verify RED.
3. Parameterize `_InteractionPosetKB` with a feature builder. Partition non-exact nodes by the returned canonical scope, not raw binary; retain the raw exact shortcut. Do not change generic defaults or its serialized results.
4. Re-run the focused tests and one existing interaction-evaluator test; commit as `[refactor] Share poset feature builder` after independent review.

### Task 4: Run the canonical four-target comparison

**Files:**
- Modify: `scripts/evaluation/evaluate_doc_tool_semantics.py`
- Modify: `tests/test_doc_tool_semantics.py`

1. Add a failing synthetic causal-stream test that checks: same command IDs/labels/availability for every arm; task-final visibility; docs fallback PMFs bit-identical to Clause-KB; and sequential/pipeline composition for latency, CPU, RSS, and Disk.
2. Implement the causal loop over existing `load_run_rows` rows. Reuse `_InteractionPosetKB`, weighted empirical draws, canonical bucket edges, current metric helpers, `tool_time.command.command_prefix_keys`, and the already selected Task-Aware components (`evaluate_semantic_work_units`, full-test phase, exact-command Disk). Fit every history on the same warm-up tasks and update only after whole-task settlement. Do not copy the old Heavy/Light resource evaluator.
3. Report Majority, raw whole-command prefix, Clause-KB, Task-Aware, generic poset, and docs poset; per-tool and macro equal-target accuracy; severe underprediction; helpful/harmful changes; task-cluster bootstrap; and available 5/10/20/40 checkpoints. The raw-prefix arm chooses the most specific observed `command_prefix_keys` node and otherwise uses the same public fallback.
4. Run focused evaluator tests and a SQLGlot plumbing subset without reading aggregate metrics. Obtain independent leakage/validity review, fix important findings, then commit `[feat] Evaluate docs-derived semantics`.

### Task 5: Freeze sources, generate once, and apply gates

**Files:**
- Create: `analysis/development/offline-tool-semantics-docs/` official source snapshots and source manifest
- Create: `analysis/results/offline-tool-semantics-sqlglot-v1/` generated specs, provenance, result, and rows

1. Snapshot official selected-version docs/`--help`, record source URL or image/probe identity, bytes, model/Codex version, prompt/schema, requested tier, token usage, wall time, and unsupported temperature.
2. Before generation, independently review split IDs, source isolation, prompt/schema, cost bounds, comparisons, and GO/NO-GO code. Commit the freeze; this commit is the preregistration identity.
3. Make exactly one structured call per supported tool. Structural or coverage failure marks that tool unsupported; do not repair or substitute.
4. Evaluate once on the frozen SQLGlot development split. Continue only if its predeclared development gate passes; otherwise commit the complete negative artifact and stop.
5. If it passes, run one PennyLane collection smoke with pre-agent binary-version probes and eBPF, profile wall/RSS/disk, estimate the full 32-task cost, and stop for approval when the estimate exceeds 30 minutes.

## Acceptance

- Focused tests pass and generic-poset regression is unchanged.
- Split artifact contains exact IDs and no outcome-bearing values.
- Generation provenance proves one tool-free call per tool within both budgets.
- No validation/final result is opened before its gate; every unsupported or invalid path falls back without changing eligible rows.
- Each completed phase has one scoped commit; the worktree is clean at handoff.
