# CPU-Idle Short-Null RSS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Evaluate the frozen short-null RSS amendment without changing the
existing conservative result or predictor decisions.

**Architecture:** Extend the existing RSS-safety evaluator with one explicit
source policy. It replaces only eligible clause RSS nulls whose provenance is
`unknown:insufficient_rss_samples` and whose clause latency is below 500 ms
with the Low upper bound of 500 MB. The existing simulator receives the set of
imputed command IDs solely to count unverified overlaps.

**Tech stack:** Python 3.12, dataclasses, existing NumPy replay, pytest, Ruff.

## Global constraints

- The prior `cpu-idle-rss-safety-v1` artifact is immutable.
- Predictor reservations, FCFS order, CPU profiles, and 32 workload orders are
  unchanged.
- Long null, unmatched, invalid, and ambiguous source RSS remain 16,000 MB.
- The amended result is exposed sensitivity evidence and can authorize only a
  controlled short-command memory calibration.
- No new dependency, evaluator framework, predictor rule, or threshold.

---

### Task 1: Provenance-aware short-null source map

**Files:**

- Modify: `scripts/evaluation/evaluate_cpu_idle_rss_safety.py`
- Modify: `tests/test_resource_admission.py`

**Interfaces:**

- Produce `_short_null_command_rows(command_rows, traces) -> (rows,
  imputed_command_ids, metadata)`.
- `rows` remains the existing `dict[(task_id, call_id), CommandRow]` shape.

- [ ] Add a failing test with measured RSS, a 499 ms insufficient-sample null,
  a 500 ms null, and a missing-profile null. Assert that only the 499 ms
  insufficient-sample clause becomes 500 MB.
- [ ] Run the exact new test and confirm the unimplemented helper fails.
- [ ] Implement the helper by reading each existing
  `resource_observations.json`, validating clause-count alignment, and using
  `dataclasses.replace`; do not change `Row` or telemetry schemas.
- [ ] Add a compound test: sequential imputed clauses compose by max and
  concurrent pipeline clauses compose by sum through the existing
  `_reservation` function.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Unverified-overlap accounting and amended gate

**Files:**

- Modify: `src/tool_resource_eval/resource_admission.py`
- Modify: `scripts/evaluation/evaluate_cpu_idle_rss_safety.py`
- Modify: `tests/test_resource_admission.py`

**Interfaces:**

- Add optional `rss_unverified_command_ids: set[str] | None = None` to
  `simulate_idle_backfill`.
- Return `rss_unverified_overlap_events` and
  `rss_unverified_overlap_command_ids`; defaults are zero and empty.
- Add `source_policy: str = "conservative"` to evaluator `run`; accepted values
  are `conservative` and `short-null-low`.

- [ ] Add a failing simulator test where one of two overlapping commands is in
  `rss_unverified_command_ids`; assert one event and both command IDs.
- [ ] Implement the two counters at the existing speculative-start boundary;
  validate that the ID set is a subset of replay commands.
- [ ] Add failing evaluator tests for the amended schema, protocol, exact
  absolute arm gate, and rejection of an unknown source policy.
- [ ] Implement conditional source-row selection, per-arm amended gates, and
  selection of the passing arm with the larger mean completion reduction.
- [ ] Add CLI `--source-policy {conservative,short-null-low}`. Make the clean
  committed-input guard select the corresponding frozen protocol.
- [ ] Run `uv run pytest tests/test_resource_admission.py
  tests/test_early_cpu_reservation.py -q`, Ruff, `py_compile`, and
  `git diff --check`.
- [ ] Obtain one bounded independent review of source provenance, unchanged
  prediction inputs, default behavior, overlap accounting, gates, and
  provenance guards; fix major findings and re-run focused verification.
- [ ] Commit the reviewed implementation with explicit paths.

### Task 3: Single sensitivity result

**Files:**

- Create:
  `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-cpu-idle-short-null-v1/result.json`
- Modify: `analysis/development/tool-resource-canonical-objective.md`

- [ ] Run one order in memory, printing only wall time, peak RSS, schema, and
  integrity status. Stop and diagnose any integrity failure.
- [ ] Confirm a clean worktree and absent output directory, then run the single
  formal command with `--source-policy short-null-low`.
- [ ] Read the result once. Report each arm's utility, confirmed exposures,
  unverified overlaps, gate, and why it passed or failed; do not tune.
- [ ] Rewrite the canonical objective to distinguish the immutable
  conservative result from the amended sensitivity and its physical limit.
- [ ] Validate schema, 32 seeds, committed implementation SHA, integrity, and
  clean JSON; commit the result and canonical objective with explicit paths.
- [ ] Re-run focused tests and confirm the worktree is clean with no evaluator
  process before reporting completion.
