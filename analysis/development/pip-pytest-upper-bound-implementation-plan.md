# pip/pytest Upper-Bound Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether perfect routing of existing pip/pytest semantic candidates, or even perfect pip/pytest prediction, can materially improve the frozen Task-Aware predictor.

**Architecture:** One evaluator reuses the frozen SQLGlot loader and existing expert evaluators, joins their causal rows by sample ID, and constructs two hindsight-only oracle arms. One focused test fixes selection and identity semantics; no parser, matcher, framework, dependency, or trace collection is added.

**Tech Stack:** Python 3.12, existing evaluation modules, pytest, stdlib JSON/random.

## Global Constraints

- Follow `analysis/development/pip-pytest-upper-bound-protocol.md` exactly.
- Use the exposed 100-warmup/100-scored SQLGlot cohort and 5/3/3/3 targets.
- Task-Aware is the baseline; non-tool rows must remain bit-identical.
- Hindsight may select an oracle bucket only after the candidate pool is built.
- Do not collect traces, tune rules, or add dependencies.

---

### Task 1: Upper-bound evaluator

**Files:**
- Create: `scripts/evaluation/evaluate_tool_modeling_upper_bound.py`
- Create: `tests/test_tool_modeling_upper_bound.py`

**Interfaces:**
- Consumes: the frozen docs-v1 Task-Aware rows, `_load_development_stream()`, `run_pytest_overlap()`, and `run_semantic_work_units()`.
- Produces: `build_oracle_rows(...) -> list[dict[str, Any]]` and a CLI writing `result.json` plus `rows.jsonl`.

- [ ] **Step 1: Write the failing focused tests**

Cover these exact invariants with small in-memory rows:

```python
assert static_oracle["prediction"]["latency"] == truth
assert static_oracle["probability_by_bucket"]["latency"] == existing_candidate_pmf
assert perfect_tool["prediction"]["latency"] == truth
assert non_tool["arms"]["static_candidate_oracle"] == non_tool["arms"]["task_aware"]
assert first_task_source == "current"
assert later_task_source == "pytest_target_overlap"
```

- [ ] **Step 2: Confirm the tests fail because the evaluator does not exist**

Run: `uv run pytest -q tests/test_tool_modeling_upper_bound.py`

Expected: import failure for `evaluate_tool_modeling_upper_bound`.

- [ ] **Step 3: Implement the minimum evaluator**

Use existing parsers to classify pip/pytest commands. Join existing arm PMFs by
`sample_id`; reject mismatched IDs, labels, availability, or order. Reuse the
frozen Task-Aware artifact and verify it against the command stream. Construct
the static oracle by choosing an already materialized candidate PMF whose hard
bucket equals truth, otherwise retain Task-Aware. Construct the perfect-tool
oracle as a one-hot PMF only for commands containing a parsed pip/pytest
clause. Compute existing per-target metrics, equal-weight macro accuracy,
task-cluster bootstrap, coverage, per-tool marginal corrections, and the frozen
decision. Exact memory must accept provenance `exact` only; every other source
fails closed.

- [ ] **Step 4: Run focused tests and a one-task plumbing smoke**

Run:

```bash
uv run pytest -q tests/test_tool_modeling_upper_bound.py tests/test_doc_tool_semantics.py tests/test_semantic_work_units.py tests/test_pytest_target_overlap.py
uv run python scripts/evaluation/evaluate_tool_modeling_upper_bound.py --profile-scored-tasks 1
```

Expected: tests pass; smoke reports one scored task without writing evidence.

- [ ] **Step 5: Obtain bounded independent review**

Review only the two new files against the frozen protocol, checking label
leakage boundaries, causal task updates, candidate provenance, row identity,
metric denominators, and decision logic. Fix critical/major findings and rerun
the focused tests.

- [ ] **Step 6: Commit the reviewed evaluator**

```bash
git add -- scripts/evaluation/evaluate_tool_modeling_upper_bound.py tests/test_tool_modeling_upper_bound.py analysis/development/pip-pytest-upper-bound-implementation-plan.md
git commit -m "[feat] Measure tool modeling ceiling"
```

### Task 2: Frozen development result

**Files:**
- Create: `analysis/results/pip-pytest-upper-bound-sqlglot-v1/result.json`
- Create: `analysis/results/pip-pytest-upper-bound-sqlglot-v1/rows.jsonl`
- Modify: `analysis/development/tool-resource-canonical-objective.md`

**Interfaces:**
- Consumes: the reviewed CLI from Task 1 and the frozen protocol.
- Produces: one decision artifact and the canonical current decision.

- [ ] **Step 1: Estimate cost without reading aggregate outcomes**

Use the one-task smoke duration to estimate the 100-task run. If the estimate
exceeds 30 minutes, report it and wait; otherwise continue.

- [ ] **Step 2: Run the frozen evaluation once**

```bash
uv run python scripts/evaluation/evaluate_tool_modeling_upper_bound.py \
  --out-dir analysis/results/pip-pytest-upper-bound-sqlglot-v1
```

- [ ] **Step 3: Validate and interpret the artifact**

Check 100 scored tasks, row/label/availability identity, non-tool identity,
coverage gates, all four absolute accuracies and gains, bootstrap intervals,
affected tasks, and the protocol decision table. Diagnose which tools,
targets, and candidate sources contain or lack headroom.

- [ ] **Step 4: Rewrite the canonical objective with the settled decision**

Record evidence separately from inference. Close only the family authorized by
the protocol; do not claim a fresh validation result or delete retained code.

- [ ] **Step 5: Verify and commit the result**

```bash
git diff --check
git add -- analysis/results/pip-pytest-upper-bound-sqlglot-v1 \
  analysis/development/tool-resource-canonical-objective.md
git commit -m "[docs] Record tool modeling ceiling"
git status --short
```

Expected: clean worktree and two coherent commits after the protocol commit.
