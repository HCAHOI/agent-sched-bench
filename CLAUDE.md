# CLAUDE.md — Python MLSys Research Project

> Authoritative instructions for AI agents working on this codebase.
> All rules here take precedence over default agent behaviors.

---

## Project Overview

This is an **academic research project** in Machine Learning Systems (MLSys).
The goal is to produce **publishable, reproducible, and scientifically rigorous**
results.

**Primary constraints:**
- Results must be obtained through legitimate, generalizable methods
- Code must be production-quality, not prototype/toy implementations
- All experiments must be reproducible with documented configurations

---

## Environment

**Single entry point** — `uv` with a `.venv` at the repo root, via
`bash scripts/setup/benchmark_server.sh`. Produces a Python 3.12 env with
the project deps (pyproject is the spec). Verified ~2 min on a fresh
Ubuntu 22.04 + GPU instance. No conda. After setup, activate with
`source .venv/bin/activate`.

There is no conda flow. Task-container agent runs (swe-rebench / swe-bench)
do not need the host interpreter mounted: each task container bootstraps its
own Python (≥3.11) in-container — see
`src/trace_collect/runtime/task_container.py`.

---

## The Research Loop

**measure → diagnose → improve → ship**, run at maximum iteration speed.
Sequencing discipline — diagnose on data you have, validate the improvement
on data it has not touched — replaces prohibition. Any process step that
slows the loop without changing a decision should be deleted.

---

## Research Integrity & Taste (CRITICAL)

> These principles are NON-NEGOTIABLE. Violating them compromises the
> scientific validity of our work.

### 1. The artifact is the system, not the claim

Measurement exists to find mechanisms and guide the next build. An evaluation
that ends at "works / doesn't work" without answering WHY and WHAT TO CHANGE
is half a result. Mechanism analysis and case studies are the primary product
of an evaluation, not an optional appendix. When a result is negative or flips
between settings, the required response is a diagnosis, not just a verdict.

### 2. No benchmark gaming

- **MUST NOT** add tricks that only work on specific datasets
- **MUST NOT** tune hyperparameters to overfit evaluation benchmarks
- **MUST NOT** cherry-pick evaluation metrics or subsets
- **MUST NOT** use dataset-specific priors disguised as "general" methods
- If a technique requires knowing properties of the test set, it is INVALID

```python
# ❌ FORBIDDEN: Dataset-specific magic numbers
if dataset_name == "HotpotQA":
    threshold = 0.73  # "tuned" to this specific benchmark

# ✅ CORRECT: Generalizable approach
threshold = config.get("threshold", 0.5)  # documented, configurable
```

A method component that must be named by token or dataset is not a method.
If a hand-written rule outperforms, it may serve as an **oracle baseline** or
a **harness positive control** — never as the shipped mechanism. What ships
must be learned or derived from data by a stated, general procedure.

### 3. No hindsight contamination

- **MUST NOT** use information unavailable at inference time
- **MUST NOT** leak ground truth into feature engineering
- **MUST NOT** use "oracle" signals in the workflow (analysis only)
- **MUST NOT** design methods around post-hoc observations of test data

```python
# ❌ FORBIDDEN: Using future/oracle information
def extract_features(query, ground_truth_answer):  # GT leaked!
    similarity_to_answer = compute_sim(query, ground_truth_answer)

# ✅ CORRECT: Only use available information
def extract_features(query, retrieved_context):
    ...
```

**Pre-registration and amendments.** Decision criteria are fixed before the
numbers exist. If a criterion must change after any number — including a
partial, smoke, or subset number — that is an **amendment**: date it, record
what was already visible when it was made, and say so wherever the criterion
is reported. An amendment made openly is legitimate; an amendment described
as a pre-registration is not, even when the statement is literally true.

### 4. Match statistical rigor to the decision being made

Systems effects are often small, heavy-tailed, and cluster-correlated, so some
protection against shipping a harmful change is warranted — but the dose must
match the decision. The right instrument is usually a single uncertainty
estimate at the granularity of the actual deployment decision (e.g. one
cluster-aware interval or test per configuration that would really ship).

Avoid imported ritual from experimental natural science: multiple-comparison
ceremonies over parameter sweeps no deployment faces, protocol lock-in that
forbids learning from your own data, or procedures that convert "consistently
positive everywhere" into "no conclusion." A method that destroys information
is not conservative; it is broken. Explore freely, label exploration honestly,
and validate improvements on data that has not shaped them.

**Accuracy is not utility.** A change that improves an estimator's error
metric can still degrade the decision the estimator feeds. Gate on the
decision, at the operating point, not on the fit statistic.

### 5. Physical honesty outranks inferential ceremony

Measure real constants on real hardware instead of assuming them. Charge every
cost the mechanism actually incurs — nothing is free just because it is
inconvenient to count. Evaluate on real workloads at realistic scale. Report
absolute numbers next to real baselines rather than only relative
improvements. Publish the settings where the method loses. An objective
function that omits a real cost will manufacture success and waste months.

- **MUST** use realistic data scales and distributions
- **MUST** test on held-out data never seen during development
- **MUST** include failure cases and limitations in analysis
- **MUST NOT** introduce mocks, simulations, stubs, or bypasses to avoid
  running real operations — even "temporarily." If a component is slow,
  expensive, or inconvenient, that is not a justification for faking it.
  Plan and wait for approval before implementing.
- Toy examples are for debugging only, never for final evaluation

### 6. Held-out data is a consumable

The moment a verdict is read from held-out data, that data is
development-exposed. Budget freshness like money: reserve untouched data for
the single decision that needs it, know which corpora are already spent, and
never plan experiments requiring data or funds the project does not have.

### 7. No unjustified complexity

- **MUST NOT** add hyperparameters without clear justification
- **MUST NOT** hardcode values that should be configurable
- **MUST NOT** add components "just in case"
- Every design choice must have a documented rationale
- Prefer simple baselines that work over complex methods that barely beat them

```python
# ❌ FORBIDDEN: Unexplained magic
alpha = 0.7823
beta = 1.2 if len(x) > 100 else 0.8

# ✅ CORRECT: Justified and documented
# Alpha controls exploration-exploitation tradeoff (see Section 3.2)
alpha = config.exploration_weight  # Default: 0.5, tuned on validation set
```

### 8. Completeness over shortcuts

- **MUST** implement full pipelines, not hacky shortcuts
- **MUST** handle edge cases properly (empty inputs, missing data, etc.)
- **MUST** preserve all relevant information in data structures
- If something is "too slow," optimize it properly, don't skip it

```python
# ❌ FORBIDDEN: Lossy shortcut
def process_trace(trace):
    return {"score": trace["final_score"]}  # Discards everything else!

# ✅ CORRECT: Preserve information
def process_trace(trace):
    return {
        "score": trace["final_score"],
        "intermediate_steps": trace["steps"],
        "metadata": trace["metadata"],
        "timing": trace["timing"],
    }
```

### 9. Use established tools

- **MUST** use mature, well-tested libraries for standard operations
- **MUST NOT** reimplement standard algorithms without justification
- Agent tracing: established frameworks. Experiment tracking: proper tools.
  Data processing: battle-tested libraries.

### 10. Speak the audience's currency

Systems venues read multipliers and native units: end-to-end latency,
percentile tails, throughput, resource footprint. Robustness mechanisms are
sold as features demonstrated in native units (what breaks without it, and by
how much), not as statistical guarantees. Keep inferential detail to a short
methods note and an appendix.

---

## Runtime, Cost, and Patience

Two rules that look opposed and are not: **never cut scope to save time**, and
**never accept avoidable slowness**. Patience applies to work that is
genuinely expensive. It is not a license to leave a pipeline unprofiled.

### Velocity is a correctness property

Time-to-insight is part of research quality. Before launching any long
computation:

- Estimate wall-clock time and say it out loud; re-estimate when evidence
  contradicts the estimate.
- **Profile before enduring.** If a run is slow, measure where the time goes
  before accepting the duration as inherent. Attribute expected cost to each
  component; any component consuming a large share must justify itself.
- **Parallelize everything provably independent** (folds, shards, corpora,
  configurations). **Sequential-by-default is a bug.** Check
  memory-footprint × concurrency against the machine before launch.
- Instrument long runs with progress signals so "slow" and "stuck" are
  distinguishable without attaching a profiler.
- Optimizations to numerically load-bearing code must be proven
  **byte-identical** on real artifacts, not merely "tests pass." Floating-point
  reassociation changes results; a faster path that alters a tie-break is a
  behavior change requiring its own decision and re-runs.

### Do not alter scope due to runtime

**MUST NOT**, without explicit human approval:

- Cancel a download/process because it's "taking too long"
- Substitute a smaller dataset, model, or subset "to save time"
- Reduce epochs, iterations, or sample size for "quick testing"
- Skip preprocessing steps that seem "expensive"
- Use cached/stale results instead of recomputing
- Switch to a "lighter" alternative

| Operation | Normal Duration |
|-----------|-----------------|
| Dataset download | Minutes to hours |
| Preprocessing/feature extraction | Minutes to hours |
| Model training | Minutes to days |
| Full evaluation pipeline | Minutes to hours |
| Hyperparameter search | Hours to days |

If runtime is genuinely problematic: report the expected duration, explain the
concern, **wait for approval**, and never silently substitute.

---

## Code Quality Standards

### 1. Correctness first

- Correct results before any optimization
- Validate assumptions with assertions or checks
- Edge cases must panic explicitly or be handled by designed fallback, never
  silently ignored
- Type hints required on all function signatures

```python
def compute_metric(predictions: list[float], labels: list[float]) -> float:
    """Compute evaluation metric.

    Raises:
        ValueError: If inputs have mismatched lengths or are empty
    """
    if len(predictions) != len(labels):
        raise ValueError(f"Length mismatch: {len(predictions)} vs {len(labels)}")
    if not predictions:
        raise ValueError("Empty input")
```

### 2. Simplicity

- Generated code must stay simple and readable
- Comments must add value, not repeat the code
- Avoid over-complex abstractions; prefer explicit over implicit
- One function does one thing

### 3. No code duplication

- **MUST NOT** duplicate same/similar logic across files
- Extract to a shared utility instead of copying
- Check whether similar functionality already exists before implementing
- Generalize existing implementations when extending functionality

### 4. Configuration management

All configurable values in config files, not hardcoded. Hierarchical config.
Separate data / model / experiment config. Log full config with every run.

**Every configuration flag has an owner.** Never inherit a template, default,
or previous invocation unexamined. Each parameter on a launched command line
must have a stated reason to be there. Parameters that contradict known
physical reality or a standing decision must be flagged and approved before
launch, even if they look like harmless plumbing.

---

## Agent Trace Standards

When working with LLM agents or multi-step pipelines:

- **MUST** preserve all intermediate outputs, not just final results
- **MUST** log timing information for each step
- **MUST** capture model responses in full (not truncated)
- **MUST** record all metadata (model version, parameters, etc.)

---

## Benchmark Plugin Architecture

All benchmarks MUST be added via the plugin layer in `src/agents/benchmarks/`
and `configs/benchmarks/<slug>.yaml`.

**FORBIDDEN:**
- Hardcoding dataset names (`princeton-nlp/SWE-bench_Verified`,
  `nebius/SWE-rebench`, etc.) in `src/trace_collect/collector.py`,
  `src/trace_collect/cli.py`, or any scaffold module.
- Adding `--harness-dataset` / `--harness-split` / `--harness-namespace`
  or similar "per-benchmark" CLI flags — those belong in the YAML.
- Adding a `from_<benchmark>_instance()` factory method on `EvalTask`.
  The canonical entry point is `EvalTask.from_benchmark_instance(row,
  workspace_base, benchmark=<plugin>)` which delegates to the plugin's
  `normalize_task` for benchmark-specific quirks.

---

## Mandatory Review Gate

Before completing a major module, committing a significant refactor, running
any experiment that produces results for analysis, or touching the evaluation
pipeline, the work **MUST** pass an independent review.

**The gate has three requirements. All are necessary.**

1. **Independent.** The reviewer is spawned by the coordinator, not by the
   author, and is not briefed by the author. An author who commissions and
   frames its own review has not passed the gate — it has selected its own
   examiner. Self-identified findings are useful input; they are not a gate.
2. **Fresh context.** The reviewer did not write the code. Authors develop
   tunnel vision: they are convinced the implementation is correct because
   they wrote it with that intent.
3. **Frozen target.** The author stops writing before the review begins and
   does not resume until findings arrive. Reviewing a moving file produces
   withdrawn findings and wasted round trips.

**The reviewer must check:** correctness (does the logic do what it claims);
research integrity (hindsight leakage, dataset-specific tricks, unjustified
magic numbers, criteria amended after seeing numbers); completeness (fields
preserved, edge cases handled); consistency with existing conventions.

**Input provenance is part of the review, not a precondition of it.** Auditing
the machinery while assuming its inputs is how a confident, wrong result gets
produced. For every corpus or dataset a run reads, confirm against the
committed manifest that defines it: the root path, the task/instance id list,
and the expected count. Confirm no configured root appears in any manifest's
exclusion list — a path taken from `excluded_trace_roots` is dev-exposed data
that was deliberately withheld, and using it silently inverts the project's
central discipline. A pipeline pointed at the wrong data passes every
statistical check ever written for it.

**Iterate until clean:** 🔴 critical or 🟠 major → fix and re-review. 🟡 minor
→ may proceed, but still fix. Log what was reviewed and how issues were
resolved; this is the audit trail.

**Verify claims rather than accepting summaries.** "Tests pass" is not proof of
equivalence; a test covers the inputs its author imagined. When a claim is
load-bearing, check it against the real artifact — regenerate and diff.

**Non-negotiable:** no experiment result is valid if the code that produced it
did not pass this gate. Running experiments on unreviewed code wastes compute
on potentially meaningless results.

---

## Records and Documents

**Rewrite documents; do not append to them.** A specification states what is
true now. Stacking status banners, "OUTCOME" blocks, and amendment notes on
top of a stale body produces a changelog that a reader must diff to
understand, and it is how a directory becomes misleading. When findings change
what a document says, rewrite the document.

History belongs in a document only when the history itself is load-bearing:
an integrity record (a criterion amended after numbers were visible), a defect
that shipped, or a decision whose reasoning constrains future work. Everything
else is noise that costs the next reader time.

- One index that maps the directory; keep it current.
- One file recording settled questions and why each died, so closed questions
  are not silently re-opened.
- Frozen result artifacts are evidence — do not delete them to reduce clutter.
- Superseded plans and design docs for abandoned work should be deleted;
  version control retains them.
- Coined shorthand is not vocabulary. Write plain language in anything a
  collaborator or reviewer will read.

---

## Delegation and Coordination

When work is split across agents:

- **Explicit file ownership.** Each lane owns a stated set of files. An agent
  that needs to edit a file it does not own must surface that, not edit it.
  Two agents reasoning about one file — two writers, or a writer and a reader
  — is the failure mode; concurrent lanes on disjoint files are safe.
- **No lane-to-lane negotiation.** All coordination routes through the
  coordinator, so conflicting reports become a decision rather than a loop.
- **Decide what is yours.** If the answer is derivable from facts already in
  hand, derive it and report the decision. Escalate only what genuinely
  requires the human: spending money, irreversible actions, changes of goal.
  A queue of questions is not delegation; it moves work onto the human.
- **Spawning a lane is not progress.** Progress is a result that passed the
  gate. Count delivered results, not lanes started.

---

## Done Criteria (Pre-Commit Checklist)

### Scope Verification
- [ ] Modified files stay within requested scope
- [ ] No accidental edits to data/artifact directories
- [ ] No changes to unrelated modules

### Functional Verification
- [ ] Run relevant test command(s) — report what was run
- [ ] Verify at least one representative output is generated
- [ ] Check no regressions in existing functionality

### Code Quality
- [ ] Type hints on all new functions
- [ ] No code duplication introduced
- [ ] No hardcoded values that should be configurable
- [ ] Comments are helpful and not excessive

### Research Integrity
- [ ] No benchmark-specific tricks introduced
- [ ] No hindsight/oracle information leakage
- [ ] Criteria unchanged since numbers appeared, or the change is recorded
      as a dated amendment
- [ ] All design choices are justified
- [ ] Generalizable to other datasets/settings

### Documentation
- [ ] Affected documents rewritten, not annotated
- [ ] Changelog updated if behavior changed
- [ ] Docstrings for new public functions
- [ ] Config changes documented

### Commit Message Format
```
[type] Brief description (max 50 chars)

- Detailed point 1
- Detailed point 2

Types: feat, fix, refactor, docs, test, config
```

---

## Agent Behavioral Rules

### DO
- Verify by reading actual source code before making claims
- Check existing code for conventions before asking
- Run the minimal test covering your changes
- Ask for clarification when requirements are ambiguous
- Preserve existing functionality unless explicitly told otherwise
- Make code fail fast

### DO NOT
- Guess or hallucinate about project internals
- Introduce new dependencies without explicit approval
- Modify files outside the scope of current task
- Make "improvements" that weren't requested
- Simplify by removing functionality (simplify implementation, not behavior)
- Add unnecessary exception handling — this is a research project; let code
  fail fast so unaligned behavior surfaces quickly

### When Uncertain
1. First: Check existing code for precedent
2. Second: Check documentation
3. Third: Ask the human explicitly
