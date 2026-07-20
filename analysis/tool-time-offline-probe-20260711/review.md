# Review Record

Date: 2026-07-11

## Pre-Result Independent Review

Reviewer: separate `sab_trace_only_review` sub-agent with no result access.

Initial major findings:

1. The CV aggregator did not prove exact fold/task/cost completeness and could
   accept stale output.
2. The provenance snapshot omitted transitive evaluation dependencies.

Corrections before any real result was generated:

- Require explicit fold count and exactly `f1..fN`.
- Reject stale or missing files and cross-fold task overlap.
- Require the full sample-by-cost panel and common configuration.
- Recompute every fold point from raw decisions and compare it to the stored
  summary.
- Validate guard nullability, accepted counts, task counts, calibration field
  completeness, and profile/probe count relationships.
- Snapshot all direct evaluation dependencies and record their hashes.
- Add adversarial tests for duplicate tasks, incomplete panels, stale folds,
  summary mismatch, missing calibration fields, impossible guard semantics,
  and inconsistent probe counts.

Final pre-result verdict: **CLEAN / APPROVE**. The reviewer explicitly did not
read real panel outputs before approval.

## Verification Before Run

- `27 passed` for `tests/test_tool_latency_offline_probe.py`
- Ruff check and format check passed.
- Bash syntax and Python byte compilation passed.
- `git diff --check` passed.

## Post-Result Audit

Reviewer: the same independent reviewer, now given access to the frozen real
outputs. No files were edited by the reviewer.

Technical verdict: **CLEAN**.

- All 55 pre-document result hashes and 44 input hashes passed.
- All 15 decision files exactly preserved the deadline, mean-hazard, and
  robust-clock baseline decisions from the preceding approved sweep.
- Every fold and pooled metric for three corpora, ten costs, and four policies
  was independently recomputed from raw decisions with zero error.
- Terminal fold 1's four inner probes were reconstructed from the original
  profile panel and exactly reproduced guard `0.2572612071278119`, objective
  `16.33941338156897`, and every calibration aggregate.

Scientific finding: **MAJOR METHOD LIMITATION**, not an implementation or
provenance invalidation.

- Terminal fold 1 alone causes the pooled 3,000--4,500 ms failure.
- `download-youtube` contributes 94.5--96.5% of those losses.
- Its 79 repeated `exec:export` calls are scored by a profile node containing
  six calls from only two tasks.
- The inner probe already observes `-1.796` normalized utility from accepted
  `exec:export` decisions, but the global call-total objective accepts them
  because positive utility elsewhere dominates.

The accepted interpretation is therefore limited to a leakage-free negative
experiment: the scalar point-estimate guard adapts to deployment samples but
does not robustly transfer across task-cluster shifts. It does not establish
universal transfer, safety, statistical significance, or superiority to the
robust clock. Dataset-, task-, command-, and exposed-threshold-specific patches
are prohibited.

## Documentation Review

The first results draft incorrectly labeled candidate-assigned rows as actual
early fires in the Terminal fold 1 case study and had three one-millisecond
rounding errors. The independent reviewer caught both before finalization. The
report now distinguishes candidate assignment from calls that survive to the
trigger, uses raw policy fire counts, and states outer-fold heterogeneity. A
correction-only rereview returned **CLEAN**.
