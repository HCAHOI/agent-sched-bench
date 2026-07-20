# Review Record

Date: 2026-07-11

## Independent Pre-Result Review

Reviewer: separate `sab_trace_only_review` sub-agent. The reviewer was
explicitly prohibited from reading real corpus rows, earlier panels, or new
outputs, and made no file edits.

Initial verdict: **NOT CLEAN** because of one protocol blocker and two minor
issues.

1. The draft's empirical success language was not falsifiable: "not
   materially worse" had no frozen metric or tolerance, and one favorable
   operating point could support a post-hoc success interpretation.
2. The protocol said exact ties while implementation used absolute `1e-12`
   numerical tolerance.
3. Synthetic tests did not demonstrate a positive robust guard selectively
   retaining and filtering baseline robust actions, or robust-specific
   `never early`.

Corrections before any real run:

- Removed the binary success declaration. The exposed panel is descriptive
  only, and no favorable point, subset, pooled sum, or tolerance can declare
  success.
- Froze the exact implementation rule: objectives within absolute `1e-12` are
  ties, larger guards win, and `never early` wins a zero tie.
- Added an end-to-end case learning guard `0.03636`, retaining one robust node
  and filtering another, with paired utility and calibration-count checks.
- Added an end-to-end robust-specific null-guard case that converts all robust
  early actions to deadline.

Final verdict: **CLEAN / APPROVE**.

The reviewer additionally checked 500 random synthetic node/parent/task
combinations against an independently expressed reference. Every robust
trigger and normalized score matched. The reviewer confirmed task-OOF and
outer boundaries, strict robust/deadline subpolicy enforcement, calibration
schema, stale-output rejection, provenance coverage, and lack of corpus/task/
command special cases.

## Verification Before Run

- 33 focused utility/offline-probe tests passed in reviewer scope.
- 66 related utility/offline-probe/profiled tests passed in the main scope.
- Ruff check and format passed.
- Bash syntax, Python byte compilation, and `git diff --check` passed.

## Independent Post-Result Audit

The same reviewer received the frozen real outputs only after the complete run
and made no file edits. Technical verdict: **CLEAN**.

- All 57 pre-report result hashes and 45 input hashes passed, and every source
  snapshot byte-matched its hashed live source.
- All 15 decision files preserved every preceding deadline, mean-hazard,
  robust-clock, and point-guard field exactly.
- Every fold and pooled metric for three corpora, ten costs, and five policies
  was independently recomputed from raw decisions with maximum error zero.
- Every gated trigger exactly applies its fold guard and equals either the
  robust candidate or deadline. Robust candidate and score eligibility also
  match on every row.
- All 15 complete four-fold inner-OOF robust calibrations were re-executed from
  original profile rows. Every guard, objective, count, accepted-task
  statistic, and worst-task value exactly matched the stored summary.

The reviewer classified the empirical result as mixed with a **major
generalization limitation**, not an implementation invalidation. The largest
Terminal improvements are task-cluster concentrated; SWE and several
Terminal/ScienceAgentBench points lose useful robust actions; and every robust
calibration has a negative worst accepted OOF task. The approved interpretation
is descriptive only, with no safety, superiority, significance, or universal
transfer claim.
