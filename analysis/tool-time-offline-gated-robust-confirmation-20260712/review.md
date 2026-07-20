# Confirmation Pipeline Review

Date: 2026-07-12

Status: phase-1 pipeline review complete. No real confirmation data has been
supplied or evaluated.

## Initial Review

Independent reviewer: `sab_trace_only_review`. The reviewer was prohibited
from using exposed or smoke traces as confirmation data and made no file edits.

Initial verdict: **NOT APPROVED**, with two major and two minor findings:

1. Missing trace-metadata `instance_id` could silently fall back to trace-path
   clusters, allowing one logical task to cross folds.
2. A standalone analyzer could bypass the frozen manifest with arbitrary fold,
   cost, replicate, confidence, and seed settings.
3. Input hashing occurred after trace extraction, leaving a TOCTOU gap.
4. Python equality allowed boolean aliases for numeric manifest fields.

## Corrections

- The runner now requires a non-empty canonical metadata `instance_id` on
  every trace and verifies every extracted row against that explicit mapping.
- The standalone analyzer and its non-protocol test were deleted. The
  manifest-gated runner is the only confirmation entry point.
- Manifest, task list, protocol, sources, fresh traces, and required
  development traces are hashed before trace parsing and reverified after all
  evaluation, before result hashes are written.
- Manifest schema types are checked explicitly; bool/int/float aliases fail.
- All three development roots are mandatory exclusions. Their trace contents
  are hashed, and byte-identical copies under new paths are rejected.
- Tests now cover the ten-cost Bonferroni tail, positive/harmful/zero labels,
  explicit logical-task metadata, copied development traces, frozen config
  types, input mutation, repeated-call clustering, and fold-level outputs.

## Final Phase-1 Verdict

**CLEAN / APPROVE**.

Verification:

- 82 related tests passed.
- Ruff and Python byte compilation passed.
- A complete five-task, five-outer-fold, ten-cost test path reached policy
  evaluation, bootstrap, provenance, and final result hashes.
- A concurrent protocol mutation correctly aborted at final input-hash
  verification; the stable rerun passed.

Phase 2 remains mandatory after a real manifest is supplied. It must inspect
collection lineage, complete task universe, actual metadata `instance_id`
values, content/task overlap beyond byte-identical files, and the proposed
output path before authorizing the confirmation run.
