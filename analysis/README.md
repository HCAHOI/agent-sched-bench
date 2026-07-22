# Analysis index

This directory separates authoritative interpretation from frozen evidence. Read
these files in order:

1. [`CLAIMS.md`](CLAIMS.md) — claims supported now, evidence boundaries, and
   explicit non-claims.
2. [`ROADMAP.md`](ROADMAP.md) — future work and go/no-go gates only.
3. [`CLOSED-QUESTIONS.md`](CLOSED-QUESTIONS.md) — compact ledger of directions
   that require new evidence before reopening.

If prose conflicts, this order controls interpretation. Frozen result artifacts
control their reported numbers and provenance.

## Evidence locations

- `certification/` — retained calibration, pre-restore accounting, and
  policy-space adjudication evidence.
- `results/prequential-task-update-20260721/` — retained development-only
  completed-task adaptation screen and original audit sidecars.
- `serving/` — hardware measurements plus inputs for the unfinished W5
  multi-tenant harness.
- `offline/` — optional background only; it is not an authority for current
  status.

## Current boundary

Retained offline evidence is limited to the claims stated in `CLAIMS.md`. The
completed-task update is the only retained adaptive candidate, but its measured
increment is development-only and order-specific.

W5 is still development work. No live multi-tenant result exists. Its configured
prefill profile is missing, and its development trigger table encodes
`rho=1.0` while runtime accounting is configured for the measured `rho=0.94`;
that contract must be resolved before a run.

Large or deleted historical outputs are recoverable only where
`CLOSED-QUESTIONS.md` or a result receipt names a Git object. Do not assume every
removed local artifact exists in Git history.
