# Analysis index

This directory separates current direction from frozen evidence. For
tool-resource prediction and scheduling, read these files in order:

1. [`development/tool-resource-canonical-objective.md`](development/tool-resource-canonical-objective.md)
   — current decisions, metrics, evidence boundaries, and frozen gates.
2. [`ROADMAP.md`](ROADMAP.md) — the compact research frontier and decision tree.
3. [`CLAIMS.md`](CLAIMS.md) — retained claims and their supporting evidence.
4. [`CLOSED-QUESTIONS.md`](CLOSED-QUESTIONS.md) — directions that require new
   evidence before reopening.

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

The active direction treats an agent as a long-lived job alternating between
GPU inference and remote CPU tool execution. `ROADMAP.md` defines its staged
frontiers; the canonical objective controls all result interpretation and
launch authorization. Older W5 material remains historical evidence, not the
current roadmap.

Large or deleted historical outputs are recoverable only where
`CLOSED-QUESTIONS.md` or a result receipt names a Git object. Do not assume every
removed local artifact exists in Git history.
