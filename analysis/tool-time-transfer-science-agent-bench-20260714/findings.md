# E2 transfer: SWE-ReBench → ScienceAgentBench Verified — findings (2026-07-14)

Frozen fitting rules (full swe-rebench-100 profile) applied to the
ScienceAgentBench Verified corpus (102 tasks, but only 173 tool calls —
~1.7 calls per task; 1730 call-cost decisions). Round-4 review APPROVE.
Exposure caveat: development-exposed target; sensitivity, not
certification.

## Result: neutral — nothing to gain on a tool-sparse workload

gated_vs_deadline totals: +12.8 s (rho=0) shrinking to −1.3 s (rho=1.0),
every cell inconclusive at every fraction. The transferred gate fires
selectively and safely (524 early fires at rho=1.0, only 3 on short
calls), but with so few calls per task there is almost no hideable latency
to win. Ungated mean-hazard is again the worst transfer (−325 s, one
harmful cell at rho=1.0).

## Interpretation

ScienceAgentBench tasks are dominated by a few long scientific program
executions rather than many heterogeneous shell calls, so per-call trigger
scheduling has little leverage; the interesting resource-management
question there is likely about the single long call's internal phases, not
about call-boundary triggers. As a transfer test this corpus mostly
demonstrates the gate's safety (no certified-harmful gated cells), not the
priors' validity — see the Terminal-Bench companion run for the
informative negative result.
