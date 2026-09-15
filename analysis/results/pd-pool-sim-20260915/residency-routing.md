# Residency routing in the pool simulator (2026-09-15)

Rule: a step is disaggregated iff its history is **not resident** on the engine that would decode it. The prefill
tier is stateless and pushes the KV to the task's home engine; warm steps prefill their small delta locally.
Neither the published PPD decision engine (append size, predicted output, QPS lookup) nor the 2026-09-11 two-sided
cost router uses residency. `pd_pool_simulation.py --pool residency:M,P`.

Mean JCT ready-to-terminal, mixed56 workload, seeds 0/1/2 (references at seed 0 from `sweep-table.md`):

| Profile | Layout | GPUs | mean JCT min | seed spread | TPOT ms | home cached share | steps disaggregated | tier busy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L40S | mixed:8 | 8 | 57.5 | 0.3 | 129.1 | 0.068 | — | — |
| L40S | pd:5,3 (best PD ratio) | 8 | 54.3 | — | 48.1 | — | 1.00 | 0.98 |
| L40S | twosided:6,2 | 8 | 54.4 | — | 99.0 | 0.050 | 0.438 | 0.99 |
| L40S | **residency:7,1** | 8 | **46.3** | 2.2 | 94.0 | 0.471 | 0.091 | 0.77 |
| L40S | residency:6,2 | 8 | 59.5 | 0.9 | 103.9 | 0.251 | 0.175 | 0.74 |
| L40S | mixed:32 | 32 | 58.0 | 0.1 | 128.9 | 0.068 | — | — |
| L40S | pd:20,12 (best PD ratio) | 32 | 55.7 | — | 47.7 | — | 1.00 | 0.98 |
| L40S | residency:28,4 | 32 | 43.8 | 0.1 | 86.8 | 0.528 | 0.085 | 0.83 |
| L40S | **residency:30,2** | 32 | **39.6** | 1.1 | 81.8 | 0.629 | 0.054 | 0.83 |
| H200 | mixed:8 | 8 | 8.0 | — | 43.2 | 0.968 | — | — |
| H200 | residency:7,1 | 8 | 8.2 | — | 44.5 | 0.968 | 0.023 | 0.03 |
| H200 | residency:6,2 | 8 | 8.5 | — | 46.2 | 0.968 | 0.025 | 0.07 |
| H200 | pd:4,4 | 8 | 9.2 | — | 43.3 | — | 1.00 | 1.00 |
| L40S | residency:1,1 | 2 | 95.3 | — | 112.3 | 0.022 | 0.205 | 0.31 |

> **Status 2026-09-15 18:18 UTC: these numbers are not evidence.** The simulator has been checked against exactly
> one measured routing policy — the 2026-09-11 two-sided cost router — and it predicted 54 min against a measured
> 75.0, optimistic by 28%, because the simulated router reads exact residency and queue state instantly while the
> real one polled snapshots. Residency routing is the same class of mechanism, so the effect below (−19.5% at 8
> GPUs, −31.7% at 32) is the same order as the instrument's known bias in the same direction. Read the table as a
> hypothesis to be measured, not as a result. The verdict language below is retained only to show what was
> pre-registered and what the simulator returned.
> What would make this instrument usable for routing questions: give the router the information the real one had
> (snapshot latency, stale state, no cross-engine residency oracle) and require it to reproduce the measured
> two-sided run within ±10% (75.0 min, TPOT 96.9 ms, 36% of later turns local). Until it passes that, it does not
> answer routing questions.

Pre-registered criterion (PENDING §4.4, written 17:47 UTC before these runs): the best residency ratio must beat
colocated by ≥ 10% at both pool sizes on all three seeds, and the home engines' cached share must rise.
**Met at both sizes.** 8 GPUs: 46.3 against 57.5, −19.5% (worst seed −17.6%). 32 GPUs: 39.6 against 58.0, −31.7%.
Against the best fixed-PD ratio: −15% and −29%. Mechanism confirmed: the home engines' cached share rises from
0.068 to 0.471 (8 GPUs) and 0.629 (32 GPUs); TPOT falls from 129 ms to 94 and 82.

Readings. (1) The win is a feedback loop, not prefill offload: taking the cache misses off the decode engines stops
them flushing their own KV with full 25K-token prefills, residency rises, so fewer steps are cold — 5–9% of steps
end up disaggregated. (2) The tier must be small and busy: 1 GPU in 8 wins, 2 in 8 loses to colocated (59.5), 2 in
32 is the best point tried (1 in 16), 1 in 2 is a disaster (95.3). (3) It is worth nothing where memory is already
sufficient: on the H200 profile, residency is 97%, only 2.3% of steps are cold, the tier idles at 3–7% and
colocated wins (8.0 against 8.2–8.5). The mechanism buys residency when memory cannot be bought.

Limits. Simulation, not measurement: the simulator's calibration gate covers colocated, fixed PD and PPD at two
L40S GPUs, and no measured run of residency routing exists. The engine model is an eager-mode 4B engine whose fixed
35 ms per iteration dominates decode, so none of this transfers to the KV-read-bound frontier regime until the
frontier profile is fitted. The workload has no tool time. One implementation defect was found and fixed before
these runs (home engines were all assigned to one engine because a task's first step is never resident).
