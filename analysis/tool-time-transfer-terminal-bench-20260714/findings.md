# E2 transfer: SWE-ReBench → Terminal-Bench — findings (2026-07-14)

Frozen fitting rules (full swe-rebench-100 profile, frozen manifest config,
guard re-selected on profile only) applied to the Terminal-Bench corpus
(100 traces; 83 tasks contributed extractable tool-latency samples, 2959
call-cost decisions). Both fit and scoring share each restore fraction.
Round-4 review APPROVE. Exposure caveat: this corpus was
development-exposed; results are transfer sensitivity, not certification.

## Result: the priors do not transfer

gated_vs_deadline totals: −13.9 s (rho=0), −86.5 s (0.25), −47.3 s (0.5),
−78.9 s (1.0) — net-negative at every fraction, no certified-positive
fraction (one isolated 1000 ms cell certifies at rho=0.25 against a
negative total; cell noise). The ungated clocks are far worse: mean-hazard
reaches −540.8 s with 4 certified-harmful cells at rho=1.0.

Mechanically, the transferred guard behaves sanely — it throttles firing
hard on the unfamiliar workload (1454 early fires at rho=0 → ~290 at
rho≥0.5, fires-on-short 125 → 23) and keeps losses bounded (no
certified-harmful cells for the gated policy). But the SWE-ReBench
command-prefix/tool latency priors simply mis-rank Terminal-Bench shell
workloads, so what firing remains is net-harmful.

## Interpretation

The generalization claim in its strongest form — priors fitted on one
benchmark transfer to another — fails on the dense shell workload. Two
honest framings survive:

1. **The gate transfers; the priors do not.** Across every experiment in
   this campaign the certification machinery consistently bounds damage
   (here: modest losses where ungated clocks are catastrophic). The
   deployable unit is "conservative deadline + gate", with early triggers
   only where locally fitted evidence supports them.
2. **Deployment-matched fitting.** In practice a serving system profiles
   its own workload history; zero-target-data transfer is the hardest
   case. The open middle ground is few-shot adaptation: how much target
   workload history does the gate need before certified-positive early
   firing returns?

Companion run: ScienceAgentBench transfer
(`analysis/tool-time-transfer-science-agent-bench-20260714/`) is neutral —
the target is tool-sparse (~1.7 calls/task), leaving nothing to gain or
lose (totals within ±13 s, no certified cells for gated).
