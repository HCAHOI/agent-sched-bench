# Gate-union on Terminal-Bench — findings (2026-07-15)

The gate-union re-scoring applied to the Terminal-Bench within-benchmark
frontier decisions (`analysis/tool-time-frontier-terminal-bench-20260715/`),
testing whether the union's SWE-ReBench dominance is a general property or
workload-specific. Sensitivity only (83 tasks, dev-exposed).

## Result: the union FAILS on Terminal-Bench

| rho | union vs deadline | union vs single GBM | (SWE-ReBench union vs GBM) |
|---|---|---|---|
| 0.0  | -257.0 s | **-312.5 s** | (+26.4 s) |
| 0.25 | -378.9 s | **-419.6 s** | (+51.8 s) |
| 0.5  | -5.1 s   | -9.1 s | (+106.1 s) |
| 1.0  | (small)  | (small) | (+112.4 s) |

The union is WORSE than the single gated GBM at every fraction and
net-negative vs the deadline at low rho — the opposite of its SWE-ReBench
behavior.

## Mechanism: OR inherits a harmful gate

On SWE-ReBench both gates individually beat the deadline (trie +118..+151 s,
GBM +23..+322 s), so OR-combining their disjoint early fires helps. On
Terminal-Bench the trie is net-HARMFUL (-298..-9 s vs deadline), so
`min(hazard, trie)` pulls in the trie's misfiring early triggers and drags
the union below the GBM alone. The OR is unconditional: it includes a gate
whether or not that gate is useful on the workload.

## The principled fix: certified union

The union should OR only the gates that individually pass deadline-
certification on the fitting partition. On Terminal-Bench the trie fails
that test (harmful), so a certified union drops it and reduces to the GBM
(the right answer there); on SWE-ReBench both pass, so it ORs both (the
+136 s winner). This makes the combiner self-certifying per component —
consistent with the campaign's through-line that certification is the
load-bearing mechanism, not the raw estimator.

This converts the union from "a combiner that won on one corpus" into a
falsifiable, deployable rule: include a gate in the OR iff it certifiably
beats the deadline on this deployment's fitting data. The certified union
is the pre-registered PRIMARY hypothesis for the fresh-corpus work; the
naive union is demoted to an ablation.

## Bottom line for the campaign

The naive gate-union's SWE-ReBench dominance does NOT generalize — it was
contingent on both gates being individually useful there. No universal
best policy exists across workloads (consistent with the E2 transfer and
TB frontier results). What generalizes is the PRINCIPLE: fit per
deployment, certify each component, combine only the certified ones.
