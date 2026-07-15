# Terminal-Bench within-benchmark frontier — findings (2026-07-15)

First replication of the trie-vs-GBM frontier on a SECOND corpus (the whole
frontier previously rested on SWE-ReBench alone). Terminal-Bench, 83 tasks
with extractable tool-latency samples, folded and evaluated with the frozen
protocol (`scripts/run_benchmark_frontier.py`, commit dfea189). Dev-exposed
corpus -> sensitivity only, not certification; the small task count makes
every cell inconclusive with wide CIs.

## Result: the frontier ordering does NOT replicate

| rho | gated_hazard vs deadline | gated_robust (trie) vs deadline | GBM vs trie |
|---|---|---|---|
| 0.0  | +55.5 s | **-298.2 s** | +353.6 s |
| 0.25 | +40.8 s | **-415.8 s** | +456.5 s |
| 0.5  | +3.9 s  | -9.1 s | +13.0 s |
| 1.0  | (small) | (small) | +13.6 s |

On SWE-ReBench the trie won head-to-head at rho >= 0.5; on Terminal-Bench
the GBM beats the trie at EVERY fraction, and the trie is net-HARMFUL
against the plain deadline at low rho (the LOTO-unanimity gate admits early
fires that misfire on this workload's exec-latency structure). All cells
inconclusive (0 certified), so this is a qualitative point-estimate flip,
not a certified reversal.

## Interpretation

Which estimator wins is a property of the WORKLOAD, not of the method. The
"trie wins at high rho" conclusion is SWE-ReBench-specific and must not be
stated generally. This strengthens two campaign positions: (1) per-
deployment fitting is mandatory (a fixed choice of estimator transfers
badly, consistent with the earlier E2 transfer result); (2) the fresh-
corpus certification must cover more than one workload before any headline.

Directly relevant to the gate-union: its premise is that the two gates fire
on disjoint calls and OR-combining helps. On TB the trie is harmful, so
min(hazard, trie) would inherit the trie's bad early fires — the union may
NOT help here. Running the gate-union re-scoring on these TB decisions is
the immediate next check; if the union degrades on TB, its SWE-ReBench win
is workload-specific and the fresh-corpus pre-registration must treat the
union as a hypothesis, not a default.

## Caveats

83 tasks (small; wide CIs, nothing certified); dev-exposed (sensitivity);
single seed; GBM feature_set=full. A larger fresh corpus is required to say
anything certified about cross-workload estimator choice.
