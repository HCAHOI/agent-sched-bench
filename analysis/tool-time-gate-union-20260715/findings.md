# Gate-union combiner — findings (2026-07-15)

Task-clustered bootstrap (50k replicates, Bonferroni over the 10-cost
family per cell) of the parameter-free OR-union: fire when either the
gated trie or the gated GBM opens early, at whichever fitted trigger comes
first. Derived re-scoring of the frozen swe-rebench-100 merged decisions
(commit 5717a5c; review APPROVE). Totals reproduce the mechanism
analysis's point estimates exactly.

## Result: the union dominates both parents at every fraction

| rho | vs deadline | vs single gated GBM | vs gated trie |
|---|---|---|---|
| 0.0  | +348.5 s (2 certified) | +26.4 s (1 certified) | +196.7 s (1 certified) |
| 0.25 | +232.2 s (2 certified) | +51.8 s (1 certified) | +112.7 s (1 certified) |
| 0.5  | +162.6 s (2 certified) | +106.1 s (2 certified) | +11.9 s (1 certified) |
| 1.0  | +136.0 s (1 certified) | +112.4 s (2 certified) | +17.9 s (1 certified) |

Zero certified-harmful cells anywhere. This is the first policy in the
campaign with certified-positive cells against the deadline at all four
restore fractions, and it certifiably beats BOTH parent policies at every
fraction. The frontier ("GBM at low rho, trie at high rho") collapses:
the union is simply better across the range, because the two gates fire
on largely disjoint calls and the OR inherits both coverage sets.

Contrast with the failed M=5 unanimity ensemble (−37 to −160 s vs single
GBM): across estimator families, requiring agreement discards
complementary coverage; taking either gate's fire keeps it. Unanimity is
the right principle WITHIN an estimator family (the trie's LOTO gate),
OR is the right principle ACROSS families with disjoint strengths.

## Deployment reading

The union costs nothing extra at runtime — both triggers are computed
offline from the same profile, and the policy is "arm both clocks, act on
whichever rings first." Worst-case exposure is the union of both gates'
short-fires (small here: ≤584 at rho=0, 14 at rho=1.0).

## Caveats (first-order this time)

The OR rule was selected after the mechanism analysis observed gate
disjointness on THIS corpus — these intervals certify the combiner on the
corpus that motivated it, i.e. they carry selection optimism on top of
being the tenth analysis of the frozen collection. The union is therefore
the pre-registered PRIMARY HYPOTHESIS for the fresh-corpus certification,
not a claimable result. Secondary pre-registrations for that corpus:
per-cell certified counts vs deadline, and the disjointness statistic
itself (fraction of early fires unique to one gate).
