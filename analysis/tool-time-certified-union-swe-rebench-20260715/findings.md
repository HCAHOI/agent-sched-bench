# Certified union — findings (2026-07-15)

The principled fix for the naive gate-union's cross-workload failure: OR
only gates that certifiably beat the deadline on the fitting partition,
inclusion decided cross-fitted leave-fold-out (fold f scored under a rule
certified only from !=f rows; review-verified leak-free, commit after this).
Two inclusion criteria: loo_point (leave-fold-out total delta > 0) and
loo_lcb (task-clustered bootstrap lower bound > 0). Run on both corpora.
Companion dirs: tool-time-certified-union-terminal-bench-20260715 (loo_point
on TB), tool-time-certified-union-tb-lcb-20260715 (loo_lcb on TB).

## SWE-ReBench (100 tasks, adequate power): certified == naive == winner

Both gates certify at all 5 folds and every rho, so the certified union
equals the naive union exactly (vs_naive = +0.0 everywhere) and keeps its
dominance: +348.5/+232.2/+162.6/+136.0 s vs deadline (2/2/2/1 certified),
+26/+52/+106/+112 s vs the single GBM, +197/+113/+12/+18 s vs the trie.
The certification does NOT break the good case — when both components earn
inclusion, it recovers the union's win.

## Terminal-Bench (83 tasks, underpowered): the two criteria bracket it

| criterion | inclusion | vs deadline | vs naive union | vs single GBM |
|---|---|---|---|---|
| loo_point | trie leaks into 1-of-5 folds | -251/-381/-12/0 s | +6/-2/-7/+38 s | -306/-422/-16/+12 s |
| loo_lcb | NEITHER gate certifies (all rho) -> deadline | +0 (is deadline) | **+257/+379/+5/+38 s** | -55/-41/-4/+12 s |

- **loo_point is too noisy** on 16-tasks-per-fold: the harmful trie's
  leave-fold-out delta flips sign across folds, so it sneaks into a fold
  and the union nearly matches the naive union's failure (-306 s vs GBM).
- **loo_lcb reverts everything to the deadline**: the bootstrap can't clear
  zero for EITHER gate on 83 tasks (not even the GBM, which was +55 s vs
  deadline but uncertified). This is SAFE — it beats the naive union's
  catastrophe by +257 to +379 s — at the cost of forgoing the GBM's
  uncertified gains (-55 s vs GBM at rho=0).

## The honest conclusion

The certified union with a conservative criterion is the deployable robust
choice: **it never has the naive union's catastrophic cross-workload
failure**, because it reverts to the conservative deadline whenever
certification fails. That is the campaign thesis realized end to end —
certify each component, fire only certified ones, fall back to the deadline
otherwise. Its only cost is conservatism on underpowered corpora, where it
leaves uncertified gains on the table.

Crucially, that cost is a DATA problem, not a method problem: on
SWE-ReBench (adequate power) it certifies and wins; on TB (too small) it
safely abstains. This is the single sharpest argument for the fresh-corpus
collection — a certification-based method needs a corpus large enough to
certify, and the payoff is a policy that is both winning AND
catastrophe-proof across workloads. Pre-registration for the fresh corpus:
certified union (loo_lcb) vs deadline / GBM / trie / naive union, with the
naive union as the negative control.

## Caveats

Both corpora are development-exposed (sensitivity, not certification); the
certified-union rule family was selected after observing these corpora, so
the cross-fitted inclusion still lives on the same data that motivated it.
Fresh-corpus certification remains mandatory.
