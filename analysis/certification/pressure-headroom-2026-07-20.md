# Footprint-aware pricing headroom screen

> **FINAL - complete corpus**
>
> EXPLORATORY, OFFLINE. Per-call swap price (resident KV footprint) replayed over the frozen manifest (swe-rebench-qwen3.7-max-fresh-seed42-skip150-n277); no GPU. Generated 2026-07-20T17:50:07 (git 90b17cdc4f7b1347d4dac238221ccd9e8fc07314).

**Verdict: DROP**

> **Arm status (read with the verdict).** The footprint-priced arm prices each call at its OWN resident KV footprint at its decision instant. An agent's context is frozen while a tool call runs, so that footprint is known at call start and equals the footprint at the swap instant: the arm consumes NO hindsight and is an ATTAINABLE bound, NOT an oracle and not unshippable. Because the price is known at call start and constant for the call, this is not a test of time-varying pressure -- it is a test of **per-call footprint-aware pricing**, i.e. a directly shippable policy iteration over a certified policy that currently charges ONE fixed `kv_cost_ms` to every call.

> **Multi-tenant contention: STRUCTURAL NEGATIVE on this corpus.** This screen bounds the SELF-FOOTPRINT axis only and does NOT bound the contention axis. The spec's original lambda (replayed concurrent occupancy) does not exist here, and pricing off it was refused. The measurement, verbatim: within-task call overlap is a timestamp artifact (336 adjacent pairs, max 22ms, median 1.2ms, 100% under 50ms, all on back-to-back fast reads -- the agent is strictly sequential); cross-task co-occupancy is exactly the collection harness's worker count (peak simultaneously-active tasks 2, median 2, mean 1.59); the apparent occupancy tail (3..9) is the same tick artifact, those calls averaging ~2ms latency and containing ZERO calls above any headline threshold; and collection ran against a cloud provider, so no shared KV cache existed and no contention could have been recorded. Pricing lambda off that would dress a `--concurrency 2` flag as physics on a near-binary signal and would manufacture a near-zero headroom -- a DROP caused by the corpus lacking pressure variation rather than by the direction lacking value. That is a false negative dressed as an empirical screen, and it is why this artifact reports the contention axis as unscreenable instead of reporting a number for it.

> **A SURVIVE here does NOT license deployment.** This is a screen, not a certificate. A positive headroom triggers a SEPARATE certified decision replay -- paired against the frozen certified policy at rho=0.94, permutation per kv cell, full H1 discipline -- exactly as A2 required its robust-clock re-confirmation. The numbers in this artifact must NEVER be quoted as a certified gain.

> **Criterion provenance (two different things).** The BAR -- DROP/PROCEED against the banked pre-restore effect (156 s/277 at kv3500) -- WAS frozen before code. The three-way POWER RULE (DROP / UNDERPOWERED / PROCEED) was NOT: it is an AMENDMENT dated 2026-07-20, made by the coordinating session with partial smoke numbers already visible, recorded in `analysis/pressure-headroom-design-20260720.md`. Read the verdict with that distinction in mind.

> **Ceiling scope.** The bound is over triggers `<= the panel cell's threshold` -- both arms search that one common domain, which is the SHIPPED policy's own domain. It is not a claim about triggers beyond the deadline.

Criterion: POWER RULE at the headline kv cell (3500) against the banked pre-restore effect (156 s/277), using the task-clustered simultaneous CI (Bonferroni over the full cost family). PROCEED iff the CI LOWER bound exceeds the bar with the permutation CI excluding zero. DROP (direction closed) iff the CI UPPER bound is BELOW the bar. Otherwise UNDERPOWERED: neither the PROCEED nor DROP criterion is met at the frozen evidence discipline, so the direction is NOT closed.

Headline kv3500: headroom **8.75 s/277** vs banked 156 s/277 (exceeds=False, CI excludes zero=False).

Secondary (NON-BINDING, not part of the verdict) kv5000: headroom -31.11 s/277 vs the kv5000 banked figure 317.9 s/277 (exceeds=False, perm=inconclusive). The verdict compares 3500-to-3500; this row is here so the kv5000 question is answered in the artifact rather than inferred.

Power rule: simultaneous CI [-85.00, 101.90] s/277 against the bar 156 s/277 (upper below bar=True, lower above bar=False).

Footprint-aware pricing DROPS and the direction is CLOSED to one future-work sentence. The CI upper bound sits BELOW the banked bar, so the screen genuinely demonstrated the headroom is smaller than what is already held. Read with the axis limitation above -- this is a null for self-footprint pricing, NOT for contention-aware policies.

## Mechanism: how much does the KV footprint actually vary?

This is why the effect could be large at all -- the certified policy charges ONE fixed `kv_cost_ms` across this entire spread.

- Corpus footprint range: 1912 .. 85189 tokens (44.6x spread).
- Within-task growth (max/min footprint), over 277 tasks: median 13.4x, p90 24.0x, max 40.7x.

277 tasks, 13410 calls, rho=0.94, guard 0ms. Permutation: sign-flip, 50000 draws, Bonferroni over 10 costs.

## Headroom = footprint-priced - fixed-price status quo (seconds per 277 tasks)

| kv | headroom s/277 | footprint-priced s | fixed-price s | fixed lambda ms | footprint early fire | fixed early fire | footprint any fire | perm label | simul CI s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | 30.08 | 605.06 | 574.97 | 1910 | 0.1625 | 0.1282 | 0.1962 | positive | [7.52, 53.47] |
| 1000 | 28.67 | 499.95 | 471.28 | 2131 | 0.1182 | 0.0938 | 0.1464 | inconclusive | [-7.54, 68.96] |
| 1500 | 68.30 | 511.45 | 443.16 | 2696 | 0.0963 | 0.0799 | 0.1192 | inconclusive | [3.17, 161.62] |
| 2000 | 37.25 | 587.86 | 550.61 | 4780 | 0.0784 | 0.0653 | 0.1007 | inconclusive | [-32.12, 117.15] |
| 2500 | 29.75 | 518.83 | 489.08 | 4016 | 0.0698 | 0.0600 | 0.0889 | inconclusive | [-42.03, 104.45] |
| 3000 | 11.48 | 452.45 | 440.97 | 4377 | 0.0638 | 0.0518 | 0.0790 | inconclusive | [-64.32, 83.51] |
| 3500 (H) | 8.75 | 352.48 | 343.74 | 5740 | 0.0595 | 0.0496 | 0.0717 | inconclusive | [-85.00, 101.90] |
| 4000 | 35.67 | 318.65 | 282.97 | 8068 | 0.0529 | 0.0450 | 0.0649 | inconclusive | [-52.45, 127.58] |
| 4500 | -26.32 | 117.87 | 144.19 | 6428 | 0.0509 | 0.0440 | 0.0611 | inconclusive | [-208.73, 145.79] |
| 5000 | -31.11 | 205.70 | 236.82 | 7649 | 0.0472 | 0.0400 | 0.0559 | inconclusive | [-205.50, 83.07] |

(H) = headline cell carrying the frozen kill readout. `fixed lambda ms` is the fit-fold-selected constant price, averaged over folds. EARLY fire = a swap strictly before the call's own deadline, i.e. the decisions the policy adds over deadline_only -- that is the rate the headroom scales against. `any fire` additionally counts deadline fires.
