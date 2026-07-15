# Mechanism analysis: GBM hazard clock vs gated robust trie (2026-07-15)

Per-call decomposition of the two deployed policies on the frozen
swe-rebench-100 merged decisions
(`analysis/tool-time-hazard-model-gbm-full-20260715/rho_*_decisions.jsonl`),
by a read-only analyst agent. Sanity gate passed: the vectorized utility
reproduces `trigger_policy_utility_ms` on sampled rows and the per-rho
totals reproduce the recorded aggregates exactly (gbm +322104/+180342/
+56424/+23604; trie +151835/+119493/+150668/+118038 ms).

## Structural facts

1. **All divergence lives on `exec`.** read/list/edit/write/message
   contribute exactly 0 to both deltas at every rho — neither gate opens
   early there. The contest is 30,030 exec calls.
2. The utility has three regions: short (early fire only loses), band
   (threshold, threshold+kv — early fire gains; the deadline captures only
   overshoot), far tail (delta ≡ 0). The game is band-gain vs
   short-penalty.

## Why each policy wins (Q1)

- **GBM at rho=0** (+170.3 s over trie): fires earlier (median 440 ms
  earlier on both-fire calls, 90% of the time) and more aggressively in
  the band (+998k ms band gain vs trie's +316k), paying more on shorts
  (−676k vs −164k) — a good trade while shorts cost only exposure.
  Divergence concentrates in command-prefix exec nodes, support 8+.
- **Trie at rho=1** (−94.4 s for GBM): the GBM's single all-or-nothing
  margin gate closes almost completely (44 early fires, 0.1%) while the
  trie's LOTO unanimity keeps 4,852 calibrated fires (10.5%) holding
  +169k band gain. The channel is **count of surviving early fires**, not
  trigger timing (26 both-early rows at rho=1 are noise).

## Trigger anatomy (Q2)

The gates fire on largely **disjoint** calls (rho=0: 6,594 vs 7,119 early
fires, only 2,664 shared; rho=1: 44 vs 4,852, 26 shared). The trie's
high-rho advantage is not firing less nor firing later — it is gate
survival under restore cost.

## Crossover rho 0.25→0.5 (Q3)

−155.1 s swing, all exec: 55% from 358 sign-flip rows, 45% broad
magnitude shift; band −229k offset by short +74k; support buckets 8–15
and 16+; top-10 tasks = 49% of the change (RDFLib-2112 alone −38.0 s).
The doubled short-restore penalty pulls the GBM gate in (2,138 → 1,205
fires); the trie holds.

## Selector headroom (Q4)

| rho | best single | per-tool oracle | per-call oracle | **gate union** |
|---|---|---|---|---|
| 0.0  | 322,104 | 322,104 (+0) | 996,784 | **348,489 (+8.2%)** |
| 0.25 | 180,342 | 180,342 (+0) | 536,861 | **232,155 (+28.7%)** |
| 0.5  | 150,668 | 150,668 (+0) | 400,057 | 162,565 (+7.9%) |
| 1.0  | 118,038 | 118,038 (+0) | 187,566 | **135,967 (+15.2%)** |

- Per-tool switching: **exactly zero headroom** (only exec diverges).
- Support/margin-threshold selectors: ≤ best single or +1–2% with tuning
  — no usable signal.
- **Gate union** — fire at min(gbm_trigger, trie_trigger), i.e. when
  either gate opens, no learning, no threshold — beats the better single
  policy at every rho, broadly across tasks (rho=0.25: +37/−6 tasks).
  The intersection (both agree) is terrible, confirming the value is the
  disjointness: the estimators catch complementary long calls.

## Verdict

A learned per-call selector is NOT worth building (no headroom at tool
level, no single-feature signal at call level). **The OR-union of the two
gates IS** — non-tuned, deployable (both triggers computed offline), +8%
to +29% over best-single at every rho. It is the unanimity principle
inverted: unanimity-to-fire hedges too hard across estimator families
(the M=5 ensemble result); disjoint families argue for either-fires.

## Caveats

Ninth analysis on the frozen corpus (sensitivity only). The union totals
carry NO confidence intervals yet — task-clustered bootstrap of the union
is the immediate next step, and fresh-corpus certification is mandatory
before any headline. Union worst-case short-fire exposure is the union of
both gates' short fires (≤584 here; corpus-dependent).
