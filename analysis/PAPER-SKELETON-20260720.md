# Paper skeleton — three contributions, one object

> **Purpose (project lead, 2026-07-20): "two different modules toward
> the same problem are meaningless."** This file fixes what the paper
> claims BEFORE the harness build, so every remaining experiment
> targets a named claim instead of accumulating results we later have
> to reconcile. Any new lane must map to C1, C2, or C3 — or it does
> not run.

## The single object (the anti-two-modules rule)

**One object:** the certified conditional-residual survival curve
S(remaining | prefix node, elapsed), fit per workload and gated by a
permutation certificate.

**One call lifecycle, two priced decision points on it:**
- `g` — swap-out trigger: when does staying resident stop paying?
- `s` — pre-restore start: when does bringing it back start paying?
- **`s >= g` by construction.** Pre-restore is only defined on calls
  the trigger already swapped. Same node samples, same cross-fit
  discipline, same expected-cost functional, same certificate, same
  hazard/utility code path.

Swap-out and pre-restore are therefore NOT two modules that both
reduce latency; they are a composed action pair over one curve. Any
draft that describes them as parallel components is wrong and must be
rewritten. The composition is the contribution — a second action is
what makes the prior a *primitive* rather than a one-off heuristic.

## C1 (MAIN) — Priced stopping-time policies over certified latency priors

**Claim:** an agent serving stack that prices KV actions against a
certified conditional-residual prior beats deadline and frozen-TTL
policies on real agent workloads under memory pressure.

**Status:** offline components banked, system claim NOT yet earned.
- Swap-out: H1 CERTIFIED on fresh-277 (+66.7s @kv3500, +150.6s
  @kv5000 vs deadline, rho=0.94, BROAD).
- Pre-restore: certified offline under BOTH trigger sources
  (+156.2s / +317.9s per 277 tasks); near-invariant to trigger
  choice, so the gain is tail-overlap structure.
- Mechanism validated on H100 (staged transfer 23 GB/s, pause/resume
  with logit identity, certified trigger wired to pause).

**What C1 still REQUIRES (the critical path):**
1. Multi-tenant harness that demonstrably thrashes (W5-7) — both
   actions executing end-to-end under real memory pressure.
2. Head-to-heads: our policy vs deadline vs Continuum-style TTL vs
   ThunderAgent, >=3 load levels, >=2 workloads.
3. Live pre-restore validation (offline accounting cannot price real
   transfer contention). **Currently behind the W8 cut line — this is
   the single largest schedule risk to the main contribution.**

**Headline form:** seconds saved per task and X.Xx on JCT/P99 TTFT.
Never statistics-forward.

## C2 — Certification decides what conditioning ships (methodological)

**Claim:** conditioning refinements must be gated on DECISION UTILITY
at the operating point, not on estimator accuracy; per-workload
permutation certification is a discipline that does this, and it
demonstrably prevents a real regression.

**Evidence already banked (the kills are the contribution):**
- **WTN**: key normalization improved MAE by 21ms — and was
  directionally HARMFUL on utility (−59.6s @kv3500). The gate caught
  it. This is the paper's proof that accuracy is the wrong target.
- The same gate rejected: atom decomposition (naive and
  argument-conditioned), stable-atom screening (no reachable mass).
- Calibration lane supplies the complement: the shipped prior is
  honest where the policy reads it (PIT/coverage/CRPS skill vs pooled
  and tool-name baselines; decision-band Brier).

**Contrast:** Continuum conditions on a frozen point TTL;
ThunderAgent assumes memoryless decay. Neither certifies its
conditioning per workload, and neither can detect an
accuracy-improving/utility-harming refinement.

## C3 — The policy space is closed (characterization)

**Claim:** the simple shipped policy is not a simplification — within
elapsed-adapted policies it is optimal, and the observable
enrichments available in this setting add nothing measurable.

**Evidence already banked:**
- **Lemma + verification:** any elapsed-only k-check policy collapses
  to a single stopping time; k=2 DP attains the k=1 optimum exactly
  (670 nodes x 10 kv cells, max value gap 0.0 ms, tol 1e-6), and is
  strictly dominated once per-check overhead is priced.
- **Measured flank:** boundary identity (which sub-command finished)
  moves 72.9% of decisions but its log-score gain CI is [−0.41,
  +0.20] — churn without information.
- Structural negative: this corpus cannot speak to multi-tenant
  contention (occupancy is a harness flag, cloud collection had no
  shared KV) — so contention claims may ONLY come from the harness.
- **Self-footprint pricing closed:** kv3500 headroom +8.75 s/277,
  simultaneous CI [−85.00, 101.90], whose upper bound is below the
  frozen +156 s/277 pre-restore bar. This does not close real
  multi-tenant pressure, which only the harness can measure.

**Why this is a contribution and not an absence:** it tells a builder
which enrichments not to implement, each backed by a proof or a
measurement rather than an untested design choice.

## Mapping rule for remaining work

| Work item | Serves | Status |
|---|---|---|
| Multi-tenant harness (W5-7) | C1 | MANDATORY, not started |
| Head-to-heads incl. ThunderAgent/Continuum (W8-9) | C1, C2 | not started |
| Live pre-restore validation | C1 | behind W8 cut line — promote |
| Online gate (W3-4) | C2 | implemented; corpus discipline review/final replay pending |
| Calibration final run | C1, C2 | banked |
| Footprint pricing screen | C3 | DROP; self-footprint axis closed |
| TraceLab replay | C1 external validity | not started |
| Candidate A (parked) | would be C2 | PARKED |

Anything that maps to no column does not run.
