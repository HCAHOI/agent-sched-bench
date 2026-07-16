# Phase 0b — power / MDE sizing for the fresh corpus (2026-07-16)

Pilot-based sample-size analysis so the one-shot fresh-corpus run (the step that
converts every sensitivity result to certified) is not collected underpowered.
Uses the coverage-valid permutation certificate (Phase 0's fix) and the observed
per-task paired-delta distribution of the primary contrast (certified-union vs
deadline, operating-point restore proxy rho=1.0) as the empirical DGP. This is
standard pilot-based sizing — using dev data to size a future study, not method
tuning or leakage. Script: `scripts/analyze_power_mde.py` (500 corpora, 4000
inner sign-flip draws). Review-gated.

## Headline: n=100 re-certifies only the ONE flagship cell, with zero margin

A cell certifies at ~80%+ power iff its observed per-task effect exceeds the
permutation MDE at that n. Per-task MDE (ms/task) vs the observed per-task effect:

| kv cell | observed ms/task | MDE @ n=100 | MDE @ n=200 | power @ n=100 | n for ~80% power |
|---|---|---|---|---|---|
| **4500** (the certifying cell) | 338 | 331 | 314 | **0.95** | ~100 (no margin) |
| 5000 | 496 | 722 | 530 | 0.34 | ~300 |
| 1500 | 135 | 248 | 176 | 0.27 | ~500 |

Power vs n for the flagship 4500 cell: 0.30 (n=50) -> 0.76 (n=75) -> **0.95
(n=100)** -> 1.00 (n>=150). So a fresh 100-task corpus has ~95% power to
re-certify an effect the size of the one we saw at kv=4500 — but its MDE (331
ms/task) essentially EQUALS the observed effect (338 ms/task), so there is no
headroom: a true effect even slightly smaller than observed would fall below
80% power at n=100. And that 0.95 is an UPPER BOUND: kv=4500 is the flagship
precisely because it certified in the pilot, so its observed effect is a
selected, regression-to-the-mean-inflated estimate — the true effect is likely
smaller, pushing real power below 0.95.

## Heavy tails cap what more tasks buy

Doubling n=100 -> 200 barely shrinks the flagship cell's per-task MDE (331 ->
314 ms/task, ~5%), far short of the ~30% a 1/sqrt(n) law would give. Mechanism:
the MDE is floored by the cell's heavy-tailed RESIDUAL NOISE (the MDE
construction holds the mean-subtracted residual fixed and varies only the mean,
so it responds to noise tails, not signal). Signal concentration is a correlated
symptom of the same big tasks and tracks it ordinally: 4500 (top-3 = 68% of
positive mass, MDE improves 5%) is noisier-tailed than 5000 (top-3 = 47%, MDE
722 -> 530) and 1500 (top-3 = 41%, 248 -> 176), which is why the latter cells'
power climbs with n while 4500's is already saturated. So the broader/weaker
cells improve with n but the flagship's detectability is capped by its tails.

## Family-any power is uninformative — pre-register the specific cell

"At least one cost cell certifies" hits 0.96 already at n=50 and pins at 1.00 for
n>=100 — NOT a real power signal. Under heavy tails a single dominant resampled
task pushes some cell's sign-flip p-value to the floor, so family-any
certification is largely single-task dominance, not broad effect. Consequence
for the fresh-run protocol: PRE-REGISTER the specific cost cell(s) / effect being
tested (the certified-union-vs-deadline contrast at the operating kv), and report
the effect-concentration (top-k task share) alongside any certificate. A
"some cell certified" result must not be quoted as support.

## Sizing recommendation

- **Minimal goal (re-certify only the flagship kv=4500 effect):** n=100 is
  adequate (~95% power) but has zero MDE margin and rides on ~3 tasks — fragile;
  a slightly smaller true effect or a corpus missing those task-types would fail.
- **Broader goal (certify more of the cost family):** ~300 tasks lifts a SUBSET
  {kv=5000 to ~0.92, kv=1500 to ~0.75, kv=3000} past ~0.8, but two cells with
  real, sizeable net effects stay underpowered at EVERY feasible n:
  kv=3500 (+8937 ms) = 0.18 at n=300, 0.36 at n=500; kv=4000 (+16548 ms) = 0.29
  at n=300, 0.54 at n=500. Their observed per-task effects (~89 / ~165 ms/task)
  sit far below their per-task MDE, so no feasible n rescues them — "robust
  multi-cell at n~300" would be an overclaim. n~300 buys a specific subset, not
  the family.
- CRUCIAL LIMITATION: these n>100 numbers use resampling WITH REPLACEMENT (only
  100 distinct pilot tasks exist), which biases power cell-dependently and
  cannot be checked against a distinct-task control beyond n=100 (see below), so
  they are optimistic and NOT a reliable size for a multi-cell claim. Reliably
  sizing a broad-family certificate needs a LARGER pilot.
- Regardless of n, report effect concentration with every certificate and
  pre-register the specific cell — heavy-tail concentration is intrinsic to this
  workload, not a fixable sample-size problem.

## With-replacement bias is cell-dependent, NOT conservative

A distinct-task control (draw n DISTINCT pilot tasks, defined only for n<=100)
quantifies the with-replacement bias directly. It is not a uniform safety margin
— it goes both ways:

| cell | n=75 with-rep | n=75 distinct | n=100 with-rep | n=100 distinct |
|---|---|---|---|---|
| 4500 (concentrated) | 0.764 | 0.924 | 0.954 | **1.000** |
| 5000 (diffuse) | 0.294 | 0.128 | 0.342 | **0.000** |
| 1500 (diffuse) | 0.288 | 0.032 | 0.274 | **0.000** |
| 1000 (diffuse) | 0.472 | 0.260 | 0.348 | **0.000** |

With-replacement INFLATES the diffuse cells ~2-15x (duplicated dominant tasks
get independent signs in the null, a configuration impossible in a fresh
distinct corpus) and DEFLATES the concentrated flagship. At n=100 the distinct
draw is the deterministic full pilot: ONLY kv=4500 certifies, every other cell
0.000 — the honest ground truth. CONSEQUENCE: for n>100 (the multi-cell regime)
there is no distinct control possible from a 100-task pilot, and with-replacement
inflates exactly the diffuse cells one would hope to certify. So the n≈200-300
broader-family power above is likely SEVERELY optimistic; the fresh run should
plan on the flagship + P1 contrast only, and treat any additional certified cell
as a bonus, not a sized deliverable. A reliable broad-family size needs a larger
pilot.

## Caveats

Operating proxy rho=1.0 (for measured rho=0.94). The DGP is the SWE-ReBench-100
pilot, so the sizing inherits that corpus's tail shape; a different fresh
workload could differ. MDE bisection holds the observed residual (noise) shape
fixed and varies only the mean — the standard "fix noise, vary signal"
construction.
