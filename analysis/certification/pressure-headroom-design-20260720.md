# Design spec — footprint-pricing headroom screen

> Formerly "λ-oracle pressure headroom screen". Retitled because both
> the λ source and the arm semantics changed during the lane; the old
> title described neither.

> **STATUS (2026-07-20): COMPLETE.** The reviewed full-corpus run produced
> **DROP**: headline kv3500 headroom 8.75 s/277, simultaneous CI
> [-85.00, 101.90] s/277, whose upper bound is below the frozen 156 s/277 bar.
> The self-footprint pricing direction is closed; the contention axis remains
> unscreenable on this corpus.
>
> **Mandatory review audit.** The first whole-diff review found tool-prior cache
> aliasing and an under-pinned final manifest; both were fixed with regressions.
> Re-review found contradictory POWER RULE provenance/UNDERPOWERED wording; all
> occurrences were corrected, then re-read clean with no remaining
> critical/major/minor findings before `--final`.

## Defect history (retained deliberately — this lane shipped two)

**Blocker 1 — the ceiling guarantee was FALSE. FIXED 2026-07-20.**
`hazard_recheck_ms` confines its trigger search to `k <= believed
price`, so a HIGHER believed price bought a LARGER search domain. When
the fixed λ̄ exceeded the per-call λ_i, the fixed-price arm reached
triggers the footprint arm structurally could not, and could beat it
at the true price — violated on 9.2% of random nodes (worst 261
ms/call), with λ̄ > λ_i on 87.8% of real calls at kv3500. "Headroom"
was therefore not a bound and a negative value was confounded with a
search-domain artifact.
*Fix:* both arms now optimize the same functional over a COMMON domain
`[0, kv_cost_ms + guard_ms]` (the panel cell's threshold); the believed
price enters only the objective, never the domain. *Independent
verification* (reviewer's own seed, 4000 nodes, three distribution
families): 0 ceiling violations, 0 triggers outside the common domain,
and **0.000000 ms shortfall against a dense 20,001-point brute-force
scan of the continuum** — i.e. the arm is an exact maximizer, not
merely best-on-its-own-grid. Probe power confirmed by reintroducing the
bug: 328/4000 violations. The bound is scoped to triggers `<=` the
panel threshold, which is the shipped policy's own domain.

**Blocker 2 — the kill criterion was amended mid-flight (process; the
coordinating session's error).** This spec shipped with a two-way
criterion (DROP if headroom below the banked bar). On 2026-07-20 the
coordinating session issued a three-way POWER RULE adding an
UNDERPOWERED state and requiring the CI upper bound below the bar for
DROP. The statistics are sound — non-inferiority framing tightens
PROCEED as well as loosening DROP — but it was issued AFTER partial
(smoke) headroom numbers were visible, and was recorded only in the
implementation docstring as "pre-registered before any full-corpus
number existed". That phrasing is literally true and materially
misleading. **Treat the power rule as a dated amendment made with
partial information visible, not as an untouched pre-registration.**

## What this screen actually tests

`threshold_ms = kv_cost_ms + guard_ms` and `restore_cost_ms =
restore_cost_fraction * kv_cost_ms`, so **`kv_cost_ms` IS the price of
the swap action** and the existing kv sweep is already a static
memory-pressure sweep. A pressure-conditioned policy is the same policy
with `kv_cost_ms` replaced by a per-call λ.

The λ originally specified (replayed concurrent occupancy) **does not
exist in this corpus** and pricing off it was refused — see the
λ-honesty section. The replacement λ is per-call resident KV footprint
from `llm_call.data.prompt_tokens` (all 13,410 calls; 1.9k→85k tokens,
44× spread; median 13.4× growth within a task). KV bytes are linear in
resident tokens, so this IS the price per call, and it is known at call
start (an agent's context is frozen while a tool call runs).

**Consequence: this is NOT a test of time-varying pressure. It is a
test of PER-CALL FOOTPRINT-AWARE PRICING** — the certified policy today
charges one fixed `kv_cost` across a 44× spread of real footprints. The
arm uses no hindsight and is implementable, so a positive result is a
directly shippable policy iteration, not headroom for a hypothetical
future policy.

## Arms

- **footprint-priced:** each call priced at its own λ_i.
- **fixed-price (status quo):** the single best constant λ̄, selected on
  FIT FOLDS only; the shipped panel cell is always a candidate, so this
  baseline is never worse than what ships today.
- **Headroom = footprint-priced − fixed-price**, seconds per 277 tasks,
  task-clustered CI, permutation label from the certified engine.

Conservative by construction: λ̄ is chosen on realized fit-fold totals
(a stronger baseline), so headroom is a lower bound and a small
negative value is a legitimate "constant price is already effectively
optimal" outcome.

## Decision rule (two parts, different provenance — state both)

- **The bar — FROZEN BEFORE CODE:** compare against the banked
  pre-restore effect (~156 s/277 at kv3500). Do not chase something
  smaller than what is already held. Exposed as
  `--banked-seconds-per-277` so it is auditable.
- **The three-way POWER RULE — DATED AMENDMENT (2026-07-20, partial
  numbers visible):**
  - **PROCEED** iff the CI lower bound exceeds the bar and the
    permutation CI excludes zero.
  - **DROP** iff the CI **upper** bound is below the bar. Only this
    closes the direction.
  - **UNDERPOWERED** otherwise. The direction is NOT closed; it may not
    be cited as evidence of absent headroom and may not serve as a
    closed axis in the paper's characterization contribution.

Mandatory reporting: headroom with CI, the early-fire fraction it
scales against, the per-fold λ̄ selected, and the within-task footprint
growth distribution (the mechanism figure). Panel coherence
(non-monotonicity across the kv panel) is reported as a DESCRIPTIVE
secondary indicator only and cannot move the verdict.

**A SURVIVE does not license deployment.** It triggers a separate decision
replay paired against the frozen policy at `rho=0.94`, with permutation tests
per KV cell over the full cost family — exactly as pre-restore required its
robust-clock re-confirmation.

## λ-honesty: why occupancy was refused (structural negative)

Measured over all 277 tasks / 13,410 calls: within-task overlap exists
(336 adjacent pairs) but is a timestamp artifact (max 22 ms, median 1.2
ms, 100% under 50 ms, all on back-to-back fast reads) — the agent is
strictly sequential. Cross-task co-occupancy is exactly the collection
harness's worker count (peak 2, median 2, mean 1.59). The apparent
occupancy tail (3–9) is sub-millisecond `read_file` bursts colliding at
tick boundaries: mean latency ~2 ms, **zero** calls above any headline
threshold. Collection ran against a cloud provider, so no shared KV
cache existed to contend for.

Pricing λ off that would dress a `--concurrency 2` flag as physics on a
near-binary signal and would have manufactured headroom ≈ 0 — a false
negative caused by the corpus lacking pressure variation rather than by
the direction lacking value.

**Therefore: the multi-tenant CONTENTION axis is a STRUCTURAL NEGATIVE
on this corpus and this screen does not bound it. Contention claims may
come only from the live W5 harness.** This finding is independent of both
blockers and stands regardless of the screen's verdict.

## Integrity rules

Footprint λ̄ and reference tokens are fit-fold only. Certified stats
engine reused verbatim (50000 draws, conf 0.95, seed 0, Bonferroni over
the cost family). No estimator changes. `prompt_tokens` excludes the
emitting call's completion tokens, so the footprint is a LOWER BOUND on
resident KV (median 0.60%, p90 3.2% understatement) — uniform
under-pricing, not a between-arm bias. Degenerate paths inherit
conservative behavior. Every artifact states the arm semantics and the
amendment provenance adjacent to the verdict.
