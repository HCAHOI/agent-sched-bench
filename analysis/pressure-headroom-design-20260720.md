# Design spec — λ-oracle pressure headroom screen

> **Status: FINAL (Fable-5 debate 2026-07-20: DEFER pressure-
> conditioning as a headline experiment; PROCEED on the multi-tenant
> harness that was never optional; RUN this screen first).** The
> screen bounds the value of EVERY pressure-aware policy — online or
> not — before any GPU hour is committed to the direction.

**Date:** 2026-07-20 · Zero GPU, existing data, ~half a day.

## The structural observation that makes this cheap

In `tool_latency_utility_clock.py`: `threshold_ms = kv_cost_ms +
guard_ms` and `restore_cost_ms = restore_cost_fraction * kv_cost_ms`.
So `kv_cost_ms` IS the price of the swap action, and the existing kv
sweep (500…5000) is already a STATIC memory-pressure sweep. A
pressure-conditioned policy is the same policy with `kv_cost_ms`
replaced by a time-varying λ(t). Its ceiling is therefore measurable
offline today.

## The screen

Over the certified fresh-277 decision corpus at rho=0.94:

- **Clairvoyant arm:** each call priced at the REALIZED λ at its own
  decision instant. This is an ORACLE — analysis only, never a
  shippable policy (hindsight rule); labeled as such in every artifact.
- **Best-fixed arm:** the single best constant λ over the same run,
  selected on FIT FOLDS only under the existing cross-fit discipline.
- **Headroom = clairvoyant − best-fixed**, seconds per 277 tasks,
  task-clustered CI, permutation label per the certified engine.

Headroom upper-bounds every pressure-aware policy, because no online
policy can beat the clairvoyant one.

## λ trajectory (must be honest, and stated)

λ(t) is derived from the replayed concurrent occupancy of the existing
corpus — the number of simultaneously in-flight calls at each decision
instant, mapped to a price by a documented monotone map (the kv panel
supplies the range; the map is config, not a fitted constant). The map
is NOT tuned to outcomes. If no defensible occupancy signal exists in
the corpus, the screen reports that and returns a structural negative
rather than inventing a load model — synthetic load on frozen traces
is forbidden.

## Pre-registered kill criterion (frozen before code)

**DROP the pressure-conditioning direction** (to one future-work
sentence) if headroom is below the already-banked pre-restore effect
(~156 s/277 at kv3500) — do not chase something smaller than what is
already held. **PROCEED to the live arm** (on the mandatory W5-7
harness) only if headroom materially exceeds it, with the CI
excluding zero.

Mandatory reporting either way: the clairvoyant gap in seconds, the
fire fraction it scales against (currently 1.4-1.5%), and the
best-fixed λ selected.

## Why a null here is publishable (and de-risks running it)

If headroom is small, the finding is: *even with an oracle view of
instantaneous memory price, the swap decision remains essentially
precomputable at call start* — A0's completeness lemma extended to
the pressure axis. That strengthens A1 rather than producing nothing,
and it is a claim neither Continuum (workload-level sliding-window
load term) nor ThunderAgent (memoryless decay) can make.

## Integrity rules

Clairvoyant arm is oracle-only and never proposed as deployable.
Best-fixed λ selected on fit folds only. Certified stats engine reused
verbatim (50000 draws, conf 0.95, seed 0, Bonferroni over the cost
family). No estimator changes. Degenerate paths inherit conservative
behavior. Every artifact states the oracle status adjacent to the
verdict, as A2 does with its trigger-source caveat.
