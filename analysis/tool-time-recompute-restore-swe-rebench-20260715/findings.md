# Reload-vs-recompute restore (P2) — findings (2026-07-15)

First step toward M(t,S|P,A): a SECOND restore mechanism. The gated
policy's restore-on-return becomes min(rho*kv, recompute_rate*context) —
Continuum/ThunderAgent's reload-vs-recompute choice — evaluated at the
measured rho=0.94 on the frozen swe-rebench-100 decisions (Mode-A
re-scoring, triggers frozen; commit after the P2 feature). Context length
is the exact llm_call prompt_tokens joined to same-iteration tool calls
(4640/4640, causal, known at dispatch): range 1908 / median 16030 / max
70220 tokens. recompute_rate is SWEPT (0.05-1.5 ms/tok), a stand-in for a
later measured H100 prefill curve exactly as rho was swept before we
measured 0.94. Review-gated APPROVE (join verified exact + causal).

## Result: recompute helps iff prefill is cheap — and the crossover is the unmeasured quantity

At rho=0.94, totals vs the deadline (5-fold, 50k task-clustered bootstrap):

| recompute rate (ms/tok) | swap-only vs deadline | min-restore vs deadline | min vs swap (restore cut) |
|---|---|---|---|
| 0.05 | -32.9 s | **+68.9 s** (0 cert) | **+101.8 s (4 cert)** |
| 0.15 | -32.9 s | -6.0 s | +26.9 s |
| 0.5  | -32.9 s | -31.1 s | +1.8 s |
| 1.5  | -32.9 s | -32.9 s | +0.0 s |

- **Swap-only is net-negative vs the deadline (-32.9 s) at honest rho** —
  the campaign's core finding, reconfirmed.
- **Recompute certifiably cuts restore cost** (min-vs-swap, +101.8 s with
  4 certified cells at cheap prefill; >=0 by construction, so the
  certification is on MAGNITUDE). As prefill gets pricier the min() always
  picks swap-in and the cut vanishes to 0.
- **At cheap prefill (0.05 ms/tok ~ 20k tok/s, plausible for an optimized
  H100 stack) the recompute option recovers the swap-only deficit and
  flips the policy to +68.9 s vs the deadline** — though uncertified (wide
  CI, frozen corpus underpowered, and triggers were fit swap-only so this
  LOWER-bounds a recompute-aware policy). At expensive prefill recompute is
  never chosen and the policy collapses to the swap-only -32.9 s.
- Identity C2 - C3 = C1 holds to machine precision (e.g. 68.9-(-32.9)=101.8).

Crossover: recompute wins when recompute_rate*context < rho*kv, i.e.
context < rho*kv / rate. At the median 16030-token context, rate 0.05 ->
801 ms recompute beats swap-in across most of the kv grid; rate 0.15 ->
2404 ms only at the largest kv. So whether recompute helps hinges entirely
on the (currently swept, later measurable) prefill rate.

## Interpretation

This is the first honest multi-action result and it behaves exactly as the
factorization predicts: a second action = a new restore-cost functional
over the SAME frozen triggers, with the action-cost swept like rho. It
does NOT claim recompute wins — it says recompute certifiably reduces
restore cost, and whether that flips the honest-rho policy positive is
decided by one unmeasured hardware number (the H100 prefill-vs-context
curve). That measurement is the concrete next fresh-GPU step, and until
then the sweep brackets the answer rather than asserting one.

## Caveats

Mode-A lower bound (triggers fit swap-only; a Mode-B refit under
min-restore would deploy a genuinely recompute-aware trigger and can only
do better). Linear recompute rate under-charges long-context prefill
(~O(c^2)) and ignores generated tokens — both conservative toward
recompute. Frozen dev-exposed corpus (sensitivity, not certification);
nothing certifies vs deadline (underpowered). Rate grid is a documented
bracketing stand-in, not tuned. Fresh corpus + measured prefill curve
remain the path to a certified recompute-aware claim.
