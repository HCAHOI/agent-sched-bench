# Research directions — current state

Rewritten 2026-07-30 after eleven autonomous loop iterations. This states what is
true now; the original survey framing is in git history. Rubric and candidate
definitions remain in `direction-candidates-20260729.md`; its *status* column is
superseded by this file.

## Related work read in full

| System | Mechanism | Where our evidence bears on it |
|---|---|---|
| [Continuum](https://arxiv.org/abs/2511.02230) | `tau* = argmax_tau P(tau,f)(T_bar*eta + PrefillReload) - tau`; `P` a per-key empirical CDF keyed on **tool name** | A tool-name key reaches **0.9%** of the available budget even fitted in-sample with ~2682 rows per key (`8e7ed28`). The per-key empirical family, evaluated without leakage, is **negative** against a fixed deadline (`f1cbee3`). Reports no fixed-TTL and no oracle baseline. |
| [ThunderAgent](https://arxiv.org/abs/2602.13692) | Program abstraction over KV + tool assets; lifecycle-hook GC; async env prep | Its stochastic-workload panels sit at **0.65x and 1.24x**; we quantified why. Its fixed-threshold ablation is a **strong** reference, not weak: `T = kv` is near-optimal among constants (`24d4344`). |
| [AgentCgroup](https://arxiv.org/html/2602.09345v2) | eBPF cgroup control at **tool-call** boundaries; agents *declare* needs | Its granularity argument applies one level down, but the payoff is small (`adeccb2`). Its declaration remedy is undercut by analogy: agent reasoning text carries **no** usable duration signal (`33fb815`). |
| [Crab](https://arxiv.org/abs/2604.28138), [DeltaBox](https://arxiv.org/html/2605.22781) | Semantics-aware sandbox checkpoint/restore | Occupies the environment-snapshot direction entirely. |
| [Seer](https://arxiv.org/abs/2511.14617) | Shared-prompt similarity for RL rollout | Occupies the rollout direction. |
| [PASTE](https://arxiv.org/html/2603.18897) | Speculative tool execution from recurring patterns | Occupies the prefetch direction. |

## The KV lane is closed. What it establishes

A complete, internally consistent negative result for tool-duration prediction under
KV swap, with every load-bearing constant checked rather than assumed.

| Policy class | Best share of the oracle budget |
|---|---|
| Constant trigger, fitted out-of-fold | **negative** (beats deadline in 1 of 20 cells) |
| Constant trigger, in-sample oracle | ≤ 7.2% |
| Per-key empirical, in-sample | 29–75% — **all overfit** |
| Per-key empirical, leave-one-out | **negative** |
| Per-call oracle | 100% by construction |

Supporting facts, each independently established:

- **The budget lives only in `kv < L < 2*kv`.** Calls beyond `2*kv` contribute
  *exactly zero*, because the deadline already hides the full swap there (`57cbbbb`).
  Every predictor in this repo targeted `P(long)`, which is the wrong quantity.
- **`T = kv` is near-optimal among constants**, because it is exactly where the
  restore penalty vanishes. The sweep rediscovers `5000 ms` in 4 of 5 folds
  (`24d4344`).
- **`rho_effective = 0.94` unconditionally** for this model and device. Reload beats
  recompute by 38.4x at every KV size, and the ratio is scale-invariant because both
  costs are linear in tokens. Recompute would need >3.77 MB/token; this model has
  98 KB (`8e5fc7d`).
- **The large budget was largely a scale artifact.** As a fraction of deadline
  utility it is 9.2–16.6% at per-request KV cost (56–160 ms) against 84.8–88.9% at the
  campaign's 3500/5000 ms cells, which are 62 and 89 *requests'* worth of KV
  (`8968434`).

### Known regime limits, and they are real

The negative result holds for **swap-back-expensive** regimes. Per-key methods do beat
the deadline below `rho ~ 0.3` (kv=3500) and `~0.6` (kv=5000) (`2316693`) — but
reaching that would require ~38x more KV per token than this model has. Everything is
also conditional on **forced eviction**: with no memory pressure the optimal policy is
never swap and every arm scores zero. `CLOSED-QUESTIONS.md` establishes Fresh-277
cannot exhibit contention, so that premise is untestable on this corpus.

## Directions

| # | Direction | Status |
|---|---|---|
| D2/D4 | Predictability boundary + mechanism-vs-prediction decomposition | **COMPLETE** — the table above is the result |
| D1 | Clause-granular resource control | **Near-dead**: corrected payoff 10.6% on 12.3% of calls; `memory.max` is a cap not a reservation; lowering it below usage OOM-kills |
| D6 | Pipeline-dominated shells (**87.7%** of multi-clause exec calls) | **OPEN, no consumer found** |
| D3 | Heterogeneous multi-tenant composition | **BLOCKED offline** — needs contention this corpus cannot exhibit |
| D5 | Clause-attribution instrument | Folded into D1 |

## No high-value offline experiment remains

Checked and exhausted: the target quantity, the baseline, the estimator family, the
key granularity, the overfit, the restore charge, and the cost scale. D6 has no
decision it changes; D3 and the forced-eviction premise both require contention the
corpus provably lacks. Further iterations of this loop would produce commits without
advancing a claim.

## Corrections made during the run, for the record

Nine, all self-caught: a transcription error (`1947` for `1089`); an envelope figure
wrong by ~5x from summing a sequential integral over concurrent pipeline clauses; a
false verification line in a commit message; the reasoning-text and runtime-activity
NO-GOs; the in-sample per-key ceilings shown to be pure overfit; the hypothesis that
`rho = 0.94` was an overcharge, refuted; and the discovery that the headline cost cells
were 62–89 requests' worth of KV rather than one.
