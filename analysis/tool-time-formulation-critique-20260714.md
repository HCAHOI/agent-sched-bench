# Tool-Time Formulation Critique & Response Plan (2026-07-14)

Critical review of the trigger-based tool-time formulation (empirical context
hierarchy + robust utility clock + DACD certification), with a ranked
experiment plan. Code referenced: `src/trace_collect/tool_latency_utility_clock.py`,
`src/trace_collect/tool_latency_confirmation.py`, `src/trace_collect/tool_latency_profiled.py`.

## Top findings (ranked by risk to validity)

### F1. Utility function omits restore cost — can manufacture success

In `_utility_matrix`, an early swap on a call that finishes before the
threshold `T` but after the swap completes earns exactly `0`: no swap-back
(restore) cost, no state churn. In a real system a completed swap-out on a
call that then returns forces a swap-in on the critical path before the agent
resumes. Unnecessary-but-fully-hidden swaps are free in the objective and
expensive in reality. Any measured gain may be an artifact of this asymmetry.

**Response: Experiment E1 (restore-cost sweep) — implemented first.**

### F2. "Hidden cost is hidden" assumption unvalidated

KV swap contends for PCIe/host memory bandwidth (and possibly CPU) with the
container running the tool. Offline latency labels were collected with no
swap running, so traces cannot reveal whether firing the action inflates tool
latency. Requires one online A/B measurement (E5).

### F3. Missing baseline: within-task last-value

Tool latencies within a task are strongly autocorrelated (warm caches,
incremental builds). "This command takes about as long as it did last time in
this task" is ~10 lines and may capture most of the certified gain. The
cross-task pooled prior discards this signal entirely. (E-B1)

## Per-question assessment

### Q1. Estimand: trigger vs. distribution

Decision-focused framing is correct, but the trigger-vs-regression dichotomy
is false: `_node_utility_curves` already estimates the conditional latency
distribution (empirical CDF per node) through a utility functional. Refusing
to name it costs:

- **Amortization**: `k(x, a)` recomputed per (cost, guard) pair (`trigger_cache`
  is the symptom). A survival curve `S(t|x)` is computed once; every action's
  trigger is a cheap functional. Required for the M(t, S | P, A) target.
- **Diagnostics**: a trigger has no calibration check; a survival curve does.
  Currently cannot distinguish "trigger is right" from "utility function
  forgives being wrong".
- `hazard_recheck_ms` is conditional survival by another name.

Verdict: keep the trigger as deployed output; factor it through an explicit
discrete-time hazard/survival estimate.

### Q2. Context hierarchy

Katz-style hard back-off. Problems:

- Hard back-off discards the parent instead of shrinking toward it.
  Empirical-Bayes partial pooling (precision-weighted node/parent blend) is
  strictly more sample-efficient, one hyperparameter instead of two support
  thresholds, no dataset-specific rules.
- `min_tool_history=1` default lets a single observation qualify a node; the
  robust-clock unanimity filter papers over this (then the hierarchy is not
  doing the work).
- `skip_leading_cd` / `max_prefix_depth=4` are defensible as general shell
  semantics but must be shown insensitive (depth ∈ {2,3,4,6}) or a reviewer
  will call them tuned.
- Exchangeability unit risk: tasks from the same repo share build systems;
  report repo-clustered intervals as sensitivity on SWE-ReBench.
- Censoring: verify how timeout-capped calls enter `read_tool_latency_jsonl`;
  truncated tails bias every trigger aggressive. Survival machinery handles
  censoring natively.

### Q3. DACD

- Percentile bootstrap LCB on heavy-tailed per-task deltas with modest task
  counts has no coverage guarantee; "certified" overclaims. Use BCa /
  studentized bootstrap, or (one-sided decision) sign/Wilcoxon on task deltas.
- Structurally this is high-confidence policy improvement (HCPI, Thomas et
  al.) / SPIBB re-derived. Position against that literature or drop the
  novelty claim; honest contribution = the task-clustered *paired*
  construction for trace data.
- Guarantee survives one pass only. Iterating method variants against the
  same certification partition spends it; headline numbers need a final
  untouched partition or fresh traces.
- Two stacked conservative filters (LOTO unanimity + Bonferroni bootstrap
  gate): report effective early-fire rate and fraction of certified gain from
  top-3 tasks (concentration diagnostic, E4).

### Q4. Sequence history and telemetry

- Features must be functions of the trace prefix at call start (or checkpoint
  time with landmarking). Grep-level failure mode: normalization by
  whole-trace statistics.
- Bigger problem is feedback, not leakage: once telemetry is both feature and
  action-affected, offline replay evaluation is invalid → logged-action OPE
  or online randomization. Current offline eval valid only while actions
  provably don't perturb latencies (F2).
- Clean architecture: discrete-time hazard with time-varying covariates
  (landmarking); `lifelines` / `scikit-survival`.

### Q5. Correlated actions

Factorize: one shared conditional hazard `h(t | x, telemetry)` (learned once)
+ per-action utility functionals (cost, benefit rate, footprint, restore
cost) whose triggers are argmax under `S(t|x)` — no learning per action.
Coupling for shared capacity: weakly coupled MDP → Lagrangian relaxation
(capacity prices decouple actions into priced stopping problems) or Whittle
indices. Start Lagrangian.

### Q6. Better formulations

Adopt: **learning-augmented ski rental**. Wait-vs-commit against unknown
duration with break-even threshold is the ski-rental problem; `T = c + guard`
is the classical deterministic 2-competitive strategy. Purohit–Svitkina–Kumar
(NeurIPS 2018) + follow-ups give prediction-with-trust-parameter-λ algorithms
with consistency/robustness guarantees. Replaces ad-hoc unanimity with a
worst-case guarantee; DACD becomes data-driven selection of λ. Competing
risks later (completion vs timeout vs cancellation); bandits/control only
once feedback effects are real.

### Q7. Baselines to run (any could invalidate the method)

- **B1 within-task last-value**: fire early iff same command prefix ran in
  this task and exceeded T; trigger at min(prev latency, T)·δ. Top candidate
  to embarrass the method.
- **B2 tool-class whitelist**: never early for read/edit/ls-class; single
  global k for {test, install, build}-class.
- **B3 global mean-hazard**: existing `mean_hazard` at global node only — no
  hierarchy, no DACD. Isolates each component's contribution.

### Q8. Experiments (ranked by discriminative power per effort)

- **E1 restore-cost sweep**: add ρ > 0 to the utility for fires-on-short;
  sweep ρ ∈ {0, 0.25c, 0.5c, c}. If certified gains vanish at realistic ρ,
  current results are an artifact. Smallest, most important.
- **E2 cross-benchmark transfer**: fit + certify on SWE-ReBench, evaluate
  frozen on Terminal-Bench and ScienceAgentBench Verified (all rotations).
- **E3 DACD empirical coverage**: repeated fit/certify/audit three-way
  splits; how often are certified-positive rules positive on the untouched
  audit partition?
- **E4 gain-concentration diagnostic**: cumulative certified gain vs number
  of contributing tasks.
- **E5 online contention**: real swaps during real tool calls; compare
  latency distributions to no-swap. Validates F2.

## Concrete alternative (implementable on existing traces)

Discrete-time hazard model + ski-rental trigger:

1. Log-spaced interval grid; per-interval survival records (censoring free).
2. One pooled gradient-boosted/logistic hazard model (scikit-survival or
   lifelines): tool id, command-prefix tokens (reuse
   `make_row_command_prefix_keys`), within-task history (prev same-prefix
   latency, call index). No back-off machinery.
3. For any (c, guard, ρ): trigger from predicted survival curve via
   learning-augmented ski-rental rule with trust parameter λ.
4. Select λ per context class on the fitting partition; gate deployment
   (λ vs λ=0) with existing paired task-clustered machinery, Wilcoxon-based
   bound instead of percentile bootstrap.
5. Evaluate vs B1–B3 with restore cost, under E2 transfer protocol.

## E1 implementation & review audit (2026-07-14)

- Implemented `restore_cost_ms` on the single-cost utility primitives
  (`_utility_matrix`, `trigger_policy_utility_ms`, `robust_utility_trigger_*`,
  `hazard_recheck_ms`) and `restore_cost_fraction` on the panel-level entry
  points (`evaluate_utility_clock_policy`, `paired_task_cluster_bootstrap`);
  defaults (0.0) reproduce frozen-protocol numerics exactly. Charged only on
  fires where the call is short: the deadline baseline never fires there, and
  long-call restores cancel in every paired contrast.
- New `scripts/analyze_restore_cost_sweep.py` re-scores frozen decision
  JSONLs (triggers fitted at rho=0) under a restore-fraction sweep for three
  contrasts: gated_vs_robust, gated_vs_deadline, robust_vs_deadline.
- Review gate: fresh `code-reviewer` sub-agent, verdict APPROVE, 0 critical,
  0 major, 3 minor (schema-note, sweep-multiplicity caveat, Mode-A pinning
  comments) — all three fixed. Reviewer independently verified the
  no-new-breakpoint exactness claim, the long-call cancellation, the frozen
  numerics at defaults, and the hand-computed test deltas. 119 focused tests
  pass.

## E1 result (2026-07-14): F1 confirmed

Mode A re-scoring of the frozen swe-rebench-100 confirmation
(`analysis/tool-time-restore-cost-sweep-20260714/`): at rho=0 the gated
policy beats the deadline (+151.8 s, one certified cost); every
certification vanishes at rho=0.25; at rho=1.0 (swap-in ~ swap-out) both
early policies are net-negative vs the plain deadline. The DACD-style gate
is protective under rho but converges toward the deadline. Top-3 tasks carry
48% of the rho=0 gain (E4 preview). Details in that directory's findings.md.
Open: Mode B refit, transfer (E2), measured rho.

## Mode B result (2026-07-14): the formulation adapts

Refit with rho in the objective on the same frozen folds
(`analysis/tool-time-restore-cost-mode-b-20260714/`): gated_vs_deadline
stays ≈ +120–150 s at every rho with certified-positive cells at each
fraction (e.g. {1000, 4500} at rho=1.0); fires-on-short collapse 90 → 11
while total early fires persist. The rho=0 refit reproduced the frozen
triggers exactly (built-in check). Conclusion: restore cost must be a
first-class action parameter; with it, the trigger machinery survives F1.
Sensitivity analysis only — a certified rho>0 claim needs a fresh corpus.

## B1 result (2026-07-14): baseline wins at rho=0, collapses at honest rho

Within-task completion-gated history baseline
(`analysis/tool-time-within-task-baseline-20260714/`): at rho=0 it beats
the deadline by +280.5 s — roughly double the gated method — confirming the
critique. At rho=0.5 it turns net-negative (two certified-harmful cells;
−397.8 s at rho=1.0) while the refit gated policy holds and certifies over
it (+515.9 s, 3 cells, at rho=1.0). Conclusions: restore-zero results are
not evidence for the method; the machinery's contribution is robustness;
within-task signal is real and should become the deepest context level of
the hierarchy (hybrid, with a gated-B1 control first).

## Gated-B1 control (2026-07-14): the gate needs cross-task pooling

Fair control (`analysis/tool-time-within-task-gated-20260714/`): giving the
within-task baseline the same margin-guard machinery does not save it — at
rho=1.0 every fold's guard selects "never early" (gated-B1 == deadline,
delta exactly 0), and at rho=0.5 the gated variant is worse than ungated.
At rho=0 the gated baseline (+205.5 s) still beats the cross-task method.
Conclusion: within-task signal and cross-task pooled certification are
complementary — direct motivation for the hybrid hierarchy.

## E2 result (2026-07-14): the gate transfers, the priors do not

SWE-ReBench-fitted rules applied frozen to Terminal-Bench
(`analysis/tool-time-transfer-terminal-bench-20260714/`): gated policy
net-negative vs the deadline at every rho (−13.9 to −86.5 s), though the
gate bounds losses (no harmful cells) while ungated clocks are
catastrophic (mean-hazard −540.8 s, 4 harmful cells at rho=1.0).
ScienceAgentBench (`...-science-agent-bench-20260714/`) is neutral — the
corpus is tool-sparse (~1.7 calls/task), nothing to win. The strong
generalization claim fails; surviving framings: (a) the certification
machinery is the transferable artifact (safety), (b) deployment-matched
fitting with a few-shot target-adaptation question as the open middle
ground. Both corpora were development-exposed (sensitivity only).

## Campaign synthesis (2026-07-14)

- Restore cost must be a first-class action parameter; rho=0 results are
  non-evidence (E1, B1).
- With rho in the objective, the cross-task gated method holds certified
  gains within-benchmark (Mode B) where every simpler variant fails:
  ungated clocks and the within-task baseline collapse under rho, and the
  gate cannot rescue within-task margins (gated-B1).
- The priors are benchmark-local: frozen transfer is neutral-to-negative,
  but the gate consistently bounds damage everywhere (E2).
- Defensible thesis for the paper: a certified trigger-gating framework
  whose value is robustness under honest cost accounting and whose priors
  are fitted per deployment workload — not universal latency priors.
- Next phase: hybrid hierarchy (within-task deepest level + cross-task
  pooled certification), few-shot target adaptation curve, measured rho on
  target hardware, fresh-corpus certification at rho>0.

## Hazard-model phase result (2026-07-14/15): trie beats linear hazard

The critique's concrete alternative was implemented in full
(docs/hazard-model-phase-plan-20260714.md; review-gated, equivalence
anchors proven, fits amortized across rho) and evaluated in three feature
arms (`analysis/tool-time-hazard-model-{full,within-task,cross-task}-
20260714/`). Verdict: the pooled penalized logistic loses head-to-head to
the gated trie at every rho in every arm (−118 to −235 s totals, zero
certified-positive cells); its guard collapses it to the deadline at
rho ≥ 0.25; more features make rho=0 worse. Diagnosis from the calibration
block: marginally calibrated but not SHARP (reliability bins show heavy
shrinkage; predicted 0.55 survival → observed 0.78). The trigger utility
pays for conditional sharpness, and at ~7k calls the memorizing trie is
the sharper survival estimator. The gate again bounded the weak predictor
safely (zero harmful cells gated; −669 s ungated). The factorization
architecture (amortization, calibration diagnostics, drop-in seam) worked
exactly as designed — the estimator, not the formulation, failed. Open:
GBM arm (nonlinear, sharpness-capable, plan-sanctioned), trie-native
within-task node level.

## GBM arm result (2026-07-15): sharpness confirmed, frontier split

Same features/seam/gate, estimator swapped to HistGradientBoosting
(`analysis/tool-time-hazard-model-gbm-full-20260715/`): head-to-head vs
the gated trie flips from −235 s (linear) to +170 s at rho=0 (certified
cell at 1000 ms), +322 s vs the deadline (2 certified; 3 at rho=0.25), and
it is the first policy to beat gated-B1 at every fraction — the nonlinear
model converts the within-task signal. At rho ≥ 0.5 the trie's LOTO
unanimity still wins (−94 s, one harmful cell at rho=1.0). Reliability
bins confirm the mechanism: shrunken middle → confident extremes. No
policy dominates; measured rho picks the regime. Next: bagged/LOTO
ensemble for the GBM (import the trie's robustness into the sharp
estimator), trie-native within-task level, measured rho.

## Execution order

1. E1 restore-cost extension + sweep — DONE 2026-07-14, F1 confirmed (above).
1b. Mode B refit — DONE 2026-07-14, method adapts (above).
2. B1 within-task baseline — DONE 2026-07-14, two-sided result (above).
2b. Gated-B1 control — DONE 2026-07-14, gate needs pooling (above).
3. E2 transfer (Terminal-Bench, ScienceAgentBench) — DONE 2026-07-14,
   priors don't transfer, gate does (above).
4. Hazard-model phase (learned survival estimator) — DONE 2026-07-15,
   linear hazard loses to trie; sharpness is the binding constraint
   (above).
2. B1 within-task last-value baseline.
3. E4 concentration diagnostic (cheap, reuses confirmation outputs).
4. E2 transfer protocol.
5. Hazard-model alternative (steps 1–4 above).
6. E3 coverage; E5 online contention (needs GPU host).
