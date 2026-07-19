# Design spec — emergent wrapper-transparency normalization (WTN)

> **OUTCOME (2026-07-19, full corpus, review-gated code,
> `wrapper-transparency-stage1-2026-07-19.md`): K1 KILL / K2 pass /
> K3 pass.** The screen itself is VALIDATED (cd emerges 5/5 folds plus
> echo/git/ls/python3/which; mass-weighted stability 0.995) and WTN is
> statistically indistinguishable from hardcoded cd-only (deltas
> ≤0.3ms, CIs straddle 0) — the emergent form costs nothing. But at the
> certified operating point (min_evidence=1) NO normalization beats
> no-normalization on MAE (WTN vs off: +25ms, CI [−79, +185]) → KILL
> as pre-registered. The de-confounding headline: cd-skip's quoted
> 25ms MAE win exists ONLY jointly with the min_evidence=5 gate
> (WTN vs off@me5: −21ms, CI [−28, −14], excludes zero); normalization
> alone improves tail (~−135ms) at MAE cost. Census: beyond-leading
> mass 6.9% → paragraph framing, as pre-committed. Live successor
> question: (normalization + evidence-gate) as ONE joint policy
> iteration through Stage-2 certified decision replay, in emergent-
> screen form since it matches hardcoded at zero cost.

> **Status: FINAL, post-debate (Fable-5 adversarial debate 2026-07-19,
> verdict PROCEED with 12 amendments, all incorporated below).**
> Motivated by the project lead's objection that cd-skip ("strip the
> literal leading `cd X &&`") is a hardcoded rule, not a method. WTN
> generalizes it: candidate wrapper positions come from shell
> STRUCTURE, transparency is decided by DATA on fit folds, and the
> normalization is certified per workload. If killed, cd-skip remains
> an honestly-labeled ablation and the shipped estimator stays
> chain_prefix (cert config).

**Date:** 2026-07-19 · **Corpus:** fresh-277 segment-timeline replay
(8,953 segments / 4,824 analysable sequential chains / 277 tasks) —
existing data only, zero GPU. **Harness:** the atom-study comparison
(`scripts/analyze_segment_variance.py`), extended.

## Debate findings that reshaped the spec

1. **The quoted cd-skip win (971 vs 996 MAE) is confounded.** In the
   study harness, `chain_prefix_cert` is fit with the frozen cert
   literal min_evidence=1 while `chain_prefix_cdskip` uses the sweep
   default min_evidence=5 — the 25ms/155ms delta bundles the
   normalization with a stricter evidence gate. cd-skip's isolated
   effect has never been measured. Stage 1 repairs this regardless of
   WTN's fate.
2. **Production keys already strip env assignments unconditionally**
   (`_ENV_ASSIGNMENT` in `src/trace_collect/command_features.py`), and
   redirections likewise. The pipeline therefore already contains
   transparency judgments by fiat; WTN's framing is "replace fiat with
   a screen." The token-level assignment-prefix arm of the original
   draft is CUT (it would re-litigate shipped normalization and
   diverge from `make_row_command_prefix_keys`).
3. **Honest size label:** reachable mass beyond cd is expected ~10% of
   chains (conda-activation preamble family, echo/which classes). WTN
   is a subsection-and-one-results-row contribution about certifying
   the conditioning key; it drops to a paragraph without renegotiation
   if the Step-0 census comes in tiny. Framing is decided by the
   census BEFORE accuracy numbers are seen.

## Mechanism (amended)

**Candidates — structural, zero token names in logic:** every
non-final segment class (keyed by its verb token) in a sequential
chain. (Token-level arm cut per debate finding 2.)

**Transparency screen — single gate, cross-fitted, fit data only:**
- *Sole gate (predictive equivalence):* pooling suffix-duration
  samples across presence/absence/spelling of the candidate class does
  not degrade prediction of `parent_total_ms` on fit-side NESTED
  task-grouped folds vs splitting on it. Statistic: paired per-task
  mean of per-chain absolute-error deltas. Equivalence tolerance fixed
  a priori (relative fit-side MAE; one primary value + a pre-declared
  2-3 point sensitivity grid, all reported). Evaluable on the full
  corpus including pipe-parents — no per-segment telemetry dependency.
- *Own-duration share* is demoted to a reported diagnostic (it is
  nearly vacuous as a gate: any preamble passes).
- *Min-support rule:* a candidate class below pre-set fit-fold support
  (tasks AND chains, documented config) defaults to NON-transparent.
  Underpowered equivalence tests must not pool — this protects rare
  heavy wrappers (e.g. `timeout`, 22 segments) from silent over-pooling.

**Normalization procedure (pre-registered, no undefined recursion):**
marginal screen per candidate class against the current key, then
JOINT application of all passing classes, then one joint fit-side
non-degradation check; both marginal and joint results reported. Chains
whose wrappers all drop collapse to the same normalized key (support
consolidation — the H2 thin-prefix-bleed mechanism). Non-transparent
wrappers stay in the key. Everything downstream (ECDF nodes,
`hazard_recheck_ms`, min-support fallback) is untouched.

## Stage 1 — one harness run, three questions (~hours, CPU)

**Step 0 (before any fit): reachable-mass census.** Chains with ≥1
candidate beyond leading cd, by class; corpus mass affected by
normalization. This fixes the contribution framing before accuracy
numbers exist.

**Knob-matched grid (de-confounds A1):**
skip ∈ {off, cd-only, WTN} × min_evidence ∈ {1, 5}, depth 4 fixed.
Same task-grouped folds, same corpus. All pairwise deltas get
task-clustered paired bootstrap CIs (statistic: paired per-task mean
of per-chain absolute-error deltas). Point deltas are not
decision-grade on a 9s-tail corpus.

**Pre-registered kill criteria:**
- **K1 (reproduce the win, knob-matched):** KILL if WTN does not beat
  the knob-matched no-normalization control on MAE with a CI excluding
  zero at the matched min_evidence, or if WTN is worse than
  knob-matched cd-only beyond paired noise. cd-only-parity is the
  minimum claim; beating it is the method claim.
- **K2 (emergence, positive control, no retuning):** all screen
  thresholds are frozen BEFORE checking emergence. KILL if the cd
  segment class does not emerge transparent in every fold — and if it
  does not, that is reported and killed, never re-tuned (cd is the one
  known ground-truth positive; a screen that misses it measures
  something else). Token-name pressure on thresholds is forbidden.
- **K3 (stability, mass-weighted):** KILL if fold agreement is poor,
  measured as the fraction of normalization-affected chain mass
  treated identically across folds (raw Jaccard on a 2-8 element set
  is degenerate and is reported only as a diagnostic).

## Stage 2 — certified decision replay (only if Stage 1 survives)

ONE paired decision replay at rho=0.94: WTN-normalized policy vs the
frozen baseline, task-clustered permutation certificate per kv cell.
cd-only becomes an ablation row, not a second certified replay
(neither normalization has passed decision replay yet; the paper needs
one certified normalization, not two replays). KILL if no cell
certifies where the policies diverge. External validity: wrapper
classes that emerge on TraceLab replay (roadmap item) are the transfer
story; a hardcoded token rule has none.

## Integrity rules

Screen thresholds are documented config applied to fit-fold statistics
— never selected on the eval fold, never per-dataset constants, never
adjusted after emergence checks. No token spelling appears in method
logic (cd appears ONLY in the K2 positive-control assertion and the
cd-only grid arm, both evaluation harness, not method). Folds
task-grouped throughout; nested folds for the screen. All degenerate
paths (no candidates, empty transparent set, thin support) degrade to
the unmodified certified config.

## Prior art (from the debate, verified by the debater)

Log template mining (Drain/Spell/LogMine) shares the normalization
instinct but is unsupervised and never gated on downstream predictive
equivalence; query fingerprinting (pt-query-digest) and template-based
query performance prediction (Akdere et al., ICDE 2012) are the
closest occupied territory with FIXED rules. The surviving delta is
learned, per-workload, cross-fitted certification of key normalization
for latency priors — the certification discipline plus the systems
seam. Subsection-sized; not a headline.

## Relation to prior kills

B died because pooled atom-duration nodes had no union mass; WTN
screens for the opposite property (classes carrying NO information)
and spends the win consolidating existing support, not minting thin
nodes. C died because runtime boundary identity added churn without
information; WTN is fit-time only and adds no runtime path.
