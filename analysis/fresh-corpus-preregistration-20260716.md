# Fresh-corpus certification — PRE-REGISTRATION (2026-07-16)

Locked and committed BEFORE any fresh trace is collected or inspected. This is
the confirmatory protocol; deviations must be logged as such. It converts the
campaign's dev-exposed sensitivity results into a certified result on untouched
data, using the coverage-valid permutation certificate (Phase 0's fix) at the
measured operating point.

## Fresh corpus (disjoint from all dev data — verified)

- Dataset: `nebius/SWE-rebench`, split `filtered` (6542 rows), `exclude_lite: false`.
- Selection: one reproducible shuffle `random.Random(42)`, contiguous window
  **skip=150, sample=200** → `shuffled[150:350]`, 200 tasks.
- Disjointness VERIFIED (2026-07-16): the dev confirmation set equals
  `shuffled[50:150]` exactly (100/100 match, pool undrifted since June);
  `shuffled[150:350]` has ZERO overlap with it and with the original 50-task set
  `shuffled[0:50]`. So the fresh set never touched method development.
- Collection config (identical to the dev corpus for apples-to-apples):
  provider `openrouter`, model `qwen/qwen3.7-max` (user-specified slug; matches
  dev `model.name=qwen3.7-max`), scaffold `openclaw`, `--mcp-config none`,
  `--max-iterations 100`, `--min-free-disk-gb 30`, runtime
  `task_container_agent`. Image lifecycle: `TASK_CONTAINER_CLEANUP_IMAGES=1`
  (per-task image removal; disk-bounded).
- Freshness attested in the run manifest (`not_used_for_method_development`,
  `not_smoke_or_synthetic`, `complete_fixed_task_set`) with SHA256 input-hash
  inventory and the three dev roots in `excluded_trace_roots`, enforced by
  `run_offline_gated_robust_confirmation.py` (content-overlap rejection).

## Operating point and certificate

- rho = KV swap-back/swap-out cost = **0.94** (measured, H100 Gen5x16). The fresh
  run scores at rho=0.94 by construction (not the low-rho probe). rho stays a
  parameter; it is not swept or featured.
- Certificate: **paired sign-flip randomization** (`permutation_draws>0` in
  `paired_task_cluster_bootstrap`), Bonferroni-simultaneous over the 10-cost
  family at one-sided tail alpha/(2m)=0.0025, draws >= 20000 (min resolvable p
  0.00005 << tail). The percentile-bootstrap `simultaneous_label` is NOT used
  for certification (measured ~2.7x anticonservative; Phase 0). Triggers are
  Mode-B (refit at the operating rho) for any deployment claim.

## Pipeline (fixed; no post-hoc rule changes)

Frozen offline pipeline, identical config to the dev confirmation (FROZEN_CONFIG:
fold_count 5, inner_folds 4, costs {500..5000} ms, guard 0, command_field
`command`, max_prefix_depth 4, skip_leading_cd false):
1. Offline-probe gated-robust command-prefix trie (`evaluate_offline_probe_clock`).
2. Hazard GBM gated arm (`run_hazard_model_confirmation`, seed 0).
3. Gate union (naive OR) — NEGATIVE CONTROL only.
4. **Certified union, `loo_lcb`** (`run_certified_union_analysis`) — the
   pre-registered PRIMARY combiner (cross-fitted leave-fold-out inclusion).

## Primary (confirmatory) hypotheses

- **H1 (flagship):** the certified-union trigger (loo_lcb) certifiably beats the
  fixed deadline at the operating point on >= 1 cost cell, permutation
  p_positive <= 0.0025. Powered: Phase 0b gives the flagship cell ~0.95 power at
  n=100 (upper bound; winner's-curse-discounted, n=200 adds margin).
- **H2 (P1 vs published SOTA):** the command-prefix trie certifiably beats the
  tool-name (Continuum-class, `command_field=None`) estimator (permutation cert).

## Secondary / exploratory (NOT confirmatory)

- Broader cost-family cells beyond the flagship. Phase 0b: with-replacement power
  is optimistically biased and not reliably sizable from a 100-task pilot;
  cells kv=3500/4000 are underpowered at any feasible n. These are reported as
  exploratory, never as a "multi-cell certified" claim.
- Naive gate union (negative control): expected to NOT be needed; if it fails
  where certified union holds, that is the intended contrast.
- GBM-vs-trie head-to-head; gate disjointness statistic.

## Decision rules

- A result is "certified" iff its permutation p_positive <= 0.0025 on the fresh
  set. Report the permutation p-value AND effect concentration (top-1/top-3 task
  share of positive mass) with EVERY certificate — heavy-tail concentration is
  intrinsic; a certificate riding on <= 3 tasks is flagged as fragile.
- **Family-any certification is NOT admissible** as confirmatory evidence
  (Phase 0b: single-task dominance inflates it); only the pre-registered cells
  H1/H2 count.
- If H1 fails to certify on fresh data, that is an honest NEGATIVE (the gate
  correctly refuses), reported as such — not re-analyzed for a passing cell.

## Telemetry (P4)

Collection preserves the existing `tool_exec.data.resource_timeline` (cgroup CPU
core-seconds + network RX/TX) already emitted for exec intervals. Host load
(CPU/mem/disk-free) is sampled into the run provenance. Multi-signal S analysis
(load-dependent benefit) is EXPLORATORY future work, not part of this
confirmation.

## Analysis-time code locked here

Permutation certificate = `_permutation_simultaneous_labels` /
`paired_task_cluster_bootstrap(permutation_draws=…)` (committed 1d1e23b,
review-gated). The fresh-run confirmation path must emit the permutation label
(the frozen path currently defaults to `bonferroni_percentile`; this wiring is a
pre-collection code change, review-gated, and does not alter triggers or the
frozen pipeline — only which certificate label is read).

## Budget / integrity

User-provided budget $120 (API). Model/scaffold match the dev corpus. No
turn-capping, task-substitution, or metric cherry-picking. All runs
review-gated per CLAUDE.md before results count.
