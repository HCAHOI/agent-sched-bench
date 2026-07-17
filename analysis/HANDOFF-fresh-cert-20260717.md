# Handoff — fresh-corpus certification in flight (2026-07-17)

Read this first to resume after compaction. The expensive fresh-corpus
collection is DONE; the **certification pipeline is mid-run** (Step 2 of 9).
Companion docs: `analysis/fresh-corpus-preregistration-20260716.md` (the locked
protocol + amendments), `analysis/HANDOFF-tool-time-campaign-20260716.md` (the
whole campaign), memories `kv-swap-eval-state`, `rho-operating-point-directive`,
`orchestration-preference`.

## 1. Fresh corpus (DONE, analysis-ready)

- `traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200/` — **277 tasks**,
  one `attempt_*/trace.jsonl` each. seed 42, skip 150 (extended toward
  `--sample 300`; credits ran out ~task 288 → 277 valid after cleanup).
- **Disjoint** from all dev roots (verified: `shuffled[150:450] ∩ dev[0:150]=0`).
- Production extractor `trace_collect.tool_latency_dataset.extract_tool_latency_
  samples` processes all 277 with **0 errors, 13,410 tool-call samples**
  (~3x the dev-100 corpus). Structure byte-identical to dev
  (`resource_timeline` on 100% of exec calls; `llm_call.prompt_tokens` 100%).
- Download archive being built at `/home/chiyu/workspace/fresh-corpus-277-traces.tar.zst`.

## 2. Collection fix (COMMITTED)

The ~5% docker-commit task failures under concurrency were root-caused: the
fixed-image derivative (`chown /testbed` + `docker commit`) raced on overlayfs.
Fix = `ensure_fixed_image` PASSTHROUGH (pull source, run it directly as root; no
build/commit) in `src/harness/container_image_prep.py`, + 3 simulator
`remove_image` guards (skip when fixed==source), dead-code removal, test rewrite.
Committed **0bb59ac** (+ pushed). Review-gated APPROVE. Traces confirmed
undegraded by the change.

## 3. THE CERTIFICATION PIPELINE — where it is + how to finish

Operating point **rho = 0.94** (measured; see rho-operating-point-directive).
Certificate = the **permutation** (paired sign-flip) cert — a cell CERTIFIES iff
`permutation_p_positive <= 0.0025` (Bonferroni family tail). Do NOT use the
percentile `simultaneous_label` (~2.7x anticonservative; Phase 0). Draws 20000.

PRIMARY hypotheses:
- **H1**: certified-union trigger (loo_lcb) certifiably beats the fixed deadline.
- **H2 (P1)**: command-prefix trie certifiably beats the tool-name (Continuum-
  class, `command_field=None`) estimator.
Decision rule: report top-1/top-3 task effect-concentration with each cert; a
cert riding on <=3 tasks is flagged FRAGILE.

Base output dir: `analysis/fresh-corpus-certification-20260717/` (`$B`).
Drivers refuse a pre-existing `--output-root` (don't pre-create leaf dirs).

**Done:** Step 0 (`$B/offline-gated-robust/task_ids.txt`, 277), Step 1
(`$B/offline-gated-robust/manifest.json`), Step 9 wrapper written
(`scripts/analyze_frontier_permutation.py`, ruff-clean — permutation cert for
one paired trigger contrast on frontier decisions).

**In flight (PARALLELIZED into 2 chains for CPU efficiency):** the pipeline runs
as TWO concurrent detached `nohup` chains (H1 chain steps 2-7 is a strict
dependency chain; H2 frontier steps 8-9 are INDEPENDENT — inputs only
`--config-manifest` + `--trace-root` — so they run in parallel):
- Chain A (H1) = `$B/chain_A_h1.sh` -> `$B/chain_A.log` (steps 2,3,4,5,6,7).
- Chain B (H2) = `$B/chain_B_h2.sh` -> `$B/chain_B.log` (steps 8,9).
Launched with `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2` (GBM uses ~4 cores;
~5 cores active vs 1 sequential). Monitor `brvv317p7`. This changes NO numerics
(same seeds, same drivers) — only the process layout.
Each chain writes to its own `--output-root`s (no collision). Completion markers:
`CHAIN_A_H1_DONE` / `CHAIN_B_H2_DONE`.

**To CHECK / FINISH:** `grep -E "CHAIN_._H._DONE|FAILED" $B/chain_A.log $B/chain_B.log`.
When BOTH show DONE, read verdicts from the artifacts (§H1/H2 below). If a chain
FAILED or died, `rm -rf` that step's partial `--output-root` (drivers refuse a
pre-existing output dir; step 2 refuses existing `results/`) and relaunch that
chain: `export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2; nohup bash
$B/chain_A_h1.sh > $B/chain_A.log 2>&1 &` (and/or chain_B). The chains run (all
with `--restore-cost-fractions 0.0,0.94 --replicates 50000 --seed 0`):
0. `run_offline_gated_robust_confirmation.py --manifest $B/offline-gated-robust/manifest.json --output-root $B/offline-gated-robust/results`  (step 2, chain A)
1. `run_restore_cost_mode_b.py --confirmation-root $B/offline-gated-robust/results --output-root $B/restore-cost-mode-b`
2. `run_within_task_baseline.py --confirmation-root .../results --mode-b-root $B/restore-cost-mode-b --output-root $B/within-task-gated`
3. `run_hazard_model_confirmation.py --confirmation-root .../results --mode-b-root $B/restore-cost-mode-b --gated-b1-root $B/within-task-gated --num-intervals 40 --model-family gbm --feature-set full --ensemble-members 0 --output-root $B/hazard-model-gbm-full`
4. `analyze_certified_union.py --hazard-root $B/hazard-model-gbm-full --inclusion-criterion loo_lcb --output-root $B/certified-union-loo-lcb`
5. **H1**: `analyze_gate_robustness.py --decisions $B/certified-union-loo-lcb/rho_0.94_decisions.jsonl --restore-cost-fraction 0.94 --replicates 20000 --seed 0 --output $B/gate-robustness/gate_robustness_rho094.json` (MUST override both defaults — they point at the committed dev rho_1.0 file).
6. `run_benchmark_frontier.py --config-manifest $B/offline-gated-robust/manifest.json --trace-root traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200 --tool-name-trie --num-intervals 40 --model-family gbm --feature-set full --ensemble-members 0 --exposure-note "..." --output-root $B/frontier-p1`
7. **H2**: `analyze_frontier_permutation.py --decisions $B/frontier-p1/rho_0.94_decisions.jsonl --treatment-field offline_gated_robust_trigger_ms --baseline-field offline_gated_tool_name_trigger_ms --restore-cost-fraction 0.94 --replicates 20000 --seed 0 --output $B/frontier-p1/permutation_p1_rho094.json`
The driver prints the H1/H2 verdicts at the end (grep `cert_driver.log` for "VERDICTS").

**Verdict artifacts:**
- H1 = `$B/gate-robustness/gate_robustness_rho094.json` →
  `deployed_certificate_task_cluster["<cost>"].permutation_p_positive` /
  `.permutation_label` / `.paired_delta_ms`. Certified iff any cell p<=0.0025.
- H2 = `$B/frontier-p1/permutation_p1_rho094.json` → per-cost
  `permutation_p_positive`. Certified iff any cell p<=0.0025.

## 4. AFTER the pipeline (do NOT skip)

1. Compute effect concentration (top-1/top-3 task share of positive mass) for any
   certified cell — flag if <=3 tasks carry it. (Re-call
   `paired_task_cluster_bootstrap` and read `task_contributions`, or a small helper.)
2. **MANDATORY review gate** (CLAUDE.md): spawn a FRESH opus code-reviewer to
   verify — leak-free cross-fitted inclusion (loo_lcb decides each fold on OTHER
   folds), freshness/disjointness, rho=0.94 applied throughout, the permutation
   cert computed correctly, and the new `analyze_frontier_permutation.py` wrapper.
   Results do NOT count until this clears. (Orchestration preference: delegate
   implement+review to subagents; Claude coordinates.)
3. Write a findings doc `$B/findings.md` + commit; update `kv-swap-eval-state`
   memory with the H1/H2 verdicts.

## 5. Live processes / monitors at handoff time

- **Certification = 2 parallel nohup chains**: `chain_A_h1.sh` (steps 2-7 ->
  `chain_A.log`) + `chain_B_h2.sh` (steps 8-9 -> `chain_B.log`). Monitor
  `brvv317p7` (fires on both-DONE with verdicts / FAILED / died; ~10-min progress
  lines). Gotcha: `chain_B_h2.sh` header MUST define `FR=traces/swe-rebench/
  qwen3.7-max/fresh-seed42-skip150-n200` (step 8's `--trace-root $FR`).
- Download archive DONE: `/home/chiyu/workspace/fresh-corpus-277-traces.tar.zst`
  (15M, single dir, 277 tasks, integrity-verified).
- Collection is DONE; all collector/prune/collection-monitor processes stopped.
- The executor subagent that started this went idle then mis-reported step 2 as
  killed — do NOT rely on it; the pipeline is now driven from the MAIN loop.

## 6. Gotchas

- rho=0.94 needs freshly generated decision files at fraction 0.94 (committed dev
  files are {0,0.25,0.5,1.0}); the whole chain must pass `--restore-cost-fractions
  0.0,0.94`. `analyze_gate_robustness.py`/analyze_frontier defaults point at dev
  rho_1.0 — override `--restore-cost-fraction 0.94` and `--decisions`.
- loo_lcb (conservative), NOT loo_point (dev used loo_point).
- Hazard params must match dev-100: num-intervals 40, gbm, full, ensemble 0, seed 0.
- The bulky per-fraction decision JSONLs are reproducible; commit aggregates +
  findings, not the giant decision files.
- **CRITICAL run-lifecycle lesson:** long compute (the 50k-replicate steps, ~50
  min each) MUST be launched via a detached `nohup ... &` from the MAIN loop
  (like the collection). Do NOT run them inside a subagent's Bash tool / harness
  `run_in_background` — those get killed when the subagent idles/ends (this
  killed the first step-2 attempt after 50 min at fold 2). nohup-from-main-loop
  survives across compaction and subagent lifecycles.
