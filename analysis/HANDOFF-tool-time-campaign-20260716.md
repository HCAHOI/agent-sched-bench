# Tool-Time Scheduling Campaign — Handoff (2026-07-16)

Standalone handoff before context compaction. Read this to pick up the whole
campaign. Branch: `dev/kv-swap-profile-sweep-test` (cloud-provider-only;
GPU/vLLM tooling was removed from it — see src/trace_collect/CLAUDE.md; do
NOT re-add `--vllm-*/--kv-*/--gpu-*` CLI flags). Everything below is
committed and pushed to origin (HEAD e0ecc3c at write time).

Companion running log (`tool-time-formulation-critique-20260714.md`) was
deleted 2026-07-20 in the doc cleanup; recover from git if the full
execution order is ever needed. Related work:
`analysis/related-work-synthesis-20260715.md`. Mechanism drill-down:
`analysis/tool-time-mechanism-analysis-20260715.md`. Memory:
`kv-swap-eval-state`, `rho-operating-point-directive`.

---

## 1. The problem

Proactively decide WHEN (if ever) to initiate a resource action (KV-cache
swap-out) during an agent's tool call, so the transfer is hidden off the
critical path. Predict from causally-available context (tool identity +
command prefix + within-task history) whether a call will run long enough.
The deployed output is an action TRIGGER k(x,a): fire the action iff the
call is still running at k. Reframed from binary long/short classification
to this decision-focused optimal-stopping form.

Target generalization: M(t, S | P, A) — multiple resource signals S
(CPU/mem/disk/accelerator, sequence state) and multiple correlated actions
A (not just KV swap).

Data: real tool traces from three benchmarks, ALL development-exposed
(sensitivity only, never fresh certification):
- SWE-ReBench (frozen 100-task confirmation set; the powered corpus; ~4640
  tool calls; has fed ~11 analyses)
- Terminal-Bench (100 traces, 83 tasks with samples; underpowered; carries
  `tool_exec.data.resource_timeline` telemetry NOT yet used)
- ScienceAgentBench Verified (102 tasks but tool-sparse, ~1.7 calls/task;
  structurally uninformative for call-boundary triggers)

---

## 2. THE OPERATING POINT (read this before quoting any number)

**rho = KV swap-back cost / swap-out cost was MEASURED = 0.94** on H100
Gen5x16 (flat across 48MB-55GB and both KV dtypes; bandwidth-bound ~54/57
GB/s; robust to GPU-compute and host-memory contention). See
`analysis/tool-time-rho-measurement-20260715/`.

DIRECTIVE (user, hard rule — see memory `rho-operating-point-directive`):
- The low-rho sweep {0, 0.25, 0.5, 1.0} was a PROBE. It revealed that
  restore cost matters (finding F1: rho=0 manufactures success) and pointed
  us to measure rho. That job is DONE.
- New experiments run at rho=0.94 SINGLE point. rho stays a code PARAMETER
  (fresh corpus / other hardware re-measures it) but is NOT swept, NOT
  specially analyzed, NOT featured. Do not hardcode 0.94 (magic number).
- Existing low-rho artifacts stay as the probe record; cite once as
  sensitivity, never as deployment claims.
- The committed sweeps used {0,0.25,0.5,1.0}; rho=1.0 is the operating-point
  PROXY for 0.94 (6% gap) until fresh runs use 0.94 by construction.
- Deployment claims REQUIRE rho=0.94 AND Mode-B (refit-at-operating-point)
  triggers. Mode-A (frozen rho=0-fit triggers scored at 0.94) is a lower
  bound only — do not conflate (this bit us once: P2's -32.9s Mode-A vs
  the +118s Mode-B for the same policy).

---

## 3. The architecture / seam (how everything composes)

One contract underlies every policy: a predictor produces, per call, "a
representation of P(latency | context)" — the empirical trie produces a
list of latency samples; the learned model produces a discrete survival
curve S(t|x) + interval masses. The SAME utility functional
(`_utility_matrix` in tool_latency_utility_clock.py) turns that into a
trigger; an equal-weight mass vector reproduces the empirical policy
exactly (proven by anchor tests). Downstream is predictor-agnostic:
- trigger derivation: `hazard_recheck_ms` / `robust_utility_trigger_stats`
  (trie), `survival_trigger_ms` (learned) — utility-optimal, restore-aware.
- CERTIFICATION GATE: `select_probe_guard` picks a cross-fitted margin
  guard on inner-OOF profile tasks; fire early only if the margin certifies.
  This gate is the ONE component that behaved well in EVERY experiment.
- `paired_task_cluster_bootstrap` (tool_latency_confirmation.py): task-
  clustered paired bootstrap, Bonferroni-simultaneous over the cost family.

Restore cost rho enters ONLY the utility functional (S(t|x) is rho-
independent), so a model is fit ONCE and every cost/rho is a cheap
functional — this amortization is why the factorization matters and why P2
(second action) is nearly free.

---

## 4. Findings arc (operating point rho=0.94, Mode-B unless noted)

1. F1 / E1 — restore cost is mandatory. rho=0 manufactures success (free
   unnecessary swaps). Charging rho on fires-on-short kills the artifact.
   [restore-cost-sweep-20260714]
2. Mode B — refit with rho in the objective: the cross-task gated trie
   holds +118 s vs deadline at rho=0.94. [restore-cost-mode-b-20260714]
3. B1 within-task baseline — task-local history is strong at rho=0 but
   NET-NEGATIVE at rho=0.94 (-398 s); its cross-fitted guard collapses to
   the deadline. Signal is real but can't self-certify.
   [within-task-baseline / within-task-gated-20260714]
4. E2 transfer — SWE-ReBench-fitted priors do NOT transfer (Terminal-Bench
   net-negative; the GATE bounds the damage while ungated clocks are
   catastrophic). ScienceAgentBench neutral (tool-sparse).
   [transfer-*-20260714]
5. Hazard model — a pooled logistic hazard LOSES to the trie (calibrated
   but not SHARP; L2 shrinkage). A GBM (HistGradientBoosting) WINS at
   rho<=0.25 but at rho=0.94 loses to the trie head-to-head. [hazard-model-
   *-20260714, hazard-model-gbm-full-20260715]
6. Ensemble (M=5 task-jackknife unanimity) — REFUTED: loses to the single
   GBM; unanimity-to-fire hedges too hard across estimator families.
   [hazard-ensemble-gbm-20260715]
7. Mechanism analysis — all policy divergence is on EXEC calls; per-tool
   switching headroom is exactly ZERO; the two gates (trie, GBM) fire on
   largely DISJOINT calls. [mechanism-analysis-20260715]
8. Gate union (naive OR: fire when either gate opens, min of triggers) —
   DOMINATES both parents on SWE-ReBench (+136 s vs deadline, 1 cert; +112
   vs GBM; +18 vs trie at rho=0.94) but FAILS on Terminal-Bench (inherits
   the harmful trie's misfires). [gate-union-*-20260715]
9. Certified union — OR only gates that certify vs deadline on the fitting
   partition (cross-fitted leave-fold-out; leak-free). SWE-ReBench: both
   certify -> == naive == winner. TB (loo_lcb): can't certify either (83
   tasks) -> reverts to deadline, SAFE (beats naive union's catastrophe by
   +257..+379 s). KEY: NEVER has the naive union's cross-workload failure;
   its only cost is conservatism on underpowered corpora = a DATA problem.
   [certified-union-*-20260715]
10. rho MEASURED = 0.94 + E5 contention probe (robust to GPU/host).
    [rho-measurement-20260715]
11. P1 tool-name baseline (Continuum's estimator class, command_field=None)
    — our command-PREFIX trie certifiably beats it on SWE-ReBench (+158.7 s,
    1 cert) and in point estimate on TB (+46.4 s). The learned GBM is NOT a
    clean further win over tool-name (2 harmful cells). So the demonstrated
    value is command-prefix conditioning, not learning per se — matches
    Continuum's own Fig 5 cd-tail pathology. [tool-name-baseline-*-20260715]
12. P2 reload-vs-recompute (second action; restore = min(rho*kv,
    rate*context)) — EXACT causal context length (llm_call prompt_tokens,
    4640/4640). At rho=0.94: recompute certifiably cuts restore cost
    (+101.8 s, 4 cert at cheap prefill); recovers swap-only's -32.9 s deficit
    to +68.9 s vs deadline at cheap prefill (uncertified). Whether it helps
    hinges on ONE unmeasured number: the H100 prefill-vs-context curve.
    [recompute-restore-swe-rebench-20260715]

---

## 5. The thesis (what generalizes)

There is NO universal best policy across workloads. What generalizes is the
MECHANISM: fit per deployment, certify each component, combine only the
certified ones, fall back to the conservative deadline otherwise. The
certification gate is the load-bearing arbiter. Related work confirms this:
Continuum (fitted per-tool-name predictor) and ThunderAgent (prediction-
refusing lifecycle scheduler; Theorem F.1 memorylessness = the null our gate
tests) are the two un-arbitrated extremes our gate unifies. Conditioning
spectrum: memoryless (ThunderAgent) -> tool-name (Continuum) -> command-
prefix (ours, the certifiable step) -> learned (no clean further gain at
this scale). Neither paper certifies; neither charges honest restore cost;
neither takes proactive in-call swap-out — all three are ours.

---

## 6. Code map (all on the branch, committed)

Core seam:
- tool_latency_utility_clock.py — `_utility_matrix` (rho + restore-aware
  utility), `robust_utility_trigger_stats` (trie LOTO-unanimity), public
  `utility_matrix` passthrough, `validate_restore_cost`.
- tool_latency_profiled.py — trie (`build_latency_prior`,
  `latency_prior_hierarchy`, `hazard_recheck_ms`).
- tool_latency_offline_probe.py — `evaluate_offline_probe_clock` (fit +
  cross-fitted guard), `select_probe_guard`, `mean_clock_region_stats`,
  public `balanced_task_folds`. command_field=None => tool-name-only nodes.
- tool_latency_confirmation.py — `paired_task_cluster_bootstrap` (+ optional
  per-row `baseline_/treatment_restore_cost_ms_field` for P2).

Predictors / actions:
- tool_latency_within_task.py — B1 (completion-gated causal history: a call
  enters history only when tool_ts_end <= current tool_ts_start; 4.2% of
  calls overlap so start-order would leak).
- tool_latency_survival_features.py, tool_latency_hazard_model.py,
  tool_latency_hazard_eval.py — learned discrete-time hazard (logistic +
  GBM families; person-period expansion; log grid; amortized across rho).
- tool_latency_recompute.py, tool_latency_context.py — P2 recompute cost +
  context-length join (prompt_tokens from same-iteration llm_call).

Drivers (restore_cost_analysis.py + scripts/):
- run_mode_b_refit, run_within_task_baseline, run_hazard_model_confirmation,
  run_gate_union_analysis, run_certified_union_analysis,
  run_recompute_restore_sweep; benchmark_frontier.run_benchmark_frontier
  (within-corpus trie-vs-GBM + `--tool-name-trie` P1 arm).
- scripts/measure_kv_swap_cost.py — rho measurement (gpu extra: vllm).
  NOTE: its auto-provenance (device/pcie) serializes null — fix before reuse.

Everything is review-gated (a fresh reviewer subagent per feature before any
recorded run) per CLAUDE.md. Anchor tests prove equivalence to the empirical
policy. Bulky per-fraction decision JSONLs are NOT committed (reproducible);
aggregate JSONs + summary.md + findings.md ARE.

---

## 7. Open items / next steps (ranked)

1. FRESH-CORPUS CERTIFICATION — the one thing that converts every result
   above from sensitivity to certified. Needs an LLM API key on a box
   (OpenRouter key was missing on the last host). Run at rho=0.94 by
   construction. Pre-registered PRIMARY hypothesis: certified union
   (loo_lcb) vs deadline / GBM / trie / naive union; naive union = negative
   control; also the command-prefix-beats-tool-name (P1) contrast and the
   gate-disjointness statistic. INSTRUMENT IT with host/GPU/queue telemetry
   at collection time (proposal P4) so multi-signal S and load-dependent
   benefit become testable — decide BEFORE collecting.
2. P2 prefill-vs-context measurement — ONE cheap (~10 min) H100 measurement
   picks the recompute operating point; do it on the SAME box as any fresh
   collection. Then P2 Mode-B refit (deploy a recompute-aware trigger, not
   just the current lower bound).
3. P3 exponential-threshold prior-free arm — ThunderAgent's Theorem F.1
   family as the policy the gate deploys when nothing certifies; connects
   the certified union to learning-augmented ski-rental.
4. Terminal-Bench resource_timeline telemetry — the unused S signal already
   on disk; first concrete multi-signal step.

---

## 8. Lessons / gotchas

- rho DISCIPLINE (section 2) — do not quote low-rho as deployment; use
  Mode-B triggers.
- GPU: keep unattended jobs simple and verify liveness, not just launch
  (a background-thread vLLM harness once wedged the box). Platform-specific;
  the next box may differ.
- Subagents in this session reliably went idle WITHOUT delivering their
  report; a follow-up "send your report" nudge always retrieved it. Expect
  this.
- Two heavy analysis runs in parallel locally just contend for CPU; serialize.
- One pre-existing unrelated test failure: tests/test_openclaw_runtime_
  selection.py (SimpleNamespace.select_window) from uncommitted collector.py
  work predating this campaign — not ours, untouched.
- benchmark_frontier recompute (seed 0) reproduces frozen trie triggers at
  rho=0 by construction but the GBM is refit (not byte-identical to the
  frozen GBM artifact); internally consistent, don't cross-equate artifacts.
