# Handoff — current stage only

Repo `/home/chiyu/workspace/agent-sched-bench`, branch
`dev/kv-swap-profile-sweep-test`. `source .venv/bin/activate`. 8 cores.
Machine is idle; nothing running.

## 1. Optimization

Merged (`fb51797`), identity proven — byte-identical artifacts after dropping
`provenance`, and 30,000/30,000 exact against a frozen `hazard_recheck_ms`:

| Script | State |
|---|---|
| `run_wtn_stage2` | parallel, 5.23× |
| `analyze_prerestore_accounting` | parallel, 2.88× |
| `analyze_boundary_evidence_stage1` | parallel, 2.87× |
| `adjudicate_k2_recheck` | parallel, 1.12× |
| `analyze_pressure_headroom` | parallel code present, **UNREVIEWED** — see §2 |
| `analyze_prior_calibration` | single-core |
| `replay_online_gate` | parallel, 3.78× |

Speedups measured on a 50-task subset. Dead ends, don't repeat: an exact
O(N log N) `hazard_recheck_ms` is **not** byte-identical (13/30,000 tie-break
divergences); the byte-identical batched version bought 6% because hazard is
not the bottleneck. Details in `CODEX-PERF-BRIEF.md`.

## 2. Probe — footprint pricing screen COMPLETE

`scripts/analyze_pressure_headroom.py` and
`tests/test_analyze_pressure_headroom.py` remain uncommitted with the reviewed
method, parallel fold scoring, fail-closed final-manifest pin, and subset
warning.

Acceleration was measured post-fix on the same 50-task subset:
- workers=1: 76.33 s
- workers=8: 51.38 s (**1.49×**)
- JSON was identical after removing only `provenance`; `.json.zst` sidecars
  were byte-identical.

Mandatory review found and fixed two major correctness issues before the run:
tool-level prior caches aliased distinct nodes, and `--final` accepted any
self-consistent 277-task manifest. Re-review then found contradictory POWER
RULE provenance/UNDERPOWERED wording; it was corrected and the gate was
declared clean. Relevant suite: 118 passed.

The reviewed full-corpus run completed in 2215.90 s:
- artifact: `analysis/pressure-headroom-2026-07-20.{json,md}`
- sidecar: `analysis/pressure-headroom-2026-07-20-decisions.json.zst`
- corpus: 277 tasks, 13,410 calls, 10 kv cells, 134,100 decision rows
- headline kv3500: 8.75 s/277, simultaneous CI [-85.00, 101.90] s/277,
  permutation inconclusive, fixed λ̄ 5740 ms, early-fire fraction 0.0595
- frozen bar: 156 s/277; CI upper bound is below it
- verdict: **DROP**, direction closed for self-footprint pricing
- kv5000 secondary: -31.11 s/277, inconclusive, non-binding

This result does not bound multi-tenant contention; that axis remains a
structural negative on this corpus. DROP does not trigger a certified decision
replay.

## 3. Online gate (W3-4) — COMPLETE, mixed/negative result

The reviewed corpus/statistics unit landed in `d9bba39`; reviewed outer-fold
parallelism landed in `6b84dba`. The final replay completed with
`--final --workers 8`. Related suite before final: **133 passed**; ruff clean.

**What it is:** a conditional-sign-symmetry test martingale with predictable,
directional absolute-Kelly bets, thresholded by Ville's inequality. It
replay-compares the incremental C2 conditioning pair
`offline_gated_robust_trigger_ms` vs `robust_trigger_ms` on exactly the same
cross-fitted task deltas and Bonferroni tail as the frozen offline permutation
gate. It is explicitly not the banked H1 certified-union-vs-deadline pair.

**Corpus discipline now enforced:**

- Development/profile runs open only the manifest-defined SWE-ReBench-100 and
  pinned Terminal-Bench-83 outcomes.
- `--final` has no corpus override. It scores both development corpora first,
  preflights the exact pinned fresh-277 identity before opening its outcomes,
  then scores fresh-277 last as a reused certified-reference row.
- Results print only after the sidecar, Markdown, and JSON completion anchor
  publish successfully. Same-path rerun failures cannot leave a stale JSON
  anchor over mixed-run artifacts.

**Review audit:** five independent review rounds found and fixed corpus leakage,
sticky-lifecycle/revocation reporting, directional lag/order-sensitivity
omissions, trigger-pair provenance, and fail-closed publication defects. The
final review reported **CLEAR: no critical, major, or minor defects**.

**Performance evidence:** a full canonical dev profile spent 1314.52 of
1333.66 sampled seconds in `cross_fitted_decisions` (1336.64 s wall), so only
the five independent outer folds were parallelized. On the same
50-task-per-corpus performance subset at the full statistical discipline,
workers=1 took 351.61 s and workers=8 took 92.99 s (**3.78x**). JSON was exact
after removing provenance, the zstd sidecar was byte-identical, and Markdown
differed only in generated time. Independent performance review: **CLEAR**.

**Final replay (commit `6b84dba`, `--final --workers 8`):**

- SWE-100 development: 9/10 agreement, all nine concordances null; kv500 is
  offline harmful / online continue.
- TB-83 development: 10/10 agreement, all null.
- fresh-277 reused reference: 7/10 agreement; concordant harmful at kv2500 and
  kv3000, concordant null at five cells, and offline-inconclusive /
  online-harmful disagreements at kv1500, kv2000, kv3500.
- No online positive certification on any corpus/cell; no revocation/lapse.

This is a mixed/negative characterization. It does not show earlier offline-gate
reproduction on development data and must not be sold as a shippable online
replacement. It leaves the paper's C2 claim resting on the banked WTN and
offline permutation-gate evidence.

The old 50-task claim ("zero certifications, effective n 1–19, threshold 400
unreachable") remains **RETRACTED**. It used a superseded trace root and must not
be restated.

## 4. Next

1. Commit the final JSON/Markdown and these status updates; keep the decisions
   sidecar local/gitignored.
2. Move to the C1 critical path: W5-7 real multi-tenant harness with live
   pre-restore, then W8-9 head-to-heads.
3. Keep the old wrong-root 50-task result retracted. The later 50-task-per-corpus
   run is performance validation only.

Two rules still apply: verify inputs against the manifest or pinned task list
that defines them, and a review the author commissions is not a review.
