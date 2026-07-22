# W3-4 online gate vs offline permutation gate (replay)

> **FINAL - complete corpus**
>
> Anytime-valid sign-symmetry test martingale (Ville; Ramdas et al. 2023) over task-clustered paired regret, replayed against the FROZEN offline permutation gate on identical cross-fitted deltas. Both gates test the SAME null (sign symmetry of the paired deltas). Generated 2026-07-20T21:46:25 (git 6b84dba4238b6dd72b691117ede738f9e0a6b97d).

Directional bet rule: `directional_abs_kelly_plug_in_truncated_one_past_scale_unit`. The positive e-process uses the absolute magnitude of the truncated signed Kelly plug-in; the harmful e-process uses its negative.

One-sided tail 0.0025 (= 0.05 / (2 x 10 kv cells), identical to the offline gate's Bonferroni family tail; e-value threshold 400), rho=0.94, guard 0ms (threshold==kv).

Comparison: treatment `offline_gated_robust_trigger_ms` vs baseline `robust_trigger_ms`. This incremental C2 conditioning comparison is not the banked H1 policy-vs-deadline pair.

Reference replay order: `unicode_lexicographic_task_id`; seeded permutations start from that order.

## swe-rebench-100 — development corpus

Corpus provenance: committed manifest -- `/home/chiyu/workspace/agent-sched-bench/traces/swe-rebench/qwen3.7-max/offline-gated-confirm-100-v2`, manifest `analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100-20260713/manifest.json`, expected task count 100.

100 tasks, 4640 calls. Agreement 9/10 kv cells, of which **0 concordant-positive, 0 concordant-harmful, 9 concordant-null** (both gates said nothing).

> **This row has no informative concordance.** Its agreement count contains only cells where both gates declined to decide. Read the one-way `ruled out` upper-bound diagnostic and achieved `logE+`; clearing the upper bound does not prove certification was attainable.

Effective n per cell (tasks with a NONZERO paired delta): 15-48 of 100.

Disagreement directions: offline=harmful/online=continue x1.

| kv | eff n | logE+ / logE- (thr) | upper bound | ruled out | total ms | offline | online lifecycle (instant) | agree | first + / - @task | revoked | lapses | seed + hit/n; min/med/max | seed - hit/n; min/med/max | seed final C/H/N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | 15 | -16.72 / 4.45 (5.99) | 9.7 | no | -628 | harmful | continue (continue) | NO | - / - | no | 0 | 0/50; never | 7/50; 77/88/100 | 0/7/43 |
| 1000 | 32 | -16.27 / -41.63 (5.99) | 21.5 | no | 4124 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 10/50; 43/54/65 | 0/10/40 |
| 1500 | 48 | -0.59 / 0.53 (5.99) | 32.6 | no | 1467 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 17/50; 18/31/46 | 0/17/33 |
| 2000 | 41 | -0.12 / -4.98 (5.99) | 27.7 | no | 1695 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 5/50; 28/37/53 | 0/5/45 |
| 2500 | 41 | -20.04 / -75.11 (5.99) | 27.7 | no | 7364 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 10/50; 33/42/51 | 0/10/40 |
| 3000 | 46 | -50.81 / -11.61 (5.99) | 31.2 | no | 17855 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 1/50; 27/27/27 | 0/1/49 |
| 3500 | 32 | -54.05 / -3.89 (5.99) | 21.5 | no | 341 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 18/50; 36/51/74 | 0/18/32 |
| 4000 | 30 | -212.83 / -1.75 (5.99) | 20.1 | no | 23158 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 0/50; never | 0/0/50 |
| 4500 | 19 | -29.00 / -0.85 (5.99) | 12.5 | no | -6624 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 4/50; 55/74/75 | 0/4/46 |
| 5000 | 37 | -4.66 / -6.33 (5.99) | 25.0 | no | -12607 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 3/50; 37/41/47 | 0/3/47 |

## terminal-bench — development corpus

Corpus provenance: pinned trace root (no manifest) -- `traces/terminal-bench/zai-org-GLM-5.2/20260709T171830`, 100 trace files, task set pinned by `analysis/tool-time-frontier-terminal-bench-20260715/folds`.

> WEAKER PROVENANCE than swe-rebench-100: no committed manifest, so no expected_task_count or freshness attestation. Task set pinned to the union of f{1..5}_eval.txt from the committed TB frontier analysis (83 ids), the set every committed TB result used.

83 tasks, 2959 calls. Agreement 10/10 kv cells, of which **0 concordant-positive, 0 concordant-harmful, 10 concordant-null** (both gates said nothing).

> **This row has no informative concordance.** Its agreement count contains only cells where both gates declined to decide. Read the one-way `ruled out` upper-bound diagnostic and achieved `logE+`; clearing the upper bound does not prove certification was attainable.

Effective n per cell (tasks with a NONZERO paired delta): 8-23 of 83.

| kv | eff n | logE+ / logE- (thr) | upper bound | ruled out | total ms | offline | online lifecycle (instant) | agree | first + / - @task | revoked | lapses | seed + hit/n; min/med/max | seed - hit/n; min/med/max | seed final C/H/N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | 17 | -4.37 / -3694.98 (5.99) | 11.1 | no | 8633 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 0/50; never | 0/0/50 |
| 1000 | 23 | -1.77 / 0.32 (5.99) | 15.2 | no | -5807 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 0/50; never | 0/0/50 |
| 1500 | 16 | -261.78 / -1.55 (5.99) | 10.4 | no | -4603 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 0/50; never | 0/0/50 |
| 2000 | 17 | -130.20 / -21.66 (5.99) | 11.1 | no | 90498 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 0/50; never | 0/0/50 |
| 2500 | 16 | -6.89 / -33.71 (5.99) | 10.4 | no | 126985 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 0/50; never | 0/0/50 |
| 3000 | 14 | -9.18 / -5.86 (5.99) | 9.0 | no | 131168 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 0/50; never | 0/0/50 |
| 3500 | 12 | -1.12 / -13.21 (5.99) | 7.6 | no | 173949 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 0/50; never | 0/0/50 |
| 4000 | 8 | -22.16 / 3.40 (5.99) | 4.9 | yes | -4685 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 0/50; never | 0/0/50 |
| 4500 | 13 | -0.34 / 0.01 (5.99) | 8.3 | no | 8298 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 0/50; never | 0/0/50 |
| 5000 | 11 | -129.09 / -3.70 (5.99) | 6.9 | no | 2856 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 0/50; never | 0/0/50 |

## fresh-277 (swe-rebench-qwen3.7-max-fresh-seed42-skip150-n277) — reused certified-reference corpus

Corpus provenance: frozen committed manifest -- `/home/chiyu/workspace/agent-sched-bench/traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200`, manifest `analysis/fresh-corpus-certification-20260717/offline-gated-robust/manifest.json`, expected task count 277.

> Previously used for H1 certification and later analyses; this is not newly unseen W3-4 validation data. It was not opened by development/profile runs in this lane.

277 tasks, 13410 calls. Agreement 7/10 kv cells, of which **0 concordant-positive, 2 concordant-harmful, 5 concordant-null** (both gates said nothing).

Effective n per cell (tasks with a NONZERO paired delta): 88-150 of 277.

Disagreement directions: offline=inconclusive/online=harmful x3.

| kv | eff n | logE+ / logE- (thr) | upper bound | ruled out | total ms | offline | online lifecycle (instant) | agree | first + / - @task | revoked | lapses | seed + hit/n; min/med/max | seed - hit/n; min/med/max | seed final C/H/N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | 145 | -30.73 / -61.79 (5.99) | 99.8 | no | 638 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 32/50; 25/38/74 | 0/32/18 |
| 1000 | 88 | -15.06 / -33.91 (5.99) | 60.3 | no | -693 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 38/50; 26/50/83 | 0/38/12 |
| 1500 | 150 | -79.69 / -451.91 (5.99) | 103.3 | no | 2063 | inconclusive | harmful (continue) | NO | - / 29 | no | 0 | 0/50; never | 42/50; 17/30/62 | 0/42/8 |
| 2000 | 143 | -41.96 / -131.83 (5.99) | 98.4 | no | 1927 | inconclusive | harmful (continue) | NO | - / 29 | no | 0 | 0/50; never | 47/50; 19/34/54 | 0/47/3 |
| 2500 | 123 | -38.53 / 15.07 (5.99) | 84.6 | no | -1102 | harmful | harmful (harmful) | yes | - / 98 | no | 0 | 0/50; never | 50/50; 27/60/152 | 0/50/0 |
| 3000 | 90 | -105.52 / 26.96 (5.99) | 61.7 | no | -6363 | harmful | harmful (harmful) | yes | - / 44 | no | 0 | 0/50; never | 47/50; 29/53/260 | 0/47/3 |
| 3500 | 110 | -26.87 / -19.04 (5.99) | 75.6 | no | -3855 | inconclusive | harmful (continue) | NO | - / 83 | no | 0 | 0/50; never | 27/50; 27/49/95 | 0/27/23 |
| 4000 | 116 | -702.99 / 3.43 (5.99) | 79.7 | no | -5326 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 30/50; 23/35/64 | 0/30/20 |
| 4500 | 119 | -81.19 / 3.38 (5.99) | 81.8 | no | -28040 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 27/50; 25/46/152 | 0/27/23 |
| 5000 | 105 | -8.61 / -703.98 (5.99) | 72.1 | no | 15090 | inconclusive | continue (continue) | yes | - / - | no | 0 | 0/50; never | 35/50; 26/40/85 | 0/35/15 |

Detection lag is tasks until the online gate FIRST crosses in either direction; the offline gate has no analogue (it speaks once, at the end of the corpus). Seed columns report crossings/total before the conditional lag spread and final lifecycle counts as certified/harmful/continue. Read them, not the reference-order columns, as the order-sensitivity result.

**Revocation vs lapse.** A REVOCATION is a certified cell whose harmful martingale later crossed. Ville bounds that crossing under the conditional sign-symmetry null used by both gates; it is not a guarantee for every distribution with nonnegative mean. A LAPSE is a certified cell whose instantaneous evidence merely fell below the threshold. Crossing back carries no type-I guarantee and is common under sustained positive effects in the validity tests, so a lapse never withdraws the policy. Revocation lag is exposure measured from first certification, not detection latency from when truth degraded.

**Agreement is not ten independent checks.** The kv cells re-price the same tasks at ten cost scalings, so the cells are near-collinear and move together; read the count with the disagreement direction, not as a rate out of ten. And a concordant-NULL cell is not a validated match: both gates declining to decide is agreement about nothing.

**Effective n and its one-way upper bound.** A task with paired delta exactly 0.0 moves neither gate. The first nonzero delta receives a zero bet, and every later nonzero delta contributes strictly less than log 2, so `(eff n - 1) x log 2` is a proven log-e upper bound. If that bound does not clear the threshold, certification is ruled out. If it does clear, reachability remains unknown under the predictable truncated bets; only the achieved `logE+`/`logE-` is observed evidence.

**Lifecycle state vs instantaneous evidence.** `agree` uses the sticky lifecycle state: certification remains deployed through a mere lapse, and changes only when the opposite martingale crosses; a later certification can deploy again. The parenthesized table value and `online_instantaneous_final_label` retain the terminal evidence diagnostic without letting an uncalibrated lapse withdraw policy.

**Scope.** This is a retrospective replay of cross-fitted deltas against the offline verdict. It does NOT establish earlier offline-verdict reproduction or prospective online deployment; a prequential variant (prior frozen on a burn-in prefix) is future work.
