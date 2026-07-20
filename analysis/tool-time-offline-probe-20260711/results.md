# Nested Offline-Probe Utility Guard Results

Date: 2026-07-11

Status: exploratory result on three analyst-exposed corpora. The proposed
point-estimate guard is rejected for adoption.

## Executive Result

The experiment answers two separate questions:

1. **Can a general calibration procedure produce deployment-specific
   thresholds without reading dataset identity?** Yes. The same code learned
   substantially different guards from each outer profile, including a
   `never early` result for one Terminal-Bench fold.
2. **Does this particular calibrated guard transfer safely to held-out
   tasks?** No. It improved the aggregate sweep on SWE-Rebench and
   ScienceAgentBench, but failed badly on Terminal-Bench because a sparse
   command node did not generalize across tasks.

The negative result is informative: a call-weighted point estimate is not an
adequate uncertainty model when calls are clustered within tasks. Offline
sampling remains useful, but the next method must treat task identity as the
statistical unit for confidence, while retaining call-total utility as the
systems estimand.

## Learned Guards

The guard is dimensionless predicted utility margin divided by action cost.
Each value was learned only from the corresponding outer profile's four-fold
task-OOF probe. `none` means the predeclared fail-closed `never early`
candidate won.

| Corpus | f1 | f2 | f3 | f4 | f5 |
|---|---:|---:|---:|---:|---:|
| SWE-Rebench | 0.000497 | 0.012859 | 0.005249 | 0.008182 | 0.000734 |
| Terminal-Bench | 0.257261 | 1.093496 | 1.329193 | 0.638234 | none |
| ScienceAgentBench | 0.003454 | 0.015350 | 0.006909 | 0.004996 | 0.005422 |

This variation is an outcome of sampled-call statistics. The policy does not
receive a corpus name, and no threshold from the earlier 100 ms sweep is an
input or candidate.

## Sweep Summary

The normalized sum is `sum_cost(delta_vs_deadline_ms / cost_ms)`. The raw sum
adds milliseconds across alternative cost settings and is only a compact
sweep summary, not a simultaneously deployable reward.

| Corpus | Policy | Positive / negative points | Normalized sum | Raw delta sum (ms) | Worst point (ms) |
|---|---|---:|---:|---:|---:|
| SWE-Rebench | mean hazard | 8 / 2 | 383.108 | 766,998 | -14,850 |
| SWE-Rebench | robust clock | 9 / 1 | 292.827 | 670,182 | -1,681 |
| SWE-Rebench | offline probe | 8 / 2 | **387.626** | **774,077** | -9,499 |
| Terminal-Bench | mean hazard | 2 / 8 | -650.473 | -2,226,923 | -468,188 |
| Terminal-Bench | robust clock | 4 / 6 | **-35.749** | **-152,785** | **-94,424** |
| Terminal-Bench | offline probe | 4 / 5 | -231.245 | -910,402 | -292,153 |
| ScienceAgentBench | mean hazard | 8 / 2 | 167.356 | 472,009 | -7,572 |
| ScienceAgentBench | robust clock | 7 / 3 | 91.383 | 237,766 | -30,028 |
| ScienceAgentBench | offline probe | 9 / 1 | **178.788** | **498,441** | **-6,826** |

Across all three exposed corpora, robust clock has the larger normalized sum
(`348.461` versus `335.169`) and a substantially better worst point. The
offline-probe arm therefore does not replace the robust baseline.

These comparisons are descriptive. This experiment did not preregister or
compute task-bootstrap confidence intervals, so the point estimates must not
be presented as statistically significant differences.

Outer-fold transfer is also heterogeneous: normalized outer objectives are
positive in SWE-Rebench 5/5 folds, in ScienceAgentBench 4/5 folds (f2 is
`-15.827`), and in Terminal-Bench only 1/5 folds (f1 `-228.885`, f2 `+0.371`,
f3 `0`, f4 `-2.731`, f5 `0`). Pooled SWE/SAB gains are therefore not evidence
of uniform task-level transfer.

## Offline-Probe Delta by Cost

Each cell is held-out outer-OOF `delta_vs_deadline_ms`.

| Cost (ms) | SWE-Rebench | Terminal-Bench | ScienceAgentBench |
|---:|---:|---:|---:|
| 500 | +113,404 | -1,481 | -6,826 |
| 1,000 | +11,456 | +14,546 | +62,633 |
| 1,500 | -7,877 | +1,286 | +27,691 |
| 2,000 | -9,499 | +1,293 | +26,959 |
| 2,500 | +14,756 | +6,237 | +24,369 |
| 3,000 | +46,890 | -171,345 | +48,612 |
| 3,500 | +86,525 | -212,634 | +49,132 |
| 4,000 | +109,469 | -256,153 | +46,471 |
| 4,500 | +177,800 | -292,153 | +107,038 |
| 5,000 | +231,152 | 0 | +112,361 |

The zero at Terminal-Bench 5,000 ms is not a selected favorable point. The
only fold that would otherwise dominate this region fired early at lower
costs but selected no eligible 5,000 ms candidate; fold 5 independently chose
the global `never early` guard.

## Terminal-Bench Failure Case

The pooled Terminal-Bench loss is localized rather than diffuse:

- Outer fold 1 contributes `-907,048 ms` summed across the ten cost settings;
  folds 2--5 contribute `+186`, `0`, `-3,540`, and `0 ms` respectively.
- At costs 3,000 / 3,500 / 4,000 / 4,500 ms, fold 1 has respectively
  76 / 75 / 73 / 73 actual early triggers, of which 71 / 72 / 73 / 73 are
  early triggers on short calls.
- One held-out task, `download-youtube`, contributes 72 / 71 / 70 / 70 of
  those actual triggers.
- The dominant node is command prefix `exec:export`. At 4,000 ms the policy
  assigns its 1,685.1 ms candidate to 79 rows; 70 calls survive to that time
  and actually trigger, and all 70 finish before the 4,000 ms deadline.
- That selected profile node contains only six profile calls from two tasks,
  with three calls surviving the candidate trigger. Its predicted normalized
  margin is 0.386 at 4,000 ms, which clears fold 1's globally learned guard of
  0.257.

The independent audit attributes 94.5--96.5% of fold 1's loss at 3,000--4,500
ms to `download-youtube`. Removing that task leaves only about 9.4--11.0
seconds of loss per point. The audit also reconstructed all four inner folds:
accepted `exec:export` probe decisions already contributed `-1.796` normalized
utility, but positive utility from other nodes caused the global sum objective
to accept the same guard.

This is a cluster-generalization failure. The inner ERM maximizes the sum of
per-call normalized deltas and can report a positive objective even when some
accepted tasks have large negative contributions. For example, Terminal fold
1's probe objective is `+16.339`, while its worst accepted probe task is
`-4.000`. A single unseen task can then repeat a sparse-node mistake many
times.

This mechanism is closely related to the original concern about many short
`read`/`edit` calls controlling a classifier. Here the direction is reversed:
a sparse apparently-long `exec` node controls repeated short calls. Both are
the same statistical error: treating correlated calls as independent support.

## Validity Checks

- Exactly five task-disjoint outer folds were pooled for every corpus:
  2,347 SWE-Rebench calls / 50 tasks; 2,959 Terminal-Bench calls / 83 tasks;
  1,730 ScienceAgentBench calls / 102 tasks.
- The strict aggregator reproduced every fold point from raw decisions and
  rejected missing, stale, duplicate-task, and incomplete-panel inputs in
  focused tests.
- All result hashes passed `sha256sum --check` after the run.
- An independent post-result audit verified 55 result hashes and 44 input
  hashes, and independently reproduced every fold and pooled metric from raw
  decisions with zero error.
- At all ten common cost points, all checked deadline, mean-hazard, and
  robust-clock fields exactly match the prior 100 ms sweep. This independently
  confirms that adding the new arm did not change the baselines.
- The 27 focused offline-probe tests, Ruff format/lint, shell syntax,
  `py_compile`, and `git diff --check` passed before the real run.

## Decision and Next Arm

Do not adopt the point-estimate offline-probe guard and do not repair it with
a corpus-name condition or a manually selected Terminal threshold.

The smallest defensible follow-up is a **task-cluster lower-confidence gate**:

1. Keep the call-total utility target, because repeated real calls incur real
   systems cost.
2. Estimate uncertainty by resampling complete tasks, not calls.
3. Use the existing LOTO-plus-parent robust trigger as the candidate, because
   it already rejected the failing `exec:export` node, and let offline
   calibration only filter those robust early actions.
4. If a higher-power method is later needed, permit an early deviation only
   when a task-grouped lower confidence bound for its utility over the
   deadline anchor is positive. That adds an explicit confidence contract and
   should be evaluated as a separate arm.
5. Use one fixed calibration/confidence contract across corpora.
   Corpus-specific behavior may emerge from sampled tasks, but no
   corpus-specific constant is allowed.

This follow-up is exploratory on these same exposed corpora. Any adoption
claim still requires a frozen implementation and a fresh bridge/deployment
sample.
