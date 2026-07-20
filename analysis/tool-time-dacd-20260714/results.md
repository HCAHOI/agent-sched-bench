# DACD development analysis results

## Direct verdict

DACD is a usable conservative policy, but not a general replacement for the
deadline anchor. Across 335 held-out tasks, 11,676 tool calls, four corpora, and
ten costs per corpus, it certified 45 of 1,470 fold-local candidate rules. Under
a single global family of 40 corpus-cost comparisons, DACD was significantly
better than deadline at 5 points, significantly harmful at 0 points, and
inconclusive or exactly equal at 35 points.

This validates the anchor-and-deviate structure: it can recover some boundary
band utility without reverting to short/long classification or always firing
early. The effect is sparse, however, and most captured headroom is small.

## Coverage

`Early` and `early-short` count policy-cost decisions, not unique calls; each
call is evaluated at all ten costs.

| Corpus | Tasks | Calls | Certified / candidate | Early | Early-short | Global-40 positive | Harmful |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWE-ReBench dev50 | 50 | 2,347 | 15 / 329 | 1,095 | 7 | 2 | 0 |
| Terminal-Bench | 83 | 2,959 | 0 / 234 | 0 | 0 | 0 | 0 |
| ScientificAgentBench Verified | 102 | 1,730 | 9 / 440 | 196 | 1 | 1 | 0 |
| SWE-ReBench 100-task | 100 | 4,640 | 21 / 467 | 458 | 5 | 2 | 0 |
| **Total** | **335** | **11,676** | **45 / 1,470** | **1,749** | **13** | **5 / 40** | **0 / 40** |

Terminal-Bench is an important negative result: every candidate interval crossed
zero, so DACD became exactly deadline at all ten costs. It did not inherit the
large negative point estimates of the ungated robust clock.

## Primary result

Cells are held-out cumulative utility delta `DACD - deadline` in seconds with a
95% simultaneous interval for the complete 40-point family. `*` means the lower
bound is strictly positive. Zero cells indicate that no rule was deployed.

| Cost | SWE dev50 | Terminal | SAB | SWE 100-task |
|---:|---:|---:|---:|---:|
| 500 | **+53.854 [40.211, 68.931]*** | 0 | +0.208 [-1.670, 1.838] | 0 |
| 1,000 | +3.333 [-11.419, 12.931] | 0 | 0 | -1.458 [-9.606, 1.866] |
| 1,500 | +0.175 [0, 0.698] | 0 | 0 | **+2.261 [0.675, 4.594]*** |
| 2,000 | 0 | 0 | 0 | **+0.909 [0.247, 1.999]*** |
| 2,500 | 0 | 0 | **+0.203 [0.048, 0.407]*** | +0.201 [0, 0.511] |
| 3,000 | 0 | 0 | +1.494 [0, 4.244] | -2.972 [-20.818, 6.133] |
| 3,500 | +0.782 [0, 2.717] | 0 | +0.011 [0, 0.043] | +0.702 [0, 2.983] |
| 4,000 | **+6.864 [1.855, 14.081]*** | 0 | 0 | +0.249 [0, 0.996] |
| 4,500 | +9.163 [0, 25.526] | 0 | +0.087 [0, 0.227] | 0 |
| 5,000 | +36.893 [-7.425, 98.913] | 0 | +1.982 [0, 8.920] | +7.179 [0, 19.204] |

## Absolute DACD utility

The following values are DACD's own cumulative `net_saved` utility in seconds,
after subtracting exposed-action penalties. They are not deltas from another
policy. Each cost is a separate system configuration and must not be summed
across columns.

| Cost | SWE dev50 | Terminal | SAB | SWE 100-task |
|---:|---:|---:|---:|---:|
| 500 | 342.964 s | 211.299 s | 220.291 s | 278.373 s |
| 1,000 | 547.159 s | 285.758 s | 298.468 s | 346.894 s |
| 1,500 | 805.178 s | 172.835 s | 290.847 s | 373.133 s |
| 2,000 | 1,034.460 s | 266.309 s | 418.222 s | 400.491 s |
| 2,500 | 1,213.020 s | 335.514 s | 485.617 s | 449.470 s |
| 3,000 | 1,369.234 s | 327.290 s | 503.313 s | 503.715 s |
| 3,500 | 1,507.078 s | 294.086 s | 526.693 s | 543.541 s |
| 4,000 | 1,656.017 s | 255.997 s | 548.909 s | 561.980 s |
| 4,500 | 1,766.727 s | 215.832 s | 582.455 s | 605.963 s |
| 5,000 | 1,867.202 s | 205.189 s | 629.934 s | 578.596 s |

Absolute DACD early firings, with early-short firings after the slash:

| Cost | SWE dev50 | Terminal | SAB | SWE 100-task | Total |
|---:|---:|---:|---:|---:|---:|
| 500 | 492 / 0 | 0 / 0 | 85 / 1 | 0 / 0 | 577 / 1 |
| 1,000 | 172 / 5 | 0 / 0 | 0 / 0 | 149 / 3 | 321 / 8 |
| 1,500 | 47 / 0 | 0 / 0 | 0 / 0 | 113 / 0 | 160 / 0 |
| 2,000 | 0 / 0 | 0 / 0 | 0 / 0 | 96 / 0 | 96 / 0 |
| 2,500 | 0 / 0 | 0 / 0 | 40 / 0 | 30 / 0 | 70 / 0 |
| 3,000 | 0 / 0 | 0 / 0 | 27 / 0 | 35 / 2 | 62 / 2 |
| 3,500 | 70 / 0 | 0 / 0 | 10 / 0 | 13 / 0 | 93 / 0 |
| 4,000 | 154 / 0 | 0 / 0 | 0 / 0 | 13 / 0 | 167 / 0 |
| 4,500 | 65 / 0 | 0 / 0 | 19 / 0 | 0 / 0 | 84 / 0 |
| 5,000 | 95 / 2 | 0 / 0 | 15 / 0 | 9 / 0 | 119 / 2 |

The five globally positive points capture the following fractions of the exact
deadline headroom: SWE dev50 at 500 ms, 35.5%; SWE dev50 at 4,000 ms, 1.72%; SAB
at 2,500 ms, 0.12%; SWE 100-task at 1,500 ms, 0.74%; and SWE 100-task at 2,000
ms, 0.26%. Only the 500 ms SWE dev point is a large practical recovery.

## Failure case

The two negative point estimates on SWE 100-task are not statistically harmful,
but they expose the remaining risk:

| Cost | Band gain | Short penalty | Net delta | Early-short |
|---:|---:|---:|---:|---:|
| 1,000 ms | +1.530 s | -2.989 s | -1.458 s | 3 |
| 3,000 ms | +2.936 s | -5.908 s | -2.972 s | 2 |

All five early-short events came from the broad command-prefix context
`exec:cd /testbed && python3`. The frozen triggers were 985.46 ms and 2,904.13
ms, close enough to deadline that a call ending just after the trigger incurred
almost the full action cost. The certification lower bounds were positive but
small (0.00363 and 0.00320 normalized utility per task). Thus a mean-positive
certificate can still admit rare, high-penalty boundary errors under shift.

This is not the old short-majority failure: only 13 of 1,749 early firings were
on short calls. It is a tail-risk problem where a few short calls dominate
utility. A future risk-aware variant should constrain task-level downside or use
prepare/commit so a late aborted preparation does not pay the full swap cost.

## Comparison with robust clock

DACD does not dominate robust clock. It deliberately gives up some positive
robust actions: within-corpus simultaneous comparisons find DACD significantly
worse than robust at SWE dev50 500 and 3,000 ms and SWE 100-task 2,000 ms; most
other comparisons are inconclusive. Conversely, Terminal shows why the anchor
is useful: DACD rejects all rules where robust point estimates change sign
across costs.

The scientifically defensible claim is therefore asymmetric: DACD is a
learning-augmented deadline policy with evidence-gated deviations, not a new
oracle predictor and not a pointwise dominant policy.

## Method limitations

- Honest fit/certification splitting uses only half of each outer profile for
  fitting and half for certification. This is statistically clean but data
  inefficient.
- Certificates are distributional expected-utility statements, not per-call
  guarantees; the outer failure case demonstrates residual shift.
- Context is currently tool/command-prefix only. Sequence and resource state
  can enter the same rule key causally, but are not present in these datasets.
- The current utility metric values hidden swap latency, not HBM byte-seconds,
  PCIe contention, or prepare-buffer occupancy. Most available headroom remains
  structurally invisible or hard to capture.
- All four corpora are now development data. Reusing them is valid for this
  analysis, but a later confirmatory claim should freeze the method first.

## Artifacts

Each `<corpus>_cv/` directory contains five fold summaries, outer decisions,
complete certification probe decisions, pooled results, within-corpus paired
uncertainty, and global-40 adjusted uncertainty. Fold summaries contain verified
input and canonical implementation hashes; aggregation recomputed every split,
fit, decision, and certificate from those inputs.

- Protocol: `analysis/tool-time-dacd-20260714/protocol.md`
- Review audit: `analysis/tool-time-dacd-20260714/review.md`
- Primary pooled results: `analysis/tool-time-dacd-20260714/<corpus>_cv/pooled_results.json`
- Primary global uncertainty: `analysis/tool-time-dacd-20260714/<corpus>_cv/deadline_minus_dacd_global40_uncertainty.json`
- DACD versus robust: `analysis/tool-time-dacd-20260714/<corpus>_cv/dacd_minus_robust_uncertainty.json`
