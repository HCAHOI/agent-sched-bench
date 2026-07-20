# Utility-Clock Fine-Grained Threshold Sweep

Date: 2026-07-11

Status: frozen before result generation. Exploratory mechanism analysis on
three analyst-exposed corpora; not a confirmation experiment.

## Question

Do the non-monotone headroom and robust-capture curves observed on a 500 ms
grid persist when the same full threshold range is sampled every 100 ms?

## Fixed Data and Policies

- Reuse the same five frozen task-disjoint profile/eval folds for SWE-Rebench
  qwen3.7-max, Terminal-Bench GLM-5.2, and ScienceAgentBench Verified
  qwen3.7-max.
- Reuse the exact prior, grouping, eligibility, utility accounting, and three
  policy arms from
  `.omc/artifacts/tool-time-utility-clock-probe-20260711/protocol.md`.
- Reuse the reviewed exact headroom, gain/exposure decomposition,
  task-cluster bootstrap, case-study grouping, and plotting implementation
  from `.omc/artifacts/tool-time-threshold-sweep-20260711/`.
- Threshold equals action proxy cost; guard remains 0 ms.
- No trace recollection, fold changes, task filtering, policy changes, or
  per-corpus tuning.

## Fixed Fine Grid

- Inclusive range: 500 through 5,000 ms.
- Uniform step: 100 ms.
- Total points: 46 per corpus, 138 corpus/threshold points overall.
- Predeclared case-study points remain 2,000 and 5,000 ms.

The full range is uniformly refined rather than adding points only around the
previously observed valleys. Every point is retained. Figures show raw point
estimates connected by lines; no interpolation-based inference, smoothing,
threshold selection, or local-extrema significance claim is allowed.

## Endpoints

At every point retain in the structured result:

- exact deadline headroom divided by oracle utility (`rho`);
- robust and mean-hazard net delta and captured headroom;
- robust and mean-hazard boundary-band gain and early-short penalty;
- task-cluster 95% percentile intervals from 10,000 replicates, seed 0.

The reused figures plot `rho`, robust captured headroom, and the robust
band-gain/short-penalty decomposition. Mean-hazard remains fully retained in
the JSON for comparison but is not added to the figures in this experiment.

At 2,000 and 5,000 ms retain the full tool/task/source/support/node case tables
defined in the preceding 500 ms sweep. These repeated cases validate exact
reproduction; they are not newly selected operating points.

## Interpretation Constraints

- Pointwise intervals are descriptive, non-simultaneous, and are not paired
  threshold-difference tests.
- Fine-grid local extrema are descriptive. A neighboring 100 ms point being
  higher or lower does not establish a stable optimal threshold.
- The exposed corpora may diagnose mechanisms but cannot confirm a method
  selected from these curves.

## Environment and Review Gate

The existing `figures` optional extra is required. The runner must preflight
Matplotlib before any fold generation and report
`uv sync --extra dev --extra figures` when unavailable.

No real fine-grid decisions or figures may be generated until an independent
reviewer approves this protocol, manifest, and runner and confirms that the
previously reviewed analysis sources are used unchanged.
