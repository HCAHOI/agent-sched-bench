# Tool-Resource Prediction — Findings (2026-07-23)

**Status: development-only.** All numbers below are read on fresh-277
(development-exposed). The terminal-bench corpus remains untouched and is
reserved for one confirmatory read of the final champion set.

## Targets and data

Per tool call, three targets: latency (ms), peak CPU (cores, resource_timeline
`cpu_core_s/dt_s` max, eligible when ≥2 samples ∧ ≥1 s), peak container memory
(MB, sampled `resources.json` max in the call window, eligible when ≥1
in-window sample; explicitly whole-container, not tool-exclusive). Fit =
SWE-100 source corpus (`offline-gated-confirm-100-v2`); eval = fresh-277
source corpus (`fresh-seed42-skip150-n200`). 26.4% of eval tasks share a
repository with fit tasks — all confirmatory-style CIs are repo-clustered;
skill numbers are partially in-distribution, not pure OOD.

Per-binary cost table: BSD process accounting during a 20× replay of SWE-100
(`analysis/results/tool-resource-20260723/per_binary_cost_table{,.flat}.json`,
163 binaries, 99.8% coverage, 0 attribution violations). Protocol: pacct-on
replays produce per-binary labels only; call-level CPU labels always come from
pacct-off runs (acct shares the cgroup counter; ~3% inflation).

## Registered comparisons (all review-gated before reading)

| # | Comparison (p90 pinball unless noted) | Result | Verdict |
|---|---|---|---|
| 1 | mem: command-prefix ECDF vs per-tool ECDF | skill −0.166, CI [−0.299, −0.045] | FAIL |
| 2 | mem: ambient-anchored per-tool residual ECDF vs per-tool ECDF | skill +0.326, CI [+0.247, +0.412] | **PASS** |
| 3 | latency: feature-MLP vs prefix ECDF | skill −0.054, CI [−0.158, +0.026] | FAIL |
| 4 | cpu-heavy (>2c) binary: feature-MLP BA, deployment-legal fit-prevalence cut | BA 0.646, CI [0.605, 0.685] vs gate 0.70 | FAIL (gate) |
| 5 | cpu-heavy: latest-observation prequential blend vs static MLP (BA diff) | +0.038, paired CI [+0.015, +0.062] | **PASS** |
| 6 | cpu-heavy: two-layer lattice vs latest-observation blend (BA diff) | −0.008, paired CI [−0.028, +0.011] | FAIL |

BERT line (B0 frozen / B1 fold-0 finetune): text embeddings add no robust
marginal over numeric features for memory/peak-cores; small consistent gain
for core-seconds only (dropped as a target — r=0.963 log-log with duration).

## Champion table (deployment recommendation per cell)

| Cell | Champion | Number (dev) |
|---|---|---|
| latency q90 (static) | command-prefix ECDF (certified tool_time prior) | 844.5 ms mean p90 pinball, coverage 0.893 |
| latency q90 (online) | two-layer lattice (repo⊕public × prefix⊕tool) | **724.0 ms**, skill +0.143 CI [+0.020, +0.262] vs prefix ECDF |
| latency long/short (3500/5000 ms) | feature-MLP | BA 0.720 / 0.737 (AUC 0.94+) |
| mem q90 (static) | ambient-anchored per-tool residual ECDF | 43.3 MB |
| mem q90 (online) | two-layer lattice | **40.2 MB**, skill +0.464 CI [+0.330, +0.559] vs prefix ECDF |
| mem heavy/light (500 MB) | anchored-ECDF survival | BA 0.867, AUC 0.922 |
| cpu-peak q90 | per-tool ECDF 0.457 cores (static); two-layer 0.463 does not beat it | coverage 0.704 — below nominal; weakest cell |
| cpu heavy/light (2 cores) | latest-observation prequential blend | BA 0.684, CI [0.650, 0.716]; cold 0.571 / repo-history 0.730 / task-repeat 0.776 |

Note: "online" rows are prequential reads (repo-layer accumulates over the
eval stream causally: candidate ts_end < row ts_start). They compare against
static champions on identical rows/metrics; the public-only arm reproduces the
static champion numbers exactly (identity checked in review).

## Mechanism summary

- Duration/work is command-shaped; container memory is task/workspace-shaped
  (between-task variance share 0.44; (repo, command-head) purity 0.846 vs
  0.653 for command alone); cpu-peak sits between and is state-dominated.
- Nonparametric per-node ECDFs are unbeaten at absolute-unit q90 among static
  models; learned features win only in relative-error (q-error p90: MLP 4.8 vs
  ECDF 13.5 on latency) and classification framings.
- The workspace (repo) layer is where remaining signal lives: two-layer
  lattice lifts q90 for latency (+14%) and memory (+7% over anchor, +46% over
  prefix), but does NOT improve the 2-core binary over simply reusing the
  latest same-repo observation (comparison #6).
- cpu-heavy@2c has an information ceiling ≈ BA 0.68 for cold calls (oracle cut
  0.620 < deployed 0.646; best sweep 0.677): near-threshold mass (17% within
  ±0.5 cores) plus state-dependence not visible pre-call. Deployment-goal
  (0.70–0.80) is met in steady state (repo history present), not cold.

## Provenance

Registered result JSONs + row dump (sha256 `71cf0efb…87c0`) archived in this
directory. Producing commits: 61ca090 (pacct), d9c2505 (tool_resource),
99d46c0 (prequential/two-layer). Cost table provenance in its wrapped JSON.
Amendment log: cost-table path switched to `.flat.json` before comparison #3's
read (loader wrapper mismatch, criterion unchanged); comparison #4's decision
cut corrected from eval- to fit-prevalence before its read (review finding).
