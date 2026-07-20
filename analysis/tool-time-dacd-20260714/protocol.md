# DACD development analysis protocol

## Purpose

Test Deadline-Anchored Certified Deviation (DACD) on the existing public trace
corpora. This is a development analysis, not a claim on an untouched benchmark.
No new traces are required and no corpus identity is available to the policy.

## Estimand and policy

The primary estimand is held-out task utility delta of DACD versus
`deadline_only` at every declared KV cost. DACD fits robust-clock triggers on a
task-disjoint subset of each outer profile split, freezes the exact
`(context, cost, threshold, trigger)` mapping, and certifies each early rule on
the other half of the outer profile tasks. Unseen, uncertified, or ambiguous
rules use the deadline.

Certification uses a one-sided task-cluster percentile bootstrap lower bound
with Bonferroni correction over the complete frozen early-rule family observed
in certification contexts. Repeated calls are summed within task before
resampling.

## Frozen configuration

- Outer folds: existing 5 task-disjoint folds per corpus
- Inner folds: 2; first balanced fold certification-only, second fit-only
- KV costs: 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000 ms
- Guard: 0 ms
- Confidence: 0.95
- Bootstrap: 50,000 task-cluster replicates, PCG64 seed 0
- Minimum tool history: 1
- Minimum profile tasks: 1
- Command field: `command`
- Maximum prefix depth: 4
- Leading `cd`: retained

All ten cost points and all four corpora will be reported. There is no
threshold selection, corpus-specific configuration, or exclusion based on the
observed result.

## Existing inputs

- SWE-ReBench development: `.omc/artifacts/tool-time-biclassifier-probe-20260710/data/swe`
- Terminal-Bench: `.omc/artifacts/tool-time-biclassifier-probe-20260710/data/terminal`
- ScientificAgentBench Verified: `.omc/artifacts/tool-time-biclassifier-sab-20260711/data/sab`
- SWE-ReBench 100-task collection: `analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100-20260713/results/data`

Each fold summary records and verifies absolute input hashes and the canonical
implementation source bundle. Aggregation reloads the hashed inputs, recomputes
the inner split and frozen models, and rejects any decision or certificate that
cannot be reproduced exactly.
