# Phase 0 — gate robustness + certificate coverage fix (2026-07-16)

Hardens the certification gate on existing SWE-ReBench data BEFORE any fresh
collection (Fable-5 review's Phase 0), then FIXES the coverage hole it exposed.
Three leakage-free re-analyses of the committed certified-union decisions at the
operating-point restore proxy (rho=1.0, the committed proxy for the measured
rho=0.94; the low-rho files are the closed sensitivity probe and are not read
here). Corpus: SWE-ReBench 100 tasks / 94 repos / 4640 calls / 10-cost family.
Certificate under test: `certified_union_trigger_ms` vs the deadline.

Script: `scripts/certification/analyze_gate_robustness.py`. Library change:
`paired_task_cluster_bootstrap` gains an opt-in `permutation_draws` that attaches
a coverage-valid certificate (default off = frozen numerics byte-identical).
Review-gated.

## Finding 1: the deployed percentile certificate is ~2.7x anticonservative

Under a **paired sign-flip null** — flip each task's whole paired-delta vector by
+/-1, the paired-randomization H0 (exact under sign-symmetry of the paired
deltas, a sharper null than E[delta]=0), preserving the heavy-tailed
cluster magnitudes and cross-cost correlation the real bootstrap faces — and
recomputing the deployed Bonferroni-percentile labels on each null draw (2000
draws x 20000 inner replicates = deployed resolution):

| certificate | one-sided family pos. rate (nominal 0.025) | two-sided any-cert (nominal 0.05) |
|---|---|---|
| **percentile (deployed)** | **0.0675 (2.70x)** | **0.1305 (2.61x)** |
| **permutation (fix)** | 0.0155 (0.62x) | 0.0295 (0.59x) |

The percentile bootstrap under-covers on this skewed ~100-cluster data: it
certifies at ~2.7x its nominal rate under no effect (Bonferroni is itself
conservative here, so the per-cell undercoverage is worse than the family ratio).
Stable across inner replicates (6.85%/13.5% at 4000 vs 6.75%/13.05% at 20000),
so it is a real estimator property, not Monte-Carlo noise. A percentile
`simultaneous_label == "positive"` therefore must NOT carry a headline
certification.

## The fix: an exact paired sign-flip randomization certificate

`_permutation_simultaneous_labels` (in `tool_latency_confirmation.py`, exposed via
`paired_task_cluster_bootstrap(..., permutation_draws=N)`) replaces the percentile
interval with a one-sided randomization p-value per cost cell (standard +1
correction), Bonferroni-simultaneous over the cost family at tail alpha/(2m). It
is exact under exchangeability and distribution-free, so it stays calibrated
under the per-task tails that break the percentile bound. Measured on the SAME
sign-flip null it holds at 0.62x nominal one-sided / 0.59x two-sided (the
comparison is fair — both certificates are scored on the identical valid null;
the percentile cert fails on the very draws where the permutation cert holds).
The randomization test's validity is a theorem under exchangeability; this
empirical pass confirms the implementation. Its p-value floor is 1/2^n, so it is
(correctly) MORE conservative than the percentile bound at small task counts — it
refuses to certify a clean 5-task win because the smallest achievable p (1/32)
exceeds the 0.0025 family tail; at n=100 it has ample resolution.

BCa was considered and rejected: its acceleration term is driven by the same
extreme clusters that cause the under-coverage, so it is fragile exactly where it
is needed; the randomization test needs no such estimate.

## The single SWE-ReBench win SURVIVES the fixed certificate

Under the coverage-valid permutation certificate on the real corpus, exactly one
cost cell certifies: **kv=4500 ms, permutation p_positive = 0.0001** (well below
the 0.0025 family tail; delta +33834 ms). The next-smallest cell p-values are
0.0101 (5000), 0.0273 (1500), 0.0312 (3000) — all above 0.0025, correctly not
certified. So the certified-union-vs-deadline win on SWE-ReBench is NOT a false
positive of the broken instrument — it holds under the exact test. (This
supersedes an earlier draft of this doc that, reasoning only from the percentile
noise floor, called the cell "plausibly a fluke" — the coverage-valid test
settles it: the cell is real.)

Two issues were conflated and are now separated:
1. **Instrument validity** (bootstrap under-coverage) — FIXED by the permutation
   certificate.
2. **Selection optimism** (dev-exposed corpus; the certified-union rule was
   itself chosen after observing gate behavior on these corpora) — NOT a
   statistical-instrument problem; still requires the fresh-corpus certification.
   The permutation certificate is now the instrument that fresh run should use.

## Finding 2 (repo clustering) — quantified non-issue

SWE-ReBench-100 is near repo-unique: 94 repos, 88 singletons, 6 repos with 2
tasks, max 2 tasks/repo. Resampling repos instead of tasks leaves the certificate
labels identical (only kv=4500 positive) and CI widths within ~5%. Honest scope:
with 94/100 clusters singleton, repo resampling is ~= task resampling and the
test has little power to detect within-repo correlation — there is almost no
within-repo replication to carry it. Not a threat on THIS corpus; a denser-repo
fresh corpus would need this re-checked.

## Finding 3 (censoring) — real but immaterial

The unambiguous censoring signature is a ~300 s exec-timeout plateau (3 calls
within 30 ms: 300074.6 / 300100.3 / 300104.9 ms) plus 1 at ~600 s (600085.2), via
a one-sided upward window `[cap, cap*1.02]` (censored calls overshoot the nominal
cap by teardown overhead). Two further lone calls near 60 s / 120 s fall in the
window but have no cluster and are as likely genuinely-long calls, so ~4 of the 6
suspect calls are real caps. Total 6/4640 = 0.13% — too few to move the
task-clustered paired totals. Constant-cap detection is a LOWER bound (this
branch's exec timeout is resource-integrated/stall-based per
src/trace_collect/CLAUDE.md → varying wall-clock cap); per-tool caps not
separated.

## Bottom line

- The certification instrument had a real coverage hole (~2.7x anticonservative);
  it is now FIXED with an exact paired randomization certificate (0.62x nominal),
  which must be the certificate the fresh-corpus run uses.
- Under the fixed instrument the one SWE-ReBench certified-union win holds
  (kv=4500, p=0.0001) — it was not an artifact. Selection optimism (dev-exposed)
  is the remaining, orthogonal reason it is still "sensitivity, not certified"
  until the fresh corpus.
- Repo clustering and censoring are quantified non-threats on this corpus.
- Cheap (~8 min CPU); no fresh data spent.
