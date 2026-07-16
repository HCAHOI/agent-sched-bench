# Phase 0 — gate robustness on the frozen corpus (2026-07-16)

Hardens the certification gate on existing SWE-ReBench data BEFORE any fresh
collection, per the Fable-5 review's Phase 0. Three leakage-free re-analyses of
the committed certified-union decisions at the operating-point restore proxy
(rho=1.0, the committed proxy for the measured rho=0.94; the low-rho files are
the closed sensitivity probe and are not read here). Corpus: SWE-ReBench 100
tasks / 94 repos / 4640 calls / 10-cost family. Certificate under test:
`certified_union_trigger_ms` vs the deadline, via the deployed
`paired_task_cluster_bootstrap` (task-cluster Bonferroni-percentile simultaneous
interval; `simultaneous_label=="positive"` is the certificate).

Script: `scripts/analyze_gate_robustness.py` (reuses the deployed
`paired_task_cluster_bootstrap` + `_resample_task_totals`, not reimplementations;
anchor assertion reproduces the library's labels). Review-gated (fresh reviewer,
two blocking findings fixed and re-confirmed).

## Result: the certificate is ~2.6-2.7x anticonservative (the load-bearing finding)

Under a **paired sign-flip null** (flip each task's whole paired-delta vector by
+/-1 — the exact paired-randomization null for a certificate testing E[delta]>0,
preserving the heavy-tailed cluster magnitudes the real bootstrap faces) and
recomputing the deployed bootstrap+Bonferroni labels on each null draw
(2000 draws, 20000 inner replicates = deployed resolution):

| quantity | nominal | empirical | ratio |
|---|---|---|---|
| one-sided family false-**positive** cert rate | 0.025 | **0.0675** | **2.70x** |
| two-sided family any-cert rate | 0.050 | **0.1305** | **2.61x** |
| family false-**harmful** rate | 0.025 | 0.0650 | 2.60x |

The percentile-bootstrap Bonferroni certificate under-covers on this skewed
~100-cluster data: it certifies at ~2.7x its nominal rate when there is NO real
effect. Bonferroni is itself conservative here (the cost columns are highly
correlated), so the per-cost undercoverage is worse than the family ratio — the
2.7x understates the per-cell problem. The magnitude is stable across inner
replicates (6.85%/13.5% at 4000 vs 6.75%/13.05% at 20000), so it is a real
property of the estimator, not Monte-Carlo noise.

**Consequence for the campaign's one SWE-ReBench win:** the certified-union-vs-
deadline certificate has exactly ONE positive cost cell (kv=4500 ms; identical
under task and repo clustering). A single positive cell against a two-sided
family false-cert rate of ~13% is at the noise floor — it is plausibly a false
positive. The "certified union beats the deadline on SWE-ReBench" claim should
be treated as UNCERTIFIED until the certificate's coverage is fixed. This is
precisely why Phase 0 runs before spending fresh data: the certification RULE
needs a coverage fix (studentized/BCa bootstrap, or a permutation/calibrated
certificate) before any fresh-corpus headline.

## Repo clustering — quantified non-issue (no aggregation to worry about)

SWE-ReBench-100 is near repo-unique: **94 repos, 88 singletons, 6 repos with 2
tasks, max 2 tasks/repo**. Resampling repos instead of tasks leaves the
certificate labels IDENTICAL (only kv=4500 positive in both) and CI widths
within ~5% (e.g. kv=4500: task 78524 vs repo 81839 ms). Honest scope: because
94/100 clusters are singletons, repo resampling is arithmetically ~= task
resampling and the test has little power to detect within-repo correlation —
there is simply almost no within-repo replication to carry it. So repo
clustering is not a threat on THIS corpus, but a fresh corpus with denser
repos would need this re-checked.

## Censoring — real but immaterial (0.13% of calls)

Tool exec timeouts do cap the upper tail. The unambiguous censoring signature is
a **~300 s cap plateau: 3 calls clustered within 30 ms (300074.6 / 300100.3 /
300104.9 ms)** plus **1 at ~600 s (600085.2)** — a one-sided upward window
`[cap, cap*1.02]` catches them (censored calls overshoot the nominal cap by
teardown overhead). Two further lone calls near 60 s / 120 s also fall in the
window but sit near round numbers with no cluster, so they are equally likely
genuinely-long calls, not caps — counted conservatively, the total is **6
suspect calls (0.129%)**, of which ~4 are real caps. The plateau is visible in
`top_latencies_ms`. My first
audit used a too-tight symmetric tolerance and printed a false "no pileup"; the
reviewer caught it (fixed). Magnitude verdict: 6/4640 calls cannot move the
task-clustered paired totals, so censoring does not threaten the aggregate
certificate. Caveat: this branch's exec timeout is resource-integrated/stall-
based (see src/trace_collect/CLAUDE.md), which censors at a VARYING wall-clock
value and leaves no constant pileup — so constant-cap detection is a LOWER
bound on true censoring. Per-tool caps were not separated.

## Bottom line

- Repo clustering and censoring are NOT threats on this corpus (quantified,
  pre-empts two of the sharpest reviewer objections).
- The gate's certificate is materially anticonservative (~2.7x). The single
  SWE-ReBench "certified" win is at the false-cert noise floor and should not
  be quoted as certified until the bootstrap CI is replaced with a
  coverage-valid procedure. This is a prerequisite fix BEFORE the fresh-corpus
  certification run (else the fresh certificate inherits the same optimism).
- Cheap to run (~4 min CPU); no fresh data spent.
