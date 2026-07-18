# Should tool-call prediction decompose chained commands into atoms?

One-page summary for advisor discussion. Full tables:
`segment-atom-study-2026-07-18.{md,json}`. EXPLORATORY study; commands
re-executed on our hardware (H100 host, 8-core replay box) via trace replay
of the fresh-277 corpus — structure analysis, not absolute-latency
benchmarking. Code review-gated (3 rounds); instrumentation = bash xtrace
wrap, documented in src/trace_collect/CLAUDE.md.

## The question

Chains like `cd X && make && pytest` are one tool call. Why not track each
atom's execution and predict atoms instead of chains? The earlier additive
segment-cost model was rejected, but its atom costs were DECONVOLVED from
chain totals. This study removes that objection: we instrumented the shell
(per-segment xtrace timing) and re-executed all 277 tasks' commands,
observing 8,953 real per-segment timelines directly.

## Headline result: atoms lose even with direct observation

Out-of-sample (task-grouped folds), predicting chain total duration,
complete corpus (4,824 analysable chains):

| model | MAE | tail (P90+) MAE |
|---|---|---|
| atom_identity (sum of observed atom medians) | 1022 ms | 9276 ms |
| atom_plus_args (+ token-count features) | 1022 ms | 9273 ms |
| chain_prefix_cert (the certified trie, exact frozen config) | 996 ms | 9060 ms |
| **chain_prefix_cdskip (trie + cd-normalization)** | **971 ms** | **8905 ms** |

Both chain models beat both atom models on every metric; argument features
add nothing to atoms. (Pooled R² ≈ 0 for all models — heavy-tail outliers
dominate SS_tot; MAE/tail-MAE ordering is the robust comparison.)

## Why atoms fail: the stability table (the mechanistic finding)

Cross-task coefficient of variation of each atom's duration:

- **Stable atoms are trivial**: `cd` (0.07 ms, CV 0.43), `case`, `source`,
  `[` — all sub-5 ms. Stability exists only where duration doesn't matter.
- **The verbs that carry all the time are wildly unstable**: `python3`
  (median 122 ms, CV 5.5), `grep` (CV 4.3), `pip` (CV 3.7), `pytest`
  (median 1s, CV 1.8), `git` (CV 2.3). An atom's identity tells you almost
  nothing — `python3` means 1 ms or 300 s depending on which script in
  which repo.
- **The one heavy stable atom is `apt-get`** (median 4992 ms, CV 0.38) —
  which is exactly why apt-get prefixes carried the certified H1 wins.
- Family decomposition: variance shares load ~100% on the final verb
  (`cd` contributes 0.00 everywhere). Chains are "cd + verb", and
  verb-times-context carries everything.

So the additive model's failure was never deconvolution error — it is that
**atom identity is the wrong conditioning unit**: duration is a property of
(verb, arguments, repo state), which is precisely what deeper command-prefix
conditioning approximates and what atom decomposition throws away.

## Secondary findings

1. **cd-normalization gains independent evidence**: chain_prefix_cdskip
   beats the exact certified config on MAE (−25 ms) and tail (−155 ms) —
   consistent with the H2 case study's fix candidate #2 (thin `cd X &&`
   prefixes fragment support). Candidate for the next method iteration,
   validated free on TraceLab replay.
2. **46% of exec calls are pipelines/loops** where per-segment durations
   are ill-defined by construction (members run concurrently). Even a
   perfect atom model could never cover half the workload — a structural
   argument for chain-level conditioning, independent of the accuracy one.
3. The per-atom dataset + instrumentation are reusable: atom boundaries as
   runtime observation events (re-condition survival at "atom k finished")
   remain the interesting future direction — as mid-call evidence, not as
   the fit-time unit.

## Verdict

The conditioning unit stays chain-prefix. The atoms question is now closed
with direct measurement rather than argument; the two live follow-ups it
produced are cd-normalization (method iteration, evidenced twice) and
atom-boundary events as runtime signals (P4-era, joins the CPU-rate lever).
