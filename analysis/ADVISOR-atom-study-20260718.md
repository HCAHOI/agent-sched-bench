# Should tool-call prediction decompose chained commands into atoms?

One-page summary for advisor discussion. Full tables:
`segment-atom-study-2026-07-19.{md,json}` (five models; supersedes the
07-18 four-model run, whose numbers reproduce to the digit). EXPLORATORY
study; commands re-executed on our hardware via trace replay of the
fresh-277 corpus — structure analysis, not absolute-latency benchmarking.
Code review-gated (4 rounds); instrumentation = bash xtrace wrap,
documented in src/trace_collect/CLAUDE.md. UPDATED 2026-07-19: added the
atom_trie model (per-atom argument-conditioned tries — the strong form of
the atoms proposal), which materially revises the headline.

## The question

Chains like `cd X && make && pytest` are one tool call. Why not track each
atom's execution and predict atoms instead of chains? The earlier additive
segment-cost model was rejected, but its atom costs were DECONVOLVED from
chain totals. This study removes that objection: we instrumented the shell
(per-segment xtrace timing) and re-executed all 277 tasks' commands,
observing 8,953 real per-segment timelines directly.

## Headline result: naive atoms lose; argument-conditioned atoms are
## competitive; chain conditioning still (narrowly) wins

Out-of-sample (task-grouped folds), predicting chain total duration,
complete corpus (4,824 analysable chains):

| model | MAE | tail (P90+) MAE |
|---|---|---|
| atom_identity (sum of observed atom medians) | 1022 ms | 9276 ms |
| atom_plus_args (+ token-count features) | 1022 ms | 9273 ms |
| atom_trie (per-atom argument tries, summed) | 974 ms | 8914 ms |
| chain_prefix_cert (the certified trie, exact frozen config) | 996 ms | 9060 ms |
| **chain_prefix_cdskip (trie + cd-normalization)** | **971 ms** | **8905 ms** |

Verb-identity atoms lose on every metric and argument features add nothing
to them — but per-atom TRIES (each atom gets its own token-prefix
conditioning and depth budget) close most of the gap: atom_trie beats the
exact certified config and trails the cd-normalized trie by ~3 ms MAE.
The frontier is now cd-skip vs atom-trie, and both wins share one cause:
giving the depth budget to the load-bearing verb instead of the cd wrapper.
(Pooled R² ~ 0 for all models — heavy-tail outliers dominate SS_tot;
MAE/tail ordering is the robust comparison.)

**Why atom_trie still can't overtake (the depth diagnostic):** the heavy,
variable verbs almost never earn argument-conditioned nodes — pytest is
served at bare verb level 88% of the time, find 98%, timeout 86% — while
python3/git condition deep (source of atom_trie's edge over flat atoms).
Argument thinness binds exactly where the tail lives. Measured, not argued:
mean matched depth 2.005 over 8,893 out-of-sample atom predictions.

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

## Verdict (revised 2026-07-19; kill tests run same day)

The shipped conditioning unit stays chain-prefix (cd-skip variant now
evidenced twice), but the question is sharper than "atoms lose": naive
atoms lose; argument-conditioned atom tries are competitive and beaten
only by argument thinness on the heavy verbs. The two pre-registered
follow-ups were run on the full corpus and **both KILLED**
(`boundary-evidence-stage1-2026-07-19.md`,
`stable-atom-overlap-2026-07-19.md`):
1. cd-normalization — method iteration, cheapest, evidenced twice.
   **Still live; now the only surviving atom-study follow-up.**
2. Atom boundaries as runtime EVIDENCE — **KILLED at Stage 1**. Boundary
   identity flips 72.9% of re-check decisions vs elapsed-only, but the
   paired log-score gain is −0.0137 nats with a task-clustered 95% CI of
   [−0.41, +0.20] covering zero: the decision churn is noise, not
   information. The worst possible deployment profile — large behavioral
   divergence with no predictive backing — which is exactly what the kill
   switch existed to catch. §Secondary-3's "interesting future direction"
   gets a measured **no**.
3. Stable-atom screening — **KILLED**. The screen is a one-atom trick:
   apt-get is the only fold-stable qualifier (pytest 1/5 folds, Jaccard
   0.8), and the pooled node fires on 29/4,824 held-out calls with only
   9 (0.19% of corpus) landing where the trie is thin — no union mass.
