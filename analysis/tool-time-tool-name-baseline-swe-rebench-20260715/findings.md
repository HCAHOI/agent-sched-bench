# Tool-name baseline (P1) — findings (2026-07-15)

Reproduces Continuum's (arXiv:2511.02230) per-tool-name empirical-CDF
estimator class inside our harness (a trie fit with command_field=None:
tool -> global back-off, no command-prefix grouping — verified tool-level
on 100% of rows) and contrasts it against our command-prefix (full) trie
and the GBM, all gated and fit identically on the same folds. Positions us
against published SOTA and tests whether command-prefix conditioning is
worth its extra granularity.

**Read at the measured operating point rho=0.94 (the rho=1.0 column; the
run's low-rho columns are the sensitivity probe, not deployment claims —
see the rho-operating-point directive).** Both corpora dev-exposed
(sensitivity, not fresh certification).

## Result: command-prefix conditioning beats the tool-name estimator

At rho=0.94, totals vs the tool-name (Continuum-class) policy:

| corpus | full-trie vs tool-name | GBM vs tool-name | tool-name vs deadline |
|---|---|---|---|
| SWE-ReBench (100 tasks) | **+158.7 s (1 certified)** | +36.8 s (2 harmful cells) | -40.7 s (4 cert cells) |
| Terminal-Bench (83 tasks) | +46.4 s (uncertified) | +60.0 s (uncertified) | -71.9 s (uncertified) |

- **Command-prefix conditioning certifiably beats tool-name on
  SWE-ReBench** (+158.7 s, 1 certified cost cell) and beats it in point
  estimate on Terminal-Bench (+46.4 s; nothing certifies on 83 tasks).
  Consistent direction on both corpora at the operating point — finer
  causal conditioning is worth it. This matches Continuum's own Figure
  5(b) ("slowest 10% of cd = 94.1% of delay"): tool-name is the wrong unit
  because the compound command (cd + build) carries the signal.
- **The GBM is NOT cleanly better than tool-name** (SWE-ReBench: +36.8 s
  but 2 certified-harmful cells; TB: +60 s uncertified). So at the
  operating point the demonstrated value is the command-PREFIX trie, not
  learning per se — the learned model does not dominate the simple
  prefix estimator here.
- Tool-name itself is net-negative vs the deadline at rho=0.94 on both
  corpora, though it certifies a few individual cost cells on SWE-ReBench.

## Interpretation

Positioned against published SOTA at the honest operating point, our
command-prefix conditioning is the certifiable improvement over Continuum's
tool-name estimator; the learned GBM is not a further clean win. This is a
fair, direct result (their own Fig 5 predicts and motivates it) and keeps
the campaign's spectrum honest: memoryless -> tool-name (Continuum) ->
command-prefix (ours, the certifiable step) -> learned (no clean further
gain at this scale/corpus). Which level a deployment should use is still
arbitrated by the certification gate.

## Caveats

Operating point rho=0.94 only; low-rho columns in the payload are the
sensitivity probe. Dev-exposed corpora (sensitivity). TB underpowered
(nothing certifies). GBM triggers recomputed (seed 0) not joined from the
frozen certification; full-trie reproduces frozen triggers by construction.
Fresh-corpus certification (at rho=0.94 by construction) converts the
SWE-ReBench command-prefix-beats-tool-name result from sensitivity to
certified.
