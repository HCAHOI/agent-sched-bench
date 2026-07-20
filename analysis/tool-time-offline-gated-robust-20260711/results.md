# Offline-Gated Robust Utility Clock Results

Date: 2026-07-11

Status: descriptive exploratory result on three analyst-exposed corpora. No
binary success or adoption claim is made.

## Executive Result

The mechanism behaved as specified: every new-arm trigger is either the
baseline robust trigger or deadline, and each outer fold learned its guard
only from nested task-OOF sampled calls. The empirical tradeoff is mixed.

Relative to `robust_clock`, summing normalized paired deltas across all ten
alternative cost points gives:

| Corpus | Gated delta vs deadline, normalized sum | Paired gated - robust, normalized sum | Paired raw sum (ms) |
|---|---:|---:|---:|
| SWE-Rebench | 286.162 | -6.665 | -19,153 |
| Terminal-Bench | 2.244 | +37.993 | +169,838 |
| ScienceAgentBench | 99.146 | +7.763 | +17,128 |

The combined paired sums are `+39.092` normalized and `+167,813 ms`, but this
is not a success criterion or a deployable combined reward: costs are
alternative operating points, corpora are exposed development data, and no
task-bootstrap confidence interval was preregistered.

## Learned Robust Guards

The dimensionless guard applies to the robust candidate's weakest normalized
advantage over every later action across full, LOTO, and parent curves.

| Corpus | f1 | f2 | f3 | f4 | f5 |
|---|---:|---:|---:|---:|---:|
| SWE-Rebench | 0.00000584 | 0.00000707 | 0.00008059 | 0.00007821 | 0.00000808 |
| Terminal-Bench | 0.00344500 | 0.00224089 | 0.01061569 | 0.00012080 | 0.00037143 |
| ScienceAgentBench | 0.00021169 | 0.00004874 | 0.00061081 | 0.00090526 | 0.00024630 |

These values were learned by one common algorithm. Neither policy nor
calibration receives corpus identity.

All 15 real folds selected non-null guards. The synthetic `never early` path
is covered by tests but was not selected on this panel.

## Paired Delta by Cost

Each cell is `offline_gated_robust_clock - robust_clock` in held-out outer-OOF
milliseconds.

| Cost (ms) | SWE-Rebench | Terminal-Bench | ScienceAgentBench |
|---:|---:|---:|---:|
| 500 | -306 | +423 | -96 |
| 1,000 | 0 | -9,453 | -25 |
| 1,500 | 0 | -2,559 | -1,750 |
| 2,000 | 0 | -1,212 | +15,329 |
| 2,500 | 0 | +4,908 | +1,732 |
| 3,000 | -6,000 | -2,553 | +4,746 |
| 3,500 | -17,300 | +82,868 | -7,500 |
| 4,000 | 0 | +92,440 | +11,076 |
| 4,500 | 0 | +848 | -6,384 |
| 5,000 | +4,454 | +4,129 | 0 |

Relative to deadline, the gated arm is positive at 9/10 SWE points, 5/10
Terminal points, and 8/10 ScienceAgentBench points. Those signs are descriptive
and do not override the paired comparison to robust clock.

## Fold Heterogeneity

Paired normalized `gated - robust` sums by outer fold:

| Corpus | f1 | f2 | f3 | f4 | f5 |
|---|---:|---:|---:|---:|---:|
| SWE-Rebench | -1.961 | 0 | -5.556 | +0.891 | -0.039 |
| Terminal-Bench | +3.827 | +2.935 | +3.703 | -1.665 | +29.193 |
| ScienceAgentBench | -2.257 | 0 | +5.124 | -2.222 | +7.118 |

The pooled effects are not uniform task-level transfer. In particular,
Terminal's aggregate improvement is dominated by fold 5, while SWE's loss is
dominated by fold 3.

## What Was Filtered

Across the ten-point sweep, robust-to-deadline filtering changes many assigned
candidates but fewer actions actually survive long enough to fire:

| Corpus | Filtered candidate assignments | Filtered actual robust fires | Robust early-short fires | Gated early-short fires |
|---|---:|---:|---:|---:|
| SWE-Rebench | 417 | 106 | 204 | 203 |
| Terminal-Bench | 9,793 | 941 | 239 | 125 |
| ScienceAgentBench | 1,942 | 342 | 266 | 239 |

This distinction matters: a deadline fallback on a call that ends before the
robust candidate changes no realized utility.

## Terminal Mechanisms

### Previously Exposed Sparse Node

For the known Terminal fold 1 `download-youtube` case, all 79
`exec:export` rows at each of 3,000--4,500 ms receive `k_r = T` from the
general robust contract. The gated arm therefore also uses deadline and has
zero early fires on these rows. No task, command, corpus, or cost special case
was added.

This verifies the mechanical correction but is not new generalization
evidence: the arm was designed after that failure was exposed, and baseline
robust clock already vetoed the node.

The rejected point arm remains unchanged and retains its
`-171,345 / -212,634 / -256,153 / -292,153 ms` fold-1 losses at these four
costs. Across all fold-1 calls, gated delta versus deadline is exactly zero at
3,000--4,500 ms; baseline robust delta in milliseconds is
`-303 / -10,952 / -7,752 / +2,409`. The gate therefore also filters unrelated
robust actions, helping at 3,000--4,000 ms and removing useful utility at
4,500 ms.

### Additional Held-Out Effect

The largest new paired improvement occurs in Terminal fold 5. At 4,000 ms,
`path-tracing` contributes `+93,396 ms` of the total `+92,440 ms` paired gain.
Its dominant `exec:python3 -c` robust node has 15 profile tasks, trigger
1,312 ms, and robust margin about `9.42e-5`, below fold 5's sampled guard
`3.71e-4`. The gate filters 26 assigned calls; 25 survive the robust trigger
and all 25 finish before deadline. This is a distinct task-cluster failure
that was not named in the method or protocol.

At 3,500 ms, the same corpus gains `+82,868 ms` relative to robust clock. These
large improvements coexist with losses at 1,000, 1,500, 2,000, and 3,000 ms;
the method is not pointwise dominant.

## Cost of Conservatism

SWE-Rebench shows the clearest cost. At 3,500 ms the gate loses `17,300 ms`
relative to robust clock while avoiding zero actual early-short fires. The
largest task losses are `hylang__hy-2250` (`-7,957 ms`),
`twisted__twisted-12366` (`-4,818 ms`), and
`codingedward__flask-sieve-30` (`-2,828 ms`). The filtered robust calls were
useful boundary-band actions, not false early actions.

ScienceAgentBench is also heterogeneous: the gate improves 2,000 and 4,000 ms
by `15,329` and `11,076 ms`, but loses 3,500 and 4,500 ms by `7,500` and
`6,385 ms`.

## Reproducibility and Integrity Checks

- Exact five-fold pooling covers 2,347 SWE-Rebench calls / 50 tasks; 2,959
  Terminal-Bench calls / 83 tasks; and 1,730 ScienceAgentBench calls / 102
  tasks.
- The strict aggregator independently derives expected gated triggers from
  each raw guard, robust score, and robust candidate, and rejects any third
  trigger value or inconsistent application.
- All deadline, mean-hazard, robust-clock, rejected point-guard metrics, and
  point calibration records exactly match the preceding approved experiment
  at every corpus/cost/fold.
- The initial 57-file result inventory passed `sha256sum --check` before this
  report was added.
- Before the run, 66 related tests, Ruff, Bash syntax, Python byte compilation,
  and `git diff --check` passed. The independent reviewer approved the run
  after separate synthetic equivalence and behavior checks.

## Interpretation

The result supports a narrow mechanism statement: nested offline statistics
can learn workload-specific filters over task-robust actions, and those filters
can substantially reduce repeated early-short exposure. It also demonstrates
the price: a global guard sometimes removes useful band actions, and the
direction varies across folds and costs.

Every fold's robust calibration still has a negative worst accepted OOF task.
Moreover, the largest observed improvement is concentrated enough that one
Terminal task exceeds the pooled 4,000 ms paired gain before other task losses
offset it. Call-total positive calibration objectives therefore do not imply
per-task safety.

The exposed panel cannot establish safety, statistical superiority, or
cross-deployment generalization. A confirmation experiment must freeze this
implementation unchanged and run on a fresh deployment/bridge sample, with a
predeclared paired task-level uncertainty analysis.
