# Fresh-corpus certification findings

The confirmatory run was finalized on 2026-07-17 under the locked
[`../certification/fresh-corpus-preregistration-20260716.md`](../certification/fresh-corpus-preregistration-20260716.md)
protocol. It used 277 previously unseen SWE-ReBench tasks (`seed42`,
`skip150`), disjoint from development roots, with 13,410 tool calls. The
operating point was `rho=0.94`; the paired task-clustered sign-flip certificate
used a Bonferroni family tail of `0.0025` and 20,000 draws.

The run passed its independent review gate with no blocking findings. Frozen
verdict artifacts are `gate-robustness/gate_robustness_rho094.json` and
`frontier-p1/permutation_p1_rho094.json`; the mechanism diagnosis is
[`case-study-h1-h2-20260717.md`](case-study-h1-h2-20260717.md).

## H1 — Certified-union trigger versus fixed deadline

**Certified.** Positive delta favors the certified-union trigger.

| KV cost | Permutation `p+` | Paired delta | Verdict |
|---:|---:|---:|---|
| `3500 ms` | `0.00115` | `+66.7 s` | Certified |
| `5000 ms` | `0.00015` | `+150.6 s` | Certified |
| `4500 ms` | `0.0094` | `+106.2 s` | Positive, not certified |
| `4000 ms` | `0.0139` | `+65.6 s` | Positive, not certified |

All ten cells were directionally positive. The two certified cells were broad:

- `kv=3500 ms`: 53 positive tasks; top three contributed 17.6% of positive
  mass; the drop-top-three diagnostic remained `p=0.0005–0.0010`.
- `kv=5000 ms`: 113 positive tasks; top three contributed 11.3%; the
  drop-top-three diagnostic remained `p<=0.0006`.

Both cells strengthened when the largest task was removed, and repository-
clustered resampling corroborated them. Exploratory mechanism analysis found
that all policy divergence occurred on `exec`; apt-related prefixes carried
most of the `kv=5000 ms` positive mass across many tasks.

## H2 — Command-prefix trie versus tool-name conditioning

**Not certified.** All ten point estimates were positive, from `+9.9 s` at
`kv=500 ms` to `+107.4 s` at `kv=5000 ms`, but the minimum `p+` was `0.0037` at
`kv=500 ms`, above the preregistered `0.0025` threshold.

The exact conclusion is directional replication without certification. The miss
was dispersion rather than a sign reversal: sign-flip `z=2.68` versus `2.81`
required, with per-task standard deviation about six times the mean at
`kv=500 ms`. Later development work on prefix normalization did not rescue this
claim and is closed in [`../CLOSED-QUESTIONS.md`](../CLOSED-QUESTIONS.md).

## Evidence boundary

H1 supports offline initialization and certification of the fixed policy at the
measured operating point. It does not certify later adaptive states and does not
establish live multi-tenant JCT, TTFT, throughput, or contention effects. H2
must remain an honest negative.

## Integrity amendments and deviations

These were reviewer-assessed as non-invalidating and remain part of the frozen
record:

- The run migrated from the 8-core collection host after a five-fold GBM
  concurrency OOM to a 28-core, 78 GB host. NumPy was pinned to 2.5.1, manifest
  paths were patched, and protocol/Git/pip provenance dependencies were added.
  Three aggregate restarts occurred before any result was written.
- `rho=0.0` was run as a Mode-B plumbing/refit anchor and consumed roughly 70%
  of step-3 wall time. This was identified post hoc. Subsequent protocols use a
  cheap trigger-equality anchor and reserve downstream scoring for the measured
  operating point.
- The corpus directory name ends in `n200`, but the preregistered extension
  contains 277 tasks. The frozen `task_ids.txt` count and manifest govern.
