# Clause Interaction KB — Query-Time Subset Matching Plan

**Effective:** 2026-08-03  
**Status:** development plan; no implementation or result claim yet  
**Scope:** SWE-ReBench clause latency first; CPU, RSS, and Disk transfer only
after the latency mechanism passes

This plan extends the representation-and-arbitration work in
`tool-resource-canonical-objective.md`. It does not change the fixed latency
buckets, resource thresholds, eligibility rules, causal visibility, or the
single offline/online predictor contract.

## 1. Question

The current KB can reuse an exact command, an ordered argv prefix, or a bare
binary. Ordered prefix matching is a poor proxy for command semantics: two
commands may share the interaction that matters even when an unrelated path or
value appears between the matching arguments.

For example, a query with features `{a, b, c}` should be able to use histories
matching `{a, c}` and `{a, b}`:

```text
{a} -> {a,c} -> {a,b,c}
{b} -> {a,b} -> {a,b,c}
```

The current prefix chain can represent only one argv-order-dependent path.

**Hypothesis:** within a repository, all shared argument interactions carry
useful causal resource information beyond exact/prefix/bin matching. Computing
those matches at query time can recover the signal without materializing an
exponential subset lattice.

## 2. Candidate mechanism

### 2.1 Structured clause features

Reuse the existing `generic-argv-v3-role` parsing and value canonicalization.
Turn its output into typed features:

- normalized binary;
- safe subcommand or its value shape;
- option name plus canonicalized value shape, retaining multiplicity;
- ordered positional slot plus canonicalized value shape;
- the explicit `--` boundary.

Options remain order-insensitive. Positional slots and subcommands remain
order-sensitive. Give repeated equal options a deterministic occurrence index,
so the representation remains a set without losing multiplicity. Raw paths,
opaque IDs, secrets, timeouts, target labels, and post-execution measurements
never become matching features.

The existing exact-clause lookup remains a shared shortcut in every arm. The
new method is consulted only after an exact miss, so the experiment isolates
non-exact sharing.

### 2.2 All subset matches without a materialized lattice

Store each causally visible historical observation once, with feature set
`H`. For a query with feature set `Q`, let:

```text
m = number of non-binary features in Q intersection H
```

Two clauses with `m` shared features have this many shared non-empty subsets:

```text
2^m - 1
```

Therefore the full subset-lattice inner product can be computed from `m`
without creating every subset node. Use its length-normalized form so long
commands do not win merely by containing more arguments:

```text
K(Q,H) = (2^m - 1)
         / sqrt((2^|Q|-1) * (2^|H|-1))
```

Here `|Q|` and `|H|` count non-binary features. A history sharing only the
binary has weight zero; it must not recreate the previously harmful hard
repo-local binary fallback. Define `K = 0` when either clause has no non-binary
feature, avoiding the empty-set denominator.

Each prior observation contributes exactly one weighted label. It is never
inserted into and then summed from several overlapping subset nodes.

For diagnosis, also report bounded-order kernels:

```text
K_k(Q,H) numerator = sum(comb(m, j), j=1..min(k,m))
```

for `k = 1, 2, 3`. These show whether the signal comes from individual
features, pairs, or higher-order interactions. They are ablations, not
alternative DVC candidates selected after seeing DVC results.

### 2.3 Prediction and public evidence

For each class, sum the kernel weights of its causal repository-local
observations and normalize them into `local_pmf`.

Kernel weights are similarities, not independent sample counts. Compute the
effective number of distinct observations as:

```text
n_eff = sum(w)^2 / sum(w^2)
```

Then combine the local PMF with the same frozen public PMF used by the control:

```text
posterior = (n_eff * local_pmf + alpha * public_pmf)
            / (n_eff + alpha)
```

Use the already development-selected `alpha = 16`; do not tune another alpha
on SQLGlot or DVC. When there is no non-binary match, return the public PMF.

This defines three required arms on identical rows:

1. **Current:** raw exact/prefix/bin hard first-hit backoff.
2. **Prefix + pooling:** current prefix candidates with the same public PMF and
   `alpha = 16` pooling.
3. **Subset kernel + pooling:** query-time all-subset matching with the same
   public PMF and `alpha = 16` pooling.

Arm 3 versus Arm 2 isolates the value of non-prefix interactions. Arm 2 versus
Arm 1 separately measures the value of pooling.

## 3. Data and causal protocol

### SQLGlot100: development only

Use the collected `tobymao/sqlglot` cohort as the development stream. All
representation inspection, debugging, ablations, and implementation choices
spend this cohort. It cannot support a confirmation claim.

Replay manifest order as a serialized deployment:

1. predict every eligible clause in one task;
2. reveal none of that task's observations while the task is running;
3. after successful task/trace finalization, release its eligible observations
   to later tasks.

The public state is frozen before replay and excludes all SQLGlot and DVC
observations. Every arm receives the same public state and the same ordered
observation stream.

### DVC72: frozen transfer

Freeze the feature extraction, kernel, alpha, tie-breaking, and all decision
criteria before reading DVC outcomes.

Seven DVC task IDs already occur in the development-exposed SWE-100/277
corpora:

```text
iterative__dvc-2254
iterative__dvc-2462
iterative__dvc-2866
iterative__dvc-3405
iterative__dvc-3727
iterative__dvc-4108
iterative__dvc-5822
```

Run all 72 tasks in their fixed order, but score the primary transfer result on
the other 65 task IDs. The seven excluded IDs may contribute their newly
collected observations to later tasks only after normal causal finalization.
Describe the result as **fresh-task transfer within a development-known
repository**, not untouched-repository confirmation.

## 4. Staged execution

### Stage 0 — validity and signal audit

Implement one slow, exact causal evaluator. It may scan all prior observations
in the active repository; do not build an index yet.

Verify on a hand-checkable example that:

- the closed-form kernel equals explicit subset enumeration;
- option reordering matches while positional reordering does not;
- one historical observation contributes once;
- observations from the current, future, failed, or unfinalized task are
  invisible;
- all arms score identical eligible row IDs and labels.

On SQLGlot, report without selecting a model:

- the distribution of shared-feature count `m` after exact misses;
- how many queries have a prior-task match with `m >= 1`, `m >= 2`, and
  `m >= 3`;
- the number of distinct prior tasks contributing to each prediction;
- effective sample size and warm-up position;
- how often subset matching changes the current prediction;
- a hindsight-only oracle over current versus subset prediction, clearly
  marked unavailable at inference time.

Stop if non-prefix matches never change a prediction or the oracle cannot beat
both current and the same-row majority baseline. That would show there is no
decision-relevant signal to justify an index or runtime work.

### Stage 1 — SQLGlot development decision

Run the three required arms and the fixed `k = 1, 2, 3` ablations. The full
all-subsets kernel is the primary candidate; bounded-order results explain the
mechanism and cannot replace a failed primary result.

Report:

- exact three-class latency accuracy, higher is better;
- same-row majority accuracy and 3x3 confusion matrix;
- paired task-cluster uncertainty for Arm 3 minus Arms 1 and 2;
- prediction changes split into helpful and harmful;
- results by task-order quartile to expose warm-up behavior;
- lookup p50/p95, peak evaluator memory, and stored observation count.

Proceed only if the full kernel beats current, prefix + pooling, and majority
on SQLGlot and the gain is actually carried by queries with non-prefix
matches. Otherwise stop with a mechanism diagnosis; do not search feature
weights, support thresholds, kernels, or argument parsers.

### Stage 2 — freeze and DVC transfer

Before opening DVC result labels, write the frozen result-affecting
configuration into the SQLGlot result artifact. Then run exactly one DVC
transfer evaluation.

The primary comparison is:

```text
accuracy(subset kernel + pooling) - accuracy(prefix + pooling)
```

GO requires:

- the paired task-cluster 95% interval for the primary difference has lower
  endpoint above zero;
- the subset arm also exceeds current and same-row majority accuracy;
- no arm changes eligible rows, labels, or prediction availability;
- non-prefix matches are selected often enough to account for the observed
  prediction changes.

Report leave-one-task influence as a fragility diagnostic, not as an additional
selection gate. A result dominated by one task must be described as fragile.

### Stage 3 — resource and systems transfer

Only after the latency GO, apply the frozen representation and matching rule to
CPU, RSS, and Disk. Preserve their current independent Heavy/Light thresholds,
short-null policy, rows, and labels. Do not retune the kernel or alpha per
target.

Only after predictive evidence exists should implementation work address
serving cost. Profile the slow evaluator first. If lookup cost matters, add the
smallest sufficient inverted index from feature to observation IDs; materialize
or cache hot intersections only if measured lookup pressure remains. A full
concept-lattice graph, ANN service, neural model, and new runtime dependency are
out of scope.

## 5. Integrity and implementation gates

- Use only information available before clause execution; timeout, outcome,
  measured latency, CPU, RSS, Disk, and task identity are forbidden features.
- Preserve `observation.ts_end < query.ts_start` and successful task/trace
  settlement before release.
- Fit no vocabulary, parameter, or public state on the target repository's
  evaluation labels.
- Keep offline replay and any eventual online path on the same
  `ClauseResourceKB` semantics; no analysis-only winning implementation may be
  described as deployable.
- Put reusable evaluation logic in one existing evaluation module and one
  focused test. The test must cover closed-form versus enumerated subsets and
  the task-settlement boundary.
- Profile a representative SQLGlot slice before any run expected to exceed 30
  minutes; report the estimate and request approval rather than shrinking the
  approved cohort.
- Obtain one bounded independent review of the evaluator and causal protocol
  before its output is used as scientific evidence.
- Do not change `resource-agentd`, snapshots, persistence, or collection while
  Stages 0–2 remain offline diagnostics.

## 6. Outputs

One result directory is sufficient:

```text
analysis/results/tool-resource-clause-interactions-<date>/
```

It should contain machine-readable SQLGlot development results, the frozen
configuration, the single DVC transfer result, and only the figures needed to
explain coverage, accuracy, warm-up, and cost. Rewrite this plan only when the
protocol changes; preserve completed result artifacts unchanged.
