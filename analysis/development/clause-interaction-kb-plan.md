# Non-Trie Clause Interaction KB — SQLGlot Plan

**Effective:** 2026-08-04
**Status:** SQLGlot development plan; command baseline visible, no subset result yet
**Scope:** SQLGlot command latency first; CPU, RSS, and Disk evaluation only
after the latency mechanism passes

This plan extends the representation-and-arbitration work in
`tool-resource-canonical-objective.md`. It does not change the fixed latency
buckets, resource thresholds, eligibility rules, causal visibility, or the
single offline/online predictor contract.

## 1. Question

The current KB can reuse an exact clause, an ordered argv prefix, or a bare
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

**Hypothesis:** within a repository, shared argument interactions carry useful
causal resource information beyond exact/prefix/bin matching. Compare two
genuinely non-trie KB architectures: a materialized feature-set poset queried by
its maximal interaction frontier, and an episodic observation memory queried by
an all-subset kernel. The current trie is a control, not a candidate.

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

The normalized binary partitions candidate histories but is not counted as an
interaction feature. Thus candidates may share arguments only within the same
binary, while a history sharing no non-binary feature receives no local weight.

The existing exact-clause hash lookup remains a shared shortcut in every arm.
Both non-trie methods are consulted only after an exact miss and never consult
prefix or binary trie nodes.

### 2.2 Architecture A — interaction-poset frontier

Store observations under their complete typed non-binary feature set. Distinct
feature sets are poset nodes ordered by strict set inclusion; a node owns only
the observations inserted at that exact set.

For query features `Q`, intersect `Q` with every causally visible node `H`, drop
empty intersections, and collapse equal intersections. Retain the maximal
intersection sets: an intersection `I` is on the frontier when no other
observed intersection is a strict superset of `I`. Every observation owned by a
node whose intersection lies on that frontier contributes once with uniform
weight. Dominated weaker matches contribute nothing. This is a feature-set
poset with a query-induced maximal frontier, not a prefix tree.

Pool the resulting local empirical distribution with the frozen public PMF
using `alpha = 16` and `n_eff = number of distinct contributing observations`.
When the frontier is empty, return the public PMF.

### 2.3 Architecture B — episodic all-subset kernel

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
features, pairs, or higher-order interactions. They are explanatory ablations,
not alternatives that may replace a failed primary candidate. For each `k`,
apply the same truncation to the query and history self-counts in the
denominator; this preserves the length normalization rather than changing only
the cross-term.

### 2.4 Prediction and public evidence

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
on SQLGlot. When there is no non-binary match, return the public PMF.

Compound commands require values rather than bucket IDs. Represent the same
posterior as a weighted empirical value distribution: normalize local weights
to total mass `n_eff`, give uniform public values total mass `alpha`, and use
the canonical 256 deterministic stratified draws over its weighted CDF before
shell-stage composition. An exact local shortcut remains its unweighted
empirical distribution and does not pool public evidence. Single-clause PMFs
are computed directly from weights without drawing.

This defines three required predictor arms on identical rows, plus majority:

1. **Current:** raw exact/prefix/bin hard first-hit backoff.
2. **Interaction poset + pooling:** maximal non-dominated shared feature sets,
   with no prefix/bin trie lookup.
3. **Subset kernel + pooling:** query-time all-subset matching with the same
   public PMF and `alpha = 16` pooling.

Arms 2 and 3 are co-primary architectural alternatives. Compare each against
Current and majority, and compare them directly to distinguish frontier
selection from dense similarity weighting. A win by either does not erase the
other result.

## 3. Data and causal protocol

### SQLGlot100: development only

Use the collected `tobymao/sqlglot` cohort as the development stream. All
representation inspection, debugging, ablations, and implementation choices
spend this cohort. It cannot support a confirmation claim.

Replay manifest order as a serialized deployment:

1. predict every eligible command in one task from its parsed clauses;
2. reveal none of that task's observations while the task is running;
3. after successful task/trace finalization, release its eligible observations
   to later tasks.

Reuse the frozen public input and filtering recorded by the current SQLGlot
command baseline; it excludes the target repository. Every arm receives that
same public state and the same ordered observation stream. Command latency truth
is the matching tool-call duration. Clauses remain internal evidence and their
empirical values are composed by the canonical shell execution graph.

## 4. Staged execution

### Stage 0 — validity and signal audit

Implement one slow, exact causal evaluator. It may scan all prior observations
in the active repository; do not build an index yet.

Verify on a hand-checkable example that:

- the closed-form kernel equals explicit subset enumeration;
- the poset returns exactly the maximal non-dominated intersections;
- option reordering matches while positional reordering does not;
- one historical observation contributes once;
- observations from the current, future, failed, or unfinalized task are
  invisible;
- both candidates avoid every prefix/bin trie lookup;
- all arms score identical eligible command IDs, labels, and availability.

On SQLGlot, report without selecting a model:

- the distribution of shared-feature count `m` after exact misses;
- how many queries have a prior-task match with `m >= 1`, `m >= 2`, and
  `m >= 3`;
- the number of distinct prior tasks contributing to each prediction;
- effective sample size and warm-up position;
- how often each non-trie architecture changes the current command prediction;
- a hindsight-only command oracle over current and both non-trie predictions,
  clearly marked unavailable at inference time.

Stop if neither non-trie architecture changes a prediction through a non-exact
match, or the three-way oracle cannot beat both current and the same-row
majority baseline. That would show there is no decision-relevant signal to
justify runtime work.

### Stage 1 — SQLGlot development decision

Run the three required predictor arms and the fixed subset-kernel `k = 1, 2, 3`
ablations. Poset and full all-subsets kernel are co-primary architectures;
bounded-order results explain the kernel and cannot replace a failed full
kernel.

Report:

- exact command-level three-class latency accuracy, higher is better;
- same-row majority accuracy and 3x3 confusion matrix;
- paired task-cluster uncertainty for each non-trie arm minus Current and for
  their direct difference;
- prediction changes split into helpful and harmful;
- results by task-order quartile to expose warm-up behavior;
- lookup p50/p95, peak evaluator memory, and stored observation count.

Use a paired task-cluster percentile bootstrap with seed `0` and `2000` draws.
A command is a non-exact carrier for an architecture when at least one clause
uses positive-weight local evidence after an exact miss. Its carrier net gain
is `helpful - harmful` among carrier commands whose hard prediction differs
from Current.

Proceed to resource evaluation for every non-trie architecture that beats both
Current and majority on SQLGlot and whose net correctness gain is positive on
commands changed through non-exact matches. Otherwise stop with a mechanism
diagnosis; do not search feature weights, support thresholds, kernels, or
argument parsers.

### Stage 2 — resource and systems evaluation

Only after the SQLGlot latency GO, apply the frozen representation and matching
rule on the same command stream to CPU, RSS, and Disk. Preserve their current
independent Heavy/Light thresholds, short-null policy, rows, and labels. Do not
retune the kernel or alpha per target.

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
- Keep command composition, public evidence, and causal replay identical across
  arms. No analysis-only winning implementation may be described as deployable.
- The two candidate predictors must not call or reconstruct exact/prefix/bin
  trie backoff. Only the current control may consult it; exact hash lookup and
  frozen public binary/global evidence are shared separately.
- Put reusable evaluation logic in one existing evaluation module and one
  focused test. The test must cover closed-form versus enumerated subsets and
  the task-settlement boundary.
- Profile a representative SQLGlot slice before any run expected to exceed 30
  minutes; report the estimate and request approval rather than shrinking the
  approved cohort.
- Obtain one bounded independent review of the evaluator and causal protocol
  before its output is used as scientific evidence.
- Do not change `resource-agentd`, snapshots, persistence, or collection; all
  stages in this plan are offline diagnostics, and no new collection is needed.

## 6. Outputs

One result directory is sufficient:

```text
analysis/results/tool-resource-clause-interactions-<date>/
```

It should contain machine-readable SQLGlot development results, the frozen
configuration, and only the figures needed to explain coverage, accuracy,
warm-up, and cost. Rewrite this plan only when the protocol changes; preserve
completed result artifacts unchanged.
