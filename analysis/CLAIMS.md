# Current claims and evidence limits

The paper studies one object: a conditional residual-time prior for tool calls,
used to price KV-cache actions. Swap-out and pre-restore are two decision points
on the same call lifecycle, not independent modules.

## C1 — Priced stopping over fixed latency priors

**Supported offline claim.** Pre-restore passed its predeclared offline accounting
gate under both trigger sources. The robust clock yields `+156.2 s/277` at
`kv=3500 ms` (clustered 95% CI `[60.4, 256.6]`) and `+317.9 s/277` at
`kv=5000 ms` (`[181.0, 460.4]`). These numbers account for restore cost but
cannot measure live PCIe contention or serving interference. See
[`certification/rolling-survival-design-20260720.md`](certification/rolling-survival-design-20260720.md).

**Not yet supported.** There is no claim that the policy improves live
multi-tenant JCT, P99 TTFT, throughput, or GPU-memory efficiency. W5 has no
result. The harness remains in development and is blocked by a missing prefill
profile and an unresolved `rho=1.0` trigger-table versus `rho=0.94` runtime
contract.

## C2 — Decision utility, not estimator fit, decides what ships

**Supported claim.** Conditioning refinements must be judged by policy utility
at the operating point. Wrapper transparency is the concrete counterexample:
the apparent 21 ms fit improvement was confounded with a support gate, while the
joint policy was directionally harmful by `-59.6 s/277` at `kv=3500 ms`. The
gate therefore prevented an estimator-motivated regression.

Future Continuum experiments provide the estimator-class comparison.

## C3 — The elapsed-only re-check space is closed

**Supported claim.** With call survival as the only runtime observation, any
elapsed-only multi-check plan reduces to one precomputable stopping time. The
independent `k=2` dynamic program matched `k=1` on all 670 prior nodes across 10
KV cells, with maximum value gap `0.0 ms` at tolerance `1e-6`, and becomes
strictly worse when check overhead is priced.

Boundary identity is the measured flank: it changed 72.9% of decisions but had
paired log-score gain `-0.0137` nats with task-clustered 95% CI
`[-0.41, +0.20]`. It adds churn without measurable information.

This closure is limited to elapsed-only and observed-boundary enrichments under
the frozen functional. It does not resolve real multi-tenant pressure, which
Fresh-277 cannot represent.

## Adaptive deployment candidate

Completed-task publication is the sole retained adaptive candidate. In the
fixed-order, five-fold development replay, it improved on the warmup snapshot
by `+0.362 s/277` at `kv=3500 ms` and `+9.050 s/277` at `kv=5000 ms`. Per-call
publication added exactly zero realized utility beyond completed-task updates.
The evidence is
[`results/prequential-task-update-20260721/`](results/prequential-task-update-20260721/).

This screen is not an unopened-stream evaluation, a deployment certificate, or
evidence of order robustness. Same-repository conditioning is dropped, and the
same-trace B1 result remains a negative baseline rather than an active policy
direction.

## Open question — is the advantage over the fixed deadline a container artifact?

**Criterion frozen 2026-07-29, before the attribution result exists.**

Visible when frozen: C1, C2, C3 and the adaptive screen above; the composition of
`analysis/serving/w5-multitenant/trigger_table_swe100_rho1_kv5000.json`, which retains
**5 group keys of 91 candidates** over 4,640 rows, four of them `apt-get` prefixes and
the fifth `cd /testbed && head`, with every other group falling through to the
deadline; and the Continuum paper (arXiv 2511.02230), which sets KV TTL from an
empirical-Bernstein bound on per-tool duration statistics, reports no fixed-TTL and no
oracle baseline, and whose own ablation states a simplified fixed-threshold variant
achieves a significant portion of its gains.

`/testbed` is the SWE-bench container path. If the priced-stopping advantage over
`deadline_only` is concentrated in container package-manager setup, it is a dataset
prior rather than a method (§3.1), and it sits underneath the live comparison W5 is
built around. The repository record contains no examination of this.

**Question.** What share of the `robust_clock` advantage over `deadline_only` comes
from container-setup commands, and does the advantage survive their removal?

**Frozen definitions.** A call is container-setup if its command, after stripping a
leading `cd <path> &&`, begins with any of `apt-get`, `apt `, `pip install`,
`conda `, `apk `, `yum `, or `dpkg`; or if the command contains the literal `/testbed`
outside a quoted string. Declared once here and never tuned against the result.

**Frozen protocol.** Fresh-277 via `configs/corpora/swe-277.json`, five outer folds,
`guard_ms = 0`, `restore_cost_fraction = 0.94`, primary cells `kv = 3500` and
`5000 ms`. Primary quantity is `net_saved_ms` of `robust_clock` minus `deadline_only`
in `s/277`, matching C1's units. Profile/eval task disjointness is asserted by the
evaluator and must be preserved by the ablation; all arms read one `decisions` list, so
row identity is structural. `absorbed_if_oracle_ms` remains analysis-only.

**Frozen bands.** Container-setup share of the advantage `>= 0.60` in both cells is an
artifact finding. `<= 0.25` in both cells is a workload regularity. Anything else is
reported as inconclusive, not resolved by moving the band. The ablated delta carries a
task-clustered bootstrap interval, as C1 does.

**Consequence declared in advance.** An artifact finding means the offline advantage
does not generalise and W5's `ours` arm should not receive GPU time until re-based. It
does not retract C1, which is a pre-restore claim on a different decision point.

## Open question — does runtime CPU activity beat survival alone on high-variance calls?

**Criterion frozen 2026-07-29, before the implementation exists.**

Visible when frozen: C1-C3, the closed-questions register, the container-artifact
attribution (`04ef2bd`), the per-group dispersion measurements (`apt-get update &&`
`P(long)=95.7%` `CV=0.40`; `python3 -m pytest` `P(long)=12.0%` `CV=1.79`; pytest the
largest long-call group at 179 long of 1,489), and the label-distribution facts in the
next paragraph. No runtime-activity arm has been implemented or run.

**Acceptance bar, derived not fitted.** An early swap saves at most `K` on a long call
and costs `rho*K` on a short one, so acting is profitable only where
`P(long | causal state) > rho/(1+rho) = 0.94/1.94 = 48.5%`. A candidate must lift a
subgroup **across** that bar; improving AUC, MAE or log-score is not evidence.

**Why this is not closed already.** C3 closes elapsed-only multi-check schemes and
scopes itself to "elapsed-only and observed-boundary enrichments". Boundary-conditioned
re-checks, atom/segment models, stable-atom screening and per-call self-footprint
pricing are closed separately. A per-500ms CPU-activity trajectory is none of those: it
is strictly richer than elapsed time, so it lies outside every recorded closure.

**Observable.** `data.resource_timeline` on `tool_exec` records, present on all 8,976
`exec` calls of the SWE277 replay trace: 500 ms sampling with `offset_s`, `dt_s`,
`cpu_core_s`, `net_rx_bytes`, `net_tx_bytes`, `cpu_quota_cores`. Only samples whose
window has completely ended before the decision may be read.

**Corpus.** The SWE277 **replay** trace, not the Fresh-277 source corpus, because the
timeline lives there. Its tool durations are real re-executions but carry container and
eBPF overhead, so they are systematically slower than source; the arms are compared
against each other on identical rows, so the overhead cancels in the comparison but the
absolute seconds are not comparable to C1.

**Landmark and population.** `t = 1000 ms`, the first point with two complete 500 ms
windows. Population is calls with `duration_ms > 1000`: 1,794 of 8,976 (20.0%).
Recorded before the run, `P(long)` among those survivors is **48.4%** at `kv=3500` and
**37.4%** at `kv=5000` — so survival alone already sits at the bar in one cell, and the
feature must beat survival, not the unconditional rate.

**Arms**, identical rows, `guard_ms = 0`, `restore = 0.94*kv`:

1. `deadline_only` — fire at the threshold. Structurally misses nothing.
2. `elapsed_only@1s` — fire at the landmark for every survivor. No feature.
3. `cpu_state@1s` — fire at the landmark only when the rule says long.

**Rule.** One decision stump on a single feature, cumulative `cpu_core_s` divided by
observed wall time over completed windows, its threshold chosen on profile tasks to
maximise net utility. No model family sweep, no second feature, no second landmark.

**Gate.** Development GO only if `cpu_state@1s` net utility is strictly greater than
**both** `deadline_only` and `elapsed_only@1s`, in **both** `kv = 3500` and `5000`,
out-of-fold under five task-grouped folds, **and** the fired subgroup's realised
`P(long)` exceeds `48.5%` in both cells. Margin over the stronger comparator carries a
task-clustered bootstrap interval.

**Not authorized.** Adding landmarks, features, or model capacity after reading;
substituting a fit metric for the utility gate; or reporting the fired subgroup's
purity without the utility comparison.

## Open question — what is the total budget available to any duration predictor?

**Criterion frozen 2026-07-29, before the numbers exist.**

Visible when frozen: C1-C3, the closed-questions register, the container-artifact
attribution (`04ef2bd`), four committed NO-GOs, and tonight's correction (`adeccb2`)
retracting the intra-call envelope figure. No oracle-versus-deadline gap has been
computed on this corpus under this functional.

**Question.** Published agent-serving systems attribute their gains to prediction
without isolating it. Fix the mechanism — a single stopping time under the utility
functional — and vary only the predictor, from none to perfect. The gap between a
fixed deadline and perfect foreknowledge is the entire budget any predictor can ever
claim.

**Arms**, identical rows, `guard_ms = 0`, `restore = 0.94*kv`, cells `kv = 3500` and
`5000 ms`:

1. `never_act` — utility 0 by construction.
2. `deadline_only` — fire at the threshold. No prediction.
3. `mean_hazard`, `robust_clock` — the shipped predictors.
4. `ORACLE` — perfect foreknowledge of the realised latency `L`. Fires only when
   `L > threshold`, at `max(0, L - kv)`, the latest trigger that still hides the full
   swap. Analysis-only, never deployable, never a feature.

**Primary quantity.** `ORACLE - deadline_only`, in `s/277`. That is the budget.
Secondary: `(robust_clock - deadline_only) / (ORACLE - deadline_only)`, the fraction
of the budget the shipped predictor captures.

**Interpretation fixed in advance.** If the budget is under 10% of `deadline_only`'s
own utility, prediction is structurally marginal for this workload regardless of
predictor quality, and no future predictor result on this corpus should be presented
as a system contribution. If it exceeds 30%, prediction is worth continued
investment. Between 10% and 30% is reported as inconclusive and not resolved by
moving the band.

**Premise, binding on every number.** The functional credits `min(kv, remaining)` for
hiding swap-out and charges the differential restore only on short calls. That is
coherent only under forced eviction. With no memory pressure the optimal policy is
never swap and every arm scores zero. `CLOSED-QUESTIONS.md` establishes Fresh-277
cannot exhibit contention. The budget computed here is therefore in
hidden-swap-milliseconds under an assumed forced-eviction regime, **not** wall clock,
and the ORACLE bound is conditional on that same regime.

**Causal contract.** The oracle reads the realised latency and is labelled
analysis-only; the deployable arms read profile-fold evidence only, with the task
settlement barrier enforced by `evaluate_utility_clock_policy`.

**Not authorized.** Reporting the oracle as an achievable result; using its per-call
triggers as features; or moving the interpretation bands after reading.

## Explicit non-claims

- No live W5 or headline systems result exists.
- No contention conclusion comes from Fresh-277; collection used a cloud model
  and had no shared KV cache.
- No per-call, same-repository, same-trace, wrapper-normalized, atom/segment, or
  boundary-conditioned policy is an active direction.
- The removed online-first replay is protocol-invalid for deployment and
  contributes no positive or negative paper result.
