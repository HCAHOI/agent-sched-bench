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

## Open question — can any per-key empirical method reach the prediction budget?

**Criterion frozen 2026-07-29, before the numbers exist.**

Visible when frozen: everything above, plus `57cbbbb` — the budget is 1392.1 s
(88.9% of deadline utility) at `kv=3500` and 1477.6 s (84.8%) at `kv=5000`, it lies
**entirely** in calls with `kv < L < 2*kv`, and the shipped `robust_clock` captures
1.4% and 8.0% of it. Also visible: `hazard_recheck_ms` in `src/tool_time/prior.py`
already computes the EXACT expected-utility-maximising trigger over a node's sample
distribution, enumerating breakpoints at `L` and `L - kv` with no grid and no tuning,
and it is shipped as `mean_hazard`, which scores 1512.9 s against `deadline_only`'s
1565.4 s at `kv=3500`.

**Why this is the right question.** The exact maximiser over per-key empirical
evidence already exists and loses to a fixed deadline. Either the per-key evidence
does not transfer from profile to evaluation folds, or the key itself carries no
information about band membership. These have opposite consequences: the first
argues for better estimation, the second kills the entire per-key empirical family —
including Continuum's `P(tau,f)`, which is exactly a per-key empirical CDF.

**Decomposition to compute.** For each command key with support in both splits:

1. `profile_trigger` — the utility-maximising trigger from profile-fold samples.
2. `eval_optimal_trigger` — the utility-maximising trigger computed in-sample on the
   evaluation rows for that key. **Analysis-only**, a per-key oracle, never
   deployable and never used as a feature.
3. `ORACLE` — per-call perfect foreknowledge, from `57cbbbb`.

Report utility for each and split the gap: `ORACLE - eval_optimal` is the part no
per-key method can ever reach because it is within-key variance; `eval_optimal -
profile_trigger` is the part attributable to estimation error and is in principle
recoverable.

**Interpretation fixed in advance.** If `eval_optimal` captures under 20% of the
budget in both cells, the per-key empirical family is dead for this workload
regardless of estimator quality, and that conclusion transfers to Continuum's
estimator by construction. If it captures over 50%, estimation is the bottleneck and
better per-key estimation is worth building. Between is inconclusive and is not
resolved by moving the band.

**Premise, binding.** The functional is conditional on forced eviction; with no
memory pressure every arm including both oracles scores zero. Fresh-277 cannot
exhibit contention. All numbers are hidden-swap-milliseconds under an assumed
regime, not wall clock.

**Causal contract.** `profile_trigger` uses profile-fold samples only, under the
existing five task-grouped folds with disjointness asserted. Both oracles are
labelled analysis-only and reported separately.

## Open question — is the within-key variance irreducible, or an artifact of key choice?

**Criterion frozen 2026-07-29, before the numbers exist.**

Visible when frozen: `57cbbbb` (budget 1392.1 s / 1477.6 s, entirely in `kv < L < 2*kv`)
and `3bc4906` (per-key ceiling 32.1% / 29.3%, with ~70% of the budget attributed to
within-key variance). That 70% was computed against **one** key: the deepest
command-prefix key at `max_prefix_depth=4`. Whether it is a property of the workload
or of that key is untested.

**Question.** Compute the per-key oracle ceiling as a function of key richness. If the
ceiling rises materially as the key sharpens, the variance is reducible and keying is
the lever. If the curve is flat past the command prefix, the variance is genuinely
within-command and no member of the per-key family — Continuum's included — can reach
it.

**Key ladder**, coarse to fine, each scored identically:

1. `tool_name` — Continuum's granularity.
2. `cmd_depth_1` — binary only.
3. `cmd_depth_2`.
4. `cmd_depth_4` — the shipped key, already measured in `3bc4906`.
5. `repo+cmd_depth_4` — adds repository identity, the richest key available offline.

**Metric.** For each key, the per-key in-sample oracle ceiling as a fraction of the
per-call ORACLE budget, at `kv = 3500` and `5000`. All ceilings are ANALYSIS-ONLY and
optimistic by construction, since each is fitted in-sample on the rows it scores; the
comparison between them is the object of interest, not their absolute level.

**Interpretation fixed in advance.** If `repo+cmd_depth_4` exceeds `cmd_depth_4` by
more than 15 percentage points of the budget in both cells, keying is the lever and
richer keys are worth building. If it gains under 5 points in both, the within-key
variance is irreducible by keying and the entire per-key empirical family is bounded
near its current ceiling. Between 5 and 15 is inconclusive and is not resolved by
moving the band.

**Expected secondary result, recorded before the run.** `tool_name` should score
lowest, quantifying how much Continuum's granularity leaves on the table relative to
a command-prefix key. That comparison is the point of including it.

**Premise, binding.** Forced eviction; with no memory pressure every arm including
every oracle scores zero. Fresh-277 cannot exhibit contention. Hidden-swap
milliseconds, not wall clock.

**Causal contract.** Ceilings are in-sample per key within each evaluation fold and
are labelled as oracles; no ceiling is presented as achievable. The five task-grouped
folds partition task ids, so fold disjointness is structural.

## Open question — the fine-key gain under an unconfounded estimator

**Criterion frozen 2026-07-29, before the numbers exist.**

Visible when frozen: `8e7ed28`, where the same measurement gave GO by the letter of
its criterion (repo gain 38.5 / 45.7 pp) and INCONCLUSIVE once restricted to groups
of five or more rows (15.0 / 13.9 pp). The disagreement is caused by a defect I
recorded there: in-sample per-key ceilings are confounded with key cardinality, and
singleton groups contribute 25.6% and 35.6% of the fine-key ceiling while carrying
zero information.

**Question.** Settle the fine-key gain with leave-one-out, which removes the
degeneracy exactly rather than by restriction. For each row, fit the trigger on the
key's **other** rows within the same evaluation fold and score the held-out row. A
singleton group then has no other rows and correctly falls back to the deadline,
contributing nothing.

**Scope, and why it is narrower than the previous ladder.** Only `cmd_depth_4` and
`repo+cmd_depth_4` are evaluated. Those are the two rungs in dispute, and their
groups are small (about 10 and 3 rows per key) so exact LOO is tractable. The coarse
rungs are excluded deliberately: `tool_name` holds roughly 2682 rows per key, where
LOO is both computationally quadratic and numerically pointless because removing one
of 2682 samples cannot move the fitted trigger materially. The previous attempt
timed out precisely because it included them.

**Metric.** LOO ceiling for each key, as a fraction of the per-call ORACLE budget, at
`kv = 3500` and `5000`. Also report the in-sample ceiling alongside so the size of
the overfit is visible.

**Interpretation, bands unchanged from `817edc7` so the two runs are comparable.**
`repo+cmd_depth_4` beating `cmd_depth_4` by more than 15 percentage points of the
budget in both cells means keying is the lever. Under 5 points in both means the
within-key variance is not materially reducible by adding repository identity.
Between 5 and 15 is inconclusive and is not resolved by moving the band.

**Premise, binding.** Forced eviction; with no memory pressure every arm including
the oracle scores zero. Fresh-277 cannot exhibit contention. Hidden-swap
milliseconds, not wall clock.

**Causal contract.** LOO is computed within an evaluation fold, so no profile-fold
information is used and no row informs its own trigger. The five task-grouped folds
partition task ids. The LOO ceiling remains an ORACLE in the sense that it uses
evaluation-fold rows of the same key; it is analysis-only and is not a deployable
result.

## Open question — is the deadline itself the best constant policy?

**Criterion frozen 2026-07-29, before the numbers exist.**

Visible when frozen: the budget (`57cbbbb`), the per-key ceiling (`3bc4906`), the key
ladder (`8e7ed28`) and the leave-one-out refutation (`f1cbee3`), which together show
that no per-key empirical method beats `deadline_only` on this corpus without
leakage. Every one of those comparisons used the same baseline: fire at
`trigger = threshold = kv + guard`. That choice has never been tested.

**Question.** The budget lies entirely in calls with `kv < L < 2*kv`, and a *lower*
constant trigger would reach them earlier, at the price of firing on calls that end
before `kv` and paying the differential restore. So there is a one-parameter,
prediction-free family — fire at a fixed `T` for every call — and `deadline_only` is
one member of it. Is it the best member?

This matters beyond our own baseline. Continuum reports no fixed-TTL baseline at all,
and ThunderAgent's ablation reports only that a fixed-threshold variant achieves "a
significant portion" without saying which threshold or how much. If a tuned constant
captures a material share of the budget, then every predictive result in this area is
being compared against an unnecessarily weak reference.

**Arms.** `deadline_only` at `T = kv`, against `best_constant`, where `T` is a single
global scalar **fitted on the profile folds** by maximising net utility and applied
unchanged to the evaluation fold. Fitting one scalar out-of-fold makes this a
genuinely deployable prediction-free policy, not an oracle. An analysis-only
`ORACLE_constant`, the best `T` fitted in-sample on the evaluation rows, is reported
alongside to expose any overfit, exactly as `f1cbee3` required.

**Metric.** Net utility in `s/277`, and the gain over `deadline_only` as a fraction of
the per-call ORACLE budget, at `kv = 3500` and `5000`.

**Interpretation fixed in advance.** If `best_constant` beats `deadline_only` by more
than 10% of the budget in both cells, the standard baseline is suboptimal and every
prediction-versus-deadline comparison in this lane and in the cited literature is
understated in the predictor's favour. If it gains under 3% in both, the deadline is
confirmed near-optimal among constants and the negative results stand as reported.
Between 3% and 10% is inconclusive and is not resolved by moving the band.

**Premise, binding.** Forced eviction; with no memory pressure every arm including
every oracle scores zero. Fresh-277 cannot exhibit contention. Hidden-swap
milliseconds, not wall clock. The label threshold separating long from short stays at
`kv`, since that is the physical condition for a call to be able to hide the swap;
only the trigger varies.

**Causal contract.** `best_constant` reads profile-fold rows only; the five
task-grouped folds partition task ids. `ORACLE_constant` is labelled analysis-only.

## Open question — are the negative results an artifact of the restore charge?

**Criterion frozen 2026-07-30, before the numbers exist.**

Visible when frozen: the full negative chain — budget `57cbbbb`, per-key ceiling
`3bc4906`, key ladder `8e7ed28`, leave-one-out refutation `f1cbee3`, and
best-constant `24d4344`. Every one of those charged a flat `restore_cost_fraction =
0.94` on short-call misfires. `24d4344` recorded the mechanism explicitly: `T = kv`
is optimal *because* the restore penalty at `rho = 0.94` is close to a full swap and
swamps everything gained in the `kv..2kv` band. It also recorded, untested, that a
smaller restore fraction would make a sub-`kv` trigger profitable.

**Why the charge is probably too high.**
`analysis/serving/tool-time-prefill-cost-20260716/findings.md` states the design
plainly: the restore action takes "the cheaper of reload (`rho*kv` over PCIe) or
recompute (prefill the context)". This lane never implemented the `min`. It always
charged reload. So every negative result to date sits under the most pessimistic
restore assumption available, and the true effective fraction is `<= 0.94`.

**Question.** Sweep `rho` and locate the value at which each negative conclusion
flips. Two policy classes are re-run at each `rho`: `best_constant`, one scalar
fitted out-of-fold, and the per-key leave-one-out ceiling for `cmd_depth_4`, both
against `deadline_only` recomputed at the same `rho`.

**Sweep.** `rho` in `{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.94}`, at
`kv = 3500` and `5000`. `rho = 0` is the most generous assumption physically
possible — a free restore — and is included deliberately as the strongest test the
data can support.

**Interpretation fixed in advance.** If neither class beats `deadline_only` at any
`rho` including `0`, the negative result is **unconditional in the restore charge**,
which is a materially stronger claim than anything committed so far. If a class
flips at some `rho* <= 0.94`, the negative result is **contingent**, and every
committed conclusion in this lane must be restated as holding only for
swap-back-expensive regimes. Report `rho*` per class and cell. There is no
pass/fail band here; the sweep's output is the boundary itself, and it must be
reported wherever it falls.

**Premise, binding.** Forced eviction; with no memory pressure every arm scores
zero. Fresh-277 cannot exhibit contention. Hidden-swap milliseconds, not wall clock.
Note that `rho` and `kv` are swept parameters of the cost model here, not derived
from any specific context length in this corpus.

**Causal contract.** `best_constant` and the per-key trigger read only rows outside
the scored row's fit, as in `24d4344` and `f1cbee3` respectively. The five
task-grouped folds partition task ids.

## Open question — what is the effective restore fraction, and is 0.94 right?

**Criterion frozen 2026-07-30, before the derivation is computed.**

Visible when frozen: `2316693`, which showed the per-key negative result flips below
`rho ~ 0.3` (kv=3500) and `rho ~ 0.6` (kv=5000), and recorded that the decisive
follow-up — computing `rho_effective = min(reload, recompute)/kv` — was blocked on
the `kv_cost_ms`-to-token mapping. Also visible, found while framing this: the
mapping exists. `interpolate_cost_grid` in `scripts/serving/measure_kv_swap_cost.py`
interpolates target swap-out cost onto measured `(tokens, swap_out_ms)` points and
refuses to extrapolate, and `rho_bf16kv_large.json` records `target_swap_out_ms=500`
mapping to `275587` tokens.

**Derivation to compute**, from two already-measured artifacts, with no new
experiment on the corpus:

1. `tokens(kv)` from the measured swap-out curve.
2. `reload_ms(kv) = rho_measured * kv`, with `rho_measured = 0.94`.
3. `recompute_ms(kv) = prefill(tokens(kv))` from
   `analysis/serving/tool-time-prefill-cost-20260716/`, floor `74.80 ms` plus
   `0.04861 ms/token`.
4. `rho_effective(kv) = min(reload_ms, recompute_ms) / kv`.

Then compare `rho_effective` against the flip boundaries from `2316693`.

**What the answer decides.** If `rho_effective` is materially below the flip boundary
at the headline cells, this lane's negative results are an artifact of an overcharged
restore and must be withdrawn. If `rho_effective` equals `0.94` there, the negative
results are confirmed and `2316693`'s contingency, while numerically valid, describes
a regime this cost grid does not represent. Report the crossover KV size at which
recompute would become cheaper than reload, and state whether any realistic
configuration reaches it.

**Premise.** Both inputs are single-measurement hardware constants on one model and
one device: Qwen3-Coder-30B-A3B-Instruct-FP8 on H100 PCIe. The derivation inherits
that scope and is not a claim about other models, KV layouts, or interconnects. The
prefill curve was measured to 32,768 tokens, so evaluating it at millions of tokens
is an extrapolation and must be labelled as one; note that attention cost grows
faster than linearly, so a linear extrapolation *understates* recompute cost and is
therefore conservative in the direction of favouring recompute.

## Open question — do the conclusions hold at the per-request KV scale?

**Criterion frozen 2026-07-30, before the numbers exist.**

Visible when frozen: the whole negative chain, plus `8e5fc7d` which established
`tokens(kv) = 551 * kv` from the measured swap-out curve. Applying that mapping to
this corpus's contexts produces a scope fact nobody has recorded: a single request at
the measured p50 final context of 30,900 tokens costs about **56 ms** to swap out, and
at the max 88,200 tokens about **160 ms**. The campaign grid starts at **500 ms**,
which is nine requests' worth at p50, and its headline cells at 3500 and 5000 ms are
**62 and 89 requests' worth**. The grid therefore never includes the per-request
scale, and 3500/5000 also sit outside the measured swap range of 238–1071 ms.

**Question.** Every conclusion in this lane — the budget, the narrow band, the
near-optimal deadline, the negative per-key result — was measured at 3500 and 5000 ms.
Do they hold at the per-request scale a system would actually face when evicting one
request's cache?

**Cells.** `kv` in `{56, 100, 160}` ms, chosen from the corpus's own context
distribution: p50 context, a round intermediate, and max context. Reported alongside
the existing 3500 and 5000 for continuity.

**Arms**, unchanged so results are comparable: `deadline_only`, `best_constant` fitted
out-of-fold, per-key `cmd_depth_4` leave-one-out, and the per-call `ORACLE`. All at
`rho = 0.94`, which `8e5fc7d` established is scale-invariant and therefore still
correct at these cells.

**Interpretation fixed in advance.** For each new cell report the budget as a fraction
of `deadline_only` utility, and whether either deployable arm beats `deadline_only`.
If the per-key arm beats the deadline at any per-request cell, the lane's negative
result is **scale-contingent** and must be restated as applying only to batch-scale
eviction. If it loses at every cell, the negative result is confirmed across two
orders of magnitude of KV cost, which is materially stronger than the present claim.

**Premise, binding.** Forced eviction; with no memory pressure every arm scores zero.
Fresh-277 cannot exhibit contention. Hidden-swap milliseconds, not wall clock. The
per-request `kv` values are derived from a measured bandwidth and a measured context
distribution, but the pairing of the two is mine and is not itself a measured
quantity.

## Open question — does the KV-lane negative replicate on a second corpus?

**Criterion frozen 2026-07-30, before the numbers exist.**

Visible when frozen: the entire closed KV lane, consolidated in
`analysis/development/research-directions-20260729.md` at `03a55fd`. Every result in
it — the budget, the narrow band, the near-optimal deadline, the negative per-key
outcome, the scale collapse — was computed on **Fresh-277 alone**, via
`configs/corpora/swe-277.json`. A second declared corpus exists,
`configs/corpora/swe-100.json` (100 tasks, same scaffold and model, trace root
`offline-gated-confirm-100-v2`), and has never been used in this lane.

**Why run it.** Replication caught a defective statistic once already this run:
`c211b37` established that the duration-weighted envelope figure replicated across
cohorts to within 0.7 pp while the median peak/min ratio moved from 36x to 13x, which
is why the median was dropped as a headline. A single-corpus negative result is weaker
than it needs to be when a second corpus is sitting unused.

**Arms and cells**, identical to `8968434` so the two corpora are directly comparable:
`deadline_only`, `best_constant` fitted out-of-fold, per-key `cmd_depth_4`
leave-one-out, and the per-call `ORACLE`, at `kv` in `{56, 100, 160, 3500, 5000}` ms,
`rho = 0.94`, `guard = 0`, five task-grouped folds.

**Interpretation fixed in advance.** The claim under test is qualitative and has two
parts. First, that no deployable arm reliably beats `deadline_only` at any cell.
Second, that the budget as a fraction of deadline utility is small at per-request
scale (56–160 ms) and large at the campaign cells (3500–5000 ms). If both hold on
SWE100, the negative result is **corpus-robust across traces** and should be stated
that way. If either fails, the negative result is **corpus-specific** and every
committed conclusion in the lane must be qualified to Fresh-277.

**What replication here does and does not buy.** Both corpora share benchmark,
scaffold and model, so this is replication **across traces, not across workloads**. It
cannot establish that the conclusion holds for non-SWE agents, and the write-up must
say so.

**Premise, binding.** Forced eviction; with no memory pressure every arm scores zero.
Neither corpus can exhibit contention. Hidden-swap milliseconds, not wall clock. The
per-request `kv` values were derived from Fresh-277's context distribution and are
reused unchanged here rather than re-derived, so they are held fixed across corpora by
construction.

## Explicit non-claims

- No live W5 or headline systems result exists.
- No contention conclusion comes from Fresh-277; collection used a cloud model
  and had no shared KV cache.
- No per-call, same-repository, same-trace, wrapper-normalized, atom/segment, or
  boundary-conditioned policy is an active direction.
- The removed online-first replay is protocol-invalid for deployment and
  contributes no positive or negative paper result.
