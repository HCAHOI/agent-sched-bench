# Design memo — atom-structured duration estimation, post-mortem edition

> Informed by the five-model atom study (segment-atom-study-2026-07-19).
> atom_trie (model #5) final outcome: beats cert-config trie on MAE (974 vs
> 996), loses narrowly to cdskip (971); heavy verbs stuck at verb-level
> fallback (pytest 88%, find 98%) — i.e., the "atom_trie loses with shallow
> fallback on heavy verbs" branch of this memo's branch-analysis is the one
> that materialized (B's screen expectation: select little; C unaffected).

**To:** user + advisor · **From:** estimator-design lane · **Date:** 2026-07-19
**Ground rules honored:** rho=0.94 fixed; existing corpora only
(8,953-segment replay corpus, fresh-277 decisions, TraceLab for external
validity); everything routes through the cross-fitted permutation-
certification machinery; the seam contract is "a sample-set/survival
representation per call + optional mid-call re-check time"
(`build_latency_prior` → ECDF nodes, `hazard_recheck_ms` for the re-check —
both in `src/trace_collect/tool_latency_profiled.py`, consumed by
`src/trace_collect/tool_latency_utility_clock.py`).

## Framing: what the atom study actually licenses

The advisor doc closed the question "atoms as *fit-time units*" —
decomposition loses because duration is a property of (verb, arguments, repo
state), and 46% of calls have no additive structure at all. But it left
three doors open, and each candidate below walks through exactly one:

- **Door 1 — state, not structure** (constraint 3): the trie conditions on
  the *within-call* prefix; nothing conditions on *cross-call* session state
  (did `apt-get update` already run this task? is pip's cache warm?).
- **Door 2 — stability certifies, not decomposition** (constraint 5):
  apt-get (CV 0.38) is the only heavy-stable atom, and it carried the
  certified H1 mass. If stability is what certifies, screen for it directly
  and make it a gate, not a model.
- **Door 3 — atoms as evidence, not units** (advisor doc §Secondary-3): at
  runtime, "atom k just completed at elapsed t" is an observation, not a
  prediction target. Composition never has to happen.

Hierarchical chain↔atom shrinkage and critical-path pipe composition were
**considered and demoted** (rationale at the end): the first collapses to
the already-queued min-support fallback (an engineering fix, not a
contribution), the second requires per-member end-times that the
concurrent-execution corpus structurally cannot supply.

---

## Candidate A — Session-state-conditioned nodes ("the task remembers")

**Mechanism.** Keep the chain-prefix trie exactly as certified. Add a
second, disjoint conditioning axis: each node key is extended with a small
*predecessor-state summary* computed from the task's earlier calls — a hash
of {set of heavy verbs already completed this task (apt-get-update-done,
pip-install-done, pytest-run-count bucket), cwd-changed-since-last-call
bit}. All fields are available at decision time from the task's own history
(no oracle; same information the online-history path already legally uses in
`_estimate_survival`). Fit: same ECDF sample-set per (prefix, state-hash)
node; hierarchy fallback drops the state hash first, then trie depth — so it
degrades to the exact current model.

**Constraints 1–4:** (1) directly targets it — `python3` CV 5.5 is
*cross-task, unconditional*; the hypothesis is that CV conditional on
session state drops. (2) worsens it — state-hashing fragments support
further; survivable only because fallback recovers the current model, and
the state vocabulary is fixed and tiny (≤ ~8 bits), not open like arguments.
(3) this IS constraint 3, made into the key. (4) unaddressed — pipes get
state-conditioning of the whole call, no decomposition needed, so it covers
pipes too (a real advantage over anything atom-shaped).

**Distribution representation.** Identical to today: a sample set at the
selected node → ECDF survival + `hazard_recheck_ms`. Zero seam changes.

**Gate/union story.** Certifies where repeated-verb calls dominate and state
flips the distribution: second `apt-get install` after an update (cache
warm), `pip install` re-runs, repeated `pytest`. These are exactly calls
where the *prefix* is identical across warm/cold, so the chain trie is
blind — disjoint by construction, the union-gain condition.

**Cost.** ~4–6 days. New: `src/trace_collect/tool_latency_session_state.py`
(state-summary extraction + keying, ~200 lines) reusing
`make_row_command_prefix_keys` plumbing; extend `latency_prior_hierarchy`
fallback order; one test file. No simulator changes.

**Falsification on existing data.** Two stages, both on the 8,953-segment
corpus + fresh-277 decision replay. Stage 1 (one day, cheap kill): recompute
the stability table conditional on state-hash — paired per-task CV of
`python3`/`pip`/`pytest`/`git` with vs without conditioning, task-grouped
folds. **Kill: if conditional CV of the heavy verbs does not drop by a
margin that survives the paired bootstrap (i.e., state explains no
variance), stop — do not build stage 2.** Stage 2: full decision replay at
rho=0.94, paired vs `chain_prefix_cdskip`, permutation certificate per kv
cell. Kill: no cell certifies where the policies diverge.

**Novelty.** [Continuum](https://arxiv.org/abs/2511.02230) maintains
global/per-tool online means with Bernstein upper bounds — no session-state
conditioning of any kind; ThunderAgent is memoryless decay. Learned cost
models in databases condition on buffer/cache state, but no occupant found
for *agent-session-state-conditioned tool-duration priors*, and none with a
certification gate. Surviving delta: modest but real — "the same command is
a different random variable depending on what the task already did,"
demonstrated and certified. Risk: a reviewer calls it feature engineering;
the defense is the CV-table mechanism, not the MAE.

---

## Candidate B — Stable-atom extraction gate ("certify the apt-get class, not the decomposition")

**Mechanism.** A mixed-unit estimator: at fit time, run a cross-fitted
**stability screen** over atoms — an atom qualifies iff (heavy: median above
the action-relevance floor) ∧ (cross-task CV below a cross-fitted cap) ∧
(min task support), all thresholds chosen on fit folds only (no dataset
constants; apt-get must *emerge*, never be named). A qualifying atom gets
one pooled survival node built from **every chain containing it, across all
chain shapes** — pooling across contexts is what defeats support
fragmentation, and it is legitimate *only* for atoms the screen has
certified as context-insensitive (that's what low cross-task CV means). At
decision time: if the call's command contains a stable atom, predict from a
shifted atom node (atom samples + the call's observed non-atom residual
median); else chain trie, unchanged. The stable-atom trigger enters the
certified union as one more gate, certified or discarded per workload like
every other member.

**Constraints 1–4:** (1) answered by refusing to model unstable verbs at
atom level at all — the screen is the answer. (2) answered by cross-context
pooling: `apt-get install` has 16-task support at atom level vs fragmenting
across `cd X && apt-get...` chain variants. (3) partially — state-sensitive
atoms fail the CV screen and are excluded, so the estimator never lies about
them; it does not *exploit* state (that's Candidate A's job — A and B are
composable). (4) genuinely covered: "call contains stable atom" is
well-defined for pipes and loops too; the atom node predicts the *call*
distribution conditional on containment, no additive composition invoked.
This is the only atom-flavored design with any claim on the 46%.

**Distribution representation.** A sample set (pooled atom-context
durations) → same ECDF/hazard path. Seam-native.

**Gate/union story.** The known risk is **overlap**: apt-get *prefixes*
already carry H1. B's union gain must come from stable atoms in chain
contexts the trie has never seen or holds thinly — precisely the H2 bleed
cells (`apt-get update -qq &&` −19.2 s at 4 calls; `apt-get install -y`
−3.5 s). The atom node sees through the prefix variation that starves those
nodes. Diagnostic before any replay: count fresh-277 decisions where the
trie is at fallback/thin-support but a stable-atom node would fire — if that
count is small, there is no union gain and B dies on arithmetic before
costing a replay.

**Cost.** ~3–4 days. New: `src/trace_collect/tool_latency_stable_atom.py`
(screen + pooled nodes) + gate registration in the union path; atom
extraction reuses the xtrace/segment parsing already built for the study
(`scripts/analyze_segment_variance.py` lineage); one test file.

**Falsification.** On the segment corpus + fresh-277: (i) the overlap
diagnostic above; (ii) cross-fitted screen — record *which* atoms qualify
per fold (if selection is unstable across folds, kill: the screen doesn't
generalize); (iii) decision replay, paired delta of union-with-B vs
union-without-B, permutation-certified. **Kill: overlap diagnostic shows
<~5% of divergent decisions land where the trie is thin, or the incremental
union delta fails certification on all kv cells.** Secondary external check,
free: TraceLab replay — does any stable-heavy atom class emerge outside
SWE-style corpora? If the screen finds *nothing but apt-get anywhere*, B is
a one-atom trick and gets reported as such (an honest negative worth one
paragraph, not a contribution).

**Novelty.** Nearest prior: nothing found doing per-program-unit *stability
screening* as a certification criterion for latency priors; hierarchical
microservice latency predictors
([GRAF](https://ina.kaist.ac.kr/assets/bibliography/GRAF_ton.pdf),
[unified-representation predictors](https://arxiv.org/html/2508.01635))
learn graph embeddings, no per-unit certificates. Surviving delta: "certify
the *unit of conditioning* itself, per workload" — a clean instantiation of
the paper's conditioning-spectrum thesis (the spectrum now has a data-chosen
operating point per command class). Weakness: if only apt-get ever
qualifies, the delta is an anecdote.

---

## Candidate C — Atom-boundary survival re-conditioning ("atoms as evidence") — **recommended**

**Mechanism.** No new fit-time unit. The chain-prefix node keeps its
certified survival curve, but the curve is now *indexed by observed
progress*: from the fit-fold segment timelines, each node stores its samples
as (total duration, boundary-event sequence [(atom₁, t₁), (atom₂, t₂), …]).
At runtime, the shell wrapper (already built — bash xtrace instrumentation
from the atom study, `src/trace_collect/CLAUDE.md`) emits "segment k
completed at elapsed t". At each event, re-select the conditional sample
subset {fit samples whose k-th boundary exists, reweighted by proximity of
their tₖ} and re-run the *existing* `hazard_recheck_ms` on the conditional
residuals. The current policy already re-checks at one precomputed k; C
replaces the fixed re-check clock with an event-driven one carrying strictly
more information. Elapsed time alone is already implicit conditioning
(survival of the residual); C's falsifiable claim is that **which atom
finished, and when, adds beyond elapsed time alone**. The stability table
says it must for the calls that matter: at elapsed t = 3 s in
`apt-get update && apt-get install -y ...`, "update still running" vs
"update done, install started" are wildly different remaining-mass regimes,
and the boundary is the observable that separates them.

**Constraints 1–4, one by one.** (1) Verb instability is irrelevant —
nothing predicts an atom's duration; the atom's *completion* is consumed as
evidence after the fact. (2) Argument thinness is irrelevant at event time —
conditioning is on (node, boundary index, elapsed), not on argument tokens;
thin nodes simply have fewer conditional samples and the existing thin-node
degradation (ties → deadline re-check, `hazard_recheck_ms` docstring)
already handles that failure mode conservatively. (3) Cross-atom state is
*captured for free*: whatever apt-get-update did to the world is baked into
the observed t₁, which is exactly the conditioning variable. (4) Pipes:
sequential boundaries don't exist, so C silently never fires there and the
call runs the unmodified certified policy — coverage is the 54%
sequential-chain mass (4,824 analysable chains), degradation is
exact-to-baseline, which is the union discipline in temporal form: *a gate
over evidence-availability rather than over command class*.

**Distribution representation.** A conditional sample set per (node,
progress-state) → the same ECDF survival + exact piecewise-linear re-check
optimizer. The seam's contract ("any predictor yielding a distribution
representation per call") is met per *re-check event*, which the utility
functional already prices.

**Gate/union story.** This is the only candidate that can win *on calls the
trie already owns*: H1's certified mass sits on apt-get prefixes where the
fixed policy still eats residual stall or fires early on the short tail;
boundary evidence moves the trigger within those same calls. It also
directly patches H2's thin-prefix bleed (`apt-get update -qq &&`): the
fit-time node is starved at 4 calls, but at runtime the boundary observation
disambiguates regardless of prefix-spelling fragmentation — evidence
substitutes for support. Certification: C vs frozen-baseline is a paired
per-call comparison under the identical utility functional, task-clustered,
permutation-gated per kv cell — it enters as a policy iteration, certified
or reverted, exactly like cd-skip did.

**Cost.** ~5–7 days, zero GPU. New:
`src/trace_collect/tool_latency_boundary_survival.py` (conditional-subset
selection + event-driven re-check schedule, ~300 lines, reusing
`hazard_recheck_ms` verbatim on conditional samples); simulator replay
extension to feed segment-timeline events (`simulate_utils.py`); tests. The
instrumentation and the 8,953 timelines already exist — the expensive part
of this idea was paid for by the atom study.

**Falsification on existing data.** All on the replay corpus (both arms
replayed on `our_hardware` durations for self-consistency — mixing original
call durations with replayed boundary fractions would be a silent
hardware-transfer assumption; we refuse it and state so). Stage 1 (2 days,
the kill switch): **information test** — at each observed boundary, compare
log-score / decision-divergence of residual survival conditioned on
(elapsed) vs (elapsed + boundary identity/index), cross-fitted,
task-grouped. **Kill: if boundary identity adds no decision divergence
beyond elapsed time (fewer than ~1% of decisions change, or paired log-score
gain CI covers zero), C is dead — drop it, keep chain-prefix + cd-skip, and
the advisor doc's §Secondary-3 sentence gets a measured "no".** Stage 2:
full decision replay at rho=0.94, paired vs `chain_prefix_cdskip` + fixed
re-check, permutation certificate per kv cell; kill if no cell certifies.
External validity: TraceLab traces lack our xtrace telemetry, so C's
TraceLab story is coverage-frequency only (how often would boundaries have
existed) — stated as a limitation, not retrofitted (no-retrofitted-telemetry
rule).

**Novelty.** Nearest priors, verified: (a) business-process remaining-time
prediction updates estimates at *activity* completions —
[Verenich et al., TIST 2019 cross-benchmark](https://dl.acm.org/doi/10.1145/3331449),
incl. [survival-analysis variants](https://doi.org/10.3390/a13110267);
(b) workflow/batch systems adjust runtime predictions at
[milestones](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11461669)
and stage completions ([Phoebe](https://arxiv.org/pdf/2110.02313),
[NURD](https://arxiv.org/pdf/2203.08339) for online straggler signals);
(c) [Continuum](https://arxiv.org/abs/2511.02230) sets one TTL at call start
and never updates mid-call — its estimator is frozen the moment the tool
launches. Surviving delta, three-part and honest: (i) the evidence events
are *intra-call, sub-command program structure* (shell segment boundaries),
a granularity none of (a)–(c) touches; (ii) the update drives a *priced,
certified cache action* through an exact expected-cost re-check optimizer,
not a point estimate; (iii) the deployment is gated — mid-call
re-conditioning ships only where it certifies per workload. (a)+(b)
establish the *family* is known, so the paper claim must be the systems
instantiation, never "we invented online re-estimation."

---

## How the atom_trie (#5) outcome, either way, feeds this

- **atom_trie loses with deep-node usage** (argument thinness binds):
  confirms constraint 2 is the binding one → A's state-hash gets *more*
  dangerous (further fragmentation) — demand Stage-1 pass with margin; B
  strengthens (cross-context pooling is the anti-thinness move); C
  unaffected (doesn't condition on arguments).
- **atom_trie loses at shallow fallback** (atoms starve before arguments
  even matter): B's screen will select almost nothing — run B's overlap
  diagnostic first and expect to kill it cheaply; C unaffected; A unaffected
  (its axis is orthogonal to atom keying).
- **atom_trie somehow wins**: reopens fit-time atoms; fold its depth
  diagnostic into B's screen thresholds before anything else. (Assign
  near-zero prior given the advisor-doc MAE table.)

Either way, no candidate here shares atom_trie's failure mode: none of the
three uses per-atom duration as a fit-time prediction unit.

## Recommendation

**Build C first.** Reasons, in order: (1) it is the only candidate whose
premise the measured data *supports* rather than merely tolerates — the
advisor doc itself names boundary events as the one live atom direction, and
the stability table is the mechanism (a CV-0.38 atom completing is a
high-information event precisely because a CV-5.5 successor follows it);
(2) its kill switch is a 2-day computation on data already on disk; (3) it
improves the calls that carry the certified H1 mass *and* patches the H2
bleed, so it strengthens both existing results rather than opening a new
front; (4) zero GPU, so it runs as an offline lane parallel to the W1–7 P4
build without touching the critical path.

**Schedule fit:** roadmap slots atom-boundary events at W10-11
(CPU-rate-lever era). Pull the *offline replay* forward: Stage-1 kill test
by ~W3, full certified replay by W6 — comfortably inside the W8 cut. If
certified by W8, C is one results row + one ablation (event-driven re-check
off) and a candidate live trigger in W10-11 alongside pre-restore (they
share the hazard plumbing). If Stage 1 kills it or W8 arrives without
numbers: one future-work sentence, per the cut-line rule, no exceptions. B
runs only its 1-day overlap diagnostic in the same window (near-free, and
its answer sharpens the paper's conditioning-spectrum text either way); A is
deferred entirely unless C dies at Stage 1, in which case A's Stage-1 CV
test is the next cheapest falsification.

## Kill criteria, restated up front (the pre-registration seeds)

| Candidate | Cheap kill (before full replay) | Final kill |
|---|---|---|
| A state-conditioned | conditional CV of heavy verbs not reduced (paired bootstrap) | no certified kv cell vs cdskip where policies diverge |
| B stable-atom gate | <~5% of divergent decisions land on thin-trie calls; or fold-unstable atom selection | incremental union delta uncertified on all cells; or screen selects only apt-get on every corpus → demote to observation |
| C boundary evidence | boundary identity adds no decision divergence over elapsed time (<~1% decisions change / log-score CI covers 0) | no certified kv cell vs cdskip+fixed-recheck on the replay corpus |

Any kill ⇒ keep chain-prefix + cd-skip as the shipped estimator, record the
negative in `analysis/`, and the paper's estimator section stays as-is.

**Demoted candidates, for the record.** *Hierarchical chain↔atom shrinkage*:
partial pooling is textbook
([standard HBM shrinkage](https://jrnold.github.io/bayesian_notes/shrinkage-and-hierarchical-models.html))
and here it shrinks thin chain nodes toward an atom-composition prior the
study just proved is wrong for every heavy verb — shrinkage toward a biased
prior is worse than the already-queued min-support fallback to tool-name,
which is the same idea with an *unbiased* coarse target. It survives only as
the principled framing of that W1 follow-up, not as a contribution.
*Critical-path pipe composition*: pipe members run concurrently and the
xtrace wrap observes starts, not per-member ends — the corpus cannot supply
the per-member durations the model needs (the atom study excluded those
4,129 calls "by construction"); building it would require new telemetry,
violating the existing-data constraint. One future-work sentence.

Sources: [Continuum (arXiv 2511.02230)](https://arxiv.org/abs/2511.02230) ·
[Verenich et al., remaining-time cross-benchmark, ACM TIST](https://dl.acm.org/doi/10.1145/3331449) ·
[survival-analysis remaining-time PPM](https://doi.org/10.3390/a13110267) ·
[workflow milestone runtime-update patent US11461669](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11461669) ·
[Phoebe checkpoint optimizer](https://arxiv.org/pdf/2110.02313) ·
[NURD online straggler prediction](https://arxiv.org/pdf/2203.08339) ·
[GRAF microservice tail-latency](https://ina.kaist.ac.kr/assets/bibliography/GRAF_ton.pdf) ·
[unified microservice tail-latency predictors](https://arxiv.org/html/2508.01635) ·
[hierarchical shrinkage notes](https://jrnold.github.io/bayesian_notes/shrinkage-and-hierarchical-models.html)
