# Direction portfolio — 2026-07-29

Deliberately **not** converged to one winner. The human directed that genuinely
mixed evidence should leave multiple directions open, each carrying its evidence,
uncertainties, dependencies and next discriminating experiment.

Rubric and candidate definitions: `direction-candidates-20260729.md`.
Scores below are my own judgement. Three parallel evaluators were spawned; two
returned nothing usable (repeated idle-without-delivery), so the novelty searches
that decide D1 were run directly and are cited inline.

## Status at a glance

| # | Direction | Status | Blocking dependency |
|---|---|---|---|
| D1 | Clause-granular prediction-free resource control | **OPEN, narrowed** | Cap-vs-reservation framing; needs a capacity number |
| D2 | Predictability boundary as a limits result | **OPEN** | Metric presupposes forced eviction |
| D3 | Heterogeneous multi-tenant composition | **PARKED** | Needs contention; corpus cannot exhibit it |
| D4 | Mechanism-vs-prediction ablation harness | **OPEN** | Cost of faithful reimplementation |
| D5 | Clause-attribution instrument | **FOLDED INTO D1** | Not standalone |

---

## D1 — Clause-granular prediction-free resource control

**Evidence in hand.** Multi-clause exec calls are 48.4% (SWE277) and 44.3%
(SWE100). A single call-level limit wastes **48.8% / 48.1%** of the RSS
resource-time envelope — replicated across cohorts to within 0.7 pp (`c211b37`).
The dominant pattern is named and quantified: a heavy command piped to a blocked
reader, 24.2% of multi-clause calls, `python3|tail` 352, `pip3|tail` 251,
`apt-get|tail` 167, `pytest|tail` 48, holding **6,512 GB-seconds** of memory-time
(`2886683`). The 29.4% label-coverage gap is resolved, not merely bounded: long
null-RSS clauses are blocked readers with nothing to sample, which makes the
envelope figure conservative.

**Novelty, narrowed by direct search — this is the important update.**
Dynamic cgroup resizing on phase transitions is **prior art**: Kubernetes
in-place pod resize updates cgroup parameters live, and patents cover
[dynamic container resource tuning and resizing](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12008411)
including classifying usage into phases to select limits. eBPF exec tracking is
also prior art, but for security and monitoring
([Falco](https://medium.com/@rameshavutu/kubernetes-threat-detection-with-falco-and-ebpf-b595cffd40aa),
[eBPF-PATROL](https://arxiv.org/pdf/2511.18155),
[eHashPipe](https://arxiv.org/pdf/2509.09879)).
No directly overlapping work retrieved for **using exec/clause boundaries as the
trigger for resource-limit adjustment**.

So the mechanism is assembled from known parts. The delta that survives is:
*exact structural phase boundaries are available for free from the command's parse
tree plus exec events, where every prior approach must detect phases statistically
from observed usage.* Plus the workload finding, which is genuinely agent-specific
because the `| tail` idiom is created by LLM context limits, not by computation.

**A paper claiming "we resize cgroups dynamically" is dead on arrival.** A paper
claiming "agent tool calls contain a measurable, context-limit-induced waste
pattern, and their phase boundaries are exactly knowable rather than inferred" is
not.

**Uncertainties.** (a) `memory.max` is a cap, not a reservation, so the waste
materialises only under a reserving admission controller — is that a real vendor
decision or a framing convenience? (b) Is downward cgroup resizing safe mid-call,
or does setting `memory.max` below current usage trigger reclaim or OOM? This is
a potential correctness killer and is unresolved.

**Next discriminating experiment.** Convert the envelope into a **capacity**
number: max concurrent sandboxes under a fixed memory budget, call-granular vs
clause-granular reservation, from real per-clause envelopes. Offline, no testbed.
Directly answers the likeliest rejection.

**Kill it if.** Downward resizing proves unsafe in cgroup v2, or the capacity gain
under realistic budgets is in the low single-digit percent.

---

## D2 — Predictability boundary as a limits result

**Evidence in hand.** `apt-get update` P(long)=95.7% CV 0.40 vs `python3 -m pytest`
P(long)=12% CV 1.79, with pytest the largest long-call group. The entire priced-
stopping advantage over a fixed deadline is carried by container-setup commands and
goes **negative** without them (`04ef2bd`). Four committed NO-GOs. ThunderAgent's own
figure corroborates the cliff (2.43–3.58× predictable, 0.65× and 1.24× stochastic)
without quantifying it; Continuum reports no fixed-TTL and no oracle baseline.

**Uncertainties.** The frozen bar `rho/(1+rho)=48.5%` was found **mis-specified**:
firing at `t` hides only `min(K, latency-t)`, so a 69%-pure subgroup was still
net-negative (`7ea7ea7`). A corrected closed-form bar from the remaining-time
distribution is not yet derived, and whether it is a contribution or bookkeeping is
open.

**The central threat.** Its numbers come from a functional conditional on forced
eviction; with no pressure the optimal policy is never swap and every arm scores
zero. **Possible resolution found tonight:** `2886683` is measured occupancy, not a
swap utility, so it does not depend on that premise. D2 may have a premise-free core
if rebuilt on occupancy rather than hidden-swap-ms.

**Next discriminating experiment.** Derive the corrected bar in closed form and test
whether it predicts, ex ante, which of our four NO-GOs would fail. A bar that
retrodicts its own refutations is a result; one that does not is bookkeeping.

**Kill it if.** The corrected bar cannot be written without workload-specific
constants, or it fails to retrodict the committed NO-GOs.

---

## D3 — Heterogeneous multi-tenant composition — PARKED

Requires contention. `CLOSED-QUESTIONS.md` establishes Fresh-277 cannot exhibit it,
and we hold traces for one agent type only. No honest offline path identified.
Revisit only if a testbed exists. Parked, not killed: the underlying observation —
that Continuum's global `eta` and ThunderAgent's global queue average over
incomparable programs — remains true and unevaluated by anyone.

---

## D4 — Mechanism-vs-prediction ablation harness

**Evidence in hand.** Demonstrated once already: `04ef2bd` decomposed a published-
style gain into structure vs prediction and found the prediction share negative
outside container setup. The methodology — frozen criteria, oracle bounds, task
settlement barriers — is built and exercised.

**Uncertainty.** Faithful reimplementation cost. Continuum's rule is verified and
implementable, but its benefit term is dominated by `T_bar`, the queueing delay,
which is unmeasurable offline — so a fair comparison may be impossible without the
live harness.

**Next discriminating experiment.** Attempt the Continuum arm with `T_bar=0` (the
correct value for a contention-free corpus) and state plainly which half of their
mechanism that tests and which it cannot.

**Kill it if.** The `T_bar=0` restriction makes the comparison uninformative about
the published system.

---

## D5 — Instrument contribution — FOLDED INTO D1

Not standalone: Crab already publishes an eBPF inspector for sandboxes and
AgentCgroup an eBPF controller. The clause-attribution telemetry is better framed as
D1's enabling instrument than as its own contribution.

---

## Cross-cutting constraints

- No further LLM API spend; Fresh-277 and the replay corpora are what exist.
- Fresh-277 provably cannot exhibit contention.
- The swap utility functional is conditional on forced eviction; its absolute
  seconds are hidden-swap-ms, not wall clock.
- Task settlement barrier on any causal evaluation; omitting it once inflated a
  result 18×.
- Replication caveat: both cohorts share benchmark, scaffold and model, so
  cross-trace replication is not cross-workload validation.
