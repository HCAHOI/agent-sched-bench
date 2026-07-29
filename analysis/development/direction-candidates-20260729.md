# Candidate directions and evaluation rubric — 2026-07-29

Five candidates, evaluated on one rubric before committing compute. Written
before the parallel evaluation so the criteria are not fitted to the answers.

## Shared rubric

Each candidate is scored 1-5 on six axes, plus a single fatal risk.

| Axis | Question |
|---|---|
| **R1 Novelty delta** | What axis differs from the closest published work — object, mechanism, granularity, or setting? Granularity alone is the weakest. |
| **R2 Evidence in hand** | How much of the required evidence already exists in this repo, measured and committed? |
| **R3 Validatable offline** | Can the central claim be established without contention, a GPU, or new API spend? Anything needing memory pressure scores <=2, since CLOSED-QUESTIONS.md proves Fresh-277 cannot exhibit it. |
| **R4 Prediction independence** | Does the mechanism survive the fact that tool duration and resource demand are unpredictable? Four committed NO-GOs and AgentCgroup independently establish this. |
| **R5 Reviewer resilience** | Does it survive the likeliest one-line rejection? |
| **R6 Effort to first result** | Weeks to a defensible number. Lower is better. |

## Candidates

### D1 — Clause-granular prediction-free resource control

**Hypothesis.** A tool call is not the atomic unit of resource behaviour; resizing
cgroups at shell-clause boundaries detected reactively via eBPF recovers headroom
that tool-call-granular control cannot.

**Novelty.** Granularity + reactive-vs-declarative, against AgentCgroup
(arXiv 2602.09345), which aligns cgroups to tool-call boundaries and proposes
agent *declaration* as its remedy.

**Impact.** Sandbox density for agent cloud providers.

**Evidence needed.** Already have: 48.4% of exec calls are multi-clause; within-call
peak/min RSS 36x median, 586x p90; one call-level limit wastes 48.8% of the RSS
resource-time envelope. Still needed: a capacity number (max concurrent sandboxes
under a memory budget) and a prototype.

**Fatal risk.** `memory.max` is a cap, not a reservation, so the 48.8% is
hypothetical over-provisioning that only materialises under a reserving admission
controller; and the gain needs a co-tenant this corpus cannot provide.

### D2 — The predictability boundary as a limits result

**Hypothesis.** There is a quantifiable condition under which duration-prediction
mechanisms can pay, agent workloads mostly sit on the wrong side of it, and
published gains concentrate on the narrow side.

**Novelty.** Setting + method. ThunderAgent's own figure shows the cliff
(2.43-3.58x predictable versus 0.65x and 1.24x stochastic) without quantifying it;
Continuum reports no fixed-TTL and no oracle baseline.

**Impact.** Redirects effort; explains three systems' results with one number.

**Evidence needed.** Mostly in hand: `apt-get` P(long)=95.7% CV 0.40 versus `pytest`
12% CV 1.79; the container-artifact attribution; four NO-GOs; the corrected bar
(the naive `rho/(1+rho)` is necessary but not sufficient because firing at `t`
hides only `min(K, latency-t)`).

**Fatal risk.** It is a limits/measurement paper, not a systems-gain paper, and its
absolute numbers are conditional on a forced-eviction premise the corpus cannot
exhibit.

### D3 — Heterogeneous multi-tenant composition

**Hypothesis.** Globally fitted parameters — Continuum's `eta`, ThunderAgent's
global queue — degrade when coding, science and browsing agents share infrastructure,
which is the real deployment condition none of the seven papers evaluates.

**Novelty.** Setting.

**Fatal risk.** Requires contention, so it is not offline-validatable here, and we
hold traces for one agent type only.

### D4 — Mechanism-versus-prediction ablation harness

**Hypothesis.** Published agent-serving gains are not decomposed into structural
mechanism versus prediction quality, and when decomposed the prediction share is
small.

**Novelty.** Method/benchmark. Demonstrated once already: the entire measured
advantage of priced stopping over a fixed deadline is carried by container-setup
commands, and goes negative without them.

**Fatal risk.** Reproducing other systems faithfully is expensive, and a benchmark
paper without a system may read as thin at a top venue.

### D5 — Clause-attribution telemetry as an instrument contribution

**Hypothesis.** Causal per-clause attribution of latency/CPU/RSS/disk inside agent
tool calls is a reusable measurement substrate nobody has published.

**Fatal risk.** Crab already publishes an eBPF inspector for sandboxes and
AgentCgroup an eBPF controller; and our own sampler misses 29.4% of clause
resource-time, so the instrument is not yet sound enough to be the contribution.

## Known constraints binding every candidate

- No further LLM API spend; Fresh-277 and the replay corpus are what exist.
- Fresh-277 provably cannot exhibit contention (CLOSED-QUESTIONS.md).
- The utility functional is conditional on forced eviction; absolute seconds in
  that lane are hidden-swap-ms, not wall clock.
- Enforce the task settlement barrier on any causal evaluation; omitting it once
  inflated a result 18x.
