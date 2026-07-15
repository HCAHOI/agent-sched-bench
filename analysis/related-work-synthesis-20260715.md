# Continuum + ThunderAgent vs our campaign — synthesis (2026-07-15)

Fable-5 research synthesis of two related systems papers against our
tool-time trigger/gate/union work. Papers fetched from arXiv HTML,
load-bearing claims cross-checked twice (not read verbatim end to end —
treat quantitative claims as approximate; flags below).

- Continuum (arXiv:2511.02230, v6 2026-05-25): multi-turn agent scheduling
  with KV-cache TTL. Verification flags: abstract "8x" JCT is best-case
  (one Tensormesh testbed); body shows ~2x vs vanilla vLLM (Fig 8);
  throughput 1.10-3.22x.
- ThunderAgent (arXiv:2602.13692, v3 2026-06-30): program-aware agentic
  inference. Flags: no ablation isolating "program-awareness"; anti-
  offloading claim (Fig 7a) has no bandwidth numbers; 1.17-3.31x-over-
  Continuum is from the main serving eval. The Theorem-F.1<->randomized-
  ski-rental correspondence below is the synthesizer's framing, unverified.

## Headline

The two papers are our central finding replayed at systems scale.
**Continuum is a fitted per-tool-name predictor; ThunderAgent refuses
prediction and beats Continuum 1.17-3.31x on ITS workloads** — the
fitted-predictor-vs-prior-free flip we measured between SWE-ReBench (GBM/
trie certify and win) and Terminal-Bench (nothing certifies), published as
two competing papers. Neither has a mechanism to decide which regime a
deployment is in. That arbiter is exactly our certification gate.

## Mechanism map

- **Continuum KV-TTL = our estimand, coarser conditioning, weaker action.**
  tau* = argmax_tau P(tau,f)*(T*eta + PrefillReload) - tau, where P(tau,f)
  is the empirical duration CDF keyed ONLY by tool name f; cold-start
  back-off tool -> global -> default (K=100). It IS a utility functional
  over an empirical survival curve (our Q1 factorization, independently
  converged). BUT: (a) conditioned on tool-name only — their own Fig 5(b)
  "slowest 10% of cd = 94.1% of total delay" proves tool-name is the wrong
  unit (the cd is glued to a compound command) — a direct advertisement for
  our command-prefix conditioning; (b) dual, weaker action: pin-at-start +
  evict-at-expiry, NO proactive in-call swap-out anywhere in the paper. Our
  early trigger sits strictly upstream of their design space. eta (turn
  index) is a sequence feature on the BENEFIT side; our GBM already has call
  index + prev-same-prefix latency on the hazard side.

- **ThunderAgent "program-aware" is NOT a predictor** (honest answer to
  "is program-awareness the better-prediction lever": no). The "LLM
  Program" is a lifecycle tuple (ID, context-length c, tool-env set,
  placement, phase, status) from a program_id request field. No DAG, no
  known tool sequence, no next-tool prediction. Scheduler uses context
  length (shortest-first eviction, recompute ~ c^2), phase, and elapsed
  acting time via decay f(t)=x^(-t). Theorem F.1: under memorylessness +
  time-homogeneity the only admissible decay is exponential/geometric —
  the anti-prediction stance made formal.

- **Unifying spectrum (the paper's framing device):**
  - exponential decay (ThunderAgent) = memoryless S(t)
  - per-tool CDF TTL (Continuum) = S(t | tool-name)
  - our trie/GBM = S(t | tool, command-prefix, within-task history)
  - **our certified gate/union = the per-deployment statistical test of
    which conditioning level the data licenses.** Theorem F.1 is the null
    hypothesis our gate tests: where our predictor certifies (SWE-ReBench)
    the workload is measurably NOT memoryless given causal context; where
    nothing certifies (Terminal-Bench) their hypothesis holds locally and
    the gate correctly collapses us to the prior-free policy.

- **ThunderAgent anti-offloading (Fig 7a: "PCIe insufficient for high-
  frequency context switching") = our F2 in the wild.** Raises E5's
  priority; does NOT refute us — their failure is INDISCRIMINATE swapping,
  ours fires only on duration-certified long calls with restore charged at
  measured rho=0.94. "Selective certified offloading survives where
  indiscriminate offloading collapses" is a claim our harness supports and
  their figure motivates. Bonus: they show Continuum at ~90% cache hit yet
  LOSING throughput because idle pinned memory dominates — the systems
  version of our F1/E1 (an objective not charging full action cost
  manufactures success).

## What they have / we have

They have, we don't: load-dependent benefit (online queueing delay T;
thrashing inequality); end-to-end online eval in real serving stacks with
contention at multiple model scales (their Fig 7a is affirmative evidence
that action perturbs latency — our F2); cross-resource actions (sandbox GC,
async env-prep); an impossibility theorem; reload-vs-recompute choice.

We have, they don't: statistical CERTIFICATION (neither paper certifies —
Continuum trusts the CDF, ThunderAgent trusts memorylessness); the
no-universal-policy finding with evidence (3 corpora + their head-to-head
as a 4th); finer causal conditioning (command-prefix vs tool-name, their
Fig 5 as the exhibit); measured rho=0.94 + honest restore accounting;
proactive in-call swap-out (an action neither takes).

## Ranked proposals (all fit our seam + bootstrap + existing corpora)

- **P1 tool-name-only baseline (highest EV, ~zero cost):** restrict the
  trie to tool-identity nodes (depth-0 back-off already exists), fit + gate
  identically -> reproduces Continuum's estimator class inside our harness.
  Contrast: gated tool-name vs full trie vs GBM vs deadline, rho=1.0, all
  corpora. Positions us vs published SOTA; their cd figure predicts
  tool-name fails exactly where all our divergence lives (exec).
- **P2 second action evict+recompute (first real M(t,S|P,A) step):** A2 =
  evict-now, out-cost ~0, restore = prefill(context length); measure
  prefill-vs-context on the H100 (extend measure_kv_swap_cost.py). A2's
  trigger is a new functional over the SAME amortized survival curve — zero
  new learning (Q5 factorization), second action from the literature.
  Contrast: swap vs two-action selector argmax(swap,evict,keep) vs
  deadline; certify the increment. Prereq: verify per-call context length
  is in our trace metadata.
- **P3 exponential-threshold prior-free arm (cheap, theory-grounded):** our
  deterministic deadline = 2-competitive ski-rental; ThunderAgent's
  exponential family ~ randomized ski-rental (hedge: synthesizer framing).
  Deploy an exponential-threshold deadline as the policy the gate uses when
  NO predictor certifies. Connects the certified union to learning-
  augmented ski-rental (Q6).
- **P4 instrument the fresh corpus with load telemetry (DECIDE BEFORE
  collection):** cannot honestly retrofit load-dependent benefit onto
  frozen traces (no queue state; synthesizing violates no-synthetic rule).
  The fresh corpus is already mandatory for the certified-union
  confirmation — add host/GPU/queue pressure telemetry now so time-varying
  cost c(t), multi-signal S, and E5 become testable on honest data.

Explicitly NOT proposed: program/DAG as a predictor feature — ThunderAgent
has no such predictor and Continuum's eta (turn index) is already in our
Q4 features. These papers are not evidence for the sequence-history lever
beyond what we have.

## M(t,S|P,A) path

- P (program state): adopt ThunderAgent's tuple as the state schema.
- A (actions): union of both papers' action sets = our catalog (swap-out/
  in, evict/recompute, pin-TTL, async env-prep, sandbox GC, pause/restore,
  placement); each a per-action functional over the same certified S(t|x).
  Env-prep is a LOW-stakes action (bounded misfire cost) -> a graded-
  certification story (stake-proportional evidence per action).
- S (coupling): ThunderAgent's thrashing inequality uses f(t)=x^(-t) as a
  hand-crafted stand-in for conditional survival; replace with certified
  S(t|x) per program -> a strictly more informed admission controller;
  Lagrangian GPU-memory price (Q5) decouples into per-program priced
  stopping. Pitch: "ThunderAgent's decay is the memoryless special case of
  our conditional survival; the certification gate decides, per deployment
  and per action, whether the data licenses deviating from it."
