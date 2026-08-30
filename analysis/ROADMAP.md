# Research frontiers

This is the compact decision roadmap. Tool-resource metrics, data exposure,
KEEP/CLOSE decisions, and frozen gates remain authoritative in
[`development/tool-resource-canonical-objective.md`](development/tool-resource-canonical-objective.md).

## Organizing question

Treat an agent session as a long-lived job alternating between inference and
tool phases while carrying GPU/KV state and tool-side process state:

> Which turn-boundary actions reduce success-adjusted physical cost and task
> completion time without starving foreground returns or losing information
> required for later actions?

The first system does not include arbitrary process checkpointing, workspace
migration, online LLM policy generation, a new RPC framework, or dynamic tensor
parallelism. SSH/RPC are deployment machinery. EAR owns work-conserving
CPU/RSS elasticity inside workers; static container right-sizing is not a
contribution.

## Evidence that changed the frontier

| Observation | What it establishes | What it does not establish |
|---|---|---|
| Low-pressure PennyLane paper baselines | At concurrency four, GPU utilization is about 15%, waiting queues are absent, and peak logged KV occupancy is below 41%; GPU policies have little opportunity. | Failure of the paper methods in their target high-pressure regimes. |
| Unique-128 official ThunderAgent | At real GPU/KV pressure, mean JCT falls 53.4% and throughput rises 57.2% versus FCFS. | A tail-safe result: p99 TTFT rises from 445.2 to 1,524.9 s. Tools are trace-timed. |
| Unique-128 CacheWise reproduction | A program/KV policy can produce a still larger observed average gain. | Policy attribution until the exact CacheWise fork has a disabled-policy control. |
| PennyLane physical feedback | Tool-phase admission can reduce mean JCT about 33% and makespan about 42% in two repetitions. | The frozen tail-safe claim; one p99 ratio is 1.115. The hard duration predictor never activates. |
| Production Copilot characterization | Idle time has distinct intra-turn and cross-turn regimes; compaction is concentrated in token-heavy sessions and creates cache-cold work. | Causal serving benefit, task correctness, server queue/KV state, or physical tool-resource interference. |

Local receipts are indexed in [`README.md`](README.md#result-entry-points).
The production study is [Agentic Coding in the Wild](https://arxiv.org/html/2608.00101v1):
its scale makes it workload evidence, not a replacement for local causal
evaluation.

## Priority 1 — Turn-structured, return-guarded phase leasing

### Residual question

Program-aware scheduling already captures a large average-JCT opportunity.
The open mechanism question is narrower:

> Can a scheduler lend otherwise idle inference capacity during a tool phase,
> then protect the foreground session's return with bounded non-preemptible work,
> revocation, return priority, and starvation-aware progress accounting?

This is not a claim of generic backfilling, “joint CPU-GPU scheduling,” output
length prediction, or raw tool-duration prediction. A new controller is useful
only if it closes a measurable Pareto gap left by faithful existing methods.

### Baseline closure before invention

1. Run a policy-disabled control on the exact CacheWise vLLM fork. Without it,
   the Unique-128 CacheWise gain is fork-confounded.
2. Keep stock FCFS and official ThunderAgent as the control and strong baseline.
   Integrate applicable public or paper-derived Continuum and Agentix mechanisms
   with their fidelity limits visible. Do not call a subset the full paper.
3. Include native priority plus aging as the simple return-protection baseline.
   The older six-cell native-priority protocol remains frozen but must be
   reconciled with this baseline matrix before launch; its cohort, amendments,
   and gates cannot be silently repurposed.
4. Add fixed concurrency, reactive phase-only admission, and a cost-charged
   future-aware oracle on identical trajectories. The oracle measures residual
   action headroom; it is never a feature or a result arm.

Only a non-dominated gap in session JCT/throughput versus all-request TTFT and
starvation authorizes a new lease controller. If official/simple baselines close
the oracle gap, this branch stops.

### Candidate action boundary

At an LLM/tool boundary, the candidate may:

- admit a bounded borrower during an observed foreground tool phase;
- cap the borrower's non-preemptible inference work before the plausible return;
- revoke future borrower work and prioritize the returning foreground request;
- preserve KV locality when its measured reuse value exceeds avoided queueing;
- use causal completed-clause/eBPF state to update the foreground phase; and
- account for each session's cumulative service so tail protection does not
  become permanent starvation of borrowers.

Unavailable tool evidence disables speculation for that phase. It does not
reserve the whole host. CPU, RSS, Disk, KV movement, proxy wait, and all server
queueing are charged. Tool-side claims require physical tools; trace-timed tools
support only GPU/KV claims.

### Evaluation order

1. **Baseline closure:** exact-fork controls and faithful integrations above.
2. **Action-headroom screen:** same workload and eligible events for reactive,
   candidate, and future-aware oracle; no outcome-tuned thresholds.
3. **Physical primary comparison:** one preregistered high-pressure workload,
   with per-task JCT, makespan/throughput, all-request mean/p95/p99 TTFT,
   starvation, prefix/KV state, policy overhead, utilization, and energy.
4. **Fresh confirmation:** only after a primary gain, on new tasks or a new time
   period. One result per method is not run-to-run uncertainty.

The older 12-task balanced PennyLane manifest is a physical low-pressure audit,
not a substitute for the pressure regime. The 70-task simulator remains
development-exposed. A new collection is not justified while reusable traces
can answer the mechanism question.

## Priority 2 — Action-anchor-preserving adaptive compaction

Production workload evidence raises compaction above speculative MoE leasing:
compaction affects 7.8% of sessions but 44.2% of tokens; in affected sessions it
cuts prompt tokens by a median 72.8% while reducing cache-hit rate by a median
66.1%. These are paper observations, not local results.

The research question is not generic summarization. It is whether compaction can
retain exact future-action anchors—paths, symbols, diffs, tests, diagnostics,
commands, unresolved constraints, and repository state—while reducing physical
prefill/KV cost without reducing task success.

Decision order:

1. Audit which retained traces expose compaction events, pre/post context,
   cache state, future actions, and task outcomes. Missing paired content or
   outcomes makes the trace characterization-only.
2. Complete full-method related-work review before claiming a residual gap.
3. Compare on identical tasks: full context; the current
   [`MemoryStore`](../src/agents/openclaw/_memory.py)-style path; deterministic
   anchor retention plus lossless artifact externalization; and a future-aware
   oracle that identifies information later reused.
4. Measure task success first, then success-adjusted prefill latency, KV
   byte-seconds, cache misses, recomputation, and total physical cost. Token
   reduction alone is not a gate.
5. Stop if deterministic retention matches the learned policy, if the oracle
   shows little residual headroom, or if token savings do not reduce physical
   cost at unchanged success.

This priority has no result, frozen threshold, or launch authorization yet.

## Conditional characterization — cross-turn KV/expert residency

Cross-turn gaps may amortize migration; short intra-turn gaps usually do not.
MoE KV/expert leasing remains an oracle/action-disagreement audit until traces
contain expert IDs, cache misses, HBM residency, PCIe queues/transfers, and
contention. Charge KV movement, expert promotion/demotion, return latency, and
byte-seconds. Existing KV/expert elasticity work is a baseline, not novelty.
Do not implement a controller before both oracle headroom and a baseline gap
exist.

## Closed or subordinate branches

- Predictive tool-gap lending under the registered five-bucket mapping: closed;
  no distinct physical action.
- Raw command-duration prediction for per-request KV eviction: scoped No-Go;
  the deadline is already near-optimal and the decision budget is small.
- CPU-only hard bucket/page carriers and further reservation tuning: closed.
- Tool-container parking and remote snapshot RPC for current PennyLane: closed
  by the free-perfect-parking No-Go.
- Tool-latency priority under the 35-task model: closed by a 0.435% exact-duration
  oracle ceiling.
- Generic KB-structure, pip-specific semantics, prediction-weighted shares,
  arbitrary generated policies, and trace-conditioned agents: closed.
- PD separation, Disk-aware placement, and RP × TP are subordinate components;
  revisit only after a primary mechanism exposes the corresponding bottleneck.

## Preserved A100 service-model amendment

The old full-corpus simulator branch is stopped, not silently superseded.

**Calibration amendment, 2026-08-19.** Task 963's replay cached-token field was
zero. Joining provider cache data still failed the frozen TTFT transfer gate
(48.83% median, 118.56% p90 error). A post-outcome diagnostic reconstructed the
exact 16-token-block common prefix and improved the exposed fit to 24.56%/41.73%;
that could not repair the original gate. The feature was frozen, then tested
unchanged on fresh task 1320. Latency passed at 12.39%/29.49%, but TTFT was
20.91%/51.22% and missed the 50% p90 limit. Task 1320 is consumed.

The subsequent cold-start interval follow-up was frozen before task 1325. Warm
coverage and median slack passed, but the cold call missed both bounds: latency
2,239.8 ms versus 2,194.0 ms and TTFT 971.3 ms versus 968.2 ms. Under the frozen
all-cold-calls condition this is a NO-GO, even though the misses are narrow.
Do not run the old concurrency probe or full-corpus scheduler from this model.

## Authorization boundary

No controller implementation, runtime integration, new collection, or run over
30 minutes is authorized by this roadmap. A launch needs a current frozen
comparison, validity gate, cost estimate, and explicit approval. A failed gate
cannot be repaired on exposed data by changing its cohort, arm, threshold,
feature, or metric.
