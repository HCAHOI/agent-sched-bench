# Related-work and executable-baseline map

Updated 2026-08-30. The non-reference body of every primary paper used below to
define an opportunity was read. This map distinguishes paper claims from code
that is actually executable here; it is not evidence that an open opportunity
works.

## Provenance vocabulary

- **Official**: an author-released policy core, with only replay/integration
  wrapping added locally.
- **Public**: author-released code, but the exposed mechanism may be narrower or
  older than the method described in the paper.
- **Reproduction**: this repository reconstructs paper logic or an unpublished
  hook; omitted details are local interpretations.
- **Subset**: only the named mechanism is executable. Results do not stand for
  the full paper system.

These labels may compose: for example, the CacheWise serving cell is a local
reproduction on the authors' public vLLM fork.

## What adjacent systems already cover

| Area | Systems read | Established mechanism | Boundary relevant here |
|---|---|---|---|
| Agent/program scheduling | [Agentix (formerly Autellix)](https://www.usenix.org/conference/nsdi26/presentation/luo), [ThunderAgent](https://arxiv.org/abs/2602.13692), [SAGA](https://arxiv.org/abs/2605.00528), [Murakkab](https://www.usenix.org/conference/osdi26/presentation/chaudhry) | Schedule a program/workflow rather than isolated requests using attained service, program phase, execution graphs, or declared DAGs. Agentix also prioritizes arrived calls and routes by program locality. | Generic program scheduling is not novel. The residual question is advance protection of a foreground return while its tool is still running, with charged tool-side state and starvation bounds. Murakkab assumes an exposed declarative workflow. |
| KV lifetime and placement | [Continuum](https://arxiv.org/abs/2511.02230), [KVFlow](https://arxiv.org/abs/2507.07400), [CacheWise](https://arxiv.org/abs/2606.16824), [PEEK](https://arxiv.org/abs/2607.02525), [A Policy-Driven Runtime Layer](https://arxiv.org/abs/2605.27744) | Choose tool-gap TTLs, exploit workflow graphs, rank paused sessions by predicted reuse, batch waiting prefixes, or expose agent-aware cache policy. | Adaptive TTL, prefix grouping, and paused-session ranking are baselines. PEEK acts on already-waiting requests; Continuum conditions on outer tool identity; KVFlow assumes an Agent Step Graph. |
| Declared semantics / programmable serving | [Parrot](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan), [Pie](https://arxiv.org/abs/2510.24051), [AgentCgroup](https://arxiv.org/abs/2602.09345) | Use semantic variables/dataflow, user-supplied generation programs, or declared tool resources. | Parrot excludes dynamic control flow and native functions; Pie is a substrate; AgentCgroup trusts outer-call declarations. None supplies causal per-shell-clause measurements from opaque commands. |
| Tool/environment speculation | [SpecBox](https://arxiv.org/abs/2607.23933), [PASTE](https://arxiv.org/abs/2603.18897), [Seer](https://arxiv.org/abs/2511.14617) | Prewarm sandboxes, speculate recurring tool calls, or prefetch shared rollout prefixes. | They predict which environment/tool/prompt state is needed, not the duration and CPU/RSS/Disk demand of processes launched by the selected call. |
| Environment state | [Crab](https://arxiv.org/abs/2604.28138), [DeltaBox](https://arxiv.org/abs/2605.22781) | Checkpoint, restore, and share tool environments. | Environment snapshotting is occupied prior work and the free-perfect-parking screen is locally negative. It is not the current contribution. |
| Workload evidence | [TraceLab](https://arxiv.org/abs/2606.30560), [Agentic Coding in the Wild](https://arxiv.org/html/2608.00101v1) | Characterize real agent sessions, tool/LLM alternation, context growth, idle time, and compaction at scale. | TraceLab removes raw messages, tool arguments, and results. The Copilot study lacks prompt/tool content, task-quality labels, server queue/KV state, and physical resource measurements. Both motivate questions; neither proves a scheduler or compaction policy. |

## Executable baseline and result status

| Baseline | Executable provenance | Implemented surface | Current physical evidence |
|---|---|---|---|
| Stock vLLM FCFS | Native control | Prefix-caching server without a paper policy | Control in both PennyLane and Unique-128. |
| ThunderAgent | **Official** policy core via [`thunderagent_official.sh`](../../scripts/baselines/thunderagent_official.sh) | Public program-aware proxy; released core does not manage tool containers | Low pressure: no pause/resume, mean JCT 0.55% worse. Unique-128: mean JCT -53.4%, throughput +57.2%, p99 TTFT 445.2→1,524.9 s. Strong baseline. |
| Continuum-public | **Public** author fork via [`continuum_public.sh`](../../scripts/baselines/continuum_public.sh) | Fixed two-second TTL exposed by the public fork, not the paper estimator | Valid low-pressure PennyLane run; no attributable end-to-end gain. Not tested on Unique-128. |
| Continuum-reproduction | **Reproduction** on the public fork via [`continuum_reproduction.sh`](../../scripts/baselines/continuum_reproduction.sh) | Paper TTL equation plus locally measured A100 prefill/reload profile | Mechanism smoke passed; low-pressure run made only 12 TTL decisions and showed no demonstrated end-to-end gain. |
| CacheWise predictor | **Official** released predictor via [`cachewise_official.sh`](../../scripts/baselines/cachewise_official.sh) | Whole-tool-argument TF-IDF/MiniBatchKMeans duration predictor | Adapter/training smoke only. The SWE pairwise gate below used an older local reproduction and does not evaluate this official predictor. |
| CacheWise serving | **Reproduction** on authors' public vLLM fork via [`cachewise_reproduction.sh`](../../scripts/baselines/cachewise_reproduction.sh) | Reconstructed causal generated-tool→resident-KV attachment; exact paper hook and split are unpublished | Low pressure: no opportunity and fork-confounded latency. Unique-128: mean JCT -59.2%, throughput +68.3%, p99 2,299.0 s. Exact-fork disabled-policy control still required. |
| Agentix/Autellix | **Reproduction subset** via [`agentix_reproduction.sh`](../../scripts/baselines/agentix_reproduction.sh) | Causally measurable PLAS arrival priority with inferred engine-step service; no ATLAS, in-flight demotion, anti-starvation, multi-engine routing, or KV-swap kernel | Mechanism smoke passed; low-pressure priority assignment had no queue choice and no demonstrated end-to-end gain. Not tested on Unique-128. |
| SAGA | **Reproduction subset** via [`saga_reproduction.sh`](../../scripts/baselines/saga_reproduction.sh) | Single-GPU WA-LRU/adaptive TTL and arrival-priority surface; no private AFS/AEG, periodic scheduler, migration, or CUDA paths | Mechanism smoke passed. Unique-128: mean JCT +34.4%; 0/128 tasks faster. This is not a full-SAGA verdict. |
| Murakkab | **No current executable baseline.** The historical reproduction subset was removed from the active baseline surface. | It covered only the declared-DAG/profile MILP and dependency-ready static epochs, not the private frontend, profiler, autoscaler, or multi-engine runtime. | No claim-bearing physical comparison. Recreate only from Git history if a new comparison makes it decision-relevant. |

Primary receipts:
[`../results/paper-baseline-physical-20260820/result.json`](../results/paper-baseline-physical-20260820/result.json),
[`../results/pennylane-paper-baseline-suite-physical-v1.md`](../results/pennylane-paper-baseline-suite-physical-v1.md),
and
[`../results/mixed128-poisson-unique-baselines-20260827/result.md`](../results/mixed128-poisson-unique-baselines-20260827/result.md).

## Pressure changes the conclusion

The PennyLane and Unique-128 results are complementary, not contradictory.

| Regime | Opportunity | Defensible reading |
|---|---|---|
| Eight or twelve tasks, concurrency four | GPU about 15% utilized, no meaningful waiting queue, peak logged KV below 41% | Faithful mechanisms have little action opportunity; small end-to-end deltas are covered by physical tool variation. |
| Unique-128, Poisson arrivals, vLLM `max_num_seqs=8` | FCFS queue reaches 119 requests and KV occupancy 99.9% | Program/KV scheduling is first order. ThunderAgent is a strong official baseline, but its average gain comes with a 3.4x p99 TTFT. |

Unique-128 uses 128 distinct tasks and identical trace-timed tools across arms,
so GPU/KV attribution is cleaner than in the physical-tool PennyLane suite. It
still has only two repositories, one repetition per policy, and no physical
tool interference. CacheWise remains fork-confounded; SAGA remains a bounded
subset.

## Historical CacheWise-style predictor reproduction: frozen No-Go

This local C20/C50/C100 reproduction predates the official predictor release;
it is not the executable official baseline listed above. It clusters one
complete outer tool-argument payload in the CacheWise style. It does not parse
shell syntax, split pipelines, observe `execve` children, or attach per-process
counters. Its predictor must therefore be reproduced before adding clause-aware
structure.

The development-only frozen comparison fit 100 SWE sessions (4,175 gaps, 4,640
outer calls) and evaluated 277 sessions (12,771 gaps, 13,410 calls). Correcting
parallel calls from one turn to one paused KV state left 2,410 pairwise ranking
events across 185 task pairs, all at concurrency two. This identifies only the
forward predictor ordering; it has no actual KV pressure, eviction, reverse
orientation, or 30–50-session paper concurrency.

| Arm | Mean hypothetical-victim regret | Top-1 oracle agreement |
|---|---:|---:|
| global | 0.778 s | 93.65% |
| tool-name batch | 0.780 s | 93.98% |
| C20 | 0.857 s | 94.40% |
| C50 | 1.179 s | 93.90% |
| C100 | 1.366 s | 94.07% |

The frozen primary delta, `C100 - tool-name`, is **+0.587 s**, with 95%
task-pair-cluster bootstrap CI **[+0.120, +1.188] s**. The gate is NO-GO:
whole-argument clustering does not reproduce the paper's predictor ordering in
this SWE orientation. It is not an eviction or JCT result.

Post-outcome diagnosis found only 90/2,410 changed rankings. The 44 harmful
changes accumulated 1,803 s of regret versus 390 s saved by 46 helpful changes;
p99 regret rose from 17.3 to 35.5 s. Repository-specific test scale, dependency
state, and cache state were missing from syntactically similar clusters. Adding
a support threshold, new tokenization, shell clauses, or synthetic concurrency
after seeing these misses would be an exposed amendment, not a reproduction.
Receipt:
[`../results/cachewise-swe-reproduction-20260731/result.json`](../results/cachewise-swe-reproduction-20260731/result.json).

## Residual opportunities

The defensible phase-scheduling target is not “use command text” or “schedule
programs.” It is to preserve the official ThunderAgent-scale average gain while
bounding foreground-return and all-request starvation, after exact-fork controls
and faithful applicable baselines. Clause/eBPF state is useful only if it changes
that charged action on identical events.

Action-anchor-preserving compaction is separately plausible because production
evidence shows concentrated token and cache costs, but its executable baseline
and full-method related-work review are not yet closed. No compaction policy
claim exists in this repository.
