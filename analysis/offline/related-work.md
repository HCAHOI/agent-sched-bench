# Related-work comparison

Updated 2026-08-19. The non-reference body of every primary paper named below
was read before using its limitations to define an opportunity. This is a
boundary map, not evidence that an untested opportunity works.

## What adjacent systems already cover

| Area | Systems read | What they already do | Boundary relevant here |
|---|---|---|---|
| Agent/program scheduling | [Agentix (formerly Autellix)](https://www.usenix.org/conference/nsdi26/presentation/luo), [ThunderAgent](https://arxiv.org/abs/2602.13692), [SAGA](https://arxiv.org/abs/2605.00528), [Murakkab](https://www.usenix.org/conference/osdi26/presentation/chaudhry) | Treat a program or workflow, rather than an isolated LLM request, as the scheduling unit; use completed LLM service, program phase, an execution graph, or a declarative workflow to allocate inference resources. Agentix additionally preempts arrived LLM calls and routes long calls by program locality across replicas. | These mechanisms improve LLM orchestration but do not observe or allocate the CPU/RSS/I/O work inside an opaque shell tool call. Agentix records an external interrupt only through the absence of LLM calls; its scheduler and load balancer act when an LLM call arrives. Murakkab requires an exposed declarative workflow; a black-box coding-agent trace does not provide that contract. |
| KV lifetime and placement | [Continuum](https://arxiv.org/abs/2511.02230), [KVFlow](https://arxiv.org/abs/2507.07400), [CacheWise](https://arxiv.org/abs/2606.16824), [PEEK](https://arxiv.org/abs/2607.02525), [A Policy-Driven Runtime Layer](https://arxiv.org/abs/2605.27744) | Choose a tool-gap TTL, exploit a workflow graph, rank paused sessions by predicted reuse, group waiting requests by prefix, or expose agent-aware cache policies through a runtime layer. | Continuum conditions on tool identity; KVFlow depends on an Agent Step Graph. PEEK acts on requests already waiting for inference and reports little headroom for coherent agentic bursts, so it does not decide what to do with paused sessions during tools. The policy-runtime cache study is preliminary and its CacheSage/CacheScout name changes across versions. |
| Declared semantics / programmable serving | [Parrot](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan), [Pie](https://arxiv.org/abs/2510.24051), [AgentCgroup](https://arxiv.org/abs/2602.09345) | Use declared semantic variables and dataflow, user-supplied generation programs, or agent-declared tool resource needs. | Parrot explicitly leaves dynamic control flow and native functions unsupported. Pie is a substrate, not a resource predictor. AgentCgroup controls at the outer tool-call boundary and trusts declarations; none observes and predicts individual shell clauses from an ordinary opaque command. |
| Tool/environment speculation | [SpecBox](https://arxiv.org/abs/2607.23933), [PASTE](https://arxiv.org/abs/2603.18897), [Seer](https://arxiv.org/abs/2511.14617) | Prewarm a likely sandbox during decoding, speculate recurring tool calls, or prefetch shared RL-rollout prefixes. | They predict which environment, tool, or prompt state will be needed, not the duration and CPU/RSS/disk demand of the processes that the selected tool launches. |
| Environment state | [Crab](https://arxiv.org/abs/2604.28138), [DeltaBox](https://arxiv.org/abs/2605.22781) | Checkpoint, restore, and share tool environments with semantic awareness. | This occupies the environment-snapshot direction. It does not remove the need to decide which running command should receive resources or which paused KV state should remain resident. |
| Workload evidence | [TraceLab](https://arxiv.org/abs/2606.30560) | Characterizes 4,265 sessions from 43 developers and exposes ordered agent traces plus a trace-driven serving harness. | Its public release removes raw user messages, tool arguments, and tool results. It can establish workload shape but cannot, by itself, test whether argument semantics predict tool duration. The paper also lacks provider-internal timing and represents one institutional population. |

## Executable baseline status

| Baseline | What is connected | Fidelity boundary |
|---|---|---|
| ThunderAgent | [Pinned official proxy and replay adapter](../../scripts/baselines/thunderagent_official.sh) | Uses the public program-aware inference scheduler. The released core does not consume `docker_ids` or manage tool containers. |
| Continuum | [Pinned public fork](../../scripts/baselines/continuum_public.sh) and [paper-estimator reproduction](../../scripts/baselines/continuum_reproduction.sh) | The public fork uses a fixed two-second TTL. The reproduction restores the published TTL equation on that fork; paper-omitted update details are fixed interpretations and measured costs are valid only for the bound A100 configuration. |
| CacheWise | [Pinned predictor](../../scripts/baselines/cachewise_official.sh) and [scheduler reproduction](../../scripts/baselines/cachewise_reproduction.sh) on the authors' vLLM fork | The serving-policy core is executable, but the paper does not publish the causal hook that attaches a newly generated tool call to already resident KV blocks. It is not yet an end-to-end paper baseline. |
| Agentix/Autellix | [PLAS arrival-priority reproduction](../../scripts/baselines/agentix_reproduction.sh) | The paper publishes no code or queue constants. This implements the causally measurable PLAS subset with inferred vLLM engine-step service; it does not claim in-flight demotion, anti-starvation, ATLAS, multi-engine routing, or the custom KV-swap kernel. |
| SAGA | [AFS arrival-priority reproduction](../../scripts/baselines/saga_reproduction.sh) | The paper publishes no code. The executable path applies AFS only when a request arrives; WA-LRU and TTL are decision outputs, not real KV-block actions. It does not claim SAGA's private preemption, migration, or CUDA paths. |
| Murakkab | [Static-epoch optimizer reproduction](../../scripts/baselines/murakkab_reproduction.sh) | The paper publishes no code. This reproduces the declared-DAG/profile MILP and dependency-ready plan; it does not claim the private frontend, profiler, autoscaler, or multi-engine runtime. |

## The CacheWise granularity is an outer tool call

CacheWise's `tool_args` are the arguments emitted by the model for an agent
tool. Examples in the paper are a complete `Bash` command such as a pipeline,
or the complete JSON arguments to `Grep`. CacheWise serializes that whole
argument payload, builds a TF-IDF vector with at most 5,000 terms, and applies
KMeans with `C=20,50,100`. It does **not** parse shell syntax, split a command
into clauses, observe `execve` children, or attach per-process counters.

Its isolated predictor result is therefore the comparison to reproduce first:
global point estimate -> tool-name history -> whole-tool-argument clusters.
The paper's largest `C=100` setting reduces session completion time by up to
19% relative to its coarser point predictor; the larger 2.7--3.5x headline also
includes prefix-aware scheduling and must not be attributed to the argument
predictor alone. The evaluation uses a random 80/20 session split, deterministic
trace replay, and 30--50 concurrent sessions. It does not establish temporal or
cross-project generalization; workload drift is left as future work.

Our traces remove the public-data blocker: canonical `tool_exec` actions retain
the raw outer arguments, and the eBPF lane can additionally decompose `exec`
commands and collect clause/process counters. That creates a testable next
question--whether outer-argument clustering works at all on independent SWE
tasks, and only then whether shell-aware structure improves it. It does not
erase the existing negative evidence: TF-IDF/KMeans and richer text features
have already performed poorly for several resource targets in this repository.

## Frozen SWE reproduction gate and outcome

This gate is development-only; both corpora have already been exposed.

- Fit: `traces/swe-rebench/qwen3.7-max/offline-gated-confirm-100-v2` (100
  sessions, 4,175 tool gaps containing 4,640 outer tool calls).
- Evaluate: `traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200` (277
  sessions, 12,771 tool gaps containing 13,410 outer tool calls).
- Unit: one paused-session gap from the preceding LLM completion to the next LLM
  start. Parallel tool calls emitted by one LLM turn form one batch and their
  ordered outer names/arguments are serialized together; treating them as
  independent paused sessions would duplicate one KV state.
- Ranking event: a real gap start at which another task remains paused. Under a
  hypothetical memory-pressure decision, the oracle victim is the task with
  the largest observed remaining gap. Only information available at that
  timestamp may be used. These traces do not observe actual KV eviction.
- Arms: global conditional remaining-time mean, tool-name conditional mean,
  and whole-argument TF-IDF/KMeans with the paper's fixed `C=20,50,100`.
- Primary comparison: `C=100` minus tool-name mean hypothetical-victim regret,
  paired by the unordered task pair. A 95% task-pair cluster bootstrap interval
  strictly below zero is the gate for a live serving replay. Top-1 oracle
  agreement, `C=20/50`, and global results are diagnostics.

The concurrency scan was performed before fitting any arm. The first scan over
individual `tool_exec` records found 2,519 overlaps, but inspection showed that
parallel calls from one LLM turn duplicate one paused KV state. The corrected
gap-level scan finds no cross-task overlap in SWE-100 and 2,410 ranking events
across 185 task pairs in fresh-277, all at concurrency two. Consequently this
can reproduce only CacheWise's pairwise predictor-ordering mechanism in the
forward orientation. It cannot reproduce actual pressure, the paper's
30--50-session concurrency, eviction count, or end-to-end completion-time
claim; the reverse orientation is unidentifiable and synthetic concurrency is
not a substitute.

The fixed run was read on 2026-07-31. Full machine-readable output is
`analysis/results/cachewise-swe-reproduction-20260731/result.json`.

| Arm | Mean hypothetical-victim regret | Top-1 oracle agreement |
|---|---:|---:|
| global | 0.778 s | 93.65% |
| tool-name batch | 0.780 s | 93.98% |
| C20 | 0.857 s | 94.40% |
| C50 | 1.179 s | 93.90% |
| C100 | 1.366 s | 94.07% |

The primary `C100 - tool-name` regret delta is **+0.587 s**, with 95%
unordered-task-pair cluster-bootstrap CI **[+0.120, +1.188] s**. The gate is a
clear NO-GO: the CacheWise argument-granularity improvement does not reproduce
on this SWE orientation. This is a predictor-ordering result, not an eviction
or JCT result.

The post-result mechanism diagnostic does not change that gate. C100 changed
only 90 of 2,410 rankings: 46 changes helped and 44 hurt, so unweighted top-1
slightly improved. The harmful changes accumulated 1,803 s of regret versus
390 s saved by the helpful changes, and p99 regret rose from 17.3 s (tool name)
to 35.5 s (C100). The ten largest positive task-pair deltas account for 91.5%
of the total harmful delta, although the pair-clustered interval remains wholly
above zero.

The failures are not simply singleton clusters: harmful C100 choices had median
surviving support 36. The largest misses are repository-specific test commands.
Their TF-IDF clusters contain syntactically similar `pytest`/test-suite calls
whose fitted maxima are often only 3--29 s, while the held-out repository calls
remain active for 108--300 s. Near the fitted cluster tail, C100 predicts only
0--5 s remaining; the broader tool-name distribution predicts 20--53 s and
preserves the correct long-tail ordering. Whole arguments identify "run tests"
but not test-suite scale, dependency state, cache state, or repository-specific
work. Adding a support threshold or changing tokenization after seeing these
misses would be a new, development-exposed amendment, not a reproduction.

## Conditional opportunity (gate did not pass)

The defensible novelty is not "use command text". It is a hierarchical,
causal decision interface:

```text
outer tool identity and whole arguments
        -> shell AST clauses / pipelines
        -> observed exec/process instances
        -> clause latency PMF + independent CPU/RSS/disk classes
        -> one measured scheduling or cache decision under contention
```

The comparison must keep the CacheWise outer-argument arm intact. A clause-aware
arm is useful only if it improves the downstream decision on identical events,
not merely clustering purity or prediction fit. Compound command resource
labels remain unavailable unless a separately approved physical composition
rule exists; boolean OR is not a valid composition rule.

Because the frozen outer-argument gate failed, this run does not authorize the
clause-aware arm or a live pressure experiment. The observed repo/environment
tail is still a plausible future question, but it needs a newly declared
mechanism and independent data; shell decomposition cannot be introduced
post-hoc to rescue this comparison.
