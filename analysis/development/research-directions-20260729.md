# Research directions — ranked, 2026-07-29

Working method, set by the human: do not hunt for an untouched zone. Find the
weakness in a published method, ask what breaks when it meets a new scenario, and
look for signals that are available but unused. SGLang and vLLM came *after* the
prefix-cache papers, not instead of them.

## Related work actually read

| System | Core mechanism | Verified weakness we can attack |
|---|---|---|
| [Continuum](https://arxiv.org/abs/2511.02230) | `tau* = argmax_tau P(tau,f)(T_bar*eta + PrefillReload) - tau`; `P` is a plain empirical CDF keyed on tool name | Tool name is degenerate on coding agents: `exec` is 99.96% of tool wall time, one `tau*=451 ms` for a 0.4 ms–346 s range. `eta` is one global scalar per workload. No fixed-TTL and no oracle baseline. |
| [ThunderAgent](https://arxiv.org/abs/2602.13692) | LLM-Program abstraction over KV + tool assets; program-aware scheduler; lifecycle-hook GC; async environment preparation | Gains collapse on stochastic tool time: 2.43–3.58x on predictable panels, **0.65x and 1.24x** on stochastic ones. Async prep still needs a trigger, which is the same prediction problem. GC is reactive (fires on Terminated only). |
| [AgentCgroup](https://arxiv.org/html/2602.09345v2) | eBPF cgroup control aligned to tool-call boundaries; agent *declares* resource needs | **Names a granularity mismatch and then stops one level too high.** Its own fix is tool-call-granular, but a single `exec` is `cd && apt-get install && pytest`. Declaration requires agent cooperation and the agent does not know either. Self-described preliminary; 144 tasks. |
| [Crab](https://arxiv.org/abs/2604.28138) | Semantics-aware checkpoint/restore for sandboxes; eBPF inspector; 75% of turns produce no recovery-relevant state | Targets fault tolerance / RL branching, not in-execution resource reclamation. |
| [Seer](https://arxiv.org/abs/2511.14617) | Exploits output-length similarity among requests sharing a prompt; 74–97% rollout throughput | Requires shared prompts — an RL-rollout property, absent in heterogeneous serving. |
| [PASTE](https://arxiv.org/html/2603.18897) | Speculative tool execution from recurring patterns; 43.5% TCT reduction | Speculating arbitrary shell is hard; 43.5% is not decomposed into speculation accuracy vs plain parallelism. |
| [SMetric](https://arxiv.org/abs/2607.08565), [SAGA](https://arxiv.org/html/2605.00528), [DriftSched](https://arxiv.org/html/2606.02982v1), AIOS, DeltaBox | session-centric / workflow-atomic / QoS scheduling | Evaluated on homogeneous workloads. |

**Pattern across all seven: everyone routes around tool-time unpredictability
rather than confronting it** — by declaration, sparsity, shared context, async
preparation, or speculate-and-rollback — and none states the condition under
which its routing pays.

## Ranked directions

### 1. Intra-tool-call resource heterogeneity — attacks AgentCgroup directly

AgentCgroup's central argument is a granularity mismatch: container-level policy
versus tool-call-level dynamics. We apply the identical argument one level down.
A single `exec` frequently contains an install phase and a test phase with
different CPU/RSS/disk profiles, and `CLOSED-QUESTIONS.md` records that **46% of
`exec` calls are pipelines or loops**. If so, one cgroup limit per tool call is
wrong for a large fraction of calls, by their own logic.

We hold the only instrument that can measure this: clause-level eBPF telemetry
that decomposes a shell command and attributes latency, peak CPU, sampled RSS and
disk bytes per clause. Falsifier: measure within-call dispersion of clause
resource profiles. If a tool call's clauses are homogeneous, the direction dies.

### 2. Reasoning text as a duration/resource signal — a signal nobody uses

Every surveyed system keys on tool name, command text, or agent declaration. None
reads the model's own reasoning immediately preceding the call. `python3 -m pytest`
is identical whether the agent intends a 3-test check or a full suite; the sentence
before it usually is not. Distinct from AgentCgroup's declaration because it needs
**no agent modification** — the text already exists in `messages_in` / `raw_response`.

Targets exactly the high-variance calls where command text provably fails
(`pytest` P(long)=12%, CV 1.79). Falsifier: does reasoning text separate long from
short `pytest` calls out-of-fold, and can it lift a subgroup across the cost bar?

### 3. Heterogeneous multi-tenant composition — the unevaluated deployment condition

Every paper evaluates one agent type at a time. Continuum's global `eta` and
ThunderAgent's global waiting queue both average over programs whose resource
profiles differ by orders of magnitude. A cloud sandbox provider runs coding,
science and browsing agents together. Falsifier: does a globally fitted parameter
degrade measurably when the workload is mixed?

## Standing corrections to carry

- The cost functional credits hiding swap-out and charges differential restore
  only on short calls. That is **conditional on forced eviction**; with no memory
  pressure the optimal policy is never swap. Absolute seconds in this lane are
  hidden-swap-ms under an assumed regime, not wall clock.
- The `rho/(1+rho) = 48.5%` acceptance bar is necessary but **not sufficient**: it
  assumes a long call hides the full `K`, while firing at `t` hides only
  `min(K, latency - t)`. Any future bar must use the remaining-time distribution.
- Enforce the task settlement barrier on every causal evaluation. Omitting it once
  inflated a result 18x.
