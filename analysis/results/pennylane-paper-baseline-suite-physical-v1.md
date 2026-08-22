# PennyLane paper-baseline physical result

## Result

On this 12-task, concurrency-4 PennyLane replay, none of the four paper
baselines provides a consistent end-to-end improvement over stock vLLM FCFS:
every method worsens makespan and throughput, while mean-JCT changes are small
enough to be covered by tool-runtime variation.  The main reason is lack of
GPU-side contention: the workload issues only about 0.05 LLM
requests/s, mean GPU utilization is 15.3--15.7%, vLLM almost never has a waiting
request, and peak logged KV-cache occupancy is below 41%.  The run is valid as
an opportunity audit, but it is not evidence that the methods fail under the
high-concurrency/cache-pressure regimes targeted by their papers.

The four baseline cells completed 48/48 replays.  Every replay reports
`success=true`, exact source action order, exact provider request order, no
missing source actions, and telemetry quality `ok`.  No OOM, NVIDIA XID, HTTP
5xx, or GPU-sampler error appears in the retained logs.

## Physical comparison

FCFS is the preceding physical control with the same manifest, model,
concurrency, CPU placement, generation inputs, and GPU.  Lower is better for
makespan, JCT, LLM latency, and TTFT; higher is better for task throughput.
Deltas in parentheses are relative to FCFS.

| Method | Makespan, h | Mean task JCT, min | Tasks/h | Mean LLM latency, s | TTFT mean / p99, ms | Mean GPU util. | GPU energy, kWh |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCFS control | 4.614 | 159.00 | 2.601 | 4.028 | 366.8 / 1196.0 | 15.56% | 0.4582 |
| Agentix PLAS subset | 4.651 (+0.80%) | 159.23 (+0.14%) | 2.580 (-0.79%) | 4.041 (+0.32%) | 374.2 / 1218.6 | 15.59% | 0.4622 |
| Continuum-public | 4.646 (+0.69%) | 158.03 (-0.62%) | 2.583 (-0.69%) | 4.052 (+0.58%) | 371.8 / 1245.1 | 15.37% | 0.4587 |
| Continuum-reproduction | 4.637 (+0.49%) | 159.09 (+0.06%) | 2.588 (-0.49%) | 3.984 (-1.09%) | 373.0 / 1173.8 | 15.69% | 0.4613 |
| CacheWise reproduction | 4.672 (+1.25%) | 161.06 (+1.29%) | 2.568 (-1.23%) | 5.181 (+28.63%) | 478.5 / 1471.1 | 15.33% | 0.4642 |

All cells execute 1,668 actions: 840 LLM requests and 828 tool calls (640
shell, 86 reads, 86 edits, and 16 directory listings).  Across methods, all
840 tuples of message hash, prompt-token hash, and requested completion length
match FCFS.  Each cell processes 34,132,332 prompt tokens and 169,692 returned
tokens.  The largest prompt has 109,934 tokens, below both Continuum context
limits.

The end-to-end deltas are smaller than tool-runtime variation.  Total tool
time ranges from 56,971 to 57,396 seconds across cells.  Each cell contains
54--56 calls at the 600-second timeout, and calls lasting at least 60 seconds
account for 54,804--55,174 seconds, about 96% of all tool time.  The replay
contract intentionally requires exact commands but not identical command
outcomes; the cells contain 4--6 extra failed tool calls relative to their
source traces.  Therefore the roughly one-percent JCT/makespan differences
cannot be attributed causally to the GPU schedulers.

## What each mechanism actually did

### Agentix

This is the paper-derived PLAS arrival-priority subset, not full Agentix: stock
vLLM cannot reproduce in-flight quantum demotion, anti-starvation, contiguous
KV swapping, ATLAS, or multi-engine routing.  All 840 requests were assigned
and completed and all 12 programs were released.  However, the chosen queue
boundaries quickly saturated: priorities were 0 for 12 requests, 2 for 27, 3
for 63, and the lowest queue 4 for 738/840 (87.9%).  vLLM logged zero waiting
requests throughout.  Thus the method assigned priorities but had almost no
queue choice to act on, producing no latency or JCT benefit.

### Continuum-public

This is the authors' public fixed-TTL implementation, not the full estimator
described in the paper.  The public parser recovered a non-empty preceding
tool name in only 13/828 follow-up opportunities (1.6%); vLLM again logged no
waiting request.  Its end-to-end and LLM metrics are effectively FCFS-level.

### Continuum-reproduction

This uses the same public vLLM 0.10.2 fork plus the reproduced paper equation
and the measured A100 prefill profile.  It made 12 TTL decisions, of which 10
were positive (1.72--2.84 s).  Compared directly with Continuum-public, it
reduced mean LLM latency by 1.66% and p99 TTFT by 5.73%, but increased mean TTFT
by 0.33% and mean task JCT by 0.68%; makespan improved by 0.20%.  With only 12
decisions, no waiting queue, one physical repetition, and tool-time variation,
these are directional observations rather than a demonstrated end-to-end
gain.

### CacheWise

The client issued 840 successful session-policy updates.  Their direct client
overhead was 33.4 ms/request on average, 53.2 ms at p95, and 28.0 seconds in
total.  But CacheWise's main eviction mechanism had no pressure to resolve:
logged KV occupancy peaked at 35.1%, and only one 10-second reporter sample had
one waiting request.  Prefix-cache hit rate ended at 97.4%, versus 97.2% for
FCFS, too small and too confounded to claim as a gain.  Mean server request
latency and mean TTFT worsened by 28.6% and 30.5%, respectively.

The CacheWise row uses the authors' newer vLLM fork
`0.1.dev1+g16cc7d43d`, while FCFS and Agentix use vLLM 0.11.2 and Continuum
uses vLLM 0.10.2.  Without a policy-disabled control on the same CacheWise
fork, the large latency regression cannot be assigned solely to CacheWise's
policy.  Its post-response tool-to-session update is also our causal
reproduction of an unpublished paper hook, not author-released end-to-end
code.

## Interpretation limits

- One physical run per method measures this execution, not run-to-run variance.
- Tools dominate JCT and have nondeterministic timeouts; LLM request metrics
  are cleaner than the small end-to-end deltas.
- GPU memory is almost fully preallocated by each vLLM server (about
  73.5--73.7 GiB), so it is not evidence of useful application memory demand.
- The methods run on different required vLLM forks.  Continuum-public versus
  Continuum-reproduction is the closest software-matched comparison; their
  configured maximum context differs, although no request exceeds the smaller
  120,000-token limit and neither run approaches KV pressure.
- Shadow generation does not choose the replayed source tool action.  This is
  especially important for Continuum, whose model-output parser recognized
  only 13 tool transitions.  CacheWise instead receives the causally revealed
  source tool call after each generated response, so semantic activation is
  asymmetric.

## Paper-ready takeaway

For long, tool-heavy coding agents at concurrency four, GPU-internal scheduling
is not the active bottleneck: the A100 is idle about 84% of the time, request
queues are absent, and KV capacity is plentiful.  Faithful public or
paper-derived implementations of Agentix, Continuum, and CacheWise therefore
do not improve end-to-end throughput on this batch; their decision mechanisms
either collapse to one priority, rarely activate, or never face eviction
pressure.  This result motivates phase-aware admission of additional agent
work during tool gaps and cross-resource scheduling.  It does not justify a
claim that the related systems fail in the higher-concurrency regimes for
which they were designed.

## Evidence and metric definitions

Baseline outputs:

- Four methods: `/home/Ubuntu/pennylane-paper-baseline-suite-v1-20260820/full`
- FCFS control: `/home/Ubuntu/pennylane-paper-baselines-mixed12-sharedcpu-v1-20260820/fcfs-r1`

JCT is `ready_to_terminal_s`, so it includes admission wait.  Makespan is
`ready_to_all_terminal_s`.  Request latency and TTFT are the 840
`shadow_generation` observations in the per-task replay JSONL files.  GPU
utilization and energy use the 1 Hz `gpu.csv` samples bounded by the common
ready time and final task completion; energy is interval-integrated power.
Policy counts come from `agentix-events.jsonl`, Continuum vLLM logs, CacheWise
HTTP/update observations, and vLLM's periodic scheduler metrics.  All aggregate
values above were recomputed from these raw files rather than copied from a
derived result artifact.
