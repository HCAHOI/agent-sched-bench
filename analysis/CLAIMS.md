# Current claims and evidence limits

This file records claims supported by retained evidence. It does not promote an
oracle, reproduction subset, failed gate, or external workload observation into
a system result. The canonical target and exposure contract is
[`development/tool-resource-canonical-objective.md`](development/tool-resource-canonical-objective.md).

## C1 — Command-level tool-resource signal transfers

**Supported development claim.** On exposed SQLGlot50, Task-Aware improves the
equal-weight latency/CPU/RSS/Disk accuracy from 80.203% for Clause-KB to 84.463%.
On the preregistered PennyLane transfer, it improves 75.680% to 78.326%; the
task-bootstrap gain interval is +0.746 to +4.515 percentage points. The pytest
worker-count × target-scope representation improves RSS accuracy/High recall to
87.799%/54.762% from 82.536%/2.381%.

**Limit.** These datasets are exposed; only the SQLGlot RSS head cleared the
existing five-point target gate. Accuracy is not scheduling utility, and no
predictor is deployed. Receipts:
[`results/tool-resource-5-3-3-3-20260804/sqlglot50-multitarget-sota-v1/result.json`](results/tool-resource-5-3-3-3-20260804/sqlglot50-multitarget-sota-v1/result.json)
and
[`results/pennylane-multitarget-transfer-validation-v1/result.json`](results/pennylane-multitarget-transfer-validation-v1/result.json).

## C2 — Causal runtime feedback reduces over-reservation

**Supported mechanism claim.** Causal command-level eBPF feedback reduced
logical CPU reservation 45.010% at 1.543% service inflation across 259 SWE100
and SWE277 tasks from 205 repositories.

**Limit.** This is a per-command counterfactual, not a scheduler or physical
throughput result. The tested hard-page and borrowing consumers exceeded their
service-inflation gates, so feedback is retained as a mechanism/control rather
than an integrated policy. Receipt:
[`results/tool-resource-5-3-3-3-20260804/swe100-277-cpu-feedback-generality-v1/result.json`](results/tool-resource-5-3-3-3-20260804/swe100-277-cpu-feedback-generality-v1/result.json).

## C3 — Alternating phases create action headroom, but the safe carrier is open

**Supported action-space claim.** On 70 exposed PennyLane trajectories, a
frozen hindsight joint arm reached 14,100.143 s mean completion and 42,692 s
makespan, versus 21,705.486 s and 55,628 s for the best tool-only arm, with no
modeled LLM-slot, CPU, or RSS violation. A temporal-RSS hindsight oracle also
improved mean completion 48.823% over static peak packing.

**Supported physical mechanism claim.** On one A100 and eight exposed PennyLane
tasks, causal feedback reduced mean JCT 32.748% and 33.102% and makespan 42.774%
and 42.443% in two paired repetitions.

**Limit.** The hindsight model used recorded request occupancy and proxy
resources, not causal A100 service. Both CPU-only causal carriers deadlocked.
Physical feedback was not tail-safe: paired p99 TTFT ratios were 1.048 and
1.115, so the second repetition failed the frozen 1.05 gate. The hard
Task-Aware bucket produced no distinct physical loan. Receipts:
[`results/pennylane-joint-phase-packing-v1/result.json`](results/pennylane-joint-phase-packing-v1/result.json)
and
[`results/pennylane-physical-gap-loan-development-v1/result.json`](results/pennylane-physical-gap-loan-development-v1/result.json).

## C4 — GPU/KV scheduling is load-dependent and already has a strong baseline

**Supported physical observation.** At low pressure, official ThunderAgent
performed no pause/resume action and changed mean JCT by 0.55% in the harmful
direction on the eight-task, concurrency-four comparison. The separate
12-task PennyLane suite also found no consistent improvement from the Agentix
PLAS subset, Continuum public/reproduction arms, or CacheWise reproduction;
GPU utilization was about 15%, waiting queues were absent, and peak logged KV
occupancy stayed below 41%.

At high pressure on Unique-128, official ThunderAgent reduced mean JCT 53.4%,
reduced makespan 36.4%, and increased throughput 57.2% relative to stock FCFS.
It also raised all-request p99 TTFT from 445.2 s to 1,524.9 s. Thus program-aware
scheduling closes much of the average-completion gap but leaves a large
all-request fairness/starvation tail. Whether explicit foreground-return
protection closes that tail is the open hypothesis.

**Limit.** Unique-128 is a deliberate two-repository stress workload with one
physical repetition per arm. Tools are trace-timed, so it supports GPU/KV
scheduling claims only. Receipts:
[`results/paper-baseline-physical-20260820/result.json`](results/paper-baseline-physical-20260820/result.json),
[`results/pennylane-paper-baseline-suite-physical-v1.md`](results/pennylane-paper-baseline-suite-physical-v1.md),
and
[`results/mixed128-poisson-unique-baselines-20260827/result.md`](results/mixed128-poisson-unique-baselines-20260827/result.md).

## C5 — Reproduction results inherit their implementation boundary

**Supported observations on Unique-128.** The CacheWise reproduction reduced
mean JCT 59.2% and increased throughput 68.3%, but raised p99 TTFT to 2,299.0 s.
The single-GPU SAGA KV subset increased mean JCT 34.4%; all 128 tasks were
slower.

**Limits.** CacheWise uses the authors' different vLLM fork plus this
repository's reconstruction of an unpublished causal attachment hook. An
exact-fork policy-disabled control is required before attributing its full gain
to CacheWise. The SAGA cell is only the published single-GPU KV/TTL subset; it
does not evaluate private AFS/AEG, migration, or CUDA paths. These results do
not support “CacheWise beats ThunderAgent” or “SAGA fails” as full-method
claims. Baseline classifications are in
[`offline/related-work.md`](offline/related-work.md#executable-baseline-and-result-status).

## Retained KV stopping findings

This historical lane assumed forced eviction and scored hidden swap
milliseconds. Fresh-277 had no shared KV cache and cannot support a contention,
JCT, or live-serving claim.

### C6 — Priced stopping has a bounded offline benefit

Under the frozen functional, robust-clock pre-restore yields +156.2 s/277 at a
3,500 ms KV cost (task-clustered 95% CI [60.4, 256.6]) and +317.9 s/277 at
5,000 ms ([181.0, 460.4]), after charging restore. This does not measure PCIe
contention or serving interference. See
[`certification/rolling-survival-design-20260720.md`](certification/rolling-survival-design-20260720.md).

Completed-task publication is the only retained adaptive variant: in a
fixed-order five-fold development replay it added +0.362 s/277 and +9.050 s/277
over the warmup snapshot at 3,500/5,000 ms. Per-call publication added exactly
zero. It is not an unopened-stream or order-robust result. See
[`results/prequential-task-update-20260721/`](results/prequential-task-update-20260721/).

### C7 — Elapsed-only re-checking is closed under this functional

With call survival as the only runtime observation, a multi-check plan reduces
to one precomputable stopping time. An independent `k=2` dynamic program matched
`k=1` on all 670 prior nodes across ten KV cells, with maximum value gap 0.0 ms
at tolerance `1e-6`; priced check overhead can only make extra checks worse.
Boundary identity changed 72.9% of decisions but had paired log-score gain
-0.0137 nats, task-clustered 95% CI [-0.41, +0.20]. This closes only
elapsed-only and observed-boundary enrichments under the frozen functional.

### C8 — Command-duration prediction is a scoped No-Go for per-request KV eviction

The earlier swap-out advantage over a fixed deadline was a container artifact,
not a general command-duration result. At 3,500/5,000 ms, container-setup calls
contributed +36.3/+124.5 s to total robust-clock deltas of +20.1/+118.0 s.
Removing those calls under the fixed fold map made the deltas -20.6 s (95% CI
[-55.1, +3.3]) and -6.5 s ([-27.7, +5.8]). This does not retract C6, which is a
pre-restore result at a different decision point. Receipt:
[`results/kv-swap-gain-attribution-20260729/attribution.json`](results/kv-swap-gain-attribution-20260729/attribution.json).

A frozen one-second mean-CPU-activity stump found a purer subgroup but still
lost to the deadline by 11.254/9.368 s at 3,500/5,000 ms; it fired on only
29/3 of 1,794 survivors. This closes that single-landmark feature, not all
runtime activity. Receipt:
[`results/runtime-activity-landmark-20260729/result.json`](results/runtime-activity-landmark-20260729/result.json).

The apparent large prediction budget was a batch-scale artifact. On Fresh-277,
the oracle budget was 88.9%/84.8% of deadline utility at 3,500/5,000 ms, but
those cells correspond to roughly 62/89 median requests and lie outside the
measured 238–1,071 ms swap range. At per-request costs of 56/100/160 ms, the
budget fell to 9.2%/11.2%/16.6%, and no deployable arm reliably beat the simple
elapsed deadline. SWE100 replicated the qualitative result: 8.5%/9.0%/13.3%
per-request budgets, no reliable deployable win, and large batch-scale budgets.

The restore-charge sensitivity did not rescue the lane: the negative per-key
result would flip only below model-dependent charge boundaries, while measured
reload/recompute accounting gave `rho_effective = 0.94` at every evaluated KV
cell. The best fitted constant did not improve on the deadline.

The in-sample key ladder's original “GO” is not a claim. It was confounded by
key cardinality: singleton groups informed the same oracle fit that scored them.
That defect was recorded rather than hidden. Leave-one-out evaluation made both
fine keys worse than the deadline and repository identity changed the ceiling
by -10.7/-11.0 percentage points at 3,500/5,000 ms.

Receipts:
[`results/per-request-scale-20260730/result.json`](results/per-request-scale-20260730/result.json),
[`results/cross-corpus-replication-20260730/result.json`](results/cross-corpus-replication-20260730/result.json),
[`results/effective-restore-fraction-20260730/result.json`](results/effective-restore-fraction-20260730/result.json),
and
[`results/loo-fine-key-20260729/result.json`](results/loo-fine-key-20260729/result.json).

## Explicit non-claims

- No current controller combines tool prediction, GPU/KV scheduling, CPU/RSS
  control, or compaction in production.
- No confirmation claim comes from SQLGlot, SWE100/277, PennyLane, or
  Unique-128; all relevant partitions and results are development-exposed.
- No trace-timed-tool result demonstrates physical CPU contention, interference,
  or CPU-GPU co-scheduling benefit.
- No CacheWise result is policy-attributable until an exact-fork disabled-policy
  control exists; no subset result stands for a full private system.
- No live W5 result exists. Fresh-277 had no shared KV cache and cannot establish
  memory-pressure behavior.
- No adaptive-compaction result exists in this repository. Production workload
  characterization motivates the question but supplies neither task-quality
  labels nor a causal serving comparison.
- No per-call, same-repository, same-trace, wrapper-normalized, atom/segment, or
  boundary-conditioned KV-duration policy is active.
- The removed online-first replay was protocol-invalid and contributes no
  positive or negative paper result.
