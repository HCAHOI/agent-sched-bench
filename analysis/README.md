# Analysis index

The current research state is the milestone series:
[`../millstone/MILESTONE-1-Single-Instance.md`](../millstone/MILESTONE-1-Single-Instance.md)
through
[`../millstone/MILESTONE-4-Pressure.md`](../millstone/MILESTONE-4-Pressure.md),
with [`../millstone/PENDING.md`](../millstone/PENDING.md) as the live queue of
experiments and open decisions. A fresh session starts from
[`../millstone/NEXT-SESSION.md`](../millstone/NEXT-SESSION.md), the onboarding
prompt.

This directory is not that record. It is the store for frozen result receipts
and the authority for the historical KV-stopping lane. The tables below index
the receipts that still carry a claim; the rest of `results/` is retained
provenance. Inside that scope, read in this order:

1. [`ROADMAP.md`](ROADMAP.md) — the frontier framing, decision order, closed
   branches, and amendments that still bind this material.
2. [`CLAIMS.md`](CLAIMS.md) — claims the retained evidence supports, with
   explicit non-claims.
3. [`CLOSED-QUESTIONS.md`](CLOSED-QUESTIONS.md) — closed historical branches
   and the evidence required to reopen them.

If prose conflicts inside that scope, that order controls interpretation;
outside it the milestone series controls. A frozen result receipt controls its
own numbers, validity, and provenance.

## Result entry points

| Evidence | Scope | Receipt |
|---|---|---|
| Unique-128 paper-baseline stress | Development-exposed, one A100, 128 unique PennyLane/SQLGlot traces, trace-timed tools | [`results/mixed128-poisson-unique-baselines-20260827/result.md`](results/mixed128-poisson-unique-baselines-20260827/result.md) |
| PennyLane paper-baseline suite | Development-exposed, one A100, 12 tasks at concurrency four, physical tools | [`results/pennylane-paper-baseline-suite-physical-v1.md`](results/pennylane-paper-baseline-suite-physical-v1.md) |
| ThunderAgent and reproduction smokes | Official ThunderAgent comparison plus Agentix, SAGA, and Continuum mechanism smokes | [`results/paper-baseline-physical-20260820/result.json`](results/paper-baseline-physical-20260820/result.json) |
| CacheWise SWE predictor ordering | Development-exposed historical local KMeans/C20–C100 reproduction; no live KV pressure or JCT claim. The official reproduction fork has since been run and is recorded in [`../millstone/MILESTONE-1-Single-Instance.md`](../millstone/MILESTONE-1-Single-Instance.md) | [`results/cachewise-swe-reproduction-20260731/result.json`](results/cachewise-swe-reproduction-20260731/result.json) |
| PennyLane joint/physical phase actions | Hindsight joint packing ceiling and paired physical feedback repetitions | [`results/pennylane-joint-phase-packing-v1/result.json`](results/pennylane-joint-phase-packing-v1/result.json), [`results/pennylane-physical-gap-loan-development-v1/result.json`](results/pennylane-physical-gap-loan-development-v1/result.json) |
| Historical KV stopping lane | Forced-eviction accounting and settled duration-prediction limits | [`CLAIMS.md`](CLAIMS.md#retained-kv-stopping-findings) |

The Unique-128 run directory is no longer present under the repository-root
`results/` tree; its receipt above is the surviving record.

## Single-instance L40S notes

These are the retained notes from the late-August and early-September 2026
single-instance runs on one NVIDIA L40S. They belong with
[`../millstone/MILESTONE-1-Single-Instance.md`](../millstone/MILESTONE-1-Single-Instance.md),
not with the two-instance milestones. None of their run directories is present
under the repository-root `results/` tree, and the absolute paths they record
point at L40S hosts that are no longer available, so the underlying artifacts are
unrecoverable. Each note is frozen: it is the only surviving evidence for its
run, and it is not to be rewritten.

| Note | What it records |
|---|---|
| [`results/mixed32-l40s-continuous-load-20260901/finite-fcfs.md`](results/mixed32-l40s-continuous-load-20260901/finite-fcfs.md) | 32-task finite-batch exact-fork FCFS; all 32 tasks succeeded. Retained as a descriptive comparison for the sustained-load FCFS. |
| [`results/mixed32-l40s-continuous-load-20260901/sustained-fcfs.md`](results/mixed32-l40s-continuous-load-20260901/sustained-fcfs.md) | 32-task sustained-load exact-fork FCFS; all 32 measured tasks succeeded. The baseline for the sustained-load comparisons. |
| [`results/mixed32-l40s-continuous-load-20260901/sustained-cachewise.md`](results/mixed32-l40s-continuous-load-20260901/sustained-cachewise.md) | Sustained-load CacheWise; one measured task starved for 1,800 s and hit the client timeout. Diagnostic only, not comparable with sustained FCFS. |
| [`results/mixed32-l40s-continuous-load-20260901/sustained-continuum-reproduction.md`](results/mixed32-l40s-continuous-load-20260901/sustained-continuum-reproduction.md) | Sustained-load Continuum reproduction; a replacement program's first request starved for 1,800 s and hit the client timeout. Diagnostic only, supports no speedup claim. |
| [`results/mixed32-l40s-continuous-load-20260901/replacement-load-implementation.md`](results/mixed32-l40s-continuous-load-20260901/replacement-load-implementation.md) | Implementation note for the sustained replacement-load option and its cohort accounting. Not a result. |
| [`results/mixed32-l40s-continuous-load-20260901/replacement-load-smoke.md`](results/mixed32-l40s-continuous-load-20260901/replacement-load-smoke.md) | Two-task plumbing smoke for the replacement load. Explicitly not a performance result. |
| [`results/mixed32-l40s-serving-metrics-20260901/serving-metrics-smoke.md`](results/mixed32-l40s-serving-metrics-20260901/serving-metrics-smoke.md) | One-task serving-telemetry smoke that found and fixed a run-start-clock defect. Plumbing validation, not evidence. |
| [`results/mixed24-l40s-cachewise-loop-l0-20260901/fcfs.md`](results/mixed24-l40s-cachewise-loop-l0-20260901/fcfs.md) and [`results/mixed24-l40s-cachewise-loop-l0-20260901/cachewise.md`](results/mixed24-l40s-cachewise-loop-l0-20260901/cachewise.md) | The mixed24 load-ladder L0 pair, exact-fork policy-disabled control against original CacheWise. Both arms completed 24/24. See the failed gate below. |
| [`results/mixed28-l40s-closed-calibration-20260902/result.md`](results/mixed28-l40s-closed-calibration-20260902/result.md) | 15-minute fixed-concurrency diagnostic that selected 16 concurrent sessions for the closed-load comparison. Intentionally stopped after its observation window. |
| [`results/mixed28-l40s-closed-c16-replenished-trace4x-20260902/native-priority.md`](results/mixed28-l40s-closed-c16-replenished-trace4x-20260902/native-priority.md) | The vLLM native-priority scheduler against FCFS at 16 sessions with replenished background and 4x tool replay; 28/28 succeeded in both arms. |
| [`results/mixed28-l40s-poisson-background-20260902/fcfs.md`](results/mixed28-l40s-poisson-background-20260902/fcfs.md) | 28-task exact-fork FCFS under a fixed low-rate Poisson background; all 28 measured tasks succeeded. |

### Mixed24 CacheWise load ladder: L0 failed its pre-registered gate

The ladder in
[`development/mixed24-l40s-poisson-real-tools-v1/load-ladder.md`](development/mixed24-l40s-poisson-real-tools-v1/load-ladder.md)
fixed its exit gate before the CacheWise result was read: CacheWise had to reach
1.30x the exact-fork policy-disabled control's task throughput, equivalently a
makespan no greater than the control's divided by 1.30. At L0 the control ran at
6.917 Task/h with a 12,490.135 s makespan and CacheWise at 8.368 Task/h with a
10,324.710 s makespan — 1.210x, and 716.914 s above the 9,607.796 s makespan
ceiling, so the gate failed. L1 and L2 were never launched, so the ladder never
exited by its own rule; their manifests
([`development/mixed24-l40s-poisson-real-tools-l1-2x/manifest.yaml`](development/mixed24-l40s-poisson-real-tools-l1-2x/manifest.yaml)
and
[`development/mixed24-l40s-poisson-real-tools-l2-4x/manifest.yaml`](development/mixed24-l40s-poisson-real-tools-l2-4x/manifest.yaml))
remain in the repository with no results. The ladder document's Outcome section
records the stop and the reason given for it.

## Historical material

The material below is retained for provenance. Nothing in it is an active
implementation or launch definition, and none of it is to be rewritten to match
a later contract.

| Material | Status |
|---|---|
| [`archive/tb-dev100-audit-20260724/`](archive/tb-dev100-audit-20260724/) | Historical Terminal-Bench audit |
| [`archive/tool-time-duration-research-202607/`](archive/tool-time-duration-research-202607/) | Historical duration-policy configuration and analysis |
| [`archive/vllm-selective-offload-w5-202607/`](archive/vllm-selective-offload-w5-202607/) | Historical W5 code/input snapshot; it is not a supported runnable package after relocation, so restore its original Git revision for reproduction from the original paths |
| [`slides/research-journey-20260819/`](slides/research-journey-20260819/) | Historical presentation, not a current progress report |
| [`slides/cachewise-reproduction-20260731/`](slides/cachewise-reproduction-20260731/) | Historical presentation predating the official/exact-fork baseline work |

Baseline provenance labels are not interchangeable:

- **Official**: an author-released policy core, possibly wrapped only for replay.
- **Public**: author-released code whose exposed behavior may be a narrower or
  older mechanism than the paper.
- **Reproduction**: this repository reconstructed paper logic or an unpublished
  integration hook; assumptions belong to this repository.
- **Subset**: only the named mechanism is present, so the full paper method is
  not being evaluated.

The exact classification of every executable baseline is in
[`offline/related-work.md`](offline/related-work.md#executable-baseline-and-result-status)
and [`../scripts/baselines/README.md`](../scripts/baselines/README.md).

## Directory roles

- `results/` contains immutable result receipts and machine-readable outputs.
- `development/` contains frozen run manifests, workload splits, and launch
  plans still referenced by those receipts.
- `certification/` contains retained calibration and adjudication evidence.
- `serving/` contains hardware measurements and historical serving inputs; it
  does not imply a live result exists.
- `offline/` is related-work and background analysis, not status authority.
- `archive/` contains retired audits, inputs, and prototypes; nothing there is
  an active implementation or launch definition.

Run artifacts themselves live in the repository-root `results/` tree, not in
`analysis/results/`: every run named in Milestones 1 through 4 writes there. That
tree is large and Git-ignored, and it is never deleted or rewritten. Because it
is outside Git, treat an absolute run path as provenance, not a portable
artifact; a result is locally recoverable only when its receipt also names a
retained in-repository file, archive, or Git object.
