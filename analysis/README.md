# Analysis index

This directory separates current decisions from frozen evidence. Read current
research state in this order:

1. [`development/tool-resource-canonical-objective.md`](development/tool-resource-canonical-objective.md)
   — authority for tool-resource targets, causal semantics, KEEP/CLOSE
   decisions, evidence exposure, and launch boundaries.
2. [`ROADMAP.md`](ROADMAP.md) — current research frontiers and decision order.
3. [`CLAIMS.md`](CLAIMS.md) — claims the retained evidence supports, with
   explicit non-claims.
4. [`CLOSED-QUESTIONS.md`](CLOSED-QUESTIONS.md) — closed historical branches
   and the evidence required to reopen them.

If prose conflicts, that order controls interpretation. A frozen result receipt
controls its own numbers, validity, and provenance.

## Current frontier

The primary question is whether turn-structured, return-guarded phase leasing
can preserve the large completion-time gains of program-aware scheduling while
bounding request starvation. The secondary question is whether compaction can
retain exact future-action anchors while reducing success-adjusted physical
prefill and KV cost. Neither is yet a supported system claim; see
[`ROADMAP.md`](ROADMAP.md).

## Result entry points

| Evidence | Scope | Receipt |
|---|---|---|
| Unique-128 paper-baseline stress | Development-exposed, one A100, 128 unique PennyLane/SQLGlot traces, trace-timed tools | [`results/mixed128-poisson-unique-baselines-20260827/result.md`](results/mixed128-poisson-unique-baselines-20260827/result.md) |
| PennyLane paper-baseline suite | Development-exposed, one A100, 12 tasks at concurrency four, physical tools | [`results/pennylane-paper-baseline-suite-physical-v1.md`](results/pennylane-paper-baseline-suite-physical-v1.md) |
| ThunderAgent and reproduction smokes | Official ThunderAgent comparison plus Agentix, SAGA, and Continuum mechanism smokes | [`results/paper-baseline-physical-20260820/result.json`](results/paper-baseline-physical-20260820/result.json) |
| CacheWise SWE predictor ordering | Development-exposed historical local KMeans/C20–C100 reproduction, predating the official release; no live KV pressure or JCT claim | [`results/cachewise-swe-reproduction-20260731/result.json`](results/cachewise-swe-reproduction-20260731/result.json) |
| Tool-resource prediction and CPU actions | Canonical 5/3/3/3 targets and related action screens | [`results/tool-resource-5-3-3-3-20260804/`](results/tool-resource-5-3-3-3-20260804/) |
| PennyLane joint/physical phase actions | Hindsight ceilings, causal failures, physical feedback, revocable lease, and parking screens | [`development/tool-resource-canonical-objective.md`](development/tool-resource-canonical-objective.md#result-ledger) |
| Historical KV stopping lane | Forced-eviction accounting and settled duration-prediction limits | [`CLAIMS.md`](CLAIMS.md#retained-kv-stopping-findings) |

## Historical material

The receipts below are retained for provenance, but their targets or selection
criteria have been superseded. They are not inputs to current 5/3/3/3 model
selection and must not be rewritten to match the current contract.

### Superseded tool-resource contracts

| Historical contract | Preserved receipts | Status |
|---|---|---|
| Per-call targets, q90 pinball, and balanced accuracy | [`results/tool-resource-20260723/`](results/tool-resource-20260723/) | Superseded historical contract |
| Old nine-bucket latency | [`results/tool-resource-canonical-signature-v1-20260727/`](results/tool-resource-canonical-signature-v1-20260727/), [`results/tool-resource-canonical-signature-v2-20260727/`](results/tool-resource-canonical-signature-v2-20260727/), [`results/tool-resource-fit-size-20260727/`](results/tool-resource-fit-size-20260727/), [`results/tool-resource-latency-p1-20260727/`](results/tool-resource-latency-p1-20260727/), [`results/tool-resource-representation-20260727/`](results/tool-resource-representation-20260727/) | Superseded historical contract |
| Old binary resource classes | [`results/tool-resource-heavy-light-20260728/`](results/tool-resource-heavy-light-20260728/) | Superseded historical contract |
| Failed shared-runtime replacements | [`results/tool-resource-local-vs-public-20260727/`](results/tool-resource-local-vs-public-20260727/), [`results/tool-resource-nighttime-3bucket-20260728/`](results/tool-resource-nighttime-3bucket-20260728/) | Superseded historical contract |

### Archived branches and presentations

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
- `development/` contains the current authority, frozen inputs, and protocols
  still referenced by evaluators or results.
- `certification/` contains retained calibration and adjudication evidence.
- `serving/` contains hardware measurements and historical serving inputs; it
  does not imply a live result exists.
- `offline/` is related-work and background analysis, not status authority.
- `archive/` contains retired audits, inputs, and prototypes; nothing there is
  an active implementation or launch definition.

Large raw runs may live outside Git. Treat an absolute run path as provenance,
not a portable artifact; a result is locally recoverable only when its receipt
also names a retained in-repository file, archive, or Git object.
