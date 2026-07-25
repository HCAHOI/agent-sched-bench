# Tool-Resource Prediction — Canonical Objective Lock

**Effective 2026-07-25.** This is the canonical contract for tool-resource work. It overrides conflicting objective, metric, and current-architecture statements in older development plans or chat summaries. Historical artifacts remain evidence about what was run, not instructions for what to optimize now.

## Primary acceptance objective

Predict three independent **command-level binary labels** before execution:

| Target | Positive label | Existing operating point |
|---|---|---|
| latency | long | `>3500 ms` and `>5000 ms` are the existing reported points |
| peak CPU | heavy | `>2 cores` |
| peak memory | heavy | `>500 MB` |

Primary metric: **balanced accuracy**, with the advisor/manager acceptance goal approximately **0.70–0.80**. Also report class prevalence, recall, precision, and confusion counts so accuracy is not hidden by imbalance.

Existing development references, which should not be needlessly rediscovered:

- latency BA: `0.720 / 0.737` at 3500/5000 ms;
- memory BA: `0.867` at 500 MB;
- CPU BA: `0.684` overall at 2 cores, `0.730` with repo history, `0.776` on task repeats.

## `q90` means q-error p90

In this project discussion, `q90` means the 90th percentile of per-example multiplicative q-error:

```text
q_error = max(predicted / observed, observed / predicted)
q90 = percentile_90(q_error over examples)
```

It does **not** mean predicting the conditional 90th-percentile resource value. Existing latency development reference: feature MLP q-error p90 `4.8` versus ECDF `13.5`.

Conditional-quantile/pinball experiments are secondary diagnostics unless explicitly requested; they must not replace the classification acceptance objective.

## Intended composition architecture

```text
shell tool call
  -> mvdan clauses / binary identities
  -> pre-call per-clause predictions from public + repo knowledge
  -> three clause flags: long, CPU-heavy, memory-heavy
  -> command flag for each target = OR over its clause flags
```

This is a three-label prediction, not one combined multiclass label. Public per-binary knowledge must remain useful for compound commands. Repo/workspace knowledge may refine exact clause/ordered-prefix/bin behavior causally.

The OR rule is the primary simple candidate and must be evaluated against observed command-level labels. It is sufficient but not logically complete for all physical cases: sequential short clauses can accumulate into a long command, and concurrent individually-light processes can jointly cross CPU/memory thresholds. Report those failure modes rather than silently adding a more complex composer.

For any secondary continuous composition: sequential durations may add; pipeline members overlap and require a wall-time envelope/critical path; quantiles are not directly additive.

## Current integration status — do not overclaim

Completed components:

1. mvdan.cc/sh/v3 v3.13.1 clause parsing (`bin`, `argv`, source span, loop/pipe/substitution context);
2. host-side eBPF lifecycle telemetry synthetic Stage-1b (`b3c3bf7`), including `t_exec_ns`, `t_exit_ns`, `wall_ns`, cumulative CPU time, peak RSS, lineage, and clause mapping;
3. **eBPF Stage-2 windowed telemetry** (`analysis/development/clause-telemetry-ebpf-stage2-20260725/`): ~10 ms perf CPU-clock sampling in 500 ms wall windows for honest per-clause `peak_cpu_cores` (quota-clipped, never `cpu_ns/wall_ns`) and `sampled_peak_rss` (max aligned distinct-mm sum, never lifetime hiwater), with exec/exit boundary snapshots, lineage attribution, coverage-gap records, provenance, and target-specific `unavailable` semantics. **Synthetic P/W/M/K/B validation now passes** (local-cgroup harness; the frozen Stage-1b docker image is gone from the host);
4. a tested asymmetric Runtime KB with frozen public state, causal repo updates, serialization, evidence/fallback metadata, and monotonic-query guard;
5. **the Stage-2 -> Runtime KB clause bridge** (`src/tool_resource/clause_bridge.py`): maps runtime exec-image occurrences to static mvdan clauses (ordered occurrence + bin/argv evidence + PID lineage), so a same-PID exec chain (`env -> nice -> workload`) and its descendants aggregate into ONE clause observation keyed by the mvdan identity; unmatched execs/clauses become explicit coverage gaps and never update the KB. `ClauseResourceKB` now scores CPU-heavy from eligible `peak_cpu_cores > 2`, memory-heavy from eligible `sampled_peak_rss_mb > 500`, latency at the frozen 3500/5000 ms points, composed by the frozen three-valued per-target OR (bridge + KB unit-tested on synthetic Stage-2 records).

**The legacy SWE pacct / bash-xtrace per-binary replay is legacy development evidence, not canonical clause telemetry** — it must not be wired into the clause KB; the intended path replays SWE traces under the non-perturbing eBPF Stage-2 monitor.

**Done:** one end-to-end Docker smoke on a development-exposed SWE attempt command (`which python3 && python3 --version`, python:3.13-slim container) wired the live Stage-2 collector output through the bridge into the KB — 2/2 clauses mapped by exact argv, 0 mapping gaps, 0 ring-buffer loss; sub-second clauses correctly returned CPU `unavailable` and the KB produced latency=False / cpu=Unknown / memory=False. The collector also fixed the exec transition (pending seq at execve entry, promote on success, clear on failed exec) and the bridge fixed CPU-merge availability (unavailable for missing/inconsistent quota or no eligible merged window — no inf/0-ok fallback).

**Still pending:** the FULL SWE eBPF replay has NOT been launched; balanced-accuracy evaluation on development-exposed data is not yet run; the Docker-cgroup attach race (~150 ms) leaves early entry-shell samples unattributed (surfaced as coverage gaps, never dropped).

Do not restart conditional-q90 shrinkage, scheduler integration, or untouched confirmation before the Docker/replay smoke and the development-exposed balanced-accuracy evaluation are working.

## Evidence boundary

- SWE/fresh-277 results are development-exposed.
- Terminal-Bench seed-42 dev-100 is development-exposed.
- The remaining 139 Terminal-Bench attempts are untouched confirmation and must not be inspected until a final candidate and criterion are frozen.

## Task-contract rule

Every CC/CX/Hermes task that changes tool-resource data, prediction, evaluation, or scheduler integration must read this file first and include this lock in its task contract:

```text
Primary = three command-level dominant-type classifications and balanced accuracy.
q90 = q-error p90.
Composition candidate = per-clause flags ORed per target (frozen three-valued OR).
Runtime KB is clause-integrated via the Stage-2 bridge; full SWE eBPF replay
and balanced-accuracy evaluation remain pending.
```
