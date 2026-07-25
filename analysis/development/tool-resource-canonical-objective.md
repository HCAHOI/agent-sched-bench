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
3. **eBPF Stage-2 windowed telemetry** (`src/trace_collect/clause_telemetry.py`, with validation artifacts under `analysis/development/clause-telemetry-ebpf-stage2-20260725/`): ~10 ms perf CPU-clock sampling in 500 ms wall windows for honest per-clause `peak_cpu_cores` (quota-clipped, never `cpu_ns/wall_ns`) and `sampled_peak_rss` (max aligned distinct-mm sum, never lifetime hiwater), plus total Linux task-I/O-accounting deltas (`read_bytes`, `write_bytes`, and `cancelled_write_bytes`) from exact exec/exit boundaries and zero baselines for newly forked TIDs. The bridge sums disjoint owned exec-image I/O totals but does not add Disk I/O to the three KB labels. Exec/exit snapshots, lineage attribution, coverage-gap records, per-TID provenance, and target-specific `unavailable` semantics are preserved. **Synthetic P/W/M/K/B and process-isolated two-cgroup runtime validation pass**;
4. a tested asymmetric Runtime KB with frozen public state, causal repo updates, serialization, evidence/fallback metadata, and monotonic-query guard;
5. **the Stage-2 -> Runtime KB clause bridge** (`src/tool_resource/clause_bridge.py`): maps runtime exec-image occurrences to static mvdan clauses (ordered occurrence + bin/argv evidence + PID lineage), so a same-PID exec chain (`env -> nice -> workload`) and its descendants aggregate into ONE clause observation keyed by the mvdan identity; unmatched execs/clauses become explicit coverage gaps and never update the KB. `ClauseResourceKB` now scores CPU-heavy from eligible `peak_cpu_cores > 2`, memory-heavy from eligible `sampled_peak_rss_mb > 500`, latency at the frozen 3500/5000 ms points, composed by the frozen three-valued per-target OR (bridge + KB unit-tested on synthetic Stage-2 records).

**The legacy SWE pacct / bash-xtrace per-binary replay is legacy development evidence, not canonical clause telemetry** — it must not be wired into the clause KB; the intended path replays SWE traces under the non-perturbing eBPF Stage-2 monitor.

**Done:** one end-to-end Docker smoke on a development-exposed SWE attempt command (`which python3 && python3 --version`, python:3.13-slim container) wired the live Stage-2 collector output through the bridge into the KB — 2/2 clauses mapped by exact argv, 0 mapping gaps, 0 ring-buffer loss; sub-second clauses correctly returned CPU `unavailable` and the KB produced latency=False / cpu=Unknown / memory=False. The collector also fixed the exec transition (pending seq at execve entry, promote on success, clear on failed exec) and the bridge fixed CPU-merge availability (unavailable for missing/inconsistent quota or no eligible merged window — no inf/0-ok fallback).

The formal simulator path now exposes `--tool-resource-telemetry {off,command,clause}`. Clause mode arms one cgroup-filtered collector per replay worker process, preserves the command envelope, disables the legacy segment timeline, supports concurrent replays with `workers=1`, and fails closed on integrity gaps. Failed `execve`/`execveat` attempts are recorded with pending sequence, argv, and errno; only command-lineage evidence may resolve one exact static clause to `no_runtime_exec`. A separate reviewed shell command-lookup contract requires anchored source/replay diagnostics, exact path-preserving executable-head and exit-code agreement, both tool-call IDs, and one exact unmatched static head. Exit 0 is accepted only for a unique nonfinal pipeline clause; it is not accepted for `|| true`, `; true`, or another later successful command. Attribution gaps persist their event identity, timestamp, PID/TID, exec sequence, entry-parent relation, fork parent, and reason. Only a sample whose TGID is exactly the long-lived container-agent entry parent (including its threads) is structural; every command descendant and every other process remains fatal.

Command-tree attribution now follows the full observed fork forest: an exec PID is a root only when it has no exec ancestor, fork-only intermediates collapse to the nearest exec ancestor, and one outside entry parent is accepted only for one connected command tree. A sentinel sample from a newly forked raw TID before that TID's first successful exec inherits to the nearest active exec ancestor only after one generation-unique, reverse-time-ordered fork chain reaches the authoritative entry parent. The raw event is unchanged; inherited-owner, fork-chain, candidate/rejected-record, timestamp-bound, and CPU-baseline/endpoint provenance is attached. Entry-only pre-exec setup is structural; missing/ambiguous/cyclic ancestry and disconnected trees remain fatal. Same-mm RSS is deduplicated and inherited CPU is added once. Successful-exec identity is retained from `sched_process_exit` through `sched_process_free` under a `(TID, task_struct*)` generation key: terminal scheduler samples retain identity but remain outside the half-open `[t_exec,t_end)` metric window, while a stale generation's free event cannot delete a reused TID's new state.

The mvdan adapter emits structured `&&`/`||` control scopes. A normally exited, uniquely mapped simple controller may resolve every exact executable in a skipped RHS subtree—including an entire RHS pipeline—to target-unavailable `no_runtime_exec`, but only when source and replay commands, visible tool results, and exit codes agree exactly. Negated or pipeline controllers, signals, unknown status, ambiguity, missing control evidence, and any contradictory RHS runtime evidence remain fatal. The final implementation passed 341 affected tests (340 in the normal user environment plus one Docker-socket test under root), six privileged BCC lifecycle/runtime tests, Ruff, Go formatting/vet/tests, and focused independent review with no critical or major findings.

A fresh fixed-image three-attempt smoke at 50×, concurrency 2, and one worker **passes all telemetry integrity gates**. All three attempts completed: 68/68 exec calls retained result and telemetry rows; 113/113 mappable clauses resolved into 104 observations and nine explicit `no_runtime_exec` rows; all 104 observations exposed available Linux task-I/O totals; source/replay commands and exit codes agreed for 68/68 calls; mapping gaps, relevant attribution gaps, ring loss, and reserve failures were zero. The three collectors used distinct cgroups and all reported clean shutdown. The causal fork bridge attributed 341 pre-exec samples exactly once; four terminal identity-only samples retained identity without extending metrics. Twelve entry-parent/setup samples were explicitly structural. Artifact: `traces/swe-rebench/qwen3.7-max/tool-resource-clause-stage2-concurrency2-fixed-images-pending-window-final-three/`.

Output bytes were exact for 49/68 calls. The remaining differences are recorded replay-fidelity caveats (for example durations, timestamps, ordering, and source host-state leakage), not silently treated as exact reproduction; every command and exit status still agreed. The smoke validates parsing, telemetry attachment, concurrency isolation, and fail-closed integrity for the actual fixed-image replay, not byte-for-byte reproduction of nondeterministic output.

**Still pending:** the FULL SWE-100 eBPF replay and development-exposed balanced-accuracy evaluation have not yet completed. The full replay may now launch at 50× with concurrency 2 and one worker, retaining the same per-attempt integrity gates and reporting replay-fidelity caveats separately.

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
