# Codex SWE-ReBench 223-Selection / 222-Trace Collection Goal

Status: ACTIVE objective lock
Owner: one Codex CLI agent in tmux session `cx-swe223`
Authorized by Chiyu: 2026-07-27
Repository: `/home/chiyu/workspace/agent-sched-bench`
Starting branch/head: `dev/kv-swap-profile-sweep-test` at `4dee7a396a14e80a70b76f6619b4422fade8b7cc`

## Goal

Select the fixed 223-task SWE-ReBench window and produce 222 accepted, real traces
using the repository's OpenClaw harness and Codex provider, with live canonical
eBPF clause-resource telemetry collected in the same execution. The one-task
difference is the declared backend attrition recorded below; it is not replaced.

Fixed experimental settings:

- benchmark: `swe-rebench`
- scaffold: `openclaw`
- provider: `codex`
- model: `gpt-5.6-sol`
- maximum iterations: `100`
- selection seed: `42`
- selection skip: `428` (zero-based; starts at shuffle ordinal 429)
- selected tasks: `223`
- accepted corpus target: `222`
- declared attrition: `gradio-app__gradio-3196`
- concurrency: `2`
- container runtime: Docker
- resource collection: enabled through the canonical tool-resource service/profile
- output run directory: `traces/swe-rebench/gpt-5.6-sol/seed42-skip428-n223-c2-ebpf`

The run is explicitly authorized even if it exceeds one night. It must be
resumable from the same run directory; never substitute a different selected
window, model, or provider.

## 2026-07-28 objective amendment — declared backend attrition

This amendment replaces the earlier requirement that all 23 pilot selections
be accepted. The original ordered 223-task selection remains fixed.

Evidence visible when the amendment was authorized:

- `cekit__cekit-474/attempt_4` had coherently completed before intervention:
  its manifest was `completed`, its result was successful with a 1,623-byte
  patch, and its resource artifact was `valid`/`ok` with 27 of 37 calls
  eligible. It is accepted.
- `gradio-app__gradio-3196` attempts 1–3 have terminal `error` manifests.
  Attempt 4 contains only its preserved partial `trace.jsonl`; it has no
  terminal manifest, result, or resource artifact.
- The SIGINT and SIGTERM affecting Gradio were explicit user cancellations.
  They require no external-reaper, old-shell, or broader cancellation
  investigation.

Gradio is declared backend attrition. Do not start or resume it, replace it,
or fabricate a terminal manifest. Preserve all of its failed and partial
artifacts. The pilot accounting is therefore 22 accepted of 23 selected. The
remaining 200 original selections make the accepted corpus target 222.

The explicit Codex backend retry fix is complete at `778c479`. Before the
remaining 200 may start, make only the agreed minimum telemetry raw-event /
collector retention fix, then pass focused tests, Ruff, one bounded review,
and a real eBPF memory smoke. A failed gate stops the operation. The Hermes
watchdog is observer-only and must not nudge or restart collection.

## Canonical context to reload before acting

Read and obey:

1. `AGENTS.md`
2. `analysis/development/tool-resource-canonical-objective.md`
3. `OPERATIONS.md`, especially the full service-backed tool-resource sequence
4. `src/trace_collect/CLAUDE.md`
5. this file again after any compaction or resumed agent session

This file is the operation-specific objective lock. Do not rewrite the goal, cohort, model, provider, concurrency, telemetry requirement, or gate based on intermediate results.

## Cohort correction and required proof

The initially suggested "start at the 378th item" is wrong for the actual existing manifests.

Observed against the current cached `nebius/SWE-rebench` data and repository manifests:

- `configs/corpora/swe-100.json`: 100 IDs; collection metadata says seed42 offset50.
- `configs/corpora/swe-277.json`: 277 IDs; collection metadata says seed42 skip150.
- old-manifest overlap: 0; old union: 377 IDs.
- the old union maps as far as shuffle index 427 under the current seed42 ordering.
- `skip=377, n=223` overlaps the old union by 50 IDs.
- `skip=378, n=223` overlaps the old union by 49 IDs.
- `skip=428, n=223` yields 223 unique IDs and zero overlap; first ID `pypa__cibuildwheel-1613`, last ID `stuartmaxwell__djpress-39`.

Before compute, independently reproduce the `skip=428` count, uniqueness, first/last IDs, and zero overlap using production benchmark selection code plus the two checked-in manifests. If any value differs, stop before collection and diagnose dataset/config drift; do not choose a new cohort ad hoc.

## Scope and decision rules

Do only work that directly protects:

- cohort identity and non-overlap,
- real Codex/provider execution,
- canonical resource telemetry validity,
- safe resumability,
- bounded image/disk lifecycle,
- the 23-selected / 22-accepted pilot followed by the remaining 200.

Do not add speculative abstractions, dashboards, reports, compatibility layers, benchmark-specific tuning, or unrelated cleanup. Reuse existing CLI, service, validation, and manifest paths.

Review is bounded: at most one focused reviewer pass after a non-trivial code change. A finding blocks only if it can change scientific validity, cohort identity, trace/resource integrity, resumability, data loss risk, or the feasibility of the authorized run. Ignore style preferences, hypothetical future needs, and unrelated pre-existing issues. Do not enter review/fix/re-review loops.

Commits are phase boundaries only when tracked source/config/documentation changed. Do not create ceremonial empty commits and do not commit traces, runtime state, logs, downloaded images, or scratch files. Never push.

## Phase 0 — Lock, inspect, and prepare

1. Confirm no other coding agent is editing this worktree.
2. Inspect Git status and preserve any pre-existing human work. At creation time only this goal file should be untracked/modified; if not, diagnose ownership before editing.
3. Commit this goal file alone with repository commit style so future sessions reload the exact objective.
4. Reproduce the cohort proof above.
5. Confirm current host prerequisites: Codex auth, passwordless sudo, Docker access via `sg docker`, BCC, cgroup v2, disk headroom, and no stale resource daemons/containers from this operation.
6. Record runtime paths in a short ignored runtime note or tmux environment, not another design document. Use a fresh persistent run-state directory outside Git (under `/home/chiyu/.cache/agent-sched-bench/`) so an interrupted multi-day run can resume; do not reuse stale `/tmp` state.

## Phase 1 — Fix concurrent image cleanup

Known blocker: in `_run_scaffold_tasks`, the `concurrency > 1` branch currently waits for `asyncio.gather` to finish all scheduled tasks before calling `_cleanup_task_images`. A 223-task run can therefore retain all source images until the end and exhaust disk. Current Docker state was 45 images / 67.02 GB; naive current-average projection for 223 images is about 332 GB versus 307 GB free. Shared layers may reduce this, but the existing lifecycle is unsafe.

Make the minimum root-cause fix:

- clean each scheduled task's source/fixed images immediately after that task reaches a terminal result;
- preserve result ordering and failure behavior;
- ensure each task is cleaned exactly once;
- do not add a refcount subsystem: the authorized 223-task cohort has 223 unique image references;
- add one focused regression test proving a fast task is cleaned while a blocked slow sibling is still running, and every task is cleaned exactly once.

Run the focused collector tests, the full test suite once because this changes shared attempt lifecycle, Ruff on changed files, `git diff --check`, and conflict-marker search. Commit only the exact fix/test. One bounded review pass is permitted under the blocking criteria above; no review loop.

## Phase 2 — Fresh services and 23-selected pilot

Use the existing canonical service architecture; do not invent an alternate resource path.

1. In tmux session `cx-swe223`, create inspectable windows named `telemetryd`, `resource-agentd`, and `collect` in addition to the Codex agent window.
2. Start `telemetryd` as root with fresh state and a fresh telemetry socket.
3. Start unprivileged `resource-agentd` with allowed UID 1000, GID 1001, a separate fresh resource socket, and the telemetry socket.
4. Create a runtime resource profile using the canonical behavior, causal update policy, snapshot policy, telemetry requirement, and latency bucket edges already specified by repository docs/objective. Keep it outside Git.
5. Launch collection under `sg docker` so the current long-lived login receives the already-configured Docker group membership.
6. Run the first 23 tasks with the fixed settings and exact shared run directory:
   - `--selection-seed 42`
   - `--skip 428`
   - `--sample 23`
   - `--concurrency 2`
   - `--max-iterations 100`
   - `--provider codex`
   - `--model gpt-5.6-sol`
   - `--container docker`
   - canonical `--tool-resource-profile`
   - explicit `--run-id traces/swe-rebench/gpt-5.6-sol/seed42-skip428-n223-c2-ebpf`

Do not call mocked APIs or accept synthetic traces as evidence. Task solve failures are legitimate benchmark outcomes and are not by themselves infrastructure failures.

### Pilot gate

Proceed to the remaining 200 only if all are true:

- exactly 23 selected unique IDs, all from the pre-verified 223 cohort and disjoint from both old manifests;
- exactly 22 accepted tasks have coherent terminal manifests/results, with
  Gradio retained only as the declared attrition above;
- real Codex calls and real task-container tool execution are present;
- each accepted task has its expected canonical trace and resource artifact;
- existing validators report no resource-integrity/cross-trace attribution failure and expected-call coverage is acceptable under the repository's canonical validity semantics;
- c2 produces distinct task/container/trace identities with no evidence of cross-talk;
- completed-task images are removed progressively and disk does not monotonically accumulate one image per completed task;
- resource daemons remain healthy;
- the run directory can be resumed without rerunning the 22 accepted tasks or
  scheduling Gradio.

If the pilot exposes a code bug, stop collection, fix only the root cause, add the smallest regression check, run affected tests, commit that fix, restart with fresh service state if required, and rerun the pilot. One bounded review pass total for that non-trivial fix; no infinite review.

If failure is environmental or credential-related, preserve artifacts, stop cleanly, and report the exact blocker rather than converting failed infrastructure attempts into scientific outcomes.

Codex credential note: at preflight the current access token expired at `2026-07-28T17:00:08Z` and a refresh token existed. The provider reloads credentials per request but does not prove autonomous refresh. Before scaling, verify that the expected remaining runtime is safe or establish a minimal proven refresh/resume path. Do not implement a speculative OAuth subsystem. On systemic 401/auth failure, stop promptly and preserve resumability; do not let many tasks become terminal auth failures.

After the gate, clean only transient pilot scratch/process leftovers. Preserve the shared run directory and persistent service/run state required for causal continuation.

## Phase 3 — Resume to 222 accepted (remaining 200)

Resume the same run directory and persistent service/run state using the same
fixed settings. Supply the original ordered 223-task selection with only
`gradio-app__gradio-3196` omitted. The collector's terminal-manifest resume
behavior must skip the 22 accepted pilot IDs and schedule exactly the remaining
200 IDs. This operational exclusion implements the declared attrition; it does
not change or replace the selected window.

Before allowing long execution, print and verify:

- 22 already terminal/skipped;
- 200 pending;
- 223 unique selected IDs and the exact one-ID Gradio exclusion;
- 222 unique accepted-target IDs;
- zero overlap with the old 100/277 union;
- same model/provider/max-iterations/concurrency/resource profile/run directory.

Keep the collector in the tmux `collect` window for human inspection. Monitor minimally: process/exit state, completed vs pending count, short error tail/counts, disk headroom, daemon health, resource artifact count, and repeated auth/rate-limit failures. Do not produce dashboards or verbose periodic reports. Do not cancel a healthy authorized run merely because it exceeds one night.

Stop and diagnose if there is a material validity/reliability signal: systemic auth/provider failures, daemon death, repeated resource-integrity failure, uncontrolled image growth, low disk guard, task/container identity cross-talk, or a stall materially beyond observed task duration. Ordinary benchmark solve failures are not a stop condition.

## Phase 4 — Completion and cleanup

At 222 accepted tasks:

1. Verify the exact 222-ID accepted set/count, its relation to the fixed
   223-task selection, the exact Gradio exclusion, and zero old-manifest
   overlap again.
2. Validate trace/resource artifact completeness and summarize collection-validity failures without hiding them.
3. Confirm images/containers from the run are cleaned and disk is stable; do not delete unrelated Docker state.
4. Gracefully close resource runs, stop operation-specific daemons/windows,
   and preserve the requested traces, Gradio failure/partial artifacts, terminal
   manifests/results, and any canonical learned snapshot needed by downstream
   analysis.
5. Remove transient scratch files and empty temporary runtime directories only after successful settlement; do not delete evidence.
6. Ensure Git is clean except for intentional, reviewed commits. Do not commit generated traces or push.
7. Leave a concise final summary in the Codex tmux window: commits, run
   directory, 22-of-23 pilot gate result, final accepted/attrition counts,
   resource validity/coverage, material caveats, and any resumable blocker.

## Autonomy boundary

This entire sequence—goal amendment, telemetry retention fix, tests, bounded
review, real eBPF memory smoke, gate evaluation, resume to 222 accepted,
monitoring, and final cleanup—is authorized. Proceed without asking about
routine derivable decisions. Stop rather than guess only if a choice would
change the fixed experimental semantics, spend a different cohort/model/provider,
risk data loss, or require an unrelated privileged/system configuration change.
