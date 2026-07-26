# Tool-Resource Service Architecture Lock

**Effective 2026-07-26.** This document is the canonical runtime architecture
contract for tool-resource prediction and telemetry. It complements
`tool-resource-canonical-objective.md`: the objective lock defines prediction
and evidence semantics; this document defines module ownership, privilege,
persistence, IPC, lifecycle, and failure boundaries.

## Decision

The runtime consists of **two independently deployed modules with fixed
privilege boundaries**. They are not two permission modes and a run-time flag
must never change which process is privileged.

1. **`resource-agentd`** is always unprivileged. It owns command parsing and
   canonicalization, prediction, causal update, KB persistence, snapshot
   management, and telemetry-session orchestration.
2. **`telemetryd`** is always privileged. It owns container/cgroup resolution,
   BCC/eBPF lifecycle, runtime-event correlation, attribution, loss/cleanup
   checks, and finalized normalized observations.

`trace collect`, `simulate`, and future schedulers are thin unprivileged
clients. They connect only to `resource-agentd`; they do not import BCC, open
the KB database, start either service, or own either service's lifecycle.
`resource-agentd` communicates with `telemetryd` over a separate local IPC
contract.

```text
trace collect / simulate / scheduler
                |
                | Resource Protocol (UDS)
                v
        resource-agentd (unprivileged)
          | parser / predictor / KB
          | snapshot / causal updates
                |
                | Telemetry Protocol (UDS)
                v
          telemetryd (privileged)
          eBPF / cgroup / attribution
```

The words `observe`, `predict`, `learn`, `frozen`, and `causal` describe
requested behavior within this fixed architecture. They are not privilege
modes.

## Module ownership

### `telemetryd`

`telemetryd` owns only measurement and telemetry validity:

- resolve a supplied live container ID to its init PID and cgroup;
- attach, run, and clean up the eBPF collector;
- register a static call/clause plan supplied by `resource-agentd`;
- correlate runtime exec/fork/exit/perf events with calls and clauses;
- compute normalized latency, CPU, RSS, and I/O observations;
- report attribution status, event loss, collector health, cleanup status, and
  formal completeness;
- retain a finalized observation until acknowledged or its bounded lease
  expires.

It must not:

- parse commands into the KB representation;
- query or update the KB;
- know latency bucket boundaries or scheduler policy;
- open SQLite, DuckDB, or a user-owned result path;
- write arbitrary client-supplied paths as root;
- expose raw eBPF events outside the privileged process;
- replace or reinterpret a workload result.

### `resource-agentd`

`resource-agentd` owns all knowledge state and policy-neutral prediction
mechanics:

- parse commands and construct versioned canonical call/clause identities;
- pin an immutable predictor snapshot when a run opens;
- return pre-execution bucket predictions and provenance;
- orchestrate telemetry sessions through the Telemetry Protocol;
- join finalized telemetry with the command identity and workload result;
- apply final KB-ingest eligibility and causal-visibility rules;
- persist normalized observations and materialized predictor state;
- expose accepted/rejected coverage and reasons;
- export immutable snapshots for reproducibility and offline analysis.

It must not import BCC/eBPF, resolve cgroups directly, or gain root privileges.
It is the sole writer of the local KB store.

### Thin clients

A trace client owns workload execution, timeout, exit status, and container
lifecycle. It sends run/trace/call lifecycle messages to `resource-agentd` and
receives prediction and observation status. A resource or telemetry failure
must never replace its workload result.

## IPC and trust boundaries

Both protocols use bounded, length-prefixed JSON over Unix-domain sockets.
The existing transport framing may be reused, but the two semantic protocols
must have separate message types and error classes.

Every request and response contains:

```text
protocol_version
request_id
operation
payload
```

Session operations additionally contain a server-issued opaque session token.
Requirements:

- strict schema validation and rejection of unknown/incompatible protocol
  versions;
- bounded message size, finite timeout, request/response ID matching;
- `SO_PEERCRED` validation and socket owner/group/mode enforcement;
- idempotent request handling where a retry can repeat a completed operation;
- bounded session lease/TTL and deterministic `Finalize`/`Abort` semantics;
- no TCP listener in the canonical host-local deployment;
- after complete migration, no dual protocol or legacy operation aliases.

Suggested sockets:

```text
/run/agent-sched/telemetry.sock             root:tool-resource 0660
$XDG_RUNTIME_DIR/agent-sched/resource.sock  owning user/group  0600/0660
```

`telemetryd` accepts requests only from the configured `resource-agentd`
identity. It derives the target cgroup from the container ID and must not trust
a client-supplied cgroup path.

## Resource Protocol

### Open run

Input:

```text
OpenRunRequest
  run_id
  workspace_scope
  snapshot_id | latest_at_run_start
  latency_bucket_edges_ms
  update_policy = frozen | causal
  telemetry_requirement = best_effort | required_for_valid_evidence
```

Output:

```text
OpenRunResponse
  run_token
  pinned_snapshot_id
  canonicalizer_version
  store_schema_version
  capabilities
```

Bucket boundaries remain explicit and must satisfy the canonical objective.
`required_for_valid_evidence` affects final evidence validity; it does not
abort or rewrite workload execution.

### Open trace

Input:

```text
OpenTraceRequest
  run_token
  trace_id
  container_runtime
  container_id
  runner_pid (when available)
  repo/workspace metadata
```

`resource-agentd` converts this into an `AttachTargetRequest` to `telemetryd`.

### Begin call

Input:

```text
BeginCallRequest
  trace_token
  call_id
  command
  query_timestamp
```

`resource-agentd` parses and canonicalizes the command, queries the pinned KB,
and registers the resulting static call plan with `telemetryd` immediately
before workload execution.

Output:

```text
BeginCallResponse
  call_token
  prediction
  probability_by_bucket
  selected_scope
  fallback_path and reason
  evidence_count and recency
  pinned_snapshot_id
  canonicalizer_version
  telemetry_status
```

### End call

Input:

```text
EndCallRequest
  call_token
  workload_result
  end_timestamp
```

Output:

```text
EndCallResponse
  workload_result                  # unchanged
  finalized_call_observation
  telemetry_status
  ingest_status and rejection reasons
```

### Close trace/run

Closing a trace finalizes collector health, loss, cleanup, call coverage, and
formal completeness. Closing a run reports workload and evidence validity
separately and exports the run manifest required to identify its pinned
snapshot and result-affecting configuration.

## Telemetry Protocol

Canonical operations:

```text
Ping / Capabilities
AttachTarget
RegisterCall
FinishCall
RecordSafetyGuardBlock
FinalizeSession
AbortSession
FetchFinalizedObservation
AcknowledgeObservation
```

`AttachTargetRequest` identifies the run/trace and live container. The response
contains an opaque telemetry session token and resolved target diagnostics.

`RegisterCallRequest` contains a call ID, command digest, and static clause plan
constructed by `resource-agentd`. It does not give `telemetryd` a KB or bucket
configuration.

`FinalizedCallObservation` contains:

```text
observation_id
run_id / trace_id / call_id
call interval
per-clause normalized resource measurements
attribution validity and invalid reasons
collector/loss counters needed for call eligibility
```

`SessionSummary` contains collector health, formal completeness, eligible and
withheld call coverage, loss counters, and cleanup status. Raw ring-buffer
records never cross this protocol.

## Eligibility ownership

Eligibility is deliberately split rather than overloaded.

`telemetryd` determines **telemetry eligibility** from physical observation:
collector state, attribution, causal boundaries, event loss, and cleanup.

`resource-agentd` determines final **KB-ingest eligibility** by combining the
telemetry verdict with workload completion, target availability,
canonicalization, scope, duplicate detection, censoring, and causal visibility.

An invalid call is withheld, never converted to a negative observation. A
partial artifact contributes its valid calls. A collector/loss/cleanup failure
makes the affected observation unavailable and produces no KB update.

## Persistence and snapshots

The canonical local store is owned exclusively by `resource-agentd`.
The initial implementation uses SQLite in WAL mode plus an in-memory
materialized predictor. DuckDB/Parquet may be used for offline analysis and
snapshot export, not as the concurrent online owner.

Persist an append-only, versioned normalized observation envelope rather than
freezing the current internal KB node/trie layout as the disk schema. The
record includes at least:

```text
observation_id (unique)
run/trace/call/clause identity
workspace/repository scope
canonical command/clause representation
canonicalizer version
observation interval
normalized target measurements
telemetry and ingest eligibility
rejection reasons
provenance and ingestion sequence
```

Do not persist secrets, opaque random IDs, or arbitrary temporary paths in
canonical keys.

Delivery from `telemetryd` to `resource-agentd` is at-least-once with
idempotent ingestion. `observation_id` is unique in SQLite; a retry cannot
produce a duplicate update.

A run pins a snapshot at `OpenRun`:

- `frozen`: all predictions use the pinned snapshot; new observations enter an
  outbox and become visible only after the run closes;
- `causal`: completed observations may enter a run-local overlay, but only when
  `observation_end < query_start` and scope matches.

Pending, overlapping, future, and cross-scope observations are never visible.
Public/frozen evidence is immutable during a run.

## Failure semantics

- `telemetryd` unavailable/disconnected/timed out: existing KB prediction may
  still be returned; telemetry is unavailable; no KB update; workload continues.
- `resource-agentd` unavailable: prediction and telemetry are unavailable;
  workload continues and the run records service unavailability.
- protocol mismatch or malformed response: reject the interaction; do not
  guess, downgrade to fabricated metrics, or update the KB.
- service restart: persisted KB survives; active telemetry sessions become
  unavailable unless explicitly recovered by the same protocol version.
- `required_for_valid_evidence`: the run may finish its workload but its final
  evidence validity is failed if required telemetry is unavailable.

Workload execution, telemetry validity, and formal mapping completeness remain
separate fields at every boundary.

## Lifecycle and deployment

Services are managed independently of trace clients, initially by systemd and
later by a per-host DaemonSet if needed. `trace collect` and `simulate` only
connect. The canonical path must not auto-start a privileged service from a
worker.

The first implementation may retain one collector instance per telemetry
session behind `telemetryd`. A future shared host BPF program with a
`cgroup_id -> session_id` map is an optimization that requires separate
multi-session/loss-isolation evidence; it is not part of this migration.

## Canonical client configuration

Expose one existing-config-backed resource profile rather than a collection of
independent DB/socket flags. The profile resolves the resource endpoint,
behavior, pinned snapshot policy, bucket edges, and evidence requirement. The
endpoint is the only service address visible to trace clients.

Example conceptually:

```yaml
tool_resource:
  endpoint: unix:///run/user/1000/agent-sched/resource.sock
  behavior: observe_predict_learn
  update_policy: frozen
  snapshot: latest_at_run_start
  telemetry_requirement: best_effort
  latency_bucket_edges_ms: [...]
```

The behavior field does not alter process privilege.

## Migration contract

Implement in this order:

1. Define strict Resource and Telemetry Protocol types and falsifying tests.
2. Rename/canonicalize the privileged server as `telemetryd`; preserve the
   already validated eBPF line behind its new protocol.
3. Add SQLite observation storage, snapshot pinning, and an unprivileged
   `resource-agentd` server.
4. Move parser, prediction, causal update, and telemetry orchestration behind
   `resource-agentd`; reduce the public SDK to a thin Resource Protocol client.
5. Migrate `trace collect` and `simulate` to the resource endpoint.
6. Verify failure isolation and one real Docker -> resource-agentd ->
   telemetryd -> eBPF -> observation -> eligible KB-ingest path.
7. After all current callers are migrated, delete per-worker sidecar auto-start,
   in-process KB cold start/update, direct telemetry observer APIs, old server
   names, and compatibility aliases. Do not keep legacy support without a live
   consumer.

## Acceptance gates

The migration is complete only when all of the following are exercised:

- unprivileged trace client and `resource-agentd` import no BCC/eBPF module;
- only `telemetryd` resolves cgroups and owns collector lifecycle;
- only `resource-agentd` opens the KB database;
- two concurrent clients receive isolated run/trace/call sessions;
- repeated observation delivery is idempotent;
- frozen and causal snapshot visibility tests exclude future/overlapping/
  cross-scope data;
- telemetry startup, disconnect, timeout, loss, cleanup, and protocol mismatch
  never alter workload results and never update the KB;
- valid calls in a partial trace remain ingestible while invalid calls are
  withheld;
- a real privileged Docker/eBPF smoke produces a finalized normalized
  observation, clean shutdown, and one idempotent KB ingestion;
- service processes, sockets, temporary containers, and test artifacts are
  cleaned after verification;
- an independent review returns GO before changed code is used for scientific
  results.
