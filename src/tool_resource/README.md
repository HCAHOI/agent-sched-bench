# Tool-resource SDK

`tool_resource` is a local command library: cold-start it from valid Stage-2
telemetry traces, then let it parse, query, predict, observe, and update around
Docker-owned execution. `ToolResourceSDK` is unprivileged. A separate local
sidecar owns the eBPF collector and exposes finalized data over a bounded,
versioned Unix-domain-socket protocol; raw events never cross the socket.

```python
from pathlib import Path

from tool_resource import DockerExecutionContext, LatencyBuckets, ToolResourceSDK

# Boundaries are explicit and have no SDK default.
sdk = ToolResourceSDK.from_traces(
    cold_start_telemetry_paths,
    LatencyBuckets(tuple(configured_latency_edges_ms)),
)
context = DockerExecutionContext(
    container_id=container_id,
    container_executable="docker",
    repo=repo,
    artifact_path=Path("command-clause-telemetry.json"),
    sidecar_socket=Path("/run/tool-resource/sidecar.sock"),
)

run = sdk.start_command(context, tool_call_id, command)
actual = docker_runner.execute(command)
result = sdk.finish_command(run, actual, replay_execution="completed")
```

Cold start accepts valid artifacts independently, exposes accepted/rejected
artifact counts, eligible/withheld call coverage, and rejection reasons in
`sdk.cold_start_report`. It fails only when none of the supplied artifacts are
usable or no valid clause latency remains. Partial artifacts contribute only
their `eligible_for_kb=True` calls; withheld telemetry never enters the KB.

`run.prediction` is the pre-execution command bucket. `finish_command` first
finalizes the collector, then reads its artifact; only a completed replay with
valid collection, clean shutdown, intact telemetry, and an eligible command
enters the causal KB. The exact `actual` mapping is returned as
`result.workload_result`; telemetry failure cannot replace it.
`result.call_telemetry` is the per-command summary and
`result.telemetry_artifact` is the authoritative finalized artifact.
At artifact level, `replay_execution` is workload status,
`telemetry_quality` is collector health, and `formal_completeness` is
`complete`, `partial`, or `unavailable`; `call_coverage` reports the exact
eligible fraction.

The Docker runner owns execution, timeout, exit status, and container
lifecycle. The existing replay runner starts and stops a sidecar automatically.
Other runners start one locally before constructing the SDK context:

```console
sudo install -d -m 0750 -o root -g 1000 /run/tool-resource
sudo python3 -m tool_resource.sidecar_server \
  --socket /run/tool-resource/sidecar.sock \
  --socket-mode 0660 --socket-gid 1000
```

The sidecar resolves the init PID and cgroup from the supplied live container.
Commands must descend from one long-lived in-container runner process;
independent `docker exec` roots have no trustworthy fork ancestry and fail
closed. Socket unavailability, disconnect, or timeout marks telemetry
unavailable without replacing the Docker result or updating the KB.

Bucket intervals are `[0, b1)`, `[b1, b2)`, ..., `[bk, +inf)`. A compound
command returns `compound_command_uncomposed`; bucket IDs are never ORed or
combined. CPU and memory measurements remain internal and are not part of the
public prediction contract.
