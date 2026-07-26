# Tool-resource services

The canonical path has two independently managed Unix-domain-socket services:

- `telemetryd` runs as root and alone resolves container cgroups and loads BCC/eBPF.
- `resource-agentd` runs without root privileges and owns parsing, prediction,
  SQLite-WAL observations, snapshots, and causal updates.

Trace collectors and simulators open one `ResourceRun` per exact repository
(or task-unique scope when no repository exists), then open all `ResourceTrace`
sessions inside that pinned run. They close the run only after every scoped
trace finishes and persist its complete manifest under `tool_resource_runs/`.
Clients connect only to `resource-agentd`; they never start either service or
import the collector.

```console
sudo resource-telemetryd \
  --socket /tmp/tool-resource/telemetry.sock \
  --allowed-uid "$(id -u)" --socket-gid "$(id -g)"

resource-agentd \
  --socket /tmp/tool-resource/resource.sock \
  --database /tmp/tool-resource/tool-resource.sqlite3 \
  --telemetry-socket /tmp/tool-resource/telemetry.sock
```

The default research setup runs `resource-agentd` and its local clients as the
same unprivileged user. Clients validate the daemon against the resource socket
owner; `telemetryd` remains the only root/BCC process.

Both `trace collect` and `trace collect simulate` accept
`--tool-resource-profile`. Omitting it disables the service path. A profile
contains the only client-visible endpoint and all result-affecting resource
configuration:

```yaml
tool_resource:
  endpoint: unix:///run/agent-sched/resource/resource.sock
  behavior: observe_predict_learn
  update_policy: frozen
  snapshot: latest_at_run_start
  telemetry_requirement: required_for_valid_evidence
  latency_bucket_edges_ms: [...]  # explicit pre-registered values; no default
```

Latency intervals are `[0, b1)`, `[b1, b2)`, ..., `[bk, +inf)`: an exact
boundary belongs to the higher bucket. There are no implicit edges and
compound-command bucket IDs are not composed.
