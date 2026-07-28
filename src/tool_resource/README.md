# Tool-resource services

This directory is self-contained: it imports nothing from the rest of this
repository and needs only `PyYAML` (client) plus the distribution's
`python3-bpfcc` (privileged telemetry only, imported lazily). Copy the whole
directory to use it elsewhere; the Go clause parser under `_mvdan_adapter/`
builds itself on first use. The offline lane that reads this repository's trace
files lives outside it, in `src/tool_resource_eval/`.

Internal lanes:

- **telemetry** — `telemetry.py` (BCC/eBPF collector), `telemetryd.py`,
  `telemetry_protocol.py`, `artifact_schema.py`, `clause_bridge.py`
- **knowledge base** — `runtime_kb.py`, `store.py`
- **parser** — `clause_parser.py`, `mvdan_client.py`, `_shell_split.py`,
  `_mvdan_adapter/`
- **service and client** — `resource_agentd.py`, `resource_protocol.py`,
  `client.py`, `profile.py`, `_uds.py`

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
sudo -n env PYTHONPATH="$PWD/src:/usr/lib/python3/dist-packages" \
  "$PWD/.venv/bin/python" -m tool_resource.telemetryd \
  --socket /tmp/tool-resource/telemetry.sock \
  --allowed-uid "$(id -u)" --socket-gid "$(id -g)"

PYTHONPATH="$PWD/src" "$PWD/.venv/bin/python" -m tool_resource.resource_agentd \
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
  latency_bucket_edges_ms: [2000, 8000]
```

Latency intervals are `[0, 2000]`, `(2000, 8000]`, and `(8000, +inf)`: an
exact boundary belongs to the lower bucket. Compound-command buckets are not
composed.

The serving default remains `raw-argv-prefix-v1`: repository exact/prefix/bin
evidence backs off to public bin/global evidence. Development evaluation may
construct the same `ClauseResourceKB` with `generic-argv-v3-role`. That arm
keeps raw repository exact keys, replaces ordered prefixes with one
privacy-preserving role signature, and adds the same signature to frozen public
evidence. Its stable subcommand vocabulary is derived without labels from at
least three distinct fit repositories. The selected representation and
canonicalizer version are carried by prediction provenance and KB snapshots;
there is no client-side predictor or per-command fallback.
