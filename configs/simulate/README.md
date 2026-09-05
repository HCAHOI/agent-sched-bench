# Simulate Manifests

`trace_collect.cli simulate` now accepts a single YAML manifest via `--manifest`.
Trace paths in the manifest must be absolute, so reusable checked-in manifests are
usually not appropriate for machine-specific trace directories.

Simple form:

```yaml
- /abs/path/task-a/attempt_1/trace.jsonl
- /abs/path/task-b/attempt_1/trace.jsonl
```

Structured form:

```yaml
version: 1
defaults:
  task_source: /abs/path/data/swe-rebench/tasks.json
traces:
  - trace: /abs/path/task-a/attempt_1/trace.jsonl
    label: task-a
  - trace: /abs/path/task-b/attempt_1/trace.jsonl
    task_source: /abs/path/other-tasks.json
  - trace: /abs/path/task-c/attempt_1/trace.jsonl
    docker_image: custom/image:tag
```

Dependency-aware form:

```yaml
version: 1
defaults:
  task_source: /abs/path/data/bfcl-v4/tasks.json
traces:
  - trace: /abs/path/bfcl_traces/memory_prereq_0.trace.jsonl
    label: memory-prereq
  - trace: /abs/path/bfcl_traces/memory_eval_0.trace.jsonl
    label: memory-eval
    depends_on:
      - memory_prereq_0
```

The same dependency can live in the task source:

```json
[
  {"instance_id": "memory_prereq_0"},
  {"instance_id": "memory_eval_0", "depends_on": ["memory_prereq_0"]}
]
```

Dependencies use `trace_metadata.instance_id` / `task_source[].instance_id`.
When dependencies are present, simulation currently requires `--workers 1`;
multi-process worker waves are rejected because they cannot release children
immediately after each parent finishes.

For container-mode traces, each admitted attempt writes `container_startup.json`
with image-fix, container-create, and agent-bootstrap timing. Runtime container
stats remain in `resources.json`.
