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
  sandbox_backend: docker
  checkpoint_backend: walk
traces:
  - trace: /abs/path/task-a/attempt_1/trace.jsonl
    label: task-a
  - trace: /abs/path/task-b/attempt_1/trace.jsonl
    task_source: /abs/path/other-tasks.json
  - trace: /abs/path/task-c/attempt_1/trace.jsonl
    docker_image: custom/image:tag
  - trace: /abs/path/task-d/attempt_1/trace.jsonl
    sandbox_backend: fake
  - trace: /abs/path/task-e/attempt_1/trace.jsonl
    checkpoint_backend: verify
```

`sandbox_backend` defaults to `docker`. The `fake` backend is only for unit/CI
mechanism tests that must avoid Docker/KVM; it is not a valid evaluation
runtime.

`checkpoint_backend` defaults to `walk`, which preserves the full-tree walk
CAS checkpoint behavior. `overlay` scans Docker overlay2 upperdirs for
net-change checkpoints, and `verify` runs both `walk` and `overlay` on the same
turn and fails if their manifests differ. `overlay` and `verify` require Docker
overlay2 and do not fall back silently on other graph drivers.

For container-mode traces, each admitted attempt writes `container_startup.json`
with image-fix, container-create, and agent-bootstrap timing. Runtime container
stats remain in `resources.json`.
