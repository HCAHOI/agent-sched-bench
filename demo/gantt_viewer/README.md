# Gantt Viewer

Dynamic local viewer for benchmark traces.

This replaces the old static `trace_collect.cli gantt` HTML generator with a
FastAPI backend plus a Solid/Vite frontend under `demo/gantt_viewer/`.

## Current Status

Implemented:

- `GET /api/health`
- `GET /api/traces`
- `POST /api/traces/register`
- `POST /api/payload`
- `POST /api/traces/reload`
- `POST /api/traces/unregister`
- `POST /api/traces/upload`
- Lazy Claude Code import cache for raw session JSONL
- Solid/Vite frontend with:
  - `SYNC` / `ABS` time mode
  - `LAYERED` / `CONCISE` layout mode
  - `Load all` and per-trace load/toggle/remove controls
  - tooltip hover + pin
  - drag-drop upload
  - Ctrl/Cmd+wheel zoom

Partially implemented:

- canvas parity with the deleted static viewer is good on the main path, but
  there is still room to polish multi-trace ergonomics and deeper tooltip/scroll
  behavior

## Layout

```text
demo/gantt_viewer/
├── backend/
├── configs/
├── frontend/
└── tests/
```

## Prerequisites

- Python deps installed into `.venv`
- Node/npm available for the frontend

Recommended setup:

```bash
uv pip install --python .venv/bin/python -e ".[dev,gantt-viewer]"
make gantt-viewer-install
```

## Run

Development mode starts the backend and a Vite dev server:

```bash
make gantt-viewer-dev
```

Equivalent CLI:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m trace_collect.cli gantt-serve --dev
```

Production mode serves `frontend/dist` directly from FastAPI:

```bash
make gantt-viewer-build
PYTHONPATH=src:. ./.venv/bin/python -m trace_collect.cli gantt-serve
```

Use the example discovery config explicitly if you want the acceptance layout:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m trace_collect.cli gantt-serve \
  --config demo/gantt_viewer/configs/example.yaml
```

## Test

Backend pytest + frontend vitest:

```bash
make gantt-viewer-test
```

Browser smoke:

```bash
make gantt-viewer-smoke
```

Current smoke coverage:

- page boot
- default AC1 auto-load
- synthetic drag-drop upload
- button-triggered JSONL upload
- pinned tooltip after lane click
- `Load all` across the discovered trace set

For agent-driven workflows, use `demo/gantt_viewer/AGENT_INTERFACE.md`.

Frontend bundle check only:

```bash
make gantt-viewer-build
```

## Acceptance Checks

The shipped example config currently discovers:

- `1` v5 openclaw trace
- `11` raw Claude Code traces

Quick API check:

```bash
./.venv/bin/python - <<'PY'
from fastapi.testclient import TestClient
from demo.gantt_viewer.backend.app import create_app

client = TestClient(create_app())
traces = client.get("/api/traces").json()["traces"]
print("n_traces", len(traces))
print("n_v5", sum(t["source_format"] == "v5" for t in traces))
print("n_cc", sum(t["source_format"] == "claude-code" for t in traces))
PY
```

Expected today:

- `n_traces 12`
- `n_v5 1`
- `n_cc 11`

AC1 payload check:

```bash
./.venv/bin/python - <<'PY'
from fastapi.testclient import TestClient
from demo.gantt_viewer.backend.app import create_app

client = TestClient(create_app())
payload = client.post("/api/payload", json={"ids": ["ac1-12rambau__sepal_ui-411"]}).json()
print(payload["traces"][0]["metadata"]["scaffold"])
print(len(payload["traces"][0]["lanes"]))
PY
```

Expected:

- `openclaw`
- `1`

## OpenAPI Types

When backend schema changes, regenerate both the frozen snapshot and the
frontend types together:

```bash
./.venv/bin/python - <<'PY' > demo/gantt_viewer/tests/fixtures/openapi.snapshot.json
import json
from demo.gantt_viewer.backend.app import create_app
print(json.dumps(create_app().openapi(), indent=2, sort_keys=True))
PY

cd demo/gantt_viewer/frontend
npx openapi-typescript ../tests/fixtures/openapi.snapshot.json -o src/api/schema.gen.ts
```
