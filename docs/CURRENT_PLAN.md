# Gantt Viewer Migration Plan

Source plan: `~/.claude/plans/iridescent-wondering-cosmos.md`

## Goal

Replace the static `src/trace_collect` Gantt HTML workflow with a dynamic
server under `demo/gantt_viewer/`, while preserving payload semantics and
supporting both:

- AC1: the single openclaw v5 trace under
  `traces/swe-rebench/smoke-20260407T121213Z/.../trace.jsonl`
- AC2: all 11 raw Claude Code traces under
  `traces/swe-rebench/claude-code-haiku/*/attempt_1/trace.jsonl`

## Current Execution Order

1. Phase 0
   Status: completed
   Create branch, add package skeleton, update packaging and make targets.
2. Phase 1
   Status: completed
   Move `src/trace_collect/gantt_data.py` to
   `demo/gantt_viewer/backend/payload.py`, delete legacy Gantt files, and
   replace CLI `gantt` with `gantt-serve`.
3. Phase 2
   Status: completed
   Implement FastAPI backend scaffold, discovery, and typed schema.
4. Phase 3
   Status: completed
   Add lazy Claude Code import cache and wire it into payload loading.
5. Phase 4+
   Status: partial
   Phase 4 frontend scaffold and a first Phase 5 canvas port are completed;
   deeper renderer parity and remaining chrome are still pending.

## This Checkpoint

Implement only through the Phase 0/1 checkpoint, then stop for human review.

Current checkpoint outcome:

- `dev/gantt-demo-server` branch created
- payload module moved to `demo/gantt_viewer/backend/payload.py`
- legacy static Gantt files and old Gantt-only tests deleted
- `gantt-serve` CLI interface reserved with a Phase 1 scaffold
- moved and directly affected tests updated to the new payload import path
- minimal verification passed

Phase 2 outcome:

- added `schema.py`, `discovery.py`, `app.py`, and `routes.py`
- implemented `/api/health`, `/api/traces`, `/api/traces/reload`, `/api/payload`
- discovery now finds AC1 + all 11 AC2 raw Claude Code traces
- `/api/payload` currently supports v5 traces and explicitly returns 501 for
  Claude Code traces until Phase 3 cache/import wiring lands
- added backend tests for discovery and routes
- verified real AC1 payload through FastAPI and matched it against
  `build_gantt_payload_multi(...)`

Phase 3 outcome:

- implemented `cc_cache.cache_key(...)` and `cc_cache.load_or_import(...)`
- wired `/api/payload` to auto-convert raw Claude Code traces on demand and
  then load the cached v5 JSONL through `TraceData`
- upgraded `sniff_format(...)` to skip Claude Code preamble records like
  `file-history-snapshot` and continue until it sees a recognizable record
- added cache and Claude Code route tests
- verified one real trace under
  `traces/swe-rebench/claude-code-haiku/*/attempt_1/trace.jsonl` through the
  API: first import returned HTTP 200 with `metadata.scaffold == "claude-code"`
  and the second call hit the cache path successfully

Phase 4 outcome:

- created the Solid + Vite + TypeScript frontend scaffold under
  `demo/gantt_viewer/frontend/`
- added a minimal health page that fetches `/api/health` through the Vite proxy
- installed npm dependencies and committed the generated `package-lock.json`
- generated `src/api/schema.gen.ts` from the backend OpenAPI schema
- verified the frontend with `npm run build`

Next checkpoint:

- Phase 5 progress:
  - added `CanvasRenderer`, `CanvasStage`, hit/layout/time helpers, state
    signals, and Solid chrome (`Header`, `TraceChipBar`, `Sidebar`, `Legend`,
    `Tooltip`)
  - app now fetches `/api/traces`, auto-loads the first v5 trace, and can load
    additional traces through `/api/payload`
  - `gantt-serve` in prod now mounts `frontend/dist`, and `gantt-serve --dev`
    starts a Vite dev server before the backend
  - browser smoke passed against `http://127.0.0.1:8765/`: the page loaded,
    AC1 auto-loaded, the chip bar listed all discovered traces, and the browser
    console was clean after adding a favicon

Remaining work after this checkpoint:

- finish deeper canvas parity (pin re-anchoring, drag-drop, upload flow,
  sidebar/scroll polish, broader multi-trace ergonomics)
- add frontend-side tests

## Verification For This Checkpoint

- `python -c "from demo.gantt_viewer.backend import payload"`
- `pytest demo/gantt_viewer/tests/test_payload.py`
- Any directly affected tests that still import the moved payload module

## Constraints

- No backward compatibility for the old static Gantt generator.
- Preserve payload behavior unless the migration itself requires API wiring.
- Do not change trace formats or `claude_code_import.py` behavior in this phase.
