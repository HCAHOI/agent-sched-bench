# GLM OpenClaw Gantt Export Plan

## Scope

- Export only the 19-task cohort in `configs/simulate/openclaw-glm-19-manifest.json`.
- Require source traces to be `scaffold=openclaw`, `model=z-ai/glm-5.1`, and `max_iterations=100`.
- Exclude Claude Code traces and import paths.

## Implementation

- Add a static Gantt exporter that reuses the existing backend payload builder and frontend snapshot bootstrap.
- Add `python -m trace_collect.cli gantt-export` with preset `swe-rebench-glm-openclaw-100`.
- Produce standalone HTML files plus an export manifest under `results/gantt-viewer/`.

## Verification

- Unit-test metadata validation, group discovery, snapshot HTML construction, and frontend snapshot defaults.
- Run Gantt backend tests, frontend tests/build, targeted exporter command, and mandatory code review before finalizing.
