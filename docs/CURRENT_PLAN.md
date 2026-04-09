# Trace Spec Unification Plan

## Goal

Remove all legacy, compact, and pre-current-spec trace compatibility. The codebase
must assume every consumed trace already uses the current canonical schema.

## Steps

1. Plan file and implementation anchor
   Status: completed
   Scope:
   - Replace the stale plan with this migration plan.

2. Core trace-path cleanup
   Status: in_progress
   Scope:
   - Remove legacy trace-path fallback from OpenClaw async status.
   - Reword trace inspection and CLI messages around "current trace" instead of
     versioned user-facing terminology.
   - Delete obsolete migration scripts for retired layouts.

3. Gantt viewer input boundary
   Status: pending
   Scope:
   - Restrict discovery, register, upload, and payload flows to canonical traces only.
   - Remove raw Claude Code direct-ingest and its cache path.
   - Collapse public `source_format` API surface to a single canonical value.

4. Tests and docs cleanup
   Status: pending
   Scope:
   - Delete or rewrite tests that pin legacy fallback or multi-format behavior.
   - Update README and Gantt docs to require explicit Claude Code conversion first.
   - Regenerate OpenAPI snapshot and frontend API types if schema changes.

5. Verification and review
   Status: pending
   Scope:
   - Run targeted tests for trace collection, import, async status, and Gantt backend.
   - Spawn an independent strict reviewer sub-agent and address findings.

## Notes

- Keep the on-disk metadata field `trace_format_version` unchanged unless a
  removal is required by implementation safety.
- No backward-compatibility shims for old traces or old runtime state.
