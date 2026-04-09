# Gantt Claude Auto-Import Plan

## Goal

Allow the Gantt viewer REST API to accept raw Claude Code session JSONL for
runtime registration/upload by reusing the existing Claude Code importer, so the
user can start the viewer without editing discovery config files and register
existing raw traces directly.

## Steps

1. Plan + scope anchor
   Status: completed
   Scope:
   - Confirm current viewer only accepts canonical traces.
   - Confirm the target 11 files under traces/swe-rebench/claude-code-haiku are raw Claude Code sessions.

2. Backend auto-conversion
   Status: completed
   Scope:
   - Add a backend helper that tries canonical sniff first.
   - If the file is a raw Claude Code session, call the existing importer
     (`trace_collect.claude_code_import.import_claude_code_session`) and then
     continue with the converted canonical output.
   - Reuse the same path for both `POST /api/traces/register` and
     `POST /api/traces/upload`.
   - Persist converted outputs under cache/runtime-managed storage; do not ask
     users to edit config.

3. Tests + docs
   Status: completed
   Scope:
   - Update backend route tests for register/upload auto-import behavior.
   - Keep canonical/legacy error handling intact.
   - Update Gantt docs/agent interface to describe the new REST behavior.

4. Review gate
   Status: in_progress
   Scope:
   - Spawn a strict independent reviewer sub-agent.
   - Fix any major issues before proceeding.

## Review audit trail

- First independent review result: **critical**
  - Issue 1: Claude sniff was too permissive and could mis-import Claude-like
    junk JSONL.
  - Interim fix: tightened Claude-specific evidence requirements and added a
    guard that imported traces must contain at least one canonical action.
- First review also flagged environment-sensitive OpenAPI drift and upload
  sidechain ambiguity.
  - Fix: documented that `upload` imports only the main session JSONL, while
    `register` preserves adjacent sidechains when present.
  - Fix: regenerated the checked-in OpenAPI snapshot + frontend schema types
    and simplified the frozen test to compare the stable API surface instead of
    brittle env-specific schema internals.
- User clarification superseded the first review's hardening goal:
  - This is an academic project; trace ingestion can assume valid inputs.
  - Final implementation intentionally removed Claude-like junk JSONL defenses
    and related tests, keeping only the two supported input classes:
    real Claude Code sessions and canonical standard traces.

5. Launch + load
   Status: pending
   Scope:
   - Start the viewer locally.
   - Register the 11 raw Claude Code traces through the REST API.
   - Verify `/api/traces` + `/api/payload` and provide access instructions.
