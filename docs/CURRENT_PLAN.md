# Kinto Task Container Root Cause And Fix

## Goal

- Explain the real behavior divergence for `Kinto__kinto-http.py-384`.
- Implement a fix in OpenClaw so the agent sees the true project workspace in
  task-container mode while preserving the existing host-mounted state
  workspace for memory, sessions, and runtime artifacts.

## Findings

1. The shell-level probe environment is not the issue: both launch modes use
   `/usr/bin/python3`, both import `kinto_http`, and both pass
   `python3 -m pytest tests/test_session.py -v`.
2. OpenClaw task-container mode does use a different controller interpreter
   (`.venv/bin/python`), but that is not the direct source of the observed
   task behavior split.
3. The concrete divergence is in the OpenClaw system prompt: it currently tells
   the model that its workspace is the host-mounted runtime state directory
   instead of the task code workspace `/testbed`.
4. Tool execution is already correctly rooted in `/testbed`; the bug is the
   prompt/context workspace mismatch.

## Work Items

1. Add a distinct prompt-visible `project_workspace` to the OpenClaw
   runner/session/loop/context chain.
2. Keep memory, sessions, and custom skills rooted in the existing state
   workspace so runtime artifacts remain unchanged.
3. Update prompt generation so bootstrap files come from the project workspace
   first, with state-workspace fallback if needed.
4. Add focused tests covering prompt content and parameter forwarding.
5. Run the narrow validation suite.
6. Run a strict independent review pass before finalizing.
7. Commit and push the fix after review.
8. Produce a fresh single-task OpenClaw Haiku trace for
   `Kinto__kinto-http.py-384` using the patched code.

## Acceptance Checks

- In task-container mode, the system prompt reports `/testbed` as the project
  workspace.
- The system prompt still points memory/history/custom skills to the host
  runtime state workspace.
- Existing non-task-container behavior remains unchanged.
- New tests cover both prompt generation and forwarding of the project
  workspace through the evaluation stack.
- The fix is committed and pushed.
- A new OpenClaw Haiku trace exists for `Kinto__kinto-http.py-384` and can be
  inspected against the prior bad trace.
