---
name: memory
description: Two-layer memory system with grep-based recall.
always: true
---

# Memory

## Runtime-managed memory

Long-term memory (`MEMORY.md`) and the append-only event log (`HISTORY.md`) live
in the agent runtime directory, **outside the task workspace**. They are managed
automatically by the runtime:

- Long-term facts are loaded into your context automatically each turn.
- Old conversation turns are auto-summarized into `HISTORY.md` and key facts are
  extracted into `MEMORY.md` as the session grows. You do not need to manage this.

## Do not create memory files in the task workspace

Never create `memory/`, `MEMORY.md`, or `HISTORY.md` inside the task workspace.
Memory is not part of the task deliverable; writing memory files into the
workspace contaminates the repository and breaks evaluation diffs.

## Recall

Because history lives in the runtime directory (not the workspace), prefer
relying on the automatically-loaded long-term memory in your context. If you
need older detail that is no longer in context, ask the user or re-derive it
from the task files — do not attempt to grep a `memory/` directory in the
workspace.
