# OpenClaw Haiku SWE-rebench Top-10 Trace Bundle

## Goal

- Run OpenClaw with `anthropic/claude-haiku-4.5` on the remaining 9 SWE-rebench
  tasks from the requested top-10 list.
- Reuse the existing completed Kinto trace from
  `traces/swe-rebench/anthropic-claude-haiku-4.5/20260409T142235Z-kinto384-openclaw-haiku-cc-aligned-100iter/`.
- Produce one new archive containing exactly 10 task trace directories by
  overlaying the existing Kinto trace with the new 9-task run.

## Fixed Decisions

1. Match the existing Kinto run parameters exactly:
   - provider: `openrouter`
   - model: `anthropic/claude-haiku-4.5`
   - scaffold: `openclaw`
   - prompt template: `cc_aligned`
   - max iterations: `100`
   - MCP config: `none`
2. Select tasks with explicit `--instance-ids`. Do not use `--sample`, because
   the collector loads the full Hugging Face split directly and its native order
   does not match the local `data/swe-rebench/tasks.json`.
3. Keep the original Kinto run untouched. Build the final 10-trace tarball from
   a temporary staging directory.

## Task Set

1. `Kinto__kinto-http.py-384` (reuse existing trace only)
2. `beeware__briefcase-817`
3. `devopshq__artifactory-255`
4. `googleapis__python-firestore-280`
5. `kobotoolbox__kobo-install-135`
6. `mozilla__bleach-259`
7. `python-graphblas__python-graphblas-217`
8. `tobymao__sqlglot-3425`
9. `tobymao__sqlglot-3848`
10. `wemake-services__wemake-python-styleguide-2343`

## Execution Steps

1. Run the review gate on the existing collection/runtime path before producing
   new experiment artifacts.
2. Run focused validation tests for the task-container OpenClaw path.
3. Launch one serial 9-task collection with a dedicated run directory under
   `traces/swe-rebench/anthropic-claude-haiku-4.5/`.
4. Verify the new run contains all 9 requested task directories and a
   `results.jsonl`.
5. Build a temporary 10-task staging directory by copying:
   - the existing Kinto task directory from the completed Kinto run
   - the 9 new task directories from the new run
6. Write a merged `results.jsonl` in the exact requested top-10 order.
7. Create a new `.tar.gz` from the staging directory and verify its contents.

## Acceptance Checks

- The reviewer finds no blocking issues in the experiment path, or all blocking
  issues are resolved before collection.
- Focused OpenClaw task-container tests pass.
- The new run contains exactly the 9 requested non-Kinto instance IDs.
- The final archive contains exactly 10 task directories plus one
  top-level `results.jsonl`.
- The final archive preserves the original Kinto trace and does not mutate the
  source run directories.
