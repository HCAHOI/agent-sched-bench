# ARM OpenClaw Batch Continuation

## Goal
Continue the validated ARM remote execution workflow on `root@1.94.166.57`
for the remaining SWE-rebench OpenClaw instances with **unchanged config**:

- provider: `openrouter`
- model: `z-ai/glm-5.1`
- benchmark: `swe-rebench`
- scaffold: `openclaw`
- container: `docker`
- mcp-config: `none`
- max-iterations: `100`

Previously completed successfully:
- `Kinto__kinto-http.py-384`
  - run dir: `traces/swe-rebench/z-ai-glm-5.1/20260411T114434`

Remaining queue:
1. `beeware__briefcase-817`
2. `devopshq__artifactory-255`
3. `googleapis__python-firestore-280`
4. `kobotoolbox__kobo-install-135`
5. `mozilla__bleach-259`
6. `python-graphblas__python-graphblas-217`
7. `tobymao__sqlglot-3425`
8. `tobymao__sqlglot-3848`
9. `wemake-services__wemake-python-styleguide-2343`

## Execution Strategy
- Use the existing CLI batch infra:
  - `--instance-ids a,b,c`
  - `--run-id <path>` for resumable runs
- Run the remaining 9 instances in one resumable batch run.
- If a blocker appears, fix it minimally, keep config semantics unchanged,
  and resume the same `--run-id` until the queue completes.

## Shared Command Template
```bash
python -m trace_collect.cli \
  --provider openrouter \
  --model z-ai/glm-5.1 \
  --benchmark swe-rebench \
  --scaffold openclaw \
  --container docker \
  --mcp-config none \
  --max-iterations 100 \
  --run-id traces/swe-rebench/z-ai-glm-5.1/batch-remaining-9 \
  --instance-ids <comma-separated remaining ids>
```

## Checkpoints
- Keep using the remote tmux session to preserve continuity.
- Record per-instance result path, trace path, and failure layer.
- Success criterion for each instance:
  - canonical `attempt_1/trace.jsonl` exists
  - `results.jsonl` entry exists
  - inspect result rather than trusting harness output blindly

## Non-goals
- Do not change benchmark/scaffold/provider/model/container semantics.
- Do not switch away from OpenClaw.
- Do not introduce persistent proxy config inside task containers.

## Review Notes
- Existing infra supports batch execution and resume.
- `load_completed_ids(run_dir)` only skips completed instances within the same
  run dir, so the remaining queue uses a dedicated stable `--run-id`.
