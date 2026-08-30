# Materialized Benchmark Data

`data/` contains machine-local benchmark inputs. It is ignored by Git; a fresh
clone does not contain these files. Dataset definitions live in
`configs/benchmarks/`, while frozen evaluation cohorts live in
`configs/corpora/`. Neither is a substitute for checking the materialized data
before a run.

## Current Checkout

| Path | Materialized content |
|---|---|
| `swebench_verified/tasks.json` | 100 selected SWE-bench Verified task rows. |
| `swe-rebench/tasks.json` | 6,542 rows from the SWE-ReBench filtered split. |
| `science-agent-bench-verified/` | 102 verified ScienceAgentBench tasks, source trees, harness checkout, and `artifact-manifest.json` with revisions and file-level provenance. |

The SWE directories currently contain task metadata only; no `repos/` clone
tree is present. Current task-container runs obtain source from their benchmark
image; host clones are needed only by workflows that explicitly request them.

## Populate or Verify

```bash
make download-swebench-verified
make download-swe-rebench
```

Use `make setup-swebench-repos` or `make setup-swe-rebench-repos` only for an
analysis that explicitly needs host-side repository clones. For
ScienceAgentBench, treat `artifact-manifest.json` as the provenance receipt;
do not infer its revision from directory names.

Before evaluation, verify the expected task IDs against the chosen corpus JSON
and verify every referenced trace separately. Counts in this file describe the
current checkout, not a guarantee that ignored data has been restored on
another machine.
