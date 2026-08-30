# Trace Map

`traces/` is machine-local working storage and is ignored by Git. This file is
a short map of the retained collections in the current checkout, not an
automatically generated inventory. A directory name or task count does not
establish evidence validity; check the result receipt and per-attempt artifacts.

## Main Retained Collections

| Path | Role in this checkout |
|---|---|
| `swe-rebench/gpt-5.6-sol/pennylane-all76-clean-ebpf-20260816/` | Cleaned expanded PennyLane collection: 76 task directories with one retained successful attempt each. Used as source traces by later mixed-workload manifests. |
| `../outputs/traces/sqlglot-200-gpt-5.6-sol-c2-fast-ebpf/` | Consolidated expanded SQLGlot collection: 200 task directories assembled from multiple collection runs without exposing run boundaries. |
| `../outputs/traces/sqlglot-200-gpt-5.6-sol-c2-fast-ebpf.tar.zst` | Shareable SQLGlot-200 archive; verify with the adjacent `.sha256` file. |
| `../outputs/traces/mixed128-poisson-unique-baselines-20260827.tar.zst` | Inputs and physical outputs for the Unique-128 baseline comparison; receipt: `analysis/results/mixed128-poisson-unique-baselines-20260827/result.md`. Verify with the adjacent `.sha256` file. |
| `terminal-bench/tb-all/canonical/` | Working Terminal-Bench canonical pool; currently 239 task directories. This is larger than either retained 100-trace release archive. |

## Frozen Development Corpora

| Corpus definition | Referenced local root | Current materialization |
|---|---|---|
| `configs/corpora/swe-100.json` | `swe-rebench/qwen3.7-max/offline-gated-confirm-100-v2/` | 99 task directories for 100 declared IDs; incomplete until the missing task is restored and validated. |
| `configs/corpora/swe-277.json` | `swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200/` | 275 task directories for 277 declared IDs; incomplete. |
| `configs/corpora/swe-sqlglot-48-gpt56-ebpf.json` | No single root declared | Frozen 24/24 fit/evaluation protocol; use its recorded evidence boundary rather than inferring a split from the SQLGlot-200 bundle. |

## Retained Release Archives

`trace_archives/` currently contains:

- `swe-rebench-qwen3.7-max-100-traces.tar.zst`
- `terminal-bench-zai-org-GLM-5.2-100-traces.tar.zst`

These tracked archives are compact releases, not mirrors of every expanded
working directory. They do not have adjacent checksum receipts in this
checkout; verify them after transport and do not silently substitute them for a
corpus whose model, task IDs, or telemetry contract differs.

## Validity and Retention

For collection resume, a `completed` or `exhausted` attempt is accepted only
under the resource-evidence rules in `OPERATIONS.md#resume`. For scientific use,
also verify the trace schema, requested telemetry, task IDs, and the result
artifact's evidence boundary.

Unlisted directories under `traces/` include development, interrupted, invalid,
and superseded runs. Do not use or delete them based on their names. First tie a
run to a tracked result receipt or confirm that a retained archive contains the
same required evidence.
