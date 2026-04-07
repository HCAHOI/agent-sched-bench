# SWE-rebench Plugin Refactor — Review Audit Trail

**Branch:** `dev/swe-rebench-plugin`
**Commits:** c7188e0 → 4a410df (7 commits + the Phase 7 docs commit)
**Reviewer:** `oh-my-claudecode:code-reviewer` (opus) in a separate context
**Date:** 2026-04-07

## Commits reviewed

- c7188e0 [plan] REVISION 1 (ralplan consensus)
- a720182 [feat] Phase 1 — Benchmark plugin base + SWEBenchVerified port
- 4c5d164 [feat] Phase 3 — SWE-rebench plugin adapter
- efd222c [refactor] Drop v4 trace support, stamp v5 at write time
- 1d3901d [refactor] Phase 4 — Collector signature surgery + EvalTask rename + YAMLs
- 6b05ce4 [feat] Phase 5 remainder — setup scripts + Makefile targets
- 4a410df [fix] Phase 6 smoke — summary None-guard + smoke targets + gitignore

## Automated gates (all clean after Phase 7 fixes)

- `make test` → **207 passed, 2 skipped** (205 pre-review + 2 new
  openclaw_import v5 contract tests)
- `ruff check src/ tests/ scripts/` → 0 errors (3 unused imports
  auto-fixed during Phase 7)
- Grep guard for hardcoded dataset names outside plugin/config:
  - `princeton-nlp/SWE-bench_Verified` → only in `swebench_data.py`
    shim default config + `configs/benchmarks/swe-bench-verified.yaml`
    (both legitimate)
  - `SWE-bench_Verified` → same (no other matches)
  - `"swebench"` as harness namespace → only in `swebench_harness.py`
    (default helper arg, overridden by plugin at call time) +
    `swebench_data.py` shim default
- Grep guard for `from_swebench_instance` outside deletion sites → 0
  matches in src/ and scripts/
- Phase 6 smoke run: 4/4 traces (2 tasks × 2 scaffolds on SWE-rebench
  via dashscope/qwen-plus-latest) produced v5-compliant metadata with
  benchmark + benchmark_split fields

## Reviewer findings — ITERATE → APPROVE after fixes

Fresh `oh-my-claudecode:code-reviewer` (opus) in a separate context
returned **ITERATE** with 2 Major issues + 7 Minor observations.
Issues were fixed in the same Phase 7 docs commit. Post-fix state:

### Major issues (fixed)

**M1 — Trace contract broken on `import-openclaw` write path.**
`src/trace_collect/openclaw_import.py::_copy_trace_for_import`
hand-rolled a `trace_metadata` dict without `trace_format_version`,
`benchmark`, or `benchmark_split`. Any trace produced via
`python -m trace_collect.cli import-openclaw …` would raise
ValueError when loaded through the strict v5 reader.

**Fix:** added `trace_format_version: 5`, `benchmark`, and
`benchmark_split` to the hand-rolled dict. Added `benchmark=` and
`benchmark_split=` kwargs to `import_openclaw_run()` and
`_copy_trace_for_import()` with defaults `"swe-bench-verified"` /
`"test"` (historical use case). Added
`tests/test_openclaw_import.py::test_imported_trace_loads_under_strict_v5_reader`
and `test_imported_trace_accepts_benchmark_override` as regression
guards — both actually call `TraceData.load()` on the produced
trace and assert the metadata shape.

**M2 — `CollectedTaskResult.trace_file` type mismatch on openclaw
error path.** `src/trace_collect/collector.py:628` passed
`trace_file=str(dest_trace)` in the exception handler, but the
dataclass declares `trace_file: Path`. Every other construction
site passes a `Path`. Latent type drift.

**Fix:** changed `trace_file=str(dest_trace)` → `trace_file=dest_trace`
at `collector.py:628`. No regression test added (would require
mocking a failing SWEBenchRunner, outsized for a 1-char fix);
relying on the type hint + future mypy pass to catch regressions.

### Minor issues (fixed in the same commit)

**m1 — Dead v3 `type: "step"` normalization branch** in
`_normalize_openclaw_trace`. The upstream openclaw path emits
`type: "action"` records exclusively (post-v4 refactor), so the
step-tool-args branch at `collector.py:800-801` was dead code.
Deleted it.

**m2 — Unprotected `json.loads` in `_normalize_openclaw_trace`** at
`collector.py:795`. All other `json.loads(line)` calls in the
collector are wrapped in try/except JSONDecodeError. Wrapped this
one to match, so a partial-write source trace (e.g. from a crashed
openclaw run) no longer aborts normalization.

### Deferred to a follow-up (tracked, not blocking)

- **m3** `parse_simulate_args` still hardcodes `data/swebench_verified`
  paths. The simulate subcommand is out of scope for Phase 4; will
  be migrated when simulate gets its own refactor.
- **m4** Stale one-line docstring in `scripts/run_nanobot_eval.py`.
- **m5** Loose regex match in `tests/test_benchmark_registry.py:21`
  (style, not correctness).
- **m6** Informational note on lazy-import discipline in
  `agents/benchmarks/__init__.py` — no current bug, noted to lock
  in for future plugin authors.
- **m7** Audit trail "Reviewer findings" placeholder → now filled
  in (this section).

### Reviewer-confirmed strengths

1. P2 trace contract holds on all three canonical write paths
   (`log_metadata`, `_session_runner`, `_normalize_openclaw_trace`),
   verified by live-loading all 4 Phase 6 smoke traces through the
   strict v5 reader.
2. `build_runner` `NotImplementedError` guard is real and tested —
   future BFCL-v4 plugin must explicitly override.
3. Collector signature surgery is complete — zero leaks of the
   removed harness_* / repos_root / output_dir kwargs, with a
   positive test at `tests/test_cli_benchmark_flag.py:21-25`
   asserting `--harness-dataset` is parser-rejected.
4. `exclude_lite` handled with research integrity — YAML default
   false + prose comment + Python default false + logic honors
   the knob + test coverage both ways.
5. `agents.swebench_data` shim is thin and non-circular. The
   `make download-swebench-verified` chain still works end to end.
6. Documentation is accurate: `docs/benchmark_plugin_spec.md`
   (new) exists with full protocol, `docs/agent_benchmark_spec.md`
   (original env spec) preserved unchanged, README and CLAUDE.md
   reference the correct new path.
7. SWE-rebench's O2 "no JSON round-trip" honored consistently —
   native list end-to-end, pinned by `test_normalize_task_keeps_native_lists`.

## Acceptance

**APPROVED for merge to `main`** after the 2 Major fixes landed.
Post-fix automated gates:

- 207 tests passing (+2 new regression tests for M1)
- ruff clean
- No new failures introduced by the fixes
- Grep guards still clean
- Phase 6 smoke traces still readable through the strict v5
  reader

Reviewer verdict: ITERATE (with specific fixes) → APPROVE (fixes
verified). The refactor unblocks adding BFCL-v4 as a plugin with
zero collector changes — the Phase 4 §D3 litmus test passes.
