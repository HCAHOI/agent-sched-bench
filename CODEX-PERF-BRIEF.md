# Task: make the CPU verification experiments fast

Repo: `/home/chiyu/workspace/agent-sched-bench`, branch
`dev/kv-swap-profile-sweep-test`. Python env: `source .venv/bin/activate`
(uv-managed, Python 3.12). Machine: 8 cores.

## The problem

Offline analysis experiments take 3 hours to 6 hours each. They are pure
CPU, single-process, on existing data. This blocks the research loop — the
project lead waits hours per iteration. Make them fast.

## Hard constraint: outputs must not change

These scripts produced results that are already committed and cited. Any
optimization must be proven **byte-identical on real artifacts**, not merely
"tests pass".

Proof procedure (do this, don't substitute a weaker one):
1. Regenerate an artifact with your change and diff it against the committed
   JSON, ignoring only the `provenance` block. Example that runs in ~3 min:
   `python scripts/adjudicate_k2_recheck.py --final` vs
   `analysis/adjudication-k2-recheck-2026-07-20.json` (6,700 cells).
2. For anything touching `hazard_recheck_ms`, also fuzz against a frozen copy
   of the original implementation: ≥30,000 randomized cases across families
   {lognormal, uniform, bimodal, tie-clusters, duplicates, edge values at
   threshold and threshold±kv}, tiny N (1,2,3), zero and huge restore cost.
   Assert **exact** equality. Floating-point reassociation silently changes
   tie-breaking — see "What I already tried" below.

## What I already tried (do not repeat)

`src/trace_collect/tool_latency_profiled.py::hazard_recheck_ms` is O(N²): a
Python loop over up to 2N+2 candidate triggers, each allocating six N-element
numpy arrays. Microbenchmark: 2.5 ms/call at N=200, 18.5 at N=1000, 78.5 at
N=3000.

- An **exact O(N log N) prefix-sum reformulation** works mathematically and is
  20–80× faster in microbenchmark. It is **NOT byte-identical**: reordering
  the summation changes which member of a tied cluster wins. Measured 13
  mismatches in 30,000 cases; all were exact utility ties (both triggers
  optimal, gap 0.0), but a different trigger is a different policy on held-out
  data, so it cannot ship silently. If you want this speed, it must be an
  explicit, separately-approved behavior change with every dependent result
  re-run.
- A **batched-candidate version** (same arithmetic, (C,N) blocks reduced with
  `np.mean(axis=1)`, chunked to ~64K elements) IS byte-identical — verified
  0 mismatches in 40,000 fuzz cases. But on the real workload it only bought
  **6%** (3:03 → 2:52 on `adjudicate_k2_recheck.py --final`). I reverted it.

**The lesson, and your actual starting point: `hazard_recheck_ms` is not the
bottleneck on the real workload.** I optimized it based on a microbenchmark
without profiling the pipeline. Do not repeat that — profile first.

## Where to start

1. **Profile the real pipeline**, not a function in isolation. cProfile or
   py-spy on a subset run, then attribute cost per component. Candidates:
   trace loading/JSON parsing, the utility-matrix scoring in
   `src/trace_collect/tool_latency_utility_clock.py`, the bootstrap/permutation
   engine in `src/trace_collect/tool_latency_confirmation.py`
   (50,000 replicates), the per-fold prior construction in
   `tool_latency_profiled.py`, or repeated recomputation across (fold, kv) pairs.
2. **Parallelize what is provably independent.** No analysis script currently
   uses more than one core; the work is embarrassingly parallel over folds and
   kv cells. `scripts/run_offline_gated_robust_confirmation.py` already
   demonstrates the pattern (`--only-fold` / `--aggregate-only`). Requirements:
   results independent of worker count and completion order (aggregate by a
   stable sort key; never let float summation order vary), and RNG seeding
   semantics preserved exactly — if a parallel split would change draw
   sequences, keep that stage sequential.
3. Cache/memoize repeated work if profiling shows it (nodes are reused across
   calls; several scripts already key trigger caches by `(source, group_key)`).

## Scripts, slowest first

| Script | Observed runtime |
|---|---|
| `scripts/run_wtn_stage2.py --final` | ~6 h |
| `scripts/analyze_prerestore_accounting.py --final` | ~2–3 h |
| `scripts/analyze_boundary_evidence_stage1.py --final` | ~2.5 h |
| `scripts/analyze_pressure_headroom.py --final` | >75 min (est. was 15 min; wrong by >3x) |
| `scripts/adjudicate_k2_recheck.py --final` | ~3 min |
| `scripts/analyze_prior_calibration.py --final` | fast |

Common invocation:
`--manifest analysis/fresh-corpus-certification-20260717/offline-gated-robust/manifest.json --final`

`analyze_boundary_evidence_stage1.py` is known to be genuinely
hazard-dominated (py-spy showed ~1.1 events/s inside `hazard_recheck_ms`), so
the picture differs per script — profile each.

## Rules

- Do not change any statistic, criterion, threshold, or default.
- Do not modify `src/trace_collect/collector.py`, `cli.py`, or the benchmark
  plugin layer — those are collection code, unrelated.
- Type hints; fail fast; no new dependencies without asking.
- Run the full test suite for every module touched, plus
  `tests/test_tool_latency_profiled.py`, `tests/test_tool_latency_confirmation.py`,
  `tests/test_analyze_prerestore_accounting.py`, `tests/test_adjudicate_k2_recheck.py`.
- Do not touch `scripts/analyze_pressure_headroom.py` or its test — another
  change is pending there.
- Report measured before/after wall-clock per script and the artifact-diff
  result explicitly. If identity cannot be proven for some change, say so and
  leave it out.

## Reference material

- A previous attempt's diff (parallelism work, unreviewed, ~400 lines) is
  saved at `/home/chiyu/workspace/perf-lane-salvage/perf-lane-20260720.patch`.
  Reuse ideas from it if useful; it was never validated.
- Project engineering and research rules: `CLAUDE.md` at the repo root.
  Section "Runtime, Cost, and Patience" is the relevant one.
