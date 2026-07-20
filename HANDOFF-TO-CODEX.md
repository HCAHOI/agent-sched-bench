# Handoff — current stage only

Repo `/home/chiyu/workspace/agent-sched-bench`, branch
`dev/kv-swap-profile-sweep-test`. `source .venv/bin/activate`. 8 cores.
Machine is idle; nothing running.

## 1. Optimization

Merged (`fb51797`), identity proven — byte-identical artifacts after dropping
`provenance`, and 30,000/30,000 exact against a frozen `hazard_recheck_ms`:

| Script | State |
|---|---|
| `run_wtn_stage2` | parallel, 5.23× |
| `analyze_prerestore_accounting` | parallel, 2.88× |
| `analyze_boundary_evidence_stage1` | parallel, 2.87× |
| `adjudicate_k2_recheck` | parallel, 1.12× |
| `analyze_pressure_headroom` | parallel code present, **UNREVIEWED** — see §2 |
| `analyze_prior_calibration` | single-core |
| `replay_online_gate` | single-core |

Speedups measured on a 50-task subset. Dead ends, don't repeat: an exact
O(N log N) `hazard_recheck_ms` is **not** byte-identical (13/30,000 tie-break
divergences); the byte-identical batched version bought 6% because hazard is
not the bottleneck. Details in `CODEX-PERF-BRIEF.md`.

## 2. Probe — footprint pricing screen COMPLETE

`scripts/analyze_pressure_headroom.py` and
`tests/test_analyze_pressure_headroom.py` remain uncommitted with the reviewed
method, parallel fold scoring, fail-closed final-manifest pin, and subset
warning.

Acceleration was measured post-fix on the same 50-task subset:
- workers=1: 76.33 s
- workers=8: 51.38 s (**1.49×**)
- JSON was identical after removing only `provenance`; `.json.zst` sidecars
  were byte-identical.

Mandatory review found and fixed two major correctness issues before the run:
tool-level prior caches aliased distinct nodes, and `--final` accepted any
self-consistent 277-task manifest. Re-review then found contradictory POWER
RULE provenance/UNDERPOWERED wording; it was corrected and the gate was
declared clean. Relevant suite: 118 passed.

The reviewed full-corpus run completed in 2215.90 s:
- artifact: `analysis/pressure-headroom-2026-07-20.{json,md}`
- sidecar: `analysis/pressure-headroom-2026-07-20-decisions.json.zst`
- corpus: 277 tasks, 13,410 calls, 10 kv cells, 134,100 decision rows
- headline kv3500: 8.75 s/277, simultaneous CI [-85.00, 101.90] s/277,
  permutation inconclusive, fixed λ̄ 5740 ms, early-fire fraction 0.0595
- frozen bar: 156 s/277; CI upper bound is below it
- verdict: **DROP**, direction closed for self-footprint pricing
- kv5000 secondary: -31.11 s/277, inconclusive, non-binding

This result does not bound multi-tenant contention; that axis remains a
structural negative on this corpus. DROP does not trigger a certified decision
replay.

## 3. Online gate (W3-4) — built, statistics verified, corpus fix unreviewed

Three untracked files: `src/trace_collect/tool_latency_online_gate.py`,
`scripts/replay_online_gate.py`, `tests/test_tool_latency_online_gate.py`.
41 tests pass in ~15 s.

**What it is:** a sign-symmetry test martingale with predictable Kelly bets,
thresholded by Ville's inequality — an anytime-valid stopping rule tested
against the offline permutation gate on the same paired-delta statistic, at the
same Bonferroni tail (0.0025).

**Independently verified and sound.** A reviewer regenerated the null
false-certification rate across four adversarial nulls it built itself (Cauchy,
heteroscedastic, sign-dependent magnitude): exit rates 0.0015–0.0037 against
nominal 0.005. It also wrote a peeking off-by-one and confirmed the
predictability guard fails on it (and that the off-by-one is genuinely
anticonservative at 0.0109).

**Open, in order:**
1. A corpus fix was just applied and is **not re-reviewed**. The lane had been
   pointed at `traces/swe-rebench/qwen3.7-max/20260624T162037` — a superseded
   50-trace root listed in `excluded_trace_roots` — instead of the
   manifest-defined 100-task corpus `offline-gated-confirm-100-v2`.
   Terminal-Bench now uses a task-id list derived from
   `analysis/tool-time-frontier-terminal-bench-20260715/folds`.
2. **RETRACTED pending re-measurement:** "zero certifications, effective n
   1–19, e-value threshold 400 unreachable." Measured on 50 tasks of the wrong
   corpus. Re-measure at n=100 before restating it anywhere.
3. `--final` requires both surviving dev corpora (ScienceAgentBench is retired,
   corpus deleted). `replay_online_gate.py` is single-core.

## 4. Next

1. Commit the reviewed pressure-headroom code, tests, design status, and final
   JSON/Markdown artifact; keep the decisions sidecar local/gitignored.
2. Re-review the online-gate corpus fix as part of the complete online-gate
   diff. The old 50-task result remains retracted.
3. Measure/profile `replay_online_gate.py` before adding parallelism; preserve
   task-order and RNG semantics exactly if it is parallelized.
4. Re-measure the canonical 100-task SWE-ReBench development corpus and the
   pinned 83-task Terminal-Bench corpus before restating any certification,
   effective-n, or attainability claim.
5. Run online gate `--final` only after its mandatory review gate is clean.

Two rules still apply: verify inputs against the manifest or pinned task list
that defines them, and a review the author commissions is not a review.
