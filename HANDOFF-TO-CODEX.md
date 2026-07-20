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

## 2. Probe — footprint pricing screen

`scripts/analyze_pressure_headroom.py`, 288 insertions **uncommitted**.
Question: does per-call KV-footprint pricing beat the single fixed `kv_cost`
the certified policy applies across a 44× spread of real footprints?

The uncommitted diff is two things:
- **Reviewed and cleared** (md5 `a4164442`): ceiling-bug fix (both arms
  optimize over a common trigger domain `[0, kv+guard]`), three-way POWER RULE
  verdict, kv5000 secondary row, mechanism figure, per-fold λ̄, wording.
- **Unreviewed, unattributed**: `ProcessPoolExecutor` parallelism (import
  :165, `_score_fold`, fan-out ~:706, `--workers` ~:1229). Landed 11:53, after
  the review cleared the file.

**The full-corpus result has never been produced.** Two attempts died: one
killed on a wrong-PID misdiagnosis, one ran single-core for 86 min and was
killed.

Verdict rule is pre-registered three-way: `DROP` only if the kv3500 CI upper
bound < 156 s/277; `UNDERPOWERED` if the CI spans it; `PROCEED` if the CI lower
bound exceeds it. A PROCEED does not license deployment — it triggers a
separate certified decision replay. Spec:
`analysis/pressure-headroom-design-20260720.md` (read the amendment block at
the top; the three-way rule was added after partial numbers were visible and
that is recorded deliberately).

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

## 4. Verify — what to do, in order

1. **Equivalence check on the parallelism** (never run): same subset,
   `--workers 1` vs `--workers 8`, diff JSON excluding `provenance`, plus the
   `.json.zst` sidecar. Byte-identical or revert the parallelism.
2. **Apply two queued changes** not yet in the file: a `--final` assertion that
   `len(task_ids) == manifest["expected_task_count"]` (277), and a subset-run
   render stating the headroom is NOT per-277 and cannot be compared against
   the 156 s bar.
3. **Review the whole diff as one unit** — none of it is committed.
4. **Run `--final --workers 8`** and report the three-way verdict.

Invocation:
```
python scripts/analyze_pressure_headroom.py \
  --manifest analysis/fresh-corpus-certification-20260717/offline-gated-robust/manifest.json \
  --final --workers 8
```

Two rules that apply to all of the above: verify inputs against the manifest
that defines them (a lane recently ran on a superseded 50-trace root taken from
`excluded_trace_roots`), and a review the author commissions is not a review.
