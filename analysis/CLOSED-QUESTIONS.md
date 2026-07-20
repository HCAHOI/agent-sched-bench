# Closed questions — what we tried, what it cost, why it died

> One entry per settled question. **Do not re-open without new data or a
> new argument that engages the stated reason.** Each entry names the
> surviving artifact holding the numbers; the design/process docs that
> produced them were deleted (recoverable from git) because they
> described plans that no longer exist and misled readers into thinking
> the work was live.

## Estimator conditioning

**Atoms as fit-time units — DEAD (2026-07-19).** Chained commands
(`cd X && make && pytest`) were decomposed into per-atom timings via
real shell instrumentation (bash xtrace, 8,953 segments re-executed on
our hardware). Naive atom identity loses outright (MAE 1022 vs the
certified trie's 996). Argument-conditioned per-atom tries (the strong
form) reach 974 — better than the certified config, still short of the
cd-normalized 971. Cause, measured: the heavy variable verbs almost
never earn argument-conditioned nodes (pytest served at bare-verb level
88% of the time, find 98%), so argument thinness binds exactly where
the tail lives. Independently, 46% of exec calls are pipelines/loops
where per-segment durations are ill-defined by construction, so no
atom model could ever cover half the workload.
*Artifacts:* `segment-atom-study-2026-07-19.{md,json}`.

**Stable-atom screening (pool an atom's samples across chain contexts)
— DEAD (2026-07-19).** A cross-fitted screen selecting heavy,
cross-task-stable atoms selects essentially one atom: apt-get qualifies
in all 5 folds, pytest in 1/5 (fold-unstable). The resulting node fires
on 29 of 4,824 held-out calls and lands on thin-trie calls only 9 times
— 0.19% of corpus mass. Died on arithmetic before any replay.
*Artifacts:* `stable-atom-overlap-2026-07-19.{md,json}`.

**Key normalization / wrapper transparency (WTN) — DEAD (2026-07-20),
and the most instructive failure.** An emergent screen (no token names
in logic) learned which leading wrapper segments carry no duration
information and dropped them from prefix keys. It worked as designed:
`cd` emerged transparent in every fold alongside echo/git/ls/which/
python3, mass-weighted fold stability 0.995, and it matched the
hardcoded cd-skip rule within noise (≤0.3 ms). It still died, twice
over. Stage 1: at the certified operating point no normalization beats
no-normalization on MAE — the previously-quoted "cd-skip wins by 25 ms"
was **confounded**, existing only jointly with a min_evidence=5 node
gate (−21 ms, CI [−28, −14]); normalization alone is MAE-worse and
tail-better. Stage 2 (certified replay): no cell certified and the
joint policy was directionally **harmful** (−59.6 s per 277 tasks at
kv3500), while the screen contradicted itself across duration sources
(cd transparent 5/5 folds on replayed timings, 4/5 on original) and
wrongly pooled apt-get — the class carrying H1's certified mass — in
every fold.
**The standing lesson: estimator-accuracy gains do not imply
decision-utility gains at the operating point. Only the replay gate
distinguishes them, and here it prevented shipping a regression that
looked like a 21 ms improvement.**
*Artifacts:* `wrapper-transparency-stage1-2026-07-19.{md,json}`,
`wrapper-transparency-stage2-2026-07-19.{md,json}`.

**Standing directive:** the hardcoded cd-skip rule is NEVER a shippable
method. It may appear only as an oracle-baseline row or a harness
positive control. Anything that ships must be screen-learned.

## Policy space

**Mid-call boundary evidence (re-condition survival when a sub-command
finishes) — DEAD (2026-07-19).** At each observed segment boundary we
compared residual survival conditioned on elapsed time alone vs elapsed
time plus boundary identity/index. Boundary identity changes **72.9%**
of re-check decisions but its paired log-score gain is −0.0137 nats
with a task-clustered 95% CI of [−0.41, +0.20]. Large behavioral churn,
no measurable information — the worst deployment profile there is, and
exactly what the cheap kill test existed to catch.
*Artifacts:* `boundary-evidence-stage1-2026-07-19.{md,json}`.

**More than one re-check (k>1) — CLOSED BY LEMMA + VERIFICATION
(2026-07-20).** Under the existing functional the policy space is one
irreversible action with "call still alive at t" as the only runtime
observable, so any elapsed-adapted k-check plan is precomputable at
call start and collapses to a single stopping time — which
`hazard_recheck_ms` already optimizes exactly. Verified rather than
assumed: an honestly-implemented k=2 DP attains the k=1 optimum on all
670 certified prior nodes × 10 kv cells, max value gap **0.0 ms**
(tol 1e-6), and is strictly dominated once per-check overhead is
priced. Together with the boundary kill above, the re-check dimension
is closed on both flanks — by lemma (elapsed) and by measurement
(boundary).
*Artifacts:* `adjudication-k2-recheck-2026-07-20.{md,json}`.

## Corpus limits

**Multi-tenant contention cannot be studied on fresh-277 — STRUCTURAL
NEGATIVE (2026-07-20).** Apparent concurrency in the corpus is the
collection harness's `--concurrency 2` flag: peak simultaneously-active
tasks 2, and the apparent occupancy tail is sub-millisecond read_file
calls colliding at tick boundaries (mean latency 2.3 ms, zero calls
above any headline threshold). Collection ran against a cloud provider,
so no shared KV cache existed to contend for. **Any contention claim
must come from the W5-7 harness, never from this corpus.**
*Artifact:* `pressure-headroom-design-20260720.md` (λ-honesty section).

**Terminal-Bench expansion — DROPPED (2026-07-20).** TB carries ~2×
the per-task swappable mass of fresh-277 (103 s vs 56 s at kv3500) and
a heavier tail, but the heavy-call *rate* is equal (~6%), and the
existing TB corpus ran GLM-5.2 against fresh-277's qwen3.7-max — so
every TB-vs-SWE difference conflates benchmark with agent model.
Buying more would spend budget we do not have on a confounded axis.
*Artifact:* `tb-sizing-memo-20260720.md`.

## Parked, not closed

**Per-call footprint pricing** — the lane is blocked and produced no
valid result; its ceiling guarantee is false and its kill criterion was
amended mid-flight. Both failures are recorded in
`pressure-headroom-design-20260720.md`. Nothing from it is citable.

**Candidate A (session-state-conditioned nodes)** — implemented,
unreviewed, never run; parked by priority call.
`scripts/analyze_session_state_cv.py` is labeled accordingly.
