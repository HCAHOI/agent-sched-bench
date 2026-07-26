# Stage 1b — clause-telemetry eBPF, single frozen run (2026-07-24)

One pass of the full P/W/M/K/B matrix against `stage1b-fixture.json`
(amendments B-1..B-6, recorded with justifications in the fixture's
`stage1b_amendments`), executed 18:49:30 → 18:53:34 UTC, no top-up, no
mechanical retry consumed. Collector and mapper were unchanged from the
preceding development run. The main-run harness outcome was FAIL because B
had a workload-design defect; all other gates passed. B was redesigned under
amendment B-7 and passed one authorized B-only confirmation, so every case now
has a passing run. Large per-event row dumps were intentionally removed before
commit; this report retains the aggregate results and the checked-in harnesses
regenerate the raw output.

## Per case

**P — pipeline/loop workload (30 pairs): all gates PASS.**
- G1 semantic equivalence: 30/30 pairs.
- G2 identity: 9 resolved + 3 unresolved in every rep, exactly the declared
  shape (entrypoint `/bin/sh` unresolved-by-declaration, `printf` builtin
  no-exec, two same-argv sleeps candidate-listed unresolved).
- G3 CPU attribution: clipped target credit 10.200/10.200 s = 100.000%,
  wrong-clause 0, missing 0, ratios 0.909–0.968 (gate ≤ 1.15).
  Reconciliation exact in every rep: raw_exit == accounted +
  runtime_housekeeping (named category; 0.0880 s over 30 reps, all of it
  sentinel-seq fork-without-exec runtime-helper exits).
- G4 timing: sequential wall MAE 0.750 ms (gate 5 ms). Pipeline gate is the
  option-B structure check — chosen because interpreter startup is a jittery
  nuisance constant and a reconstructed startup-inclusive MAE target would
  re-introduce the spec fragility run 6 exposed. 30/30 reps pass presence,
  startup non-negativity, parallel overlap, and c1/c8 ordering. Reported:
  interpreter startup mean 23.5 ms (median 23.3, range 18.6–43.5); legacy
  pipeline MAE vs busy-only targets 23.5 ms (reported only, ≈ the startup
  constant, as predicted from run 6).
- G6 overhead: absolute median +4.50 ms — PASS (< 5 ms) but with little
  margin; p95 60.1 ms (docker launch jitter). Relative median +0.53%,
  bootstrap CI95 [−1.52%, +3.14%] REPORTED ONLY per amendment B-5: the
  docker-per-rep protocol cannot establish < 1% at n=30.

**W — exec chain (10 reps): GW PASS.** Every rep shows exactly 4 execs in
order `sh → env → nice → python3`. All 10 reps took the forked shape:
entrypoint `sh` on its own pid, the 3-exec tail chain on one child pid
(exec indexes 0,1,2); env/nice hold 0 CPU, terminal python3 holds the CPU.
Entrypoint shell CPU 16–40 ms reported, not gated. This confirms the B-4
operationalization and contradicts the "all four on the same PID" narrative
in the Stage-1b contract: the entrypoint exec sits on a separate pid in
practice.

**M — peak RSS (2×10 reps): GM PASS.** Peaks 74168–74436 KiB, all inside
the frozen band [65536, 90668]; live-VmHWM relative error ≤ 0.30% (gate 5%);
liveness marker 20/20.

**K — SIGKILL mid-burn (10 reps): GK PASS.** Exactly one python3 child per
rep, signal 9, nonzero CPU captured at exit, lineage parent present.

**B — background child: original workload GB FAIL (spec defect); redesigned
workload (amendment B-7) GB PASS 10/10 in a single confirmation pass.**

- Original run: all 10 reps observed 1 exec vs 2 —
  `python3 … & exit 0` makes container pid 1 exit immediately after the
  fork; docker tears the cgroup down and the background child is SIGKILLed
  ~150 µs later, before execve, in 10/10 reps. The collector captured
  exactly this reality (fork event, signal-9 exit with sentinel exec_seq,
  zero loss, balanced exec/exit); the strace oracle, whose ptrace overhead
  slows the container, shows the child *did* exec in the oracle lane. A
  background child cannot outlive container pid 1 — the frozen expectation
  was unreachable as designed. Workload-design defect, not telemetry.
- Amendment B-7 redesign (frozen before the confirmation numbers existed;
  GB gate logic unchanged): `( python3 -c '<busy 0.4s>' & sleep 0.1 );
  sleep 1`. The subshell is the parent — it backgrounds python3 and exits
  at ~0.1 s, after the child has reliably exec'd; the child outlives it and
  exits normally at ~0.4 s; pid 1's foreground `sleep 1` keeps the
  container alive past the child's exit. Parent still returns before the
  child finishes, so the background/outliving-child semantic is preserved.
- Confirmation run (19:06:53 → 19:07:28 UTC,
  one pass, no top-up): GB and G5 PASS, 10/10 reps, 4 execs each (sh,
  python3, sleep 0.1, sleep 1). Representative rep: child execs at 1.4 ms,
  parent exits at 102 ms, child exits at 428 ms with 428 ms CPU, pid 1
  alive to 1104 ms; lineage parent's exit precedes the child's exit in
  every rep. Mechanism integrity clean (0 loss, balanced, argv complete,
  attached before launch); oracle presence check passes.

## Mechanism integrity (all cases)

Across all 80 candidate repetitions: reserve failures 0, sequence failures
0, exec/exit balanced, argv truncation 0, attached-before-launch true,
cgroup configured. Oracle presence/absence comparison passes in every case
(oracle-only `env` is the strace wrapper). The scoped mechanical retry was
never needed.

## Verdict

Host-side eBPF process-lifecycle telemetry now has demonstrated end-to-end
support across all five cases P, W, M, K, and B: exact clause identity,
100% clipped CPU credit with a fully named accounting decomposition,
sub-millisecond sequential wall accuracy, structurally correct pipeline
timing with a measured ~23 ms interpreter-startup constant, in-band
peak-RSS capture, correct signal/lineage capture, and — after the B-7
redesign — correct observation of a background child that execs, outlives
its parent's exit, and exits normally while the container stays alive. Cost
is +4.5 ms median per container. The one caveat carried into any
production-integration decision: the < 1% relative-overhead claim is not
certifiable under the docker-per-rep protocol at n=30 — the absolute median
gate passes, but with limited margin (4.50 of 5 ms). Nothing else blocks.

## Cleanup and provenance

- BPF: `bpftool prog show` before and after each run shows no
  tracing/kprobe programs; only the host's pre-existing
  cgroup_device/cgroup_skb programs remain (a transient cgroup_device seen
  after the main run belonged to the host's `upower.service` and is gone).
  No `fable-stage1*` containers remain.
- Before cleanup, the preceding run artifacts and Stage-1b main-run artifacts
  were verified byte-identical after the B confirmation. The committed record
  intentionally keeps only the collector, mapper, two frozen fixtures, two
  harnesses, and this aggregate report. Large per-event JSON, privileged-command
  logs, bytecode, agent state, and superseded Stage-1 intermediates were removed
  under explicit cleanup authorization; they are reproducible by rerunning the
  checked-in harnesses.
- Amendments B-1..B-7 are recorded in `stage1b-fixture-bconf.json`
  (B-1..B-6 also in `stage1b-fixture.json`) with justifications tied to
  observed rows. B-6 (entrypoint-inclusive exec counts for M/K/B) extends
  the contract's five named items by applying the B-1 declared-entrypoint
  principle and is flagged as such. B-4 records the observed
  forked-entrypoint shape as a correction to the contract's same-PID
  narrative. B-7 (B-workload redesign) was frozen before its confirmation
  numbers existed.
