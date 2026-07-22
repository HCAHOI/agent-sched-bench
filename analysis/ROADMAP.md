# Roadmap

Only future work and decision gates belong here. Current evidence boundaries are
in [`CLAIMS.md`](CLAIMS.md); settled directions are in
[`CLOSED-QUESTIONS.md`](CLOSED-QUESTIONS.md).

## 1. Make W5 launchable

The multi-tenant harness and its development workloads exist, but no run is
ready to launch.

Required before any GPU execution:

1. Produce the configured Llama-3.1-8B prefill profile at
   `analysis/serving/w5-multitenant/prefill_result_llama31_8b.json` on the target
   hardware, with the exact serving configuration recorded.
2. Resolve the policy contract between the development trigger table, whose
   metadata encodes `rho=1.0`, and runtime accounting at the measured
   `rho=0.94`. Regenerate the table or approve an explicit contract; do not
   silently relabel it.
3. Complete the focused CPU tests and independent correctness review for the
   connector, pause/resume path, multi-tenant scheduler, pre-restore timing, and
   result provenance.

**Go gate:** every result-affecting input exists, the `rho` contract is
internally consistent, tests pass, and review has no blocking finding.

## 2. Run one minimal live development smoke

Run the smallest real-GPU case that can falsify the mechanism. It must show:

- actual memory pressure and co-tenant admission into freed blocks;
- retain, swap-out, restore, pause/resume, and fallback paths;
- bit-faithful resumed generation for the supported decoding mode;
- measured transfer, queue, JCT, TTFT, and GPU-memory telemetry;
- no missing request, task-order, policy-version, or input-provenance metadata.

This is a plumbing and mechanism check, not paper evidence.

**No-go gate:** stop if blocks are not truly freed, the resumed request is not
faithful, pressure is absent, or accounting cannot be reconciled with measured
serving events.

## 3. Freeze the adaptive deployment contract

Use completed-task updates only. For task `t`, predict from the initialized
profile plus outcomes from completed tasks `<t`; publish the update only after
`t` resolves. Per-call and same-repository updates remain excluded.

Fix before opening an evaluation stream:

- initialization corpus and fallback;
- task order and completion semantics;
- update rule, model-version/state-hash logging, and failure behavior;
- bounded canary allocation and the one-sided harmful revocation rule;
- primary systems comparison and stopping criterion.

The initial offline certificate covers the initialized policy only. Canary and
harm monitoring are safety mechanisms, not replacement certificates.

**Go gate:** one end-to-end development replay proves past-only causality and
complete state auditing. A final claim then requires an unopened chronological
stream; Fresh-277 is already development-exposed.

## 4. Run the minimum credible live comparison

Only after the prior gates pass, compare the fixed adaptive algorithm against:

- vanilla/fallback scheduling;
- deadline scheduling;
- a Continuum-style TTL baseline;
- ThunderAgent where the public implementation can be matched honestly.

Use at least three load levels and at least two real workload classes. Report
JCT, P99 TTFT, throughput, GPU-memory pressure, maximum wait, and separate
results for profile-defined light, middle, and heavy tool-call strata.
Throughput claims must come from the live stack, not simulation.

**No-go gate:** if the primary operating point is harmful or the harness does
not create memory pressure, stop and diagnose rather than expanding the matrix.

## 5. Validate pre-restore live

Offline pre-restore accounting is banked; live transfer contention is not.
Measure early restore on the same policy curve and charge the memory occupancy
created by a misfire. Promote it to the main system only if it improves the
predeclared live comparison without violating fairness or pressure accounting.
Otherwise retain it as an offline result and report the live negative.

## 6. Package the evidence

After the live gates settle, update `CLAIMS.md`, freeze exact configs and input
manifests, and produce the paper figures. Do not convert development screens,
smokes, or already-exposed streams into confirmatory evidence.
