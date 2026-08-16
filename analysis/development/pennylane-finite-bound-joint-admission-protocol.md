# PennyLane Finite-Bound Joint Admission Protocol

**Status:** frozen before finite-bound outcomes
**Role:** development-exposed carrier test; not fresh or physical evidence

## Question

Can settled same-repository tasks convert the existing CPU/RSS bucket
predictions into finite, composable reservations that preserve liveness and
safe completion-time gain?

The preceding hard-page result is not repaired or reinterpreted: its candidate
deadlocked because RSS-High reserved the full 80 GB host. This is a new carrier
evaluated on the same already exposed 35-fit/35-replay lexical split, input
tree, public predictor inputs, cap-8 task queue, four LLM slots, and shifted
piecewise-constant action model frozen in
`pennylane-causal-joint-tool-admission-protocol.md`.

## Frozen finite-bound carrier

1. On the 35 settled fit tasks only, associate each eligible exec command's
   unambiguous canonical CPU and RSS label with the maximum task-profile CPU or
   RSS value observed during that exec action.
2. For each target and label, use the maximum observed action value. Apply a
   cumulative maximum from Low through High so a higher class never requests
   less. Cap the result at the frozen host capacity.
3. A class without fit support uses its canonical finite upper edge for Low or
   Medium and full host capacity for High. A replay command without a target
   prediction or eligible command row also uses full capacity for that target.
4. The unchanged Task-Aware hard bucket selects the corresponding fitted bound
   when each replay exec becomes ready. No replay label, future action, command
   family, parser change, quantile, safety threshold, or tuned margin is used.
5. `finite_fit_static` holds the request for the full exec.
   `finite_fit_feedback` replaces it only after a completed sampled segment,
   using the same causal feedback semantics as the reviewed hard-page
   evaluator.

The old `serial_tool` arm is the prediction-free safe-side baseline. The old
hard-page result remains a fixed negative reference. The exact cap-8 joint
oracle remains context only.

## Metrics and gate

Report fit support and fitted bounds by target/class, prediction availability,
completion and makespan, LLM/tool queue time, utilization and peaks, actual
capacity-violation seconds, reservation-underprediction seconds, overlap task
count, and any deadlock state.

`finite_fit_feedback` advances only if:

1. both it and `serial_tool` complete all 35 replay tasks with zero LLM, CPU,
   and RSS capacity-violation seconds;
2. its mean task completion is at least 5% below `serial_tool`;
3. its makespan is no higher than `serial_tool`;
4. at least two exec actions overlap across at least 20 replay tasks.

A liveness failure, replay exposure, or concentrated overlap is NO-GO. Do not
add a margin, quantile sweep, emergency bypass, or command-specific exception
after reading outcomes. A GO authorizes the separate dynamic task-admission
phase; otherwise CPU-only expansion stops and the next meaningful experiment
requires a physical resource action.

## Cost and provenance

Use the previously reviewed evaluator and add only the fit-bound carrier and
focused regression tests. Commit before formal evaluation. Run locally, then
recompute unchanged on `Ubuntu@216.81.248.69`; results must be byte-identical.
No new trace collection, LM call, GPU, service, or runtime integration is part
of this phase.
