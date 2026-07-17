# ML/MLSys Research Guidance

General operating principles for machine-learning-systems research. These are
field-level rules, independent of any particular project or dataset.

## 1. The artifact is the system, not the claim

Measurement exists to find mechanisms and guide the next build. An evaluation
that ends at "works / doesn't work" without answering WHY and WHAT TO CHANGE
is half a result. Mechanism analysis and case studies are the primary product
of an evaluation, not an optional appendix. When a result is negative or flips
between settings, the required response is a diagnosis, not just a verdict.

## 2. Match statistical rigor to the decision being made

Systems effects are often small, heavy-tailed, and cluster-correlated, so some
protection against shipping a harmful change is warranted — but the dose must
match the decision. The right instrument is usually a single uncertainty
estimate at the granularity of the actual deployment decision (e.g., one
cluster-aware interval or test per configuration that would really ship).
Avoid imported ritual from experimental natural science: multiple-comparison
ceremonies over parameter sweeps no deployment faces, protocol lock-in that
forbids learning from your own data, or procedures that convert "consistently
positive everywhere" into "no conclusion." A method that destroys information
is not conservative; it is broken. Explore freely, label exploration honestly,
and validate improvements on data that has not shaped them.

## 3. Velocity is a correctness property

Time-to-insight is part of research quality. Before launching any long
computation:

- Estimate wall-clock time and say it out loud; re-estimate when evidence
  contradicts the estimate.
- Parallelize everything provably independent (folds, shards, corpora,
  configurations). Sequential-by-default is a bug.
- Check memory-footprint × concurrency against the machine before launch.
- Attribute expected cost to each component of the run; any component that
  consumes a large share must justify itself explicitly.
- Instrument long runs with progress signals so "slow" and "stuck" are
  distinguishable.

## 4. Every configuration flag has an owner

Never inherit a template, default, or previous invocation unexamined. Each
parameter on a launched command line must have a stated reason to be there.
Parameters that contradict known physical reality or a standing decision must
be flagged and approved before launch, even if they look like harmless
plumbing.

## 5. Physical honesty outranks inferential ceremony

The integrity that matters most in systems work: measure real constants on
real hardware instead of assuming them; charge every cost the mechanism
actually incurs (nothing is free just because it is inconvenient to count);
evaluate on real workloads at realistic scale; report absolute numbers next to
real baselines rather than only relative improvements; publish the settings
where the method loses. An objective function that omits a real cost will
manufacture success and waste months.

## 6. Speak the audience's currency

Systems venues read multipliers and native units: end-to-end latency,
percentile tails, throughput, resource footprint. Robustness mechanisms are
sold as features demonstrated in native units (what breaks without it, and by
how much), not as statistical guarantees. Keep inferential detail to a short
methods note and an appendix. Write the paper the reviewers you will actually
get can champion.

## 7. Held-out data is a consumable

The moment a verdict is read from held-out data, that data is development-
exposed. Budget freshness like money: reserve untouched data for the single
decision that needs it, know which corpora are already spent, and never plan
experiments that require data or funds the project does not have.

## 8. The loop is measure → diagnose → improve → ship

Run it at maximum iteration speed. Sequencing discipline — diagnose on data
you have, validate the improvement on data it has not touched — replaces
prohibition. Any process step that slows the loop without changing a decision
should be deleted.
