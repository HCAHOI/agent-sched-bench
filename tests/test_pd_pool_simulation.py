"""The three pool layouts route by role. Synthetic workload: timings here are not performance evidence."""
import pytest

from scripts.evaluation.pd_pool_simulation import HW, Sim

WORKLOAD = {"source_run": "synthetic", "concurrency": 2, "replacement_delay_mean_s": 0,
            "tasks": [{"task": f"t{i}", "steps": [{"prompt": 400 + 100 * s, "gen": 8, "gap_s": 0.0}
                                                  for s in range(3)]} for i in range(2)]}


def simulate(pool: str) -> Sim:
    sim = Sim(WORKLOAD, pool, dict(HW["l40s"]), agents=2, max_seqs=8, budget=2048, shared_prefix=0,
              seed=0, tasks_per_gpu=1, d_max_seqs=None)
    sim.run()
    return sim


def test_mixed_never_disaggregates():
    sim = simulate("mixed:2")
    assert sim.requests and all(r.p_engine is None for r in sim.requests)


def test_pd_disaggregates_every_step():
    sim = simulate("pd:1,1")
    assert sim.requests and all(r.p_engine in sim.P and r.d_target is None for r in sim.requests)


def test_ppd_disaggregates_turn_one_and_keeps_later_turns_on_the_home_pd():
    sim = simulate("ppd:1,1")
    first = [r for r in sim.requests if r.step == 0]
    later = [r for r in sim.requests if r.step > 0]
    assert first and later
    # Turn one prefills on P and its KV lands on the task's home pD; later turns never leave that pD.
    assert all(r.p_engine in sim.P and r.d_target is sim.mixed[0] for r in first)
    assert all(r.p_engine is None for r in later)
    assert all(r.task.sticky["mixed"] is sim.mixed[0] for r in sim.requests)


def test_unknown_layout_is_refused():
    with pytest.raises(AssertionError, match="pool must be"):
        simulate("residency:1,1")
