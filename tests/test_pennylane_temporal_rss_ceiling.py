from scripts.evaluation.evaluate_pennylane_temporal_rss_ceiling import (
    RssJob,
    simulate,
)


def test_temporal_packing_uses_complementary_profiles_without_exposure() -> None:
    jobs = [
        RssJob("a", (0.0, 5.0, 10.0), (9_000.0, 1_000.0, 1_000.0)),
        RssJob("b", (0.0, 5.0, 10.0), (1_000.0, 9_000.0, 9_000.0)),
        RssJob("c", (0.0, 5.0, 10.0), (8_000.0, 8_000.0, 8_000.0)),
    ]

    static = simulate(jobs, "static_peak")
    temporal = simulate(jobs, "temporal_oracle")
    unconstrained = simulate(jobs, "unconstrained")

    assert static["starts_before_first_completion"] == 1
    assert temporal["starts_before_first_completion"] == 2
    assert temporal["mean_task_completion_s"] < static["mean_task_completion_s"]
    assert temporal["maximum_sampled_aggregate_rss_mb"] == 10_000.0
    assert not temporal["capacity_violation"]
    assert unconstrained["capacity_violation"]


def test_terminal_sample_is_counted_before_completion() -> None:
    jobs = [
        RssJob("a", (0.0, 1.0), (1_000.0, 9_000.0)),
        RssJob("b", (0.0, 1.0), (1_000.0, 9_000.0)),
    ]

    unconstrained = simulate(jobs, "unconstrained")

    assert unconstrained["maximum_sampled_aggregate_rss_mb"] == 18_000.0
    assert unconstrained["capacity_violation"]


def test_terminal_and_internal_samples_are_aggregated_at_same_time() -> None:
    jobs = [
        RssJob("ending", (0.0, 1.0), (1_000.0, 9_000.0)),
        RssJob("continuing", (0.0, 1.0, 2.0), (1_000.0, 9_000.0, 9_000.0)),
    ]

    unconstrained = simulate(jobs, "unconstrained")

    assert unconstrained["maximum_sampled_aggregate_rss_mb"] == 18_000.0
    assert unconstrained["capacity_violation"]
