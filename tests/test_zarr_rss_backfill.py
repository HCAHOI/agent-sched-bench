from scripts.evaluation.evaluate_zarr_rss_backfill import (
    RSS_CAPACITY_MB,
    _composed_rss_source,
    _phase_raises,
    _upper_index,
    _validated_rss_source,
)


def _clause(rss, latency, *, pipeline_position=-1):
    return {
        "sampled_peak_rss_mb": rss,
        "latency_ms": latency,
        "in_pipe": pipeline_position >= 0,
        "in_subst": False,
        "pipeline_position": pipeline_position,
    }


def test_zarr_rss_source_composition_and_upper_reservation() -> None:
    assert _composed_rss_source(
        {
            "eligible_for_kb": True,
            "clauses": [_clause(None, 10), _clause(700, 1_000)],
        }
    ) == (700.0, "clause_short_null_upper")
    assert _composed_rss_source(
        {
            "eligible_for_kb": True,
            "clauses": [
                _clause(300, 1_000, pipeline_position=0),
                _clause(400, 1_000, pipeline_position=1),
            ],
        }
    ) == (700.0, "observed_clause_composition")
    assert _composed_rss_source(
        {"eligible_for_kb": True, "clauses": [_clause(None, 500)]}
    ) == (
        RSS_CAPACITY_MB,
        "full_fallback",
    )
    assert _upper_index((0.75, 0.25, 0.0)) == 1
    assert _phase_raises(1, 0, 0)
    assert not _phase_raises(1, 0, None)
    assert not _phase_raises(1, 0, 2)
    assert _validated_rss_source(
        {"eligible_for_kb": False, "clauses": [_clause(10, 1_000)]}
    ) == (RSS_CAPACITY_MB, "full_fallback")
    assert _validated_rss_source(
        {
            "eligible_for_kb": True,
            "invalid_reasons": ["lossy_mapping"],
            "clauses": [_clause(10, 1_000)],
        }
    ) == (RSS_CAPACITY_MB, "full_fallback")
