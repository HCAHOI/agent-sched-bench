from __future__ import annotations

import json

import pytest

from tool_resource.runtime_kb import (
    CANONICAL_LATENCY_BUCKET_EDGES_MS,
    GENERIC_ARGV_CANONICALIZER_VERSION,
    ClauseObservation,
    ClauseResourceKB,
    LatencyBuckets,
    generic_argv_keys,
)


def _obs(
    repo: str,
    bin_: str,
    argv: tuple[str, ...],
    start: float,
    end: float,
    *,
    latency_ms: float | None = 100.0,
    cpu: float | None = None,
    rss: float | None = None,
    disk: float | None = None,
    impute_short_null: bool = False,
) -> ClauseObservation:
    return ClauseObservation(
        repo=repo,
        bin=bin_,
        argv=argv,
        ts_start=start,
        ts_end=end,
        latency_ms=latency_ms,
        peak_cpu_cores=cpu,
        sampled_peak_rss_mb=rss,
        disk_read_write_bytes_total=disk,
        impute_short_null_resources_as_light=impute_short_null,
    )


def _fit(*observations: ClauseObservation) -> ClauseResourceKB:
    return ClauseResourceKB.fit_public(observations)


def _clauses(*specs: tuple[str, list[str]]) -> list[dict]:
    return [{"bin": bin_, "argv": argv} for bin_, argv in specs]


def _generic_exact(bin_: str, argv: tuple[str, ...]) -> str:
    return generic_argv_keys(bin_, argv)[0][1]


def test_generic_argv_collapses_opaque_values_and_has_explicit_version() -> None:
    first = _generic_exact(
        "runner",
        (
            "/usr/bin/runner",
            "deploy",
            "--path=/tmp/build-1",
            "https://example.test/jobs/1",
            "123e4567-e89b-12d3-a456-426614174000",
            "deadbeef1234",
            "1500",
        ),
    )
    second = _generic_exact(
        "runner",
        (
            "runner",
            "publish",
            "--path=./build-2",
            "s3://other/jobs/2",
            "987e6543-e21b-12d3-a456-426614174999",
            "cafebabe5678",
            "9000",
        ),
    )

    assert GENERIC_ARGV_CANONICALIZER_VERSION == "generic-argv-v2-shape"
    assert first == second
    assert first.split("\x00") == [
        "runner",
        "<ARG>",
        "--path=<PATH>",
        "<URL>",
        "<ID>",
        "<ID>",
        "<NUM:+E3>",
    ]


def test_generic_argv_preserves_option_shape_without_plaintext_values() -> None:
    keys = {
        _generic_exact("tool", ("tool", subcommand, first, second))
        for subcommand, first, second in (
            ("fetch", "--mode=fast", "target"),
            ("push", "--other=fast", "target"),
            ("fetch", "target", "--mode=fast"),
        )
    }
    redacted = _generic_exact(
        "tool",
        (
            "tool",
            "HOME=/private/work",
            "--password=hunter2",
            "--api-key",
            "split-secret",
            "--mode=stable",
            "-phunter2",
            "-H",
            "Authorization: Bearer private-token",
        ),
    )

    assert len(keys) == 3
    assert redacted.split("\x00") == [
        "tool",
        "HOME=<PATH>",
        "--password=<ARG>",
        "--api-key",
        "<ARG>",
        "--mode=<ARG>",
        "-p=<ARG>",
        "-H",
        "<ARG>",
    ]
    for secret in ("private", "hunter2", "split-secret", "private-token"):
        assert secret not in redacted


def test_generic_argv_keeps_numeric_order_of_magnitude() -> None:
    small = _generic_exact("tool", ("tool", "-j2", "5"))
    large = _generic_exact("tool", ("tool", "-j64", "600"))
    nearby = _generic_exact("tool", ("tool", "-j70", "900"))

    assert small != large
    assert large == nearby
    assert large.split("\x00") == ["tool", "-j=<NUM:+E1>", "<NUM:+E2>"]
    assert _generic_exact("tool", ("tool", "1e1000000000000000000")).endswith(
        "<NUM:EXTREME>"
    )


def test_latency_buckets_match_strict_threshold_decisions() -> None:
    buckets = LatencyBuckets((100.0, 1000.0))

    assert [buckets.bucket_id(value) for value in (0.0, 99.999, 100.0)] == [
        0,
        0,
        0,
    ]
    assert buckets.bucket_id(999.999) == 1
    assert buckets.bucket_id(1000.0) == 1
    assert buckets.bucket_id(1e12) == 2
    assert CANONICAL_LATENCY_BUCKET_EDGES_MS == (
        500.0,
        1000.0,
        2000.0,
        4000.0,
        8000.0,
        16000.0,
        32000.0,
        64000.0,
    )


@pytest.mark.parametrize(
    "edges",
    [
        (),
        (0.0,),
        (-1.0,),
        (100.0, 100.0),
        (100.0, 50.0),
        (float("inf"),),
        (True,),
        ("100",),
    ],
)
def test_latency_buckets_reject_invalid_edges(edges: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        LatencyBuckets(edges)


@pytest.mark.parametrize("latency", [-1.0, float("inf"), float("nan")])
def test_latency_buckets_reject_invalid_values(latency: float) -> None:
    with pytest.raises(ValueError):
        LatencyBuckets((100.0,)).bucket_id(latency)


@pytest.mark.parametrize("latency", [-1.0, float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_invalid_latency_fails_closed_at_any_position(
    latency: float, position: int
) -> None:
    """An unusable latency must be refused wherever it sits in the node.

    Nodes are held sorted so predictions can bisect them, and NaN compares
    false against everything, so it can land anywhere in that order. Checking
    only the ends of a node would let a NaN in the middle through and yield a
    PMF that silently counts it as the lowest bucket.

    Refusal happens as the value enters the node. That is deliberately earlier
    than the original per-value check, which lived in the bucket histogram and
    so only fired when the affected node was predicted from; a corpus carrying
    an unusable latency now fails at fit rather than at the first query that
    happens to reach it. Both fail closed; this one fails sooner and names the
    corpus rather than the query.
    """

    values = [10.0, 20.0, 30.0]
    values[position] = latency
    observations = [
        ClauseObservation(
            repo="owner__repo",
            bin="tool",
            argv=("tool",),
            ts_start=float(index),
            ts_end=float(index) + 1.0,
            latency_ms=value,
        )
        for index, value in enumerate(values)
    ]
    with pytest.raises(ValueError, match="finite and non-negative"):
        ClauseResourceKB.fit_public(observations)

    kb = _fit(
        ClauseObservation(
            repo="other__repo",
            bin="tool",
            argv=("tool",),
            ts_start=0.0,
            ts_end=1.0,
            latency_ms=10.0,
        )
    )
    for observation in observations:
        kb.observe_completed_clause(observation)
    with pytest.raises(ValueError, match="finite and non-negative"):
        kb.predict_clause_latency_bucket(
            "owner__repo",
            "tool",
            ("tool",),
            LatencyBuckets(CANONICAL_LATENCY_BUCKET_EDGES_MS),
            ts_start=100.0,
        )


def test_cold_clause_uses_public_bin_and_modal_bucket() -> None:
    kb = _fit(
        _obs("pub", "pytest", ("pytest", "-q"), 0.0, 1.0, latency_ms=50.0),
        _obs("pub", "pytest", ("pytest", "-k"), 1.0, 2.0, latency_ms=150.0),
        _obs("pub", "pytest", ("pytest", "-x"), 2.0, 3.0, latency_ms=200.0),
    )

    prediction = kb.predict_clause_latency_bucket(
        "r1", "pytest", ("pytest", "tests/x.py"), LatencyBuckets((100.0,))
    )

    assert prediction.probability_by_bucket == pytest.approx((1 / 3, 2 / 3))
    assert prediction.scope == "public"
    assert prediction.key_kind == "bin"


def test_empirical_pmf_preserves_tied_bucket_mass() -> None:
    kb = _fit(
        _obs("pub", "pytest", ("pytest", "-q"), 0.0, 1.0, latency_ms=50.0),
        _obs("pub", "pytest", ("pytest", "-x"), 1.0, 2.0, latency_ms=150.0),
    )

    prediction = kb.predict_clause_latency_bucket(
        "r1", "pytest", ("pytest",), LatencyBuckets((100.0,))
    )

    assert prediction.probability_by_bucket == (0.5, 0.5)


def test_causal_exact_clause_becomes_available_strictly_after_end() -> None:
    buckets = LatencyBuckets((1000.0, 5000.0))
    kb = _fit(_obs("pub", "pytest", ("pytest", "-q"), 0.0, 1.0, latency_ms=100.0))
    kb.observe_completed_clause(
        _obs("r1", "pytest", ("pytest", "-q"), 10.0, 12.0, latency_ms=6000.0)
    )

    same_end = kb.predict_command_latency_bucket(
        "r1", "pytest -q", 12.0, buckets
    ).prediction
    after_end = kb.predict_command_latency_bucket(
        "r1", "pytest -q", 12.1, buckets
    ).prediction

    assert same_end is not None and same_end.scope == "public"
    assert after_end is not None and after_end.scope == "repo"
    assert after_end.key_kind == "exact_clause"
    assert after_end.probability_by_bucket == (0.0, 0.0, 1.0)


def test_repo_prefix_backoff_and_isolation() -> None:
    buckets = LatencyBuckets((1000.0,))
    kb = _fit(_obs("pub", "make", ("make", "-j2", "all"), 0.0, 1.0, latency_ms=100.0))
    kb.observe_completed_clause(
        _obs("r1", "make", ("make", "-j2", "all"), 10.0, 12.0, latency_ms=2000.0)
    )

    prefix = kb.predict_command_latency_bucket(
        "r1", "make -j2 clean", 13.0, buckets
    ).prediction
    other = kb.predict_command_latency_bucket(
        "r2", "make -j2 clean", 14.0, buckets
    ).prediction

    assert prefix is not None and prefix.scope == "repo"
    assert prefix.key_kind == "argv_prefix_depth_2"
    assert prefix.probability_by_bucket == (0.0, 1.0)
    assert other is not None and other.scope == "public"
    assert other.probability_by_bucket == (1.0, 0.0)


def test_leading_cd_command_is_still_compound_and_not_composed() -> None:
    buckets = LatencyBuckets((1000.0,))
    kb = _fit(
        _obs("pub", "cd", ("cd", "/tmp"), 0.0, 1.0, latency_ms=1.0),
        _obs("pub", "cat", ("cat", "x"), 1.0, 2.0, latency_ms=2000.0),
    )

    leading_cd = kb.predict_command_latency_bucket(
        "r1", "cd /tmp && cat x", 10.0, buckets
    )
    compound = kb.predict_command_latency_bucket(
        "r1", "cat x && echo done", 11.0, buckets
    )

    assert leading_cd.clause_bins == ("cd", "cat")
    assert leading_cd.prediction is None
    assert leading_cd.unavailable_reason == "compound_command_uncomposed"
    assert compound.clause_bins == ("cat", "echo")
    assert compound.prediction is None
    assert compound.unavailable_reason == "compound_command_uncomposed"


def test_external_clauses_advance_state_and_backdated_queries_fail() -> None:
    buckets = LatencyBuckets((1000.0,))
    kb = _fit(_obs("pub", "x", ("x",), 0.0, 1.0, latency_ms=100.0))
    kb.observe_completed_clause(_obs("r1", "x", ("x",), 10.0, 12.0, latency_ms=2000.0))

    warm = kb.predict_command_latency_bucket_from_clauses(
        "r1", _clauses(("x", ["x"])), 13.0, buckets
    ).prediction
    assert warm is not None and warm.probability_by_bucket == (0.0, 1.0)
    with pytest.raises(ValueError, match="backdated query"):
        kb.predict_command_latency_bucket_from_clauses(
            "r1", _clauses(("x", ["x"])), 5.0, buckets
        )


def test_cpu_and_rss_measurements_are_preserved_but_not_in_latency_output() -> None:
    kb = _fit(
        _obs(
            "pub",
            "runner",
            ("runner",),
            0.0,
            1.0,
            latency_ms=100.0,
            cpu=3.0,
            rss=700.0,
        )
    )

    prediction = kb.predict_clause_latency_bucket(
        "r1", "runner", ("runner",), LatencyBuckets((1000.0,))
    )

    assert prediction.probability_by_bucket == (1.0, 0.0)
    assert kb._public["peak_cpu_cores"][("bin", "runner")] == (3.0,)
    assert kb._public["sampled_peak_rss_mb"][("bin", "runner")] == (700.0,)


def test_resource_classes_reuse_backoff_and_impute_only_strictly_short_nulls() -> None:
    mib = 1024 * 1024
    kb = _fit(
        _obs(
            "pub",
            "runner",
            ("runner", "heavy-a"),
            0.0,
            1.0,
            latency_ms=1000.0,
            cpu=3.0,
            rss=700.0,
            disk=200 * mib,
        ),
        _obs(
            "pub",
            "runner",
            ("runner", "heavy-b"),
            1.0,
            2.0,
            latency_ms=1000.0,
            cpu=4.0,
            rss=800.0,
            disk=300 * mib,
        ),
        _obs(
            "pub",
            "runner",
            ("runner", "short-null"),
            2.0,
            2.499,
            latency_ms=499.0,
            impute_short_null=True,
        ),
        _obs(
            "pub",
            "runner",
            ("runner", "boundary-null"),
            3.0,
            3.5,
            latency_ms=500.0,
            impute_short_null=True,
        ),
    )

    public = kb.predict_clause_resource_classes(
        "repo", "runner", ("runner", "new"), ts_start=10.0
    )
    assert set(public) == {
        "peak_cpu_cores",
        "sampled_peak_rss_mb",
        "disk_read_write_bytes_total",
    }
    for prediction in public.values():
        assert prediction is not None
        assert prediction.label == "heavy"
        assert prediction.probability_heavy == pytest.approx(2 / 3)
        assert prediction.evidence_count == 3

    kb.observe_completed_clause(
        _obs(
            "repo",
            "runner",
            ("runner", "new"),
            11.0,
            12.0,
            latency_ms=100.0,
            impute_short_null=True,
        )
    )
    at_end = kb.predict_clause_heavy_light(
        "repo", "runner", ("runner", "new"), "peak_cpu_cores", ts_start=12.0
    )
    after_end = kb.predict_clause_heavy_light(
        "repo", "runner", ("runner", "new"), "peak_cpu_cores", ts_start=12.1
    )
    assert at_end is not None and at_end.scope == "public"
    assert after_end is not None and after_end.scope == "repo"
    assert after_end.label == "light" and after_end.evidence_count == 1


def test_serialization_round_trip_preserves_latency_predictions_and_pending() -> None:
    buckets = LatencyBuckets((1000.0,))
    kb = _fit(_obs("pub", "pytest", ("pytest", "-q"), 0.0, 1.0))
    kb.observe_completed_clause(
        _obs("r1", "pytest", ("pytest", "-q"), 10.0, 12.0, latency_ms=2000.0)
    )
    kb.predict_command_latency_bucket("r1", "pytest -q", 13.0, buckets)
    kb.observe_completed_clause(
        _obs("r1", "pytest", ("pytest", "-q"), 20.0, 30.0, latency_ms=2000.0)
    )

    restored = ClauseResourceKB.from_json_obj(json.loads(json.dumps(kb.to_json_obj())))

    assert restored.predict_clause_latency_bucket(
        "r1", "pytest", ("pytest", "-q"), buckets
    ) == kb.predict_clause_latency_bucket("r1", "pytest", ("pytest", "-q"), buckets)
    late = restored.predict_command_latency_bucket(
        "r1", "pytest -q", 31.0, buckets
    ).prediction
    assert late is not None and late.evidence_count == 2


def test_v5_snapshot_requires_refit_for_resource_labels() -> None:
    with pytest.raises(ValueError, match="refit the snapshot"):
        ClauseResourceKB.from_json_obj({"schema": "runtime_clause_resource_kb_v5"})


def test_parse_failure_is_explicitly_unavailable() -> None:
    kb = _fit(_obs("pub", "echo", ("echo",), 0.0, 1.0))

    prediction = kb.predict_command_latency_bucket(
        "r1", "echo 'unterminated", 10.0, LatencyBuckets((100.0,))
    )

    assert prediction.parse_failed
    assert prediction.prediction is None
    assert prediction.unavailable_reason == "parse_failed"


def test_invalid_observation_and_unfit_public_fail_fast() -> None:
    with pytest.raises(ValueError):
        ClauseObservation(repo="r", bin="x", argv=(), ts_start=0.0, ts_end=1.0)
    with pytest.raises(ValueError):
        ClauseObservation(repo="r", bin="x", argv=("x",), ts_start=1.0, ts_end=0.0)
    with pytest.raises(ValueError):
        ClauseResourceKB.fit_public(
            [_obs("pub", "x", ("x",), 0.0, 1.0, latency_ms=None)]
        )


@pytest.mark.parametrize("ts_start", [float("inf"), float("nan")])
def test_nonfinite_query_time_fails_before_absorbing_pending(ts_start: float) -> None:
    kb = _fit(_obs("pub", "x", ("x",), 0.0, 1.0, latency_ms=100.0))
    kb.observe_completed_clause(_obs("r1", "x", ("x",), 10.0, 12.0, latency_ms=2000.0))

    with pytest.raises(ValueError, match="finite"):
        kb.predict_command_latency_bucket(
            "r1", "x", ts_start, LatencyBuckets((1000.0,))
        )

    prediction = kb.predict_command_latency_bucket(
        "r1", "x", 11.0, LatencyBuckets((1000.0,))
    ).prediction
    assert prediction is not None and prediction.scope == "public"
