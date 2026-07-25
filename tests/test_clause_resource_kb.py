from __future__ import annotations

import json

import pytest

from tool_resource.runtime_kb import (
    CPU_HEAVY_TARGET,
    MEMORY_HEAVY_TARGET,
    ClauseFlagPrediction,
    ClauseObservation,
    ClauseResourceKB,
    _aggregate_or,
)


def _flag(value: bool | None) -> ClauseFlagPrediction:
    return ClauseFlagPrediction(
        target=CPU_HEAVY_TARGET,
        flag=value,
        threshold=2.0,
        source="peak_cpu_cores",
        scope="public",
        key_kind="bin",
        evidence_count=1,
        fallback_path=(),
    )


def _obs(
    repo: str,
    bin_: str,
    argv: tuple[str, ...],
    start: float,
    end: float,
    *,
    latency_ms: float | None = 100.0,
    cpu: float | None = 0.5,
    rss: float | None = 50.0,
    cpu_ns: int | None = 10_000_000,
) -> ClauseObservation:
    return ClauseObservation(
        repo=repo,
        bin=bin_,
        argv=tuple(argv),
        ts_start=start,
        ts_end=end,
        latency_ms=latency_ms,
        peak_cpu_cores=cpu,
        sampled_peak_rss_mb=rss,
        cpu_ns_cumulative=cpu_ns,
    )


def _fit(*obs: ClauseObservation) -> ClauseResourceKB:
    return ClauseResourceKB.fit_public(obs)


def _clauses(*specs: tuple[str, list[str]]) -> list[dict]:
    return [{"bin": b, "argv": argv} for b, argv in specs]


def test_cold_clause_uses_public_bin_not_prefix_or_repo() -> None:
    kb = _fit(_obs("pub", "pytest", ("pytest", "-q"), 0.0, 1.0, cpu=3.0))
    flags = kb.predict_clause("r1", "pytest", ("pytest", "-k", "x"))

    cpu = flags[CPU_HEAVY_TARGET]
    assert cpu.scope == "public"
    assert cpu.key_kind == "bin"
    assert cpu.flag is True  # 3.0 > 2 cores
    assert cpu.fallback_path[0] == "repo:exact_clause"
    assert not any(p.startswith("public:exact") for p in cpu.fallback_path)
    assert not any("prefix" in p and p.startswith("public") for p in cpu.fallback_path)


def test_public_holds_only_bin_and_global_and_is_immutable() -> None:
    kb = _fit(
        _obs("pub", "pytest", ("pytest", "-q"), 0.0, 1.0),
        _obs("pub", "make", ("make", "all"), 1.0, 2.0),
    )
    for nodes in kb._public.values():
        assert {kind for kind, _ in nodes} <= {"bin", "global"}
    snapshot = {s: dict(n) for s, n in kb._public.items()}
    kb.observe_completed_clause(
        _obs("r1", "pytest", ("pytest", "-q"), 10.0, 20.0, cpu=8.0)
    )
    kb.predict_command_from_clauses(
        "r1", _clauses(("pytest", ["pytest", "-q"])), 10.0
    )
    assert kb._public == snapshot


def test_causal_exact_clause_becomes_available_and_all_thresholds() -> None:
    kb = _fit(_obs("pub", "pytest", ("pytest", "-q"), 0.0, 1.0))
    kb.observe_completed_clause(
        _obs(
            "r1", "pytest", ("pytest", "-q"), 10.0, 12.0,
            latency_ms=6000.0, cpu=3.0, rss=600.0,
        )
    )
    pred = kb.predict_command("r1", "pytest -q", 13.0)
    assert pred.targets["latency_long_3500ms"].flag is True
    assert pred.targets["latency_long_5000ms"].flag is True
    assert pred.targets[CPU_HEAVY_TARGET].flag is True  # 3.0 > 2
    assert pred.targets[MEMORY_HEAVY_TARGET].flag is True  # 600 > 500
    clause_cpu = pred.targets[CPU_HEAVY_TARGET].clause_flags[0]
    assert clause_cpu.scope == "repo"
    assert clause_cpu.key_kind == "exact_clause"


def test_external_clauses_advance_causal_state() -> None:
    kb = _fit(_obs("pub", "pytest", ("pytest", "-q"), 0.0, 1.0, cpu=0.5))
    kb.observe_completed_clause(
        _obs("r1", "pytest", ("pytest", "-q"), 10.0, 12.0, cpu=3.0)
    )

    cpu = kb.predict_command_from_clauses(
        "r1", _clauses(("pytest", ["pytest", "-q"])), 13.0
    ).targets[CPU_HEAVY_TARGET].clause_flags[0]

    assert cpu.scope == "repo"
    assert cpu.flag is True


def test_argv_prefix_and_bin_backoff() -> None:
    kb = _fit(_obs("pub", "make", ("make", "-j2", "all"), 0.0, 1.0))
    kb.observe_completed_clause(
        _obs("r1", "make", ("make", "-j2", "all"), 10.0, 12.0, cpu=4.0)
    )
    prefix = kb.predict_command("r1", "make -j2 clean", 13.0).targets[
        CPU_HEAVY_TARGET
    ].clause_flags[0]
    assert prefix.scope == "repo"
    assert prefix.key_kind == "argv_prefix_depth_2"
    binlevel = kb.predict_command("r1", "make install", 14.0).targets[
        CPU_HEAVY_TARGET
    ].clause_flags[0]
    assert binlevel.scope == "repo"
    assert binlevel.key_kind == "bin"


def test_repo_isolation() -> None:
    kb = _fit(_obs("pub", "pytest", ("pytest", "-q"), 0.0, 1.0, cpu=0.5))
    kb.observe_completed_clause(
        _obs("r1", "pytest", ("pytest", "-q"), 10.0, 12.0, cpu=8.0)
    )
    other = kb.predict_command("r2", "pytest -q", 13.0).targets[
        CPU_HEAVY_TARGET
    ].clause_flags[0]
    assert other.scope == "public"
    assert other.flag is False  # public cpu is 0.5


def test_running_overlapping_same_start_future_and_monotonic_guard() -> None:
    kb = _fit(_obs("pub", "pytest", ("pytest", "-q"), 0.0, 1.0, cpu=0.5))
    kb.observe_completed_clause(
        _obs("r1", "pytest", ("pytest", "-q"), 0.0, 10.0, cpu=8.0)
    )
    kb.observe_completed_clause(
        _obs("r1", "pytest", ("pytest", "-q"), 20.0, 21.0, cpu=8.0)
    )
    assert kb.predict_command("r1", "pytest -q", 5.0).targets[
        CPU_HEAVY_TARGET
    ].clause_flags[0].scope == "public"
    assert kb.predict_command("r1", "pytest -q", 10.0).targets[
        CPU_HEAVY_TARGET
    ].clause_flags[0].scope == "public"
    warm = kb.predict_command("r1", "pytest -q", 10.5).targets[
        CPU_HEAVY_TARGET
    ].clause_flags[0]
    assert warm.scope == "repo"
    assert warm.evidence_count == 1
    with pytest.raises(ValueError, match="backdated query"):
        kb.predict_command("r1", "pytest -q", 5.0)


def test_command_three_valued_or_via_kb() -> None:
    # cpu global non-empty -> clauses resolve True/False (never per-clause unknown)
    kb = _fit(
        _obs("pub", "light", ("light",), 0.0, 1.0, cpu=0.5),
        _obs("pub", "heavy", ("heavy",), 1.0, 2.0, cpu=3.0),
    )
    assert kb.predict_command_from_clauses(
        "r1", _clauses(("light", ["light"]), ("heavy", ["heavy"])), 10.0
    ).targets[CPU_HEAVY_TARGET].flag is True  # any True -> True
    assert kb.predict_command_from_clauses(
        "r1", _clauses(("light", ["light"]), ("light", ["light", "x"])), 10.0
    ).targets[CPU_HEAVY_TARGET].flag is False  # all False -> False

    # a target whose source has NO global evidence -> every clause unknown
    kb_noc = _fit(_obs("pub", "light", ("light",), 0.0, 1.0, cpu=None))
    assert kb_noc.predict_command_from_clauses(
        "r1", _clauses(("light", ["light"]), ("heavy", ["heavy"])), 10.0
    ).targets[CPU_HEAVY_TARGET].flag is None  # all unknown -> Unknown


def test_three_valued_or_composer_semantics() -> None:
    # the frozen rule, including the mixed branch the KB alone cannot produce
    assert _aggregate_or(CPU_HEAVY_TARGET, [_flag(True), _flag(None)]).flag is True
    assert _aggregate_or(CPU_HEAVY_TARGET, [_flag(False), _flag(None)]).flag is None
    assert _aggregate_or(CPU_HEAVY_TARGET, [_flag(False), _flag(False)]).flag is False
    assert _aggregate_or(CPU_HEAVY_TARGET, []).flag is False  # OR identity
    assert "does not sum concurrent" in _aggregate_or(CPU_HEAVY_TARGET, [_flag(False)]).note


def test_cpu_heavy_is_real_from_peak_cores_and_unknown_without_evidence() -> None:
    kb = _fit(_obs("pub", "pytest", ("pytest",), 0.0, 1.0, cpu=3.0))
    assert kb.predict_clause("r1", "pytest", ("pytest",))[CPU_HEAVY_TARGET].flag is True
    # a KB fit with no cpu evidence anywhere -> cpu unknown, never fabricated
    kb_noc = _fit(_obs("pub", "cat", ("cat",), 0.0, 1.0, cpu=None))
    cat = kb_noc.predict_clause("r1", "cat", ("cat",))[CPU_HEAVY_TARGET]
    assert cat.flag is None
    assert "no peak_cpu_cores evidence" in cat.note


def test_memory_unknown_when_no_rss_evidence() -> None:
    kb = _fit(_obs("pub", "pytest", ("pytest",), 0.0, 1.0, rss=None))
    mem = kb.predict_clause("r1", "pytest", ("pytest",))[MEMORY_HEAVY_TARGET]
    assert mem.flag is None
    assert "no sampled_peak_rss_mb evidence" in mem.note
    # latency still resolves
    assert kb.predict_clause("r1", "pytest", ("pytest",))["latency_long_3500ms"].flag is False


def test_serialization_round_trip_preserves_predictions_and_pending() -> None:
    kb = _fit(
        _obs("pub", "pytest", ("pytest", "-q"), 0.0, 1.0),
        _obs("pub", "make", ("make", "all"), 1.0, 2.0),
    )
    kb.observe_completed_clause(
        _obs("r1", "pytest", ("pytest", "-q"), 10.0, 12.0, cpu=8.0, rss=600.0)
    )
    kb.predict_command("r1", "pytest -q", 13.0)
    kb.observe_completed_clause(
        _obs("r1", "pytest", ("pytest", "-q"), 20.0, 30.0, cpu=8.0)
    )
    restored = ClauseResourceKB.from_json_obj(json.loads(json.dumps(kb.to_json_obj())))
    for repo, bin_, argv in (
        ("r1", "pytest", ("pytest", "-q")),
        ("r2", "make", ("make", "all")),
    ):
        assert restored.predict_clause(repo, bin_, argv) == kb.predict_clause(
            repo, bin_, argv
        )
    warm = restored.predict_command("r1", "pytest -q", 31.0).targets[
        CPU_HEAVY_TARGET
    ].clause_flags[0]
    assert warm.evidence_count == 2


def test_real_mvdan_parse_path() -> None:
    kb = _fit(
        _obs("pub", "sleep", ("sleep", "1"), 0.0, 1.0),
        _obs("pub", "echo", ("echo", "x"), 1.0, 2.0),
    )
    pred = kb.predict_command("r1", "echo hi && sleep 1", 10.0)
    assert not pred.parse_failed
    assert pred.clause_bins == ("echo", "sleep")
    for target in pred.targets.values():
        assert target.flag in (True, False, None)


def test_invalid_observation_and_unfit_public_fail_fast() -> None:
    with pytest.raises(ValueError):
        ClauseObservation(repo="r", bin="x", argv=(), ts_start=0.0, ts_end=1.0)
    with pytest.raises(ValueError):
        ClauseObservation(repo="r", bin="x", argv=("x",), ts_start=1.0, ts_end=0.0)
    with pytest.raises(ValueError):
        ClauseResourceKB.fit_public(
            [_obs("pub", "x", ("x",), 0.0, 1.0, latency_ms=None)]
        )
