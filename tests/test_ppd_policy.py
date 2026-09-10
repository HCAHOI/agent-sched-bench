"""CPU decision tests; synthetic scores establish semantics, not performance."""
import json

import pytest

from scripts.baselines.ppd_policy import add_huge_context_table, protect_decode


def test_measured_huge_table_replaces_extrapolation_without_changing_public_rules(tmp_path, monkeypatch):
    from ppd.optimizer.ppd_decision_engine import PPDDecisionEngine, QPS_POINTS, T2_WORKLOAD_CONFIGS

    monkeypatch.setenv("PPD_BYPASS_THRESHOLD", "512")
    for mode in ("1P_1D", "1P_1pD"):
        (tmp_path / mode).mkdir()
        for context in ("small", "large", "huge"):
            for workload in T2_WORKLOAD_CONFIGS:
                for qps in QPS_POINTS:
                    ttft = 1 if mode == "1P_1D" else (2 if context == "huge" else 0.5)
                    (tmp_path / mode / f"{mode}_{context}_{workload}_{qps}.json").write_text(json.dumps(
                        {"success_rate": 100, "turn2": {"avg_ttft_ms": ttft, "avg_tpot_ms": 1}}))
    engine = PPDDecisionEngine(str(tmp_path), base_config="1P_1D")
    before = dict(engine.lookup_table)
    assert engine.should_use_ppd(2, 1024, 32, 1, 32000)
    add_huge_context_table(engine, tmp_path)
    assert len(engine.lookup_table) == 270
    assert all(engine.lookup_table[k] == value for k, value in before.items())
    assert not engine.should_use_ppd(2, 1024, 32, 1, 32000)
    assert engine.should_use_ppd(2, 511, 32, 1, 32000)  # Original short-input rule remains.
    assert not engine.should_use_ppd(1, 16, 32, 1, 0)
    next((tmp_path / "1P_1D").glob('*_huge_*.json')).unlink()
    with pytest.raises(AssertionError, match="Missing measured huge-context"):
        add_huge_context_table(engine, tmp_path)


def test_decode_guard_uses_actual_missing_tokens_and_engine_capacity():
    state = dict(cached_tokens=0, running=8, waiting=0, max_num_seqs=8)
    assert protect_decode(True, 32000, state, 512) == (False, "uncached_prefill_on_busy_decode")
    assert protect_decode(True, 32000, dict(state, cached_tokens=31872), 512)[0]
    assert protect_decode(True, 32000, dict(state, running=1), 512)[0]
    assert not protect_decode(True, 32000, dict(state, running=1, waiting=1), 512)[0]
    assert not protect_decode(False, 32000, dict(state, running=0), 512)[0]
    with pytest.raises(AssertionError):
        protect_decode(True, 32000, dict(state, cached_tokens=32001), 512)
