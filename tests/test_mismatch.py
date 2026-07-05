from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_collect.mismatch import (
    CATEGORY_CONTENT_DIVERGENT,
    CATEGORY_COSMETIC,
    CATEGORY_UNCLASSIFIED,
    MismatchOracle,
)


def test_mismatch_oracle_tier_1_timeout_mismatch() -> None:
    verdict = MismatchOracle().verdict(
        source_returncode=0,
        replay_returncode=0,
        source_timed_out=False,
        replay_timed_out=True,
        normalized_output_match=True,
    )

    assert verdict is not None
    assert verdict.tier == 1
    assert verdict.category == CATEGORY_CONTENT_DIVERGENT
    assert verdict.semantic_match is False
    assert verdict.evidence["timeout_mismatch"] is True


def test_mismatch_oracle_tier_1_exit_code_mismatch() -> None:
    verdict = MismatchOracle().verdict(
        source_returncode=0,
        replay_returncode=2,
        source_timed_out=False,
        replay_timed_out=False,
        normalized_output_match=True,
    )

    assert verdict is not None
    assert verdict.tier == 1
    assert verdict.category == CATEGORY_CONTENT_DIVERGENT
    assert verdict.semantic_match is False


def test_mismatch_oracle_tier_2_output_mismatch() -> None:
    verdict = MismatchOracle().verdict(
        source_returncode=0,
        replay_returncode=0,
        source_timed_out=False,
        replay_timed_out=False,
        normalized_output_match=False,
        output_diff_snippet="- expected\n+ actual",
    )

    assert verdict is not None
    assert verdict.tier == 2
    assert verdict.category == CATEGORY_CONTENT_DIVERGENT
    assert verdict.semantic_match is False


def test_mismatch_oracle_tier_2_unclassified_empty_diff() -> None:
    verdict = MismatchOracle().verdict(
        source_returncode=0,
        replay_returncode=0,
        normalized_output_match=False,
        output_diff_snippet="",
    )

    assert verdict is not None
    assert verdict.tier == 2
    assert verdict.category == CATEGORY_UNCLASSIFIED
    assert verdict.semantic_match is False


def test_mismatch_oracle_tier_3_cas_match_overrides_lower_tiers() -> None:
    verdict = MismatchOracle().verdict(
        source_returncode=0,
        replay_returncode=0,
        normalized_output_match=False,
        output_diff_snippet="- expected\n+ actual",
        cas_manifest_match=True,
        cas_modified_count=0,
        cas_removed_count=0,
        cas_added_count=0,
    )

    assert verdict is not None
    assert verdict.tier == 3
    assert verdict.category == CATEGORY_COSMETIC
    assert verdict.semantic_match is True


def test_mismatch_oracle_tier_3_cas_mismatch_is_authoritative() -> None:
    verdict = MismatchOracle().verdict(
        source_returncode=0,
        replay_returncode=0,
        normalized_output_match=True,
        cas_manifest_match=False,
        cas_modified_count=1,
        cas_removed_count=0,
        cas_added_count=0,
        mismatch_reason="cas_state_mismatch",
    )

    assert verdict is not None
    assert verdict.tier == 3
    assert verdict.category == CATEGORY_CONTENT_DIVERGENT
    assert verdict.semantic_match is False


def test_mismatch_oracle_legacy_fallback_uses_raw_match() -> None:
    data = {"replay_outcome_match": False}

    oracle = MismatchOracle()

    assert oracle.verdict_from_action_data(data) is None
    assert oracle.semantic_match_from_action_data(data) is False


def test_mismatch_oracle_fails_fast_on_invalid_structured_fields() -> None:
    with pytest.raises(ValueError, match="source_returncode"):
        MismatchOracle().verdict(source_returncode=True)

    with pytest.raises(ValueError, match="normalized_output_match"):
        MismatchOracle().verdict(normalized_output_match="yes")


def test_mismatch_oracle_round_trips_minimal_simulate_trace(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "simulate.jsonl"
    records = [
        _tool_record(
            "timeout",
            {
                "replay_outcome_match": False,
                "mismatch_reason": "timeout_mismatch",
                "source_returncode": 124,
                "replay_returncode": 0,
                "source_timed_out": True,
                "replay_timed_out": False,
                "normalized_output_match": False,
                "output_diff_snippet": "- [timeout]\n+ ok",
            },
        ),
        _tool_record(
            "exit-code",
            {
                "replay_outcome_match": False,
                "mismatch_reason": "command_exit_code_mismatch",
                "source_returncode": 0,
                "replay_returncode": 1,
                "source_timed_out": False,
                "replay_timed_out": False,
                "normalized_output_match": True,
            },
        ),
        _tool_record(
            "output",
            {
                "replay_outcome_match": True,
                "source_returncode": 0,
                "replay_returncode": 0,
                "source_timed_out": False,
                "replay_timed_out": False,
                "normalized_output_match": False,
                "output_diff_snippet": "- alpha\n+ beta",
            },
        ),
        _tool_record(
            "cas",
            {
                "replay_outcome_match": False,
                "mismatch_reason": "cas_state_mismatch",
                "source_returncode": 0,
                "replay_returncode": 0,
                "normalized_output_match": True,
                "cas_manifest_match": False,
                "cas_modified_count": 1,
                "cas_removed_count": 0,
                "cas_added_count": 0,
            },
        ),
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    oracle = MismatchOracle()
    loaded = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    verdicts = [
        oracle.verdict_from_action_data(record["data"]) for record in loaded
    ]

    assert [(verdict.tier, verdict.category, verdict.semantic_match) for verdict in verdicts if verdict] == [
        (1, CATEGORY_CONTENT_DIVERGENT, False),
        (1, CATEGORY_CONTENT_DIVERGENT, False),
        (2, CATEGORY_CONTENT_DIVERGENT, False),
        (3, CATEGORY_CONTENT_DIVERGENT, False),
    ]


def _tool_record(action_id: str, data: dict[str, object]) -> dict[str, object]:
    return {
        "type": "action",
        "action_type": "tool_exec",
        "action_id": action_id,
        "data": data,
    }
