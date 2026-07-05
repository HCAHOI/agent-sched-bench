from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CATEGORY_COSMETIC = "cosmetic"
CATEGORY_CONTENT_DIVERGENT = "content_divergent"
CATEGORY_UNCLASSIFIED = "unclassified"
TIER_SKIPPED = "skipped"

_CATEGORIES = {
    CATEGORY_COSMETIC,
    CATEGORY_CONTENT_DIVERGENT,
    CATEGORY_UNCLASSIFIED,
}


@dataclass(frozen=True)
class MismatchVerdict:
    tier: int
    category: str
    semantic_match: bool
    evidence: dict[str, Any]

    def __post_init__(self) -> None:
        if self.tier not in {1, 2, 3}:
            raise ValueError(f"mismatch verdict tier must be 1, 2, or 3: {self.tier}")
        if self.category not in _CATEGORIES:
            raise ValueError(f"unknown mismatch verdict category: {self.category!r}")


@dataclass(frozen=True)
class TierVerdicts:
    tier_1: str
    tier_2: str
    tier_3: str


@dataclass(frozen=True)
class _TierDecision:
    tier: int
    category: str
    reason: str

    @property
    def semantic_match(self) -> bool:
        return self.category == CATEGORY_COSMETIC


class MismatchOracle:
    """Classify replay divergence using transport, output, and state signals."""

    def verdict_from_action_data(
        self,
        data: dict[str, Any],
    ) -> MismatchVerdict | None:
        return self.verdict(
            source_returncode=data.get("source_returncode"),
            replay_returncode=data.get("replay_returncode", data.get("returncode")),
            source_timed_out=data.get("source_timed_out"),
            replay_timed_out=data.get("replay_timed_out", data.get("timed_out")),
            normalized_output_match=data.get("normalized_output_match"),
            output_diff_snippet=data.get("output_diff_snippet"),
            cas_manifest_match=data.get("cas_manifest_match"),
            cas_modified_count=data.get("cas_modified_count"),
            cas_removed_count=data.get("cas_removed_count"),
            cas_added_count=data.get("cas_added_count"),
            mismatch_reason=data.get("mismatch_reason"),
        )

    def verdict(
        self,
        *,
        source_returncode: Any = None,
        replay_returncode: Any = None,
        source_timed_out: Any = None,
        replay_timed_out: Any = None,
        normalized_output_match: Any = None,
        output_diff_snippet: Any = None,
        cas_manifest_match: Any = None,
        cas_modified_count: Any = None,
        cas_removed_count: Any = None,
        cas_added_count: Any = None,
        mismatch_reason: Any = None,
    ) -> MismatchVerdict | None:
        source_rc = _optional_int("source_returncode", source_returncode)
        replay_rc = _optional_int("replay_returncode", replay_returncode)
        source_timeout = _optional_bool("source_timed_out", source_timed_out)
        replay_timeout = _optional_bool("replay_timed_out", replay_timed_out)
        normalized_match = _optional_bool(
            "normalized_output_match",
            normalized_output_match,
        )
        output_diff = _optional_str("output_diff_snippet", output_diff_snippet)
        cas_match = _optional_bool("cas_manifest_match", cas_manifest_match)
        cas_modified = _optional_int("cas_modified_count", cas_modified_count)
        cas_removed = _optional_int("cas_removed_count", cas_removed_count)
        cas_added = _optional_int("cas_added_count", cas_added_count)
        reason = _optional_str("mismatch_reason", mismatch_reason)

        tier_1 = self._tier_1_decision(
            source_returncode=source_rc,
            replay_returncode=replay_rc,
            source_timed_out=source_timeout,
            replay_timed_out=replay_timeout,
            mismatch_reason=reason,
        )
        tier_2 = self._tier_2_decision(
            normalized_output_match=normalized_match,
            output_diff_snippet=output_diff,
        )
        tier_3 = self._tier_3_decision(
            cas_manifest_match=cas_match,
            mismatch_reason=reason,
        )

        evidence = {
            "source_returncode": source_rc,
            "replay_returncode": replay_rc,
            "source_timed_out": source_timeout,
            "replay_timed_out": replay_timeout,
            "timeout_mismatch": (
                source_timeout != replay_timeout
                if source_timeout is not None and replay_timeout is not None
                else reason == "timeout_mismatch"
            ),
            "returncode_mismatch": (
                source_rc != replay_rc
                if source_rc is not None and replay_rc is not None
                else None
            ),
            "normalized_output_match": normalized_match,
            "output_diff_snippet_present": bool(output_diff),
            "cas_manifest_match": cas_match,
            "cas_modified_count": cas_modified,
            "cas_removed_count": cas_removed,
            "cas_added_count": cas_added,
            "mismatch_reason": reason,
            "tiers": {
                "1": _decision_evidence(tier_1),
                "2": _decision_evidence(tier_2),
                "3": _decision_evidence(tier_3),
            },
        }

        if tier_3 is not None:
            return _to_verdict(tier_3, evidence)
        lower_tier = _select_lower_tier(tier_1, tier_2)
        if lower_tier is None:
            return None
        return _to_verdict(lower_tier, evidence)

    def tier_verdicts_from_action_data(self, data: dict[str, Any]) -> TierVerdicts:
        verdict = self.verdict_from_action_data(data)
        tiers = {}
        if verdict is not None:
            tiers = verdict.evidence.get("tiers", {})
        return TierVerdicts(
            tier_1=_category_from_tier_evidence(tiers.get("1")),
            tier_2=_category_from_tier_evidence(tiers.get("2")),
            tier_3=_category_from_tier_evidence(tiers.get("3")),
        )

    def semantic_match_from_action_data(self, data: dict[str, Any]) -> bool | None:
        """Determine semantic match status from action data with backward-compat fallback.

        Priority order:
        1. ``oracle_semantic_match`` — the modern oracle field (state-level)
        2. ``semantic_match`` — direct field from verdict evidence
        3. ``verdict.semantic_match`` — extracted from structured verdict
        4. ``replay_outcome_match`` — **legacy fallback** for traces recorded
           before the mismatch oracle was added; this is a **transport-level**
           signal (tool return codes / output equality), not a state-level
           semantic match.  The fallback therefore conflates precision levels
           when the oracle is unavailable — callers must treat the result as a
           lower-confidence approximation.
        """
        oracle_semantic_match = data.get("oracle_semantic_match")
        if oracle_semantic_match is not None:
            return _optional_bool("oracle_semantic_match", oracle_semantic_match)

        semantic_match = data.get("semantic_match")
        if semantic_match is not None:
            return _optional_bool("semantic_match", semantic_match)

        verdict = self.verdict_from_action_data(data)
        if verdict is not None:
            return verdict.semantic_match

        replay_outcome_match = data.get("replay_outcome_match")
        if replay_outcome_match is None:
            return None
        return _optional_bool("replay_outcome_match", replay_outcome_match)

    def _tier_1_decision(
        self,
        *,
        source_returncode: int | None,
        replay_returncode: int | None,
        source_timed_out: bool | None,
        replay_timed_out: bool | None,
        mismatch_reason: str | None,
    ) -> _TierDecision | None:
        if (
            source_timed_out is not None
            and replay_timed_out is not None
            and source_timed_out != replay_timed_out
        ) or mismatch_reason == "timeout_mismatch":
            return _TierDecision(1, CATEGORY_CONTENT_DIVERGENT, "timeout_mismatch")

        compared = False
        if source_timed_out is not None and replay_timed_out is not None:
            compared = True

        if source_returncode is not None and replay_returncode is not None:
            compared = True
            if source_returncode != replay_returncode:
                return _TierDecision(
                    1,
                    CATEGORY_CONTENT_DIVERGENT,
                    "returncode_mismatch",
                )

        if not compared:
            return None
        return _TierDecision(1, CATEGORY_COSMETIC, "transport_match")

    def _tier_2_decision(
        self,
        *,
        normalized_output_match: bool | None,
        output_diff_snippet: str | None,
    ) -> _TierDecision | None:
        if normalized_output_match is None:
            return None
        if normalized_output_match:
            return _TierDecision(2, CATEGORY_COSMETIC, "normalized_output_match")
        if not output_diff_snippet:
            return _TierDecision(
                2,
                CATEGORY_UNCLASSIFIED,
                "normalized_output_mismatch_without_diff",
            )
        return _TierDecision(
            2,
            CATEGORY_CONTENT_DIVERGENT,
            "normalized_output_mismatch",
        )

    def _tier_3_decision(
        self,
        *,
        cas_manifest_match: bool | None,
        mismatch_reason: str | None,
    ) -> _TierDecision | None:
        if cas_manifest_match is None:
            return None
        if cas_manifest_match:
            return _TierDecision(3, CATEGORY_COSMETIC, "cas_manifest_match")
        reason = (
            "cas_state_mismatch"
            if mismatch_reason == "cas_state_mismatch"
            else "cas_manifest_mismatch"
        )
        return _TierDecision(3, CATEGORY_CONTENT_DIVERGENT, reason)


def _optional_bool(name: str, value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean when present, got {value!r}")
    return value


def _optional_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be int when present, got {value!r}")
    return value


def _optional_str(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be string when present, got {value!r}")
    return value


def _decision_evidence(decision: _TierDecision | None) -> dict[str, Any]:
    if decision is None:
        return {"category": TIER_SKIPPED}
    return {
        "category": decision.category,
        "reason": decision.reason,
        "semantic_match": decision.semantic_match,
    }


def _category_from_tier_evidence(value: Any) -> str:
    if not isinstance(value, dict):
        return TIER_SKIPPED
    category = value.get("category")
    if isinstance(category, str):
        return category
    return TIER_SKIPPED


def _select_lower_tier(
    tier_1: _TierDecision | None,
    tier_2: _TierDecision | None,
) -> _TierDecision | None:
    if tier_1 is not None and tier_1.category == CATEGORY_CONTENT_DIVERGENT:
        return tier_1
    if tier_2 is not None and tier_2.category != CATEGORY_COSMETIC:
        return tier_2
    if tier_2 is not None:
        return tier_2
    return tier_1


def _to_verdict(
    decision: _TierDecision,
    evidence: dict[str, Any],
) -> MismatchVerdict:
    return MismatchVerdict(
        tier=decision.tier,
        category=decision.category,
        semantic_match=decision.semantic_match,
        evidence=evidence,
    )
