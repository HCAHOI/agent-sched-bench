from __future__ import annotations

from scripts.evaluation.evaluate_resource_prediction import (
    ambient_residual_prediction,
    build_ambient_residual_ecdfs,
    repo_cluster_key,
)


def test_ambient_residual_prediction_uses_tool_then_global_fallback() -> None:
    rows = [
        {"tool_name": "exec", "peak_memory_mb": 110.0, "ambient_before_mb": 100.0},
        {"tool_name": "exec", "peak_memory_mb": 140.0, "ambient_before_mb": 120.0},
        {
            "tool_name": "read_file",
            "peak_memory_mb": 200.0,
            "ambient_before_mb": 100.0,
        },
    ]
    by_tool, global_values = build_ambient_residual_ecdfs(rows)

    assert ambient_residual_prediction("exec", 200.0, by_tool, global_values) == (
        220.0,
        "tool_residual",
    )
    assert ambient_residual_prediction("unseen", 200.0, by_tool, global_values) == (
        300.0,
        "global_residual",
    )


def test_repo_cluster_key_strips_only_trailing_numeric_issue() -> None:
    assert repo_cluster_key("tobymao__sqlglot-1234") == "tobymao__sqlglot"
    assert repo_cluster_key("owner__repo-name-12") == "owner__repo-name"
    assert repo_cluster_key("owner__repo-name") == "owner__repo-name"
    assert repo_cluster_key("owner__repo-12x") == "owner__repo-12x"
