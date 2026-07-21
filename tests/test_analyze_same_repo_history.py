from __future__ import annotations

import hashlib
import os
from pathlib import Path
import pytest

from scripts.exploration.analyze_same_repo_history import (
    _apply_guard_and_baseline,
    _load_config,
    _publish_staged_outputs,
    _repository_identity,
    _score_repository_task,
    _verify_trace_inventories,
)


def _row(sample_id: str, task_id: str, latency_ms: float) -> dict[str, object]:
    return {
        "sample_id": f"/local/{task_id}/trace.jsonl:agent:0:{sample_id}",
        "task_id": task_id,
        "source_trace": f"/local/{task_id}/trace.jsonl",
        "tool_name": "exec",
        "tool_ts_start": 0.0,
        "tool_ts_end": latency_ms / 1000.0,
        "latency_ms": latency_ms,
        "tool_args": {"command": "pytest -q"},
    }


def _score(
    eval_row: dict[str, object], profile_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    return _score_repository_task(
        [eval_row],
        identity=_repository_identity(str(eval_row["task_id"])),
        profile_rows=profile_rows,
        costs_ms=[3500.0],
        guard_ms=0.0,
        restore_cost_fraction=0.94,
        command_field="command",
        max_prefix_depth=4,
        skip_leading_cd=False,
        minimum_distinct_prior_tasks=2,
    )


def test_repository_parser_uses_final_numeric_issue_and_casefolds() -> None:
    parsed = _repository_identity("GerbenOostra__poetry-plugin-mono-repo-deps-24")
    assert parsed["repository_raw"] == "GerbenOostra__poetry-plugin-mono-repo-deps"
    assert (
        parsed["repository_canonical"] == "gerbenoostra__poetry-plugin-mono-repo-deps"
    )
    assert not _repository_identity("owner__repo-not-an-issue")["repository_supported"]


def test_same_repo_requires_two_other_tasks_and_falls_back_exactly() -> None:
    eval_row = _row("eval", "Owner__Repo-3", 8000.0)
    one_task = [_row("p1", "Owner__Repo-1", 5000.0)]
    unsupported = _score(eval_row, one_task)
    assert unsupported[0]["repository_supported"] is False

    two_tasks = one_task + [_row("p2", "owner__repo-2", 5000.0)]
    candidate = _score(eval_row, two_tasks)
    assert candidate[0]["repository_supported"] is True
    assert candidate[0]["same_repo_prior_task_ids"] == [
        "Owner__Repo-1",
        "owner__repo-2",
    ]
    assert "Owner__Repo-3" not in candidate[0]["same_repo_prior_task_ids"]
    assert float(candidate[0]["same_repo_candidate_trigger_ms"]) < 3500.0

    baseline = [
        {
            "sample_id": "/remote/Owner__Repo-3/trace.jsonl:agent:0:eval",
            "task_id": "Owner__Repo-3",
            "source_trace": "/remote/Owner__Repo-3/trace.jsonl",
            "tool_name": "exec",
            "tool_ts_start": 0.0,
            "tool_ts_end": 8.0,
            "latency_ms": 8000.0,
            "kv_cost_ms": 3500.0,
            "threshold_ms": 3500.0,
            "restore_cost_ms": 3290.0,
            "warmup_snapshot_trigger_ms": 1234.0,
        }
    ]
    fallback = _apply_guard_and_baseline(
        unsupported,
        baseline,
        selected_guard=0.0,
        outer_fold=1,
        baseline_sidecar={"file": "f1.zst", "sha256": "0" * 64},
    )
    assert fallback[0]["same_repo_trigger_ms"] == 1234.0
    assert fallback[0]["same_repo_fallback_reason"] == (
        "fewer_than_two_prior_repository_tasks"
    )
    assert fallback[0]["sample_id"] == baseline[0]["sample_id"]
    assert fallback[0]["local_sample_id"] == eval_row["sample_id"]

    accepted = _apply_guard_and_baseline(
        candidate,
        baseline,
        selected_guard=0.0,
        outer_fold=1,
        baseline_sidecar={"file": "f1.zst", "sha256": "0" * 64},
    )
    assert accepted[0]["same_repo_override"] is True
    assert (
        accepted[0]["same_repo_trigger_ms"]
        == candidate[0]["same_repo_candidate_trigger_ms"]
    )


def test_frozen_same_repo_config_loads() -> None:
    config, digest = _load_config(Path("configs/experiments/same_repo_history.yaml"))
    assert config["arm"] == "same_repo"
    assert len(digest) == 64


def test_declared_trace_inventory_must_match_live_corpus(tmp_path: Path) -> None:
    declared: dict[str, dict[str, object]] = {}
    live: dict[str, dict[str, str]] = {}
    for role, digest in (("initialization", "1" * 64), ("development", "2" * 64)):
        trace_path = str((tmp_path / f"{role}.jsonl").resolve())
        inventory_path = tmp_path / f"{role}.sha256"
        inventory_path.write_text(f"{digest}  {trace_path}\n", encoding="utf-8")
        declared[role] = {
            "file": str(inventory_path),
            "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            "trace_count": 1,
        }
        live[role] = {trace_path: digest}

    _verify_trace_inventories(declared, live["initialization"], live["development"])
    with pytest.raises(
        ValueError, match="live traces differ from the declared pre-run inventory"
    ):
        _verify_trace_inventories(
            declared,
            {next(iter(live["initialization"])): "3" * 64},
            live["development"],
        )


def test_staged_output_publication_rolls_back_partial_set(tmp_path: Path) -> None:
    staged = [tmp_path / "one.partial", tmp_path / "two.partial"]
    for path in staged:
        path.write_text(path.name, encoding="utf-8")
    destinations = [tmp_path / "one", tmp_path / "missing" / "two"]

    with pytest.raises(OSError):
        _publish_staged_outputs(staged, destinations)

    assert not destinations[0].exists()
    assert not any(path.exists() for path in staged)


def test_staged_publication_never_clobbers_interleaving_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "result.partial"
    destination = tmp_path / "result"
    staged.write_text("ours", encoding="utf-8")
    real_link = os.link

    def raced_link(source: Path, target: Path) -> None:
        Path(target).write_text("theirs", encoding="utf-8")
        real_link(source, target)

    monkeypatch.setattr(os, "link", raced_link)
    with pytest.raises(FileExistsError):
        _publish_staged_outputs([staged], [destination])

    assert destination.read_text(encoding="utf-8") == "theirs"
