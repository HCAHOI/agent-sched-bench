from __future__ import annotations

import ast
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.baselines import cachewise_reproduction as reproduction


SCRIPT = Path(__file__).parents[1] / "scripts/baselines/cachewise_reproduction.sh"
OFFICIAL_SCRIPT = Path(__file__).parents[1] / "scripts/baselines/cachewise_official.sh"


def test_policy_payload_uses_one_official_selected_curve(monkeypatch) -> None:
    curve = [
        {"t_ms": 0.0, "prob_still_running": 1.0},
        {"t_ms": 1000.0, "prob_still_running": 0.0},
    ]
    model = object()
    fake_predictor = SimpleNamespace(
        load_models=lambda _models: {"Bash": model},
        canonicalize_text=lambda text: text.strip().lower(),
        select_curve=lambda selected, text: (
            curve,
            0.2 if selected is model and text == "pytest" else -1.0,
            -1,
        ),
    )
    monkeypatch.setattr(
        reproduction, "verify_checkout", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(reproduction, "_load_predictor", lambda _path: fake_predictor)

    policy = reproduction.build_policy(
        Path("predictor"),
        Path("models"),
        "session-a",
        "Bash",
        " PYTEST ",
        125.0,
    )

    assert policy["hints"] == {
        "duration_curve": curve,
        "elapsed_ms": 125.0,
    }
    assert policy["scope"]["session_id"] == "session-a"
    assert policy["predictor_provenance"]["cluster_id"] == -1
    assert policy["predictor_provenance"]["similarity"] == 0.2


def test_idle_and_oracle_are_separate_paper_ablations() -> None:
    assert reproduction.idle_policy("session-a")["scope"] == {
        "session_id": "session-a",
        "idle_session": True,
    }
    oracle = reproduction.oracle_policy("session-a", 5000.0, 1250.0)
    assert oracle["hints"] == {
        "oracle": {"total_duration_ms": 5000.0},
        "elapsed_ms": 1250.0,
    }

    manifest = reproduction.inference_manifest()
    assert manifest["published"]["rebuild_interval_engine_iterations"] == 3
    assert manifest["published"]["miss_feedback"] is None
    assert "rebuild_phase" in manifest["inferred_not_tuned"]
    assert manifest["inferred_not_tuned"]["waiting_ties"] == (
        "arrival time then request ID"
    )
    assert "causal generated-tool-call attachment hook" in manifest["unpublished"]


def test_patch_contract_and_cli_help() -> None:
    patch = reproduction.VLLM_PATCH
    assert reproduction.N_REBUILD == 3
    assert "_CACHEWISE_REBUILD_INTERVAL = 3" in patch
    assert "rebuild_cachewise_heap" in patch
    assert "CachewiseSessionPolicies" in patch
    assert "update_cachewise_session_policy" in patch
    assert "missing_tokens = max(0, req.num_tokens - matched_tokens)" in patch
    assert "cachewise_eviction_score(block.cachewise_policy, now_s)" in patch
    added_lines = "\n".join(
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    assert "hash(name" not in added_lines
    subprocess.run(["bash", "-n", SCRIPT], check=True)
    help_text = subprocess.run(
        ["bash", SCRIPT, "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "conditional-remaining-time eviction" in help_text
    assert "N_rebuild=3" in help_text
    assert "does not publish its tool-call-to-engine attachment hook" in help_text


def test_checkout_integrity_allows_only_explicit_model_artifacts(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "predictor"
    checkout.mkdir()
    subprocess.run(["git", "init", "--quiet", checkout], check=True)
    subprocess.run(
        ["git", "-C", checkout, "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", checkout, "config", "user.name", "Test"], check=True
    )
    source = checkout / "predictor.py"
    source.write_text("clean = True\n")
    subprocess.run(["git", "-C", checkout, "add", "predictor.py"], check=True)
    subprocess.run(
        ["git", "-C", checkout, "commit", "--quiet", "-m", "fixture"], check=True
    )
    remote = "https://example.test/predictor.git"
    subprocess.run(
        ["git", "-C", checkout, "remote", "add", "origin", remote], check=True
    )
    commit = subprocess.run(
        ["git", "-C", checkout, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    allowed = checkout / "tool_duration_prediction/models/all_models.pkl"
    allowed.parent.mkdir(parents=True)
    allowed.write_bytes(b"model")

    reproduction.verify_checkout(
        checkout,
        commit,
        remote,
        allowed_untracked=("tool_duration_prediction/models/",),
    )

    model_source = allowed.parent / "override.py"
    model_source.write_text("pollution = True\n")
    with pytest.raises(RuntimeError, match="unexpected untracked"):
        reproduction.verify_checkout(
            checkout,
            commit,
            remote,
            allowed_untracked=("tool_duration_prediction/models/",),
        )
    model_source.unlink()

    rogue = checkout / "replacement_predictor.py"
    rogue.write_text("pollution = True\n")
    with pytest.raises(RuntimeError, match="unexpected untracked"):
        reproduction.verify_checkout(
            checkout,
            commit,
            remote,
            allowed_untracked=("tool_duration_prediction/models/",),
        )
    rogue.unlink()

    source.write_text("clean = False\n")
    with pytest.raises(RuntimeError, match="tracked changes"):
        reproduction.verify_checkout(checkout, commit, remote)
    subprocess.run(["git", "-C", checkout, "add", "predictor.py"], check=True)
    with pytest.raises(RuntimeError, match="tracked changes"):
        reproduction.verify_checkout(checkout, commit, remote)


def _cached_vllm() -> Path | None:
    candidates = (
        Path.home() / ".cache/agent-sched-bench/cachewise-official-vllm",
        Path.home()
        / ".cache/agent-sched-bench"
        / f"cachewise-vllm-reproduction-{reproduction.VLLM_COMMIT}",
    )
    return next((path for path in candidates if path.is_dir()), None)


@pytest.mark.slow
def test_official_entrypoint_rejects_predictor_source_pollution(
    tmp_path: Path,
) -> None:
    source = (
        Path.home()
        / ".cache/agent-sched-bench"
        / f"cachewise-{reproduction.PREDICTOR_COMMIT}"
    )
    if not source.is_dir():
        pytest.skip("pinned official CacheWise predictor is not cached")
    checkout = tmp_path / "predictor"
    subprocess.run(
        [
            "git",
            "-C",
            source,
            "worktree",
            "add",
            "--quiet",
            "--detach",
            checkout,
            reproduction.PREDICTOR_COMMIT,
        ],
        check=True,
    )
    env = {
        **os.environ,
        "CACHEWISE_CHECKOUT": str(checkout),
        "CACHEWISE_REPRO_PYTHON": sys.executable,
    }
    try:
        model = checkout / "tool_duration_prediction/models/all_models.pkl"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"allowed model artifact")
        subprocess.run(["bash", OFFICIAL_SCRIPT, "verify"], env=env, check=True)

        rogue = checkout / "tool_duration_prediction/override.py"
        rogue.write_text("pollution = True\n")
        rejected = subprocess.run(
            ["bash", OFFICIAL_SCRIPT, "verify"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0
        assert "unexpected untracked" in rejected.stderr
        rogue.unlink()

        infer = checkout / "tool_duration_prediction/infer.py"
        infer.write_text(infer.read_text() + "\n# tracked pollution\n")
        rejected = subprocess.run(
            ["bash", OFFICIAL_SCRIPT, "infer", "--demo"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0
        assert "tracked changes" in rejected.stderr
    finally:
        subprocess.run(
            ["git", "-C", source, "worktree", "remove", "--force", checkout],
            check=True,
        )


@pytest.mark.slow
def test_clean_patch_and_deterministic_eviction_scores(tmp_path: Path) -> None:
    source = _cached_vllm()
    if source is None:
        pytest.skip("pinned official CacheWise vLLM fork is not cached")
    checkout = tmp_path / "vllm"
    subprocess.run(
        [
            "git",
            "-C",
            source,
            "worktree",
            "add",
            "--quiet",
            "--detach",
            checkout,
            reproduction.VLLM_COMMIT,
        ],
        check=True,
    )
    try:
        reproduction.apply_vllm_patch(checkout)
        reproduction.verify_applied_patch(checkout)

        policy_path = checkout / "vllm/v1/core/cachewise_policy.py"
        spec = importlib.util.spec_from_file_location(
            "patched_cachewise_policy", policy_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        raw = {
            "version": 1,
            "scope": {"session_id": "session-a", "idle_session": False},
            "hints": {
                "duration_curve": [
                    {"t_ms": 0.0, "prob_still_running": 1.0},
                    {"t_ms": 1000.0, "prob_still_running": 0.0},
                ],
                "elapsed_ms": 0.0,
            },
        }
        parsed = module.parse_cachewise_policy_body(raw)
        assert parsed is not None
        attached = parsed["hints"]["attached_monotonic_s"]
        assert module.cachewise_eviction_score(parsed, attached) == pytest.approx(0.5)
        assert module.cachewise_eviction_score(
            parsed, attached + 0.5
        ) == pytest.approx(0.25)
        assert math.isinf(
            module.cachewise_eviction_score(
                module.parse_cachewise_policy_body(
                    reproduction.idle_policy("session-a")
                ),
                attached,
            )
        )

        oracle = module.parse_cachewise_policy_body(
            reproduction.oracle_policy("session-a", 5000.0)
        )
        assert oracle is not None
        oracle_attached = oracle["hints"]["attached_monotonic_s"]
        assert module.cachewise_eviction_score(
            oracle, oracle_attached + 2.0
        ) == pytest.approx(3.0)

        first = module.parse_cachewise_policy_body(
            {
                "version": 1,
                "scope": {"session_id": "session-shared"},
                "hints": {
                    "duration_curve": [
                        {"t_ms": 0.0, "prob_still_running": 1.0},
                        {"t_ms": 10000.0, "prob_still_running": 0.0},
                    ]
                },
            }
        )
        second = module.parse_cachewise_policy_body(
            {
                "version": 1,
                "scope": {"session_id": "session-shared"},
                "hints": {
                    "duration_curve": [
                        {"t_ms": 0.0, "prob_still_running": 1.0},
                        {"t_ms": 1000.0, "prob_still_running": 0.0},
                    ]
                },
            }
        )
        assert first is not None and second is not None
        old_block = SimpleNamespace(cachewise_policy=first)
        policies = module.CachewiseSessionPolicies()
        shared = policies.update(first, [old_block])
        assert shared is not None
        first_score = module.cachewise_eviction_score(
            old_block.cachewise_policy,
            first["hints"]["attached_monotonic_s"],
        )
        block_pool_path = checkout / "vllm/v1/core/block_pool.py"
        block_pool_tree = ast.parse(block_pool_path.read_text())
        block_pool_class = next(
            node
            for node in block_pool_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "BlockPool"
        )
        update_method_node = next(
            node
            for node in block_pool_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "update_cachewise_session_policy"
        )
        method_module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                update_method_node,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(method_module)
        method_namespace: dict[str, object] = {}
        exec(compile(method_module, str(block_pool_path), "exec"), method_namespace)
        rebuild_times: list[float] = []
        fake_pool = SimpleNamespace(
            _cachewise_session_policies=policies,
            blocks=[old_block],
            rebuild_cachewise_heap=rebuild_times.append,
        )
        updated = method_namespace["update_cachewise_session_policy"](
            fake_pool, second, 123.0
        )
        new_block = SimpleNamespace(cachewise_policy=policies.canonical(second))
        assert updated is shared
        assert rebuild_times == [123.0]
        assert old_block.cachewise_policy is new_block.cachewise_policy is shared
        update_time = shared["hints"]["attached_monotonic_s"]
        old_score = module.cachewise_eviction_score(
            old_block.cachewise_policy, update_time
        )
        new_score = module.cachewise_eviction_score(
            new_block.cachewise_policy, update_time
        )
        assert first_score == pytest.approx(5.0)
        assert old_score == new_score == pytest.approx(0.5)

        scheduler = (checkout / "vllm/v1/core/sched/scheduler.py").read_text()
        assert "_CACHEWISE_REBUILD_INTERVAL = 3" in scheduler
        assert "additional_blocks = (" in scheduler

        unexpected = checkout / "unexpected.txt"
        unexpected.write_text("pollution\n")
        with pytest.raises(RuntimeError, match="untracked"):
            reproduction.verify_applied_patch(checkout)
        unexpected.unlink()

        readme = checkout / "README.md"
        readme.write_text(readme.read_text() + "\nunrelated edit\n")
        with pytest.raises(RuntimeError, match="differs from the intended patch"):
            reproduction.verify_applied_patch(checkout)
    finally:
        subprocess.run(
            ["git", "-C", source, "worktree", "remove", "--force", checkout],
            check=True,
        )
