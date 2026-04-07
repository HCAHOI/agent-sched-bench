"""Regression test: every public symbol the legacy swebench_data module
exported must still be importable after the shim refactor."""

from __future__ import annotations


EXPECTED_LEGACY_SYMBOLS = {
    "load_swebench_verified",
    "derive_test_cmd",
    "_count_fail_to_pass",
    "select_tool_intensive_tasks",
    "download_and_save",
    "REPO_QUOTAS",
    "HEAVY_REPOS",
}


def test_shim_exposes_all_legacy_symbols() -> None:
    import agents.swebench_data as shim

    missing = EXPECTED_LEGACY_SYMBOLS - set(dir(shim))
    assert not missing, f"Shim missing legacy symbols: {missing}"


def test_shim_repo_quotas_matches_class_level() -> None:
    """REPO_QUOTAS exported by the shim must match the plugin's class-level copy."""
    from agents.swebench_data import REPO_QUOTAS as shim_quotas
    from agents.benchmarks.swe_bench_verified import SWEBenchVerified

    assert dict(shim_quotas) == dict(SWEBenchVerified.CLASS_LEVEL_REPO_QUOTAS)


def test_shim_heavy_repos_matches_class_level() -> None:
    from agents.swebench_data import HEAVY_REPOS as shim_heavy
    from agents.benchmarks.swe_bench_verified import SWEBenchVerified

    assert set(shim_heavy) == set(SWEBenchVerified.CLASS_LEVEL_HEAVY_REPOS)


def test_derive_test_cmd_shim_accepts_list() -> None:
    from agents.swebench_data import derive_test_cmd

    result = derive_test_cmd({"FAIL_TO_PASS": ["tests/a.py::test_x"]})
    assert "tests/a.py::test_x" in result
    assert "pytest" in result


def test_count_fail_to_pass_shim_accepts_list() -> None:
    from agents.swebench_data import _count_fail_to_pass

    assert _count_fail_to_pass({"FAIL_TO_PASS": ["a", "b", "c"]}) == 3
