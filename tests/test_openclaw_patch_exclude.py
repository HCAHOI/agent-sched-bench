"""Unit tests for source-only model_patch extraction.

This code gates the SWE-bench/SWE-rebench resolved verdict: the extracted
patch must contain the agent's real source edits and nothing else. The
container path runs `git add -A -- . <exclude pathspecs>` then
`git diff <base> -- . <exclude pathspecs>` (see
`agents.openclaw.eval.types.git_diff_excluding`).

The two failure modes that matter:
  * dropping a real source change  -> spurious *unresolved* (integrity bug)
  * leaking venv/egg-info/pycache  -> patch noise + slow `git add` pathstat

These tests pin both: a nested dir sharing a generic runtime-state name
(`memory/`) must SURVIVE, while venv/egg-info/__pycache__ at any depth must
be DROPPED.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from agents.openclaw.eval.types import EvalResult, git_diff_excluding  # noqa: E402


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _init_repo(root: Path) -> str:
    """Init a repo with a single committed file; return the base commit sha."""
    _git(root, "init")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "README.md").write_text("base\n")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "base")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()


def _write(root: Path, rel: str, content: str = "x\n") -> None:
    fp = root / rel
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content)


def _patched_files(stdout: str) -> set[str]:
    return set(re.findall(r"^\+\+\+ b/(.+)$", stdout, re.M))


def test_patch_keeps_source_drops_runtime_and_detritus(tmp_path: Path) -> None:
    base = _init_repo(tmp_path)

    # Legitimate source the agent created -> MUST appear in the patch.
    _write(tmp_path, "src/new_module.py")
    # A real source dir that happens to share a generic runtime-state name.
    # Anchored exclusion must NOT drop this.
    _write(tmp_path, "src/pkg/memory/store.py")

    # OpenClaw runtime state at the workspace root -> dropped (anchored).
    _write(tmp_path, "memory/state.json")
    _write(tmp_path, "sessions/s.json")
    _write(tmp_path, ".openclaw/x")
    _write(tmp_path, "trace.jsonl")

    # Python env detritus at top-level AND nested -> dropped at any depth.
    _write(tmp_path, "venv/lib/site.py")
    _write(tmp_path, "src/sub/venv/lib/site.py")
    _write(tmp_path, "pkg.egg-info/PKG-INFO")
    _write(tmp_path, "src/pkg/foo.egg-info/PKG-INFO")
    _write(tmp_path, "__pycache__/a.pyc")
    _write(tmp_path, "src/pkg/__pycache__/b.pyc")

    result = git_diff_excluding(
        tmp_path, base, EvalResult.exclude_pathspecs(), add_excludes=True
    )

    assert result.returncode == 0
    assert _patched_files(result.stdout) == {
        "src/new_module.py",
        "src/pkg/memory/store.py",
    }


def test_nested_egg_info_is_dropped(tmp_path: Path) -> None:
    """Regression guard: a bare `*.egg-info` pathspec is a silent no-op, so the
    nested-and-anywhere exclusion must use glob magic. Pins against reverting."""
    base = _init_repo(tmp_path)
    _write(tmp_path, "src/real.py")
    _write(tmp_path, "deep/a/b/c.egg-info/PKG-INFO")

    result = git_diff_excluding(
        tmp_path, base, EvalResult.exclude_pathspecs(), add_excludes=True
    )
    assert _patched_files(result.stdout) == {"src/real.py"}


def test_exclude_pathspecs_forms() -> None:
    specs = EvalResult.exclude_pathspecs()
    # Generic runtime-state names are anchored (top-level only).
    assert ":(exclude)memory" in specs
    assert ":(exclude)build" in specs
    # Env detritus is excluded at any depth via both entry + subtree globs.
    for name in ("venv", ".venv", "*.egg-info", "__pycache__"):
        assert f":(exclude,glob)**/{name}" in specs
        assert f":(exclude,glob)**/{name}/**" in specs
    # No name is globbed AND anchored.
    assert ":(exclude,glob)**/memory/**" not in specs


def test_timeout_branch_returns_124_without_raising(tmp_path: Path) -> None:
    base = _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("changed\n")
    # Sub-microsecond timeouts force both the add and diff TimeoutExpired
    # branches; they must be swallowed (rc=124), not propagate. A regression to
    # stdlib `logging` here would raise TypeError on the brace-kwarg warning.
    result = git_diff_excluding(
        tmp_path,
        base,
        EvalResult.exclude_pathspecs(),
        add_excludes=True,
        add_timeout=1e-6,
        diff_timeout=1e-6,
    )
    assert result.returncode == 124
    assert result.stdout == ""
