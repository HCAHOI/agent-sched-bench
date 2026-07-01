"""Tests for task-container runtime helpers."""

from __future__ import annotations

import json
import ssl
import subprocess
import urllib.error
from pathlib import Path

import pytest

from agents.openclaw.runtime_deps import OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS
from trace_collect.runtime import task_container as task_container_module
from trace_collect.runtime.task_container import (
    TaskContainerExecConfig,
    TaskContainerRunResult,
    bootstrap_task_container_python,
    preflight_task_container_runtime,
    project_mount_args,
    resolve_task_container_exec_config,
    resolve_running_container_exec_config,
    run_task_container_agent,
)


_PYTHON_FINGERPRINT = {
    "version": "3.12.1",
    "implementation": "CPython",
    "cache_tag": "cpython-312",
    "ext_suffix": ".cpython-312-x86_64-linux-gnu.so",
    "machine": "x86_64",
    "libc": "glibc 2.36",
    "os_id": "debian",
    "os_version_id": "12",
}


@pytest.fixture(autouse=True)
def _isolate_shared_bootstrap_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "trace_collect.runtime.task_container._SHARED_BOOTSTRAP_CACHE",
        tmp_path / "shared-bootstrap-cache",
    )
    monkeypatch.setattr(
        "trace_collect.runtime.task_container._container_python_fingerprint",
        lambda **_: dict(_PYTHON_FINGERPRINT),
    )


def test_project_mount_args_include_attempt_dir_and_repo(
    tmp_path: Path,
) -> None:
    args = project_mount_args(tmp_path / "attempt")
    joined = " ".join(args)

    assert str((tmp_path / "attempt").resolve()) in joined
    assert str((Path(__file__).resolve().parents[1]).resolve()) in joined


def test_resolve_task_container_exec_config_uses_bootstrap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "trace_collect.runtime.task_container._inspect_image_platform",
        lambda image, *, container_executable: "linux/amd64",
    )

    config = resolve_task_container_exec_config(
        attempt_dir=tmp_path / "attempt",
        image="localhost/example:latest",
        container_executable="docker",
    )

    assert isinstance(config, TaskContainerExecConfig)
    assert config.bootstrap is True
    assert config.runtime == "/usr/bin/python3"
    assert config.image_platform == "linux/amd64"
    assert config.start_extra_args[:2] == ("--platform", "linux/amd64")
    assert all("/etc:/etc:ro" not in arg for arg in config.start_extra_args)
    assert config.bootstrap_site_dir is not None
    assert str(config.bootstrap_site_dir) in config.pythonpath


def test_resolve_running_container_exec_config_probes_python(monkeypatch) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath="/deps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=Path("/tmp/pydeps"),
        image_platform="linux/amd64",
    )
    seen: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        seen["cmd"] = args[0]

        class Result:
            returncode = 0
            stdout = "/opt/miniconda3/bin/python3\n"
            stderr = ""

        return Result()

    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    resolved = resolve_running_container_exec_config(
        container_id="cid-1",
        exec_config=exec_config,
        container_executable="docker",
    )

    cmd = seen["cmd"]
    assert isinstance(cmd, list)
    assert "/opt/miniconda3/bin/python3" in cmd
    assert resolved.runtime == "/opt/miniconda3/bin/python3"
    assert resolved.pythonpath == exec_config.pythonpath


def test_resolve_running_container_exec_config_skips_invalid_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bad_python = tmp_path / "python3.10"
    good_python = tmp_path / "miniconda-python3"
    bad_python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    good_python.write_text(
        "#!/bin/sh\n"
        'case "$2" in\n'
        "  *version_info*) exit 0 ;;\n"
        '  *sys.executable*) echo "$0"; exit 0 ;;\n'
        "esac\n"
        "exit 1\n",
        encoding="utf-8",
    )
    bad_python.chmod(0o755)
    good_python.chmod(0o755)
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath="/deps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=Path("/tmp/pydeps"),
        image_platform="linux/amd64",
    )
    monkeypatch.setattr(
        "trace_collect.runtime.task_container._CONTAINER_PYTHON_CANDIDATES",
        (str(bad_python), str(good_python)),
    )
    real_run = subprocess.run

    def fake_run(*args, **kwargs):
        cmd = args[0]
        candidates = cmd[cmd.index("--") + 1 :]
        return real_run(
            ["/bin/sh", "-s", "--", *candidates],
            input=kwargs["input"],
            capture_output=True,
            text=True,
            check=False,
            timeout=kwargs["timeout"],
        )

    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    resolved = resolve_running_container_exec_config(
        container_id="cid-1",
        exec_config=exec_config,
        container_executable="docker",
    )

    assert resolved.runtime == str(good_python)


def test_resolve_running_container_exec_config_raises_without_python(
    monkeypatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath="/deps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=Path("/tmp/pydeps"),
        image_platform="linux/amd64",
    )

    def fake_run(*args, **kwargs):
        class Result:
            returncode = 1
            stdout = "probe stdout\n"
            stderr = "probe stderr\n"

        return Result()

    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    try:
        resolve_running_container_exec_config(
            container_id="cid-1",
            exec_config=exec_config,
            container_executable="docker",
        )
    except RuntimeError as exc:
        assert "no Python >=3.11 interpreter found" in str(exc)
        assert "stdout: probe stdout" in str(exc)
        assert "stderr: probe stderr" in str(exc)
    else:
        raise AssertionError("expected probe failure")


def test_bootstrap_task_container_python_uses_resolved_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath=f"{tmp_path}/pydeps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=tmp_path / "pydeps",
        image_platform="linux/amd64",
    )
    seen: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"print('ok')\n"

    def fake_urlopen(url: str, timeout: int):
        seen["url"] = url
        seen["timeout"] = timeout
        return FakeResponse()

    def fake_run(*args, **kwargs):
        seen["cmd"] = args[0]
        seen["input"] = kwargs["input"]

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)
    monkeypatch.setenv("PIP_INDEX_URL", "https://host-only.invalid/simple")
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:7897")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7897")
    monkeypatch.setenv(
        "TASK_CONTAINER_PIP_EXTRA_INDEX_URL", "https://extra.example/simple"
    )
    monkeypatch.setenv("TASK_CONTAINER_HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("TASK_CONTAINER_SSL_CERT_FILE", "/certs/ca.pem")

    bootstrap_task_container_python(
        container_id="cid-1",
        exec_config=exec_config,
        extra_requirements=("mcp>=1.0",),
        container_executable="docker",
    )

    assert seen["url"] == "https://bootstrap.pypa.io/get-pip.py"
    assert "/usr/bin/python3" in seen["cmd"]
    input_script = str(seen["input"])
    assert "https://host-only.invalid/simple" not in input_script
    assert "https://pypi.org/simple" in input_script
    assert "PIP_NO_INDEX" not in input_script
    assert "explicit_env_map" in input_script
    assert "TASK_CONTAINER_PIP_EXTRA_INDEX_URL" in input_script
    assert "TASK_CONTAINER_HTTPS_PROXY" in input_script
    assert "TASK_CONTAINER_SSL_CERT_FILE" in input_script
    assert 'env["PIP_CONFIG_FILE"] = os.devnull' in input_script
    for requirement in OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS:
        assert requirement in input_script
    assert "mcp>=1.0" in str(seen["input"])
    for heavy_dep in (
        "datasets",
        "terminal-bench",
        "trafilatura",
    ):
        assert heavy_dep not in input_script


def test_bootstrap_task_container_python_retries_transient_get_pip_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath=f"{tmp_path}/pydeps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=tmp_path / "pydeps",
        image_platform="linux/amd64",
    )
    seen: dict[str, object] = {"attempts": 0, "sleeps": []}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"print('ok')\n"

    def fake_urlopen(url: str, timeout: int):
        seen["attempts"] = int(seen["attempts"]) + 1
        if int(seen["attempts"]) < 3:
            raise urllib.error.URLError(ssl.SSLEOFError("eof"))
        seen["url"] = url
        seen["timeout"] = timeout
        return FakeResponse()

    def fake_sleep(delay: float) -> None:
        sleeps = seen["sleeps"]
        assert isinstance(sleeps, list)
        sleeps.append(delay)

    def fake_run(*args, **kwargs):
        del args, kwargs

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr("trace_collect.runtime.task_container.time.sleep", fake_sleep)
    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    bootstrap_task_container_python(
        container_id="cid-1",
        exec_config=exec_config,
        extra_requirements=(),
        container_executable="docker",
    )

    assert seen["attempts"] == 3
    assert seen["sleeps"] == [1.0, 2.0]
    assert seen["url"] == "https://bootstrap.pypa.io/get-pip.py"


def test_bootstrap_task_container_python_does_not_retry_http_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath=f"{tmp_path}/pydeps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=tmp_path / "pydeps",
        image_platform="linux/amd64",
    )
    seen = {"attempts": 0}

    def fake_urlopen(url: str, timeout: int):
        del url, timeout
        seen["attempts"] += 1
        raise urllib.error.HTTPError(
            "https://bootstrap.pypa.io/get-pip.py",
            404,
            "not found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        "trace_collect.runtime.task_container.time.sleep", lambda *_: None
    )

    with pytest.raises(urllib.error.HTTPError):
        bootstrap_task_container_python(
            container_id="cid-1",
            exec_config=exec_config,
            extra_requirements=(),
            container_executable="docker",
        )

    assert seen["attempts"] == 1


def _bootstrap_cache_key_for(
    exec_config: TaskContainerExecConfig,
    requirements: tuple[str, ...] | None = None,
) -> str:
    pip_fingerprint = task_container_module._bootstrap_pip_resolution_fingerprint(
        "https://pypi.org/simple"
    )
    return task_container_module._bootstrap_cache_key(
        requirements=requirements or tuple(OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS),
        runtime=exec_config.runtime,
        pip_index_url="https://pypi.org/simple",
        arch="amd64",
        image_platform=exec_config.image_platform,
        python_fingerprint=dict(_PYTHON_FINGERPRINT),
        pip_resolution_fingerprint=pip_fingerprint,
    )


def _write_ready_shared_cache(
    exec_config: TaskContainerExecConfig,
    *,
    requirements: tuple[str, ...] | None = None,
    packages: list[str] | None = None,
    userbase_packages: list[str] | None = None,
    generation: str = "gen-existing",
) -> Path:
    resolved_requirements = requirements or tuple(
        OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS
    )
    cache_key = _bootstrap_cache_key_for(exec_config, resolved_requirements)
    cache_root = (
        task_container_module._shared_bootstrap_dir(exec_config.image_platform)
        / cache_key
    )
    generation_dir = cache_root / generation
    site_dir = generation_dir / "pydeps"
    userbase = generation_dir / ".pyuserbase"
    site_dir.mkdir(parents=True)
    package_list = ["openai==2.0"] if packages is None else packages
    userbase_package_list = (
        ["pip==24.0"] if userbase_packages is None else userbase_packages
    )
    for package in package_list:
        (site_dir / f"{package.replace('==', '-')}.dist-info").mkdir()
    (userbase / "bin").mkdir(parents=True)
    (userbase / "bin" / "pip").write_text("", encoding="utf-8")
    userbase_site = userbase / "lib" / "python3.12" / "site-packages"
    userbase_site.mkdir(parents=True)
    for package in userbase_package_list:
        (userbase_site / f"{package.replace('==', '-')}.dist-info").mkdir()
    pip_fingerprint = task_container_module._bootstrap_pip_resolution_fingerprint(
        "https://pypi.org/simple"
    )
    (site_dir / ".bootstrap-ready.json").write_text(
        json.dumps(
            {
                "requirements": list(resolved_requirements),
                "python": exec_config.runtime,
                "pip_index_url": "https://pypi.org/simple",
                "arch": "amd64",
                "image_platform": exec_config.image_platform,
                "python_fingerprint": dict(_PYTHON_FINGERPRINT),
                "pip_resolution_fingerprint": pip_fingerprint,
                "cache_key": cache_key,
                "packages": package_list,
                "userbase_packages": userbase_package_list,
            }
        ),
        encoding="utf-8",
    )
    (cache_root / "current.json").write_text(
        json.dumps(
            {
                "generation": generation,
                "cache_key": cache_key,
                "python_fingerprint": dict(_PYTHON_FINGERPRINT),
                "pip_resolution_fingerprint": pip_fingerprint,
            }
        ),
        encoding="utf-8",
    )
    return site_dir


def test_bootstrap_task_container_python_ignores_attempt_local_partial_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath=f"{tmp_path}/pydeps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=tmp_path / "pydeps",
        image_platform="linux/amd64",
    )
    exec_config.bootstrap_site_dir.mkdir(parents=True, exist_ok=True)
    stale_pydeps = exec_config.bootstrap_site_dir / "stale.txt"
    stale_pydeps.write_text("stale", encoding="utf-8")
    userbase = exec_config.bootstrap_site_dir.parent / ".pyuserbase"
    userbase.mkdir(parents=True, exist_ok=True)
    stale_userbase = userbase / "stale.txt"
    stale_userbase.write_text("stale", encoding="utf-8")
    seen: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"print('ok')\n"

    def fake_urlopen(url: str, timeout: int):
        seen["url"] = url
        return FakeResponse()

    def fake_run(*args, **kwargs):
        seen["cmd"] = args[0]

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    resolved = bootstrap_task_container_python(
        container_id="cid-1",
        exec_config=exec_config,
        extra_requirements=(),
        container_executable="docker",
    )

    assert stale_pydeps.exists()
    assert stale_userbase.exists()
    assert seen["url"] == "https://bootstrap.pypa.io/get-pip.py"
    assert resolved.bootstrap_site_dir is not None
    assert resolved.bootstrap_site_dir != exec_config.bootstrap_site_dir
    assert "shared-bootstrap-cache" in str(resolved.bootstrap_site_dir)


def test_bootstrap_task_container_python_fails_on_generation_cleanup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath=f"{tmp_path}/pydeps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=tmp_path / "pydeps",
        image_platform="linux/amd64",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"print('ok')\n"

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.urllib.request.urlopen",
        lambda url, timeout: FakeResponse(),
    )

    def fake_run(*args, **kwargs):
        class Result:
            returncode = 1
            stdout = "bootstrap failed"
            stderr = ""

        return Result()

    def fake_rmtree(path: Path) -> None:
        raise OSError(f"cannot remove {path}")

    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)
    monkeypatch.setattr(
        "trace_collect.runtime.task_container.shutil.rmtree", fake_rmtree
    )

    with pytest.raises(OSError, match="cannot remove"):
        bootstrap_task_container_python(
            container_id="cid-1",
            exec_config=exec_config,
            extra_requirements=(),
            container_executable="docker",
        )


def test_bootstrap_task_container_python_publishes_new_generation_when_marker_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath=f"{tmp_path}/pydeps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=tmp_path / "pydeps",
        image_platform="linux/amd64",
    )
    stale_site_dir = _write_ready_shared_cache(
        exec_config,
        requirements=("openai>=2.0,<3.0",),
        packages=["openai==2.0"],
        generation="gen-stale",
    )
    stale_file = stale_site_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")
    seen: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"print('ok')\n"

    def fake_urlopen(url: str, timeout: int):
        seen["url"] = url
        return FakeResponse()

    def fake_run(*args, **kwargs):
        seen["cmd"] = args[0]

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    resolved = bootstrap_task_container_python(
        container_id="cid-1",
        exec_config=exec_config,
        extra_requirements=(),
        container_executable="docker",
    )

    assert stale_file.exists()
    assert seen["url"] == "https://bootstrap.pypa.io/get-pip.py"
    assert resolved.bootstrap_site_dir is not None
    assert resolved.bootstrap_site_dir != stale_site_dir


def test_bootstrap_task_container_python_reuses_valid_shared_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath=f"{tmp_path}/pydeps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=tmp_path / "pydeps",
        image_platform="linux/amd64",
    )
    site_dir = _write_ready_shared_cache(exec_config)

    def fail_run(*args, **kwargs):
        raise AssertionError("shared cache hit should not run bootstrap")

    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fail_run)

    resolved = bootstrap_task_container_python(
        container_id="cid-1",
        exec_config=exec_config,
        extra_requirements=(),
        container_executable="docker",
    )

    assert resolved.bootstrap_site_dir == site_dir
    assert str(site_dir) in resolved.pythonpath


def test_bootstrap_task_container_python_rebuilds_when_pip_resolution_env_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath=f"{tmp_path}/pydeps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=tmp_path / "pydeps",
        image_platform="linux/amd64",
    )
    site_dir = _write_ready_shared_cache(exec_config)
    seen: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"print('ok')\n"

    def fake_urlopen(url: str, timeout: int):
        seen["url"] = url
        return FakeResponse()

    def fake_run(*args, **kwargs):
        seen["cmd"] = args[0]

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setenv(
        "TASK_CONTAINER_PIP_EXTRA_INDEX_URL", "https://extra.example/simple"
    )
    monkeypatch.setattr(
        "trace_collect.runtime.task_container.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    resolved = bootstrap_task_container_python(
        container_id="cid-1",
        exec_config=exec_config,
        extra_requirements=(),
        container_executable="docker",
    )

    assert resolved.bootstrap_site_dir is not None
    assert resolved.bootstrap_site_dir != site_dir
    assert seen["url"] == "https://bootstrap.pypa.io/get-pip.py"


def test_bootstrap_task_container_python_rebuilds_contaminated_shared_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exec_config = TaskContainerExecConfig(
        runtime="/usr/bin/python3",
        pythonpath=f"{tmp_path}/pydeps:/repo/src:/repo",
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=tmp_path / "pydeps",
        image_platform="linux/amd64",
    )
    stale_site_dir = _write_ready_shared_cache(
        exec_config,
        packages=[],
        generation="gen-contaminated",
    )
    stale_package = stale_site_dir / "stale-1.0.dist-info"
    stale_package.mkdir()
    seen: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"print('ok')\n"

    def fake_urlopen(url: str, timeout: int):
        seen["url"] = url
        return FakeResponse()

    def fake_run(*args, **kwargs):
        seen["cmd"] = args[0]

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    resolved = bootstrap_task_container_python(
        container_id="cid-1",
        exec_config=exec_config,
        extra_requirements=(),
        container_executable="docker",
    )

    assert stale_package.exists()
    assert seen["url"] == "https://bootstrap.pypa.io/get-pip.py"
    assert resolved.bootstrap_site_dir is not None
    assert resolved.bootstrap_site_dir != stale_site_dir


def test_preflight_task_container_runtime_reads_runtime_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.chdir(tmp_path)
    attempt_dir = Path("relative-attempt")
    result_path = (
        tmp_path
        / "relative-attempt"
        / "_task_container_runtime"
        / "preflight"
        / "result.json"
    )

    def fake_exec(**kwargs):
        request = json.loads(Path(kwargs["request_path"]).read_text(encoding="utf-8"))
        seen.update(request)
        seen["runtime"] = kwargs["runtime"]
        seen["pythonpath"] = kwargs["pythonpath"]
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "runtime_proof": {
                        "container_id": "cid-1",
                        "hostname": "host-a",
                        "cwd": "/testbed",
                        "python_executable": "/usr/bin/python3",
                        "python_prefix": "/usr",
                        "project_root": "/repo",
                        "sys_path": ["/repo/src"],
                    },
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    proof = preflight_task_container_runtime(
        container_id="cid-1",
        attempt_dir=attempt_dir,
        imports=["trace_collect.runtime.entrypoint", "agents.openclaw.eval.runner"],
        runtime="/usr/bin/python3",
        pythonpath="/tmp/site:/repo/src:/repo",
        container_executable="docker",
    )

    assert proof.container_id == "cid-1"
    assert proof.python_executable == "/usr/bin/python3"
    assert Path(str(seen["result_path"])).is_absolute()
    assert Path(str(seen["writable_probe"])).is_absolute()
    assert Path(str(seen["result_path"])) == result_path
    assert seen["imports"] == [
        "trace_collect.runtime.entrypoint",
        "agents.openclaw.eval.runner",
    ]
    assert seen["runtime"] == "/usr/bin/python3"
    assert seen["pythonpath"] == "/tmp/site:/repo/src:/repo"


def test_run_task_container_agent_reads_result_and_writes_raw_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_path = tmp_path / "_task_container_runtime" / "openclaw" / "run.result.json"
    stdout_path = tmp_path / "_task_container_runtime" / "openclaw" / "stdout.txt"
    stderr_path = tmp_path / "_task_container_runtime" / "openclaw" / "stderr.txt"
    trace_path = tmp_path / "_task_container_runtime" / "openclaw" / "trace.jsonl"

    def fake_exec(**kwargs):
        assert kwargs["runtime"] == "/usr/bin/python3"
        assert kwargs["pythonpath"] == "/tmp/site:/repo/src:/repo"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "success": True,
                    "trace_path": str(trace_path),
                    "model_patch": "diff --git a/x b/x",
                    "exit_status": "Submitted",
                    "error": None,
                    "n_iterations": 3,
                    "total_llm_ms": 1.0,
                    "total_tool_ms": 2.0,
                    "total_tokens": 4,
                    "runtime_proof": {"hostname": "container-a"},
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = "stdout text"
            stderr = "stderr text"

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    result = run_task_container_agent(
        container_id="cid-2",
        timeout=10,
        runtime="/usr/bin/python3",
        pythonpath="/tmp/site:/repo/src:/repo",
        container_executable="docker",
        request={
            "scaffold": "openclaw",
            "result_path": str(result_path),
            "trace_file": str(trace_path),
            "raw_stdout_path": str(stdout_path),
            "raw_stderr_path": str(stderr_path),
        },
    )

    assert isinstance(result, TaskContainerRunResult)
    assert result.success is True
    assert stdout_path.read_text(encoding="utf-8") == "stdout text"
    assert stderr_path.read_text(encoding="utf-8") == "stderr text"


def test_run_task_container_agent_preserves_existing_raw_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_path = tmp_path / "_task_container_runtime" / "openclaw" / "run.result.json"
    stdout_path = tmp_path / "_task_container_runtime" / "openclaw" / "stdout.txt"
    stderr_path = tmp_path / "_task_container_runtime" / "openclaw" / "stderr.txt"
    trace_path = tmp_path / "_task_container_runtime" / "openclaw" / "trace.jsonl"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("container stdout", encoding="utf-8")
    stderr_path.write_text("container stderr", encoding="utf-8")

    def fake_exec(**kwargs):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "success": True,
                    "trace_path": str(trace_path),
                    "model_patch": "diff --git a/x b/x",
                    "exit_status": "Submitted",
                    "error": None,
                    "n_iterations": 3,
                    "total_llm_ms": 1.0,
                    "total_tool_ms": 2.0,
                    "total_tokens": 4,
                    "runtime_proof": {"hostname": "container-a"},
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    run_task_container_agent(
        container_id="cid-2",
        timeout=10,
        runtime="/usr/bin/python3",
        pythonpath="/tmp/site:/repo/src:/repo",
        container_executable="docker",
        request={
            "kind": "run_openclaw",
            "scaffold": "openclaw",
            "result_path": str(result_path),
            "trace_file": str(trace_path),
            "raw_stdout_path": str(stdout_path),
            "raw_stderr_path": str(stderr_path),
        },
    )

    assert stdout_path.read_text(encoding="utf-8") == "container stdout"
    assert stderr_path.read_text(encoding="utf-8") == "container stderr"


def test_run_task_container_agent_updates_raw_logs_from_exec_pipe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_path = tmp_path / "_task_container_runtime" / "openclaw" / "run.result.json"
    stdout_path = tmp_path / "_task_container_runtime" / "openclaw" / "stdout.txt"
    stderr_path = tmp_path / "_task_container_runtime" / "openclaw" / "stderr.txt"
    trace_path = tmp_path / "_task_container_runtime" / "openclaw" / "trace.jsonl"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("tee stdout", encoding="utf-8")
    stderr_path.write_text("tee stderr", encoding="utf-8")

    def fake_exec(**kwargs):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "success": True,
                    "trace_path": str(trace_path),
                    "model_patch": "diff --git a/x b/x",
                    "exit_status": "Submitted",
                    "error": None,
                    "runtime_proof": {},
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = "pipe stdout"
            stderr = "pipe stderr"

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    run_task_container_agent(
        container_id="cid-2",
        timeout=10,
        runtime="/usr/bin/python3",
        pythonpath="/tmp/site:/repo/src:/repo",
        container_executable="docker",
        request={
            "kind": "run_openclaw",
            "scaffold": "openclaw",
            "result_path": str(result_path),
            "trace_file": str(trace_path),
            "raw_stdout_path": str(stdout_path),
            "raw_stderr_path": str(stderr_path),
        },
    )

    assert stdout_path.read_text(encoding="utf-8") == "pipe stdout"
    assert stderr_path.read_text(encoding="utf-8") == "pipe stderr"


def test_run_task_container_agent_prefers_explicit_success_over_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_path = tmp_path / "_task_container_runtime" / "openclaw" / "run.result.json"
    stdout_path = tmp_path / "_task_container_runtime" / "openclaw" / "stdout.txt"
    stderr_path = tmp_path / "_task_container_runtime" / "openclaw" / "stderr.txt"
    trace_path = tmp_path / "_task_container_runtime" / "openclaw" / "trace.jsonl"

    def fake_exec(**kwargs):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "success": True,
                    "trace_path": str(trace_path),
                    "model_patch": "",
                    "exit_status": "completed",
                    "error": None,
                    "runtime_proof": {},
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    result = run_task_container_agent(
        container_id="cid-3",
        timeout=10,
        container_executable="docker",
        request={
            "kind": "run_openclaw",
            "scaffold": "openclaw",
            "result_path": str(result_path),
            "trace_file": str(trace_path),
            "raw_stdout_path": str(stdout_path),
            "raw_stderr_path": str(stderr_path),
        },
    )

    assert result.success is True


def test_run_task_container_agent_timeout_writes_partial_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_path = tmp_path / "_task_container_runtime" / "openclaw" / "run.result.json"
    stdout_path = tmp_path / "_task_container_runtime" / "openclaw" / "stdout.txt"
    stderr_path = tmp_path / "_task_container_runtime" / "openclaw" / "stderr.txt"
    trace_path = tmp_path / "_task_container_runtime" / "openclaw" / "trace.jsonl"

    def fake_exec(**kwargs):
        raise __import__("subprocess").TimeoutExpired(
            cmd="podman exec ...",
            timeout=10,
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    try:
        run_task_container_agent(
            container_id="cid-2",
            timeout=10,
            container_executable="docker",
            request={
                "kind": "run_openclaw",
                "scaffold": "openclaw",
                "result_path": str(result_path),
                "trace_file": str(trace_path),
                "raw_stdout_path": str(stdout_path),
                "raw_stderr_path": str(stderr_path),
            },
        )
    except RuntimeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected timeout failure")

    assert stdout_path.read_text(encoding="utf-8") == "partial stdout"
    assert stderr_path.read_text(encoding="utf-8") == "partial stderr"


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_preflight_task_container_runtime_passes_container_executable_to_exec(
    tmp_path: Path,
    monkeypatch,
    container_executable: str,
) -> None:
    seen: dict[str, object] = {}
    result_path = (
        tmp_path / "attempt" / "_task_container_runtime" / "preflight" / "result.json"
    )

    def fake_exec(**kwargs):
        seen["container_executable"] = kwargs["container_executable"]
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "runtime_proof": {
                        "container_id": "cid-1",
                        "hostname": "host-a",
                        "cwd": "/testbed",
                        "python_executable": "/usr/bin/python3",
                        "python_prefix": "/usr",
                        "project_root": "/repo",
                        "sys_path": ["/repo/src"],
                    }
                }
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.exec_task_container_entrypoint",
        fake_exec,
    )

    preflight_task_container_runtime(
        container_id="cid-1",
        attempt_dir=tmp_path / "attempt",
        container_executable=container_executable,
    )

    assert seen["container_executable"] == container_executable
