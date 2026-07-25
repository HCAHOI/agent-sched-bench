from __future__ import annotations

import os
from pathlib import Path
import signal
import textwrap

import pytest

from tool_resource.mvdan_client import (
    ADAPTER_PROTOCOL_VERSION,
    MvdanClient,
    MvdanClientError,
    PARSER_VERSION,
    REQUIRED_CAPABILITIES,
    default_binary_path,
    ensure_compatible_adapter,
)


def _write_fake_adapter(
    path: Path,
    *,
    advertise_protocol: bool,
    capabilities: list[str] | None = None,
    parse_marker: Path | None = None,
) -> None:
    advertised_capabilities = (
        sorted(REQUIRED_CAPABILITIES) if capabilities is None else capabilities
    )
    protocol = (
        f'"protocol": {{"version": {ADAPTER_PROTOCOL_VERSION}, '
        f'"capabilities": {advertised_capabilities!r}}},'
        if advertise_protocol
        else ""
    )
    marker_action = (
        f"open({str(parse_marker)!r}, 'w').close()"
        if parse_marker is not None
        else "pass"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import json
            import sys

            for line in sys.stdin:
                request = json.loads(line)
                if request["op"] == "parse":
                    {marker_action}
                print(json.dumps({{
                    "id": request["id"],
                    "ok": True,
                    "parser": {{
                        "name": "mvdan.cc/sh/v3",
                        "version": {PARSER_VERSION!r},
                    }},
                    {protocol}
                    "clauses": [],
                    "control_edges": [],
                }}), flush=True)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_two_parses_reuse_one_process() -> None:
    with MvdanClient() as client:
        first = client.parse("echo one")
        pid = client.pid
        second = client.parse("printf two")

        assert first["ok"] and second["ok"]
        assert client.pid == pid
        assert client.start_count == 1


def test_crash_restarts_process_once() -> None:
    with MvdanClient() as client:
        assert client.parse("echo before")["ok"]
        assert client.pid is not None
        os.kill(client.pid, signal.SIGKILL)

        assert client.parse("echo after")["ok"]
        assert client.start_count == 2


def test_missing_binary_names_build_contract(tmp_path: Path) -> None:
    client = MvdanClient(tmp_path / "missing")

    with pytest.raises(
        MvdanClientError,
        match=r"scripts/setup/build_mvdan_adapter\.sh",
    ):
        client.parse("echo one")


def test_cache_identity_includes_protocol_and_parser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert default_binary_path().name == (
        f"mvdan-clause-adapter-protocol-{ADAPTER_PROTOCOL_VERSION}"
        f"-mvdan-{PARSER_VERSION}"
    )


def test_empty_xdg_cache_home_matches_shell_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", "")

    assert default_binary_path().parent == tmp_path / ".cache" / "agent-sched-bench"


def test_old_schema_with_same_parser_is_rejected_before_parse(
    tmp_path: Path,
) -> None:
    parse_marker = tmp_path / "parse-called"
    adapter = tmp_path / "old-adapter"
    _write_fake_adapter(
        adapter,
        advertise_protocol=False,
        parse_marker=parse_marker,
    )

    with pytest.raises(MvdanClientError, match="protocol mismatch"):
        MvdanClient(adapter).parse("echo must-not-run")

    assert not parse_marker.exists()


@pytest.mark.parametrize(
    ("capabilities", "missing"),
    (
        (["word_intents"], "structural_context"),
        (["structural_context"], "word_intents"),
    ),
)
def test_missing_required_capability_is_rejected_before_parse(
    tmp_path: Path,
    capabilities: list[str],
    missing: str,
) -> None:
    parse_marker = tmp_path / "parse-called"
    adapter = tmp_path / "old-capabilities"
    _write_fake_adapter(
        adapter,
        advertise_protocol=True,
        capabilities=capabilities,
        parse_marker=parse_marker,
    )

    with pytest.raises(MvdanClientError, match=missing):
        MvdanClient(adapter).parse("echo must-not-run")

    assert not parse_marker.exists()


def test_preflight_atomically_rebuilds_stale_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    adapter = default_binary_path()
    _write_fake_adapter(adapter, advertise_protocol=False)
    builds: list[tuple[object, object]] = []

    def fake_build(command: object, *, cwd: object, check: bool) -> None:
        assert check
        builds.append((command, cwd))
        replacement = adapter.with_suffix(".replacement")
        _write_fake_adapter(replacement, advertise_protocol=True)
        replacement.replace(adapter)

    monkeypatch.setattr("tool_resource.mvdan_client.subprocess.run", fake_build)

    assert ensure_compatible_adapter() == adapter
    assert len(builds) == 1


def test_preflight_rebuilds_unlaunchable_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    adapter = default_binary_path()
    adapter.parent.mkdir(parents=True)
    adapter.write_text("not an executable format\n", encoding="utf-8")
    adapter.chmod(0o755)
    builds = 0

    def fake_build(command: object, *, cwd: object, check: bool) -> None:
        nonlocal builds
        assert command and cwd and check
        builds += 1
        replacement = adapter.with_suffix(".replacement")
        _write_fake_adapter(replacement, advertise_protocol=True)
        replacement.replace(adapter)

    monkeypatch.setattr("tool_resource.mvdan_client.subprocess.run", fake_build)

    assert ensure_compatible_adapter() == adapter
    assert builds == 1
