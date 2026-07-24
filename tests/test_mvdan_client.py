from __future__ import annotations

import os
from pathlib import Path
import signal

import pytest

from tool_resource.mvdan_client import MvdanClient, MvdanClientError


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
