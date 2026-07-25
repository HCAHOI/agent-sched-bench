from __future__ import annotations

import importlib.util
import inspect

import pytest

from trace_collect.cli import parse_simulate_args
from trace_collect.openclaw_tools import _REPLAY_AGENT_SCRIPT
from trace_collect.simulate_outputs import _split_trace_by_agent
from trace_collect.simulate_types import WorkerReplayResult
from trace_collect.simulator import simulate


@pytest.mark.parametrize("legacy_flag", ["--pacct", "--no-segment-timeline"])
def test_simulate_cli_rejects_removed_legacy_telemetry_flags(legacy_flag: str) -> None:
    with pytest.raises(SystemExit):
        parse_simulate_args(["--manifest", "manifest.yaml", legacy_flag])


def test_simulate_api_has_no_legacy_telemetry_parameters() -> None:
    parameters = inspect.signature(simulate).parameters
    assert "pacct" not in parameters
    assert "segment_timeline" not in parameters


def test_worker_result_has_no_pacct_metadata_channel() -> None:
    assert "pacct_metadata_by_agent" not in WorkerReplayResult.__dataclass_fields__
    assert "metadata_by_agent" not in inspect.signature(_split_trace_by_agent).parameters


def test_replay_agent_has_no_pacct_or_xtrace_instrumentation() -> None:
    assert "OPENCLAW_PACCT" not in _REPLAY_AGENT_SCRIPT
    assert "OPENCLAW_SEGMENT_TIMELINE" not in _REPLAY_AGENT_SCRIPT
    assert "BASH_XTRACEFD" not in _REPLAY_AGENT_SCRIPT
    assert "segment_timeline" not in _REPLAY_AGENT_SCRIPT
    assert "per_process" not in _REPLAY_AGENT_SCRIPT


def test_pacct_decoder_module_is_removed() -> None:
    assert importlib.util.find_spec("trace_collect.pacct") is None
