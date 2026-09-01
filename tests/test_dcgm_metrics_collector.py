from __future__ import annotations

import csv
import signal
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts.evaluation import collect_dcgm_metrics as collector


_DOUBLE = ord("d")


def _sample(timestamp_us: int, value: float = 0.25) -> collector._Sample:
    return collector._Sample(
        timestamp_us=timestamp_us,
        gpu_id=0,
        field_id=1005,
        field_type=_DOUBLE,
        status=0,
        value=value,
        blank=False,
    )


class _Source:
    double_field_type = _DOUBLE

    def __init__(self, batches: list[list[collector._Sample]]) -> None:
        self.batches = iter(batches)
        self.on_read: Any = None
        self.read_count = 0
        self.closed = False

    def __enter__(self) -> _Source:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.closed = True

    def read(self) -> list[collector._Sample]:
        self.read_count += 1
        if self.on_read is not None:
            self.on_read(self.read_count)
        return next(self.batches)


def _install_fake_signals(monkeypatch: pytest.MonkeyPatch) -> dict[int, Any]:
    handlers: dict[int, Any] = {
        signal.SIGTERM: signal.SIG_DFL,
        signal.SIGINT: signal.SIG_DFL,
    }
    installed: dict[int, Any] = {}

    def set_signal(signum: int, handler: Any) -> Any:
        previous = handlers[signum]
        handlers[signum] = handler
        if callable(handler):
            installed[signum] = handler
        return previous

    monkeypatch.setattr(collector.signal, "signal", set_signal)
    return installed


def test_downsamples_and_keeps_first_sample_after_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _Source(
        [
            [
                _sample(1_000_000, 0.1),
                _sample(1_002_000, 0.2),
                _sample(1_999_000, 0.3),
                _sample(2_000_500, 0.4),
            ],
            [
                _sample(2_400_000, 0.5),
                _sample(2_600_000, 0.6),
                _sample(2_700_000, 0.7),
            ],
        ]
    )
    installed = _install_fake_signals(monkeypatch)
    now_ns = 2_500_000 * 1_000
    monkeypatch.setattr(collector.time, "time_ns", lambda: now_ns)
    monkeypatch.setattr(collector.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(collector, "_DcgmSource", lambda: source)

    def stop_on_second_read(read_count: int) -> None:
        if read_count == 2:
            installed[signal.SIGTERM](signal.SIGTERM, None)

    source.on_read = stop_on_second_read
    output = tmp_path / "dcgm.csv"
    ready = tmp_path / "ready"

    collector.collect(output, ready)

    assert ready.exists()
    assert source.closed
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "timestamp_us": "1000000",
            "gpu_id": "0",
            "field_id": "1005",
            "status": "OK",
            "dram_active_ratio": "0.1",
        },
        {
            "timestamp_us": "2000500",
            "gpu_id": "0",
            "field_id": "1005",
            "status": "OK",
            "dram_active_ratio": "0.4",
        },
        {
            "timestamp_us": "2600000",
            "gpu_id": "0",
            "field_id": "1005",
            "status": "OK",
            "dram_active_ratio": "0.6",
        },
    ]


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (replace(_sample(1), gpu_id=1), "expected GPU 0"),
        (replace(_sample(1), field_id=1006), "expected field 1005"),
        (replace(_sample(1), field_type=ord("i")), "return a double"),
        (replace(_sample(1), status=4), "status 4"),
        (replace(_sample(1), blank=True), "blank value"),
        (replace(_sample(1), timestamp_us=0), "non-positive timestamp"),
        (replace(_sample(1), value=float("nan")), "finite"),
        (replace(_sample(1), value=1.01), "within"),
    ],
)
def test_invalid_sample_fails_before_ready(
    sample: collector._Sample,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _Source([[sample]])
    _install_fake_signals(monkeypatch)
    monkeypatch.setattr(collector, "_DcgmSource", lambda: source)

    with pytest.raises(ValueError, match=message):
        collector.collect(tmp_path / "dcgm.csv", tmp_path / "ready")

    assert not (tmp_path / "ready").exists()
    assert source.closed


def test_rejects_non_increasing_raw_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _Source([[_sample(10), _sample(10)]])
    _install_fake_signals(monkeypatch)
    monkeypatch.setattr(collector, "_DcgmSource", lambda: source)

    with pytest.raises(ValueError, match="strictly increasing"):
        collector.collect(tmp_path / "dcgm.csv", tmp_path / "ready")

    assert source.closed
