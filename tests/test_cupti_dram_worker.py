from __future__ import annotations

import csv
import importlib.util
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class _Sample:
    start_timestamp: int
    end_timestamp: int
    metric_values: list[float]


class _CounterData(list[_Sample]):
    metrics = [
        "dram__bytes_read.sum.per_second",
        "dram__bytes_write.sum.per_second",
    ]


def _load_worker(monkeypatch: pytest.MonkeyPatch, events: list[object]):
    class BaseWorker:
        def init_device(self) -> None:
            events.append("base-init")

        def shutdown(self) -> None:
            events.append("base-shutdown")

    gpu_worker = types.ModuleType("vllm.v1.worker.gpu_worker")
    gpu_worker.Worker = BaseWorker
    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", gpu_worker)

    class Collector:
        instances: list[Collector] = []

        def __init__(self, device_index: int) -> None:
            events.append(("collector", device_index))
            self.decodes: list[_CounterData] = []
            Collector.instances.append(self)

        def enable(self) -> None:
            events.append("enable")

        def configure(self, **kwargs: object) -> None:
            events.append(("configure", kwargs))

        def start(self) -> None:
            events.append("start")

        def decode(self) -> _CounterData:
            events.append("decode")
            return self.decodes.pop(0) if self.decodes else _CounterData()

        def stop(self) -> None:
            events.append("stop")

        def disable(self) -> None:
            events.append("disable")

    cupti = types.ModuleType("cupti")
    cupti.__path__ = []
    pm_sampling = types.ModuleType("cupti.pm_sampling")
    pm_sampling.Collector = Collector
    monkeypatch.setitem(sys.modules, "cupti", cupti)
    monkeypatch.setitem(sys.modules, "cupti.pm_sampling", pm_sampling)
    path = Path(__file__).parents[1] / "scripts/evaluation/cupti_dram_worker.py"
    spec = importlib.util.spec_from_file_location("_cupti_dram_worker_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, Collector


def _set_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path]:
    csv_path = tmp_path / "dram.csv"
    ready_path = tmp_path / "dram.ready"
    error_path = tmp_path / "dram.error"
    monkeypatch.setenv("CUPTI_DRAM_CSV", str(csv_path))
    monkeypatch.setenv("CUPTI_DRAM_READY", str(ready_path))
    monkeypatch.setenv("CUPTI_DRAM_ERROR", str(error_path))
    return csv_path, ready_path, error_path


def test_worker_records_valid_samples_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[object] = []
    module, collectors = _load_worker(monkeypatch, events)
    csv_path, ready_path, error_path = _set_paths(monkeypatch, tmp_path)

    worker = module.CuptiDramWorker()
    worker.init_device()
    collector = collectors.instances[0]
    collector.decodes.append(
        _CounterData(
            [
                _Sample(1_000_000_000, 3_000_000_000, [3.0, 4.0]),
                _Sample(3_000_000_000, 4_000_000_000, [3.0, 4.0]),
            ]
        )
    )
    worker._decode_dram_samples()
    collector.decodes.append(
        _CounterData(
            [
                _Sample(3_000_000_000, 4_000_000_000, [3.0, 4.0]),
                _Sample(3_500_000_000, 6_000_000_000, [4.0, 5.0]),
                _Sample(4_000_000_000, 5_000_000_000, [5.0, 6.0]),
            ]
        )
    )
    worker.shutdown()

    configure = next(event[1] for event in events if event[0] == "configure")
    assert events[:4] == [
        "base-init",
        ("collector", 0),
        "enable",
        ("configure", configure),
    ]
    assert configure["metrics"] == _CounterData.metrics
    assert configure["sampling_interval"] == 1_000_000_000
    assert configure["single_pass_metric_set_name"] == "TriageSCG"
    assert events[-4:] == ["stop", "decode", "disable", "base-shutdown"]
    assert ready_path.exists()
    assert not error_path.exists()
    with csv_path.open(newline="") as source:
        assert list(csv.reader(source)) == [
            [
                "start_timestamp_ns",
                "end_timestamp_ns",
                "gpu_id",
                "read_bytes_per_s",
                "write_bytes_per_s",
            ],
            ["3000000000", "4000000000", "0", "3.0", "4.0"],
            ["4000000000", "5000000000", "0", "5.0", "6.0"],
        ]


@pytest.mark.parametrize(
    ("decodes", "message"),
    [
        (
            [
                _CounterData([_Sample(1_000_000_000, 2_000_000_000, [1.0, 2.0])]),
                _CounterData(
                    [_Sample(2_000_000_000, 4_000_000_000, [float("nan"), 4.0])]
                ),
            ],
            "finite non-negative",
        ),
    ],
)
def test_sampling_anomaly_is_fatal_at_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    decodes: list[_CounterData],
    message: str,
) -> None:
    events: list[object] = []
    module, collectors = _load_worker(monkeypatch, events)
    module._DECODE_SECONDS = 0.001
    _, _, error_path = _set_paths(monkeypatch, tmp_path)

    worker = module.CuptiDramWorker()
    worker.init_device()
    collector = collectors.instances[0]
    collector.decodes.extend(decodes)

    deadline = time.monotonic() + 1
    while not error_path.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert error_path.exists()
    assert message in error_path.read_text()

    with pytest.raises(RuntimeError, match="CUPTI DRAM sampling failed"):
        worker.shutdown()
    assert events[-3:] == ["decode", "disable", "base-shutdown"]
