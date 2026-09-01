#!/usr/bin/env python3
"""Collect one-second DCGM DRAM-active samples for GPU 0."""

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any


_GPU_ID = 0
_FIELD_ID = 1005
_PERIOD_US = 1_000_000
_POLL_SECONDS = 0.1
_FINAL_SAMPLE_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True, slots=True)
class _Sample:
    timestamp_us: int
    gpu_id: int
    field_id: int
    field_type: int
    status: int
    value: float
    blank: bool


class _DcgmSource:
    def __init__(self) -> None:
        self._stack = ExitStack()
        self._pending: list[_Sample] = []
        self._callback_error: BaseException | None = None
        self._since_us = 0
        self._agent: Any = None
        self._group: Any = None
        self._field_group: Any = None
        self._handle: Any = None
        self._callback: Any = None
        self._double_field_type = 0

    def __enter__(self) -> _DcgmSource:
        import dcgm_agent
        import dcgm_fields
        import dcgmvalue
        import pydcgm

        if dcgm_fields.DCGM_FI_PROF_DRAM_UTIL_RATIO != _FIELD_ID:
            raise RuntimeError("DCGM DRAM-active field ID is not 1005")

        self._agent = dcgm_agent
        self._double_field_type = ord(dcgm_fields.DCGM_FT_DOUBLE)
        self._handle = pydcgm.DcgmHandle(ipAddress="localhost")
        self._stack.callback(self._handle.Shutdown)
        system = self._handle.GetSystem()
        name = f"agent-sched-dram-{os.getpid()}-{time.time_ns()}"
        self._group = system.GetGroupWithGpuIds(name, [_GPU_ID])
        self._stack.callback(self._group.Delete)
        self._field_group = pydcgm.DcgmFieldGroup(
            self._handle, name=f"{name}-field", fieldIds=[_FIELD_ID]
        )
        self._stack.callback(self._field_group.Delete)

        def receive(gpu_id: int, values: Any, count: int, _user_data: Any) -> int:
            try:
                for index in range(count):
                    raw = values[index]
                    value = float(raw.value.dbl)
                    self._pending.append(
                        _Sample(
                            timestamp_us=int(raw.ts),
                            gpu_id=int(gpu_id),
                            field_id=int(raw.fieldId),
                            field_type=int(raw.fieldType),
                            status=int(raw.status),
                            value=value,
                            blank=bool(dcgmvalue.DCGM_FP64_IS_BLANK(value)),
                        )
                    )
            except BaseException as exc:
                self._callback_error = exc
                return 1
            return 0

        self._callback = dcgm_agent.dcgmFieldValueEnumeration_f(receive)
        self._since_us = time.time_ns() // 1_000
        self._group.samples.WatchFields(
            self._field_group,
            updateFreq=_PERIOD_US,
            maxKeepAge=10.0,
            maxKeepSamples=0,
        )
        self._stack.callback(self._group.samples.UnwatchFields, self._field_group)
        system.UpdateAllFields(True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return self._stack.__exit__(exc_type, exc, traceback)

    @property
    def double_field_type(self) -> int:
        return self._double_field_type

    def read(self) -> list[_Sample]:
        self._pending = []
        self._callback_error = None
        next_since_us = self._agent.dcgmGetValuesSince(
            self._handle.handle,
            self._group.GetId(),
            self._field_group.fieldGroupId,
            self._since_us,
            self._callback,
            None,
        )
        if self._callback_error is not None:
            raise RuntimeError("DCGM sample callback failed") from self._callback_error
        if next_since_us < self._since_us:
            raise RuntimeError("DCGM sample cursor moved backwards")
        self._since_us = next_since_us
        return self._pending


def _validate_sample(
    sample: _Sample, double_field_type: int, previous_timestamp_us: int | None
) -> None:
    if sample.gpu_id != _GPU_ID:
        raise ValueError(f"DCGM returned GPU {sample.gpu_id}, expected GPU 0")
    if sample.field_id != _FIELD_ID:
        raise ValueError(f"DCGM returned field {sample.field_id}, expected field 1005")
    if sample.field_type != double_field_type:
        raise ValueError("DCGM field 1005 did not return a double")
    if sample.status != 0:
        raise ValueError(f"DCGM field 1005 returned status {sample.status}")
    if sample.blank:
        raise ValueError("DCGM field 1005 returned a blank value")
    if sample.timestamp_us <= 0:
        raise ValueError("DCGM returned a non-positive timestamp")
    if (
        previous_timestamp_us is not None
        and sample.timestamp_us <= previous_timestamp_us
    ):
        raise ValueError("DCGM timestamps are not strictly increasing")
    if not math.isfinite(sample.value) or not 0 <= sample.value <= 1:
        raise ValueError("DCGM field 1005 value must be finite and within [0, 1]")


def _write_sample(writer: Any, output: Any, sample: _Sample) -> None:
    writer.writerow(
        [
            sample.timestamp_us,
            sample.gpu_id,
            sample.field_id,
            "OK",
            repr(sample.value),
        ]
    )
    output.flush()


def collect(output_csv: Path, ready_file: Path) -> None:
    if ready_file.exists():
        raise FileExistsError(f"ready file already exists: {ready_file}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ready_file.parent.mkdir(parents=True, exist_ok=True)

    stopped_at_us: int | None = None

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped_at_us
        if stopped_at_us is None:
            stopped_at_us = time.time_ns() // 1_000

    previous_sigterm = signal.signal(signal.SIGTERM, stop)
    previous_sigint = signal.signal(signal.SIGINT, stop)
    try:
        with (
            _DcgmSource() as source,
            output_csv.open("x", encoding="utf-8", newline="") as output,
        ):
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(
                [
                    "timestamp_us",
                    "gpu_id",
                    "field_id",
                    "status",
                    "dram_active_ratio",
                ]
            )
            output.flush()
            previous_timestamp_us: int | None = None
            next_regular_sample_us: int | None = None
            final_sample_deadline: float | None = None
            ready = False

            while True:
                for sample in source.read():
                    _validate_sample(
                        sample, source.double_field_type, previous_timestamp_us
                    )
                    previous_timestamp_us = sample.timestamp_us

                    if (
                        stopped_at_us is not None
                        and sample.timestamp_us >= stopped_at_us
                    ):
                        _write_sample(writer, output, sample)
                        if not ready:
                            ready_file.touch(exist_ok=False)
                        return

                    if (
                        next_regular_sample_us is None
                        or sample.timestamp_us >= next_regular_sample_us
                    ):
                        _write_sample(writer, output, sample)
                        if not ready:
                            ready_file.touch(exist_ok=False)
                            ready = True
                        if next_regular_sample_us is None:
                            next_regular_sample_us = sample.timestamp_us + _PERIOD_US
                        else:
                            while next_regular_sample_us <= sample.timestamp_us:
                                next_regular_sample_us += _PERIOD_US

                if stopped_at_us is not None:
                    if final_sample_deadline is None:
                        final_sample_deadline = (
                            time.monotonic() + _FINAL_SAMPLE_TIMEOUT_SECONDS
                        )
                    elif time.monotonic() >= final_sample_deadline:
                        raise TimeoutError(
                            "DCGM produced no sample after the stop request"
                        )
                time.sleep(_POLL_SECONDS)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--ready-file", required=True, type=Path)
    args = parser.parse_args()
    collect(args.output_csv, args.ready_file)


if __name__ == "__main__":
    main()
