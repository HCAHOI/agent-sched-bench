"""Record DRAM read and write rates from the vLLM worker CUDA context."""

from __future__ import annotations

import csv
import math
import os
import threading
from pathlib import Path
from typing import Any

from vllm.v1.worker.gpu_worker import Worker


_METRICS = [
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum.per_second",
]
_SAMPLE_PERIOD_NS = 1_000_000_000
_DECODE_SECONDS = 30.0
_HARDWARE_BUFFER_BYTES = 16 * 1024 * 1024


class CuptiDramWorker(Worker):
    """GPU worker with one-second CUDA-context DRAM bandwidth samples."""

    def init_device(self) -> None:
        super().init_device()

        paths = []
        for name in ("CUPTI_DRAM_CSV", "CUPTI_DRAM_READY", "CUPTI_DRAM_ERROR"):
            value = os.environ.get(name)
            if not value:
                raise RuntimeError(f"{name} must name a file")
            paths.append(Path(value))
        if len(set(paths)) != len(paths):
            raise RuntimeError("CUPTI DRAM output paths must be distinct")

        self._dram_csv_path, self._dram_ready_path, self._dram_error_path = paths
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(f"CUPTI DRAM output already exists: {path}")

        self._dram_stop = threading.Event()
        self._dram_error: BaseException | None = None
        self._dram_last_start_ns: int | None = None
        self._dram_last_end_ns: int | None = None
        self._dram_discarded_leading = False
        self._dram_collector: Any = None
        self._dram_started = False
        self._dram_thread: threading.Thread | None = None
        self._dram_output = self._dram_csv_path.open("x", encoding="utf-8", newline="")
        self._dram_writer = csv.writer(self._dram_output, lineterminator="\n")
        self._dram_writer.writerow(
            [
                "start_timestamp_ns",
                "end_timestamp_ns",
                "gpu_id",
                "read_bytes_per_s",
                "write_bytes_per_s",
            ]
        )
        self._dram_output.flush()

        try:
            from cupti.pm_sampling import Collector

            self._dram_collector = Collector(0)
            self._dram_collector.enable()
            self._dram_collector.configure(
                metrics=_METRICS,
                hardware_buffer_size=_HARDWARE_BUFFER_BYTES,
                sampling_interval=_SAMPLE_PERIOD_NS,
                single_pass_metric_set_name="TriageSCG",
            )
            self._dram_collector.start()
            self._dram_started = True
            self._dram_thread = threading.Thread(
                target=self._dram_decode_loop,
                name="cupti-dram-decode",
                daemon=True,
            )
            self._dram_thread.start()
            self._dram_ready_path.touch(exist_ok=False)
        except BaseException as exc:
            self._record_dram_error(exc)
            self._dram_stop.set()
            if self._dram_thread is not None:
                self._dram_thread.join()
            self._cleanup_dram_collector(final_decode=False)
            raise

    def _record_dram_error(self, exc: BaseException) -> None:
        if self._dram_error is None:
            self._dram_error = exc
            self._dram_error_path.write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )

    def _dram_decode_loop(self) -> None:
        while not self._dram_stop.wait(_DECODE_SECONDS):
            try:
                self._decode_dram_samples()
            except BaseException as exc:
                self._record_dram_error(exc)
                return

    def _decode_dram_samples(self) -> None:
        counter_data = self._dram_collector.decode()
        if list(counter_data.metrics) != _METRICS:
            raise ValueError("CUPTI returned unexpected DRAM metrics")

        wrote = False
        for sample in counter_data:
            start_ns = int(sample.start_timestamp)
            end_ns = int(sample.end_timestamp)
            if start_ns <= 0 or end_ns <= start_ns:
                raise ValueError("CUPTI sample timestamps must be positive and ordered")
            if self._dram_last_end_ns is not None and start_ns < self._dram_last_end_ns:
                continue

            duration_ns = end_ns - start_ns
            if not _SAMPLE_PERIOD_NS // 2 <= duration_ns <= 3 * _SAMPLE_PERIOD_NS // 2:
                if (
                    self._dram_last_start_ns is None
                    and not self._dram_discarded_leading
                ):
                    self._dram_discarded_leading = True
                    continue
                raise ValueError("CUPTI sample duration is outside 0.5-1.5 seconds")

            values = [float(value) for value in sample.metric_values]
            if len(values) != 2 or not all(
                math.isfinite(value) and value >= 0 for value in values
            ):
                raise ValueError(
                    "CUPTI DRAM rates must be two finite non-negative values"
                )

            self._dram_writer.writerow([start_ns, end_ns, 0, *values])
            self._dram_last_start_ns = start_ns
            self._dram_last_end_ns = end_ns
            wrote = True
        if wrote:
            self._dram_output.flush()

    def _cleanup_dram_collector(self, *, final_decode: bool) -> None:
        collector = self._dram_collector
        try:
            if collector is not None and self._dram_started:
                try:
                    collector.stop()
                except BaseException as exc:
                    self._record_dram_error(exc)
                if final_decode:
                    try:
                        self._decode_dram_samples()
                    except BaseException as exc:
                        self._record_dram_error(exc)
            if collector is not None:
                try:
                    collector.disable()
                except BaseException as exc:
                    self._record_dram_error(exc)
        finally:
            self._dram_collector = None
            self._dram_started = False
            self._dram_output.close()

    def shutdown(self) -> None:
        stop = getattr(self, "_dram_stop", None)
        if stop is not None:
            stop.set()
            thread = self._dram_thread
            if thread is not None:
                thread.join()
            self._cleanup_dram_collector(final_decode=True)

        super_error: BaseException | None = None
        try:
            super().shutdown()
        except BaseException as exc:
            super_error = exc

        dram_error = getattr(self, "_dram_error", None)
        if dram_error is not None:
            raise RuntimeError("CUPTI DRAM sampling failed") from dram_error
        if super_error is not None:
            raise super_error
