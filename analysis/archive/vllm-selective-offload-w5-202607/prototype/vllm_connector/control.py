"""Out-of-band control channel between the spike driver and the connector.

The driver owns an :class:`OffloadControl`; the connector (which vLLM may
instantiate in a *separate* scheduler and/or worker process) reads the same
channel. We back it with a single JSON sentinel file written atomically rather
than a bare in-process object, because vLLM v1 runs the scheduler and workers
in their own processes -- an in-memory flag on the driver would not be visible
to the connector. A file is the laziest transport that actually crosses that
boundary and needs no REST server. The driver still treats it as "an object it
owns": construct one, call :meth:`request_offload` / :meth:`request_restore`.
"""

from __future__ import annotations

import enum
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


class OffloadPhase(str, enum.Enum):
    """What the connector / scheduler should do with the target request's KV.

    OFFLOAD/RESTORE drive the W1 *copy-only* scenario (connector reads this
    file directly). PAUSE/RESUME drive the P4 *evict* scenario and are consumed
    by :class:`PausableScheduler` (which then pokes the connector in-process) --
    a PAUSE saves the KV, frees the blocks, and holds the request out of every
    queue until a RESUME reallocates + reloads it. Same atomic-rename channel,
    zero new IPC.
    """

    RESIDENT = "resident"  # blocks live on GPU, no action
    OFFLOAD = "offload"  # copy target blocks GPU -> host, then mark offloaded
    RESTORE = "restore"  # copy target blocks host -> GPU, then mark resident
    PAUSE = "pause"  # save KV -> evict blocks -> hold (scheduler-driven)
    RESUME = "resume"  # reallocate -> load saved KV -> continue generation


@dataclass
class ControlState:
    """Serializable control payload shared via the sentinel file."""

    target_request_id: str | None
    phase: OffloadPhase
    epoch: int  # bumped on every state change so the connector detects edges

    def to_json(self) -> str:
        return json.dumps(
            {
                "target_request_id": self.target_request_id,
                "phase": self.phase.value,
                "epoch": self.epoch,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> ControlState:
        d = json.loads(text)
        return cls(
            target_request_id=d["target_request_id"],
            phase=OffloadPhase(d["phase"]),
            epoch=int(d["epoch"]),
        )


class OffloadControl:
    """File-backed offload control. Thread/process safe via atomic rename."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        if not self.path.exists():
            self._write(ControlState(target_request_id=None, phase=OffloadPhase.RESIDENT, epoch=0))

    def _write(self, state: ControlState) -> None:
        # Atomic replace so a concurrent reader never sees a half-written file.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(state.to_json())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def read(self) -> ControlState:
        return ControlState.from_json(self.path.read_text())

    def _transition(self, target_request_id: str | None, phase: OffloadPhase) -> ControlState:
        cur = self.read()
        new = ControlState(
            target_request_id=target_request_id,
            phase=phase,
            epoch=cur.epoch + 1,
        )
        self._write(new)
        return new

    def request_offload(self, request_id: str) -> ControlState:
        return self._transition(request_id, OffloadPhase.OFFLOAD)

    def request_restore(self, request_id: str) -> ControlState:
        return self._transition(request_id, OffloadPhase.RESTORE)

    def request_pause(self, request_id: str) -> ControlState:
        return self._transition(request_id, OffloadPhase.PAUSE)

    def request_resume(self, request_id: str) -> ControlState:
        return self._transition(request_id, OffloadPhase.RESUME)

    def clear(self) -> ControlState:
        return self._transition(None, OffloadPhase.RESIDENT)
