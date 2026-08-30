#!/usr/bin/env python3
"""Add bounded-bypass aging to stock vLLM 0.11.2 priority scheduling."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

VLLM_VERSION = "0.11.2"
ORIGINAL_SCHEDULER_SHA256 = (
    "82d9cbbc71e147ba3b0e3623f72931f383e07ae72cb3aa9fc2eb4b4c427350e8"
)
PATCHED_SCHEDULER_SHA256 = (
    "52a5d8acf0f5ffeff130d1e27cc24ed79537a1bff7469670c5d550e847ccaef4"
)
_MARKER = "# NATIVE_PRIORITY_ONE_BATCH_AGING_V1"
_IMPORT_ANCHOR = (
    "from vllm.v1.core.sched.request_queue import SchedulingPolicy, "
    "create_request_queue\n"
)
_ADMISSION_ANCHOR = """                req_index += 1
                self.running.append(request)
"""
_ADMISSION_PATCH = """                # NATIVE_PRIORITY_ONE_BATCH_AGING_V1
                if native_priority_record_admission(self.waiting, request):
                    pending_requests = list(self.waiting)
                    self.waiting = create_request_queue(self.policy)
                    for pending_request in pending_requests:
                        self.waiting.add_request(pending_request)

                req_index += 1
                self.running.append(request)
"""


def record_admission(waiting: Any, admitted: Any) -> bool:
    """Age eligible initial requests after one batch of return admissions."""

    limit_raw = os.environ.get("NATIVE_PRIORITY_AGING_BYPASS_LIMIT")
    event_log = os.environ.get("NATIVE_PRIORITY_AGING_EVENT_LOG")
    if limit_raw is None or event_log is None:
        raise RuntimeError("native-priority aging environment is incomplete")
    limit = int(limit_raw)
    if limit < 1:
        raise RuntimeError("native-priority aging bypass limit must be positive")

    admitted_base = getattr(admitted, "_native_priority_base", admitted.priority)
    admitted._native_priority_base = admitted_base
    if admitted_base != 0:
        return False

    promoted: list[dict[str, Any]] = []
    for request in waiting:
        base = getattr(request, "_native_priority_base", request.priority)
        request._native_priority_base = base
        if base != 1 or request.priority != 1:
            continue
        bypasses = getattr(request, "_native_priority_bypasses", 0) + 1
        request._native_priority_bypasses = bypasses
        if bypasses == limit:
            request.priority = 0
            promoted.append(
                {
                    "schema": "native-priority-aging-v1",
                    "request_id": request.request_id,
                    "bypass_count": bypasses,
                }
            )
    if promoted:
        with Path(event_log).open("a", encoding="utf-8") as handle:
            for event in promoted:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")
    return bool(promoted)


def patch_scheduler(path: Path) -> bool:
    """Patch an exact stock scheduler; return False when already patched."""

    source = path.read_text(encoding="utf-8")
    if _MARKER in source:
        return False
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual != ORIGINAL_SCHEDULER_SHA256:
        raise ValueError(f"refusing to patch unexpected scheduler.py sha256 {actual}")
    for anchor in (_IMPORT_ANCHOR, _ADMISSION_ANCHOR):
        if source.count(anchor) != 1:
            raise ValueError(f"unexpected scheduler.py anchor: {anchor!r}")
    source = source.replace(
        _IMPORT_ANCHOR,
        _IMPORT_ANCHOR
        + "from scripts.baselines.native_priority_aging import (\n"
        + "    record_admission as native_priority_record_admission,\n"
        + ")\n",
        1,
    ).replace(_ADMISSION_ANCHOR, _ADMISSION_PATCH, 1)
    temporary = path.with_name(path.name + ".native-priority-aging.tmp")
    temporary.write_text(source, encoding="utf-8")
    os.chmod(temporary, path.stat().st_mode)
    os.replace(temporary, path)
    return True
